#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_visual.py —— 价格曲线 + 买卖点 + 净值曲线 可视化 (PNG & HTML)
用法：
    python scripts/evaluate_visual.py          # 随机动作
    python scripts/evaluate_visual.py --det    # 确定性动作
"""
from __future__ import annotations
import sys
from pathlib import Path

# ---------------- 全局路径 ----------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 导入其他模块
import argparse, yaml
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import BTCTradingEnv  # 现在导入 trading_env 应该没问题了
from tqdm import tqdm  # 导入进度条库
from utils.logger import get_logger  # 导入日志记录器

# 创建日志记录器
logger = get_logger("evaluate")

CFG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text("utf-8"))
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports";
REPORT_DIR.mkdir(exist_ok=True)
VEC_PKL = MODEL_DIR / "vec_norm.pkl"


# =====================================================================
# 工具函数
# =====================================================================

def load_latest_model() -> Path:
    zips = sorted(MODEL_DIR.glob("ppo_btc_latest_*.zip"),
                  key=lambda p: p.stat().st_mtime)
    if not zips:
        raise FileNotFoundError("❌ models/ 目录下找不到 ppo_btc_latest_*.zip")
    logger.info(f"📊 使用模型: {zips[-1].name}")
    return zips[-1]


def build_env(test_df: pd.DataFrame, env_cfg: dict):
    """
    返回 trading_env, 以及布尔 is_vec_env
    """
    base_env = DummyVecEnv(
        [lambda: BTCTradingEnv(test_df.copy(), **env_cfg)]
    )  # 用 DummyVecEnv 包一层，方便后续统一接口

    if VEC_PKL.exists():
        vec_env: VecNormalize = VecNormalize.load(VEC_PKL, base_env)
        vec_env.training = False  # 冻结均值方差
        vec_env.norm_reward = False
        return vec_env, True
    else:
        return base_env, True  # DummyVecEnv 也是 VecEnv


# =====================================================================
# reset / step 兼容包装
# =====================================================================
def env_reset(env):
    """
    兼容 DummyVecEnv(VecNormalize) 只返回 obs 的情况
    return: obs (np.ndarray)
    """
    out = env.reset()
    # DummyVecEnv/VecNormalize -> 1 个返回值 (obs batch)
    if isinstance(out, tuple):
        return out[0]
    return out


def env_step(env, action):
    """
    统一返回: obs, reward(float), done(bool), info(dict)
    """
    # VecEnv 必须传 batch action
    if hasattr(env, "num_envs"):
        # 增强型修复：首先检查动作类型
        try:
            # 只有在支持len()操作且长度为1时才递归展开
            while isinstance(action, (list, np.ndarray)) and hasattr(action, "__len__") and len(action) == 1:
                action = action[0]  # 递归提取内层值
        except (TypeError, AttributeError):
            # 捕获任何类型错误，保持动作不变
            pass
        
        obs, rewards, dones, infos = env.step([action])
        return obs[0], float(rewards[0]), bool(dones[0]), infos[0]
    else:  # 单环境（此脚本永远不会到这分支，因为外层 DummyVecEnv）
        obs, reward, done, trunc, info = env.step(action)
        return obs, float(reward), done or trunc, info


# =====================================================================
# 主回测逻辑
# =====================================================================
def run_backtest(det: bool):
    data_cfg = CFG["data"]
    env_cfg = CFG["env"]  

    # ---------- 载入测试集 ----------
    df = pd.read_parquet(ROOT / data_cfg["processed_path"])
    start = pd.to_datetime(data_cfg["test_start"])
    end = pd.to_datetime(data_cfg["test_end"])
    test_df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)

    # ---------- 环境 ----------
    env, is_vec = build_env(test_df, env_cfg)

    # ---------- 模型 ----------
    model_path = load_latest_model()
    model = PPO.load(model_path)
    logger.info(f"🤖 模型加载成功: {model_path.name}")
    logger.info(f"👉 确定性模式: {'启用' if det else '禁用'}")
    
    # 初始化动作计数器
    action_counts = {0: 0, 1: 0, 2: 0}  # 假设动作空间为：0=持有, 1=买入, 2=卖出
    
    # ---------- 回测循环 ----------
    logger.info("⏳ 开始回测...")
    obs = env_reset(env)
    equity_curve = []
    trades = []
    trade_count = {'BUY': 0, 'SELL': 0}

    # 使用tqdm创建进度条，总长度为测试数据的长度
    total_steps = len(test_df)
    with tqdm(total=total_steps, desc="回测进度", ncols=100, 
            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=det)
            
            # 记录动作频率
            action_val = action.item() if hasattr(action, 'item') else action
            action_counts[action_val] = action_counts.get(action_val, 0) + 1
            
            # 记录动作前的状态
            if hasattr(env, "venv"):
                inner_env = env.venv.envs[0]
            else:
                inner_env = env.envs[0]
                
            # 获取交易前状态
            pre_position = getattr(inner_env, "position", "未知")
            pre_cash = getattr(inner_env, "cash", "未知")
                
            obs, reward, done, info = env_step(env, action)
            
            # 记录净值
            equity_curve.append((test_df.loc[min(step, len(test_df) - 1), "timestamp"],
                                info["account_value"]))

            # 若刚执行完交易，VecNormalize -> info 不含 trade_log, 只能用 trading_env
            if hasattr(env, "venv"):
                inner_env: BTCTradingEnv = env.venv.envs[0]  # deepest trading_env
            else:
                inner_env: BTCTradingEnv = env.envs[0]

            # 添加交易记录捕获逻辑 - 不输出中间统计
            if hasattr(inner_env, "trade_log") and inner_env.trade_log:
                last_trade = inner_env.trade_log[-1]
                    
                # 放宽条件，不仅检查最后一步
                trade_step_diff = abs(last_trade[1] - (inner_env._cursor - 1))
                if trade_step_diff <= 5:  # 允许5步以内的交易也被记录
                    side = last_trade[0]
                    px = last_trade[2]
                    t = test_df.loc[min(last_trade[1], len(test_df) - 1), "timestamp"]
                    trades.append(dict(t=t, px=px, side=side))
                    # 计数但不打印
                    trade_count[side] = trade_count.get(side, 0) + 1

            # 更新进度条
            pbar.update(1)
            
            if done or step >= len(test_df) - 2:
                break
            step += 1

    # 打印交易统计汇总
    # 进度条完成后添加一个空行，防止被其他输出打断
    logger.info("")
    logger.info("\n📈 回测结果汇总:")
    logger.info(f"总步数: {step}")
    logger.info(f"动作分布: 持有 {action_counts.get(0, 0)} 次, 买入 {action_counts.get(1, 0)} 次, 卖出 {action_counts.get(2, 0)} 次")
    logger.info(f"总交易次数: {len(trades)} 笔 (买入: {trade_count.get('BUY', 0)}, 卖出: {trade_count.get('SELL', 0)})")
    
    # 计算收益率
    if len(equity_curve) > 0:
        first_equity = equity_curve[0][1]
        last_equity = equity_curve[-1][1]
        total_return = (last_equity / first_equity - 1) * 100
        logger.info(f"起始资金: {CFG['env']['init_cash']:.2f}, 最终资产: {last_equity:.2f}")
        logger.info(f"策略收益率: {total_return:.2f}%")
        
        # 计算买入持有收益率
        first_price = test_df['close'].iloc[0]
        last_price = test_df['close'].iloc[-1]
        bh_return = (last_price / first_price - 1) * 100
        logger.info(f"买入持有收益率: {bh_return:.2f}%")
        logger.info(f"超额收益率: {total_return - bh_return:.2f}%")

    equity_series = pd.Series(
        [v for _, v in equity_curve],
        index=[t for t, _ in equity_curve],
        name="equity"
    )
    trades_df = pd.DataFrame(trades)
    return test_df, equity_series, trades_df


# =====================================================================
# 可视化 & 存盘
# =====================================================================
def draw_matplotlib(test_df, equity, trades, file_png, model_name=""):
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 确保时间戳是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(test_df["timestamp"]):
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )
    
    # 添加标题显示模型名称
    if model_name:
        fig.suptitle(f"BTC Trading Strategy Backtest - Model: {model_name}", fontsize=14)
    
    # ------- 价格折线 -------
    ax1.plot(test_df["timestamp"], test_df["close"],
             lw=1.2, label="BTC Price", color="#444")
    
    # 检查trades是否为空或是否有side列
    if not trades.empty and 'side' in trades.columns:
        buys = trades[trades["side"] == "BUY"]
        sells = trades[trades["side"] == "SELL"]
        
        if not buys.empty:
            ax1.scatter(buys["t"], buys["px"], marker="^", c="green", s=100, label="BUY", zorder=5)
        if not sells.empty:
            ax1.scatter(sells["t"], sells["px"], marker="v", c="red", s=100, label="SELL", zorder=5)
    else:
        logger.warning("⚠️ 没有检测到交易记录，只显示价格和净值曲线")
    
    # 优化图表显示
    ax1.set_ylabel("Price (USDT)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("BTC Price and Trade Signals", fontsize=12)
    
    # ------- 净值曲线 -------
    init_cash = CFG["env"]["init_cash"]
    bh_equity = (test_df["close"] / test_df["close"].iloc[0]) * init_cash
    ax2.plot(equity.index, equity.values, label="Strategy", lw=1.5, color="#1f77b4")
    ax2.plot(bh_equity.index, bh_equity.values, label="Buy & Hold", lw=1.3, ls="--", color="#ff7f0e")
    
    # 优化下方子图
    ax2.set_ylabel("Equity (USDT)", fontsize=12)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Strategy vs Buy & Hold Performance", fontsize=12)
    
    # 导入日期处理相关模块
    import matplotlib.dates as mdates
    from datetime import timedelta
    
    # 获取数据的日期范围
    date_min, date_max = test_df["timestamp"].min(), test_df["timestamp"].max()
    date_range = (date_max - date_min).total_seconds() / 86400  # 转换为天数
    
    # 根据日期范围动态设置最佳显示方式
    if date_range <= 5:  # 5天以内
        # 每12小时一个刻度
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    elif date_range <= 30:  # 一个月以内
        # 每3天一个刻度
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    elif date_range <= 90:  # 三个月以内
        # 每7天一个刻度
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    else:
        # 每月一个刻度
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 使用次级刻度补充日期轴
    if date_range <= 30:
        ax2.xaxis.set_minor_locator(mdates.DayLocator())
        ax2.grid(True, which='minor', alpha=0.1)
    
    # 设置X轴限制确保只显示实际数据范围
    ax1.set_xlim(date_min - timedelta(hours=12), date_max + timedelta(hours=12))
    
    # 调整日期标签角度
    fig.autofmt_xdate(rotation=30)
    
    # 调整布局
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.1)  # 增加底部空间，减少子图间距
    
    # 保存图表
    fig.savefig(file_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def draw_plotly(test_df, equity, trades, file_html, model_name=""):
    # 创建子图布局
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("BTC Price and Trading Signals", "Strategy Performance"),
                        row_heights=[0.7, 0.3])

    # ------- 上图：价格与买卖点 -------
    # 价格线
    fig.add_trace(
        go.Scatter(
            x=test_df["timestamp"],
            y=test_df["close"],
            mode="lines",
            name="BTC Price",
            line=dict(color="#444", width=1.5)
        ),
        row=1, col=1
    )

    # 检查trades是否为空或是否有side列
    if not trades.empty and 'side' in trades.columns:
        # 买入点
        buys = trades[trades["side"] == "BUY"]
        if not buys.empty:
            buy_text = [f"Buy Price: {px:.2f}<br>Time: {t.strftime('%Y-%m-%d')}" for t, px in zip(buys["t"], buys["px"])]
            fig.add_trace(
                go.Scatter(
                    x=buys["t"],
                    y=buys["px"],
                    mode="markers",
                    name="Buy",
                    marker=dict(symbol="triangle-up", color="green", size=14),
                    hoverinfo="text",
                    hovertext=buy_text
                ),
                row=1, col=1
            )

        # 卖出点
        sells = trades[trades["side"] == "SELL"]
        if not sells.empty:
            sell_text = [f"Sell Price: {px:.2f}<br>Time: {t.strftime('%Y-%m-%d')}" for t, px in zip(sells["t"], sells["px"])]
            fig.add_trace(
                go.Scatter(
                    x=sells["t"],
                    y=sells["px"],
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", color="red", size=14),
                    hoverinfo="text",
                    hovertext=sell_text
                ),
                row=1, col=1
            )

    # ------- 下图：收益率对比 -------
    # 策略收益率
    init_cash = CFG["env"]["init_cash"]
    strategy_return = (equity / equity.iloc[0] - 1) * 100  # 转换为百分比
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=strategy_return,
            mode="lines",
            name="Strategy Return (%)",
            line=dict(color="blue", width=2)
        ),
        row=2, col=1
    )

    # 买入持有收益率
    bh_equity = (test_df["close"] / test_df["close"].iloc[0]) * init_cash
    bh_return = (bh_equity / bh_equity.iloc[0] - 1) * 100  # 转换为百分比
    fig.add_trace(
        go.Scatter(
            x=test_df["timestamp"],
            y=bh_return,
            mode="lines",
            name="Buy & Hold Return (%)",
            line=dict(color="orange", width=1.5, dash="dash")
        ),
        row=2, col=1
    )

    # 使用英文标题避免中文问题
    title = "BTC Trading Strategy Backtest"
    if model_name:
        title += f" - Model: {model_name}"
        
    # 美化图表布局
    fig.update_layout(
        title=title,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=800,
        width=1200,  # 增加图表宽度
        template="plotly_white",  # 使用白色模板，更加美观
        font=dict(family="Arial", size=12),
    )
    
    # 更新Y轴标签
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    
    # 优化X轴日期格式 - 减少刻度密度
    fig.update_xaxes(
        tickformat="%Y-%m",  # 只显示年-月
        tickangle=30,
        dtick="M3",  # 每三个月一个刻度
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220,220,220,0.5)',
        row=2, col=1
    )

    # 保存为HTML
    fig.write_html(file_html, include_plotlyjs="cdn")


def save_stats(equity: pd.Series, file_csv: Path, trades_df=None):
    ret = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    ann_ret = (1 + total_return) ** (365 * 24 / len(equity)) - 1  # 小时 K
    ann_vol = ret.std() * np.sqrt(365 * 24)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd = (equity.cummax() - equity) / equity.cummax()
    mdd = dd.max()
    
    # 添加交易统计信息
    stats_dict = {
        "total_return": total_return,
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }
    
    # 如果有交易记录，添加交易统计
    if trades_df is not None and not trades_df.empty:
        n_trades = len(trades_df)
        n_buys = len(trades_df[trades_df["side"] == "BUY"])
        n_sells = len(trades_df[trades_df["side"] == "SELL"])
        stats_dict.update({
            "n_trades": n_trades,
            "n_buys": n_buys,
            "n_sells": n_sells
        })
    else:
        stats_dict.update({
            "n_trades": 0,
            "n_buys": 0,
            "n_sells": 0
        })

    pd.DataFrame([stats_dict]).to_csv(file_csv, index=False)


# =====================================================================
def main(det: bool):
    # 从run_backtest获取必要数据
    test_df, equity, trades = run_backtest(det)

    # 获取模型文件名用于报告文件名
    model_path = load_latest_model()
    model_name = model_path.stem.replace("ppo_btc_latest_", "")
    
    # 文件名包含时间戳和模型信息
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    png = REPORT_DIR / f"backtest_{model_name}_{ts}.png"
    html = REPORT_DIR / f"backtest_{model_name}_{ts}.html"
    csv = REPORT_DIR / f"backtest_{model_name}_{ts}_summary.csv"

    # 将模型信息添加到绘图标题
    draw_matplotlib(test_df, equity, trades, png, model_name)
    draw_plotly(test_df, equity, trades, html, model_name)
    # 将交易记录也传递给save_stats
    save_stats(equity, csv, trades)

    logger.info(f"✅ PNG  保存: {png}")
    logger.info(f"✅ HTML 保存: {html}")
    logger.info(f"✅ CSV  保存: {csv}")


# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det", action="store_true",
                        help="use deterministic policy action")
    args = parser.parse_args()
    main(args.det)
