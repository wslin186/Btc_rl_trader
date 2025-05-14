#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_visual.py —— 价格曲线 + 买卖点 + 净值曲线 可视化 (PNG & HTML)
用法：
    python scripts/evaluate_visual.py          # 随机动作
    python scripts/evaluate_visual.py --det    # 确定性动作
"""
from __future__ import annotations
import argparse, yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import BTCTradingEnv

# ---------------- 全局路径 ----------------
ROOT       = Path(__file__).resolve().parents[1]
CFG        = yaml.safe_load((ROOT / "config" / "config.yaml").read_text("utf-8"))
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
VEC_PKL    = MODEL_DIR / "vec_norm.pkl"


# =====================================================================
# 工具函数
# =====================================================================

def load_latest_model() -> Path:
    zips = sorted(MODEL_DIR.glob("ppo_btc_latest_*.zip"),
                  key=lambda p: p.stat().st_mtime)
    if not zips:
        raise FileNotFoundError("❌ models/ 目录下找不到 ppo_btc_latest_*.zip")
    return zips[-1]


def build_env(test_df: pd.DataFrame, env_cfg: dict):
    """
    返回 env, 以及布尔 is_vec_env
    """
    base_env = DummyVecEnv(
        [lambda: BTCTradingEnv(test_df.copy(), **env_cfg)]
    )  # 用 DummyVecEnv 包一层，方便后续统一接口

    if VEC_PKL.exists():
        vec_env: VecNormalize = VecNormalize.load(VEC_PKL, base_env)
        vec_env.training = False          # 冻结均值方差
        vec_env.norm_reward = False
        return vec_env, True
    else:
        return base_env, True             # DummyVecEnv 也是 VecEnv


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
        obs, rewards, dones, infos = env.step([action])
        return obs[0], float(rewards[0]), bool(dones[0]), infos[0]
    else:  # 单环境（此脚本永远不会到这分支，因为外层 DummyVecEnv）
        obs, reward, done, trunc, info = env.step(action)
        return obs, float(reward), done or trunc, info


# =====================================================================
# 主回测逻辑
# =====================================================================
def run_backtest(det: bool):
    data_cfg   = CFG["data"]
    env_cfg    = CFG["env"]

    # ---------- 载入测试集 ----------
    df = pd.read_parquet(ROOT / data_cfg["processed_path"])
    start = pd.to_datetime(data_cfg["test_start"])
    end   = pd.to_datetime(data_cfg["test_end"])
    test_df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)

    # ---------- 环境 ----------
    env, is_vec = build_env(test_df, env_cfg)

    # ---------- 模型 ----------
    model = PPO.load(load_latest_model())

    # ---------- 回测循环 ----------
    obs   = env_reset(env)
    equity_curve = []
    trades = []

    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=det)
        obs, reward, done, info = env_step(env, action)

        # 记录净值
        equity_curve.append((test_df.loc[min(step, len(test_df)-1), "timestamp"],
                             info["account_value"]))

        # 若刚执行完交易，VecNormalize -> info 不含 trade_log, 只能用 env
        if hasattr(env, "venv"):
            inner_env: BTCTradingEnv = env.venv.envs[0]  # deepest env
        else:
            inner_env: BTCTradingEnv = env.envs[0]

        if inner_env.trade_log and inner_env.trade_log[-1][1] == inner_env._cursor - 1:
            side = inner_env.trade_log[-1][0]
            px   = inner_env.trade_log[-1][2]
            t    = test_df.loc[inner_env._cursor - 1, "timestamp"]
            trades.append(dict(t=t, px=px, side=side))

        if done or step >= len(test_df) - 2:
            break
        step += 1

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
def draw_matplotlib(test_df, equity, trades, file_png):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )
    # ------- 价格折线 -------
    ax1.plot(test_df["timestamp"], test_df["close"],
             lw=0.8, label="BTC price", color="#999")
    buys  = trades[trades["side"] == "BUY"]
    sells = trades[trades["side"] == "SELL"]
    ax1.scatter(buys["t"],  buys["px"],  marker="^", c="green", s=60, label="BUY")
    ax1.scatter(sells["t"], sells["px"], marker="v", c="red",   s=60, label="SELL")
    ax1.set_ylabel("Price (USDT)"); ax1.legend(loc="upper left")

    # ------- 净值曲线 -------
    init_cash = CFG["env"]["init_cash"]         # ← 用全局 CFG
    bh_equity = (test_df["close"] / test_df["close"].iloc[0]) * init_cash
    ax2.plot(equity.index, equity.values, label="Strategy", lw=1.3)
    ax2.plot(bh_equity.index, bh_equity.values, label="Buy & Hold", lw=1.0, ls="--")
    ax2.set_ylabel("Equity (USDT)"); ax2.legend(loc="upper left")

    fig.tight_layout();
    fig.savefig(file_png, dpi=150);
    plt.close(fig)


def draw_plotly(test_df, equity, trades, file_html):
    price = go.Scatter(x=test_df["timestamp"], y=test_df["close"],
                       mode="lines", name="BTC price", line=dict(color="#999"))
    buy_scat = go.Scatter(x=trades[trades["side"] == "BUY"]["t"],
                          y=trades[trades["side"] == "BUY"]["px"],
                          mode="markers", marker=dict(symbol="triangle-up",
                                                      color="green", size=9),
                          name="BUY")
    sell_scat = go.Scatter(x=trades[trades["side"] == "SELL"]["t"],
                           y=trades[trades["side"] == "SELL"]["px"],
                           mode="markers", marker=dict(symbol="triangle-down",
                                                       color="red", size=9),
                           name="SELL")
    eq_curve = go.Scatter(x=equity.index, y=equity.values,
                          mode="lines", name="Equity", yaxis="y2")

    layout = go.Layout(
        title="BTC Strategy Backtest",
        xaxis=dict(domain=[0, 1]),
        yaxis=dict(title="Price (USDT)"),
        yaxis2=dict(title="Equity", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    go.Figure([price, buy_scat, sell_scat, eq_curve], layout).write_html(
        file_html, include_plotlyjs="cdn"
    )


def save_stats(equity: pd.Series, file_csv: Path):
    ret = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    ann_ret = (1 + total_return) ** (365 * 24 / len(equity)) - 1    # 小时 K
    ann_vol = ret.std() * np.sqrt(365 * 24)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd      = (equity.cummax() - equity) / equity.cummax()
    mdd     = dd.max()

    pd.DataFrame([dict(
        total_return  = total_return,
        annual_return = ann_ret,
        annual_vol    = ann_vol,
        sharpe        = sharpe,
        max_drawdown  = mdd
    )]).to_csv(file_csv, index=False)


# =====================================================================
def main(det: bool):
    test_df, equity, trades = run_backtest(det)

    # 文件名包含时间戳
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    png  = REPORT_DIR / f"backtest_{ts}.png"
    html = REPORT_DIR / f"backtest_{ts}.html"
    csv  = REPORT_DIR / f"backtest_{ts}_summary.csv"

    draw_matplotlib(test_df, equity, trades, png)
    draw_plotly(test_df, equity, trades, html)
    save_stats(equity, csv)

    print(f"✅ PNG  保存: {png}")
    print(f"✅ HTML 保存: {html}")
    print(f"✅ CSV  保存: {csv}")


# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det", action="store_true",
                        help="use deterministic policy action")
    args = parser.parse_args()
    main(args.det)
