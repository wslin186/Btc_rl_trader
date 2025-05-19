# btc_rl_trader/trading_env/btc_trading_env.py
# --------------------------------------------------------------
# 单币全仓多头环境：0=保持 1=全仓买入 2=全仓卖出/平仓
# reward = reward_scale * log(account_value_t / account_value_{t-1})
# 若动作发生改变且实际成交，额外 +action_reward
# --------------------------------------------------------------
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class BTCTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # -----------------------------------------------------------
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 50,
        init_cash: float = 10_000.0,
        fee: float = 0.0004,
        slippage: float = 0.0,
        reward_scale: float = 100.0,     # ← 放大 log 回报
        action_reward: float = 0.01,     # ← 动作切换奖励
        trade_penalty: float = 0.001,    # ← 交易惩罚
        sharpe_coef: float = 0.1,        # ← 夏普比率奖励系数
        position_steps: int = 1,         # ← 仓位步数（1=全仓）
        use_stop_loss: bool = False,     # ← 是否启用止损
        stop_loss_pct: float = 0.05,     # ← 止损百分比
        position_penalty_ratio: float = 0.5,  # ← 持仓惩罚系数
        stop_loss_penalty_ratio: float = 2.0, # ← 止损惩罚系数
        trade_cooldown: int = 0,         # ← 交易冷却期（小时）
        balance_buy_sell: bool = False,  # ← 是否平衡买卖奖励
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
    
        # 添加交易日志记录
        self.trade_log = []
    
        # ---------- 数据与参数 ----------
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.init_cash = float(init_cash)
        self.fee = fee
        self.slippage = slippage
        self.reward_scale = reward_scale
        self.action_reward = action_reward
        self.trade_penalty = trade_penalty
        self.sharpe_coef = sharpe_coef
        self.position_steps = position_steps
        self.use_stop_loss = use_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.position_penalty_ratio = position_penalty_ratio
        self.stop_loss_penalty_ratio = stop_loss_penalty_ratio
        self.trade_cooldown = trade_cooldown  # 新增交易冷却期
        self.balance_buy_sell = balance_buy_sell  # 新增买卖平衡开关
        self.max_steps = max_steps or len(self.df)
        self.render_mode = render_mode
        
        # 记录买入平均价格（用于止损）
        self.entry_price = 0.0
        self.max_drawdown = 0.0  # 应当改为最小值
        self.peak_value = 0.0    # 添加新变量记录峰值
        
        # 增加交易统计与冷却倒计时
        self.buy_count = 0
        self.sell_count = 0
        self.cooldown_counter = 0  # 新增冷却倒计时
        self.holding_time = 0  # 新增持仓时间计数
        
        # ---------- 回报历史 ----------
        self.return_history = []
        self.sharpe_window = 20

        # ---------- Gym spaces ----------
        # 观察空间: 所有特征 + 持仓信息
        feature_dim = self.df.shape[1] - 1  # 减去timestamp
        position_dim = 5  # 持仓状态、现金比例、仓位比例、净值比例、入场价格比例
        obs_dim = self.window_size * feature_dim + position_dim
        self.observation_space = Box(-np.inf, np.inf, (obs_dim,), np.float32)
        
        # 动作空间: 0=保持，1-position_steps=买入不同比例，position_steps+1=卖出
        self.action_space = Discrete(position_steps + 2)

        # ---------- 缓存 ----------
        self._price_series = self.df["close"].values.astype(np.float32)

        self._reset_internal_state()

    # ==================================================================
    # Gym 接口
    # ==================================================================
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_internal_state()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行交易动作，返回新观察、奖励、结束标志、信息"""
        assert self.action_space.contains(action), f"无效动作: {action}"
        
        # ---------- 记录上一步状态 ----------
        prev_action = self._current_action
        prev_value = self.account_value
        prev_position = self.position
        
        # ---------- 执行交易 ----------
        self._cursor += 1
        price_raw = self._price_series[self._cursor - 1]
        
        # 更新冷却倒计时
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        
        # 如果持有仓位，增加持仓时间
        if self.position > 0:
            self.holding_time += 1
        else:
            self.holding_time = 0
        
        # 止损检查（如果启用）
        stop_loss_triggered = False
        if self.use_stop_loss and self.position > 0 and self.entry_price > 0:
            # 如果当前价格低于入场价格的(1-stop_loss_pct)，触发止损
            if price_raw <= self.entry_price * (1 - self.stop_loss_pct):
                action = self.position_steps + 1  # 强制卖出
                stop_loss_triggered = True
                self.trade_log.append(("STOP_LOSS", self._cursor - 1, price_raw))
        
        # 检查交易冷却期
        in_cooldown = self.cooldown_counter > 0
        
        # 计算滑点
        buy_action = action >= 1 and action <= self.position_steps
        sell_action = action == self.position_steps + 1
        
        # 如果在冷却期且要执行交易，修改为保持当前状态
        if in_cooldown and (buy_action or sell_action):
            # 将动作改为保持，不执行交易
            buy_action = False
            sell_action = False
            action = 0  # 改为不操作
        
        slippage_impact = self.slippage * price_raw if buy_action else -self.slippage * price_raw if sell_action else 0
        execution_price = price_raw + slippage_impact
        
        # 执行交易
        trade_executed = False
        
        if buy_action and self.cash > 0:  # 买入操作
            # 确保全仓模式下使用全部资金
            if self.position_steps == 1:
                buy_ratio = 1.0
            else:
                # 多级仓位模式下使用指定比例
                buy_ratio = action / self.position_steps
                
            cash_to_use = self.cash * buy_ratio
            
            # 计算可买数量 (考虑手续费)
            btc_amount = cash_to_use / execution_price / (1 + self.fee)
            
            if btc_amount > 0:
                # 如果是新开仓或加仓，更新平均入场价格
                if self.position == 0:
                    self.entry_price = execution_price
                else:
                    # 计算加权平均入场价格
                    self.entry_price = (self.position * self.entry_price + btc_amount * execution_price) / (self.position + btc_amount)
                
                self.position += btc_amount
                self.cash -= cash_to_use
                trade_executed = True
                # 记录买入交易
                self.trade_log.append(("BUY", self._cursor - 1, execution_price, buy_ratio))
                self.buy_count += 1
                
                # 设置交易冷却期
                if self.trade_cooldown > 0:
                    self.cooldown_counter = self.trade_cooldown
                
        elif sell_action and self.position > 0:  # 卖出操作
            # 计算卖出所得 (考虑手续费)
            self.cash += self.position * execution_price * (1 - self.fee)
            self.position = 0
            self.entry_price = 0  # 清空入场价
            self.holding_time = 0  # 重置持仓时间
            trade_executed = True
            # 记录卖出交易

            if stop_loss_triggered:
                # 已经在止损逻辑中记录了
                pass
            else:
                self.trade_log.append(("SELL", self._cursor - 1, execution_price))
                self.sell_count += 1
            
            # 设置交易冷却期
            if self.trade_cooldown > 0:
                self.cooldown_counter = self.trade_cooldown
        
        # 更新账户价值
        self.account_value = self.cash + self.position * price_raw
        
        # 更新最大回撤 - 使用峰值法计算
        if self.account_value > self.peak_value:
            self.peak_value = self.account_value
        
        if self.peak_value > 0:
            current_drawdown = (self.account_value / self.peak_value - 1)
            if current_drawdown < self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        # ---------- 奖励计算 ----------
        # 记录回报率，用于夏普比率计算
        returns_pct = self.account_value / prev_value - 1
        self.return_history.append(returns_pct)
        
        # 基础奖励 - 净值对数变化
        reward = self.reward_scale * np.log(self.account_value / prev_value)
        
        # 动作改变且成交的奖励
        if trade_executed and action != prev_action:
            reward += self.action_reward
            
        # 交易惩罚
        if trade_executed:
            reward -= self.trade_penalty
            
        # 持仓惩罚 - 当价格下跌时惩罚持仓
        if self.position > 0 and returns_pct < 0:
            reward -= self.trade_penalty * self.position_penalty_ratio
        
        # 止损惩罚 - 如果触发止损，给予更大惩罚
        if stop_loss_triggered:
            reward -= self.trade_penalty * self.stop_loss_penalty_ratio
            
        # 买卖平衡奖励 - 当买卖比例接近1:1时给予额外奖励
        if self.balance_buy_sell and trade_executed:
            # 计算买卖比例的平衡度
            if self.buy_count > 0 and self.sell_count > 0:
                ratio = min(self.buy_count, self.sell_count) / max(self.buy_count, self.sell_count)
                # 当比例接近1时，给予奖励
                balance_reward = ratio * self.action_reward
                reward += balance_reward
        
        # 夏普比率奖励 (当有足够样本时)
        if len(self.return_history) >= self.sharpe_window:
            returns = np.array(self.return_history[-self.sharpe_window:])
            if returns.std() > 0:
                sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(self.sharpe_window)
                reward += self.sharpe_coef * sharpe
                
        # 更新当前动作
        self._current_action = action
        
        # ---------- 检查是否结束 ----------
        done = self._cursor >= min(len(self.df) - 1, self.max_steps)
        terminated = done
        truncated = False  # 提前终止标志
        
        # ---------- 返回 SARTD + info ----------
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    # ==================================================================
    # 内部方法
    # ==================================================================
    def _reset_internal_state(self) -> None:
        """重置内部状态"""
        # 游标位置、现金、持仓
        self._cursor = self.window_size
        self.cash = self.init_cash
        self.position = 0.0
        
        # 账户价值、当前动作
        self.account_value = self.cash
        self._current_action = 0  # 默认不操作
        
        # 回报历史
        self.return_history = []
        
        # 清空交易日志
        self.trade_log = []
        
        # 重置统计指标
        self.buy_count = 0
        self.sell_count = 0
        self.cooldown_counter = 0
        self.holding_time = 0
        self.peak_value = self.init_cash  # 初始化峰值为初始资金
    
    def _get_observation(self) -> np.ndarray:
        """获取当前窗口的市场数据 + 持仓状态"""
        # 获取窗口数据，排除timestamp列
        window = self.df.iloc[self._cursor - self.window_size : self._cursor]
        market_obs = window.drop(columns=["timestamp"]).values.flatten().astype(np.float32)
        
        # 当前价格
        current_price = self._price_series[self._cursor - 1]
        
        # 持仓信息 (5个特征)
        position_info = np.array([
            self.position / (self.init_cash / current_price) if current_price > 0 else 0,  # 归一化持仓量
            self.cash / self.init_cash,  # 现金比例
            self.position * current_price / self.account_value if self.account_value > 0 else 0,  # 持仓价值比例
            self.account_value / self.init_cash,  # 当前净值比例
            self.entry_price / current_price if self.entry_price > 0 and current_price > 0 else 1.0  # 入场价格/当前价格比例
        ], dtype=np.float32)
        
        # 合并特征
        obs = np.concatenate([market_obs, position_info])
        
        # 安全检查
        if np.isnan(obs).any() or np.isinf(obs).any():
            # 如果有NaN或Inf，替换为0，避免模型崩溃
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            
        return obs
        
    def _get_info(self) -> dict:
        """获取额外信息"""
        return {
            "account_value": self.account_value,
            "cash": self.cash,
            "position": self.position,
            "current_price": self._price_series[self._cursor - 1],
            "current_step": self._cursor,
            "return_pct": (self.account_value / self.init_cash - 1) * 100,
            "entry_price": self.entry_price,
            "max_drawdown": self.max_drawdown * 100,  # 转为百分比
            "position_ratio": self.position * self._price_series[self._cursor - 1] / self.account_value if self.account_value > 0 else 0
        }
        
    def render(self):
        """环境可视化"""
        if self.render_mode != "human":
            return
            
        print(f"步数: {self._cursor}/{len(self.df)}")
        print(f"价格: {self._price_series[self._cursor - 1]:.2f}")
        print(f"现金: {self.cash:.2f}")
        print(f"持仓: {self.position:.6f}")
        print(f"净值: {self.account_value:.2f}")
        print(f"回报率: {(self.account_value / self.init_cash - 1) * 100:.2f}%")
        print("=" * 30)
        
    def close(self):
        """关闭环境"""
        pass
