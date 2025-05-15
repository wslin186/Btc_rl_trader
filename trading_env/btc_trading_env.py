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
        self.max_steps = max_steps or len(self.df)
        self.render_mode = render_mode
        
        # ---------- 回报历史 ----------
        self.return_history = []
        self.sharpe_window = 20

        # ---------- Gym spaces ----------
        # 观察空间: 所有特征 + 持仓信息
        feature_dim = self.df.shape[1] - 1  # 减去timestamp
        position_dim = 4  # 持仓状态、现金比例、仓位比例、净值比例
        obs_dim = self.window_size * feature_dim + position_dim
        self.observation_space = Box(-np.inf, np.inf, (obs_dim,), np.float32)
        
        # 动作空间: 保持、买入、卖出
        self.action_space = Discrete(3)

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
        
        # ---------- 执行交易 ----------
        self._cursor += 1
        price_raw = self._price_series[self._cursor - 1]
        
        # 计算滑点 (买入时价格上涨，卖出时价格下跌)
        slippage_impact = self.slippage * price_raw if action == 1 else -self.slippage * price_raw if action == 2 else 0
        execution_price = price_raw + slippage_impact
        
        # 检查是否与当前持仓状态一致
        trade_executed = False
        if action == 1 and self.position == 0:  # 买入
            # 计算可买数量 (考虑手续费)
            btc_amount = self.cash / execution_price / (1 + self.fee)
            self.position = btc_amount
            self.cash = 0
            trade_executed = True
            # 记录买入交易
            self.trade_log.append(("BUY", self._cursor - 1, execution_price))
                
        elif action == 2 and self.position > 0:  # 卖出
            # 计算卖出所得 (考虑手续费)
            self.cash = self.position * execution_price * (1 - self.fee)
            self.position = 0
            trade_executed = True
            # 记录卖出交易
            self.trade_log.append(("SELL", self._cursor - 1, execution_price))
        
        # 更新账户价值
        self.account_value = self.cash + self.position * price_raw
        
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
            reward -= self.trade_penalty * 0.5
        
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
        
    def _get_observation(self) -> np.ndarray:
        """获取当前窗口的市场数据 + 持仓状态"""
        # 获取窗口数据，排除timestamp列
        window = self.df.iloc[self._cursor - self.window_size : self._cursor]
        market_obs = window.drop(columns=["timestamp"]).values.flatten().astype(np.float32)
        
        # 当前价格
        current_price = self._price_series[self._cursor - 1]
        
        # 持仓信息 (4个特征)
        position_info = np.array([
            float(self.position > 0),  # 是否持仓
            self.cash / self.init_cash,  # 现金比例
            self.position * current_price / self.account_value if self.account_value > 0 else 0,  # 持仓价值比例
            self.account_value / self.init_cash  # 当前净值比例
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
            "return_pct": (self.account_value / self.init_cash - 1) * 100
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
