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
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        # ---------- 数据与参数 ----------
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.init_cash = float(init_cash)
        self.fee = fee
        self.slippage = slippage
        self.reward_scale = reward_scale
        self.action_reward = action_reward
        self.max_steps = max_steps or len(self.df)
        self.render_mode = render_mode

        # ---------- Gym spaces ----------
        feature_dim = self.df.shape[1] - 1
        obs_dim = self.window_size * feature_dim
        self.observation_space = Box(-np.inf, np.inf, (obs_dim,), np.float32)
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
        return self._get_observation(), {}

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action)

        price_raw = self._price_series[self._cursor]
        buy_price  = price_raw * (1 + self.slippage)
        sell_price = price_raw * (1 - self.slippage)

        prev_action = self.last_action
        trade_executed = False

        # ============ 交易逻辑 ============
        if action == 1 and self.position == 0:
            qty = self.cash / buy_price
            fee_cost = qty * buy_price * self.fee
            self.cash -= qty * buy_price + fee_cost
            self.position = qty
            trade_executed = True
            self.trade_log.append(("BUY", self._cursor, buy_price, qty, fee_cost))

        elif action == 2 and self.position > 0:
            proceeds = self.position * sell_price
            fee_cost = proceeds * self.fee
            self.cash += proceeds - fee_cost
            trade_executed = True
            self.trade_log.append(("SELL", self._cursor, sell_price, self.position, fee_cost))
            self.position = 0.0

        self.last_action = action

        # ============ 更新净值 & 奖励 ============
        prev_value = self.account_value
        self.account_value = self.cash + self.position * price_raw
        reward = self.reward_scale * np.log(self.account_value / prev_value)

        if trade_executed and action != prev_action:
            reward += self.action_reward

        # ============ 推进时间 ============
        self._cursor += 1
        terminated = self._cursor >= len(self.df) or self._cursor >= self.max_steps
        truncated = False

        info = {"account_value": self.account_value}

        if self.render_mode == "human":
            self._render_step(price_raw, reward)

        return self._get_observation(), float(reward), terminated, truncated, info

    # ==================================================================
    # Internal helpers
    # ==================================================================
    def _reset_internal_state(self):
        self._cursor = self.window_size
        self.cash = self.init_cash
        self.position = 0.0
        self.account_value = self.init_cash
        self.last_action = 0        # 初始视为“保持”
        self.trade_log: List[Tuple] = []

    def _get_observation(self) -> np.ndarray:
        window = self.df.iloc[self._cursor - self.window_size : self._cursor]
        obs = window.drop(columns="timestamp").values.flatten().astype(np.float32)
        if np.isnan(obs).any() or np.isinf(obs).any():
            raise ValueError(f"NaN/Inf in obs at idx {self._cursor}")
        return obs

    def _render_step(self, price: float, reward: float):
        print(
            f"t={self._cursor:5d} | price={price:,.2f} | "
            f"cash={self.cash:,.2f} | pos={self.position:.6f} | "
            f"equity={self.account_value:,.2f} | r={reward:+.4f}"
        )
