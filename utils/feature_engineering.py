# utils/feature_engineering.py
# --------------------------------------------------------------
# 给 K 线 DataFrame 添加技术指标，并确保最终无 NaN/Inf
# --------------------------------------------------------------
import pandas as pd
import numpy as np
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """返回包含 RSI / MACD / EMA 的新数据框，已去除所有 NaN/Inf"""
    df = df.copy()

    # === 示例指标 ===
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd_diff()
    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)

    # === 清洗 ===
    df = df.ffill()               # 前向填充
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df
