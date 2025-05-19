# utils/feature_engineering.py
# --------------------------------------------------------------
# 给 K 线 DataFrame 添加核心技术指标，并确保最终无 NaN/Inf
# --------------------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # 获取项目根目录
sys.path.insert(0, str(ROOT))  # 添加到Python路径

import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger("feature_engineering")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """返回包含核心技术指标的新数据框，已去除所有 NaN/Inf"""
    df = df.copy()

    # === 价格收益率指标 ===
    df["return_1d"] = df["close"].pct_change(periods=1)              # 日收益率
    df["return_5d"] = df["close"].pct_change(periods=5)              # 5日收益率
    
    # === 振幅指标 ===
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14)               # ATR波动率
    
    # 简化版布林带
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()  # 布林带宽度
    
    # === 成交量变化 ===
    df["vol_ema"] = ta.trend.ema_indicator(df["volume"], window=20)  # 成交量EMA
    df["vol_ratio"] = df["volume"] / df["vol_ema"]                   # 相对成交量
    
    # === 均线系统 ===
    df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)      # 短期EMA
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)    # 长期EMA
    df["ema_diff"] = (df["ema_9"] - df["ema_50"]) / df["close"]      # 均线差值(归一化)
    
    # === 趋势指标 ===
    # 简化版MACD
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd_diff()                                    # MACD柱状图
    
    # 趋势强度
    df["adx"] = ta.trend.ADXIndicator(
        df["high"], df["low"], df["close"], window=14).adx()         # ADX趋势强度
    
    # === RSI ===
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)              # RSI

    # === 清洗 ===
    # 使用直接方法而不是已弃用的fillna(method=...)
    df = df.ffill()                                                  # 前向填充
    df = df.bfill()                                                  # 后向填充
    df = df.replace([np.inf, -np.inf], np.nan)                       # 替换无穷值
    df = df.dropna().reset_index(drop=True)                          # 删除剩余NaN

    # 使用日志记录器替换print
    n_features = len(df.columns) - 6  # 减去OHLCV和timestamp
    logger.info(f"📊 共添加 {n_features} 个核心技术指标")

    return df


def normalize_features(df: pd.DataFrame, method='minmax', exclude_cols=None) -> pd.DataFrame:
    """
    对特征进行归一化或标准化处理
    
    参数:
        df: 输入DataFrame
        method: 'minmax'表示Min-Max归一化，'standard'表示Z-score标准化
        exclude_cols: 不需要处理的列名列表
    
    返回:
        处理后的DataFrame
    """
    if exclude_cols is None:
        exclude_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    
    # 确定需要处理的列
    columns_to_process = [col for col in df.columns if col not in exclude_cols]
    if not columns_to_process:
        logger.warning("⚠️ 没有需要处理的列")
        return df
    
    # 拷贝输入数据
    df_processed = df.copy()
    feature_data = df_processed[columns_to_process].values
    
    # 选择处理方法
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        logger.info(f"🔄 使用Min-Max归一化处理 {len(columns_to_process)} 个特征")
    else:  # 'standard'
        scaler = StandardScaler()
        logger.info(f"🔄 使用Z-score标准化处理 {len(columns_to_process)} 个特征")
    
    # 归一化/标准化
    try:
        normalized_data = scaler.fit_transform(feature_data)
        df_processed[columns_to_process] = normalized_data
        logger.info("✅ 特征处理完成")
    except Exception as e:
        logger.error(f"❌ 特征处理失败: {str(e)}")
    
    return df_processed


def standardize_features(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    """对特征进行Z-score标准化的便捷函数"""
    return normalize_features(df, method='standard', exclude_cols=exclude_cols)
