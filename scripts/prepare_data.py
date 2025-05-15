# -*- coding: utf-8 -*-
"""
prepare_data.py
--------------------------------------------------
读取 Binance .zip / .csv（无表头），合并 BTCUSDT 1h K 线：
1. 递归处理 monthly + daily 目录
2. 过滤异常时间戳、NaN/Inf
3. 添加多种技术指标
4. 统计数据质量与分布
5. 保存到 data/processed/btc.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # 获取项目根目录
sys.path.insert(0, str(ROOT))  # 添加到Python路径

import os
import time
from itertools import chain
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.feature_engineering import add_indicators, normalize_features
from utils.logger import get_logger

logger = get_logger("prepare_data")

# ----------------------------------------------------------------------
# 路径 & 配置
# ----------------------------------------------------------------------
RAW_DIR = Path("data/raw/data/spot")          # 下载脚本默认存放根目录
OUT_PATH = Path("data/processed/btc.parquet")
STATS_DIR = Path("data/stats")                # 数据统计目录
SYMBOLS = ["BTCUSDT"]                         # 多币种可扩展此列表

COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]
DTYPES = {c: "Int64" if c.endswith("_time") else "float64" for c in COLS}
NEEDED = {"open_time", "open", "high", "low", "close", "volume"}

# 合理时间戳范围（毫秒）
T_MIN, T_MAX = 1_483_228_800_000, 4_107_926_400_000  # 2017‑01‑01 ~ 2100‑01‑01

# CHUNK_SIZE = 100_000  # 分块处理大文件，提高内存效率


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def read_kline(path: Path) -> pd.DataFrame:
    """读取单个 ZIP/CSV，无表头 -> names=COLS，支持大文件优化"""
    try:
        # 性能优化：添加low_memory=False减少数据类型推断的开销
        return pd.read_csv(
            path,
            header=None,
            names=COLS,
            compression="zip" if path.suffix == ".zip" else None,
            dtype=DTYPES,
            na_values=["", "nan", "null", "None"],
            low_memory=False,
        )
    except Exception as e:
        logger.error(f"❌ 读取文件 {path.name} 失败: {e}")
        raise


def analyze_data_quality(df: pd.DataFrame, save_dir: Path) -> dict:
    """分析数据质量，生成统计报告和可视化"""
    save_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    
    # 1. 基本统计
    stats["总行数"] = len(df)
    stats["起始日期"] = df["timestamp"].min().strftime("%Y-%m-%d")
    stats["结束日期"] = df["timestamp"].max().strftime("%Y-%m-%d")
    stats["时间区间(天)"] = (df["timestamp"].max() - df["timestamp"].min()).days
    stats["每日K线数(平均)"] = len(df) / stats["时间区间(天)"] if stats["时间区间(天)"] > 0 else 0
    
    # 2. 可视化价格趋势
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["close"])
    plt.title("BTC 价格走势")
    plt.xlabel("日期")
    plt.ylabel("价格 (USDT)")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "price_trend.png", dpi=150)
    plt.close()
    
    # 3. 可视化交易量趋势
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["volume"], color="orange")
    plt.title("BTC 交易量走势")
    plt.xlabel("日期")
    plt.ylabel("交易量")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "volume_trend.png", dpi=150)
    plt.close()
    
    # 4. 检查连续性 - 查找时间缺口
    df_sorted = df.sort_values("timestamp")
    df_sorted["next_ts"] = df_sorted["timestamp"].shift(-1)
    df_sorted["gap_hours"] = (df_sorted["next_ts"] - df_sorted["timestamp"]).dt.total_seconds() / 3600
    
    # 筛选超过预期间隔的缺口（对于1h数据，应该是1小时）
    gaps = df_sorted[df_sorted["gap_hours"] > 1.5]
    if len(gaps) > 0:
        stats["数据缺口数"] = len(gaps)
        stats["最大缺口(小时)"] = gaps["gap_hours"].max()
        
        # 记录明显的缺口
        with open(save_dir / "data_gaps.csv", "w") as f:
            f.write("start_time,end_time,gap_hours\n")
            for _, row in gaps.iterrows():
                f.write(f"{row['timestamp']},{row['next_ts']},{row['gap_hours']:.1f}\n")
    else:
        stats["数据缺口数"] = 0
        stats["最大缺口(小时)"] = 0
    
    # 5. 生成统计报告
    with open(save_dir / "data_stats.txt", "w", encoding="utf-8") as f:
        f.write("=== BTC 数据集统计 ===\n\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    
    return stats


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    start_time = time.time()
    
    # 1. 收集文件 (.zip + .csv)
    files = [
        f for f in chain(RAW_DIR.rglob("*.zip"), RAW_DIR.rglob("*.csv"))
        if any(sym in f.name for sym in SYMBOLS)
    ]
    logger.info(f"🔍 匹配到文件数: {len(files)}")
    if not files:
        logger.error(f"❌ 未找到 {SYMBOLS} 的 zip/csv")
        return

    # 2. 读取 & 合并数据
    frames = []
    for f in tqdm(files, desc="读取文件"):
        try:
            df = read_kline(f)
            
            # 筛选所需列
            if not NEEDED.issubset(df.columns):
                logger.warning(f"⚠️ 文件 {f.name} 缺少必要列: {NEEDED}")
                continue
                
            # 过滤异常时间戳
            df = df[(df["open_time"] >= T_MIN) & (df["open_time"] <= T_MAX)]
            
            # 取必要列，转换时间戳
            df = df[["open_time", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df = df.drop(columns=["open_time"])
            
            frames.append(df)
        except Exception as e:
            logger.error(f"❌ 处理文件 {f.name} 失败: {e}")
    
    if not frames:
        logger.error("❌ 没有有效数据帧")
        return
        
    # 3. 合并 & 排序
    logger.info("📊 合并数据帧...")
    df_all = pd.concat(frames, ignore_index=True)
    
    # 删除重复记录
    n_before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    n_after = len(df_all)
    if n_before > n_after:
        logger.info(f"🧹 已删除 {n_before - n_after} 条重复记录")
    
    # 按时间排序
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"✅ 完成! 总行数: {len(df_all)}")
    
    # 数据基本情况
    logger.info(f"📅 数据区间: {df_all['timestamp'].min()} ~ {df_all['timestamp'].max()}")
    
    # 4. 添加技术指标
    logger.info("🔄 添加技术指标...")
    df_with_indicators = add_indicators(df_all)
    logger.info(f"✅ 添加指标后行数: {len(df_with_indicators)}")
    
    # 增加特征归一化步骤
    logger.info("🔄 对技术指标进行归一化处理...")
    # 排除原始价格和交易量列，只标准化衍生指标
    exclude_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df_normalized = normalize_features(df_with_indicators, method='minmax', exclude_cols=exclude_cols)
    
    # 5. 保存处理后的数据
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_normalized.to_parquet(OUT_PATH, compression="snappy")
    logger.info(f"💾 数据已保存: {OUT_PATH}")
    
    # 6. 生成数据质量报告
    logger.info("📝 生成数据质量报告...")
    stats = analyze_data_quality(df_normalized, STATS_DIR)
    
    # 打印一些关键特征的统计信息
    logger.info("📊 特征统计信息:")
    for col in ["return_1d", "rsi", "macd"]:
        if col in df_normalized.columns:
            logger.info(f"  - {col}: 均值={df_normalized[col].mean():.4f}, 标准差={df_normalized[col].std():.4f}")
    
    elapsed = time.time() - start_time
    logger.info(f"🎉 数据处理完成! 用时: {elapsed:.2f}秒")
    logger.info(f"💡 数据统计: 行数={stats['总行数']}, 时间区间={stats['时间区间(天)']}天")


if __name__ == "__main__":
    main()