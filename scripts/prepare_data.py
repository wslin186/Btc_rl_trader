# -*- coding: utf-8 -*-
"""
prepare_data.py
--------------------------------------------------
读取 Binance .zip / .csv（无表头），合并 BTCUSDT 1h K 线：
1. 递归处理 monthly + daily 目录
2. 过滤异常时间戳、NaN/Inf
3. 添加 RSI / MACD / EMA
4. 保存到 data/processed/btc.parquet
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.feature_engineering import add_indicators
from utils.logger import get_logger

logger = get_logger("prepare_data")

# ----------------------------------------------------------------------
# 路径 & 配置
# ----------------------------------------------------------------------
RAW_DIR = Path("data/raw/data/spot")          # 下载脚本默认存放根目录
OUT_PATH = Path("data/processed/btc.parquet")
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


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def read_kline(path: Path) -> pd.DataFrame:
    """读取单个 ZIP/CSV，无表头 -> names=COLS"""
    return pd.read_csv(
        path,
        header=None,
        names=COLS,
        compression="zip" if path.suffix == ".zip" else None,
        dtype=DTYPES,
        na_values=["", "nan"],
    )


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    # 1. 收集文件 (.zip + .csv)
    files = [
        f for f in chain(RAW_DIR.rglob("*.zip"), RAW_DIR.rglob("*.csv"))
        if any(sym in f.name for sym in SYMBOLS)
    ]
    logger.info(f"DEBUG | 匹配到文件数: {len(files)}")
    if not files:
        logger.error(f"❌ 未找到 {SYMBOLS} 的 zip/csv")
        return

    # 2. 读取并基本校验
    dfs: list[pd.DataFrame] = []
    total_rows, skipped = 0, 0

    for f in tqdm(files, desc="Reading files", ncols=80):
        try:
            df = read_kline(f)
        except Exception as e:
            logger.warning(f"⚠️  读取失败 {f.name}: {e}")
            skipped += 1
            continue

        if df.empty or not NEEDED.issubset(df.columns):
            logger.warning(f"⚠️  跳过空表/列缺失: {f.name}")
            skipped += 1
            continue

        dfs.append(df)
        total_rows += len(df)
        if total_rows % 1_000_000 < len(df):
            logger.info(f"💾 已累积 {total_rows:,} 行…")

    if not dfs:
        logger.error("❌ 所有文件被跳过，无法继续")
        return

    # 3. 合并
    df = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"📊 有效文件 {len(dfs)}/{len(files)}，拼接后 {len(df):,} 行"
    )

    # 4. 过滤异常时间戳并整理列
    df = df[df["open_time"].between(T_MIN, T_MAX)]
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], unit="ms", errors="coerce"
    )
    df = df.dropna(subset=["timestamp"])

    # 5. 技术指标 + 清洗 (函数已确保无 NaN/Inf)
    df = add_indicators(df)

    # 6. 保存
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info(f"✅ Saved processed data to {OUT_PATH}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
