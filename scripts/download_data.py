# -*- coding: utf-8 -*-
"""
读取 config.yaml 的 download 节点，批量下载 Binance K 线
------------------------------------------------------------
用法：
    python scripts/download_data.py
"""
from __future__ import annotations
import yaml
from pathlib import Path

from data_fetch.download_kline import download_klines
from utils.logger import get_logger

logger = get_logger("download")

CONFIG_PATH = Path("config/config.yaml")


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dl = cfg["download"]

    logger.info(
        f"开始下载 {dl['symbols']} {dl['intervals']} "
        f"{dl['start_date']} ▶ {dl['end_date']}"
    )

    download_klines(
        trading_type = dl["trading_type"],
        symbols      = dl["symbols"],
        intervals    = dl["intervals"],
        start_date   = dl["start_date"],
        end_date     = dl["end_date"],
        folder       = None,                 # 可在 YAML 加字段自定义
        daily        = dl["skip_daily"]   == 0,
        monthly      = dl["skip_monthly"] == 0,
        checksum     = bool(dl["checksum"]),
        jobs         = dl["threads"],
    )

    logger.info("✅ 数据下载完成")


if __name__ == "__main__":
    main()
