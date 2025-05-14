# -*- coding: utf-8 -*-
"""
Binance 批量 Kline 下载工具
用法：
  python -m data_fetch.download_kline -t spot -s BTCUSDT -i 1h 4h -startDate 2020-01-01 -endDate 2023-12-31 -j 6
作为函数：
  from data_fetch.download_kline import download_klines
"""
from __future__ import annotations

import sys
from datetime import datetime, date
from pathlib import Path
from itertools import product, chain
from typing import Sequence, List

from .utility import (
    get_parser, convert_to_date_object, get_path, download_batch, get_all_symbols
)
from .enums import BinanceConst as C

# -------------- 业务函数（供脚本和其他模块复用） ------------------
def download_klines(
    trading_type: str,
    symbols: Sequence[str],
    intervals: Sequence[str],
    start_date: str | date,
    end_date: str | date,
    folder: str | None = None,
    daily: bool = True,
    monthly: bool = True,
    checksum: bool = False,
    jobs: int = 4,
):
    """下载 klines，日期闭区间 [start_date, end_date]"""
    start = convert_to_date_object(str(start_date)) if isinstance(start_date, str) else start_date
    end   = convert_to_date_object(str(end_date))   if isinstance(end_date, str)   else end_date

    years  = [str(y) for y in range(start.year, end.year + 1)]
    months = list(range(1, 13))
    dates  = [d.strftime("%Y-%m-%d") for d in
              (date.fromordinal(d) for d in range(start.toordinal(), end.toordinal()+1))]

    task_args: list[tuple] = []

    # -------- monthly --------
    if monthly:
        for sym, inter, y, m in product(symbols, intervals, years, months):
            cur = date(int(y), int(m), 1)
            if start <= cur <= end:
                base = get_path(trading_type, "klines", "monthly", sym, inter)
                fname = f"{sym.upper()}-{inter}-{y}-{m:02d}.zip"
                task_args.append((base, fname, None, folder))
                if checksum:
                    task_args.append((base, f"{fname}.CHECKSUM", None, folder))

    # -------- daily ----------
    if daily:
        valid_intervals = set(intervals) & set(C.DAILY_INTERVALS)
        for sym, inter, d in product(symbols, valid_intervals, dates):
            cur = convert_to_date_object(d)
            if start <= cur <= end:
                base = get_path(trading_type, "klines", "daily", sym, inter)
                fname = f"{sym.upper()}-{inter}-{d}.zip"
                task_args.append((base, fname, None, folder))
                if checksum:
                    task_args.append((base, f"{fname}.CHECKSUM", None, folder))

    downloaded = download_batch(task_args, max_workers=jobs,
                                desc=f"Klines {trading_type} ({len(task_args)} files)")
    ok = sum(x is True for x in downloaded)
    miss = sum(x is None for x in downloaded)
    print(f"\n✅ 完成: 新下 {ok} 个, 服务器无文件 {miss} 个, 已存在 {len(downloaded)-ok-miss} 个。")

# ----------------------- CLI 模式 -----------------------------
def cli_main(argv: list[str] | None = None):
    parser = get_parser('klines')
    parser.add_argument('-j', dest='jobs', type=int, default=4, help='并行下载线程数')
    args = parser.parse_args(argv or sys.argv[1:])

    symbols = args.symbols or get_all_symbols(args.type)
    download_klines(
        trading_type=args.type,
        symbols=symbols,
        intervals=args.intervals,
        start_date=args.startDate or C.PERIOD_START_DATE,
        end_date=args.endDate or datetime.today().strftime("%Y-%m-%d"),
        folder=args.folder,
        daily=(args.skip_daily == 0),
        monthly=(args.skip_monthly == 0),
        checksum=bool(args.checksum),
        jobs=args.jobs,
    )


if __name__ == "__main__":
    cli_main()
