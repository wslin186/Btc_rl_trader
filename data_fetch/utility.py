# -*- coding: utf-8 -*-
"""
utility.py
--------------------------------------------------
Binance 历史数据下载的通用工具函数

✅ 主要变更
1. 使用 BinanceConst (enums.BinanceConst) 统一常量
2. 提供多线程批量下载 download_batch()，供 download_kline.py 调用
3. 依旧保留 download_file() 逐个下载函数，向后兼容
"""

from __future__ import annotations

import os
import sys
import re
import shutil
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from argparse import (
    ArgumentParser,
    RawTextHelpFormatter,
    ArgumentTypeError,
)

from tqdm import tqdm

from .enums import BinanceConst as C

# ----------------------------------------------------------------------
# 路径 & URL 处理
# ----------------------------------------------------------------------
def get_destination_dir(file_url: str, folder: str | None = None) -> str:
    """
    返回本地保存路径；默认落到项目根目录下 data/raw/
    可通过：
      1. 环境变量 STORE_DIRECTORY
      2. CLI 参数 --folder
    覆盖
    """
    store_directory = folder or os.getenv("STORE_DIRECTORY") \
        or os.path.join(Path(__file__).resolve().parents[1], "data", "raw")
    return os.path.join(store_directory, file_url)


def get_download_url(file_url: str) -> str:
    """拼接成完整下载地址"""
    return f"{C.BASE_URL}{file_url}"


# ----------------------------------------------------------------------
# 交易所元信息
# ----------------------------------------------------------------------
def get_all_symbols(typ: str):
    """
    根据交易类型获取全部交易对：
    - spot: https://api.binance.com
    - um  : USDT‑M  永续
    - cm  : COIN‑M 永续/交割
    """
    url = {
        "um": "https://fapi.binance.com/fapi/v1/exchangeInfo",
        "cm": "https://dapi.binance.com/dapi/v1/exchangeInfo",
    }.get(typ, "https://api.binance.com/api/v3/exchangeInfo")

    resp = urllib.request.urlopen(url).read()
    symbols = [symbol["symbol"] for symbol in json.loads(resp)["symbols"]]
    return symbols


# ----------------------------------------------------------------------
# 下载核心（多线程 & 进度条）
# ----------------------------------------------------------------------
def _download_one(args: tuple):
    """
    内部函数：下载单个文件
    args = (base_path, file_name, date_range, folder)
    返回：
        True  -> 新下载
        False -> 本地已存在
        None  -> 服务器无文件
    """
    base_path, file_name, date_range, folder = args

    # 按日期分子目录（可选）
    if date_range:
        date_range = date_range.replace(" ", "_")
        base_path = os.path.join(base_path, date_range)

    path_on_disk = get_destination_dir(os.path.join(base_path, file_name), folder)

    # 已存在
    if os.path.exists(path_on_disk):
        return False

    # 确保目录
    os.makedirs(os.path.dirname(path_on_disk), exist_ok=True)

    # 真实 URL
    download_url = get_download_url(f"{base_path}{file_name}")

    try:
        with urllib.request.urlopen(download_url) as resp, open(path_on_disk, "wb") as out:
            shutil.copyfileobj(resp, out)
        return True
    except urllib.error.HTTPError:
        # 404 等
        return None


def download_batch(task_args: list[tuple], max_workers: int = 4, desc: str = "Download"):
    """
    批量下载，带进度条
    task_args: 每个元素都是 _download_one 的参数 tuple
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_download_one, task_args),
                            total=len(task_args), desc=desc, ncols=80))
    return results


# ----------------------------------------------------------------------
# 兼容旧脚本的逐个下载实现（保留）
# ----------------------------------------------------------------------
def download_file(base_path: str, file_name: str,
                  date_range: str | None = None, folder: str | None = None):
    """单线程、单文件下载（带进度条 ASCII）"""
    if date_range:
        date_range = date_range.replace(" ", "_")
        base_path = os.path.join(base_path, date_range)

    save_path = get_destination_dir(os.path.join(base_path, file_name), folder)
    if os.path.exists(save_path):
        print(f"\nfile already exists! {save_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    download_url = get_download_url(f"{base_path}{file_name}")
    try:
        with urllib.request.urlopen(download_url) as dl_file:
            length = dl_file.getheader("content-length")
            length = int(length) if length else None
            blocksize = max(4096, length // 100) if length else 8192

            print(f"\nFile Download: {save_path}")
            dl_progress = 0
            with open(save_path, "wb") as out_file:
                while True:
                    buf = dl_file.read(blocksize)
                    if not buf:
                        break
                    dl_progress += len(buf)
                    out_file.write(buf)
                    if length:
                        done = int(50 * dl_progress / length)
                        sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50 - done)))
                        sys.stdout.flush()
            print()
    except urllib.error.HTTPError:
        print(f"\nFile not found: {download_url}")
        pass


# ----------------------------------------------------------------------
# 日期 & CLI 工具函数
# ----------------------------------------------------------------------
def convert_to_date_object(d: str) -> date:
    y, m, day = (int(x) for x in d.split("-"))
    return date(y, m, day)


def get_start_end_date_objects(date_range: str):
    start, end = date_range.split()
    return convert_to_date_object(start), convert_to_date_object(end)


def match_date_regex(arg_value, pat=re.compile(r"\d{4}-\d{2}-\d{2}")):
    if not pat.match(arg_value):
        raise ArgumentTypeError("日期格式应为 YYYY-MM-DD")
    return arg_value


def check_directory(arg_value: str):
    """CLI 参数 --folder 的合法性检查（若已存在则询问是否覆盖）"""
    if os.path.exists(arg_value):
        while True:
            option = input("Folder already exists! Do you want to overwrite it? y/n  ")
            if option not in ("y", "n"):
                print("Invalid Option!")
                continue
            elif option == "y":
                shutil.rmtree(arg_value)
            break
    return arg_value


def raise_arg_error(msg: str):
    raise ArgumentTypeError(msg)


def get_path(trading_type: str, market_data_type: str,
             time_period: str, symbol: str, interval: str | None = None) -> str:
    """
    生成远端存储路径：
      spot:   data/spot/monthly/klines/BTCUSDT/1h/
      um/cm:  data/futures/um/...
    """
    trading_type_path = "data/spot"
    if trading_type != "spot":
        trading_type_path = f"data/futures/{trading_type}"

    if interval is not None:
        path = f"{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/"
    else:
        path = f"{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/"
    return path


# ----------------------------------------------------------------------
# CLI 解析器（download_kline.py 调用）
# ----------------------------------------------------------------------
def get_parser(parser_type: str):
    """生成 argparse.ArgumentParser"""
    parser = ArgumentParser(
        description=f"This is a script to download historical {parser_type} data",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument("-s", dest="symbols", nargs="+",
                        help="Single symbol or multiple symbols separated by space")
    parser.add_argument("-y", dest="years", default=C.YEARS, nargs="+", choices=C.YEARS,
                        help="Years to download")
    parser.add_argument("-m", dest="months", default=C.MONTHS, nargs="+", type=int, choices=C.MONTHS,
                        help="Months to download")
    parser.add_argument("-d", dest="dates", nargs="+", type=match_date_regex,
                        help="Specific dates YYYY-MM-DD")
    parser.add_argument("-startDate", dest="startDate", type=match_date_regex,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("-endDate", dest="endDate", type=match_date_regex,
                        help="End date YYYY-MM-DD")
    parser.add_argument("-folder", dest="folder", type=check_directory,
                        help="Destination directory")
    parser.add_argument("-skip-monthly", dest="skip_monthly",
                        default=0, type=int, choices=[0, 1],
                        help="1 to skip monthly data")
    parser.add_argument("-skip-daily", dest="skip_daily",
                        default=0, type=int, choices=[0, 1],
                        help="1 to skip daily data")
    parser.add_argument("-c", dest="checksum",
                        default=0, type=int, choices=[0, 1],
                        help="1 to download checksum file")
    parser.add_argument("-t", dest="type", required=True,
                        choices=C.TRADING_TYPE,
                        help=f"Trading types: {C.TRADING_TYPE}")

    if parser_type == "klines":
        parser.add_argument("-i", dest="intervals",
                            default=C.INTERVALS, nargs="+", choices=C.INTERVALS,
                            help="Kline intervals")

    return parser
