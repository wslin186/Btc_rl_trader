# utils/data_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
import datetime as dt


def _to_ts(x):
    if x is None:
        return None
    if isinstance(x, (pd.Timestamp, dt.date, dt.datetime)):
        return pd.to_datetime(x)
    return pd.to_datetime(str(x))


def _filter(df: pd.DataFrame, s, e):
    if s is not None:
        df = df[df["timestamp"] >= s]
    if e is not None:
        df = df[df["timestamp"] <= e]
    return df


def load_btc_data(
    processed_path: Path,
    train_ratio: float | None = None,
    train_start: str | dt.date | None = None,
    train_end: str | dt.date | None = None,
    test_start: str | dt.date | None = None,
    test_end: str | dt.date | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.read_parquet(processed_path).sort_values("timestamp").reset_index(drop=True)

    # —— 显式日期切分 ——
    if all([train_start, train_end, test_start, test_end]):
        train_df = _filter(df, _to_ts(train_start), _to_ts(train_end))
        test_df  = _filter(df, _to_ts(test_start),  _to_ts(test_end))
        return train_df, test_df

    # —— 比例切分 ——
    if train_ratio is None:
        train_ratio = 0.8
    split = int(len(df) * float(train_ratio))
    return df.iloc[:split], df.iloc[split:]
