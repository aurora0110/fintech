from __future__ import annotations

import os
from typing import Dict

import pandas as pd

from core.market_rules import detect_board

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _read_price_file(path: str) -> pd.DataFrame | None:
    encodings = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]
    for encoding in encodings:
        try:
            df = pd.read_csv(path, sep=r"\s+|\t+", engine="python", skiprows=1, header=None, encoding=encoding)
        except Exception:
            continue

        if df.shape[1] < 6:
            continue

        cols = ["date", "open", "high", "low", "close", "volume", "amount"][: df.shape[1]]
        df = df.iloc[:, : len(cols)].copy()
        df.columns = cols
        if "amount" not in df.columns:
            df["amount"] = 0.0

        df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
        if df.empty:
            continue

        df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        return df

    return None


def _valid_price_frame(df: pd.DataFrame) -> bool:
    if df.empty or len(df) < 150:
        return False
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        return False
    price_invalid = (
        (df["high"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
    )
    return not bool(price_invalid.any())


def load_stock_directory(data_dir: str) -> Dict[str, pd.DataFrame]:
    stock_data: Dict[str, pd.DataFrame] = {}
    for filename in tqdm(sorted(os.listdir(data_dir)), desc="Loading price files", unit="file"):
        if not filename.endswith(".txt"):
            continue
        code = filename[:-4]
        path = os.path.join(data_dir, filename)
        df = _read_price_file(path)
        if df is None or not _valid_price_frame(df):
            continue
        df["code"] = code
        df["board"] = detect_board(code)
        stock_data[code] = df
    return stock_data
