from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd

from core.market_rules import detect_board, detect_st, limit_pct_for


PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]


def _read_txt(path: str) -> pd.DataFrame | None:
    encodings = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]
    for encoding in encodings:
        try:
            df = pd.read_csv(path, sep=r"\s+|\t+", engine="python", skiprows=1, header=None, encoding=encoding)
            if df.shape[1] < 6:
                continue
            cols = ["date", "open", "high", "low", "close", "volume", "amount"][: df.shape[1]]
            df = df.iloc[:, : len(cols)]
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
        except Exception:
            continue
    return None


def _has_basic_anomaly(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    if (df["open"] <= 0).any() or (df["high"] <= 0).any() or (df["low"] <= 0).any() or (df["close"] <= 0).any():
        return True
    if ((df["high"] < df["low"]) | (df["open"] > df["high"]) | (df["open"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])).any():
        return True
    return False


def load_price_directory(data_dir: str) -> Tuple[Dict[str, pd.DataFrame], List[pd.Timestamp]]:
    stock_data: Dict[str, pd.DataFrame] = {}
    all_dates = set()

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue
        code = filename[:-4]
        path = os.path.join(data_dir, filename)
        df = _read_txt(path)
        if df is None or len(df) < 30:
            continue
        if _has_basic_anomaly(df):
            continue

        df["code"] = code
        df["board"] = detect_board(code)
        df["is_st"] = detect_st(code)
        df["limit_pct"] = df["date"].apply(lambda dt: limit_pct_for(code, dt, bool(df["is_st"].iloc[0])))
        df["is_suspended"] = (df["volume"].fillna(0) <= 0) | (df["open"].fillna(0) <= 0)
        df = df.set_index("date")
        stock_data[code] = df
        all_dates.update(df.index.tolist())

    return stock_data, sorted(all_dates)
