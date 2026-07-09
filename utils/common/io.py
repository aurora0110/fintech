from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DATE_COL_CANDIDATES = ["date", "Date", "trade_date", "日期", "DATE"]
OPEN_COL_CANDIDATES = ["open", "Open", "开盘", "OPEN"]
HIGH_COL_CANDIDATES = ["high", "High", "最高", "HIGH"]
LOW_COL_CANDIDATES = ["low", "Low", "最低", "LOW"]
CLOSE_COL_CANDIDATES = ["close", "Close", "收盘", "CLOSE"]
VOL_COL_CANDIDATES = ["volume", "vol", "Volume", "成交量", "VOL"]
AMOUNT_COL_CANDIDATES = ["amount", "Amount", "成交额", "AMOUNT"]
CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码", "CODE"]


def extract_stock_code_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    return stem.split("#")[-1]


def read_stock_name_from_file(path: str | Path) -> str:
    raw_path = Path(path)
    if raw_path.parent.name == "normal":
        candidate = raw_path.parent.parent / raw_path.name
        if candidate.exists():
            raw_path = candidate
    first_line = ""
    for encoding in ("gbk", "gb2312", "utf-8", "latin-1"):
        try:
            with open(raw_path, "r", encoding=encoding, errors="ignore") as f:
                first_line = f.readline().strip()
            break
        except Exception:
            first_line = ""
    if not first_line:
        return ""
    parts = first_line.split()
    if len(parts) >= 2:
        return parts[1].strip()
    return "".join(c for c in first_line if "\u4e00" <= c <= "\u9fff").strip()


def pick_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"missing required columns from candidates={list(candidates)}")
    return None


def read_table_auto(path: str | Path) -> pd.DataFrame:
    path = str(path)
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+|\t+", engine="python")


def standardize_ohlcv_columns(
    raw: pd.DataFrame,
    *,
    path: str | Path | None = None,
    amount_default: float = np.nan,
) -> pd.DataFrame:
    date_col = pick_col(raw, DATE_COL_CANDIDATES)
    open_col = pick_col(raw, OPEN_COL_CANDIDATES)
    high_col = pick_col(raw, HIGH_COL_CANDIDATES)
    low_col = pick_col(raw, LOW_COL_CANDIDATES)
    close_col = pick_col(raw, CLOSE_COL_CANDIDATES)
    vol_col = pick_col(raw, VOL_COL_CANDIDATES)
    amount_col = pick_col(raw, AMOUNT_COL_CANDIDATES, required=False)
    code_col = pick_col(raw, CODE_COL_CANDIDATES, required=False)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "open": pd.to_numeric(raw[open_col], errors="coerce"),
            "high": pd.to_numeric(raw[high_col], errors="coerce"),
            "low": pd.to_numeric(raw[low_col], errors="coerce"),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
            "volume": pd.to_numeric(raw[vol_col], errors="coerce"),
            "amount": pd.to_numeric(raw[amount_col], errors="coerce") if amount_col else amount_default,
        }
    )
    if code_col:
        df["code"] = raw[code_col].astype(str).iloc[0]
    else:
        df["code"] = extract_stock_code_from_filename(path or "")
    return df


def read_stock_daily(
    path: str | Path,
    *,
    min_bars: int = 0,
    allow_zero_volume: bool = True,
    amount_default: float = np.nan,
) -> Optional[pd.DataFrame]:
    raw = read_table_auto(path)
    df = standardize_ohlcv_columns(raw, path=path, amount_default=amount_default)
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    volume_mask = df["volume"] >= 0 if allow_zero_volume else df["volume"] > 0
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & volume_mask
    ].copy()
    if len(df) < min_bars:
        return None
    return df


def load_all_stock_files(
    data_dir: str | Path,
    *,
    pattern: str = "*",
    min_bars: int = 0,
    allow_zero_volume: bool = True,
) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    result: dict[str, pd.DataFrame] = {}
    for path in sorted(root.glob(pattern)):
        if path.suffix.lower() not in {".csv", ".txt"}:
            continue
        try:
            df = read_stock_daily(path, min_bars=min_bars, allow_zero_volume=allow_zero_volume)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        result[extract_stock_code_from_filename(path)] = df
    return result


def build_code_name_map(file_paths: Iterable[str | Path]) -> dict[str, str]:
    code_name_map: dict[str, str] = {}
    for file_path in file_paths:
        path = Path(file_path)
        code = extract_stock_code_from_filename(path)
        if code in code_name_map:
            continue
        stock_name = read_stock_name_from_file(path)
        code_name_map[code] = stock_name or code
    return code_name_map
