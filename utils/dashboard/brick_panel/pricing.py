from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .paths import DATA_DIR
except ImportError:  # pragma: no cover - for direct script execution
    from paths import DATA_DIR


def _iter_date_dirs() -> list[Path]:
    dirs = [p for p in DATA_DIR.glob("20*") if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def _candidate_paths(date_dir: Path, code: str) -> list[Path]:
    return [
        date_dir / f"SZ#{code}.txt",
        date_dir / f"SH#{code}.txt",
        date_dir / f"BJ#{code}.txt",
        date_dir / "normal" / f"SZ#{code}.txt",
        date_dir / "normal" / f"SH#{code}.txt",
        date_dir / "normal" / f"BJ#{code}.txt",
    ]


@lru_cache(maxsize=12000)
def find_stock_file(code: str) -> Path | None:
    code = str(code).zfill(6)
    for date_dir in _iter_date_dirs():
        for candidate in _candidate_paths(date_dir, code):
            if candidate.exists():
                return candidate
    return None


def _read_lines(path: Path) -> list[str]:
    encodings = ("gbk", "gb2312", "utf-8", "latin-1")
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as fh:
                return [line.rstrip("\n") for line in fh if line.strip()]
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


@lru_cache(maxsize=12000)
def get_stock_name(code: str) -> str:
    code = str(code).zfill(6)
    for date_dir in _iter_date_dirs():
        for candidate in [
            date_dir / f"SZ#{code}.txt",
            date_dir / f"SH#{code}.txt",
            date_dir / f"BJ#{code}.txt",
        ]:
            if not candidate.exists():
                continue
            lines = _read_lines(candidate)
            if not lines:
                continue
            first = " ".join(lines[0].split())
            parts = first.split(" ")
            if len(parts) >= 2 and "/" not in parts[1] and parts[1] not in {"开盘", "最高", "最低", "收盘"}:
                return parts[1]
    file_path = find_stock_file(code)
    if file_path is None:
        return ""
    lines = _read_lines(file_path)
    if not lines:
        return ""
    first = " ".join(lines[0].split())
    parts = first.split(" ")
    if len(parts) >= 2 and "/" not in parts[1] and parts[1] not in {"开盘", "最高", "最低", "收盘"}:
        return parts[1]
    return ""


@lru_cache(maxsize=12000)
def load_price_df(code: str) -> pd.DataFrame:
    file_path = find_stock_file(code)
    if file_path is None:
        return pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"])
    lines = _read_lines(file_path)
    rows: list[list[str]] = []
    for line in lines:
        if any(keyword in line for keyword in ("日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额")):
            continue
        clean = " ".join(line.replace("\t", " ").split())
        parts = clean.split(" ")
        if len(parts) == 7 and "/" in parts[0]:
            rows.append(parts)
    if not rows:
        return pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"])
    df = pd.DataFrame(rows, columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"])
    df["日期"] = pd.to_datetime(df["日期"], format="%Y/%m/%d", errors="coerce")
    for col in ["开盘", "最高", "最低", "收盘", "成交量", "成交额"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["日期", "开盘", "最高", "最低", "收盘"]).reset_index(drop=True)
    return df


def _with_atr20(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    prev_close = out["收盘"].shift(1)
    tr = pd.concat(
        [
            out["最高"] - out["最低"],
            (out["最高"] - prev_close).abs(),
            (out["最低"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR20"] = tr.rolling(20, min_periods=20).mean()
    return out


@lru_cache(maxsize=12000)
def latest_snapshot(code: str) -> dict[str, Any]:
    df = _with_atr20(load_price_df(str(code).zfill(6)))
    if df.empty:
        return {
            "code": str(code).zfill(6),
            "name": "",
            "latest_close": np.nan,
            "latest_price_date": None,
            "atr20": np.nan,
        }
    last = df.iloc[-1]
    return {
        "code": str(code).zfill(6),
        "name": get_stock_name(str(code).zfill(6)),
        "latest_close": float(last["收盘"]),
        "latest_price_date": pd.Timestamp(last["日期"]).strftime("%Y-%m-%d"),
        "atr20": float(last["ATR20"]) if pd.notna(last["ATR20"]) else np.nan,
    }


def get_signal_bar(code: str, signal_date: str) -> dict[str, Any] | None:
    df = load_price_df(str(code).zfill(6))
    if df.empty:
        return None
    ts = pd.Timestamp(signal_date)
    matched = df[df["日期"] == ts]
    if matched.empty:
        return None
    row = matched.iloc[-1]
    return {
        "code": str(code).zfill(6),
        "name": get_stock_name(str(code).zfill(6)),
        "signal_close": float(row["收盘"]),
        "signal_low": float(row["最低"]),
        "signal_date": ts.strftime("%Y-%m-%d"),
    }


def next_open_after_signal(code: str, signal_date: str) -> tuple[str | None, float | None]:
    df = load_price_df(str(code).zfill(6))
    if df.empty:
        return None, None
    ts = pd.Timestamp(signal_date)
    future = df[df["日期"] > ts]
    if future.empty:
        return None, None
    row = future.iloc[0]
    return pd.Timestamp(row["日期"]).strftime("%Y-%m-%d"), float(row["开盘"])


def trading_days_between(code: str, start_date: str, end_date: str) -> int:
    df = load_price_df(str(code).zfill(6))
    if df.empty:
        return 0
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    subset = df[(df["日期"] >= start_ts) & (df["日期"] <= end_ts)]
    return int(len(subset))


def enrich_positions(positions_df: pd.DataFrame) -> pd.DataFrame:
    if positions_df.empty:
        return positions_df.copy()
    rows = []
    for row in positions_df.to_dict("records"):
        snap = latest_snapshot(row["code"])
        latest_close = snap["latest_close"]
        latest_date = snap["latest_price_date"]
        atr20 = snap["atr20"]
        entry_price = float(row["entry_price"])
        current_return = np.nan
        atr_tp_price = np.nan
        if pd.notna(latest_close):
            current_return = (latest_close - entry_price) / entry_price
        if pd.notna(atr20):
            atr_tp_price = entry_price + atr20 * 2.0
        holding_days = trading_days_between(row["code"], row["entry_date"], latest_date) if latest_date else 0
        rows.append(
            {
                **row,
                "display_name": row["name"] or snap["name"] or row["code"],
                "latest_close": latest_close,
                "latest_price_date": latest_date,
                "atr20": atr20,
                "current_return": current_return,
                "atr_tp_price": atr_tp_price,
                "fixed_tp_3_price": entry_price * 1.03,
                "fixed_tp_8_price": entry_price * 1.08,
                "stop_loss_price": float(row["entry_signal_low"]),
                "holding_days": holding_days,
                "warning_over_3d": holding_days > 3,
                "unrealized_pnl_amount": (latest_close - entry_price) * float(row["quantity"]) if pd.notna(latest_close) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_basket_snapshot(signal_date: str, strategy: str, codes: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    members: list[dict[str, Any]] = []
    returns: list[float] = []
    latest_price_date: str | None = None
    for code in sorted({str(c).zfill(6) for c in codes}):
        entry_date, entry_price = next_open_after_signal(code, signal_date)
        snap = latest_snapshot(code)
        latest_close = snap["latest_close"]
        if entry_date is None or entry_price is None or pd.isna(latest_close):
            continue
        latest_price_date = snap["latest_price_date"] or latest_price_date
        ret = (latest_close - entry_price) / entry_price
        returns.append(ret)
        members.append(
            {
                "signal_date": signal_date,
                "strategy": strategy,
                "code": code,
                "name": snap["name"] or code,
                "entry_date": entry_date,
                "entry_price": float(entry_price),
                "latest_close": float(latest_close),
                "latest_price_date": latest_price_date,
                "return_to_latest_close": float(ret),
            }
        )
    summary = {
        "signal_date": signal_date,
        "strategy": strategy,
        "stock_count": len(members),
        "avg_return_to_latest_close": float(np.mean(returns)) if returns else None,
        "latest_price_date": latest_price_date,
    }
    return summary, members
