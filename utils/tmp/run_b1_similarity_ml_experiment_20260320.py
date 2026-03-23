from __future__ import annotations

import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "20260315" / "normal"
RAW_DIR = ROOT / "data" / "20260315"
PERFECT_DIR = ROOT / "data" / "完美图" / "B1"
NAME_CODE_CACHE = ROOT / "results" / "b1_name_code_map_cache_20260315.json"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = ROOT / "results" / f"b1_similarity_ml_signal_{RUN_TS}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MIN_BARS = 160
SEQ_LEN = 21
MAX_WORKERS = 8

RESEARCH_RATIO = 0.60
VALIDATION_RATIO = 0.20
FINAL_TEST_RATIO = 0.20

BUY_DELAY_DAYS = 1
MAX_HOLD_DAYS = 60
STOP_LOSS_MULTIPLIER = 0.95
TARGET_RETURNS = [0.10, 0.20, 0.30, 0.40, 0.50]
DAILY_TOPN_LIST = [3, 5, 8, 10]

PERFECT_POSITIVE_MIN_RET20 = 0.0
NEGATIVE_MAX_DRAWDOWN_30 = -0.10
NEGATIVE_MIN_RET20 = -0.05
MAX_NEGATIVE_MULTIPLIER = 5

SIMILARITY_VARIANTS = [
    ("corr", "close_norm"),
    ("corr", "returns"),
    ("corr", "close_vol_concat"),
    ("cosine", "close_norm"),
    ("cosine", "returns"),
    ("cosine", "close_vol_concat"),
    ("euclidean", "close_norm"),
    ("euclidean", "returns"),
    ("euclidean", "close_vol_concat"),
]

CASE_NAME_ALIASES: Dict[str, str] = {
    "亨通光电踩白线": "亨通光电",
}


def safe_div(a: Any, b: Any, default: float = np.nan) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > 1e-12)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    return series.ewm(alpha=m / n, adjust=False).mean()


def zscore_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def minmax_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mx - mn) or abs(mx - mn) < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def compute_slope(series: pd.Series, window: int) -> pd.Series:
    def _slope(arr: np.ndarray) -> float:
        a = np.asarray(arr, dtype=float)
        if len(a) < window or np.any(np.isnan(a)):
            return np.nan
        x = np.arange(window, dtype=float)
        slope, _ = np.polyfit(x, a, 1)
        return float(slope)

    return series.rolling(window).apply(_slope, raw=True)


def read_first_line(path: Path) -> str:
    for enc in ("gbk", "gb2312", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="ignore").splitlines()[0].strip()
        except Exception:
            continue
    return ""


def load_name_code_map() -> Dict[str, str]:
    if NAME_CODE_CACHE.exists():
        try:
            return json.loads(NAME_CODE_CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    code_to_name: Dict[str, str] = {}
    for fp in RAW_DIR.glob("*.txt"):
        first = read_first_line(fp)
        m = re.match(r"([A-Z0-9#]+)\s+(.+?)\s+", first)
        if not m:
            continue
        raw_code, stock_name = m.group(1).strip(), m.group(2).strip()
        code_to_name[raw_code] = stock_name

    normal_tail_map: Dict[str, str] = {}
    for fp in DATA_DIR.glob("*.txt"):
        m = re.search(r"(\d{6})", fp.stem)
        if m:
            normal_tail_map[m.group(1)] = fp.stem

    name_to_normal: Dict[str, str] = {}
    for raw_code, stock_name in code_to_name.items():
        tail = re.search(r"(\d{6})", raw_code)
        if not tail:
            continue
        normal_code = normal_tail_map.get(tail.group(1))
        if normal_code:
            name_to_normal[stock_name] = normal_code
    try:
        NAME_CODE_CACHE.write_text(json.dumps(name_to_normal, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return name_to_normal


def resolve_code(stock_name: str, mapping: Dict[str, str]) -> Optional[str]:
    alias = CASE_NAME_ALIASES.get(stock_name, stock_name)
    if alias in mapping:
        return mapping[alias]
    for mapped_name, code in mapping.items():
        if alias in mapped_name or mapped_name in alias:
            return code
    return None


def load_stock_data(file_path: str) -> Optional[pd.DataFrame]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None

    if len(lines) < MIN_BARS + 1:
        return None

    rows: List[Dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            rows.append(
                {
                    "date": parts[0],
                    "open": float(parts[1]),
                    "high": float(parts[2]),
                    "low": float(parts[3]),
                    "close": float(parts[4]),
                    "volume": float(parts[5]),
                }
            )
        except ValueError:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < MIN_BARS:
        return None
    df["code"] = Path(file_path).stem
    return df


def compute_b1_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["ret1"] = x["close"].pct_change()
    x["ret3"] = x["close"].pct_change(3)
    x["ret5"] = x["close"].pct_change(5)
    x["ret10"] = x["close"].pct_change(10)
    x["ret20"] = x["close"].pct_change(20)
    x["signal_ret"] = safe_div(x["close"] - x["open"], x["open"], default=np.nan)

    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0

    x["ma5"] = x["close"].rolling(5).mean()
    x["ma10"] = x["close"].rolling(10).mean()
    x["ma20"] = x["close"].rolling(20).mean()
    x["ma30"] = x["close"].rolling(30).mean()
    x["ma60"] = x["close"].rolling(60).mean()

    x["ma5_slope_5"] = compute_slope(x["ma5"], 5)
    x["ma10_slope_5"] = compute_slope(x["ma10"], 5)
    x["ma20_slope_5"] = compute_slope(x["ma20"], 5)
    x["trend_slope_5"] = compute_slope(x["trend_line"], 5)
    x["long_slope_5"] = compute_slope(x["long_line"], 5)

    llv9 = x["low"].rolling(9).min()
    hhv9 = x["high"].rolling(9).max()
    rsv = safe_div(x["close"] - llv9, (hhv9 - llv9).replace(0, np.nan)) * 100
    x["K"] = pd.Series(rsv, index=x.index).ewm(alpha=1 / 3, adjust=False).mean()
    x["D"] = x["K"].ewm(alpha=1 / 3, adjust=False).mean()
    x["J"] = 3 * x["K"] - 2 * x["D"]

    x["trend_spread"] = safe_div(x["trend_line"] - x["long_line"], x["close"], default=np.nan)
    x["close_to_trend"] = safe_div(x["close"] - x["trend_line"], x["trend_line"], default=np.nan)
    x["close_to_long"] = safe_div(x["close"] - x["long_line"], x["long_line"], default=np.nan)

    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5_prev"], default=np.nan)
    x["vol_vs_prev"] = safe_div(x["volume"], x["volume"].shift(1), default=np.nan)

    candle_range = (x["high"] - x["low"]).replace(0, np.nan)
    body = (x["close"] - x["open"]).abs()
    x["body_ratio"] = safe_div(body, candle_range, default=np.nan)
    x["upper_shadow_pct"] = safe_div(x["high"] - x[["open", "close"]].max(axis=1), candle_range, default=np.nan)
    x["lower_shadow_pct"] = safe_div(x[["open", "close"]].min(axis=1) - x["low"], candle_range, default=np.nan)
    x["close_location"] = safe_div(x["close"] - x["low"], candle_range, default=np.nan)

    x["minimal_b1_candidate"] = (x["J"] < 13) & (x["trend_line"] > x["long_line"])
    return x


def extract_sequence(window_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    close = window_df["close"].to_numpy(dtype=float)
    volume = window_df["volume"].to_numpy(dtype=float)
    close_norm = minmax_1d(close)
    returns = safe_div(np.diff(close), close[:-1], default=0.0)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    returns = zscore_1d(returns)
    volume_norm = minmax_1d(volume)
    close_vol_concat = np.concatenate([close_norm, volume_norm], axis=0)
    return {
        "close_norm": close_norm,
        "returns": returns,
        "close_vol_concat": close_vol_concat,
    }


def future_metrics(x: pd.DataFrame, idx: int) -> Dict[str, Any]:
    entry_idx = idx + BUY_DELAY_DAYS
    if entry_idx >= len(x):
        return {}

    entry_price = float(x.iloc[entry_idx]["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {}

    fut = x.iloc[entry_idx + 1 : entry_idx + MAX_HOLD_DAYS + 1].copy()
    if fut.empty:
        return {}

    out: Dict[str, Any] = {
        "entry_date": pd.Timestamp(x.iloc[entry_idx]["date"]),
        "entry_price": entry_price,
        "stop_loss_price": float(x.iloc[entry_idx]["low"]) * STOP_LOSS_MULTIPLIER,
    }

    for horizon in (3, 5, 10, 20, 30):
        sub = fut.head(horizon)
        if sub.empty:
            out[f"ret_{horizon}d"] = np.nan
            out[f"up_{horizon}d"] = np.nan
        else:
            ret = float(sub.iloc[-1]["close"] / entry_price - 1)
            out[f"ret_{horizon}d"] = ret
            out[f"up_{horizon}d"] = float(ret > 0)

    min_close_ret_30 = float(fut.head(30)["close"].min() / entry_price - 1)
    max_high_ret_30 = float(fut.head(30)["high"].max() / entry_price - 1)
    out["min_close_ret_30"] = min_close_ret_30
    out["max_high_ret_30"] = max_high_ret_30
    out["negative_30d"] = bool(min_close_ret_30 <= NEGATIVE_MAX_DRAWDOWN_30 and out.get("ret_20d", 0.0) <= NEGATIVE_MIN_RET20)
    return out


def build_candidates_for_one_stock(file_path: str) -> List[Dict[str, Any]]:
    df = load_stock_data(file_path)
    if df is None or df.empty:
        return []

    x = compute_b1_features(df)
    code = str(x["code"].iloc[0])
    rows: List[Dict[str, Any]] = []

    for idx in range(max(MIN_BARS, SEQ_LEN - 1), len(x) - (BUY_DELAY_DAYS + 1)):
        row = x.iloc[idx]
        if not bool(row["minimal_b1_candidate"]):
            continue

        metrics = future_metrics(x, idx)
        if not metrics:
            continue

        seq_window = x.iloc[idx - SEQ_LEN + 1 : idx + 1]
        if len(seq_window) != SEQ_LEN:
            continue

        seq_map = extract_sequence(seq_window)
        rows.append(
            {
                "code": code,
                "signal_date": pd.Timestamp(row["date"]),
                "signal_idx": idx,
                "entry_date": metrics["entry_date"],
                "entry_price": metrics["entry_price"],
                "stop_loss_price": metrics["stop_loss_price"],
                "ret_3d": metrics["ret_3d"],
                "ret_5d": metrics["ret_5d"],
                "ret_10d": metrics["ret_10d"],
                "ret_20d": metrics["ret_20d"],
                "ret_30d": metrics["ret_30d"],
                "up_3d": metrics["up_3d"],
                "up_5d": metrics["up_5d"],
                "up_10d": metrics["up_10d"],
                "up_20d": metrics["up_20d"],
                "up_30d": metrics["up_30d"],
                "min_close_ret_30": metrics["min_close_ret_30"],
                "max_high_ret_30": metrics["max_high_ret_30"],
                "negative_30d": metrics["negative_30d"],
                "J": float(row["J"]) if pd.notna(row["J"]) else np.nan,
                "ret1": float(row["ret1"]) if pd.notna(row["ret1"]) else 0.0,
                "ret3": float(row["ret3"]) if pd.notna(row["ret3"]) else 0.0,
                "ret5": float(row["ret5"]) if pd.notna(row["ret5"]) else 0.0,
                "ret10": float(row["ret10"]) if pd.notna(row["ret10"]) else 0.0,
                "signal_ret": float(row["signal_ret"]) if pd.notna(row["signal_ret"]) else 0.0,
                "trend_spread": float(row["trend_spread"]) if pd.notna(row["trend_spread"]) else 0.0,
                "close_to_trend": float(row["close_to_trend"]) if pd.notna(row["close_to_trend"]) else 0.0,
                "close_to_long": float(row["close_to_long"]) if pd.notna(row["close_to_long"]) else 0.0,
                "signal_vs_ma5": float(row["signal_vs_ma5"]) if pd.notna(row["signal_vs_ma5"]) else 0.0,
                "vol_vs_prev": float(row["vol_vs_prev"]) if pd.notna(row["vol_vs_prev"]) else 0.0,
                "body_ratio": float(row["body_ratio"]) if pd.notna(row["body_ratio"]) else 0.0,
                "upper_shadow_pct": float(row["upper_shadow_pct"]) if pd.notna(row["upper_shadow_pct"]) else 0.0,
                "lower_shadow_pct": float(row["lower_shadow_pct"]) if pd.notna(row["lower_shadow_pct"]) else 0.0,
                "close_location": float(row["close_location"]) if pd.notna(row["close_location"]) else 0.0,
                "ma5_slope_5": float(row["ma5_slope_5"]) if pd.notna(row["ma5_slope_5"]) else 0.0,
                "ma10_slope_5": float(row["ma10_slope_5"]) if pd.notna(row["ma10_slope_5"]) else 0.0,
                "ma20_slope_5": float(row["ma20_slope_5"]) if pd.notna(row["ma20_slope_5"]) else 0.0,
                "trend_slope_5": float(row["trend_slope_5"]) if pd.notna(row["trend_slope_5"]) else 0.0,
                "long_slope_5": float(row["long_slope_5"]) if pd.notna(row["long_slope_5"]) else 0.0,
                "seq_map": seq_map,
            }
        )

    return rows


def parse_perfect_b1_cases(name_code_map: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in sorted(PERFECT_DIR.glob("*.png")):
        m = re.match(r"(.+?)(\d{8})\.png$", p.name)
        if not m:
            continue
        stock_name = m.group(1).strip()
        signal_date = m.group(2)
        code = resolve_code(stock_name, name_code_map)
        rows.append(
            {
                "stock_name": stock_name,
                "signal_date": signal_date,
                "code": code,
                "image_name": p.name,
                "status": "name_unmapped" if not code else "pending",
            }
        )
    return pd.DataFrame(rows)


def enrich_perfect_case(case_row: pd.Series) -> Dict[str, Any]:
    out = case_row.to_dict()
    code = out.get("code")
    if not code:
        out["status"] = "name_unmapped"
        return out

    path = DATA_DIR / f"{code}.txt"
    if not path.exists():
        out["status"] = "normal_missing"
        return out

    df = load_stock_data(str(path))
    if df is None or df.empty:
        out["status"] = "load_failed"
        return out

    x = compute_b1_features(df)
    ds = pd.to_datetime(x["date"]).dt.strftime("%Y%m%d")
    idxs = np.flatnonzero(ds.to_numpy() == out["signal_date"])
    if len(idxs) == 0:
        out["status"] = "date_missing"
        return out

    idx = int(idxs[-1])
    if idx < SEQ_LEN - 1 or idx + BUY_DELAY_DAYS + 1 >= len(x):
        out["status"] = "bars_insufficient"
        return out

    row = x.iloc[idx]
    metrics = future_metrics(x, idx)
    if not metrics:
        out["status"] = "future_missing"
        return out

    seq_map = extract_sequence(x.iloc[idx - SEQ_LEN + 1 : idx + 1])
    out.update(
        {
            "status": "ok",
            "signal_date": pd.Timestamp(row["date"]),
            "entry_date": metrics["entry_date"],
            "entry_price": metrics["entry_price"],
            "stop_loss_price": metrics["stop_loss_price"],
            "ret_3d": metrics["ret_3d"],
            "ret_5d": metrics["ret_5d"],
            "ret_10d": metrics["ret_10d"],
            "ret_20d": metrics["ret_20d"],
            "ret_30d": metrics["ret_30d"],
            "up_3d": metrics["up_3d"],
            "up_5d": metrics["up_5d"],
            "up_10d": metrics["up_10d"],
            "up_20d": metrics["up_20d"],
            "up_30d": metrics["up_30d"],
            "min_close_ret_30": metrics["min_close_ret_30"],
            "max_high_ret_30": metrics["max_high_ret_30"],
            "J": float(row["J"]) if pd.notna(row["J"]) else np.nan,
            "ret1": float(row["ret1"]) if pd.notna(row["ret1"]) else 0.0,
            "ret3": float(row["ret3"]) if pd.notna(row["ret3"]) else 0.0,
            "ret5": float(row["ret5"]) if pd.notna(row["ret5"]) else 0.0,
            "ret10": float(row["ret10"]) if pd.notna(row["ret10"]) else 0.0,
            "signal_ret": float(row["signal_ret"]) if pd.notna(row["signal_ret"]) else 0.0,
            "trend_spread": float(row["trend_spread"]) if pd.notna(row["trend_spread"]) else 0.0,
            "close_to_trend": float(row["close_to_trend"]) if pd.notna(row["close_to_trend"]) else 0.0,
            "close_to_long": float(row["close_to_long"]) if pd.notna(row["close_to_long"]) else 0.0,
            "signal_vs_ma5": float(row["signal_vs_ma5"]) if pd.notna(row["signal_vs_ma5"]) else 0.0,
            "vol_vs_prev": float(row["vol_vs_prev"]) if pd.notna(row["vol_vs_prev"]) else 0.0,
            "body_ratio": float(row["body_ratio"]) if pd.notna(row["body_ratio"]) else 0.0,
            "upper_shadow_pct": float(row["upper_shadow_pct"]) if pd.notna(row["upper_shadow_pct"]) else 0.0,
            "lower_shadow_pct": float(row["lower_shadow_pct"]) if pd.notna(row["lower_shadow_pct"]) else 0.0,
            "close_location": float(row["close_location"]) if pd.notna(row["close_location"]) else 0.0,
            "ma5_slope_5": float(row["ma5_slope_5"]) if pd.notna(row["ma5_slope_5"]) else 0.0,
            "ma10_slope_5": float(row["ma10_slope_5"]) if pd.notna(row["ma10_slope_5"]) else 0.0,
            "ma20_slope_5": float(row["ma20_slope_5"]) if pd.notna(row["ma20_slope_5"]) else 0.0,
            "trend_slope_5": float(row["trend_slope_5"]) if pd.notna(row["trend_slope_5"]) else 0.0,
            "long_slope_5": float(row["long_slope_5"]) if pd.notna(row["long_slope_5"]) else 0.0,
            "seq_map": seq_map,
        }
    )
    return out


def split_three_way_by_date(signal_df: pd.DataFrame) -> Dict[str, pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(signal_df["signal_date"]).drop_duplicates())
    n = len(unique_dates)
    if n < 10:
        raise ValueError("历史信号日期太少，无法做三段切分")

    research_end = int(n * RESEARCH_RATIO)
    validation_end = int(n * (RESEARCH_RATIO + VALIDATION_RATIO))
    research_end = min(max(1, research_end), n - 2)
    validation_end = min(max(research_end + 1, validation_end), n - 1)

    return {
        "research_end": unique_dates[research_end - 1],
        "validation_start": unique_dates[research_end],
        "validation_end": unique_dates[validation_end - 1],
        "final_start": unique_dates[validation_end],
        "final_end": unique_dates[-1],
    }


def split_three_way_by_dates(dates: List[pd.Timestamp]) -> Dict[str, pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(pd.Series(dates)).drop_duplicates())
    n = len(unique_dates)
    if n < 5:
        raise ValueError("有效日期太少，无法做三段切分")

    research_end = int(n * RESEARCH_RATIO)
    validation_end = int(n * (RESEARCH_RATIO + VALIDATION_RATIO))
    research_end = min(max(1, research_end), n - 2)
    validation_end = min(max(research_end + 1, validation_end), n - 1)

    return {
        "research_end": unique_dates[research_end - 1],
        "validation_start": unique_dates[research_end],
        "validation_end": unique_dates[validation_end - 1],
        "final_start": unique_dates[validation_end],
        "final_end": unique_dates[-1],
    }


def assign_split(df: pd.DataFrame, cutoffs: Dict[str, pd.Timestamp]) -> pd.Series:
    date_s = pd.to_datetime(df["signal_date"])
    out = pd.Series("", index=df.index, dtype="object")
    out[date_s <= cutoffs["research_end"]] = "research"
    out[(date_s >= cutoffs["validation_start"]) & (date_s <= cutoffs["validation_end"])] = "validation"
    out[date_s >= cutoffs["final_start"]] = "final_test"
    return out


def stack_templates(rows: Iterable[Dict[str, Any]], rep_name: str) -> np.ndarray:
    arrs = [np.asarray(r["seq_map"][rep_name], dtype=float) for r in rows]
    if not arrs:
        return np.empty((0, 0), dtype=float)
    return np.vstack(arrs)


def similarity_scores(seqs: np.ndarray, templates: np.ndarray, scorer: str) -> np.ndarray:
    if len(seqs) == 0:
        return np.zeros(0, dtype=float)
    if templates.size == 0:
        return np.zeros(len(seqs), dtype=float)

    if scorer == "corr":
        seq_z = np.vstack([zscore_1d(s) for s in seqs])
        tpl_z = np.vstack([zscore_1d(t) for t in templates])
        num = seq_z @ tpl_z.T
        den = np.linalg.norm(seq_z, axis=1, keepdims=True) * np.linalg.norm(tpl_z, axis=1)
        sims = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-12)
        return np.nanmax(sims, axis=1)

    if scorer == "cosine":
        num = seqs @ templates.T
        den = np.linalg.norm(seqs, axis=1, keepdims=True) * np.linalg.norm(templates, axis=1)
        sims = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-12)
        return np.nanmax(sims, axis=1)

    if scorer == "euclidean":
        dists = np.sqrt(((seqs[:, None, :] - templates[None, :, :]) ** 2).sum(axis=2))
        min_dist = np.nanmin(dists, axis=1)
        return 1.0 / (1.0 + min_dist)

    raise ValueError(f"未知 scorer: {scorer}")


def select_daily_topn(df: pd.DataFrame, score_col: str, topn: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = df.sort_values(["signal_date", score_col, "code"], ascending=[True, False, True])
    return ranked.groupby("signal_date", group_keys=False).head(topn).copy()


def summarize_signal_df(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "sample_count": 0,
            "date_count": 0,
            "up_5d_rate": np.nan,
            "up_10d_rate": np.nan,
            "up_20d_rate": np.nan,
            "ret_5d_mean": np.nan,
            "ret_10d_mean": np.nan,
            "ret_20d_mean": np.nan,
            "min_close_ret_30_mean": np.nan,
        }
    return {
        "sample_count": int(len(df)),
        "date_count": int(df["signal_date"].nunique()),
        "up_5d_rate": float(df["up_5d"].mean()),
        "up_10d_rate": float(df["up_10d"].mean()),
        "up_20d_rate": float(df["up_20d"].mean()),
        "ret_5d_mean": float(df["ret_5d"].mean()),
        "ret_10d_mean": float(df["ret_10d"].mean()),
        "ret_20d_mean": float(df["ret_20d"].mean()),
        "min_close_ret_30_mean": float(df["min_close_ret_30"].mean()),
    }


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, max_iter: int = 400, lr: float = 0.05, l2: float = 1e-3) -> Dict[str, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Xs = (X - mean) / std

    w = np.zeros(Xs.shape[1], dtype=float)
    b = 0.0

    for _ in range(max_iter):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = (Xs.T @ (p - y)) / len(y) + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b

    return {"mean": mean, "std": std, "w": w, "b": np.array([b], dtype=float)}


def predict_logistic(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mean = model["mean"]
    std = model["std"]
    w = model["w"]
    b = float(model["b"][0])
    Xs = (X - mean) / std
    z = Xs @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def build_feature_matrix(df: pd.DataFrame, include_similarity: bool) -> Tuple[np.ndarray, List[str]]:
    cols = [
        "J",
        "ret1",
        "ret3",
        "ret5",
        "ret10",
        "signal_ret",
        "trend_spread",
        "close_to_trend",
        "close_to_long",
        "signal_vs_ma5",
        "vol_vs_prev",
        "body_ratio",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "close_location",
        "ma5_slope_5",
        "ma10_slope_5",
        "ma20_slope_5",
        "trend_slope_5",
        "long_slope_5",
    ]
    if include_similarity:
        cols.extend([f"sim_{scorer}_{rep}" for scorer, rep in SIMILARITY_VARIANTS])
    return df[cols].fillna(0.0).to_numpy(dtype=float), cols


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(stage: str, **kwargs: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    payload.update(kwargs)
    write_json(RESULT_DIR / "progress.json", payload)


def map_candidate_build(file_paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for part in executor.map(build_candidates_for_one_stock, [str(p) for p in file_paths], chunksize=20):
                rows.extend(part)
        return rows
    except Exception as exc:
        print(f"多进程构建候选池失败，回退到串行: {type(exc).__name__}: {exc}")
        for i, p in enumerate(file_paths, start=1):
            rows.extend(build_candidates_for_one_stock(str(p)))
            if i % 300 == 0:
                print(f"串行候选池进度: {i}/{len(file_paths)}")
        return rows


def main() -> None:
    update_progress("starting")
    print("开始构建 B1 极简候选池...")
    name_code_map = load_name_code_map()

    candidate_pkl = RESULT_DIR / "candidate_rows.pkl"
    candidate_csv = RESULT_DIR / "candidate_rows.csv"
    if candidate_pkl.exists():
        print("发现已有候选池中间结果，直接复用。")
        candidate_df = pd.read_pickle(candidate_pkl)
    else:
        file_paths = sorted(DATA_DIR.glob("*.txt"))
        rows = map_candidate_build(file_paths)
        candidate_df = pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)
        candidate_df.to_pickle(candidate_pkl)
        candidate_df.drop(columns=["seq_map"]).to_csv(candidate_csv, index=False, encoding="utf-8-sig")
    print(f"候选池信号数: {len(candidate_df)}")
    update_progress("candidates_built", candidate_signal_count=int(len(candidate_df)))

    cutoffs = split_three_way_by_date(candidate_df)
    candidate_df["split"] = assign_split(candidate_df, cutoffs)
    write_json(RESULT_DIR / "split_cutoffs.json", cutoffs)

    perfect_raw = parse_perfect_b1_cases(name_code_map)
    perfect_enriched = pd.DataFrame([enrich_perfect_case(r) for _, r in perfect_raw.iterrows()])
    perfect_enriched["split"] = assign_split(perfect_enriched[perfect_enriched["status"] == "ok"], cutoffs).reindex(perfect_enriched.index).fillna("")
    perfect_enriched.to_csv(RESULT_DIR / "perfect_positive_manifest.csv", index=False, encoding="utf-8-sig")
    update_progress(
        "perfect_cases_built",
        perfect_total=int(len(perfect_raw)),
        perfect_ok=int((perfect_enriched["status"] == "ok").sum()),
    )

    positive_df = perfect_enriched[perfect_enriched["status"] == "ok"].copy().reset_index(drop=True)
    if positive_df.empty:
        raise ValueError("完美图/B1 没有成功映射出的正样本")

    if not (positive_df["split"] == "research").any():
        print("默认切分下 research 段没有完美图正样本，改用完美图正样本日期重建切分。")
        positive_cutoffs = split_three_way_by_dates(list(pd.to_datetime(positive_df["signal_date"])))
        candidate_df["split"] = assign_split(candidate_df, positive_cutoffs)
        perfect_enriched["split"] = assign_split(perfect_enriched[perfect_enriched["status"] == "ok"], positive_cutoffs).reindex(perfect_enriched.index).fillna("")
        perfect_enriched.to_csv(RESULT_DIR / "perfect_positive_manifest.csv", index=False, encoding="utf-8-sig")
        write_json(RESULT_DIR / "split_cutoffs.json", positive_cutoffs)
        cutoffs = positive_cutoffs
        positive_df = perfect_enriched[perfect_enriched["status"] == "ok"].copy().reset_index(drop=True)
        update_progress("split_rebuilt_on_positive_dates", cutoffs=cutoffs)

    baseline_summary = []
    for split_name in ["validation", "final_test"]:
        part = candidate_df[candidate_df["split"] == split_name].copy()
        row = {"family": "baseline", "variant": "all_candidates", "topn": 0, "split": split_name}
        row.update(summarize_signal_df(part))
        baseline_summary.append(row)
        for topn in DAILY_TOPN_LIST:
            top_df = part.sort_values(["signal_date", "J", "code"], ascending=[True, True, True]).groupby("signal_date", group_keys=False).head(topn)
            row = {"family": "baseline", "variant": "lowest_J", "topn": topn, "split": split_name}
            row.update(summarize_signal_df(top_df))
            baseline_summary.append(row)

    template_rows = positive_df[positive_df["split"] == "research"].to_dict("records")
    if not template_rows:
        raise ValueError("research 段没有完美图/B1 正样本，无法构建模板")

    for scorer, rep in SIMILARITY_VARIANTS:
        templates = stack_templates(template_rows, rep)
        seqs = np.vstack([r["seq_map"][rep] for r in candidate_df.to_dict("records")])
        candidate_df[f"sim_{scorer}_{rep}"] = similarity_scores(seqs, templates, scorer)
        pos_seqs = np.vstack([r["seq_map"][rep] for r in positive_df.to_dict("records")])
        positive_df[f"sim_{scorer}_{rep}"] = similarity_scores(pos_seqs, templates, scorer)
    update_progress("similarity_scores_ready")

    sim_leaderboard: List[Dict[str, Any]] = []
    for split_name in ["validation", "final_test"]:
        part = candidate_df[candidate_df["split"] == split_name].copy()
        for scorer, rep in SIMILARITY_VARIANTS:
            score_col = f"sim_{scorer}_{rep}"
            for topn in DAILY_TOPN_LIST:
                top_df = select_daily_topn(part, score_col, topn)
                row = {"family": "similarity", "variant": f"{scorer}_{rep}", "topn": topn, "split": split_name}
                row.update(summarize_signal_df(top_df))
                sim_leaderboard.append(row)

    research_positive = positive_df[positive_df["split"] == "research"].copy()
    research_negative = candidate_df[(candidate_df["split"] == "research") & (candidate_df["negative_30d"])].copy()
    if research_negative.empty:
        raise ValueError("research 段没有自动负样本，无法训练机器学习模型")

    max_negative = max(len(research_positive) * MAX_NEGATIVE_MULTIPLIER, len(research_positive))
    research_negative = research_negative.sort_values("signal_date").head(max_negative).copy()

    train_plain = pd.concat(
        [
            research_positive.assign(label=1, source="perfect_positive"),
            research_negative.assign(label=0, source="auto_negative"),
        ],
        ignore_index=True,
    )

    X_plain, feature_cols_plain = build_feature_matrix(train_plain, include_similarity=False)
    y_plain = train_plain["label"].to_numpy(dtype=float)
    plain_model = fit_logistic_regression(X_plain, y_plain)

    X_mix, feature_cols_mix = build_feature_matrix(train_plain, include_similarity=True)
    mix_model = fit_logistic_regression(X_mix, y_plain)
    update_progress("ml_models_trained", positive_train_count=int(len(research_positive)), negative_train_count=int(len(research_negative)))

    for split_name in ["validation", "final_test"]:
        part = candidate_df[candidate_df["split"] == split_name].copy()
        Xp, _ = build_feature_matrix(part, include_similarity=False)
        Xm, _ = build_feature_matrix(part, include_similarity=True)
        part["ml_plain_score"] = predict_logistic(plain_model, Xp)
        part["ml_mix_score"] = predict_logistic(mix_model, Xm)
        candidate_df.loc[part.index, "ml_plain_score"] = part["ml_plain_score"]
        candidate_df.loc[part.index, "ml_mix_score"] = part["ml_mix_score"]

    ml_leaderboard: List[Dict[str, Any]] = []
    for split_name in ["validation", "final_test"]:
        part = candidate_df[candidate_df["split"] == split_name].copy()
        for score_col, family, variant in [
            ("ml_plain_score", "ml", "logistic_features_only"),
            ("ml_mix_score", "ml_plus_similarity", "logistic_features_plus_similarity"),
        ]:
            for topn in DAILY_TOPN_LIST:
                top_df = select_daily_topn(part, score_col, topn)
                row = {"family": family, "variant": variant, "topn": topn, "split": split_name}
                row.update(summarize_signal_df(top_df))
                ml_leaderboard.append(row)

    baseline_df = pd.DataFrame(baseline_summary)
    sim_df = pd.DataFrame(sim_leaderboard)
    ml_df = pd.DataFrame(ml_leaderboard)
    leaderboard_df = pd.concat([baseline_df, sim_df, ml_df], ignore_index=True)
    leaderboard_df.to_csv(RESULT_DIR / "signal_layer_leaderboard.csv", index=False, encoding="utf-8-sig")
    update_progress("leaderboard_ready", leaderboard_rows=int(len(leaderboard_df)))

    validation_df = leaderboard_df[leaderboard_df["split"] == "validation"].copy()
    validation_best_rows: List[pd.Series] = []
    for family in ["baseline", "similarity", "ml", "ml_plus_similarity"]:
        fam = validation_df[validation_df["family"] == family].copy()
        if fam.empty:
            continue
        fam = fam.sort_values(["ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[False, False, False])
        validation_best_rows.append(fam.iloc[0])
    validation_best_df = pd.DataFrame(validation_best_rows)
    validation_best_df.to_csv(RESULT_DIR / "validation_family_best.csv", index=False, encoding="utf-8-sig")

    final_reports: List[Dict[str, Any]] = []
    final_selected_frames: List[pd.DataFrame] = []
    final_part = candidate_df[candidate_df["split"] == "final_test"].copy()
    for _, best_row in validation_best_df.iterrows():
        family = str(best_row["family"])
        variant = str(best_row["variant"])
        topn = int(best_row["topn"])
        if family == "baseline" and variant == "all_candidates":
            selected = final_part.copy()
        elif family == "baseline":
            selected = final_part.sort_values(["signal_date", "J", "code"], ascending=[True, True, True]).groupby("signal_date", group_keys=False).head(topn)
        elif family == "similarity":
            score_col = f"sim_{variant}"
            selected = select_daily_topn(final_part, score_col, topn)
        elif family == "ml":
            selected = select_daily_topn(final_part, "ml_plain_score", topn)
        elif family == "ml_plus_similarity":
            selected = select_daily_topn(final_part, "ml_mix_score", topn)
        else:
            continue

        report = {"family": family, "variant": variant, "topn": topn}
        report.update(summarize_signal_df(selected))
        final_reports.append(report)

        tag = f"{family}_{variant}_top{topn}"
        tmp = selected.drop(columns=["seq_map"]).copy()
        tmp["strategy_tag"] = tag
        final_selected_frames.append(tmp)

    final_report_df = pd.DataFrame(final_reports).sort_values(["ret_20d_mean", "up_20d_rate"], ascending=[False, False])
    final_report_df.to_csv(RESULT_DIR / "final_test_report.csv", index=False, encoding="utf-8-sig")
    if final_selected_frames:
        pd.concat(final_selected_frames, ignore_index=True).to_csv(
            RESULT_DIR / "final_test_selected_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = {
        "result_dir": str(RESULT_DIR),
        "candidate_signal_count": int(len(candidate_df)),
        "candidate_split_counts": candidate_df["split"].value_counts().sort_index().to_dict(),
        "perfect_positive_total": int(len(perfect_raw)),
        "perfect_positive_ok": int((perfect_enriched["status"] == "ok").sum()),
        "perfect_split_counts": positive_df["split"].value_counts().sort_index().to_dict(),
        "negative_train_count": int(len(research_negative)),
        "positive_train_count": int(len(research_positive)),
        "feature_cols_plain": feature_cols_plain,
        "feature_cols_mix": feature_cols_mix,
    }
    write_json(RESULT_DIR / "summary.json", summary)
    update_progress("finished", summary=summary)

    print("信号层实验完成。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
