from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import pinfilter, stoploss, technical_indicators


CACHED_UNIVERSE = ROOT / "results" / "pin_rebuild_opt_20260314" / "pin_universe_features.csv"
CASE_FILE = ROOT / "results" / "pin_success_fail_compare_20260314.csv"
RECENT_SIGNAL_FILE = ROOT / "results" / "pinfilter_recent_week_signals_20260314_thread" / "all_signals.csv"
FORWARD_DIR = ROOT / "data" / "forward_data"
LATEST_DIR = ROOT / "data" / "20260313" / "normal"
OUT_DIR = ROOT / "results" / "pin_negative_factor_fast_20260314"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
MIN_BARS = 160
EPS = 1e-12


def safe_div(a, b):
    if a is None or b is None:
        return np.nan
    try:
        a_f = float(a)
        b_f = float(b)
    except Exception:
        return np.nan
    if not math.isfinite(a_f) or not math.isfinite(b_f) or abs(b_f) <= EPS:
        return np.nan
    return a_f / b_f


def build_stair_volume_flag(x: pd.DataFrame, lookback: int = 30) -> pd.Series:
    out = np.zeros(len(x), dtype=bool)
    closes = x["close"].to_numpy(dtype=float)
    opens = x["open"].to_numpy(dtype=float)
    vols = x["volume"].to_numpy(dtype=float)
    for idx in range(len(x)):
        left = max(0, idx - lookback + 1)
        hit = False
        for anchor in range(left, idx - 2):
            if not (closes[anchor] > opens[anchor]):
                continue
            hist_left = max(0, anchor - lookback + 1)
            anchor_hist = vols[hist_left : anchor + 1]
            if len(anchor_hist) < 5:
                continue
            if np.sum(anchor_hist >= vols[anchor]) > 3:
                continue
            for stair_len in (2, 3, 4):
                end = anchor + stair_len
                if end >= idx:
                    continue
                ok = True
                prev_vol = vols[anchor]
                for j in range(anchor + 1, end + 1):
                    if not (closes[j] < opens[j]):
                        ok = False
                        break
                    if not (vols[j] < prev_vol and vols[j] <= prev_vol * 0.95 and vols[j] >= prev_vol * 0.45):
                        ok = False
                        break
                    prev_vol = vols[j]
                if ok:
                    hit = True
                    break
            if hit:
                break
        out[idx] = hit
    return pd.Series(out, index=x.index)


def build_failed_breakout_flag(x: pd.DataFrame, lookback: int = 60) -> pd.Series:
    prior_high = x["high"].shift(1).rolling(lookback, min_periods=20).max()
    return (
        (x["high"] >= prior_high * 0.995)
        & (x["close"] < prior_high * 0.995)
        & (x["upper_shadow_ratio"] >= (1.0 / 3.0))
    ).fillna(False)


def build_code_features(file_path: Path) -> Optional[pd.DataFrame]:
    df, err = stoploss.load_data(str(file_path))
    if err or df is None or len(df) < MIN_BARS:
        return None
    df = df[(df["日期"] < EXCLUDE_START) | (df["日期"] > EXCLUDE_END)].copy()
    if len(df) < MIN_BARS:
        return None

    df = technical_indicators.calculate_trend(df)
    df = technical_indicators.calculate_kdj(df)
    x = pd.DataFrame(
        {
            "date": pd.to_datetime(df["日期"]),
            "open": df["开盘"].astype(float),
            "high": df["最高"].astype(float),
            "low": df["最低"].astype(float),
            "close": df["收盘"].astype(float),
            "volume": df["成交量"].astype(float),
            "code": file_path.stem,
            "trend_line": df["知行短期趋势线"].astype(float),
            "long_line": df["知行多空线"].astype(float),
            "J": df["J"].astype(float),
        }
    ).reset_index(drop=True)

    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    body_low = np.minimum(x["open"], x["close"])
    body_high = np.maximum(x["open"], x["close"])
    x["upper_shadow_ratio"] = (x["high"] - body_high) / full_range

    x = pinfilter.add_structure_tags(x)
    x["ret3"] = x["close"] / x["close"].shift(3) - 1.0
    x["ret5"] = x["close"] / x["close"].shift(5) - 1.0
    x["ret10"] = x["close"] / x["close"].shift(10) - 1.0
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["signal_vs_ma20"] = x["volume"] / x["vol_ma20"]
    x["vol_vs_prev"] = x["volume"] / x["volume"].shift(1)
    x["trend_slope_3"] = x["trend_line"] / x["trend_line"].shift(3) - 1.0
    x["trend_slope_5"] = x["trend_line"] / x["trend_line"].shift(5) - 1.0
    x["long_slope_5"] = x["long_line"] / x["long_line"].shift(5) - 1.0
    x["trend_line_lead"] = (x["trend_line"] - x["long_line"]) / x["close"]
    x["close_position"] = (x["close"] - x["low"]) / full_range
    x["lower_shadow_ratio"] = (body_low - x["low"]) / full_range

    x["subtype_a"] = (
        (x["trend_slope_5"].fillna(-np.inf) >= 0.03)
        & (x["trend_line_lead"].fillna(-np.inf) >= 0.03)
        & (x["ret10"].fillna(-np.inf) >= 0.05)
        & (x["ret3"].fillna(np.inf) <= 0.01)
        & (x["signal_vs_ma20"].fillna(np.inf) <= 1.0)
        & (x["vol_vs_prev"].fillna(np.inf) <= 1.1)
        & (x["lower_shadow_ratio"].fillna(np.inf) <= 0.25)
    )
    x["subtype_b"] = (
        (x["trend_slope_3"].fillna(-np.inf) >= 0.02)
        & (x["trend_slope_5"].fillna(-np.inf) >= 0.04)
        & (x["long_slope_5"].fillna(-np.inf) >= 0.0)
        & (x["trend_line_lead"].fillna(-np.inf) >= 0.02)
        & (x["ret10"].fillna(-np.inf) >= 0.10)
        & (x["ret3"].fillna(np.inf) <= 0.05)
        & (x["signal_vs_ma20"].fillna(-np.inf) >= 0.90)
        & (x["signal_vs_ma20"].fillna(np.inf) <= 1.50)
        & (x["vol_vs_prev"].fillna(np.inf) <= 1.50)
        & (x["close_position"].fillna(np.inf) <= 0.20)
        & (x["lower_shadow_ratio"].fillna(np.inf) <= 0.10)
    )
    x["subtype_c"] = (
        (x["along_trend_up"] | x["n_up_any"] | x["keyk_support_active"])
        & (x["trend_slope_5"].fillna(-np.inf) >= 0.015)
        & (x["trend_line_lead"].fillna(-np.inf) >= 0.02)
        & (x["ret10"].fillna(-np.inf) >= -0.04)
        & (x["ret3"].fillna(np.inf) <= 0.01)
        & (x["signal_vs_ma20"].fillna(np.inf) <= 1.0)
        & (x["vol_vs_prev"].fillna(np.inf) <= 1.1)
        & (x["close_position"].fillna(np.inf) <= 0.30)
        & (x["lower_shadow_ratio"].fillna(np.inf) <= 0.25)
    )
    x["entry_ok"] = x["subtype_a"] | x["subtype_b"] | x["subtype_c"]
    x["subtype_group"] = np.select(
        [x["subtype_a"] & x["subtype_c"], x["subtype_a"], x["subtype_b"], x["subtype_c"]],
        ["A+C", "A_only", "B_any", "C_only"],
        default="",
    )
    x["stair_volume_30d"] = build_stair_volume_flag(x, 30)
    x["too_fast_up_5d40"] = (x["ret5"] > 0.40).fillna(False)
    x["failed_breakout_long_upper"] = build_failed_breakout_flag(x)
    x["negative_any"] = x["stair_volume_30d"] | x["too_fast_up_5d40"] | x["failed_breakout_long_upper"]
    return x


def attach_for_targets(targets: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    code_map = {code: grp[["code", "date"]].drop_duplicates().copy() for code, grp in targets.groupby("code")}
    out_rows: List[pd.DataFrame] = []
    codes = sorted(code_map.keys())
    for idx, code in enumerate(codes, 1):
        file_path = data_dir / f"{code}.txt"
        if not file_path.exists():
            continue
        feat = build_code_features(file_path)
        if feat is None:
            continue
        merged = code_map[code].merge(
            feat[
                [
                    "code",
                    "date",
                    "subtype_a",
                    "subtype_b",
                    "subtype_c",
                    "entry_ok",
                    "subtype_group",
                    "stair_volume_30d",
                    "too_fast_up_5d40",
                    "failed_breakout_long_upper",
                    "negative_any",
                ]
            ],
            on=["code", "date"],
            how="left",
        )
        out_rows.append(merged)
        if idx % 200 == 0:
            print(f"目标附加特征进度: {idx}/{len(codes)}")
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def summarize_subset(df: pd.DataFrame, tag: str) -> dict:
    if df.empty:
        return {
            "variant": tag,
            "count": 0,
            "avg_close_ret_5d": np.nan,
            "avg_max_float_5d": np.nan,
            "close_positive_5d": np.nan,
            "escape_rate_5d": np.nan,
        }
    return {
        "variant": tag,
        "count": int(len(df)),
        "avg_close_ret_5d": float(df["close_ret_5d"].mean()),
        "avg_max_float_5d": float(df["max_float_5d"].mean()),
        "close_positive_5d": float((df["close_ret_5d"] > 0).mean()),
        "escape_rate_5d": float((df["max_float_5d"] >= 0.05).mean()),
    }


def main():
    universe = pd.read_csv(CACHED_UNIVERSE, parse_dates=["date"])
    base_pin = universe.loc[universe["base_pin"] == True].copy()

    enrich = attach_for_targets(base_pin[["code", "date"]], FORWARD_DIR)
    base_pin = base_pin.merge(enrich, on=["code", "date"], how="left")
    base_pin.to_csv(OUT_DIR / "base_pin_with_negative.csv", index=False)

    entry_df = base_pin.loc[base_pin["entry_ok"] == True].copy()
    entry_df.to_csv(OUT_DIR / "current_entry_with_negative.csv", index=False)

    case_df = pd.read_csv(CASE_FILE, parse_dates=["date"])
    case_eval = case_df.merge(
        base_pin[
            [
                "code",
                "date",
                "base_pin",
                "entry_ok",
                "subtype_group",
                "stair_volume_30d",
                "too_fast_up_5d40",
                "failed_breakout_long_upper",
                "negative_any",
                "close_ret_5d",
                "max_float_5d",
            ]
        ],
        on=["code", "date"],
        how="left",
        suffixes=("_case", ""),
    )
    case_eval.to_csv(OUT_DIR / "case_negative_compare.csv", index=False)

    base_case = case_eval.loc[case_eval["base_pin"] == True].copy()
    rows = []
    for label, sub in base_case.groupby("label"):
        rows.append(
            {
                "label": label,
                "count": int(len(sub)),
                "stair_volume_30d_rate": float(sub["stair_volume_30d"].fillna(False).mean()),
                "too_fast_up_5d40_rate": float(sub["too_fast_up_5d40"].fillna(False).mean()),
                "failed_breakout_long_upper_rate": float(sub["failed_breakout_long_upper"].fillna(False).mean()),
                "negative_any_rate": float(sub["negative_any"].fillna(False).mean()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "base_case_group_compare.csv", index=False)

    variants = [
        ("baseline", entry_df),
        ("exclude_stair_volume", entry_df.loc[~entry_df["stair_volume_30d"].fillna(False)]),
        ("exclude_fast_rise", entry_df.loc[~entry_df["too_fast_up_5d40"].fillna(False)]),
        ("exclude_failed_breakout", entry_df.loc[~entry_df["failed_breakout_long_upper"].fillna(False)]),
        ("exclude_any_negative", entry_df.loc[~entry_df["negative_any"].fillna(False)]),
    ]
    filter_rows = [summarize_subset(df, tag) for tag, df in variants]
    pd.DataFrame(filter_rows).to_csv(OUT_DIR / "entry_negative_filter_ab.csv", index=False)

    recent_signals = pd.read_csv(RECENT_SIGNAL_FILE, parse_dates=["signal_date"])
    recent_targets = recent_signals.rename(columns={"signal_date": "date"})[["code", "date"]].drop_duplicates()
    recent_enrich = attach_for_targets(recent_targets, LATEST_DIR)
    recent_eval = recent_signals.rename(columns={"signal_date": "date"}).merge(
        recent_enrich, on=["code", "date"], how="left"
    )
    recent_eval.to_csv(OUT_DIR / "recent_week_signal_negative.csv", index=False)
    latest_date = recent_eval["date"].max()
    latest_day = recent_eval.loc[recent_eval["date"] == latest_date].copy()
    latest_day.to_csv(OUT_DIR / "latest_day_signal_negative.csv", index=False)

    summary = {
        "base_pin_count": int(len(base_pin)),
        "current_entry_count": int(len(entry_df)),
        "latest_recent_signal_date": None if pd.isna(latest_date) else str(pd.Timestamp(latest_date).date()),
        "latest_recent_signal_count": int(len(latest_day)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
