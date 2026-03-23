from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b1filter, b2filter, b3filter, brick_filter, pinfilter, technical_indicators  # type: ignore
from utils.market_risk_tags import add_risk_features  # type: ignore
from utils.shared_market_features import compute_base_features  # type: ignore


DATA_DIR = ROOT / "data/20260315/normal"
RESULT_DIR = ROOT / "results/strategy_risk_ab_20260319"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))

HORIZONS = (3, 5, 10)
RISK_FLAGS = [
    "recent_heavy_bear_top_20",
    "recent_failed_breakout_20",
    "top_distribution_20",
    "recent_stair_bear_20",
    "risk_fast_rise_5d_30",
    "risk_fast_rise_5d_40",
    "risk_fast_rise_10d_40",
    "risk_segment_rise_slope_10_006",
    "risk_segment_rise_slope_15_005",
    "risk_distribution_any_20",
]

PENALTY_WEIGHTS = {
    "recent_heavy_bear_top_20": 2,
    "recent_failed_breakout_20": 2,
    "top_distribution_20": 2,
    "risk_segment_rise_slope_10_006": 2,
    "risk_fast_rise_5d_40": 1,
    "risk_fast_rise_10d_40": 1,
    "risk_distribution_any_20": 1,
}


def future_metrics(x: pd.DataFrame, signal_idx: int) -> Dict[str, float]:
    x = x.reset_index(drop=True)
    entry_idx = signal_idx + 1
    if entry_idx >= len(x):
        return {}
    entry_open = float(x.iloc[entry_idx]["open"])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return {}

    out: Dict[str, float] = {
        "entry_idx": entry_idx,
        "entry_open": entry_open,
    }
    for h in HORIZONS:
        end_idx = entry_idx + h - 1
        if end_idx >= len(x):
            out[f"ret_close_{h}"] = np.nan
            out[f"ret_max_{h}"] = np.nan
            out[f"up_close_{h}"] = np.nan
            out[f"up_max_{h}"] = np.nan
            continue
        close_ret = float(x.iloc[end_idx]["close"]) / entry_open - 1.0
        max_ret = float(x.loc[entry_idx:end_idx, "high"].max()) / entry_open - 1.0
        out[f"ret_close_{h}"] = close_ret
        out[f"ret_max_{h}"] = max_ret
        out[f"up_close_{h}"] = float(close_ret > 0)
        out[f"up_max_{h}"] = float(max_ret > 0)
    return out


def build_brick_signal_mask(x: pd.DataFrame) -> pd.Series:
    latest_mode = brick_filter.MODE
    mask_a = x["pattern_a"] & (x["rebound_ratio"] >= (1.2 if latest_mode == "legacy" else 0.8))
    mask_b = x["pattern_b"] & (x["rebound_ratio"] >= 1.0)
    legacy_ok = (
        x["signal_base"].fillna(False)
        & x["ret1"].le(0.08)
        & (mask_a | mask_b)
        & x["trend_line"].gt(x["long_line"])
    )
    if latest_mode == "legacy":
        return legacy_ok.fillna(False)
    return (
        legacy_ok
        & x["trend_line"].gt(x["long_line"])
        & x["ret1"].between(-0.03, 0.11, inclusive="both")
    ).fillna(False)


def build_b1_signal_mask(df_cn: pd.DataFrame, weekly_ok: pd.Series) -> pd.Series:
    df_trend = technical_indicators.calculate_trend(df_cn.copy())
    df_kdj = technical_indicators.calculate_kdj(df_cn.copy())
    df_ma = technical_indicators.calculate_daily_ma(df_cn.copy())
    old_b1_passed = b1filter._compute_old_b1_passed(df_cn, df_trend, df_kdj, df_ma)
    today_confirm = b1filter._compute_today_confirm(df_cn, df_kdj)
    return (old_b1_passed.shift(1).fillna(False) & today_confirm & weekly_ok).fillna(False)


def build_pin_signal_mask(df: pd.DataFrame, df_cn: pd.DataFrame, weekly_ok: pd.Series) -> pd.Series:
    x = compute_base_features(df)
    x = x.copy()
    x["open"] = df["open"].astype(float).to_numpy()
    x["high"] = df["high"].astype(float).to_numpy()
    x["low"] = df["low"].astype(float).to_numpy()
    x["close"] = df["close"].astype(float).to_numpy()
    x["volume"] = df["volume"].astype(float).to_numpy()
    x = pinfilter.add_structure_tags(x)

    close = x["close"].astype(float)
    volume = x["volume"].astype(float)
    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    body_low = pd.concat([x["open"], x["close"]], axis=1).min(axis=1)

    x["trend_slope_3"] = x["trend_line"] / x["trend_line"].shift(3) - 1.0
    x["trend_slope_5"] = x["trend_line"] / x["trend_line"].shift(5) - 1.0
    x["long_slope_5"] = x["long_line"] / x["long_line"].shift(5) - 1.0
    x["trend_line_lead"] = (x["trend_line"] - x["long_line"]) / close.replace(0, np.nan)
    x["ret10"] = close.pct_change(10)
    x["ret3"] = close.pct_change(3)
    x["signal_vs_ma20"] = volume / volume.rolling(20, min_periods=20).mean()
    x["vol_vs_prev"] = volume / volume.shift(1)
    x["close_position"] = (close - x["low"]) / full_range
    x["lower_shadow_ratio"] = (body_low - x["low"]) / full_range

    llv_l_n1 = df_cn["最低"].rolling(window=3).min()
    hhv_c_n1 = df_cn["收盘"].rolling(window=3).max()
    short_term = (df_cn["收盘"] - llv_l_n1) / (hhv_c_n1 - llv_l_n1) * 100
    llv_l_n2 = df_cn["最低"].rolling(window=21).min()
    hhv_l_n2 = df_cn["收盘"].rolling(window=21).max()
    long_term = (df_cn["收盘"] - llv_l_n2) / (hhv_l_n2 - llv_l_n2) * 100
    pin_ok = (short_term <= 30) & (long_term >= 85)

    subtype_a = (
        x["trend_slope_5"].ge(0.03)
        & x["trend_line_lead"].ge(0.03)
        & x["ret10"].ge(0.05)
        & x["ret3"].le(0.01)
        & x["signal_vs_ma20"].le(1.0)
        & x["vol_vs_prev"].le(1.1)
        & x["lower_shadow_ratio"].le(0.25)
    )
    subtype_b = (
        x["trend_slope_3"].ge(0.02)
        & x["trend_slope_5"].ge(0.04)
        & x["long_slope_5"].ge(0.0)
        & x["trend_line_lead"].ge(0.02)
        & x["ret10"].ge(0.10)
        & x["ret3"].le(0.05)
        & x["signal_vs_ma20"].between(0.90, 1.50, inclusive="both")
        & x["vol_vs_prev"].le(1.50)
        & x["close_position"].le(0.20)
        & x["lower_shadow_ratio"].le(0.10)
    )
    subtype_c = (
        (x["along_trend_up"] | x["n_up_any"] | x["keyk_support_active"])
        & x["trend_slope_5"].ge(0.015)
        & x["trend_line_lead"].ge(0.02)
        & x["ret10"].ge(-0.04)
        & x["ret3"].le(0.01)
        & x["signal_vs_ma20"].le(1.0)
        & x["vol_vs_prev"].le(1.1)
        & x["close_position"].le(0.30)
        & x["lower_shadow_ratio"].le(0.25)
    )
    return (weekly_ok & x["trend_line"].gt(x["long_line"]) & pin_ok.fillna(False) & (subtype_a | subtype_b | subtype_c)).fillna(False)


def penalty_score(row: pd.Series) -> int:
    return int(sum(PENALTY_WEIGHTS.get(flag, 0) for flag in PENALTY_WEIGHTS if bool(row.get(flag, False))))


def scan_one_file(path: Path) -> List[Dict[str, object]]:
    df = b2filter.load_one_csv(str(path))
    if df is None or df.empty:
        return []
    df = df.reset_index(drop=True)
    base_df = compute_base_features(df)
    risk_df = add_risk_features(df, precomputed_base=base_df)

    df_cn = pd.DataFrame(
        {
            "日期": df["date"],
            "开盘": df["open"],
            "最高": df["high"],
            "最低": df["low"],
            "收盘": df["close"],
            "成交量": df["volume"],
            "成交额": 0.0,
        }
    )
    weekly_map = b1filter.map_weekly_screen_to_daily_df(df_cn)
    weekly_ok = weekly_map["weekly_ok"].reindex(df_cn.index, fill_value=False)

    b1_mask = build_b1_signal_mask(df_cn, weekly_ok)
    b2_df = b2filter.add_features(df, precomputed_base=base_df)
    b3_df = b3filter.add_features(df, precomputed_b2=b2_df)
    b3_mask = (b3_df["b3_signal"].fillna(False) & weekly_ok)
    pin_mask = build_pin_signal_mask(df, df_cn, weekly_ok)
    brick_df = brick_filter.add_features(df)
    brick_mask = build_brick_signal_mask(brick_df)

    strategy_masks = {
        "B1": b1_mask,
        "B3": b3_mask,
        "PIN": pin_mask,
        "BRICK": brick_mask,
    }

    rows: List[Dict[str, object]] = []
    code = path.stem
    for strategy_name, mask in strategy_masks.items():
        idxs = np.flatnonzero(mask.to_numpy(dtype=bool))
        for idx in idxs:
            metrics = future_metrics(df, int(idx))
            if not metrics:
                continue
            risk_row = risk_df.iloc[int(idx)]
            row: Dict[str, object] = {
                "strategy": strategy_name,
                "code": code,
                "signal_idx": int(idx),
                "signal_date": pd.Timestamp(df.iloc[int(idx)]["date"]),
            }
            for flag in RISK_FLAGS:
                row[flag] = bool(risk_row[flag])
            row["penalty_score"] = penalty_score(pd.Series(row))
            row.update(metrics)
            rows.append(row)
    return rows


def summarize_subset(df: pd.DataFrame, strategy: str, scheme: str, subset: pd.DataFrame) -> Dict[str, object]:
    row: Dict[str, object] = {
        "strategy": strategy,
        "scheme": scheme,
        "sample_count": int(len(subset)),
    }
    for h in HORIZONS:
        close_ret = pd.to_numeric(subset[f"ret_close_{h}"], errors="coerce").dropna()
        up_close = pd.to_numeric(subset[f"up_close_{h}"], errors="coerce").dropna()
        row[f"ret_close_{h}_mean"] = round(float(close_ret.mean()), 4) if not close_ret.empty else np.nan
        row[f"up_close_{h}_rate"] = round(float(up_close.mean()), 4) if not up_close.empty else np.nan
    return row


def build_ab_rows(signal_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for strategy in sorted(signal_df["strategy"].unique()):
        base = signal_df[signal_df["strategy"] == strategy].copy()
        rows.append(summarize_subset(signal_df, strategy, "base", base))

        hard_schemes = {
            "hard_recent_failed_breakout_20": ~base["recent_failed_breakout_20"],
            "hard_top_distribution_20": ~base["top_distribution_20"],
            "hard_recent_heavy_bear_top_20": ~base["recent_heavy_bear_top_20"],
            "hard_risk_segment_rise_slope_10_006": ~base["risk_segment_rise_slope_10_006"],
            "hard_distribution_any_20": ~base["risk_distribution_any_20"],
            "hard_core_distribution_combo": ~(
                base["recent_failed_breakout_20"]
                | base["top_distribution_20"]
                | base["recent_heavy_bear_top_20"]
            ),
        }
        for scheme, keep_mask in hard_schemes.items():
            rows.append(summarize_subset(signal_df, strategy, scheme, base[keep_mask].copy()))

        penalty_schemes = {
            "penalty_keep_0": base["penalty_score"] <= 0,
            "penalty_keep_1": base["penalty_score"] <= 1,
            "penalty_keep_2": base["penalty_score"] <= 2,
            "penalty_keep_3": base["penalty_score"] <= 3,
        }
        for scheme, keep_mask in penalty_schemes.items():
            rows.append(summarize_subset(signal_df, strategy, scheme, base[keep_mask].copy()))
    return pd.DataFrame(rows).sort_values(["strategy", "scheme"]).reset_index(drop=True)


def build_delta(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for strategy in sorted(summary_df["strategy"].unique()):
        base_row = summary_df[(summary_df["strategy"] == strategy) & (summary_df["scheme"] == "base")]
        if base_row.empty:
            continue
        base_row = base_row.iloc[0]
        subset_df = summary_df[(summary_df["strategy"] == strategy) & (summary_df["scheme"] != "base")]
        for _, row in subset_df.iterrows():
            rows.append(
                {
                    "strategy": strategy,
                    "scheme": row["scheme"],
                    "sample_count": int(row["sample_count"]),
                    "sample_keep_ratio": round(float(row["sample_count"]) / float(base_row["sample_count"]), 4) if base_row["sample_count"] else np.nan,
                    "delta_ret_close_3_mean": round(float(row["ret_close_3_mean"] - base_row["ret_close_3_mean"]), 4),
                    "delta_ret_close_5_mean": round(float(row["ret_close_5_mean"] - base_row["ret_close_5_mean"]), 4),
                    "delta_ret_close_10_mean": round(float(row["ret_close_10_mean"] - base_row["ret_close_10_mean"]), 4),
                    "delta_up_close_3_rate": round(float(row["up_close_3_rate"] - base_row["up_close_3_rate"]), 4),
                    "delta_up_close_5_rate": round(float(row["up_close_5_rate"] - base_row["up_close_5_rate"]), 4),
                    "delta_up_close_10_rate": round(float(row["up_close_10_rate"] - base_row["up_close_10_rate"]), 4),
                }
            )
    return pd.DataFrame(rows).sort_values(["strategy", "delta_ret_close_5_mean", "delta_up_close_5_rate"], ascending=[True, False, False]).reset_index(drop=True)


def summarize_risk_flag(df: pd.DataFrame, strategy: str, risk_flag: str, flag_value: bool) -> Dict[str, object]:
    g = df[(df["strategy"] == strategy) & (df[risk_flag] == flag_value)].copy()
    row: Dict[str, object] = {
        "strategy": strategy,
        "risk_flag": risk_flag,
        "flag_value": int(flag_value),
        "sample_count": int(len(g)),
    }
    for h in HORIZONS:
        close_ret = pd.to_numeric(g[f"ret_close_{h}"], errors="coerce").dropna()
        up_close = pd.to_numeric(g[f"up_close_{h}"], errors="coerce").dropna()
        row[f"ret_close_{h}_mean"] = round(float(close_ret.mean()), 4) if not close_ret.empty else np.nan
        row[f"up_close_{h}_rate"] = round(float(up_close.mean()), 4) if not up_close.empty else np.nan
    return row


def build_recommendations(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for strategy in sorted(delta_df["strategy"].unique()):
        sub = delta_df[delta_df["strategy"] == strategy].copy()
        for _, row in sub.iterrows():
            decision = "忽略"
            if row["sample_keep_ratio"] >= 0.70 and row["delta_up_close_5_rate"] >= 0.015 and row["delta_ret_close_5_mean"] >= 0.002:
                decision = "优先硬过滤"
            elif row["sample_keep_ratio"] >= 0.50 and (row["delta_up_close_5_rate"] >= 0.008 or row["delta_ret_close_5_mean"] >= 0.001):
                decision = "适合做扣分项"
            rows.append(
                {
                    "strategy": strategy,
                    "scheme": row["scheme"],
                    "decision": decision,
                    "sample_keep_ratio": row["sample_keep_ratio"],
                    "delta_up_close_5_rate": row["delta_up_close_5_rate"],
                    "delta_ret_close_5_mean": row["delta_ret_close_5_mean"],
                    "delta_up_close_10_rate": row["delta_up_close_10_rate"],
                    "delta_ret_close_10_mean": row["delta_ret_close_10_mean"],
                }
            )
    return pd.DataFrame(rows).sort_values(["strategy", "decision", "delta_ret_close_5_mean"], ascending=[True, True, False]).reset_index(drop=True)


def main() -> None:
    files = sorted(DATA_DIR.glob("*.txt"))
    all_rows: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(2, WORKERS)) as executor:
        for idx, rows in enumerate(executor.map(scan_one_file, files, chunksize=8), start=1):
            all_rows.extend(rows)
            if idx % 100 == 0 or idx == len(files):
                print({"scan_progress": idx, "total": len(files), "signals": len(all_rows)}, flush=True)

    signal_df = pd.DataFrame(all_rows).sort_values(["strategy", "signal_date", "code"]).reset_index(drop=True)
    signal_df.to_csv(RESULT_DIR / "signal_risk_rows.csv", index=False, encoding="utf-8-sig")

    strategy_summary = (
        signal_df.groupby("strategy")
        .agg(signal_count=("code", "size"))
        .reset_index()
        .sort_values("strategy")
    )
    strategy_summary.to_csv(RESULT_DIR / "strategy_signal_counts.csv", index=False, encoding="utf-8-sig")

    risk_rows: List[Dict[str, object]] = []
    for strategy in sorted(signal_df["strategy"].unique()):
        for risk_flag in RISK_FLAGS:
            risk_rows.append(summarize_risk_flag(signal_df, strategy, risk_flag, False))
            risk_rows.append(summarize_risk_flag(signal_df, strategy, risk_flag, True))
    risk_summary = pd.DataFrame(risk_rows).sort_values(["strategy", "risk_flag", "flag_value"]).reset_index(drop=True)
    risk_summary.to_csv(RESULT_DIR / "risk_flag_summary.csv", index=False, encoding="utf-8-sig")

    ab_summary = build_ab_rows(signal_df)
    ab_summary.to_csv(RESULT_DIR / "ab_summary.csv", index=False, encoding="utf-8-sig")

    ab_delta = build_delta(ab_summary)
    ab_delta.to_csv(RESULT_DIR / "ab_delta.csv", index=False, encoding="utf-8-sig")

    recommendations = build_recommendations(ab_delta)
    recommendations.to_csv(RESULT_DIR / "recommendations.csv", index=False, encoding="utf-8-sig")

    print(strategy_summary.to_dict("records"), flush=True)


if __name__ == "__main__":
    main()
