from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b2filter, b3filter, brick_filter  # type: ignore
from utils.market_risk_tags import add_risk_features  # type: ignore


DATA_DIR = ROOT / "data/20260315/normal"
RESULT_DIR = ROOT / "results/strategy_risk_signal_analysis_20260319"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = (3, 5, 10)
RISK_FLAGS = [
    "risk_fast_rise_5d_30",
    "risk_fast_rise_5d_40",
    "risk_fast_rise_10d_40",
    "risk_segment_rise_slope_10_006",
    "risk_segment_rise_slope_15_005",
    "recent_heavy_bear_top_20",
    "recent_failed_breakout_20",
    "top_distribution_20",
    "recent_stair_bear_20",
    "risk_distribution_any_20",
]


def build_brick_signal_mask(x: pd.DataFrame) -> pd.Series:
    mask_a = x["pattern_a"] & (x["rebound_ratio"] >= 0.8)
    mask_b = x["pattern_b"] & (x["rebound_ratio"] >= 1.0)
    return (
        x["signal_base"].fillna(False)
        & x["trend_line"].gt(x["long_line"])
        & x["ret1"].between(-0.03, 0.11, inclusive="both")
        & (mask_a | mask_b)
    )


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


def scan_one_file(path: Path) -> List[Dict[str, object]]:
    df = b2filter.load_one_csv(str(path))
    if df is None or df.empty:
        return []
    df = df.reset_index(drop=True)

    risk_df = add_risk_features(df)
    b2_df = b2filter.add_features(df)
    b3_df = b3filter.add_features(df, precomputed_b2=b2_df)
    brick_df = brick_filter.add_features(df)
    brick_mask = build_brick_signal_mask(brick_df)

    strategy_masks = {
        "B2": b2_df["b2_signal"].fillna(False),
        "B3": b3_df["b3_signal"].fillna(False),
        "BRICK": brick_mask.fillna(False),
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
            row.update(metrics)
            rows.append(row)
    return rows


def summarize_group(df: pd.DataFrame, strategy: str, risk_flag: str, flag_value: bool) -> Dict[str, object]:
    g = df[(df["strategy"] == strategy) & (df[risk_flag] == flag_value)].copy()
    row: Dict[str, object] = {
        "strategy": strategy,
        "risk_flag": risk_flag,
        "flag_value": int(flag_value),
        "sample_count": int(len(g)),
    }
    for h in HORIZONS:
        close_ret = pd.to_numeric(g[f"ret_close_{h}"], errors="coerce").dropna()
        max_ret = pd.to_numeric(g[f"ret_max_{h}"], errors="coerce").dropna()
        up_close = pd.to_numeric(g[f"up_close_{h}"], errors="coerce").dropna()
        up_max = pd.to_numeric(g[f"up_max_{h}"], errors="coerce").dropna()
        row[f"ret_close_{h}_mean"] = round(float(close_ret.mean()), 4) if not close_ret.empty else np.nan
        row[f"ret_close_{h}_median"] = round(float(close_ret.median()), 4) if not close_ret.empty else np.nan
        row[f"ret_max_{h}_mean"] = round(float(max_ret.mean()), 4) if not max_ret.empty else np.nan
        row[f"up_close_{h}_rate"] = round(float(up_close.mean()), 4) if not up_close.empty else np.nan
        row[f"up_max_{h}_rate"] = round(float(up_max.mean()), 4) if not up_max.empty else np.nan
    return row


def main() -> None:
    files = sorted(DATA_DIR.glob("*.txt"))
    all_rows: List[Dict[str, object]] = []
    for idx, path in enumerate(files, start=1):
        all_rows.extend(scan_one_file(path))
        if idx % 300 == 0:
            print({"scan_progress": idx, "total": len(files), "signals": len(all_rows)}, flush=True)

    signal_df = pd.DataFrame(all_rows).sort_values(["strategy", "signal_date", "code"]).reset_index(drop=True)
    signal_df.to_csv(RESULT_DIR / "signal_risk_rows.csv", index=False, encoding="utf-8-sig")

    summary_rows: List[Dict[str, object]] = []
    for strategy in sorted(signal_df["strategy"].unique()):
        for risk_flag in RISK_FLAGS:
            summary_rows.append(summarize_group(signal_df, strategy, risk_flag, False))
            summary_rows.append(summarize_group(signal_df, strategy, risk_flag, True))

    summary_df = pd.DataFrame(summary_rows).sort_values(["strategy", "risk_flag", "flag_value"]).reset_index(drop=True)
    summary_df.to_csv(RESULT_DIR / "risk_flag_summary.csv", index=False, encoding="utf-8-sig")

    strategy_summary = (
        signal_df.groupby("strategy")
        .agg(signal_count=("code", "size"))
        .reset_index()
        .sort_values("strategy")
    )
    strategy_summary.to_csv(RESULT_DIR / "strategy_signal_counts.csv", index=False, encoding="utf-8-sig")
    print(strategy_summary.to_dict("records"))


if __name__ == "__main__":
    main()
