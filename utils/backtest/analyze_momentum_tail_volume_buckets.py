from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_SCRIPT = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/run_momentum_tail_experiment.py")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/momentum_tail_volume_buckets")


def load_base_module():
    spec = importlib.util.spec_from_file_location("momentum_tail_base_buckets", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_long_line(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0
    return x


def assign_bucket(value: float, edges: List[float]) -> str:
    if not np.isfinite(value):
        return "nan"
    prev = -np.inf
    for edge in edges:
        if value <= edge:
            return f"({prev if np.isfinite(prev) else '-inf'},{edge}]"
        prev = edge
    return f"({edges[-1]},inf)"


def summarize_buckets(df: pd.DataFrame, value_col: str, edges: List[float]) -> List[dict]:
    work = df.copy()
    work["bucket"] = work[value_col].apply(lambda v: assign_bucket(v, edges))
    out = []
    for bucket, g in work.groupby("bucket", sort=False):
        out.append(
            {
                "bucket": bucket,
                "sample_count": int(len(g)),
                "avg_trade_return": float(g["ret"].mean()),
                "success_rate": float(g["success"].mean()),
                "pattern_a_share": float(g["pattern_a"].mean()),
                "pattern_b_share": float(g["pattern_b"].mean()),
            }
        )
    return out


def build_trade_df(base) -> pd.DataFrame:
    feature_map: Dict[str, pd.DataFrame] = base.load_feature_map(base.DATA_DIR)
    combo = base.Combo(rebound_threshold=1.5, gain_limit=0.07, take_profit=0.045, stop_mode="entry_low")
    trades: List[dict] = []
    for code, raw_df in feature_map.items():
        df = add_long_line(raw_df)
        mask_a = df["pattern_a"] & (df["rebound_ratio"] >= combo.rebound_threshold)
        mask_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
        mask = df["signal_base"] & (df["ret1"] <= combo.gain_limit) & (mask_a | mask_b) & (df["trend_line"] > df["long_line"])
        for signal_idx in np.flatnonzero(mask.to_numpy()):
            trade = base.simulate_trade(df, int(signal_idx), combo)
            if trade is None:
                continue
            prev_vol = float(df.at[int(signal_idx) - 1, "volume"]) if int(signal_idx) >= 1 else np.nan
            ma5_vol = float(df["volume"].shift(1).rolling(5).mean().iloc[int(signal_idx)])
            up_leg_avg_vol = float(df.at[int(signal_idx), "up_leg_avg_vol"])
            pullback_avg_vol = float(df.at[int(signal_idx), "pullback_avg_vol"])
            trade["code"] = code
            trade["signal_vol"] = float(df.at[int(signal_idx), "volume"])
            trade["prev_vol"] = prev_vol
            trade["vol_ratio_prev1"] = trade["signal_vol"] / prev_vol if np.isfinite(prev_vol) and prev_vol > 0 else np.nan
            trade["vol_ratio_ma5"] = trade["signal_vol"] / ma5_vol if np.isfinite(ma5_vol) and ma5_vol > 0 else np.nan
            trade["pullback_shrink_ratio"] = pullback_avg_vol / up_leg_avg_vol if np.isfinite(up_leg_avg_vol) and up_leg_avg_vol > 0 else np.nan
            trades.append(trade)
    return pd.DataFrame(trades).sort_values(["signal_date", "code"]).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_module()
    trade_df = build_trade_df(base)
    if trade_df.empty:
        raise ValueError("当前口径下没有交易样本")

    bucket_summary = {
        "tested_combo": {
            "rebound_threshold": 1.5,
            "gain_limit": 0.07,
            "take_profit": 0.045,
            "stop_mode": "entry_low",
            "require_trend_above_long": True,
        },
        "overall_sample_count": int(len(trade_df)),
        "vol_ratio_prev1": summarize_buckets(trade_df, "vol_ratio_prev1", [0.8, 1.1, 1.4, 1.8, 2.5]),
        "vol_ratio_ma5": summarize_buckets(trade_df, "vol_ratio_ma5", [0.8, 1.0, 1.3, 1.8, 2.2]),
        "pullback_shrink_ratio": summarize_buckets(trade_df, "pullback_shrink_ratio", [0.5, 0.7, 0.9, 1.1]),
    }
    trade_df.to_csv(OUTPUT_DIR / "trades_with_volume_features.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(bucket_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(bucket_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
