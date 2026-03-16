from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd


MOD_PATH = "/Users/lidongyang/Desktop/Qstrategy/utils/brick_filter.py"
INPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260312/normal")
OUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/brick_t1_ge4_analysis_20260312")
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")


def load_mod():
    spec = importlib.util.spec_from_file_location("brick_filter_mod", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_research_features(mod, df: pd.DataFrame) -> pd.DataFrame:
    x = mod.add_features(df)
    x["body"] = (x["close"] - x["open"]).abs()
    x["upper_shadow"] = x["high"] - x[["open", "close"]].max(axis=1)
    x["lower_shadow"] = x[["open", "close"]].min(axis=1) - x["low"]
    x["range"] = x["high"] - x["low"]
    x["body_ratio"] = np.where(x["range"] > 0, x["body"] / x["range"], np.nan)
    x["upper_shadow_ratio"] = np.where(x["range"] > 0, x["upper_shadow"] / x["range"], np.nan)
    x["lower_shadow_ratio"] = np.where(x["range"] > 0, x["lower_shadow"] / x["range"], np.nan)
    x["close_position"] = np.where(x["range"] > 0, (x["close"] - x["low"]) / x["range"], np.nan)
    x["trend_distance"] = np.where(x["trend_line"] > 0, (x["close"] - x["trend_line"]) / x["trend_line"], np.nan)
    x["long_distance"] = np.where(x["long_line"] > 0, (x["close"] - x["long_line"]) / x["long_line"], np.nan)
    x["trend_long_spread"] = np.where(x["long_line"] > 0, (x["trend_line"] - x["long_line"]) / x["long_line"], np.nan)
    x["trend_slope_3"] = np.where(x["trend_line"].shift(3) > 0, x["trend_line"] / x["trend_line"].shift(3) - 1.0, np.nan)
    x["trend_slope_5"] = np.where(x["trend_line"].shift(5) > 0, x["trend_line"] / x["trend_line"].shift(5) - 1.0, np.nan)
    x["long_slope_3"] = np.where(x["long_line"].shift(3) > 0, x["long_line"] / x["long_line"].shift(3) - 1.0, np.nan)
    x["long_slope_5"] = np.where(x["long_line"].shift(5) > 0, x["long_line"] / x["long_line"].shift(5) - 1.0, np.nan)
    x["slope_spread_3"] = x["trend_slope_3"] - x["long_slope_3"]
    x["slope_spread_5"] = x["trend_slope_5"] - x["long_slope_5"]
    x["close_10_high_dist"] = np.where(x["close"].rolling(10).max() > 0, x["close"] / x["close"].rolling(10).max() - 1.0, np.nan)
    x["close_20_high_dist"] = np.where(x["close"].rolling(20).max() > 0, x["close"] / x["close"].rolling(20).max() - 1.0, np.nan)
    x["close_20_low_dist"] = np.where(x["close"].rolling(20).min() > 0, x["close"] / x["close"].rolling(20).min() - 1.0, np.nan)
    x["vol_vs_prev"] = mod.safe_div(x["volume"], x["volume"].shift(1))
    x["vol_ma10_prev"] = x["volume"].shift(1).rolling(10).mean()
    x["vol_vs_ma10"] = mod.safe_div(x["volume"], x["vol_ma10_prev"])
    x["vol_rank_20"] = x["volume"].rolling(20).rank(pct=True)
    x["ret_5d_before"] = x["close"].shift(1) / x["close"].shift(6) - 1.0
    x["ret_10d_before"] = x["close"].shift(1) / x["close"].shift(11) - 1.0
    x["pullback_shrink_ratio"] = np.where(x["up_leg_avg_vol"] > 0, x["pullback_avg_vol"] / x["up_leg_avg_vol"], np.nan)
    x["next_open"] = x["open"].shift(-1)
    x["next_close"] = x["close"].shift(-1)
    x["t1_close_vs_signal_close"] = np.where(x["close"] > 0, x["next_close"] / x["close"] - 1.0, np.nan)
    x["t1_close_vs_next_open"] = np.where(x["next_open"] > 0, x["next_close"] / x["next_open"] - 1.0, np.nan)
    return x


def bucket_share(series: pd.Series, cond) -> float:
    valid = series.notna()
    if valid.sum() == 0:
        return np.nan
    return float(cond[valid].mean())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_mod()
    rows = []
    files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in {".txt", ".csv"}])
    total = len(files)
    for idx, path in enumerate(files, 1):
        df = mod.load_one_csv(str(path))
        if df is None or df.empty:
            continue
        code = str(df["code"].iloc[0])
        x = add_research_features(mod, df)
        mask_a = x["pattern_a"] & (x["rebound_ratio"] >= 1.2)
        mask_b = x["pattern_b"] & (x["rebound_ratio"] >= 1.0)
        mask = (
            x["signal_base"]
            & (x["ret1"] <= 0.08)
            & (mask_a | mask_b)
            & (x["trend_line"] > x["long_line"])
        )
        sig = x.loc[mask].copy()
        sig = sig[(sig["date"] < EXCLUDE_START) | (sig["date"] > EXCLUDE_END)]
        if sig.empty:
            continue
        for _, r in sig.iterrows():
            rows.append(
                {
                    "date": pd.Timestamp(r["date"]),
                    "code": code,
                    "pattern": "3绿1红" if bool(r["pattern_a"]) else "3绿1红1绿1红",
                    "signal_ret1": float(r["ret1"]),
                    "t1_close_vs_signal_close": float(r["t1_close_vs_signal_close"]) if pd.notna(r["t1_close_vs_signal_close"]) else np.nan,
                    "t1_close_vs_next_open": float(r["t1_close_vs_next_open"]) if pd.notna(r["t1_close_vs_next_open"]) else np.nan,
                    "signal_vs_ma5": float(r["signal_vs_ma5"]) if pd.notna(r["signal_vs_ma5"]) else np.nan,
                    "vol_vs_prev": float(r["vol_vs_prev"]) if pd.notna(r["vol_vs_prev"]) else np.nan,
                    "vol_vs_ma10": float(r["vol_vs_ma10"]) if pd.notna(r["vol_vs_ma10"]) else np.nan,
                    "vol_rank_20": float(r["vol_rank_20"]) if pd.notna(r["vol_rank_20"]) else np.nan,
                    "rebound_ratio": float(r["rebound_ratio"]) if pd.notna(r["rebound_ratio"]) else np.nan,
                    "pullback_shrink_ratio": float(r["pullback_shrink_ratio"]) if pd.notna(r["pullback_shrink_ratio"]) else np.nan,
                    "trend_distance": float(r["trend_distance"]) if pd.notna(r["trend_distance"]) else np.nan,
                    "long_distance": float(r["long_distance"]) if pd.notna(r["long_distance"]) else np.nan,
                    "trend_long_spread": float(r["trend_long_spread"]) if pd.notna(r["trend_long_spread"]) else np.nan,
                    "trend_slope_3": float(r["trend_slope_3"]) if pd.notna(r["trend_slope_3"]) else np.nan,
                    "trend_slope_5": float(r["trend_slope_5"]) if pd.notna(r["trend_slope_5"]) else np.nan,
                    "long_slope_3": float(r["long_slope_3"]) if pd.notna(r["long_slope_3"]) else np.nan,
                    "long_slope_5": float(r["long_slope_5"]) if pd.notna(r["long_slope_5"]) else np.nan,
                    "slope_spread_3": float(r["slope_spread_3"]) if pd.notna(r["slope_spread_3"]) else np.nan,
                    "slope_spread_5": float(r["slope_spread_5"]) if pd.notna(r["slope_spread_5"]) else np.nan,
                    "close_position": float(r["close_position"]) if pd.notna(r["close_position"]) else np.nan,
                    "body_ratio": float(r["body_ratio"]) if pd.notna(r["body_ratio"]) else np.nan,
                    "upper_shadow_ratio": float(r["upper_shadow_ratio"]) if pd.notna(r["upper_shadow_ratio"]) else np.nan,
                    "lower_shadow_ratio": float(r["lower_shadow_ratio"]) if pd.notna(r["lower_shadow_ratio"]) else np.nan,
                    "close_10_high_dist": float(r["close_10_high_dist"]) if pd.notna(r["close_10_high_dist"]) else np.nan,
                    "close_20_high_dist": float(r["close_20_high_dist"]) if pd.notna(r["close_20_high_dist"]) else np.nan,
                    "close_20_low_dist": float(r["close_20_low_dist"]) if pd.notna(r["close_20_low_dist"]) else np.nan,
                    "ret_5d_before": float(r["ret_5d_before"]) if pd.notna(r["ret_5d_before"]) else np.nan,
                    "ret_10d_before": float(r["ret_10d_before"]) if pd.notna(r["ret_10d_before"]) else np.nan,
                }
            )
        if idx % 500 == 0 or idx == total:
            print(f"扫描进度: {idx}/{total}")

    df = pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)
    df["t1_ge_4"] = df["t1_close_vs_signal_close"] >= 0.04
    df.to_csv(OUT_DIR / "signals_with_t1.csv", index=False, encoding="utf-8-sig")

    focus = df[df["t1_ge_4"]].copy()
    other = df[~df["t1_ge_4"]].copy()
    feature_cols = [
        "signal_ret1",
        "signal_vs_ma5",
        "vol_vs_prev",
        "vol_vs_ma10",
        "vol_rank_20",
        "rebound_ratio",
        "pullback_shrink_ratio",
        "trend_distance",
        "long_distance",
        "trend_long_spread",
        "trend_slope_3",
        "trend_slope_5",
        "long_slope_3",
        "long_slope_5",
        "slope_spread_3",
        "slope_spread_5",
        "close_position",
        "body_ratio",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "close_10_high_dist",
        "close_20_high_dist",
        "close_20_low_dist",
        "ret_5d_before",
        "ret_10d_before",
    ]
    compare_rows = []
    for col in feature_cols:
        compare_rows.append(
            {
                "feature": col,
                "focus_mean": focus[col].mean(),
                "focus_median": focus[col].median(),
                "other_mean": other[col].mean(),
                "other_median": other[col].median(),
                "focus_minus_other_mean": focus[col].mean() - other[col].mean(),
                "focus_minus_other_median": focus[col].median() - other[col].median(),
            }
        )
    compare = pd.DataFrame(compare_rows)
    compare.to_csv(OUT_DIR / "focus_vs_other_compare.csv", index=False, encoding="utf-8-sig")

    payload = {
        "summary": {
            "signal_count": int(len(df)),
            "focus_count": int(len(focus)),
            "focus_share": float(len(focus) / len(df)) if len(df) else np.nan,
            "date_min": df["date"].min().date().isoformat() if len(df) else None,
            "date_max": df["date"].max().date().isoformat() if len(df) else None,
        },
        "focus_commonality": {
            "signal_vs_ma5_1_3_to_1_8_share": bucket_share(focus["signal_vs_ma5"], focus["signal_vs_ma5"].between(1.3, 1.8, inclusive="both")),
            "signal_vs_ma5_1_8_to_2_2_share": bucket_share(focus["signal_vs_ma5"], focus["signal_vs_ma5"].between(1.8, 2.2, inclusive="right")),
            "pullback_shrink_0_7_to_0_9_share": bucket_share(focus["pullback_shrink_ratio"], focus["pullback_shrink_ratio"].between(0.7, 0.9, inclusive="both")),
            "trend_distance_0_to_3pct_share": bucket_share(focus["trend_distance"], focus["trend_distance"].between(0.0, 0.03, inclusive="both")),
            "long_distance_0_to_5pct_share": bucket_share(focus["long_distance"], focus["long_distance"].between(0.0, 0.05, inclusive="both")),
            "trend_slope_3_positive_share": bucket_share(focus["trend_slope_3"], focus["trend_slope_3"] > 0),
            "trend_slope_5_positive_share": bucket_share(focus["trend_slope_5"], focus["trend_slope_5"] > 0),
            "slope_spread_3_positive_share": bucket_share(focus["slope_spread_3"], focus["slope_spread_3"] > 0),
            "upper_shadow_ratio_le_0_15_share": bucket_share(focus["upper_shadow_ratio"], focus["upper_shadow_ratio"] <= 0.15),
            "close_position_ge_0_7_share": bucket_share(focus["close_position"], focus["close_position"] >= 0.7),
            "close_within_5pct_of_20d_high_share": bucket_share(focus["close_20_high_dist"], focus["close_20_high_dist"] >= -0.05),
            "close_above_20d_low_by_10pct_share": bucket_share(focus["close_20_low_dist"], focus["close_20_low_dist"] >= 0.10),
            "ret_10d_before_positive_share": bucket_share(focus["ret_10d_before"], focus["ret_10d_before"] > 0),
            "ret_10d_before_negative_5pct_share": bucket_share(focus["ret_10d_before"], focus["ret_10d_before"] <= -0.05),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("\nFOCUS_SIGNALS_TAIL")
    print(focus[["date", "code", "pattern", "t1_close_vs_signal_close"]].tail(60).to_string(index=False))


if __name__ == "__main__":
    main()
