from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.data_loader import load_price_directory
from strategies.common import calc_trend_cn


def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    avg_up = up.ewm(span=period, adjust=False).mean()
    avg_down = down.ewm(span=period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def add_pin_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    low_3 = out["low"].rolling(window=3).min()
    close_high_3 = out["close"].rolling(window=3).max()
    low_21 = out["low"].rolling(window=21).min()
    close_high_21 = out["close"].rolling(window=21).max()
    short_denom = (close_high_3 - low_3).replace(0.0, np.nan)
    long_denom = (close_high_21 - low_21).replace(0.0, np.nan)
    out["pin_short"] = ((out["close"] - low_3) / short_denom) * 100.0
    out["pin_long"] = ((out["close"] - low_21) / long_denom) * 100.0
    out["pin_short"] = out["pin_short"].replace([np.inf, -np.inf], np.nan)
    out["pin_long"] = out["pin_long"].replace([np.inf, -np.inf], np.nan)
    out["pin_shape"] = (out["pin_short"] <= 30.0) & (out["pin_long"] >= 85.0)
    return out


def top_volume_days_all_bullish(volume: pd.Series, bullish: pd.Series, window: int = 60, top_n: int = 2) -> pd.Series:
    values = volume.to_numpy(dtype=float)
    bull = bullish.fillna(False).to_numpy(dtype=bool)
    result = np.zeros(len(volume), dtype=bool)
    for i in range(len(volume)):
        start = max(0, i - window + 1)
        window_vol = values[start : i + 1]
        window_bull = bull[start : i + 1]
        if len(window_vol) < top_n:
            continue
        top_idx = np.argpartition(window_vol, -top_n)[-top_n:]
        result[i] = bool(window_bull[top_idx].all())
    return pd.Series(result, index=volume.index)


def prepare_stock(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["short_trend"], out["long_trend"] = calc_trend_cn(out["close"])
    out = add_pin_features(out)

    prev_close = out["close"].shift(1)
    prev_open = out["open"].shift(1)
    prev_volume = out["volume"].shift(1)
    price_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    body_high = out[["open", "close"]].max(axis=1)
    body_low = out[["open", "close"]].min(axis=1)

    out["rsi14"] = calc_rsi(out["close"], 14)
    out["rsi28"] = calc_rsi(out["close"], 28)
    out["rsi57"] = calc_rsi(out["close"], 57)
    out["return_1d"] = out["close"].pct_change(1)
    out["return_3d"] = out["close"].pct_change(3)
    out["return_5d"] = out["close"].pct_change(5)
    out["return_10d"] = out["close"].pct_change(10)
    out["return_20d"] = out["close"].pct_change(20)
    out["gap_return"] = out["open"] / prev_close - 1.0
    out["intraday_return"] = out["close"] / out["open"] - 1.0
    out["signal_day_return"] = out["close"] / prev_close - 1.0
    out["prev_day_return"] = prev_close / out["close"].shift(2) - 1.0
    out["amplitude"] = out["high"] / out["low"] - 1.0
    out["body_ratio"] = (out["close"] - out["open"]).abs() / price_range
    out["close_in_range"] = (out["close"] - out["low"]) / price_range
    out["open_in_range"] = (out["open"] - out["low"]) / price_range
    out["lower_shadow_ratio"] = (body_low - out["low"]) / price_range
    out["upper_shadow_ratio"] = (out["high"] - body_high) / price_range
    out["volume_ratio_prev"] = out["volume"] / prev_volume.replace(0.0, np.nan)
    out["volume_ratio_ma5"] = out["volume"] / out["volume"].rolling(5).mean().replace(0.0, np.nan)
    out["volume_ratio_ma10"] = out["volume"] / out["volume"].rolling(10).mean().replace(0.0, np.nan)
    out["volume_ratio_ma20"] = out["volume"] / out["volume"].rolling(20).mean().replace(0.0, np.nan)
    out["distance_to_trend"] = out["close"] / out["short_trend"] - 1.0
    out["distance_to_long"] = out["close"] / out["long_trend"] - 1.0
    out["trend_spread"] = out["short_trend"] / out["long_trend"] - 1.0
    out["trend_slope_3"] = out["short_trend"] / out["short_trend"].shift(3) - 1.0
    out["trend_slope_5"] = out["short_trend"] / out["short_trend"].shift(5) - 1.0
    out["long_slope_3"] = out["long_trend"] / out["long_trend"].shift(3) - 1.0
    out["long_slope_5"] = out["long_trend"] / out["long_trend"].shift(5) - 1.0
    out["volatility_5"] = out["return_1d"].rolling(5).std()
    out["volatility_10"] = out["return_1d"].rolling(10).std()
    out["rsi_stack_bull"] = (out["rsi14"] > out["rsi28"]) & (out["rsi28"] > out["rsi57"])
    out["close_up"] = out["close"] > prev_close
    out["volume_up"] = out["volume"] > prev_volume
    out["today_bearish"] = out["close"] < out["open"]
    out["yesterday_bullish"] = prev_close > prev_open
    out["volume_below_half_30d_max"] = out["volume"] <= (out["volume"].rolling(30).max() / 2.0)
    out["top2_volume_days_all_bullish_60d"] = top_volume_days_all_bullish(out["volume"], out["close"] > out["open"])
    out["moderate_shrink_volume"] = (out["volume_ratio_prev"] > 0.7) & (out["volume_ratio_prev"] <= 0.9)
    out["mild_expand_volume"] = (out["volume_ratio_prev"] > 1.1) & (out["volume_ratio_prev"] <= 1.5)
    out["deep_signal_drop"] = out["signal_day_return"] <= -0.05
    out["strong_5d_momentum"] = out["return_5d"] > 0.10
    out["friday_signal"] = out.index.weekday == 4
    out["gem_board"] = out["board"] == "GEM"
    out["close_above_prev_close"] = out["close"] > prev_close
    out["trend_above_long"] = out["short_trend"] > out["long_trend"]
    out["signal"] = out["trend_above_long"] & out["pin_shape"]

    out["next_date"] = out.index.to_series().shift(-1)
    out["next_open"] = out["open"].shift(-1)
    out["next_close"] = out["close"].shift(-1)
    out["next_high"] = out["high"].shift(-1)
    out["next_low"] = out["low"].shift(-1)
    out["close_t2"] = out["close"].shift(-2)
    out["close_t3"] = out["close"].shift(-3)
    out["next_day_return_close_to_close"] = out["next_close"] / out["close"] - 1.0
    out["next_day_return_open_to_close"] = out["next_close"] / out["next_open"] - 1.0
    out["next_day_bull"] = out["next_close"] > out["next_open"]
    out["hold_1d_return"] = out["next_close"] / out["next_open"] - 1.0
    out["hold_2d_return"] = out["close_t2"] / out["next_open"] - 1.0
    out["hold_3d_return"] = out["close_t3"] / out["next_open"] - 1.0
    out["hold_1d_win"] = out["hold_1d_return"] > 0
    out["hold_2d_win"] = out["hold_2d_return"] > 0
    out["hold_3d_win"] = out["hold_3d_return"] > 0
    out["weekday"] = out.index.weekday
    out["year"] = out.index.year
    return out


def collect_signals(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: List[pd.DataFrame] = []
    for code, raw_df in stock_data.items():
        df = prepare_stock(raw_df)
        signal_df = df.loc[df["signal"].fillna(False)].copy()
        if signal_df.empty:
            continue
        signal_df = signal_df.loc[signal_df["next_date"].notna()].copy()
        signal_df["code"] = code
        signal_df["signal_date"] = signal_df.index
        records.append(signal_df.reset_index(drop=True))
    if not records:
        return pd.DataFrame()
    combined = pd.concat(records, ignore_index=True)
    columns = [
        "code",
        "board",
        "signal_date",
        "next_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "short_trend",
        "long_trend",
        "pin_short",
        "pin_long",
        "rsi14",
        "rsi28",
        "rsi57",
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "gap_return",
        "intraday_return",
        "signal_day_return",
        "prev_day_return",
        "amplitude",
        "body_ratio",
        "close_in_range",
        "open_in_range",
        "lower_shadow_ratio",
        "upper_shadow_ratio",
        "volume_ratio_prev",
        "volume_ratio_ma5",
        "volume_ratio_ma10",
        "volume_ratio_ma20",
        "distance_to_trend",
        "distance_to_long",
        "trend_spread",
        "trend_slope_3",
        "trend_slope_5",
        "long_slope_3",
        "long_slope_5",
        "volatility_5",
        "volatility_10",
        "rsi_stack_bull",
        "close_up",
        "volume_up",
        "today_bearish",
        "yesterday_bullish",
        "volume_below_half_30d_max",
        "top2_volume_days_all_bullish_60d",
        "moderate_shrink_volume",
        "mild_expand_volume",
        "deep_signal_drop",
        "strong_5d_momentum",
        "friday_signal",
        "gem_board",
        "close_above_prev_close",
        "next_open",
        "next_close",
        "next_high",
        "next_low",
        "close_t2",
        "close_t3",
        "next_day_return_close_to_close",
        "next_day_return_open_to_close",
        "next_day_bull",
        "hold_1d_return",
        "hold_2d_return",
        "hold_3d_return",
        "hold_1d_win",
        "hold_2d_win",
        "hold_3d_win",
        "weekday",
        "year",
    ]
    return combined[columns].sort_values(["code", "signal_date"]).reset_index(drop=True)


def numeric_feature_summary(signals: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in signals.columns
        if col
        not in {"next_day_bull", "weekday", "year"}
        and pd.api.types.is_numeric_dtype(signals[col])
        and col not in {"next_open", "next_close", "next_high", "next_low"}
    ]
    success = signals.loc[signals["next_day_bull"]].copy()
    failure = signals.loc[~signals["next_day_bull"]].copy()
    rows = []
    for col in numeric_cols:
        s = success[col].replace([np.inf, -np.inf], np.nan).dropna()
        f = failure[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 20 or len(f) < 20:
            continue
        success_mean = float(s.mean())
        failure_mean = float(f.mean())
        pooled_std = float(pd.concat([s, f]).std(ddof=0))
        std_diff = 0.0 if pooled_std == 0 else (success_mean - failure_mean) / pooled_std
        rows.append(
            {
                "feature": col,
                "success_mean": success_mean,
                "failure_mean": failure_mean,
                "success_median": float(s.median()),
                "failure_median": float(f.median()),
                "mean_diff": success_mean - failure_mean,
                "std_diff": std_diff,
                "abs_std_diff": abs(std_diff),
            }
        )
    summary = pd.DataFrame(rows).sort_values("abs_std_diff", ascending=False).reset_index(drop=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def boolean_feature_summary(signals: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    bool_cols = [
        col
        for col in signals.columns
        if pd.api.types.is_bool_dtype(signals[col]) and col != "next_day_bull"
    ]
    base_rate = float(signals["next_day_bull"].mean())
    rows = []
    for col in bool_cols:
        subset = signals.loc[signals[col].fillna(False)].copy()
        if len(subset) < 20:
            continue
        hit_rate = float(subset["next_day_bull"].mean())
        rows.append(
            {
                "feature": col,
                "sample_count": int(len(subset)),
                "sample_ratio": float(len(subset) / len(signals)),
                "hit_rate": hit_rate,
                "base_rate": base_rate,
                "lift": hit_rate - base_rate,
            }
        )
    summary = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def bucketed_summary(signals: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    bucket_specs = {
        "volume_ratio_prev": [-np.inf, 0.7, 0.9, 1.1, 1.5, np.inf],
        "rsi14": [-np.inf, 30, 40, 50, 60, 70, np.inf],
        "return_5d": [-np.inf, -0.10, -0.05, 0.0, 0.05, 0.10, np.inf],
        "signal_day_return": [-np.inf, -0.05, -0.03, -0.01, 0.0, 0.02, np.inf],
        "distance_to_trend": [-np.inf, -0.03, -0.01, 0.0, 0.01, 0.03, np.inf],
    }
    rows = []
    for feature, bins in bucket_specs.items():
        subset = signals[[feature, "next_day_bull"]].replace([np.inf, -np.inf], np.nan).dropna()
        if subset.empty:
            continue
        cats = pd.cut(subset[feature], bins=bins, include_lowest=True)
        grouped = subset.groupby(cats, observed=False)["next_day_bull"].agg(["size", "mean"]).reset_index()
        for _, row in grouped.iterrows():
            if int(row["size"]) == 0:
                continue
            rows.append(
                {
                    "feature": feature,
                    "bucket": str(row[feature]),
                    "sample_count": int(row["size"]),
                    "hit_rate": float(row["mean"]),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def board_summary(signals: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = (
        signals.groupby("board", observed=False)["next_day_bull"]
        .agg(signal_count="size", hit_rate="mean")
        .reset_index()
        .sort_values("hit_rate", ascending=False)
    )
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def factor_audit(signals: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    factor_columns = [
        "rsi_stack_bull",
        "today_bearish",
        "yesterday_bullish",
        "volume_below_half_30d_max",
        "top2_volume_days_all_bullish_60d",
        "moderate_shrink_volume",
        "mild_expand_volume",
        "deep_signal_drop",
        "strong_5d_momentum",
        "friday_signal",
        "gem_board",
        "close_above_prev_close",
        "volume_up",
        "close_up",
    ]
    base_hit_rate = float(signals["next_day_bull"].mean())
    rows = []
    for factor in factor_columns:
        subset = signals.loc[signals[factor].fillna(False)].copy()
        if len(subset) < 20:
            continue
        rows.append(
            {
                "factor": factor,
                "sample_count": int(len(subset)),
                "sample_ratio": float(len(subset) / len(signals)),
                "next_day_bull_rate": float(subset["next_day_bull"].mean()),
                "next_day_bull_lift": float(subset["next_day_bull"].mean() - base_hit_rate),
                "hold_1d_mean_return": float(subset["hold_1d_return"].mean()),
                "hold_2d_mean_return": float(subset["hold_2d_return"].mean()),
                "hold_3d_mean_return": float(subset["hold_3d_return"].mean()),
                "hold_1d_win_rate": float(subset["hold_1d_win"].mean()),
                "hold_2d_win_rate": float(subset["hold_2d_win"].mean()),
                "hold_3d_win_rate": float(subset["hold_3d_win"].mean()),
            }
        )
    audit = pd.DataFrame(rows).sort_values(
        ["next_day_bull_lift", "hold_1d_mean_return"], ascending=False
    ).reset_index(drop=True)
    audit.to_csv(output_path, index=False, encoding="utf-8-sig")
    return audit


def holding_period_summary(signals: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for days in [1, 2, 3]:
        ret_col = f"hold_{days}d_return"
        win_col = f"hold_{days}d_win"
        subset = signals[[ret_col, win_col]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "holding_days": days,
                "sample_count": int(len(subset)),
                "mean_return": float(subset[ret_col].mean()),
                "median_return": float(subset[ret_col].median()),
                "win_rate": float(subset[win_col].mean()),
                "positive_return_share": float((subset[ret_col] > 0).mean()),
                "p10_return": float(subset[ret_col].quantile(0.10)),
                "p90_return": float(subset[ret_col].quantile(0.90)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def top_examples(signals: pd.DataFrame) -> Dict[str, List[dict]]:
    winners = signals.loc[signals["next_day_bull"]].sort_values(
        "next_day_return_open_to_close", ascending=False
    )
    losers = signals.loc[~signals["next_day_bull"]].sort_values(
        "next_day_return_open_to_close", ascending=True
    )

    def _pack(df: pd.DataFrame) -> List[dict]:
        cols = [
            "code",
            "signal_date",
            "next_date",
            "board",
            "signal_day_return",
            "volume_ratio_prev",
            "rsi14",
            "return_5d",
            "distance_to_trend",
            "next_day_return_open_to_close",
        ]
        sample = df[cols].head(10).copy()
        sample["signal_date"] = sample["signal_date"].dt.strftime("%Y-%m-%d")
        sample["next_date"] = sample["next_date"].dt.strftime("%Y-%m-%d")
        return sample.to_dict(orient="records")

    return {"top_winners": _pack(winners), "top_losers": _pack(losers)}


def build_summary(
    signals: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    boolean_summary: pd.DataFrame,
    board_stats: pd.DataFrame,
) -> dict:
    success = signals.loc[signals["next_day_bull"]]
    failure = signals.loc[~signals["next_day_bull"]]
    per_stock = signals.groupby("code", observed=False).size()
    positive_numeric = numeric_summary.loc[numeric_summary["std_diff"] > 0].head(12)
    positive_bool = boolean_summary.loc[boolean_summary["lift"] > 0].head(12)
    summary = {
        "data_scope": "data/20260311/normal",
        "signal_definition": {
            "trend_filter": "short_trend > long_trend",
            "pin_filter": "pin_short <= 30 and pin_long >= 85",
            "removed_filters": [
                "RSI_14 > RSI_28 > RSI_57",
                "today_volume <= max_30_volume / 2",
                "today_close < today_open",
                "yesterday_close > yesterday_open",
                "top_volume_days_all_bullish",
            ],
        },
        "signal_count": int(len(signals)),
        "stock_count_with_signal": int(signals["code"].nunique()),
        "avg_signals_per_stock": float(per_stock.mean()),
        "median_signals_per_stock": float(per_stock.median()),
        "next_day_bull_count": int(len(success)),
        "next_day_bull_rate": float(success.shape[0] / len(signals)) if len(signals) else 0.0,
        "next_day_open_to_close_mean": float(signals["next_day_return_open_to_close"].mean()),
        "next_day_open_to_close_mean_success": float(success["next_day_return_open_to_close"].mean()) if len(success) else 0.0,
        "next_day_open_to_close_mean_failure": float(failure["next_day_return_open_to_close"].mean()) if len(failure) else 0.0,
        "top_positive_numeric_commonalities": positive_numeric.to_dict(orient="records"),
        "top_positive_boolean_commonalities": positive_bool.to_dict(orient="records"),
        "best_boards": board_stats.head(5).to_dict(orient="records"),
        "worst_boards": board_stats.sort_values("hit_rate", ascending=True).head(5).to_dict(orient="records"),
    }
    summary.update(top_examples(signals))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze minimal pinfilter historical signals")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="/Users/lidongyang/Desktop/Qstrategy/data/20260311/normal",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/lidongyang/Desktop/Qstrategy/results/pin_minimal_signal_analysis_20260311",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_data, _ = load_price_directory(args.data_dir)
    signals = collect_signals(stock_data)
    if signals.empty:
        raise SystemExit("No signals found.")

    signals.to_csv(output_dir / "historical_signals.csv", index=False, encoding="utf-8-sig")
    numeric_summary = numeric_feature_summary(signals, output_dir / "numeric_commonality_rank.csv")
    boolean_summary = boolean_feature_summary(signals, output_dir / "boolean_commonality_rank.csv")
    bucketed_summary(signals, output_dir / "bucket_hit_rates.csv")
    board_stats = board_summary(signals, output_dir / "board_hit_rates.csv")
    factor_audit(signals, output_dir / "factor_audit.csv")
    holding_period_summary(signals, output_dir / "holding_period_summary.csv")

    per_stock = (
        signals.groupby("code", observed=False)
        .agg(signal_count=("code", "size"), bull_count=("next_day_bull", "sum"))
        .reset_index()
    )
    per_stock["next_day_bull_rate"] = per_stock["bull_count"] / per_stock["signal_count"]
    per_stock.to_csv(output_dir / "per_stock_signal_summary.csv", index=False, encoding="utf-8-sig")

    summary = build_summary(signals, numeric_summary, boolean_summary, board_stats)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
