from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_SCRIPT = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/run_momentum_tail_experiment.py")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/momentum_tail_ranking_models")
TOP_N = 10


def load_base_module():
    spec = importlib.util.spec_from_file_location("momentum_tail_base_ranking", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def add_long_line(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0
    return x


def triangle_quality(value: float, center: float, half_width: float) -> float:
    if not np.isfinite(value) or half_width <= 0:
        return 0.0
    score = 1.0 - abs(value - center) / half_width
    return float(max(score, 0.0))


def compute_signal_features(df: pd.DataFrame, signal_idx: int) -> dict:
    trend_line = float(df.at[signal_idx, "trend_line"])
    long_line = float(df.at[signal_idx, "long_line"])
    close = float(df.at[signal_idx, "close"])
    ret1 = float(df.at[signal_idx, "ret1"])
    signal_vs_ma5 = float(df.at[signal_idx, "signal_vs_ma5"])
    rebound_ratio = float(df.at[signal_idx, "rebound_ratio"])
    brick_red_len = float(df.at[signal_idx, "brick_red_len"])
    pullback_avg_vol = float(df.at[signal_idx, "pullback_avg_vol"])
    up_leg_avg_vol = float(df.at[signal_idx, "up_leg_avg_vol"])
    pullback_shrink_ratio = pullback_avg_vol / up_leg_avg_vol if np.isfinite(up_leg_avg_vol) and up_leg_avg_vol > 0 else np.nan
    trend_spread_pct = (trend_line - long_line) / close if np.isfinite(close) and close > 0 else np.nan
    return {
        "ret1": ret1,
        "rebound_ratio": rebound_ratio,
        "brick_red_len": brick_red_len,
        "signal_vs_ma5": signal_vs_ma5,
        "pullback_shrink_ratio": pullback_shrink_ratio,
        "trend_spread_pct": trend_spread_pct,
        "pattern_a": bool(df.at[signal_idx, "pattern_a"]),
        "pattern_b": bool(df.at[signal_idx, "pattern_b"]),
        "ret1_quality": triangle_quality(ret1, 0.03, 0.03),
        "signal_vs_ma5_quality": triangle_quality(signal_vs_ma5, 1.7, 0.5),
        "shrink_quality": triangle_quality(pullback_shrink_ratio, 0.8, 0.3),
    }


def assign_rank_scores(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return trade_df
    x = trade_df.copy()
    x["trend_spread_clip"] = x["trend_spread_pct"].clip(lower=0.0).fillna(0.0)
    x["rebound_rank"] = x.groupby("signal_date")["rebound_ratio"].rank(pct=True)
    x["brick_rank"] = x.groupby("signal_date")["brick_red_len"].rank(pct=True)
    x["trend_rank"] = x.groupby("signal_date")["trend_spread_clip"].rank(pct=True)
    x["signal_vs_ma5_rank"] = x.groupby("signal_date")["signal_vs_ma5_quality"].rank(pct=True)
    x["shrink_rank"] = x.groupby("signal_date")["shrink_quality"].rank(pct=True)
    x["ret1_rank"] = x.groupby("signal_date")["ret1_quality"].rank(pct=True)
    return x


def build_sort_score(df: pd.DataFrame, model_name: str) -> pd.Series:
    if model_name == "baseline_rebound_heat":
        return df["rebound_ratio"] / np.maximum(df["ret1"].abs(), 0.01)
    if model_name == "rebound_only":
        return df["rebound_ratio"]
    if model_name == "structure_trend":
        return 0.65 * df["rebound_rank"] + 0.35 * df["trend_rank"]
    if model_name == "structure_price":
        return 0.55 * df["rebound_rank"] + 0.25 * df["brick_rank"] + 0.20 * df["ret1_rank"]
    if model_name == "structure_trend_price":
        return 0.45 * df["rebound_rank"] + 0.20 * df["brick_rank"] + 0.20 * df["trend_rank"] + 0.15 * df["ret1_rank"]
    if model_name == "vol_ma5_focus":
        return 0.55 * df["signal_vs_ma5_rank"] + 0.25 * df["rebound_rank"] + 0.20 * df["ret1_rank"]
    if model_name == "shrink_focus":
        return 0.50 * df["shrink_rank"] + 0.30 * df["rebound_rank"] + 0.20 * df["trend_rank"]
    if model_name == "structure_volume_balance":
        return 0.35 * df["rebound_rank"] + 0.15 * df["brick_rank"] + 0.20 * df["trend_rank"] + 0.20 * df["signal_vs_ma5_rank"] + 0.10 * df["shrink_rank"]
    if model_name == "structure_volume_price":
        return 0.30 * df["rebound_rank"] + 0.15 * df["brick_rank"] + 0.15 * df["trend_rank"] + 0.20 * df["signal_vs_ma5_rank"] + 0.10 * df["shrink_rank"] + 0.10 * df["ret1_rank"]
    if model_name == "conservative_quality":
        return 0.25 * df["rebound_rank"] + 0.20 * df["trend_rank"] + 0.25 * df["signal_vs_ma5_rank"] + 0.15 * df["shrink_rank"] + 0.15 * df["ret1_rank"]
    if model_name == "aggressive_structure":
        return 0.60 * df["rebound_rank"] + 0.20 * df["brick_rank"] + 0.20 * df["trend_rank"]
    if model_name == "pattern_balanced":
        pattern_bonus = np.where(df["pattern_b"], 0.05, 0.0)
        return 0.32 * df["rebound_rank"] + 0.18 * df["brick_rank"] + 0.15 * df["trend_rank"] + 0.15 * df["signal_vs_ma5_rank"] + 0.10 * df["shrink_rank"] + 0.10 * df["ret1_rank"] + pattern_bonus
    raise ValueError(f"未知排序模型: {model_name}")


def build_trade_df(feature_map: Dict[str, pd.DataFrame], combo) -> pd.DataFrame:
    trades: List[dict] = []
    for code, raw_df in feature_map.items():
        df = add_long_line(raw_df)
        mask_a = df["pattern_a"] & (df["rebound_ratio"] >= combo.rebound_threshold)
        mask_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
        mask = (
            df["signal_base"]
            & (df["ret1"] <= combo.gain_limit)
            & (mask_a | mask_b)
            & (df["trend_line"] > df["long_line"])
        )
        signal_idxs = np.flatnonzero(mask.to_numpy())
        for signal_idx in signal_idxs:
            trade = base.simulate_trade(df, int(signal_idx), combo)
            if trade is None:
                continue
            trade["code"] = code
            trade.update(compute_signal_features(df, int(signal_idx)))
            trades.append(trade)
    if not trades:
        return pd.DataFrame()
    trade_df = pd.DataFrame(trades).sort_values(["signal_date", "code"]).reset_index(drop=True)
    return assign_rank_scores(trade_df)


def build_portfolio_for_model(trade_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["signal_date", "portfolio_ret", "equity"])
    x = trade_df.copy()
    x["sort_score"] = build_sort_score(x, model_name)
    x = x.sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    return base.build_portfolio_curve(x.groupby("signal_date", group_keys=False).head(TOP_N))


def summarize_model(trade_df: pd.DataFrame, model_name: str) -> dict:
    if trade_df.empty:
        return {
            "model_name": model_name,
            "sample_count": 0,
            "crowded_day_count": 0,
            "crowded_signal_count": 0,
            "crowded_day_avg_candidates": np.nan,
        }
    x = trade_df.copy()
    x["sort_score"] = build_sort_score(x, model_name)
    x = x.sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    selected = x.groupby("signal_date", group_keys=False).head(TOP_N).reset_index(drop=True)
    portfolio_df = base.build_portfolio_curve(selected)
    summary = base.summarize_combo(
        combo=combo,
        trade_df=selected,
        portfolio_df=portfolio_df,
    )
    daily_counts = x.groupby("signal_date").size()
    crowded = daily_counts[daily_counts > TOP_N]
    summary["model_name"] = model_name
    summary["crowded_day_count"] = int(len(crowded))
    summary["crowded_signal_count"] = int(crowded.sum()) if not crowded.empty else 0
    summary["crowded_day_avg_candidates"] = float(crowded.mean()) if not crowded.empty else np.nan
    return summary


combo = base.Combo(rebound_threshold=1.2, gain_limit=0.08, take_profit=0.03, stop_mode="entry_low_x_0.99")


def run_ranking_experiment() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_map = base.load_feature_map(base.DATA_DIR)
    trade_df = build_trade_df(feature_map, combo)
    model_names = [
        "baseline_rebound_heat",
        "rebound_only",
        "structure_trend",
        "structure_price",
        "structure_trend_price",
        "vol_ma5_focus",
        "shrink_focus",
        "structure_volume_balance",
        "structure_volume_price",
        "conservative_quality",
        "aggressive_structure",
        "pattern_balanced",
    ]
    rows = [summarize_model(trade_df, model_name) for model_name in model_names]
    result_df = pd.DataFrame(rows).sort_values(
        ["annual_return", "max_drawdown", "avg_trade_return", "success_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    summary = {
        "tested_combo": {
            "rebound_threshold": combo.rebound_threshold,
            "gain_limit": combo.gain_limit,
            "take_profit": combo.take_profit,
            "stop_mode": combo.stop_mode,
            "require_trend_above_long": True,
            "signal_vs_ma5_filter": "1.3~2.2",
            "top_n": TOP_N,
        },
        "ranking_models": model_names,
        "results": result_df.to_dict(orient="records"),
    }
    result_df.to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_ranking_experiment(), ensure_ascii=False, indent=2))
