from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_SCRIPT = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/run_momentum_tail_experiment.py")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/momentum_tail_board_gain_limits")
TOP_N = 10


def load_base_module():
    spec = importlib.util.spec_from_file_location("momentum_tail_base_board_gain", BASE_SCRIPT)
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
    return float(max(1.0 - abs(value - center) / half_width, 0.0))


def classify_board(code: str) -> str:
    pure = str(code).split("#")[-1]
    if pure.startswith("688"):
        return "star"
    if pure.startswith("300") or pure.startswith("301"):
        return "gem"
    return "main"


def board_gain_ok(ret1: float, board: str, limit_main: float, limit_gem: float, limit_star: float) -> bool:
    if not np.isfinite(ret1):
        return False
    if board == "gem":
        return ret1 <= limit_gem
    if board == "star":
        return ret1 <= limit_star
    return ret1 <= limit_main


def build_trade_df(feature_map: Dict[str, pd.DataFrame], combo, gain_scheme: dict) -> pd.DataFrame:
    trades: List[dict] = []
    for code, raw_df in feature_map.items():
        df = add_long_line(raw_df)
        board = classify_board(code)
        mask_a = df["pattern_a"] & (df["rebound_ratio"] >= combo.rebound_threshold)
        mask_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
        gain_mask = df["ret1"].apply(
            lambda x: board_gain_ok(
                float(x),
                board,
                gain_scheme["main"],
                gain_scheme["gem"],
                gain_scheme["star"],
            )
        )
        mask = (
            df["signal_base"]
            & gain_mask
            & (mask_a | mask_b)
            & (df["trend_line"] > df["long_line"])
        )
        for signal_idx in np.flatnonzero(mask.to_numpy()):
            trade = base.simulate_trade(df, int(signal_idx), combo)
            if trade is None:
                continue
            close = float(df.at[int(signal_idx), "close"])
            trend_line = float(df.at[int(signal_idx), "trend_line"])
            long_line = float(df.at[int(signal_idx), "long_line"])
            rebound_ratio = float(df.at[int(signal_idx), "rebound_ratio"])
            brick_red_len = float(df.at[int(signal_idx), "brick_red_len"])
            ret1 = float(df.at[int(signal_idx), "ret1"])
            signal_vs_ma5 = float(df.at[int(signal_idx), "signal_vs_ma5"])
            pullback_avg_vol = float(df.at[int(signal_idx), "pullback_avg_vol"])
            up_leg_avg_vol = float(df.at[int(signal_idx), "up_leg_avg_vol"])
            pullback_shrink_ratio = pullback_avg_vol / up_leg_avg_vol if np.isfinite(up_leg_avg_vol) and up_leg_avg_vol > 0 else np.nan
            trade["code"] = code
            trade["board"] = board
            trade["rebound_ratio"] = rebound_ratio
            trade["brick_red_len"] = brick_red_len
            trade["ret1_quality"] = triangle_quality(ret1, 0.03, 0.03)
            trade["signal_vs_ma5_quality"] = triangle_quality(signal_vs_ma5, 1.7, 0.5)
            trade["shrink_quality"] = triangle_quality(pullback_shrink_ratio, 0.8, 0.3)
            trade["trend_spread_clip"] = max((trend_line - long_line) / close, 0.0) if np.isfinite(close) and close > 0 else 0.0
            trades.append(trade)

    if not trades:
        return pd.DataFrame()
    x = pd.DataFrame(trades).sort_values(["signal_date", "code"]).reset_index(drop=True)
    x["rebound_rank"] = x.groupby("signal_date")["rebound_ratio"].rank(pct=True)
    x["trend_rank"] = x.groupby("signal_date")["trend_spread_clip"].rank(pct=True)
    x["shrink_rank"] = x.groupby("signal_date")["shrink_quality"].rank(pct=True)
    x["sort_score"] = 0.50 * x["shrink_rank"] + 0.30 * x["rebound_rank"] + 0.20 * x["trend_rank"]
    x["score_pct_rank"] = x.groupby("signal_date")["sort_score"].rank(pct=True)
    x = x[x["score_pct_rank"] >= 0.50].copy()
    x = x.sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True]).groupby("signal_date", group_keys=False).head(TOP_N).reset_index(drop=True)
    return x


def summarize_scheme(trade_df: pd.DataFrame, scheme_name: str, gain_scheme: dict, combo) -> dict:
    portfolio_df = base.build_portfolio_curve(trade_df)
    summary = base.summarize_combo(combo=combo, trade_df=trade_df, portfolio_df=portfolio_df)
    summary["scheme_name"] = scheme_name
    summary["gain_main"] = gain_scheme["main"]
    summary["gain_gem"] = gain_scheme["gem"]
    summary["gain_star"] = gain_scheme["star"]
    if trade_df.empty:
        summary["main_share"] = np.nan
        summary["gem_share"] = np.nan
        summary["star_share"] = np.nan
    else:
        counts = trade_df["board"].value_counts(normalize=True)
        summary["main_share"] = float(counts.get("main", 0.0))
        summary["gem_share"] = float(counts.get("gem", 0.0))
        summary["star_share"] = float(counts.get("star", 0.0))
    return summary


combo = base.Combo(rebound_threshold=1.2, gain_limit=0.08, take_profit=0.03, stop_mode="entry_low_x_0.99")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_map = base.load_feature_map(base.DATA_DIR)

    schemes = [{"name": "uniform_0.08", "main": 0.08, "gem": 0.08, "star": 0.08}]
    for main_limit in [0.05, 0.06, 0.07]:
        for gem_limit in [0.07, 0.08, 0.09]:
            for star_limit in [0.07, 0.08, 0.09]:
                schemes.append(
                    {
                        "name": f"main_{main_limit:.2f}__gem_{gem_limit:.2f}__star_{star_limit:.2f}",
                        "main": main_limit,
                        "gem": gem_limit,
                        "star": star_limit,
                    }
                )

    rows = []
    print(f"组合数: {len(schemes)}")
    for idx, scheme in enumerate(schemes, 1):
        trade_df = build_trade_df(feature_map, combo, scheme)
        rows.append(summarize_scheme(trade_df, scheme["name"], scheme, combo))
        print(f"组合进度: {idx}/{len(schemes)}")

    result_df = pd.DataFrame(rows).sort_values(
        ["annual_return", "max_drawdown", "avg_trade_return", "success_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    result_df.to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    summary = {
        "tested_combo": {
            "rebound_threshold": combo.rebound_threshold,
            "uniform_gain_limit_baseline": combo.gain_limit,
            "take_profit": combo.take_profit,
            "stop_mode": combo.stop_mode,
            "ranking_model": "shrink_focus",
            "rank_cutoff": "pct_rank_ge_0.50",
            "require_trend_above_long": True,
            "signal_vs_ma5_filter": "1.3~2.2",
            "top_n": TOP_N,
        },
        "results": result_df.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
