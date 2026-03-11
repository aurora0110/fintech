from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_SCRIPT = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/run_momentum_tail_experiment.py")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/momentum_tail_volume_filters_ab")


def load_base_module():
    spec = importlib.util.spec_from_file_location("momentum_tail_base_filters", BASE_SCRIPT)
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
    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    return x


def build_trade_df(base, feature_map: Dict[str, pd.DataFrame], combo, scenario_name: str) -> pd.DataFrame:
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

        pullback_shrink_ratio = df["pullback_avg_vol"] / df["up_leg_avg_vol"]
        signal_vs_ma5 = df["volume"] / df["vol_ma5_prev"]

        if scenario_name == "filter_signal_vs_ma5":
            mask = mask & signal_vs_ma5.between(1.3, 2.2, inclusive="both")
        elif scenario_name == "filter_pullback_shrink":
            mask = mask & pullback_shrink_ratio.between(0.7, 1.1, inclusive="both")
        elif scenario_name == "filter_both":
            mask = mask & signal_vs_ma5.between(1.3, 2.2, inclusive="both") & pullback_shrink_ratio.between(0.7, 1.1, inclusive="both")

        for signal_idx in np.flatnonzero(mask.to_numpy()):
            trade = base.simulate_trade(df, int(signal_idx), combo)
            if trade is None:
                continue
            trade["code"] = code
            trade["sort_score"] = float(df.at[int(signal_idx), "rebound_ratio"] / max(abs(float(df.at[int(signal_idx), "ret1"])), 0.01))
            trades.append(trade)
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades).sort_values(["signal_date", "code"]).reset_index(drop=True)


def run_ab_test() -> dict:
    base = load_base_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_map = base.load_feature_map(base.DATA_DIR)
    combo = base.Combo(rebound_threshold=1.5, gain_limit=0.07, take_profit=0.045, stop_mode="entry_low")

    scenarios = [
        "baseline",
        "filter_signal_vs_ma5",
        "filter_pullback_shrink",
        "filter_both",
    ]

    rows = []
    for scenario_name in scenarios:
        trade_df = build_trade_df(base, feature_map, combo, scenario_name)
        portfolio_df = base.build_portfolio_curve(trade_df)
        summary = base.summarize_combo(combo, trade_df, portfolio_df)
        summary["scenario_name"] = scenario_name
        rows.append(summary)

    result_df = pd.DataFrame(rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    summary = {
        "tested_combo": {
            "rebound_threshold": combo.rebound_threshold,
            "gain_limit": combo.gain_limit,
            "take_profit": combo.take_profit,
            "stop_mode": combo.stop_mode,
            "require_trend_above_long": True,
        },
        "volume_filter_design": {
            "filter_signal_vs_ma5": "只保留信号日量 / 前5日均量在 1.3~2.2 的样本",
            "filter_pullback_shrink": "只保留回撤缩量比在 0.7~1.1 的样本",
            "filter_both": "同时满足上面两条",
        },
        "results": result_df.to_dict(orient="records"),
    }
    result_df.to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_ab_test(), ensure_ascii=False, indent=2))
