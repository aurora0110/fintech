from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_SCRIPT = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/run_momentum_tail_experiment.py")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/momentum_tail_refined_volume_score_ab")


def load_base_module():
    spec = importlib.util.spec_from_file_location("momentum_tail_base_refined", BASE_SCRIPT)
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


def refined_volume_score(df: pd.DataFrame, signal_idx: int) -> float:
    score = 0.0
    signal_vol = float(df.at[signal_idx, "volume"])
    vol_ma5_prev = float(df.at[signal_idx, "vol_ma5_prev"])
    up_leg_avg_vol = float(df.at[signal_idx, "up_leg_avg_vol"])
    pullback_avg_vol = float(df.at[signal_idx, "pullback_avg_vol"])

    if np.isfinite(up_leg_avg_vol) and up_leg_avg_vol > 0 and np.isfinite(pullback_avg_vol) and pullback_avg_vol > 0:
        shrink_ratio = pullback_avg_vol / up_leg_avg_vol
        if 0.7 < shrink_ratio <= 0.9:
            score += 1.4
        elif 0.9 < shrink_ratio <= 1.1:
            score += 0.8
        elif 0.5 < shrink_ratio <= 0.7:
            score += 0.3
        elif shrink_ratio <= 0.5:
            score -= 0.5
        else:
            score -= 0.2

    if np.isfinite(vol_ma5_prev) and vol_ma5_prev > 0:
        signal_vs_ma5 = signal_vol / vol_ma5_prev
        if 1.8 < signal_vs_ma5 <= 2.2:
            score += 1.2
        elif 1.3 < signal_vs_ma5 <= 1.8:
            score += 1.0
        elif 1.0 <= signal_vs_ma5 <= 1.3:
            score += 0.2
        elif signal_vs_ma5 <= 0.8:
            score -= 0.3
        elif signal_vs_ma5 > 2.2:
            score -= 0.1

    return score


def build_trade_df(base, feature_map: Dict[str, pd.DataFrame], combo, use_refined_volume_score: bool) -> pd.DataFrame:
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
        for signal_idx in np.flatnonzero(mask.to_numpy()):
            trade = base.simulate_trade(df, int(signal_idx), combo)
            if trade is None:
                continue
            base_score = float(df.at[int(signal_idx), "rebound_ratio"] / max(abs(float(df.at[int(signal_idx), "ret1"])), 0.01))
            vol_score = refined_volume_score(df, int(signal_idx)) if use_refined_volume_score else 0.0
            trade["code"] = code
            trade["base_sort_score"] = base_score
            trade["refined_volume_score"] = vol_score
            trade["sort_score"] = max(base_score + vol_score, 0.01)
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
        {"name": "baseline_with_trend_filter", "use_refined_volume_score": False},
        {"name": "refined_volume_score_with_trend_filter", "use_refined_volume_score": True},
    ]

    rows = []
    for scenario in scenarios:
        trade_df = build_trade_df(
            base=base,
            feature_map=feature_map,
            combo=combo,
            use_refined_volume_score=scenario["use_refined_volume_score"],
        )
        portfolio_df = base.build_portfolio_curve(trade_df)
        summary = base.summarize_combo(combo, trade_df, portfolio_df)
        summary.update(scenario)
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
        "refined_volume_score_design": {
            "pullback_shrink": "回撤缩量比 0.7~0.9 最高加分，0.9~1.1 次之，<=0.5 反而扣分",
            "signal_vs_ma5": "信号日量 / 前5日均量在 1.3~2.2 最高加分，过低或过热都降分",
        },
        "results": result_df.to_dict(orient="records"),
    }
    result_df.to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_ab_test(), ensure_ascii=False, indent=2))
