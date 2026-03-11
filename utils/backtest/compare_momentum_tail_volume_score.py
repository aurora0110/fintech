from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_SCRIPT = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/run_momentum_tail_experiment.py")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/momentum_tail_volume_score_ab")


def load_base_module():
    spec = importlib.util.spec_from_file_location("momentum_tail_base", BASE_SCRIPT)
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


def volume_score(df: pd.DataFrame, signal_idx: int) -> float:
    score = 0.0
    today_vol = float(df.at[signal_idx, "volume"])
    prev_vol = float(df.at[signal_idx - 1, "volume"]) if signal_idx >= 1 else np.nan
    up_leg_avg_vol = float(df.at[signal_idx, "up_leg_avg_vol"])
    pullback_avg_vol = float(df.at[signal_idx, "pullback_avg_vol"])
    vol_ma5 = float(df["volume"].shift(1).rolling(5).mean().iloc[signal_idx])

    if np.isfinite(up_leg_avg_vol) and np.isfinite(pullback_avg_vol) and up_leg_avg_vol > 0 and pullback_avg_vol > 0:
        shrink_ratio = pullback_avg_vol / up_leg_avg_vol
        if shrink_ratio <= 0.55:
            score += 1.0
        elif shrink_ratio <= 0.75:
            score += 0.7
        elif shrink_ratio <= 0.95:
            score += 0.3
        else:
            score -= 0.3

    if np.isfinite(prev_vol) and prev_vol > 0:
        day_over_prev = today_vol / prev_vol
        if 1.1 <= day_over_prev <= 1.8:
            score += 1.0
        elif 0.9 <= day_over_prev < 1.1:
            score += 0.3
        elif 1.8 < day_over_prev <= 2.5:
            score += 0.2
        elif day_over_prev > 2.5:
            score -= 0.5
        else:
            score -= 0.2

    if np.isfinite(vol_ma5) and vol_ma5 > 0:
        day_over_ma5 = today_vol / vol_ma5
        if 1.0 <= day_over_ma5 <= 1.8:
            score += 0.8
        elif 0.8 <= day_over_ma5 < 1.0:
            score += 0.2
        elif day_over_ma5 > 2.2:
            score -= 0.4

    return score


def build_trade_df(base, feature_map: Dict[str, pd.DataFrame], combo, require_trend_above_long: bool, use_volume_score: bool) -> pd.DataFrame:
    trades: List[dict] = []
    for code, raw_df in feature_map.items():
        df = add_long_line(raw_df)
        mask_a = df["pattern_a"] & (df["rebound_ratio"] >= combo.rebound_threshold)
        mask_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
        mask = df["signal_base"] & (df["ret1"] <= combo.gain_limit) & (mask_a | mask_b)
        if require_trend_above_long:
            mask = mask & (df["trend_line"] > df["long_line"])
        signal_idxs = np.flatnonzero(mask.to_numpy())
        for signal_idx in signal_idxs:
            trade = base.simulate_trade(df, int(signal_idx), combo)
            if trade is None:
                continue
            base_score = float(df.at[int(signal_idx), "rebound_ratio"] / max(abs(float(df.at[int(signal_idx), "ret1"])), 0.01))
            v_score = volume_score(df, int(signal_idx)) if use_volume_score else 0.0
            trade["code"] = code
            trade["base_sort_score"] = base_score
            trade["volume_score"] = v_score
            trade["sort_score"] = max(base_score + v_score, 0.01)
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
        {"name": "baseline_no_trend_filter", "require_trend_above_long": False, "use_volume_score": False},
        {"name": "volume_score_no_trend_filter", "require_trend_above_long": False, "use_volume_score": True},
        {"name": "baseline_with_trend_filter", "require_trend_above_long": True, "use_volume_score": False},
        {"name": "volume_score_with_trend_filter", "require_trend_above_long": True, "use_volume_score": True},
    ]

    rows = []
    for scenario in scenarios:
        trade_df = build_trade_df(
            base=base,
            feature_map=feature_map,
            combo=combo,
            require_trend_above_long=scenario["require_trend_above_long"],
            use_volume_score=scenario["use_volume_score"],
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
        },
        "volume_score_design": {
            "pullback_shrink": "回撤3日均量 / 上涨段3日均量，越低越好，极度缩量最高加1分",
            "signal_vs_prev_day": "信号日量 / 前一日量，温和放量(1.1~1.8倍)最高加1分，爆量>2.5倍扣0.5分",
            "signal_vs_ma5": "信号日量 / 过去5日均量，温和放量(1.0~1.8倍)加0.8分，过热放量>2.2倍扣0.4分",
        },
        "results": result_df.to_dict(orient="records"),
    }
    result_df.to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_ab_test(), ensure_ascii=False, indent=2))
