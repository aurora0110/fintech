from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils.multi_factor_research.data_processor import load_stock_directory
from utils.multi_factor_research.exit_models import (
    FixedDaysExit,
    TieredExit,
    exit_config_label,
    simulate_signal_exits,
)
from utils.multi_factor_research.factor_calculator import build_prepared_stock_data
from utils.multi_factor_research.research_metrics import summarize_trade_metrics


def _rolling_rank_pct(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).rank(method="average", pct=True)


def _calc_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    vol_lt_1 = out["volume"].lt(out["volume"].shift(1))
    vol_lt_2 = vol_lt_1 & out["volume"].lt(out["volume"].shift(2))
    vol_lt_3 = vol_lt_2 & out["volume"].lt(out["volume"].shift(3))
    out["quality_shrink_volume"] = (
        0.3 * vol_lt_1.astype(float)
        + 0.3 * vol_lt_2.astype(float)
        + 1.0 * vol_lt_3.astype(float)
    )

    ret = out["close"] / out["close"].shift(1) - 1.0
    main_ok = out["board"].eq("MAIN") & ret.gt(-0.035) & ret.lt(0.02)
    growth_ok = out["board"].isin(["GEM", "STAR"]) & ret.gt(-0.05) & ret.lt(0.03)
    out["quality_price_amplitude"] = (main_ok | growth_ok).astype(float)

    burst_event = (out["close"] > out["open"]) & out["volume"].gt(out["volume"].shift(1) * 2.0)
    burst_count = burst_event.rolling(window=60, min_periods=1).sum()
    out["quality_staged_volume_burst"] = np.select(
        [burst_count.eq(1), burst_count.eq(2), burst_count.ge(3)],
        [0.3, 0.6, 1.0],
        default=0.0,
    )

    out["quality_long_bear_short_volume"] = out["long_bear_short_volume_factor"].fillna(0.0).gt(0.0).astype(float)
    out["quality_total"] = (
        out["quality_shrink_volume"]
        + out["quality_price_amplitude"]
        + out["quality_staged_volume_burst"]
        + out["quality_long_bear_short_volume"]
    )
    return out


def _calc_risk_penalties(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close_max_10 = out["close"].rolling(window=10, min_periods=10).max()
    close_min_10 = out["close"].rolling(window=10, min_periods=10).min()
    sideways_10 = ((close_max_10 - close_min_10) / close_min_10.replace(0, np.nan)).lt(0.03)
    out["risk_sideways_10"] = sideways_10.astype(float) * 1.0

    bearish = out["close"] < out["open"]
    volume = out["volume"]
    penalty_bear_vol = np.select(
        [
            bearish & volume.eq(volume.rolling(60, min_periods=60).max()),
            bearish & volume.eq(volume.rolling(30, min_periods=30).max()),
            bearish & volume.eq(volume.rolling(15, min_periods=15).max()),
            bearish & volume.eq(volume.rolling(10, min_periods=10).max()),
        ],
        [4.0, 3.0, 2.0, 1.0],
        default=0.0,
    )
    out["risk_bearish_volume"] = penalty_bear_vol

    low_30 = out["low"].rolling(window=30, min_periods=30).min()
    high_30 = out["high"].rolling(window=30, min_periods=30).max()
    rise_30 = high_30 / low_30.replace(0, np.nan) - 1.0
    out["risk_rise_30_over_50"] = rise_30.gt(0.5).astype(float) * 1.0

    out["risk_break_key_close"] = out["key_k_close_break_penalty"].fillna(0.0).gt(0.0).astype(float) * 1.0
    out["risk_flat_trend_chop"] = out["flat_trend_slope_penalty"].fillna(0.0).ge(0.5).astype(float) * 2.0
    out["risk_line_entanglement"] = out["line_entanglement_penalty"].fillna(0.0).ge(0.5).astype(float) * 2.0
    out["risk_sideways_no_confirm"] = out["sideways_without_confirmation_penalty"].fillna(0.0).ge(0.5).astype(float) * 1.0
    out["risk_amp_no_progress"] = out["amplitude_without_progress_penalty"].fillna(0.0).ge(0.5).astype(float) * 1.0

    out["risk_total"] = (
        out["risk_sideways_10"]
        + out["risk_bearish_volume"]
        + out["risk_rise_30_over_50"]
        + out["risk_break_key_close"]
        + out["risk_flat_trend_chop"]
        + out["risk_line_entanglement"]
        + out["risk_sideways_no_confirm"]
        + out["risk_amp_no_progress"]
    )
    return out


def _apply_four_layer_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["J_rank_30"] = _rolling_rank_pct(out["J"], 30)
    out["hard_b1_dynamic_j"] = out["J_rank_30"].le(0.1) & out["J"].gt(out["J"].shift(1))
    out["hard_max_bear_60"] = out["bearish_max_volume_60_penalty"].fillna(0.0).lt(0.5)
    out["hard_double_break_bull_bear"] = out["double_break_bull_bear_penalty"].fillna(0.0).lt(0.5)
    low_30 = out["low"].rolling(window=30, min_periods=30).min()
    high_30 = out["high"].rolling(window=30, min_periods=30).max()
    out["hard_not_extreme_hot"] = (high_30 / low_30.replace(0, np.nan) - 1.0).lt(1.0).fillna(False)
    out["hard_trend_above_bull_bear"] = out["trend_line"].gt(out["bull_bear_line"])

    out["hard_pass"] = (
        out["hard_max_bear_60"]
        & out["hard_double_break_bull_bear"]
        & out["hard_not_extreme_hot"]
        & out["hard_trend_above_bull_bear"]
        & out["hard_b1_dynamic_j"]
    )

    out["structure_pullback"] = out["pullback_confirmation_factor"].fillna(0.0).gt(0.0).astype(float)
    out["structure_key_k_support"] = out["key_k_support_factor"].fillna(0.0).gt(0.0).astype(float)
    out["structure_daily_ma_bull"] = out["daily_ma_bull_factor"].fillna(0.0).ge(0.5).astype(float)
    out["structure_hits"] = (
        out["structure_pullback"] + out["structure_key_k_support"] + out["structure_daily_ma_bull"]
    )

    out = _calc_quality_scores(out)
    out = _calc_risk_penalties(out)
    out["score_total"] = out["structure_hits"] + out["quality_total"] - out["risk_total"]
    out["candidate_signal"] = out["hard_pass"] & out["structure_hits"].ge(1.0)
    return out


def build_scorecard_signals(prepared_stock_data: Dict[str, pd.DataFrame]) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    updated: Dict[str, pd.DataFrame] = {}
    records: List[dict] = []

    for code, raw_df in prepared_stock_data.items():
        df = _apply_four_layer_scorecard(raw_df)
        updated[code] = df
        signal_idx = np.flatnonzero(df["candidate_signal"].to_numpy(dtype=bool))
        signal_idx = signal_idx[signal_idx + 1 < len(df)]
        for idx in signal_idx:
            records.append(
                {
                    "code": code,
                    "signal_idx": int(idx),
                    "signal_date": df.at[idx, "date"],
                    "board": df.at[idx, "board"],
                    "j_value": float(df.at[idx, "J"]),
                    "trend_line": float(df.at[idx, "trend_line"]),
                    "bull_bear_line": float(df.at[idx, "bull_bear_line"]),
                    "structure_hits": float(df.at[idx, "structure_hits"]),
                    "quality_total": float(df.at[idx, "quality_total"]),
                    "risk_total": float(df.at[idx, "risk_total"]),
                    "net_factor_score": float(df.at[idx, "score_total"]),
                    "structure_pullback": float(df.at[idx, "structure_pullback"]),
                    "structure_key_k_support": float(df.at[idx, "structure_key_k_support"]),
                    "structure_daily_ma_bull": float(df.at[idx, "structure_daily_ma_bull"]),
                    "quality_shrink_volume": float(df.at[idx, "quality_shrink_volume"]),
                    "quality_price_amplitude": float(df.at[idx, "quality_price_amplitude"]),
                    "quality_staged_volume_burst": float(df.at[idx, "quality_staged_volume_burst"]),
                    "quality_long_bear_short_volume": float(df.at[idx, "quality_long_bear_short_volume"]),
                    "risk_sideways_10": float(df.at[idx, "risk_sideways_10"]),
                    "risk_bearish_volume": float(df.at[idx, "risk_bearish_volume"]),
                    "risk_rise_30_over_50": float(df.at[idx, "risk_rise_30_over_50"]),
                    "risk_break_key_close": float(df.at[idx, "risk_break_key_close"]),
                    "risk_flat_trend_chop": float(df.at[idx, "risk_flat_trend_chop"]),
                    "risk_line_entanglement": float(df.at[idx, "risk_line_entanglement"]),
                    "risk_sideways_no_confirm": float(df.at[idx, "risk_sideways_no_confirm"]),
                    "risk_amp_no_progress": float(df.at[idx, "risk_amp_no_progress"]),
                }
            )

    if not records:
        return updated, pd.DataFrame()
    signals = pd.DataFrame(records).sort_values(["signal_date", "net_factor_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    return updated, signals


def summarize_score_groups(dataset: pd.DataFrame, top_quantile: float) -> dict:
    if dataset.empty:
        return {"samples": 0, "top_group_samples": 0, "baseline": {}, "top_group": {}}
    cutoff = dataset["net_factor_score"].quantile(max(0.0, min(1.0, 1.0 - top_quantile)))
    top_group = dataset[dataset["net_factor_score"] >= cutoff].copy()
    return {
        "samples": int(len(dataset)),
        "top_group_samples": int(len(top_group)),
        "baseline": summarize_trade_metrics(dataset),
        "top_group": summarize_trade_metrics(top_group),
        "score_cutoff": float(cutoff),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-layer scorecard backtest")
    parser.add_argument("data_dir", nargs="?", default="/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
    parser.add_argument("--burst-window", type=int, default=20)
    parser.add_argument("--top-quantile", type=float, default=0.30)
    parser.add_argument("--output-dir", default="results/four_layer_scorecard_backtest")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_data = load_stock_directory(args.data_dir)
    prepared_stock_data = build_prepared_stock_data(stock_data, burst_window=args.burst_window)
    scored_stock_data, signal_candidates = build_scorecard_signals(prepared_stock_data)

    signal_candidates.to_csv(output_dir / "signal_candidates.csv", index=False, encoding="utf-8-sig")

    exit_configs: list[object] = [
        FixedDaysExit(10),
        FixedDaysExit(20),
        FixedDaysExit(30),
        TieredExit(60),
    ]

    summaries = []
    for exit_config in exit_configs:
        label = exit_config_label(exit_config)
        dataset = simulate_signal_exits(
            prepared_stock_data=scored_stock_data,
            signal_candidates=signal_candidates,
            exit_config=exit_config,
            success_return_threshold=0.0,
        )
        score_summary = summarize_score_groups(dataset, args.top_quantile)
        run_dir = output_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(run_dir / "trades.csv", index=False, encoding="utf-8-sig")
        summary = {
            "exit_model": label,
            "signal_count": int(len(dataset)),
            "score_summary": score_summary,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(summary)

    final_summary = {
        "signal_candidate_count": int(len(signal_candidates)),
        "assumptions": {
            "hard_filter_requires": [
                "60日最大量不是阴线",
                "最近两天不连续收盘跌破多空线",
                "30日最高/最低涨幅不超过100%",
                "趋势线高于多空线",
                "J_rank_30<=0.1且J上拐",
            ],
            "candidate_gate": "硬过滤通过且结构确认>=1",
            "score_formula": "structure_hits + quality_total - risk_total",
            "top_quantile": args.top_quantile,
        },
        "runs": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
