from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tmp.run_b2_type14_exit_and_param_opt import (  # type: ignore
    DATA_DIR,
    EXIT_RULES,
    ExitRule,
    EXCLUDE_END,
    EXCLUDE_START,
    build_trade_table,
    load_all_data,
    simulate_portfolio,
    summarize_trades,
)

RESULT_DIR = ROOT / "results/b2_type14_param_search_cached_20260313"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TYPE1_EXIT = "hold30_close"
TYPE4_EXIT = "hold20_close"


def score_volume_quality(signal_vs_ma5: pd.Series) -> pd.Series:
    return (1.0 - np.minimum(np.abs(signal_vs_ma5.fillna(0.0) - 1.9) / 0.6, 1.0)).clip(lower=0.0)


def valid_date_mask(dates: pd.Series) -> pd.Series:
    return ~((dates >= EXCLUDE_START) & (dates <= EXCLUDE_END))


def build_type4_touch_flags(x: pd.DataFrame, touch_ratio: float, lookback: int = 20) -> pd.Series:
    result = pd.Series(False, index=x.index)
    bull_cross = (
        (x["trend_line"] > x["long_line"])
        & (x["trend_line"].shift(1) <= x["long_line"].shift(1))
    ).fillna(False).to_numpy(dtype=bool)
    lows = x["low"].to_numpy(dtype=float)
    closes = x["close"].to_numpy(dtype=float)
    trends = x["trend_line"].to_numpy(dtype=float)
    n = len(x)
    for i in range(10, n):
        left = max(1, i - lookback)
        cross_idx = None
        for j in range(i - 1, left - 1, -1):
            if bull_cross[j]:
                cross_idx = j
                break
        if cross_idx is None:
            continue
        prev_touch = (lows[i - 1] <= trends[i - 1] * touch_ratio) or (closes[i - 1] <= trends[i - 1] * touch_ratio)
        if not prev_touch:
            continue
        had_touch = False
        for k in range(cross_idx + 1, i - 1):
            if (lows[k] <= trends[k] * touch_ratio) or (closes[k] <= trends[k] * touch_ratio):
                had_touch = True
                break
        if not had_touch:
            result.iat[i] = True
    return result


def build_candidates(dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    type1_rows: list[dict[str, object]] = []
    type4_rows: list[dict[str, object]] = []
    total = len(dfs)
    for idx, (code, x) in enumerate(dfs.items(), start=1):
        valid = valid_date_mask(x["date"])
        base_invariant = (
            x["trend_ok"]
            & (x["close"] > x["open"])
            & (x["volume"] > x["volume"].shift(1))
            & (x["volume"] > x["vol_ma5"])
            & (x["J"] > x["J"].shift(1))
            & (x["J"].shift(1) < x["J"].shift(2))
            & (x["J"].shift(2) < x["J"].shift(3))
            & valid
        )
        upper_body_ratio = (
            x["upper_shadow"] / x["body"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        touch_100 = build_type4_touch_flags(x, 1.00)
        touch_101 = build_type4_touch_flags(x, 1.01)
        touch_102 = build_type4_touch_flags(x, 1.02)

        for i in np.where(base_invariant.to_numpy(dtype=bool))[0]:
            entry_idx = i + 1
            if entry_idx >= len(x):
                continue
            common = {
                "code": code,
                "signal_idx": int(i),
                "signal_date": x.at[i, "date"],
                "entry_idx": int(entry_idx),
                "entry_date": x.at[entry_idx, "date"],
                "entry_open": float(x.at[entry_idx, "open"]),
                "signal_low": float(x.at[i, "low"]),
                "ret1": float(x.at[i, "ret1"]),
                "j_value": float(x.at[i, "J"]),
                "j_rank20_prev": float(x.at[i, "j_rank20_prev"]) if pd.notna(x.at[i, "j_rank20_prev"]) else np.nan,
                "near_long_prev_ratio": float(x.at[i - 1, "close"] / x.at[i - 1, "long_line"]) if i >= 1 and pd.notna(x.at[i - 1, "long_line"]) and x.at[i - 1, "long_line"] != 0 else np.nan,
                "upper_body_ratio": float(upper_body_ratio.iat[i]),
                "close_position": float(x.at[i, "close_position"]) if pd.notna(x.at[i, "close_position"]) else 0.0,
                "signal_vs_ma5": float(x.at[i, "signal_vs_ma5"]) if pd.notna(x.at[i, "signal_vs_ma5"]) else np.nan,
                "trend_line_lead": float(x.at[i, "trend_line_lead"]) if pd.notna(x.at[i, "trend_line_lead"]) else 0.0,
                "trend_slope_5": float(x.at[i, "trend_slope_5"]) if pd.notna(x.at[i, "trend_slope_5"]) else 0.0,
                "type4_touch_100": bool(touch_100.iat[i]),
                "type4_touch_101": bool(touch_101.iat[i]),
                "type4_touch_102": bool(touch_102.iat[i]),
            }
            type1_rows.append(common)
            type4_rows.append(common)
        if idx % 500 == 0:
            print(f"候选构建进度: {idx}/{total}")

    return pd.DataFrame(type1_rows), pd.DataFrame(type4_rows)


def select_type1(candidates: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    mask = (
        (candidates["ret1"] >= params["ret1_min"])
        & (candidates["j_value"] < params["j_max"])
        & (candidates["upper_body_ratio"] <= params["upper_shadow_body_ratio"] + 1e-12)
        & (candidates["near_long_prev_ratio"] <= params["type1_near_ratio"] + 1e-12)
        & (candidates["j_rank20_prev"] <= params["type1_j_rank20_max"] + 1e-12)
    )
    out = candidates.loc[mask].copy()
    if out.empty:
        return out
    out["sort_score"] = (
        0.35 * out["close_position"].clip(0.0, 1.0)
        + 0.20 * score_volume_quality(out["signal_vs_ma5"])
        + 0.20 * out["trend_line_lead"].clip(lower=0.0)
        + 0.10 * (1.0 - np.minimum(out["j_value"] / params["j_max"], 1.0)).clip(lower=0.0)
        + 0.10 * np.maximum(out["trend_slope_5"], 0.0)
        + 0.05
    )
    out["type1"] = True
    out["type4"] = False
    return out[
        ["code", "signal_idx", "signal_date", "entry_idx", "entry_date", "entry_open", "signal_low", "sort_score", "type1", "type4"]
    ].reset_index(drop=True)


def select_type4(candidates: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    touch_col = {
        1.00: "type4_touch_100",
        1.01: "type4_touch_101",
        1.02: "type4_touch_102",
    }[round(params["type4_touch_ratio"], 2)]
    mask = (
        (candidates["ret1"] >= params["ret1_min"])
        & (candidates["j_value"] < params["j_max"])
        & (candidates["upper_body_ratio"] <= params["upper_shadow_body_ratio"] + 1e-12)
        & candidates[touch_col]
    )
    out = candidates.loc[mask].copy()
    if out.empty:
        return out
    out["sort_score"] = (
        0.35 * out["close_position"].clip(0.0, 1.0)
        + 0.20 * score_volume_quality(out["signal_vs_ma5"])
        + 0.20 * out["trend_line_lead"].clip(lower=0.0)
        + 0.10 * (1.0 - np.minimum(out["j_value"] / params["j_max"], 1.0)).clip(lower=0.0)
        + 0.10 * np.maximum(out["trend_slope_5"], 0.0)
        + 0.05
    )
    out["type1"] = False
    out["type4"] = True
    return out[
        ["code", "signal_idx", "signal_date", "entry_idx", "entry_date", "entry_open", "signal_low", "sort_score", "type1", "type4"]
    ].reset_index(drop=True)


def summarize_combo(trades: pd.DataFrame) -> dict[str, float]:
    s = summarize_trades(trades)
    p = simulate_portfolio(trades)
    return {**s, **p}


def pick_best(df: pd.DataFrame, min_count: int) -> pd.Series:
    cand = df[df["sample_count"] >= min_count].copy()
    if cand.empty:
        cand = df.copy()
    cand["score"] = (
        0.45 * cand["annual_return"].fillna(-9.0)
        + 0.25 * cand["success_rate"].fillna(0.0)
        + 0.20 * cand["avg_return"].fillna(-9.0)
        - 0.10 * cand["max_drawdown"].abs().fillna(1.0)
    )
    cand = cand.sort_values(
        by=["score", "avg_return", "success_rate", "sample_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return cand.iloc[0]


def get_rule(name: str) -> ExitRule:
    return next(rule for rule in EXIT_RULES if rule.name == name)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = load_all_data()
    type1_candidates, type4_candidates = build_candidates(dfs)
    type1_candidates.to_csv(RESULT_DIR / "type1_candidates.csv", index=False)
    type4_candidates.to_csv(RESULT_DIR / "type4_candidates.csv", index=False)

    type1_rule = get_rule(TYPE1_EXIT)
    type4_rule = get_rule(TYPE4_EXIT)

    type1_rows: list[dict[str, object]] = []
    total1 = 2 * 2 * 2 * 3 * 3
    idx = 0
    for ret1_min in [0.03, 0.04]:
        for upper_ratio in [0.3, 0.5]:
            for j_max in [90.0, 100.0]:
                for near_ratio in [1.01, 1.02, 1.03]:
                    for jrank in [0.05, 0.10, 0.15]:
                        idx += 1
                        params = {
                            "ret1_min": ret1_min,
                            "upper_shadow_body_ratio": upper_ratio,
                            "j_max": j_max,
                            "type1_near_ratio": near_ratio,
                            "type1_j_rank20_max": jrank,
                        }
                        signals = select_type1(type1_candidates, params)
                        trades = build_trade_table(signals, dfs, type1_rule)
                        row = summarize_combo(trades)
                        row.update(params)
                        type1_rows.append(row)
                        if idx % 5 == 0 or idx == total1:
                            print(f"type1 参数进度: {idx}/{total1}")

    type4_rows: list[dict[str, object]] = []
    total4 = 2 * 2 * 2 * 3
    idx = 0
    for ret1_min in [0.03, 0.04]:
        for upper_ratio in [0.3, 0.5]:
            for j_max in [90.0, 100.0]:
                for touch_ratio in [1.00, 1.01, 1.02]:
                    idx += 1
                    params = {
                        "ret1_min": ret1_min,
                        "upper_shadow_body_ratio": upper_ratio,
                        "j_max": j_max,
                        "type4_touch_ratio": touch_ratio,
                    }
                    signals = select_type4(type4_candidates, params)
                    trades = build_trade_table(signals, dfs, type4_rule)
                    row = summarize_combo(trades)
                    row.update(params)
                    type4_rows.append(row)
                    print(f"type4 参数进度: {idx}/{total4}")

    type1_df = pd.DataFrame(type1_rows)
    type4_df = pd.DataFrame(type4_rows)
    type1_df.to_csv(RESULT_DIR / "type1_param_search.csv", index=False)
    type4_df.to_csv(RESULT_DIR / "type4_param_search.csv", index=False)

    best1 = pick_best(type1_df, 800)
    best4 = pick_best(type4_df, 200)
    summary = {
        "type1_exit": TYPE1_EXIT,
        "type4_exit": TYPE4_EXIT,
        "best_type1": best1.to_dict(),
        "best_type4": best4.to_dict(),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
