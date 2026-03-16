from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brick_filter


INPUT_DIR = brick_filter.INPUT_DIR
EPS = brick_filter.EPS
CASE_CATEGORY_WEIGHTS = {
    "candle": 0.0296,
    "trend": 0.0550,
    "keyk": 0.1266,
    "volume": 0.1908,
    "support": 0.0000,
    "structure": 0.5980,
}


def triangle_quality(value: float, center: float, half_width: float) -> float:
    if not np.isfinite(value) or half_width <= 0:
        return 0.0
    score = 1.0 - abs(value - center) / half_width
    return float(max(score, 0.0))


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def calc_case_scores(row: pd.Series) -> dict:
    close_position_score = triangle_quality(float(row["close_position"]), 0.82, 0.22)
    body_ratio_score = triangle_quality(float(row["body_ratio"]), 0.58, 0.28)
    upper_shadow_score = triangle_quality(float(row["upper_shadow_ratio"]), 0.05, 0.10)
    lower_shadow_score = triangle_quality(float(row["lower_shadow_ratio"]), 0.08, 0.12)
    candle_score = (
        0.35 * close_position_score
        + 0.30 * body_ratio_score
        + 0.20 * upper_shadow_score
        + 0.15 * lower_shadow_score
    )

    trend_slope_3_score = triangle_quality(float(row["trend_slope_3"]), 0.012, 0.018)
    trend_slope_5_score = triangle_quality(float(row["trend_slope_5"]), 0.018, 0.025)
    long_slope_5_score = triangle_quality(float(row["long_slope_5"]), 0.008, 0.020)
    trend_gap = max(float(row["trend_line"]) - float(row["long_line"]), 0.0) / max(float(row["close"]), EPS)
    trend_gap_score = triangle_quality(trend_gap, 0.04, 0.05)
    trend_score = (
        0.30 * trend_slope_3_score
        + 0.30 * trend_slope_5_score
        + 0.15 * long_slope_5_score
        + 0.25 * trend_gap_score
    )

    signal_vs_ma5_score = triangle_quality(float(row["signal_vs_ma5"]), 1.35, 0.95)
    signal_vs_ma10_score = triangle_quality(float(row["signal_vs_ma10"]), 1.15, 0.65)
    vol_vs_prev_score = triangle_quality(float(row["vol_vs_prev"]), 1.30, 0.90)
    t1_vs_prev5_score = triangle_quality(float(row["t1_vs_prev5"]), 0.98, 0.22)
    pullback_vs_prev10_score = triangle_quality(float(row["pullback_vs_prev10"]), 0.78, 0.32)
    volume_score = (
        0.30 * signal_vs_ma5_score
        + 0.15 * signal_vs_ma10_score
        + 0.15 * vol_vs_prev_score
        + 0.20 * t1_vs_prev5_score
        + 0.20 * pullback_vs_prev10_score
    )

    keyk_high_score = 1.0 if bool(row["above_double_bull_high"]) else 0.0
    keyk_close_score = 1.0 if bool(row["above_double_bull_close"]) else 0.0
    keyk_low_score = 1.0 if bool(row["above_double_bull_low"]) else 0.0
    keyk_anchor_score = 1.0 if bool(row["has_double_bull_anchor"]) else 0.0
    keyk_score = (
        0.45 * keyk_high_score
        + 0.25 * keyk_close_score
        + 0.15 * keyk_low_score
        + 0.15 * keyk_anchor_score
    )

    along_trend_score = 1.0 if bool(row["along_trend_rise"]) else 0.0
    dist_trend_score = triangle_quality(float(row["dist_trend"]), 0.02, 0.035)
    dist_long_score = triangle_quality(float(row["dist_long"]), 0.06, 0.08)
    dist_20d_high_score = triangle_quality(abs(float(row["dist_20d_high"])), 0.02, 0.05)
    support_score = (
        0.30 * along_trend_score
        + 0.30 * dist_trend_score
        + 0.20 * dist_long_score
        + 0.20 * dist_20d_high_score
    )

    rebound_score = triangle_quality(float(row["rebound_ratio"]), 1.45, 1.10)
    green_streak_score = triangle_quality(float(row["prev_green_streak"]), 2.5, 1.5)
    structure_score = 0.65 * rebound_score + 0.35 * green_streak_score

    total_score = (
        CASE_CATEGORY_WEIGHTS["candle"] * candle_score
        + CASE_CATEGORY_WEIGHTS["trend"] * trend_score
        + CASE_CATEGORY_WEIGHTS["keyk"] * keyk_score
        + CASE_CATEGORY_WEIGHTS["volume"] * volume_score
        + CASE_CATEGORY_WEIGHTS["support"] * support_score
        + CASE_CATEGORY_WEIGHTS["structure"] * structure_score
    )
    return {
        "总分": round(total_score, 4),
        "K线质量分": round(candle_score, 4),
        "趋势质量分": round(trend_score, 4),
        "关键K分": round(keyk_score, 4),
        "量能质量分": round(volume_score, 4),
        "支撑位置分": round(support_score, 4),
        "结构分": round(structure_score, 4),
        "收盘位置分": round(close_position_score, 4),
        "实体分": round(body_ratio_score, 4),
        "上影线分": round(upper_shadow_score, 4),
        "下影线分": round(lower_shadow_score, 4),
        "趋势线3日斜率分": round(trend_slope_3_score, 4),
        "趋势线5日斜率分": round(trend_slope_5_score, 4),
        "多空线5日斜率分": round(long_slope_5_score, 4),
        "趋势领先分": round(trend_gap_score, 4),
        "量比五日分": round(signal_vs_ma5_score, 4),
        "量比十日分": round(signal_vs_ma10_score, 4),
        "量比前一日分": round(vol_vs_prev_score, 4),
        "T-1缩量分": round(t1_vs_prev5_score, 4),
        "回调缩量分": round(pullback_vs_prev10_score, 4),
    }


def calc_future_metrics(x: pd.DataFrame, signal_idx: int) -> dict:
    entry_idx = signal_idx + 1
    if entry_idx >= len(x):
        return {}
    entry_open = float(x.at[entry_idx, "open"])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return {}

    future_3 = x.iloc[entry_idx : min(entry_idx + 3, len(x))].copy()
    future_5 = x.iloc[entry_idx : min(entry_idx + 5, len(x))].copy()
    if future_3.empty or future_5.empty:
        return {}

    max_gain_3d = float(future_3["high"].max() / entry_open - 1.0)
    max_gain_5d = float(future_5["high"].max() / entry_open - 1.0)

    prev_closes = x["close"].shift(1)
    future_up_days = (x["close"] > prev_closes).iloc[entry_idx : min(entry_idx + 3, len(x))]
    up_days_ratio_3d = float(future_up_days.mean()) if len(future_up_days) > 0 else 0.0

    gain3_score = clip01(max_gain_3d / 0.12)
    up_days_score = clip01(up_days_ratio_3d)
    later_gain_score = clip01(max_gain_5d / 0.18)
    future_target = 0.50 * gain3_score + 0.20 * up_days_score + 0.30 * later_gain_score

    return {
        "entry_open": entry_open,
        "max_gain_3d": max_gain_3d,
        "up_days_ratio_3d": up_days_ratio_3d,
        "max_gain_5d": max_gain_5d,
        "future_target": float(future_target),
    }


def build_case_pool_df(input_dir: Path, include_future: bool = False) -> pd.DataFrame:
    rows: List[dict] = []
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".csv", ".txt"}])
    total = len(files)
    for idx, path in enumerate(files, 1):
        df = brick_filter.load_one_csv(str(path))
        if df is None or df.empty:
            continue
        code = str(df["code"].iloc[0])
        x = brick_filter.add_features(df)
        x["dist_20d_high"] = brick_filter.safe_div(x["close"], x["high"].rolling(20).max()) - 1.0

        pattern_a_relaxed = (
            (x["prev_green_streak"] >= 2)
            & x["brick_red"]
            & x["close_pullback_white"].shift(1).fillna(False)
            & x["close_above_white"]
        )
        pattern_b_relaxed = (
            (pd.Series(brick_filter.calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(3) >= 2)
            & x["brick_red"]
            & x["brick_green"].shift(1).fillna(False)
            & x["brick_red"].shift(2).fillna(False)
            & x["close_pullback_white"].shift(1).fillna(False)
            & x["close_above_white"]
        )
        relaxed_mask = (
            (pattern_a_relaxed | pattern_b_relaxed)
            & x["brick_red"]
            & x["close_above_white"]
            & (x["trend_line"] > x["long_line"])
            & x["signal_vs_ma5"].between(0.8, 2.5, inclusive="both")
            & x["ret1"].between(0.0, 0.10, inclusive="both")
            & (x["rebound_ratio"] >= 0.8)
        )
        signal_idxs = np.flatnonzero(relaxed_mask.to_numpy())
        for signal_idx in signal_idxs:
            row = x.iloc[int(signal_idx)]
            score_map = calc_case_scores(row)
            future_map = calc_future_metrics(x, int(signal_idx)) if include_future else {}
            rows.append(
                {
                    "date": row["date"],
                    "code": code,
                    "signal_close": float(row["close"]),
                    "signal_open": float(row["open"]),
                    "signal_high": float(row["high"]),
                    "signal_low": float(row["low"]),
                    "signal_volume": float(row["volume"]),
                    "rebound_ratio": float(row["rebound_ratio"]),
                    "signal_vs_ma5": float(row["signal_vs_ma5"]),
                    "signal_vs_ma10": float(row["signal_vs_ma10"]),
                    "vol_vs_prev": float(row["vol_vs_prev"]),
                    "close_position": float(row["close_position"]),
                    "body_ratio": float(row["body_ratio"]),
                    "upper_shadow_ratio": float(row["upper_shadow_ratio"]),
                    "lower_shadow_ratio": float(row["lower_shadow_ratio"]),
                    "trend_slope_3": float(row["trend_slope_3"]),
                    "trend_slope_5": float(row["trend_slope_5"]),
                    "long_slope_5": float(row["long_slope_5"]),
                    "dist_trend": float(row["dist_trend"]),
                    "dist_long": float(row["dist_long"]),
                    "dist_20d_high": float(row["dist_20d_high"]),
                    "along_trend_rise": bool(row["along_trend_rise"]),
                    "has_double_bull_anchor": bool(row["has_double_bull_anchor"]),
                    "above_double_bull_high": bool(row["above_double_bull_high"]),
                    "above_double_bull_close": bool(row["above_double_bull_close"]),
                    "above_double_bull_low": bool(row["above_double_bull_low"]),
                    "t1_vs_prev5": float(row["t1_vs_prev5"]),
                    "pullback_vs_prev10": float(row["pullback_vs_prev10"]),
                    "prev_green_streak": float(row["prev_green_streak"]),
                    **score_map,
                    **future_map,
                }
            )
        if idx % 500 == 0 or idx == total:
            print(f"案例候选池进度: {idx}/{total}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "总分", "code"], ascending=[True, False, True]).reset_index(drop=True)


def remove_strict_selected(case_pool_df: pd.DataFrame, strict_selected_df: pd.DataFrame) -> pd.DataFrame:
    if case_pool_df.empty:
        return case_pool_df
    strict_pairs = set(
        zip(pd.to_datetime(strict_selected_df["date"]).dt.strftime("%Y-%m-%d"), strict_selected_df["code"])
    )
    keep_mask = []
    for _, row in case_pool_df.iterrows():
        key = (pd.to_datetime(row["date"]).strftime("%Y-%m-%d"), row["code"])
        keep_mask.append(key not in strict_pairs)
    return case_pool_df[pd.Series(keep_mask, index=case_pool_df.index)].reset_index(drop=True)


def build_latest_case_pool(input_dir: Path) -> pd.DataFrame:
    strict_signal_df = brick_filter.build_signal_df(input_dir)
    strict_selected_df = brick_filter.apply_selection(strict_signal_df)
    case_pool_df = build_case_pool_df(input_dir)
    case_pool_df = remove_strict_selected(case_pool_df, strict_selected_df)
    if case_pool_df.empty:
        return case_pool_df
    latest_trade_date = pd.to_datetime(case_pool_df["date"]).max()
    latest_df = case_pool_df[pd.to_datetime(case_pool_df["date"]) == latest_trade_date].copy()
    return latest_df.sort_values(["总分", "code"], ascending=[False, True]).reset_index(drop=True)


def to_main_rows(case_pool_df: pd.DataFrame) -> List[list]:
    if case_pool_df.empty:
        return []
    rows: List[list] = []
    for _, row in case_pool_df.iterrows():
        stop_price = round(float(row["signal_low"]) * 0.99, 2)
        buy_price = round(float(row["signal_close"]), 2)
        detail = (
            f"案例池 | 总分={row['总分']:.4f} | K线质量分={row['K线质量分']:.4f} | "
            f"趋势质量分={row['趋势质量分']:.4f} | 关键K分={row['关键K分']:.4f} | "
            f"量能质量分={row['量能质量分']:.4f} | 支撑位置分={row['支撑位置分']:.4f} | "
            f"结构分={row['结构分']:.4f} | 收盘位置分={row['收盘位置分']:.4f} | "
            f"实体分={row['实体分']:.4f} | 上影线分={row['上影线分']:.4f} | "
            f"下影线分={row['下影线分']:.4f} | 趋势线3日斜率分={row['趋势线3日斜率分']:.4f} | "
            f"趋势线5日斜率分={row['趋势线5日斜率分']:.4f} | 趋势领先分={row['趋势领先分']:.4f} | "
            f"量比五日分={row['量比五日分']:.4f} | T-1缩量分={row['T-1缩量分']:.4f} | "
            f"回调缩量分={row['回调缩量分']:.4f}"
        )
        rows.append(
            [
                row["code"].split("#")[-1],
                f"{stop_price:.2f}",
                f"{buy_price:.2f}",
                f"{float(row['总分']):.4f}",
                detail,
            ]
        )
    return rows


def main() -> None:
    case_pool_df = build_latest_case_pool(INPUT_DIR)
    if case_pool_df.empty:
        return
    print(
        case_pool_df[
            [
                "date",
                "code",
                "总分",
                "K线质量分",
                "趋势质量分",
                "关键K分",
                "量能质量分",
                "支撑位置分",
                "结构分",
                "收盘位置分",
                "实体分",
                "上影线分",
                "下影线分",
                "趋势线3日斜率分",
                "趋势线5日斜率分",
                "趋势领先分",
                "量比五日分",
                "T-1缩量分",
                "回调缩量分",
                "rebound_ratio",
                "signal_vs_ma5",
                "close_position",
                "upper_shadow_ratio",
                "lower_shadow_ratio",
                "trend_slope_3",
                "trend_slope_5",
                "above_double_bull_high",
                "along_trend_rise",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
