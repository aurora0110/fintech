from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sys

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brick_filter, pinfilter, technical_indicators
from utils.tmp import brick_case_semantics_v1_20260326 as case_semantics

DATA_DIR = ROOT / "data" / "20260324"
PIN_CASE_DIR = ROOT / "data" / "完美图" / "单针"
BRICK_CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
RESULT_DIR = ROOT / "results" / "pin_brick_relation_analysis_v1_20260402_r1"

EPS = 1e-12


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_price_df(path: Path) -> pd.DataFrame | None:
    df, err = pinfilter.stoploss.load_data(str(path))
    if err or df is None or df.empty:
        return None
    return df.sort_values("日期").drop_duplicates("日期").reset_index(drop=True)


def parse_pin_cases() -> pd.DataFrame:
    pat = case_semantics.parse_case_images(PIN_CASE_DIR, skip_counter_examples=False)
    if pat.empty:
        return pat
    pat["case_group"] = "pin"
    return pat


def parse_brick_cases() -> pd.DataFrame:
    df = case_semantics.parse_case_images(BRICK_CASE_DIR, skip_counter_examples=True)
    if df.empty:
        return df
    df["case_group"] = "brick"
    return df


def build_case_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    name_map = case_semantics.build_name_code_map(DATA_DIR)
    pin_df = parse_pin_cases()
    brick_df = parse_brick_cases()
    for df in (pin_df, brick_df):
        if df.empty:
            continue
        df["code"] = df["stock_name"].map(name_map)
        df["code_key"] = df["code"].map(case_semantics.code_key)
        df.dropna(subset=["code"], inplace=True)
        df.sort_values(["code_key", "signal_date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return pin_df, brick_df


def build_brick_signal_mask(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    x = brick_filter.add_features(
        pd.DataFrame(
            {
                "date": work["日期"],
                "open": work["开盘"],
                "high": work["最高"],
                "low": work["最低"],
                "close": work["收盘"],
                "volume": work["成交量"],
                "code": "NA",
            }
        )
    )
    mask_a = x["pattern_a"] & (x["rebound_ratio"] >= 0.8)
    mask_b = x["pattern_b"] & (x["rebound_ratio"] >= 1.0)
    legacy_mask = (
        x["signal_base"]
        & (x["ret1"] <= 0.08)
        & (mask_a | mask_b)
        & (x["trend_line"] > x["long_line"])
    )
    perfect_mask = (
        legacy_mask
        & x["trend_line"].gt(x["long_line"])
        & x["ret1"].between(-0.03, 0.11, inclusive="both")
    )
    out = x.copy()
    out["signal_date"] = pd.to_datetime(df["日期"]).reset_index(drop=True)
    out["brick_signal"] = perfect_mask.fillna(False).to_numpy(dtype=bool)
    return out


def pin_signal_snapshot(df: pd.DataFrame, idx: int) -> dict[str, Any] | None:
    sub = df.iloc[: idx + 1].copy()
    feat = pinfilter.build_today_features(sub)
    if feat is None:
        return None
    if feat["trend_line"] <= feat["long_line"]:
        return None
    if not technical_indicators.caculate_pin(sub):
        return None
    matched = pinfilter.detect_subtypes(feat)
    if not matched:
        return None
    return {
        "matched_subtypes": matched,
        "trend_line_lead": float(feat["trend_line_lead"]) if np.isfinite(feat["trend_line_lead"]) else np.nan,
        "trend_slope_5": float(feat["trend_slope_5"]) if np.isfinite(feat["trend_slope_5"]) else np.nan,
        "ret10": float(feat["ret10"]) if np.isfinite(feat["ret10"]) else np.nan,
        "ret3": float(feat["ret3"]) if np.isfinite(feat["ret3"]) else np.nan,
        "signal_vs_ma20": float(feat["signal_vs_ma20"]) if np.isfinite(feat["signal_vs_ma20"]) else np.nan,
        "vol_vs_prev": float(feat["vol_vs_prev"]) if np.isfinite(feat["vol_vs_prev"]) else np.nan,
        "close_position": float(feat["close_position"]) if np.isfinite(feat["close_position"]) else np.nan,
        "lower_shadow_ratio": float(feat["lower_shadow_ratio"]) if np.isfinite(feat["lower_shadow_ratio"]) else np.nan,
        "n_up_any": bool(feat["n_up_any"]),
        "along_trend_up": bool(feat["along_trend_up"]),
        "keyk_support_active": bool(feat["keyk_support_active"]),
    }


@dataclass
class FutureBrickHit:
    future_signal_date: pd.Timestamp | None
    lag_days: int | None
    lag_trading_days: int | None


def find_future_brick_signal(brick_feat: pd.DataFrame, pin_date: pd.Timestamp, max_lag_trading_days: int = 20) -> FutureBrickHit:
    later = brick_feat[(brick_feat["signal_date"] > pin_date) & (brick_feat["brick_signal"])].copy()
    if later.empty:
        return FutureBrickHit(None, None, None)
    later = later.reset_index(drop=True)
    first = later.iloc[0]
    all_dates = brick_feat["signal_date"].tolist()
    start_idx = all_dates.index(pin_date)
    future_idx = all_dates.index(first["signal_date"])
    lag_trading = future_idx - start_idx
    if lag_trading > max_lag_trading_days:
        return FutureBrickHit(None, None, None)
    lag_days = int((pd.Timestamp(first["signal_date"]) - pin_date).days)
    return FutureBrickHit(pd.Timestamp(first["signal_date"]), lag_days, lag_trading)


def find_recent_pin_signal(pin_rows: list[dict[str, Any]], brick_date: pd.Timestamp, max_lag_trading_days: int = 20) -> dict[str, Any] | None:
    candidates = [r for r in pin_rows if r["signal_date"] < brick_date]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r["signal_date"])
    lag_trading = int(best["trade_index_to_signal"])
    if lag_trading > max_lag_trading_days:
        return None
    return best


def analyze() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pin_cases, brick_cases = build_case_tables()
    brick_feat_cache: dict[str, pd.DataFrame] = {}
    pin_case_rows: list[dict[str, Any]] = []
    same_stock_links: list[dict[str, Any]] = []
    brick_case_feat_df = case_semantics.load_case_day_features(BRICK_CASE_DIR, DATA_DIR)

    overlap_stocks = sorted(set(pin_cases["code_key"]) & set(brick_cases["code_key"]))
    overlap_names = sorted(set(pin_cases["stock_name"]) & set(brick_cases["stock_name"]))

    for row in pin_cases.itertuples(index=False):
        code = str(row.code)
        file_path = DATA_DIR / f"{code}.txt"
        price_df = load_price_df(file_path)
        if price_df is None:
            continue
        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(pd.to_datetime(price_df["日期"]))}
        signal_date = pd.Timestamp(row.signal_date)
        if signal_date not in date_to_idx:
            continue
        idx = date_to_idx[signal_date]
        pin_snap = pin_signal_snapshot(price_df, idx)
        if pin_snap is None:
            continue
        if code not in brick_feat_cache:
            brick_feat_cache[code] = build_brick_signal_mask(price_df)
        fut = find_future_brick_signal(brick_feat_cache[code], signal_date, max_lag_trading_days=20)
        pin_case_rows.append(
            {
                "stock_name": row.stock_name,
                "code": code,
                "signal_date": signal_date,
                "pin_subtypes": "+".join(pin_snap["matched_subtypes"]),
                "future_brick_signal_date": fut.future_signal_date,
                "future_brick_lag_days": fut.lag_days,
                "future_brick_lag_trading_days": fut.lag_trading_days,
                "future_brick_within_20d": fut.future_signal_date is not None,
                "trend_line_lead": pin_snap["trend_line_lead"],
                "trend_slope_5": pin_snap["trend_slope_5"],
                "ret10": pin_snap["ret10"],
                "ret3": pin_snap["ret3"],
                "signal_vs_ma20": pin_snap["signal_vs_ma20"],
                "vol_vs_prev": pin_snap["vol_vs_prev"],
                "close_position": pin_snap["close_position"],
                "lower_shadow_ratio": pin_snap["lower_shadow_ratio"],
                "n_up_any": pin_snap["n_up_any"],
                "along_trend_up": pin_snap["along_trend_up"],
                "keyk_support_active": pin_snap["keyk_support_active"],
            }
        )

    pin_case_df = pd.DataFrame(pin_case_rows).sort_values(["signal_date", "code"]).reset_index(drop=True)

    # Perfect-case image overlap on same stock.
    for code_key in overlap_stocks:
        pin_sub = pin_cases[pin_cases["code_key"] == code_key].sort_values("signal_date")
        brick_sub = brick_cases[brick_cases["code_key"] == code_key].sort_values("signal_date")
        for prow in pin_sub.itertuples(index=False):
            later = brick_sub[brick_sub["signal_date"] > prow.signal_date]
            if later.empty:
                continue
            brow = later.iloc[0]
            same_stock_links.append(
                {
                    "stock_name": prow.stock_name,
                    "code": prow.code,
                    "pin_signal_date": pd.Timestamp(prow.signal_date),
                    "brick_case_date": pd.Timestamp(brow["signal_date"]),
                    "lag_days": int((pd.Timestamp(brow["signal_date"]) - pd.Timestamp(prow.signal_date)).days),
                }
            )

    overlap_df = pd.DataFrame(same_stock_links).sort_values(["pin_signal_date", "code"]).reset_index(drop=True) if same_stock_links else pd.DataFrame()

    pin_case_df.to_csv(RESULT_DIR / "pin_cases_with_future_brick.csv", index=False)
    if not overlap_df.empty:
        overlap_df.to_csv(RESULT_DIR / "pin_to_brick_same_stock_links.csv", index=False)
    if not brick_case_feat_df.empty:
        brick_case_feat_df.to_csv(RESULT_DIR / "brick_case_features.csv", index=False)

    subtype_counts = {}
    subtype_future_rate = {}
    if not pin_case_df.empty:
        for subtype, g in pin_case_df.groupby("pin_subtypes"):
            subtype_counts[subtype] = int(len(g))
            subtype_future_rate[subtype] = float(g["future_brick_within_20d"].mean())

    summary = {
        "pin_case_count": int(len(pin_cases)),
        "brick_case_count": int(len(brick_cases)),
        "pin_case_count_with_code": int(pin_cases["code"].notna().sum()) if not pin_cases.empty else 0,
        "brick_case_count_with_code": int(brick_cases["code"].notna().sum()) if not brick_cases.empty else 0,
        "same_stock_name_overlap_count": int(len(overlap_names)),
        "same_stock_code_overlap_count": int(len(overlap_stocks)),
        "same_stock_pin_then_brick_link_count": int(len(overlap_df)),
        "pin_cases_with_future_brick_signal_within_20d_count": int(pin_case_df["future_brick_within_20d"].sum()) if not pin_case_df.empty else 0,
        "pin_cases_with_future_brick_signal_within_20d_ratio": float(pin_case_df["future_brick_within_20d"].mean()) if not pin_case_df.empty else 0.0,
        "pin_to_brick_lag_trading_days_median": float(pin_case_df.loc[pin_case_df["future_brick_within_20d"], "future_brick_lag_trading_days"].median()) if not pin_case_df.empty and pin_case_df["future_brick_within_20d"].any() else None,
        "pin_to_brick_lag_trading_days_mean": float(pin_case_df.loc[pin_case_df["future_brick_within_20d"], "future_brick_lag_trading_days"].mean()) if not pin_case_df.empty and pin_case_df["future_brick_within_20d"].any() else None,
        "pin_subtype_counts": subtype_counts,
        "pin_subtype_future_brick_rate": subtype_future_rate,
        "pin_feature_medians": {
            "trend_line_lead": float(pin_case_df["trend_line_lead"].median()) if not pin_case_df.empty else None,
            "trend_slope_5": float(pin_case_df["trend_slope_5"].median()) if not pin_case_df.empty else None,
            "ret10": float(pin_case_df["ret10"].median()) if not pin_case_df.empty else None,
            "ret3": float(pin_case_df["ret3"].median()) if not pin_case_df.empty else None,
            "signal_vs_ma20": float(pin_case_df["signal_vs_ma20"].median()) if not pin_case_df.empty else None,
            "close_position": float(pin_case_df["close_position"].median()) if not pin_case_df.empty else None,
            "lower_shadow_ratio": float(pin_case_df["lower_shadow_ratio"].median()) if not pin_case_df.empty else None,
        },
        "brick_feature_medians": {
            "prev_green_streak": float(brick_case_feat_df["prev_green_streak"].median()) if not brick_case_feat_df.empty else None,
            "rebound_ratio": float(brick_case_feat_df["rebound_ratio"].median()) if not brick_case_feat_df.empty else None,
            "signal_ret": float(brick_case_feat_df["signal_ret"].median()) if not brick_case_feat_df.empty else None,
            "upper_shadow_pct": float(brick_case_feat_df["upper_shadow_pct"].median()) if not brick_case_feat_df.empty else None,
            "close_to_trend": float(brick_case_feat_df["close_to_trend"].median()) if not brick_case_feat_df.empty else None,
            "close_to_long": float(brick_case_feat_df["close_to_long"].median()) if not brick_case_feat_df.empty else None,
            "trend_spread": float(brick_case_feat_df["trend_spread"].median()) if not brick_case_feat_df.empty else None,
        },
    }
    write_json(RESULT_DIR / "summary.json", summary)


if __name__ == "__main__":
    analyze()
