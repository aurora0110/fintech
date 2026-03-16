from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tmp.run_b2_type14_exit_and_param_opt import add_base_features, load_one_csv  # type: ignore


DATA_DIR = ROOT / "data/20260313/normal"
RESULT_DIR = ROOT / "results/distribution_point_zone_calibration_20260314"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def add_distribution_point_zone_labels(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["vol_rank30"] = x["volume"].rolling(30, min_periods=30).apply(
        lambda v: pd.Series(v).rank(pct=True).iloc[-1], raw=False
    )
    x["vol_rank60"] = x["volume"].rolling(60, min_periods=60).apply(
        lambda v: pd.Series(v).rank(pct=True).iloc[-1], raw=False
    )
    x["high30_prev"] = x["high"].shift(1).rolling(30).max()
    x["high60_prev"] = x["high"].shift(1).rolling(60).max()
    x["close60_prev"] = x["close"].shift(1).rolling(60).max()
    x["close20_max_prev"] = x["close"].shift(1).rolling(20).max()
    x["ret5"] = x["close"] / x["close"].shift(5) - 1.0
    x["ret10"] = x["close"] / x["close"].shift(10) - 1.0
    x["ret20"] = x["close"] / x["close"].shift(20) - 1.0
    x["ret5_prev"] = x["close"].shift(1) / x["close"].shift(6) - 1.0
    x["ret10_prev"] = x["close"].shift(1) / x["close"].shift(11) - 1.0
    x["ret20_prev"] = x["close"].shift(1) / x["close"].shift(21) - 1.0
    x["trend_slope_10"] = x["trend_line"] / x["trend_line"].shift(10) - 1.0
    x["trend_slope_5_prev"] = x["trend_line"].shift(1) / x["trend_line"].shift(6) - 1.0
    x["trend_slope_10_prev"] = x["trend_line"].shift(1) / x["trend_line"].shift(11) - 1.0
    x["is_bear"] = x["close"] < x["open"]
    x["is_bull"] = x["close"] > x["open"]
    x["bear_vol"] = np.where(x["is_bear"], x["volume"], 0.0)
    x["bull_vol"] = np.where(x["is_bull"], x["volume"], 0.0)
    x["upper_range_ratio"] = pd.Series(
        safe_div(x["upper_shadow"], (x["high"] - x["low"]).replace(0.0, np.nan)),
        index=x.index,
    ).clip(lower=0.0)
    x["body_ratio"] = pd.Series(
        safe_div((x["close"] - x["open"]).abs(), (x["high"] - x["low"]).replace(0.0, np.nan)),
        index=x.index,
    ).clip(lower=0.0)
    x["top_zone_now"] = (
        (x["high"] >= x["high30_prev"] * 0.97)
        | (x["high"] >= x["high60_prev"] * 0.95)
        | (x["close"] >= x["close20_max_prev"] * 0.94)
    )
    x["top_zone_recent"] = pd.Series(x["top_zone_now"].fillna(False)).rolling(12, min_periods=1).max().astype(bool)

    point_accel_heavy_bear = (
        x["is_bear"]
        & (x["body_ratio"] >= 0.45)
        & (x["close_position"] <= 0.36)
        & ((x["vol_rank30"] >= 0.90) | (x["vol_rank60"] >= 0.93) | (x["volume"] >= x["vol_ma20"] * 1.15))
        & (
            (x["ret5_prev"] >= 0.08)
            | (x["ret10_prev"] >= 0.12)
            | (x["ret20_prev"] >= 0.20)
            | (x["trend_slope_5_prev"] >= 0.03)
            | (x["trend_slope_10_prev"] >= 0.06)
        )
        & x["top_zone_now"]
    )

    point_failed_breakout = (
        (x["high"] >= x["high30_prev"] * 0.995)
        & (x["close"] <= x["high30_prev"] * 0.985)
        & (x["upper_range_ratio"] >= 0.28)
        & (x["volume"] >= x["vol_ma20"] * 1.2)
        & x["top_zone_now"]
    )

    point_heavy_bear = point_accel_heavy_bear | point_failed_breakout

    x["point_accel_heavy_bear"] = point_accel_heavy_bear.fillna(False)
    x["point_failed_breakout"] = point_failed_breakout.fillna(False)
    x["point_any"] = point_heavy_bear.fillna(False)

    # 区间：顶部阴量持续主导
    x["bear_vol_sum8"] = pd.Series(x["bear_vol"]).rolling(8).sum()
    x["bull_vol_sum8"] = pd.Series(x["bull_vol"]).rolling(8).sum()
    x["bear_days_8"] = pd.Series(x["is_bear"].astype(int)).rolling(8).sum()
    x["bear_vol_sum10"] = pd.Series(x["bear_vol"]).rolling(10).sum()
    x["bull_vol_sum10"] = pd.Series(x["bull_vol"]).rolling(10).sum()
    x["bear_days_10"] = pd.Series(x["is_bear"].astype(int)).rolling(10).sum()
    zone_top_distribution = (
        (x["bear_days_10"] >= 5)
        & (x["bear_vol_sum10"] >= x["bull_vol_sum10"] * 1.20)
        & x["top_zone_recent"]
    )

    # 区间：新高后连续放量下跌 / 两根巨阴后续弱化
    had_new_high_recent = pd.Series(x["top_zone_now"].shift(1).fillna(False)).rolling(8, min_periods=1).max().astype(bool)
    recent_bear_4 = pd.Series(x["is_bear"].astype(int)).rolling(4).sum() >= 2
    zone_post_new_high_selloff = (
        had_new_high_recent.fillna(False)
        & recent_bear_4.fillna(False)
        & (x["close"] <= x["close20_max_prev"] * 0.95)
        & ((x["volume"] >= x["vol_ma20"] * 1.00) | (x["vol_rank30"] >= 0.85))
    )

    # 区间：巨阴后阶梯量阴线下跌
    stair_active = pd.Series(False, index=x.index)
    point_arr = x["point_any"].fillna(False).to_numpy(dtype=bool)
    bear_arr = x["is_bear"].fillna(False).to_numpy(dtype=bool)
    vols = x["volume"].to_numpy(dtype=float)
    closes = x["close"].to_numpy(dtype=float)
    for i in range(len(x)):
        anchor = None
        for j in range(i - 1, max(-1, i - 6), -1):
            if j >= 0 and point_arr[j]:
                anchor = j
                break
        if anchor is None or i - anchor < 2:
            continue
        sub = x.iloc[anchor + 1 : i + 1]
        if len(sub) < 2:
            continue
        if not bool((sub["close"] < sub["open"]).all()):
            continue
        if not bool((sub["volume"].diff().fillna(0) <= 0).iloc[1:].all()):
            continue
        if not bool((sub["close"].diff().fillna(0) <= 0).iloc[1:].all()):
            continue
        stair_active.iat[i] = True

    x["zone_top_distribution"] = zone_top_distribution.fillna(False)
    x["zone_post_new_high_selloff"] = zone_post_new_high_selloff.fillna(False)
    x["zone_stair_bear"] = stair_active.fillna(False)
    x["zone_any"] = (
        x["zone_top_distribution"] | x["zone_post_new_high_selloff"] | x["zone_stair_bear"]
    )

    x["zone_start"] = x["zone_any"] & (~x["zone_any"].shift(1).fillna(False))
    x["zone_end"] = x["zone_any"] & (~x["zone_any"].shift(-1).fillna(False))
    x["point_or_zone"] = x["point_any"] | x["zone_any"]
    return x


def check_focus_cases() -> pd.DataFrame:
    focus: List[Tuple[str, str, List[str]]] = [
        ("东方财富", "SZ#300059", ["2020-07-16"]),
        ("光线传媒", "SZ#300251", ["2025-02-17"]),
        ("华谊兄弟", "SZ#300027", ["2013-10-08", "2013-10-22"]),
        ("晋亿实业", "SH#601002", ["2025-08-14"]),
        ("民生银行", "SH#600016", ["2025-07-31", "2013-02-07"]),
        ("宁德时代", "SZ#300750", ["2021-12-07"]),
        ("万科A", "SZ#000002", ["2018-01-29"]),
        ("卫宁健康", "SZ#300253", ["2025-02-17", "2025-03-10"]),
        ("镇洋发展", "SH#603213", ["2025-09-03"]),
        ("中材科技", "SZ#002080", ["2025-08-29"]),
        ("中国中铁", "SH#601390", ["2024-12-22", "2025-06-09"]),
    ]
    rows = []
    for name, code, dates in focus:
        path = DATA_DIR / f"{code}.txt"
        if not path.exists():
            rows.append({"name": name, "code": code, "check_date": "", "status": "missing_file"})
            continue
        raw = load_one_csv(path)
        if raw is None:
            rows.append({"name": name, "code": code, "check_date": "", "status": "load_failed"})
            continue
        x = add_distribution_point_zone_labels(add_base_features(raw))
        x["date_str"] = x["date"].dt.strftime("%Y-%m-%d")
        for d in dates:
            sub = x.loc[x["date_str"] == d]
            if sub.empty:
                rows.append({"name": name, "code": code, "check_date": d, "status": "date_missing"})
                continue
            r = sub.iloc[0]
            rows.append(
                {
                    "name": name,
                    "code": code,
                    "check_date": d,
                    "status": "focus_date",
                    "point_accel_heavy_bear": bool(r["point_accel_heavy_bear"]),
                    "point_failed_breakout": bool(r["point_failed_breakout"]),
                    "point_any": bool(r["point_any"]),
                    "zone_top_distribution": bool(r["zone_top_distribution"]),
                    "zone_post_new_high_selloff": bool(r["zone_post_new_high_selloff"]),
                    "zone_stair_bear": bool(r["zone_stair_bear"]),
                    "zone_any": bool(r["zone_any"]),
                    "zone_start": bool(r["zone_start"]),
                    "zone_end": bool(r["zone_end"]),
                }
            )
            target = pd.Timestamp(d)
            near = x[
                (x["date"] >= target - pd.Timedelta(days=7))
                & (x["date"] <= target + pd.Timedelta(days=7))
                & (x["point_or_zone"])
            ]
            for _, rr in near.iterrows():
                rows.append(
                    {
                        "name": name,
                        "code": code,
                        "check_date": rr["date_str"],
                        "status": "nearby_tag",
                        "point_accel_heavy_bear": bool(rr["point_accel_heavy_bear"]),
                        "point_failed_breakout": bool(rr["point_failed_breakout"]),
                        "point_any": bool(rr["point_any"]),
                        "zone_top_distribution": bool(rr["zone_top_distribution"]),
                        "zone_post_new_high_selloff": bool(rr["zone_post_new_high_selloff"]),
                        "zone_stair_bear": bool(rr["zone_stair_bear"]),
                        "zone_any": bool(rr["zone_any"]),
                        "zone_start": bool(rr["zone_start"]),
                        "zone_end": bool(rr["zone_end"]),
                    }
                )
    return pd.DataFrame(rows)


def main():
    focus_df = check_focus_cases()
    focus_df.to_csv(RESULT_DIR / "focus_point_zone_check.csv", index=False)
    summary = {
        "focus_rows": int(len(focus_df)),
        "focus_date_rows": int((focus_df["status"] == "focus_date").sum()),
        "nearby_tag_rows": int((focus_df["status"] == "nearby_tag").sum()),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
