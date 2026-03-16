from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/lidongyang/Desktop/Qstrategy")
from utils import brick_filter


RAW_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260312")
NORMAL_DIR = RAW_DIR / "normal"
CASE_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/完美图/砖型图")
OUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/perfect_brick_case_analysis_deep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_cases() -> pd.DataFrame:
    rows: List[dict] = []
    pat = re.compile(r"(.+?)(\d{8})\.png$")
    for path in sorted(CASE_DIR.glob("*.png")):
        m = pat.match(path.name)
        if not m:
            continue
        rows.append(
            {
                "股票名称": m.group(1),
                "信号日期": pd.to_datetime(m.group(2), format="%Y%m%d"),
                "案例文件": str(path),
            }
        )
    return pd.DataFrame(rows)


def build_name_code_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    valid_codes = {p.stem for p in NORMAL_DIR.glob("*.txt")}
    for path in RAW_DIR.glob("*.txt"):
        try:
            with open(path, "r", encoding="gbk", errors="ignore") as f:
                first_line = f.readline().strip()
        except Exception:
            continue
        parts = first_line.split()
        if len(parts) >= 2 and parts[0].isdigit() and path.stem in valid_codes:
            mapping[parts[1]] = path.stem
    fallback = Path("/Users/lidongyang/Desktop/Qstrategy/results/perfect_brick_case_analysis_v2/case_match_results.csv")
    if fallback.exists():
        old = pd.read_csv(fallback)
        for _, row in old.iterrows():
            name = str(row.get("股票名称", "")).strip()
            code = str(row.get("股票代码", "")).strip()
            if name and code and code in valid_codes and code.lower() != "nan":
                mapping.setdefault(name, code)
    return mapping


def calc_base_signal_row(row: pd.Series) -> bool:
    mask_a = bool(row["pattern_a"]) and float(row["rebound_ratio"]) >= 1.2
    mask_b = bool(row["pattern_b"]) and float(row["rebound_ratio"]) >= 1.0
    return (
        (mask_a or mask_b)
        and bool(row["pullback_shrinking"])
        and bool(row["signal_vs_ma5_valid"])
        and bool(row["not_sideways"])
        and pd.notna(row["ret1"])
        and float(row["ret1"]) <= 0.08
        and float(row["trend_line"]) > float(row["long_line"])
    )


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    x = brick_filter.add_features(df).copy()

    prev_low = x["low"].shift(1)
    prev_close = x["close"].shift(1)
    prev_trend = x["trend_line"].shift(1)
    prev_long = x["long_line"].shift(1)

    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    real_body = (x["close"] - x["open"]).abs()
    upper_shadow = x["high"] - np.maximum(x["open"], x["close"])
    lower_shadow = np.minimum(x["open"], x["close"]) - x["low"]

    x["close_position"] = (x["close"] - x["low"]) / full_range
    x["upper_shadow_ratio"] = upper_shadow / full_range
    x["lower_shadow_ratio"] = lower_shadow / full_range
    x["body_ratio"] = real_body / full_range

    x["signal_vs_ma10"] = x["volume"] / x["volume"].shift(1).rolling(10).mean()
    x["vol_vs_prev"] = x["volume"] / x["volume"].shift(1)
    x["vol_rank_20"] = x["volume"].rolling(20).rank(pct=True)
    x["ret_5d_before"] = x["close"] / x["close"].shift(5) - 1.0
    x["ret_10d_before"] = x["close"] / x["close"].shift(10) - 1.0

    x["trend_slope_3"] = x["trend_line"] / x["trend_line"].shift(3) - 1.0
    x["trend_slope_5"] = x["trend_line"] / x["trend_line"].shift(5) - 1.0
    x["long_slope_3"] = x["long_line"] / x["long_line"].shift(3) - 1.0
    x["long_slope_5"] = x["long_line"] / x["long_line"].shift(5) - 1.0
    x["trend_long_slope_gap_5"] = x["trend_slope_5"] - x["long_slope_5"]

    x["dist_trend"] = x["close"] / x["trend_line"] - 1.0
    x["dist_long"] = x["close"] / x["long_line"] - 1.0
    x["dist_20d_high"] = x["close"] / x["high"].rolling(20).max() - 1.0
    x["dist_20d_low"] = x["close"] / x["low"].rolling(20).min() - 1.0

    x["回踩趋势线"] = (prev_low <= prev_trend * 1.01) & (prev_close >= prev_trend * 0.99)
    x["回踩多空线"] = (prev_low <= prev_long * 1.01) & (prev_close >= prev_long * 0.99)
    x["趋势线支撑"] = (x["close"] > x["trend_line"]) & (x["dist_trend"] <= 0.03)
    x["多空线支撑"] = (x["close"] > x["long_line"]) & (x["dist_long"] <= 0.05)
    x["沿趋势线上涨"] = (
        (x["close"] > x["trend_line"])
        & (x["close"].shift(1) > prev_trend)
        & (x["trend_slope_5"] > 0)
        & (x["dist_trend"].between(0, 0.03))
    )

    x["倍量阳柱"] = (x["close"] > x["open"]) & (x["volume"] >= x["volume"].shift(1) * 2.0)
    x["60日倍量阳柱数"] = x["倍量阳柱"].rolling(60).sum()
    x["30日倍量阳柱数"] = x["倍量阳柱"].rolling(30).sum()

    last_high = np.full(len(x), np.nan)
    last_low = np.full(len(x), np.nan)
    last_close = np.full(len(x), np.nan)
    last_idx = np.full(len(x), np.nan)
    recent_any = np.zeros(len(x), dtype=bool)
    recent_count = np.zeros(len(x), dtype=float)

    high_vals = x["high"].to_numpy(dtype=float)
    low_vals = x["low"].to_numpy(dtype=float)
    close_vals = x["close"].to_numpy(dtype=float)
    flag = x["倍量阳柱"].to_numpy(dtype=bool)

    for i in range(len(x)):
        start = max(0, i - 60)
        recent_idx = np.flatnonzero(flag[start:i])
        if recent_idx.size == 0:
            continue
        actual = recent_idx + start
        j = actual[-1]
        last_high[i] = high_vals[j]
        last_low[i] = low_vals[j]
        last_close[i] = close_vals[j]
        last_idx[i] = float(j)
        recent_any[i] = True
        recent_count[i] = float(len(actual))

    x["前60日存在倍量阳柱"] = recent_any
    x["前60日倍量阳柱个数"] = recent_count
    x["最近倍量阳柱最高价"] = last_high
    x["最近倍量阳柱最低价"] = last_low
    x["最近倍量阳柱收盘价"] = last_close
    x["站上倍量柱最高价"] = x["close"] > x["最近倍量阳柱最高价"]
    x["站上倍量柱收盘价"] = x["close"] > x["最近倍量阳柱收盘价"]
    x["站上倍量柱最低价"] = x["close"] > x["最近倍量阳柱最低价"]
    x["距倍量柱最高价"] = x["close"] / x["最近倍量阳柱最高价"] - 1.0
    x["距倍量柱收盘价"] = x["close"] / x["最近倍量阳柱收盘价"] - 1.0
    x["距倍量柱最低价"] = x["close"] / x["最近倍量阳柱最低价"] - 1.0

    return x


def build_case_date_pool(case_dates: List[pd.Timestamp]) -> pd.DataFrame:
    rows: List[dict] = []
    files = sorted(NORMAL_DIR.glob("*.txt"))
    total = len(files)
    case_date_set = set(case_dates)
    for idx, path in enumerate(files, 1):
        if idx % 500 == 0 or idx == total:
            print(f"特征进度: {idx}/{total}")
        df = brick_filter.load_one_csv(str(path))
        if df is None or df.empty:
            continue
        x = enrich_df(df)
        code = str(x["code"].iloc[0])
        date_map = {pd.Timestamp(d): i for i, d in enumerate(x["date"])}
        for d in case_date_set:
            signal_idx = date_map.get(pd.Timestamp(d))
            if signal_idx is None:
                continue
            row = x.iloc[signal_idx]
            if not calc_base_signal_row(row):
                continue
            pullback_ratio = float(row["pullback_avg_vol"] / row["up_leg_avg_vol"]) if pd.notna(row["up_leg_avg_vol"]) and float(row["up_leg_avg_vol"]) > 0 else np.nan
            trend_spread = float((row["trend_line"] - row["long_line"]) / row["close"]) if pd.notna(row["close"]) and float(row["close"]) > 0 else np.nan
            rows.append(
                {
                    "date": pd.Timestamp(d),
                    "code": code,
                    "rebound_ratio": float(row["rebound_ratio"]),
                    "signal_vs_ma5": float(row["signal_vs_ma5"]),
                    "signal_vs_ma10": float(row["signal_vs_ma10"]),
                    "vol_vs_prev": float(row["vol_vs_prev"]),
                    "vol_rank_20": float(row["vol_rank_20"]),
                    "pullback_shrink_ratio": pullback_ratio,
                    "ret1": float(row["ret1"]),
                    "trend_spread": trend_spread,
                    "close_position": float(row["close_position"]),
                    "upper_shadow_ratio": float(row["upper_shadow_ratio"]),
                    "lower_shadow_ratio": float(row["lower_shadow_ratio"]),
                    "body_ratio": float(row["body_ratio"]),
                    "trend_slope_3": float(row["trend_slope_3"]),
                    "trend_slope_5": float(row["trend_slope_5"]),
                    "long_slope_3": float(row["long_slope_3"]),
                    "long_slope_5": float(row["long_slope_5"]),
                    "trend_long_slope_gap_5": float(row["trend_long_slope_gap_5"]),
                    "dist_trend": float(row["dist_trend"]),
                    "dist_long": float(row["dist_long"]),
                    "dist_20d_high": float(row["dist_20d_high"]),
                    "dist_20d_low": float(row["dist_20d_low"]),
                    "ret_5d_before": float(row["ret_5d_before"]),
                    "ret_10d_before": float(row["ret_10d_before"]),
                    "回踩趋势线": bool(row["回踩趋势线"]),
                    "回踩多空线": bool(row["回踩多空线"]),
                    "趋势线支撑": bool(row["趋势线支撑"]),
                    "多空线支撑": bool(row["多空线支撑"]),
                    "沿趋势线上涨": bool(row["沿趋势线上涨"]),
                    "前60日存在倍量阳柱": bool(row["前60日存在倍量阳柱"]),
                    "前60日倍量阳柱个数": float(row["前60日倍量阳柱个数"]),
                    "站上倍量柱最高价": bool(row["站上倍量柱最高价"]) if pd.notna(row["站上倍量柱最高价"]) else False,
                    "站上倍量柱收盘价": bool(row["站上倍量柱收盘价"]) if pd.notna(row["站上倍量柱收盘价"]) else False,
                    "站上倍量柱最低价": bool(row["站上倍量柱最低价"]) if pd.notna(row["站上倍量柱最低价"]) else False,
                    "距倍量柱最高价": float(row["距倍量柱最高价"]) if pd.notna(row["距倍量柱最高价"]) else np.nan,
                    "距倍量柱收盘价": float(row["距倍量柱收盘价"]) if pd.notna(row["距倍量柱收盘价"]) else np.nan,
                    "距倍量柱最低价": float(row["距倍量柱最低价"]) if pd.notna(row["距倍量柱最低价"]) else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame()
    pool = pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)
    pool["rebound_rank"] = pool.groupby("date")["rebound_ratio"].rank(pct=True)
    pool["trend_rank"] = pool.groupby("date")["trend_spread"].rank(pct=True)
    pool["shrink_quality"] = 1.0 - np.minimum(np.abs(pool["pullback_shrink_ratio"] - 0.8) / 0.3, 1.0)
    pool["shrink_rank"] = pool.groupby("date")["shrink_quality"].rank(pct=True)
    pool["sort_score"] = 0.50 * pool["shrink_rank"] + 0.30 * pool["rebound_rank"] + 0.20 * pool["trend_rank"]
    pool["score_pct_rank"] = pool.groupby("date")["sort_score"].rank(pct=True)
    pool = pool.sort_values(["date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    pool["daily_rank"] = pool.groupby("date").cumcount() + 1
    return pool


def main() -> None:
    cases = parse_cases()
    name_code = build_name_code_map()
    cases["股票代码"] = cases["股票名称"].map(name_code)
    valid_cases = cases[cases["股票代码"].notna()].copy()
    case_dates = sorted(valid_cases["信号日期"].drop_duplicates().tolist())

    pool = build_case_date_pool(case_dates)
    selected = pool[(pool["score_pct_rank"] >= brick_filter.PCT_RANK_THRESHOLD) & (pool["daily_rank"] <= brick_filter.TOP_N)].copy()

    merged = valid_cases.merge(pool, how="left", left_on=["信号日期", "股票代码"], right_on=["date", "code"])
    merged["命中信号池"] = merged["code"].notna()
    selected_keys = set(zip(selected["date"].dt.strftime("%Y-%m-%d"), selected["code"]))
    merged["命中最终筛选"] = [
        (d.strftime("%Y-%m-%d"), c) in selected_keys if pd.notna(c) else False
        for d, c in zip(merged["信号日期"], merged["股票代码"])
    ]
    merged["未命中原因"] = np.where(
        ~merged["命中信号池"],
        "未进入信号池",
        np.where(~merged["命中最终筛选"], "进入信号池但未进前50%或前10", ""),
    )

    case_features = merged[merged["命中信号池"]].copy()
    compare_numeric = [
        "rebound_ratio",
        "signal_vs_ma5",
        "signal_vs_ma10",
        "vol_vs_prev",
        "vol_rank_20",
        "pullback_shrink_ratio",
        "ret1",
        "trend_spread",
        "close_position",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "body_ratio",
        "trend_slope_3",
        "trend_slope_5",
        "long_slope_3",
        "long_slope_5",
        "trend_long_slope_gap_5",
        "dist_trend",
        "dist_long",
        "dist_20d_high",
        "dist_20d_low",
        "ret_5d_before",
        "ret_10d_before",
        "前60日倍量阳柱个数",
        "距倍量柱最高价",
        "距倍量柱收盘价",
        "距倍量柱最低价",
    ]
    compare_bool = [
        "回踩趋势线",
        "回踩多空线",
        "趋势线支撑",
        "多空线支撑",
        "沿趋势线上涨",
        "前60日存在倍量阳柱",
        "站上倍量柱最高价",
        "站上倍量柱收盘价",
        "站上倍量柱最低价",
    ]

    compare_rows: List[dict] = []
    for col in compare_numeric:
        compare_rows.append(
            {
                "指标": col,
                "类型": "数值",
                "案例中位数": float(case_features[col].median()) if not case_features.empty else np.nan,
                "同期信号池中位数": float(pool[col].median()) if not pool.empty else np.nan,
                "案例均值": float(case_features[col].mean()) if not case_features.empty else np.nan,
                "同期信号池均值": float(pool[col].mean()) if not pool.empty else np.nan,
            }
        )
    for col in compare_bool:
        compare_rows.append(
            {
                "指标": col,
                "类型": "布尔",
                "案例命中率": float(case_features[col].mean()) if not case_features.empty else np.nan,
                "同期信号池命中率": float(pool[col].mean()) if not pool.empty else np.nan,
            }
        )
    compare_df = pd.DataFrame(compare_rows)

    summary = {
        "案例文件总数": int(len(cases)),
        "可解析案例数": int(len(valid_cases)),
        "命中信号池数": int(merged["命中信号池"].sum()) if not merged.empty else 0,
        "命中最终筛选数": int(merged["命中最终筛选"].sum()) if not merged.empty else 0,
        "案例日期数": len(case_dates),
        "案例日期列表": [d.strftime("%Y-%m-%d") for d in case_dates],
    }

    merged.to_csv(OUT_DIR / "case_match_results.csv", index=False, encoding="utf-8-sig")
    pool.to_csv(OUT_DIR / "case_date_signal_pool.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "case_date_selected.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(OUT_DIR / "case_vs_same_date_pool_compare.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
