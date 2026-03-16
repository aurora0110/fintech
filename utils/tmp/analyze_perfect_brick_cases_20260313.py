from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RAW_DIR = ROOT / "data" / "20260313"
NORMAL_DIR = RAW_DIR / "normal"
CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
OUT_DIR = ROOT / "results" / "perfect_brick_case_analysis_20260313"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > 1e-12)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    return series.ewm(alpha=m / n, adjust=False).mean()


def read_daily_file(path: Path) -> pd.DataFrame | None:
    try:
        raw = pd.read_csv(path)
        if raw.shape[1] == 1:
            raw = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
    except Exception:
        try:
            raw = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
        except Exception:
            return None

    cols = list(raw.columns)
    if not cols:
        return None
    date_col = cols[0]
    if len(cols) < 6:
        return None
    open_col, high_col, low_col, close_col, vol_col = cols[1:6]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "open": pd.to_numeric(raw[open_col], errors="coerce"),
            "high": pd.to_numeric(raw[high_col], errors="coerce"),
            "low": pd.to_numeric(raw[low_col], errors="coerce"),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
            "volume": pd.to_numeric(raw[vol_col], errors="coerce"),
        }
    )
    df = df.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)]
    if len(df) < 160:
        return None
    return df


def parse_cases() -> pd.DataFrame:
    pat = re.compile(r"(.+?)(\d{8})\.png$")
    rows: List[dict] = []
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
    for path in RAW_DIR.glob("*.txt"):
        try:
            with open(path, "r", encoding="gbk", errors="ignore") as f:
                first = f.readline().strip()
        except Exception:
            continue
        parts = first.split()
        if len(parts) >= 2 and parts[0].isdigit():
            mapping[parts[1]] = path.stem
    return mapping


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


def add_features(df: pd.DataFrame, code: str) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x["code"] = code
    x["ret1"] = x["close"].pct_change()
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0
    x["trend_ok"] = x["trend_line"] > x["long_line"]
    x["close_pullback_white"] = x["close"] < x["trend_line"] * 1.01
    x["close_above_white"] = x["close"] > x["trend_line"]

    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = pd.Series(safe_div(x["volume"], x["vol_ma5_prev"]), index=x.index)

    hhv4 = x["high"].rolling(4).max()
    llv4 = x["low"].rolling(4).min()
    den4 = (hhv4 - llv4).replace(0, np.nan)
    var1a = safe_div((hhv4 - x["close"]), den4) * 100 - 90
    var2a = tdx_sma(pd.Series(var1a, index=x.index), 4, 1) + 100
    var3a = safe_div((x["close"] - llv4), den4) * 100
    var4a = tdx_sma(pd.Series(var3a, index=x.index), 6, 1)
    var5a = tdx_sma(var4a, 6, 1) + 100
    var6a = var5a - var2a
    x["brick"] = np.where(var6a > 4, var6a - 4, 0.0)
    x["brick_prev"] = x["brick"].shift(1)
    x["brick_red_len"] = np.where(x["brick"] > x["brick_prev"], x["brick"] - x["brick_prev"], 0.0)
    x["brick_green_len"] = np.where(x["brick"] < x["brick_prev"], x["brick_prev"] - x["brick"], 0.0)
    x["brick_red"] = x["brick_red_len"] > 0
    x["brick_green"] = x["brick_green_len"] > 0
    x["prev_green_streak"] = pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(1)

    x["close_slope_10"] = (
        x["close"]
        .rolling(10)
        .apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if np.isfinite(s).all() else np.nan, raw=False)
    )
    x["not_sideways"] = np.abs(safe_div(x["close_slope_10"], x["close"].rolling(10).mean())) > 0.002

    x["up_leg_avg_vol"] = x["volume"].shift(4).rolling(3).mean()
    x["pullback_avg_vol"] = x["volume"].shift(1).rolling(3).mean()
    x["pullback_shrinking"] = x["pullback_avg_vol"] < x["up_leg_avg_vol"]
    x["pullback_shrink_ratio"] = pd.Series(safe_div(x["pullback_avg_vol"], x["up_leg_avg_vol"]), index=x.index)

    x["pattern_a"] = (
        (x["prev_green_streak"] >= 3)
        & x["brick_red"]
        & x["close_pullback_white"].shift(1).fillna(False)
        & x["close_above_white"]
    )
    x["pattern_b"] = (
        (pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(3) >= 3)
        & x["brick_red"]
        & x["brick_green"].shift(1).fillna(False)
        & x["brick_red"].shift(2).fillna(False)
        & x["close_pullback_white"].shift(1).fillna(False)
        & x["close_above_white"]
    )
    x["rebound_ratio"] = pd.Series(safe_div(x["brick_red_len"], x["brick_green_len"].shift(1)), index=x.index)

    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    real_body = (x["close"] - x["open"]).abs()
    upper_shadow = x["high"] - np.maximum(x["open"], x["close"])
    lower_shadow = np.minimum(x["open"], x["close"]) - x["low"]
    x["close_position"] = pd.Series(safe_div(x["close"] - x["low"], full_range), index=x.index)
    x["upper_shadow_ratio"] = pd.Series(safe_div(upper_shadow, full_range), index=x.index)
    x["lower_shadow_ratio"] = pd.Series(safe_div(lower_shadow, full_range), index=x.index)
    x["body_ratio"] = pd.Series(safe_div(real_body, full_range), index=x.index)

    x["signal_vs_ma10"] = pd.Series(safe_div(x["volume"], x["volume"].shift(1).rolling(10).mean()), index=x.index)
    x["vol_vs_prev"] = pd.Series(safe_div(x["volume"], x["volume"].shift(1)), index=x.index)
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

    prev_low = x["low"].shift(1)
    prev_close = x["close"].shift(1)
    prev_trend = x["trend_line"].shift(1)
    prev_long = x["long_line"].shift(1)
    x["回踩趋势线"] = (prev_low <= prev_trend * 1.01) & (prev_close >= prev_trend * 0.99)
    x["回踩多空线"] = (prev_low <= prev_long * 1.01) & (prev_close >= prev_long * 0.99)
    x["趋势线支撑"] = (x["close"] > x["trend_line"]) & (x["dist_trend"] <= 0.03)
    x["多空线支撑"] = (x["close"] > x["long_line"]) & (x["dist_long"] <= 0.05)
    x["沿趋势线上涨"] = (
        (x["close"] > x["trend_line"])
        & (x["close"].shift(1) > prev_trend)
        & (x["trend_slope_5"] > 0)
        & (x["dist_trend"].between(0.0, 0.03))
    )

    x["倍量阳柱"] = (x["close"] > x["open"]) & (x["volume"] >= x["volume"].shift(1) * 2.0)
    x["前60日倍量阳柱个数"] = x["倍量阳柱"].rolling(60).sum()

    last_high = np.full(len(x), np.nan)
    last_low = np.full(len(x), np.nan)
    last_close = np.full(len(x), np.nan)
    exist = np.zeros(len(x), dtype=bool)
    flag = x["倍量阳柱"].to_numpy(dtype=bool)
    highs = x["high"].to_numpy(dtype=float)
    lows = x["low"].to_numpy(dtype=float)
    closes = x["close"].to_numpy(dtype=float)
    for i in range(len(x)):
        start = max(0, i - 60)
        idx = np.flatnonzero(flag[start:i])
        if idx.size == 0:
            continue
        j = idx[-1] + start
        last_high[i] = highs[j]
        last_low[i] = lows[j]
        last_close[i] = closes[j]
        exist[i] = True
    x["前60日存在倍量阳柱"] = exist
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


def is_signal(row: pd.Series) -> bool:
    mask_a = bool(row["pattern_a"]) and float(row["rebound_ratio"]) >= 1.2
    mask_b = bool(row["pattern_b"]) and float(row["rebound_ratio"]) >= 1.0
    return (
        (mask_a or mask_b)
        and bool(row["pullback_shrinking"])
        and pd.notna(row["signal_vs_ma5"])
        and 1.3 <= float(row["signal_vs_ma5"]) <= 2.2
        and bool(row["not_sideways"])
        and pd.notna(row["ret1"])
        and float(row["ret1"]) <= 0.08
        and float(row["trend_line"]) > float(row["long_line"])
    )


def main() -> None:
    cases = parse_cases()
    name_code = build_name_code_map()
    cases["股票代码"] = cases["股票名称"].map(name_code)
    cases = cases[cases["股票代码"].notna()].copy()
    case_dates = sorted(cases["信号日期"].drop_duplicates().tolist())

    pool_rows: List[dict] = []
    total = len(list(NORMAL_DIR.glob("*.txt")))
    for idx, path in enumerate(sorted(NORMAL_DIR.glob("*.txt")), 1):
        if idx % 500 == 0 or idx == total:
            print(f"特征进度: {idx}/{total}")
        df = read_daily_file(path)
        if df is None or df.empty:
            continue
        x = add_features(df, path.stem)
        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(x["date"])}
        for d in case_dates:
            i = date_to_idx.get(pd.Timestamp(d))
            if i is None:
                continue
            row = x.iloc[i]
            if not is_signal(row):
                continue
            record = row.to_dict()
            pool_rows.append(record)

    pool = pd.DataFrame(pool_rows)
    if pool.empty:
        raise SystemExit("案例日期对应的信号池为空")
    pool = pool.sort_values(["date", "code"]).reset_index(drop=True)
    pool["rebound_rank"] = pool.groupby("date")["rebound_ratio"].rank(pct=True)
    pool["trend_rank"] = pool.groupby("date")["dist_trend"].rank(pct=True)
    pool["shrink_quality"] = 1.0 - np.minimum(np.abs(pool["pullback_shrink_ratio"] - 0.8) / 0.3, 1.0)
    pool["shrink_rank"] = pool.groupby("date")["shrink_quality"].rank(pct=True)
    pool["sort_score"] = 0.50 * pool["shrink_rank"] + 0.30 * pool["rebound_rank"] + 0.20 * pool["trend_rank"]
    pool["score_pct_rank"] = pool.groupby("date")["sort_score"].rank(pct=True)
    pool = pool.sort_values(["date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    pool["daily_rank"] = pool.groupby("date").cumcount() + 1

    selected = pool[(pool["score_pct_rank"] >= 0.50) & (pool["daily_rank"] <= 10)].copy()
    selected_keys = set(zip(selected["date"].dt.strftime("%Y-%m-%d"), selected["code"]))

    merged = cases.merge(pool, how="left", left_on=["信号日期", "股票代码"], right_on=["date", "code"])
    merged["命中信号池"] = merged["code"].notna()
    merged["命中最终筛选"] = [
        (d.strftime("%Y-%m-%d"), c) in selected_keys if pd.notna(c) else False
        for d, c in zip(merged["信号日期"], merged["股票代码"])
    ]
    merged["未命中原因"] = np.where(
        ~merged["命中信号池"],
        "未进入信号池",
        np.where(~merged["命中最终筛选"], "进入信号池但未进前50%或前10", ""),
    )

    num_cols = [
        "rebound_ratio",
        "signal_vs_ma5",
        "signal_vs_ma10",
        "vol_vs_prev",
        "vol_rank_20",
        "pullback_shrink_ratio",
        "ret1",
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
    bool_cols = [
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

    case_feat = merged[merged["命中信号池"]].copy()
    compare_rows: List[dict] = []
    for col in num_cols:
        compare_rows.append(
            {
                "指标": col,
                "类型": "数值",
                "案例中位数": float(case_feat[col].median()) if not case_feat.empty else np.nan,
                "同期信号池中位数": float(pool[col].median()),
                "案例均值": float(case_feat[col].mean()) if not case_feat.empty else np.nan,
                "同期信号池均值": float(pool[col].mean()),
            }
        )
    for col in bool_cols:
        compare_rows.append(
            {
                "指标": col,
                "类型": "布尔",
                "案例命中率": float(case_feat[col].mean()) if not case_feat.empty else np.nan,
                "同期信号池命中率": float(pool[col].mean()),
            }
        )
    compare = pd.DataFrame(compare_rows)

    merged.to_csv(OUT_DIR / "case_match_results.csv", index=False, encoding="utf-8-sig")
    pool.to_csv(OUT_DIR / "case_date_signal_pool.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "case_date_selected.csv", index=False, encoding="utf-8-sig")
    compare.to_csv(OUT_DIR / "case_vs_pool_compare.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "案例数": int(len(cases)),
                "命中信号池数": int(merged["命中信号池"].sum()),
                "命中最终筛选数": int(merged["命中最终筛选"].sum()),
                "案例日期数": len(case_dates),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"案例数": int(len(cases)), "命中信号池数": int(merged["命中信号池"].sum()), "命中最终筛选数": int(merged["命中最终筛选"].sum())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
