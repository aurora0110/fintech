from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


INPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260226/normal")
MIN_BARS = 160
EPS = 1e-12
TOP_N = 10
PCT_RANK_THRESHOLD = 0.50

DATE_COL_CANDIDATES = ["date", "Date", "trade_date", "日期", "DATE"]
OPEN_COL_CANDIDATES = ["open", "Open", "开盘", "OPEN"]
HIGH_COL_CANDIDATES = ["high", "High", "最高", "HIGH"]
LOW_COL_CANDIDATES = ["low", "Low", "最低", "LOW"]
CLOSE_COL_CANDIDATES = ["close", "Close", "收盘", "CLOSE"]
VOL_COL_CANDIDATES = ["volume", "vol", "Volume", "成交量", "VOL"]
CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码", "CODE"]


def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > EPS)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    return series.ewm(alpha=m / n, adjust=False).mean()


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"缺少字段，候选字段={candidates}")
    return None


def read_csv_auto(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+|\t+", engine="python")


def load_one_csv(path: str) -> Optional[pd.DataFrame]:
    raw = read_csv_auto(path)
    date_col = pick_col(raw, DATE_COL_CANDIDATES)
    open_col = pick_col(raw, OPEN_COL_CANDIDATES)
    high_col = pick_col(raw, HIGH_COL_CANDIDATES)
    low_col = pick_col(raw, LOW_COL_CANDIDATES)
    close_col = pick_col(raw, CLOSE_COL_CANDIDATES)
    vol_col = pick_col(raw, VOL_COL_CANDIDATES)
    code_col = pick_col(raw, CODE_COL_CANDIDATES, required=False)

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
    if code_col:
        df["code"] = raw[code_col].astype(str).iloc[0]
    else:
        df["code"] = os.path.splitext(os.path.basename(path))[0]
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)].copy()
    if len(df) < MIN_BARS:
        return None
    return df


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


def triangle_quality(value: float, center: float, half_width: float) -> float:
    if not np.isfinite(value) or half_width <= 0:
        return 0.0
    return float(max(1.0 - abs(value - center) / half_width, 0.0))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x["ret1"] = x["close"].pct_change()
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["close_pullback_white"] = x["close"] < x["trend_line"] * 1.01
    x["close_above_white"] = x["close"] > x["trend_line"]

    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0

    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5_prev"])
    x["signal_vs_ma5_valid"] = x["signal_vs_ma5"].between(1, 2.2, inclusive="both")

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

    x["rebound_ratio"] = safe_div(x["brick_red_len"], x["brick_green_len"].shift(1))
    x["signal_base"] = (
        (x["pattern_a"] | x["pattern_b"])
        & x["pullback_shrinking"].fillna(False)
        & x["signal_vs_ma5_valid"].fillna(False)
        & x["not_sideways"].fillna(False)
        & x["ret1"].notna()
    )
    return x


def build_signal_df(input_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".csv", ".txt"}])
    total = len(files)
    for idx, path in enumerate(files, 1):
        df = load_one_csv(str(path))
        if df is None or df.empty:
            continue
        code = str(df["code"].iloc[0])
        x = add_features(df)
        mask_a = x["pattern_a"] & (x["rebound_ratio"] >= 1.2)
        mask_b = x["pattern_b"] & (x["rebound_ratio"] >= 1.0)
        mask = (
            x["signal_base"]
            & (x["ret1"] <= 0.08)
            & (mask_a | mask_b)
            & (x["trend_line"] > x["long_line"])
        )
        signal_idxs = np.flatnonzero(mask.to_numpy())
        for signal_idx in signal_idxs:
            close = float(x.at[int(signal_idx), "close"])
            trend_line = float(x.at[int(signal_idx), "trend_line"])
            long_line = float(x.at[int(signal_idx), "long_line"])
            ret1 = float(x.at[int(signal_idx), "ret1"])
            signal_vs_ma5 = float(x.at[int(signal_idx), "signal_vs_ma5"])
            rebound_ratio = float(x.at[int(signal_idx), "rebound_ratio"])
            pullback_avg_vol = float(x.at[int(signal_idx), "pullback_avg_vol"])
            up_leg_avg_vol = float(x.at[int(signal_idx), "up_leg_avg_vol"])
            pullback_shrink_ratio = pullback_avg_vol / up_leg_avg_vol if np.isfinite(up_leg_avg_vol) and up_leg_avg_vol > 0 else np.nan
            trend_spread_clip = max((trend_line - long_line) / close, 0.0) if np.isfinite(close) and close > 0 else 0.0
            rows.append(
                {
                    "date": x.at[int(signal_idx), "date"],
                    "code": code,
                    "signal_close": close,
                    "signal_open": float(x.at[int(signal_idx), "open"]),
                    "signal_high": float(x.at[int(signal_idx), "high"]),
                    "signal_low": float(x.at[int(signal_idx), "low"]),
                    "signal_volume": float(x.at[int(signal_idx), "volume"]),
                    "rebound_ratio": rebound_ratio,
                    "pullback_shrink_ratio": pullback_shrink_ratio,
                    "trend_spread_clip": trend_spread_clip,
                    "signal_vs_ma5": signal_vs_ma5,
                    "ret1_quality": triangle_quality(ret1, 0.03, 0.03),
                    "signal_vs_ma5_quality": triangle_quality(signal_vs_ma5, 1.7, 0.5),
                    "shrink_quality": triangle_quality(pullback_shrink_ratio, 0.8, 0.3),
                    "pattern_a": bool(x.at[int(signal_idx), "pattern_a"]),
                    "pattern_b": bool(x.at[int(signal_idx), "pattern_b"]),
                }
            )
        if idx % 500 == 0 or idx == total:
            print(f"特征进度: {idx}/{total}")

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)
    out["rebound_rank"] = out.groupby("date")["rebound_ratio"].rank(pct=True)
    out["trend_rank"] = out.groupby("date")["trend_spread_clip"].rank(pct=True)
    out["shrink_rank"] = out.groupby("date")["shrink_quality"].rank(pct=True)
    out["sort_score"] = 0.50 * out["shrink_rank"] + 0.30 * out["rebound_rank"] + 0.20 * out["trend_rank"]
    out["score_pct_rank"] = out.groupby("date")["sort_score"].rank(pct=True)
    out["daily_rank"] = out.sort_values(["date", "sort_score", "code"], ascending=[True, False, True]).groupby("date").cumcount() + 1
    return out


def apply_selection(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df
    x = signal_df[signal_df["score_pct_rank"] >= PCT_RANK_THRESHOLD].copy()
    x = x.sort_values(["date", "sort_score", "code"], ascending=[True, False, True])
    x["daily_rank"] = x.groupby("date").cumcount() + 1
    return x.groupby("date", group_keys=False).head(TOP_N).reset_index(drop=True)


def check(file_path, hold_list=None):
    df = load_one_csv(str(file_path))
    if df is None or df.empty:
        return [-1]
    x = add_features(df)
    latest_idx = len(x) - 1
    if latest_idx < 0:
        return [-1]
    latest = x.iloc[latest_idx]
    mask_a = bool(latest["pattern_a"]) and float(latest["rebound_ratio"]) >= 1.2
    mask_b = bool(latest["pattern_b"]) and float(latest["rebound_ratio"]) >= 1.0
    signal_ok = (
        bool(latest["signal_base"])
        and float(latest["ret1"]) <= 0.08
        and (mask_a or mask_b)
        and float(latest["trend_line"]) > float(latest["long_line"])
    )
    if not signal_ok:
        return [-1]
    pullback_avg_vol = float(latest["pullback_avg_vol"])
    up_leg_avg_vol = float(latest["up_leg_avg_vol"])
    pullback_shrink_ratio = pullback_avg_vol / up_leg_avg_vol if np.isfinite(up_leg_avg_vol) and up_leg_avg_vol > 0 else np.nan
    sort_score = (
        0.50 * triangle_quality(pullback_shrink_ratio, 0.8, 0.3)
        + 0.30 * 1.0
        + 0.20 * 1.0
    )
    stop_loss_price = round(float(latest["low"]) * 0.99, 3)
    return [1, stop_loss_price, float(latest["close"]), round(sort_score, 4), "brick动量续冲"]


def main() -> None:
    signal_df = build_signal_df(INPUT_DIR)
    selected_df = apply_selection(signal_df)
    if selected_df.empty:
        return
    latest_trade_date = pd.to_datetime(selected_df["date"]).max()
    latest_df = selected_df[pd.to_datetime(selected_df["date"]) == latest_trade_date].copy()
    print(latest_df[["date", "code", "daily_rank", "sort_score", "rebound_ratio", "signal_vs_ma5", "pullback_shrink_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
