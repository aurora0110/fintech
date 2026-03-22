from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb


INPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260226/normal")
MODEL_FILE = Path("/Users/lidongyang/Desktop/Qstrategy/models/lgbm_p3_shallower_core10_daily_top9.txt")

MIN_BARS = 160
EPS = 1e-12
TOP_N = 9

DATE_COL_CANDIDATES = ["date", "Date", "trade_date", "日期", "DATE"]
OPEN_COL_CANDIDATES = ["open", "Open", "开盘", "OPEN"]
HIGH_COL_CANDIDATES = ["high", "High", "最高", "HIGH"]
LOW_COL_CANDIDATES = ["low", "Low", "最低", "LOW"]
CLOSE_COL_CANDIDATES = ["close", "Close", "收盘", "CLOSE"]
VOL_COL_CANDIDATES = ["volume", "vol", "Volume", "成交量", "VOL"]
CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码", "CODE"]

# 最终定版主策略核心特征
CORE10_FEATURE_COLS = [
    "sim_euclidean",
    "sim_rank_today",
    "ret1",
    "ret5",
    "trend_spread",
    "ma20_slope_5",
    "close_to_long",
    "brick_green_len_prev",
    "brick_red_len",
    "signal_ret",
]


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
    df = df[
        (df["open"] > 0) &
        (df["high"] > 0) &
        (df["low"] > 0) &
        (df["close"] > 0) &
        (df["volume"] >= 0)
    ].copy()

    if len(df) < MIN_BARS:
        return None
    return df


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


def compute_slope(series: pd.Series, window: int) -> pd.Series:
    def _slope(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) < window or np.any(np.isnan(arr)):
            return np.nan
        x = np.arange(window)
        slope, _ = np.polyfit(x, arr, 1)
        return slope
    return series.rolling(window).apply(_slope, raw=False)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)

    x["ret1"] = x["close"].pct_change()
    x["ret5"] = x["close"].pct_change(5)
    x["ret10"] = x["close"].pct_change(10)
    x["ret20"] = x["close"].pct_change(20)

    x["signal_ret"] = safe_div(x["close"] - x["open"], x["open"], default=np.nan)

    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["close_pullback_white"] = x["close"] < x["trend_line"] * 1.01
    x["close_above_white"] = x["close"] > x["trend_line"]

    x["ma10"] = x["close"].rolling(10).mean()
    x["ma20"] = x["close"].rolling(20).mean()

    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0

    x["ma20_slope_5"] = compute_slope(x["ma20"], 5)

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

    x["trend_spread"] = safe_div(x["trend_line"] - x["long_line"], x["close"], default=np.nan)
    x["close_to_trend"] = safe_div(x["close"] - x["trend_line"], x["trend_line"], default=np.nan)
    x["close_to_long"] = safe_div(x["close"] - x["long_line"], x["long_line"], default=np.nan)

    return x


def build_daily_signal_df(input_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".csv", ".txt"}])

    model = lgb.Booster(model_file=str(MODEL_FILE))

    total = len(files)
    for idx, path in enumerate(files, 1):
        df = load_one_csv(str(path))
        if df is None or df.empty:
            continue

        code = str(df["code"].iloc[0])
        x = add_features(df)

        latest_idx = len(x) - 1
        if latest_idx < 20:
            continue

        latest = x.iloc[latest_idx]

        mask_a = bool(latest["pattern_a"]) and float(latest["rebound_ratio"]) >= 0.8
        mask_b = bool(latest["pattern_b"]) and float(latest["rebound_ratio"]) >= 1.0

        signal_ok = (
            bool(latest["signal_base"])
            and float(latest["ret1"]) <= 0.08
            and (mask_a or mask_b)
            and float(latest["trend_line"]) > float(latest["long_line"])
            and (-0.03 <= float(latest["ret1"]) <= 0.11)
        )
        if not signal_ok:
            continue

        # 先把除 sim_euclidean / sim_rank_today 外的 core10 特征准备好
        row = {
            "date": latest["date"],
            "code": code,
            "signal_close": float(latest["close"]),
            "signal_open": float(latest["open"]),
            "signal_high": float(latest["high"]),
            "signal_low": float(latest["low"]),
            "signal_volume": float(latest["volume"]),
            "pattern_a": bool(latest["pattern_a"]),
            "pattern_b": bool(latest["pattern_b"]),
            "rebound_ratio": float(latest["rebound_ratio"]) if pd.notna(latest["rebound_ratio"]) else np.nan,
            "signal_vs_ma5": float(latest["signal_vs_ma5"]) if pd.notna(latest["signal_vs_ma5"]) else np.nan,
            "pullback_shrink_ratio": (
                float(latest["pullback_avg_vol"]) / float(latest["up_leg_avg_vol"])
                if pd.notna(latest["pullback_avg_vol"]) and pd.notna(latest["up_leg_avg_vol"]) and float(latest["up_leg_avg_vol"]) > 0
                else np.nan
            ),
            "ret1": float(latest["ret1"]) if pd.notna(latest["ret1"]) else np.nan,
            "ret5": float(latest["ret5"]) if pd.notna(latest["ret5"]) else np.nan,
            "trend_spread": float(latest["trend_spread"]) if pd.notna(latest["trend_spread"]) else np.nan,
            "ma20_slope_5": float(latest["ma20_slope_5"]) if pd.notna(latest["ma20_slope_5"]) else np.nan,
            "close_to_long": float(latest["close_to_long"]) if pd.notna(latest["close_to_long"]) else np.nan,
            "brick_green_len_prev": float(x["brick_green_len"].shift(1).iloc[latest_idx]) if latest_idx >= 1 else np.nan,
            "brick_red_len": float(latest["brick_red_len"]) if pd.notna(latest["brick_red_len"]) else np.nan,
            "signal_ret": float(latest["signal_ret"]) if pd.notna(latest["signal_ret"]) else np.nan,
        }
        rows.append(row)

        if idx % 500 == 0 or idx == total:
            print(f"扫描进度: {idx}/{total}")

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)

    # ============================================================
    # 这里是为了适配主策略：
    # 由于盘中/当日扫描时无法拿到真正“历史成功模板相似度”，
    # 这里先用当前候选池做一个横向 proxy：
    # sim_euclidean 先用趋势+节奏+砖型的综合标准化得分近似，
    # sim_rank_today 再按日内候选排名。
    #
    # 如果你后面要做到完全一致版本，
    # 我可以再给你接入“历史模板库 + 真正欧氏相似度”的版本。
    # ============================================================
    z_cols = ["ret1", "ret5", "trend_spread", "ma20_slope_5", "close_to_long", "brick_green_len_prev", "brick_red_len", "signal_ret"]
    for c in z_cols:
        out[f"{c}_z"] = out.groupby("date")[c].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) and np.isfinite(s.std(ddof=0)) and s.std(ddof=0) > 1e-12 else 0.0
        )

    out["sim_euclidean"] = (
        0.20 * out["ret1_z"]
        + 0.20 * out["ret5_z"]
        + 0.18 * out["trend_spread_z"]
        + 0.12 * out["ma20_slope_5_z"]
        + 0.10 * out["close_to_long_z"]
        + 0.08 * out["brick_green_len_prev_z"]
        + 0.07 * out["brick_red_len_z"]
        + 0.05 * out["signal_ret_z"]
    )

    out["sim_euclidean"] = out.groupby("date")["sim_euclidean"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if (s.max() - s.min()) > 1e-12 else 0.5
    )

    out = out.sort_values(["date", "sim_euclidean", "code"], ascending=[True, False, True]).reset_index(drop=True)
    out["sim_rank_today"] = out.groupby("date").cumcount() + 1

    # LightGBM 最终主策略打分
    X = out[CORE10_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["sort_score"] = model.predict(X)

    out = out.sort_values(["date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    out["daily_rank"] = out.groupby("date").cumcount() + 1

    return out


def apply_selection(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df.copy()
    x = signal_df.sort_values(["date", "sort_score", "code"], ascending=[True, False, True]).copy()
    x["daily_rank"] = x.groupby("date").cumcount() + 1
    return x.groupby("date", group_keys=False).head(TOP_N).reset_index(drop=True)


def check(file_path, hold_list=None):
    model = lgb.Booster(model_file=str(MODEL_FILE))

    df = load_one_csv(str(file_path))
    if df is None or df.empty:
        return [-1]

    x = add_features(df)
    latest_idx = len(x) - 1
    if latest_idx < 20:
        return [-1]

    latest = x.iloc[latest_idx]

    mask_a = bool(latest["pattern_a"]) and float(latest["rebound_ratio"]) >= 0.8
    mask_b = bool(latest["pattern_b"]) and float(latest["rebound_ratio"]) >= 1.0

    signal_ok = (
        bool(latest["signal_base"])
        and float(latest["ret1"]) <= 0.08
        and (mask_a or mask_b)
        and float(latest["trend_line"]) > float(latest["long_line"])
        and (-0.03 <= float(latest["ret1"]) <= 0.11)
    )
    if not signal_ok:
        return [-1]

    # 单票 check 时 sim_rank_today 无法知道当天全市场候选中的真实名次
    # 这里临时给一个中性值；系统日批处理时应以 build_daily_signal_df 为准
    feature_row = pd.DataFrame([{
        "sim_euclidean": 0.5,
        "sim_rank_today": 5,
        "ret1": float(latest["ret1"]) if pd.notna(latest["ret1"]) else 0.0,
        "ret5": float(latest["ret5"]) if pd.notna(latest["ret5"]) else 0.0,
        "trend_spread": float(latest["trend_spread"]) if pd.notna(latest["trend_spread"]) else 0.0,
        "ma20_slope_5": float(latest["ma20_slope_5"]) if pd.notna(latest["ma20_slope_5"]) else 0.0,
        "close_to_long": float(latest["close_to_long"]) if pd.notna(latest["close_to_long"]) else 0.0,
        "brick_green_len_prev": float(x["brick_green_len"].shift(1).iloc[latest_idx]) if latest_idx >= 1 and pd.notna(x["brick_green_len"].shift(1).iloc[latest_idx]) else 0.0,
        "brick_red_len": float(latest["brick_red_len"]) if pd.notna(latest["brick_red_len"]) else 0.0,
        "signal_ret": float(latest["signal_ret"]) if pd.notna(latest["signal_ret"]) else 0.0,
    }])

    sort_score = float(model.predict(feature_row[CORE10_FEATURE_COLS])[0])
    stop_loss_price = round(float(latest["low"]) * 0.99, 3)

    return [1, stop_loss_price, float(latest["close"]), round(sort_score, 6), "lgbm_core10_top9"]


def main() -> None:
    signal_df = build_daily_signal_df(INPUT_DIR)
    selected_df = apply_selection(signal_df)

    if selected_df.empty:
        print("当日无候选信号")
        return

    latest_trade_date = pd.to_datetime(selected_df["date"]).max()
    latest_df = selected_df[pd.to_datetime(selected_df["date"]) == latest_trade_date].copy()

    show_cols = [
        "date",
        "code",
        "daily_rank",
        "sort_score",
        "sim_euclidean",
        "sim_rank_today",
        "ret1",
        "ret5",
        "trend_spread",
        "ma20_slope_5",
        "close_to_long",
        "brick_green_len_prev",
        "brick_red_len",
        "signal_ret",
        "rebound_ratio",
        "signal_vs_ma5",
        "pullback_shrink_ratio",
    ]
    print(latest_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()