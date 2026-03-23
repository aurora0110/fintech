from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb


INPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260226/normal")
MODEL_FILE = Path("/Users/lidongyang/Desktop/Qstrategy/models/lgbm_p3_shallower_core10_daily_top9.txt")
HOLDINGS_FILE = Path("/Users/lidongyang/Desktop/Qstrategy/data/current_holdings.csv")

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

HOLDING_CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码", "CODE"]
HOLDING_BUY_PRICE_COL_CANDIDATES = ["buy_price", "entry_price", "成本价", "买入价"]
HOLDING_BUY_DATE_COL_CANDIDATES = ["buy_date", "entry_date", "买入日期"]
HOLDING_SHARES_COL_CANDIDATES = ["shares", "qty", "数量", "持仓数量"]

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

# 卖出参数
STOP_LOSS_PCT = 0.99
TAKE_PROFIT_PCT = 1.035


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
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ].copy()

    if len(df) < MIN_BARS:
        return None
    return df


def load_holdings(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=["code", "buy_price", "buy_date", "shares"])

    raw = read_csv_auto(str(file_path))
    code_col = pick_col(raw, HOLDING_CODE_COL_CANDIDATES)
    buy_price_col = pick_col(raw, HOLDING_BUY_PRICE_COL_CANDIDATES, required=False)
    buy_date_col = pick_col(raw, HOLDING_BUY_DATE_COL_CANDIDATES, required=False)
    shares_col = pick_col(raw, HOLDING_SHARES_COL_CANDIDATES, required=False)

    out = pd.DataFrame()
    out["code"] = raw[code_col].astype(str).str.extract(r"(\d{6})", expand=False).fillna(raw[code_col].astype(str))

    if buy_price_col:
        out["buy_price"] = pd.to_numeric(raw[buy_price_col], errors="coerce")
    else:
        out["buy_price"] = np.nan

    if buy_date_col:
        out["buy_date"] = pd.to_datetime(raw[buy_date_col], errors="coerce")
    else:
        out["buy_date"] = pd.NaT

    if shares_col:
        out["shares"] = pd.to_numeric(raw[shares_col], errors="coerce")
    else:
        out["shares"] = np.nan

    out = out.drop_duplicates(subset=["code"]).reset_index(drop=True)
    return out


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

    prev_close_pullback = x["close_pullback_white"].shift(1).fillna(False).astype(bool)
    prev_brick_green = x["brick_green"].shift(1).fillna(False).astype(bool)
    prev2_brick_red = x["brick_red"].shift(2).fillna(False).astype(bool)

    x["pattern_a"] = (
        (x["prev_green_streak"] >= 3)
        & x["brick_red"]
        & prev_close_pullback
        & x["close_above_white"]
    )
    x["pattern_b"] = (
        (pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(3) >= 3)
        & x["brick_red"]
        & prev_brick_green
        & prev2_brick_red
        & prev_close_pullback
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

    # 卖出判定辅助字段
    x["close_below_long"] = (x["close"] < x["long_line"]).fillna(False)
    x["close_below_trend"] = (x["close"] < x["trend_line"]).fillna(False)

    return x


def evaluate_buy_signal(latest: pd.Series) -> bool:
    mask_a = bool(latest["pattern_a"]) and float(latest["rebound_ratio"]) >= 0.8
    mask_b = bool(latest["pattern_b"]) and float(latest["rebound_ratio"]) >= 1.0

    signal_ok = (
        bool(latest["signal_base"])
        and float(latest["ret1"]) <= 0.08
        and (mask_a or mask_b)
        and float(latest["trend_line"]) > float(latest["long_line"])
        and (-0.03 <= float(latest["ret1"]) <= 0.11)
    )
    return signal_ok


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
        if not evaluate_buy_signal(latest):
            continue

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
            print(f"买入扫描进度: {idx}/{total}")

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)

    z_cols = [
        "ret1", "ret5", "trend_spread", "ma20_slope_5",
        "close_to_long", "brick_green_len_prev", "brick_red_len", "signal_ret"
    ]
    for c in z_cols:
        out[f"{c}_z"] = out.groupby("date")[c].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0)
            if s.std(ddof=0) and np.isfinite(s.std(ddof=0)) and s.std(ddof=0) > 1e-12
            else 0.0
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


def evaluate_sell_signal(x: pd.DataFrame, holding_info: Optional[dict] = None) -> Dict[str, object]:
    latest_idx = len(x) - 1
    if latest_idx < 5:
        return {"sell": False, "reasons": []}

    latest = x.iloc[latest_idx]
    reasons: List[str] = []

    # 卖出条件1：中期结构走坏
    if latest_idx >= 2:
        last3 = x.iloc[latest_idx-2:latest_idx+1]
        last3_below_long = bool(last3["close_below_long"].all())
        last3_low_min = float(last3["low"].min())
        if last3_below_long and float(latest["low"]) < last3_low_min + 1e-12:
            reasons.append("连续3天收盘在多空线下方且跌破3天最低价")

    # 卖出条件2：短期趋势转弱
    if bool(latest["close_below_trend"]) and bool(latest["brick_green"]) and float(latest["ret1"]) < 0:
        reasons.append("收盘跌破趋势线且砖型转绿且当日为阴线")

    # 卖出条件3：买入价止损
    if holding_info is not None:
        buy_price = holding_info.get("buy_price", np.nan)
        if pd.notna(buy_price) and np.isfinite(float(buy_price)) and float(buy_price) > 0:
            if float(latest["close"]) <= float(buy_price) * STOP_LOSS_PCT:
                reasons.append(f"跌破买入价止损线({STOP_LOSS_PCT:.2f})")

    # 卖出条件4：买入价止盈提示
    if holding_info is not None:
        buy_price = holding_info.get("buy_price", np.nan)
        if pd.notna(buy_price) and np.isfinite(float(buy_price)) and float(buy_price) > 0:
            if float(latest["high"]) >= float(buy_price) * TAKE_PROFIT_PCT:
                reasons.append(f"达到止盈目标({TAKE_PROFIT_PCT:.3f})")

    return {"sell": len(reasons) > 0, "reasons": reasons}


def build_daily_sell_df(input_dir: Path, holdings_df: pd.DataFrame) -> pd.DataFrame:
    if holdings_df.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    total = len(holdings_df)

    for idx, holding in holdings_df.iterrows():
        code = str(holding["code"])
        matched_files = list(input_dir.glob(f"*{code}*.txt")) + list(input_dir.glob(f"*{code}*.csv"))
        if not matched_files:
            continue

        df = load_one_csv(str(matched_files[0]))
        if df is None or df.empty:
            continue

        x = add_features(df)
        sell_info = evaluate_sell_signal(x, holding.to_dict())
        if not bool(sell_info["sell"]):
            continue

        latest = x.iloc[-1]
        buy_price = holding.get("buy_price", np.nan)
        current_ret = np.nan
        if pd.notna(buy_price) and np.isfinite(float(buy_price)) and float(buy_price) > 0:
            current_ret = float(latest["close"]) / float(buy_price) - 1.0

        rows.append({
            "date": latest["date"],
            "code": code,
            "buy_price": buy_price,
            "buy_date": holding.get("buy_date", pd.NaT),
            "shares": holding.get("shares", np.nan),
            "close": float(latest["close"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "ret1": float(latest["ret1"]) if pd.notna(latest["ret1"]) else np.nan,
            "trend_line": float(latest["trend_line"]) if pd.notna(latest["trend_line"]) else np.nan,
            "long_line": float(latest["long_line"]) if pd.notna(latest["long_line"]) else np.nan,
            "brick_green": bool(latest["brick_green"]),
            "current_ret": current_ret,
            "sell_reason": " | ".join(sell_info["reasons"]),
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"卖出扫描进度: {idx + 1}/{total}")

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)
    return out


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
    if not evaluate_buy_signal(latest):
        return [-1]

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
    print("开始扫描买入候选...")
    signal_df = build_daily_signal_df(INPUT_DIR)
    selected_df = apply_selection(signal_df)

    print("\n开始扫描持仓卖出候选...")
    holdings_df = load_holdings(HOLDINGS_FILE)
    sell_df = build_daily_sell_df(INPUT_DIR, holdings_df)

    if selected_df.empty:
        print("\n当日无买入候选信号")
    else:
        latest_trade_date = pd.to_datetime(selected_df["date"]).max()
        latest_df = selected_df[pd.to_datetime(selected_df["date"]) == latest_trade_date].copy()

        show_buy_cols = [
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
        print("\n================ 买入候选 ================")
        print(latest_df[show_buy_cols].to_string(index=False))

    if sell_df.empty:
        print("\n当前持仓无卖出候选")
    else:
        show_sell_cols = [
            "date",
            "code",
            "buy_price",
            "close",
            "current_ret",
            "trend_line",
            "long_line",
            "brick_green",
            "ret1",
            "sell_reason",
        ]
        print("\n================ 卖出候选 ================")
        print(sell_df[show_sell_cols].to_string(index=False))


if __name__ == "__main__":
    main()