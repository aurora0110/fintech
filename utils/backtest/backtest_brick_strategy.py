# -*- coding: utf-8 -*-
import os
import math
import json
import hashlib
import itertools
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# =========================================================
# 配置区
# =========================================================
DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
OUTPUT_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/brick_strategy_full_experiment_output_optimized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_BARS = 160
EPS = 1e-12

DATE_COL_CANDIDATES = ["date", "Date", "trade_date", "日期", "DATE"]
OPEN_COL_CANDIDATES = ["open", "Open", "开盘", "OPEN"]
HIGH_COL_CANDIDATES = ["high", "High", "最高", "HIGH"]
LOW_COL_CANDIDATES = ["low", "Low", "最低", "LOW"]
CLOSE_COL_CANDIDATES = ["close", "Close", "收盘", "CLOSE"]
VOL_COL_CANDIDATES = ["volume", "vol", "Volume", "成交量", "VOL"]
CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码", "CODE"]

# 输出控制
SAVE_TRADE_DETAIL = False   # 若为 True，会保存所有逐笔交易文件；实验很多时会非常耗时、占空间
SAVE_SIGNAL_DETAIL = False  # 若为 True，会保存信号池


# =========================================================
# 通用工具
# =========================================================
def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > EPS)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def make_cache_key(*parts) -> str:
    raw = "||".join(stable_json_dumps(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


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

    try:
        df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
        return df
    except Exception as e:
        raise ValueError(f"文件读取失败: {path}, error={e}")


def to_datetime_auto(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


def to_numeric_auto(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_one_csv(path: str) -> pd.DataFrame:
    raw = read_csv_auto(path)

    date_col = pick_col(raw, DATE_COL_CANDIDATES)
    open_col = pick_col(raw, OPEN_COL_CANDIDATES)
    high_col = pick_col(raw, HIGH_COL_CANDIDATES)
    low_col = pick_col(raw, LOW_COL_CANDIDATES)
    close_col = pick_col(raw, CLOSE_COL_CANDIDATES)
    vol_col = pick_col(raw, VOL_COL_CANDIDATES)
    code_col = pick_col(raw, CODE_COL_CANDIDATES, required=False)

    df = pd.DataFrame({
        "date": to_datetime_auto(raw[date_col]),
        "open": to_numeric_auto(raw[open_col]),
        "high": to_numeric_auto(raw[high_col]),
        "low": to_numeric_auto(raw[low_col]),
        "close": to_numeric_auto(raw[close_col]),
        "volume": to_numeric_auto(raw[vol_col]),
    })

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
        return pd.DataFrame()
    return df


def load_all_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    data_map = {}
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".txt"))]
    total = len(files)

    for i, f in enumerate(files, 1):
        path = os.path.join(data_dir, f)
        try:
            df = load_one_csv(path)
            if not df.empty:
                code = str(df["code"].iloc[0])
                data_map[code] = df
        except Exception as e:
            print(f"加载失败: {f} -> {e}")

        if i % 500 == 0 or i == total:
            print(f"加载进度: {i}/{total}")

    return data_map


# =========================================================
# 通达信 SMA 等价实现
# SMA(X,N,M) = (M*X + (N-M)*REF(SMA,1))/N
# 等价于 ewm(alpha=M/N, adjust=False)
# =========================================================
def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    alpha = m / n
    return series.ewm(alpha=alpha, adjust=False).mean()


# =========================================================
# 特征构建
# =========================================================
def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    streak = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        if green_flag[i]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 0
    return streak


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # 基础价量特征
    x["range"] = (x["high"] - x["low"]).replace(0, np.nan)
    x["body"] = (x["close"] - x["open"]).abs()
    x["body_pct_range"] = safe_div(x["body"], x["range"])
    x["close_pos"] = safe_div(x["close"] - x["low"], x["range"])

    x["ret1"] = x["close"].pct_change()
    x["ret2"] = x["close"].pct_change(2)
    x["ret3"] = x["close"].pct_change(3)
    x["ret5"] = x["close"].pct_change(5)

    x["ma5"] = x["close"].rolling(5).mean()
    x["ma10"] = x["close"].rolling(10).mean()
    x["ma20"] = x["close"].rolling(20).mean()
    x["ma60"] = x["close"].rolling(60).mean()

    x["vol_ma5"] = x["volume"].rolling(5).mean()
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["vol_ratio_5_20"] = safe_div(x["vol_ma5"], x["vol_ma20"])
    x["vol_today_ratio_5"] = safe_div(x["volume"], x["vol_ma5"])
    x["amount"] = x["close"] * x["volume"]

    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - x["close"].shift(1)).abs(),
        (x["low"] - x["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean()

    # 你的趋势线 / 多空线
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean()
    x["trend_line"] = x["trend_line"].ewm(span=10, adjust=False).mean()

    x["MA14"] = x["close"].rolling(14).mean()
    x["MA28"] = x["close"].rolling(28).mean()
    x["MA57"] = x["close"].rolling(57).mean()
    x["MA114"] = x["close"].rolling(114).mean()
    x["bull_bear_line"] = (x["MA14"] + x["MA28"] + x["MA57"] + x["MA114"]) / 4.0

    x["trend_gt_bullbear"] = (x["trend_line"] > x["bull_bear_line"])
    x["close_above_ma20"] = (x["close"] > x["ma20"])
    x["ma5_gt_ma10_gt_ma20"] = ((x["ma5"] > x["ma10"]) & (x["ma10"] > x["ma20"]))
    x["close_above_ma60"] = (x["close"] > x["ma60"])
    x["ma20_slope_up"] = (x["ma20"] > x["ma20"].shift(5))

    # 前序走势结构
    x["prev_1_down"] = (x["close"].shift(1) < x["close"].shift(2))
    x["prev_2_down"] = (x["close"].shift(2) < x["close"].shift(3))
    x["prev_3_down"] = (x["close"].shift(3) < x["close"].shift(4))
    x["prev_3day_all_down"] = x["prev_1_down"] & x["prev_2_down"] & x["prev_3_down"]

    x["prev_2day_1up1down"] = (
        ((x["prev_1_down"]) & (~x["prev_2_down"])) |
        ((~x["prev_1_down"]) & (x["prev_2_down"]))
    )

    rolling_max_5 = x["close"].shift(1).rolling(5).max()
    x["prev_5day_drawdown"] = safe_div(x["close"].shift(1) - rolling_max_5, rolling_max_5)
    x["prev_5day_drawdown_gt3"] = (x["prev_5day_drawdown"] <= -0.03)
    x["prev_5day_drawdown_gt5"] = (x["prev_5day_drawdown"] <= -0.05)

    # 通达信砖型图
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
    x["brick_prev2"] = x["brick"].shift(2)

    x["brick_red"] = (x["brick"] > x["brick_prev"])
    x["brick_green"] = (x["brick"] < x["brick_prev"])

    x["brick_bar_len"] = (x["brick"] - x["brick_prev"]).abs()
    x["brick_red_len"] = np.where(x["brick"] > x["brick_prev"], x["brick"] - x["brick_prev"], 0.0)
    x["brick_green_len"] = np.where(x["brick"] < x["brick_prev"], x["brick_prev"] - x["brick"], 0.0)

    x["today_red"] = (x["brick_red_len"] > 0)
    x["yesterday_green"] = (pd.Series(x["brick_green_len"]).shift(1).fillna(0) > 0).values
    x["today_red_turn"] = x["today_red"] & x["yesterday_green"]

    prev_bar_len = x["brick_bar_len"].shift(1)
    prev_green_len = x["brick_green_len"].shift(1)

    x["brick_ratio_vs_prev_bar"] = safe_div(x["brick_red_len"], prev_bar_len)
    x["brick_ratio_vs_prev_green"] = safe_div(x["brick_red_len"], prev_green_len)
    x["brick_delta"] = x["brick"] - x["brick_prev"]

    green_flag = (x["brick_green_len"].values > 0)
    green_streak = calc_green_streak(green_flag)
    x["green_streak"] = green_streak
    x["prev_green_streak"] = pd.Series(green_streak, index=x.index).shift(1)

    # 回测辅助列
    x["entry_low_for_trade"] = x["low"]
    x["prev_day_low_for_trade"] = x["low"].shift(1)
    x["prev2_day_low_for_trade"] = x["low"].shift(2)

    # 静态有效掩码
    x["valid_base"] = (
        x["brick"].notna() &
        x["brick_prev"].notna() &
        x["close"].notna() &
        x["low"].notna() &
        x["high"].notna() &
        x["date"].notna() &
        np.isfinite(x["close"]) &
        np.isfinite(x["low"]) &
        np.isfinite(x["high"]) &
        (x["close"] > 0)
    )

    # 常用过滤预生成
    x["filter_prev_structure_none"] = True
    x["filter_prev_structure_3down"] = x["prev_3day_all_down"]
    x["filter_prev_structure_2down1up"] = x["prev_2day_1up1down"]
    x["filter_prev_structure_5day_drawdown_gt3"] = x["prev_5day_drawdown_gt3"]
    x["filter_prev_structure_5day_drawdown_gt5"] = x["prev_5day_drawdown_gt5"]

    x["filter_ma_none"] = True
    x["filter_ma_close_above_ma20"] = x["close_above_ma20"]
    x["filter_ma_ma5>ma10>ma20"] = x["ma5_gt_ma10_gt_ma20"]
    x["filter_ma_close_above_ma60"] = x["close_above_ma60"]

    return x


def preprocess_all_data(data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    feature_map = {}
    total = len(data_map)

    for i, (code, df) in enumerate(data_map.items(), 1):
        try:
            x = add_features(df)
            x.attrs["code"] = code
            x.attrs["date_to_idx"] = pd.Series(x.index.values, index=x["date"]).to_dict()
            feature_map[code] = x
        except Exception as e:
            print(f"特征构建失败: {code} -> {e}")

        if i % 500 == 0 or i == total:
            print(f"特征构建进度: {i}/{total}")

    return feature_map


# =========================================================
# 信号构建
# =========================================================
def get_prev_structure_mask(df: pd.DataFrame, prev_structure: Optional[str]) -> pd.Series:
    if prev_structure is None:
        return df["filter_prev_structure_none"]
    col = f"filter_prev_structure_{prev_structure}"
    return df[col] if col in df.columns else df["filter_prev_structure_none"]


def get_ma_filter_mask(df: pd.DataFrame, ma_filter: Optional[str]) -> pd.Series:
    if ma_filter is None:
        return df["filter_ma_none"]
    col = f"filter_ma_{ma_filter}"
    return df[col] if col in df.columns else df["filter_ma_none"]


def build_signal_mask(df: pd.DataFrame, signal_params: Dict[str, Any]) -> pd.Series:
    cond = df["valid_base"].copy()

    require_today_red = signal_params.get("require_today_red", True)
    if require_today_red:
        cond &= df["today_red"]

    require_yesterday_green = signal_params.get("require_yesterday_green", False)
    if require_yesterday_green:
        cond &= df["yesterday_green"]

    require_today_red_turn = signal_params.get("require_today_red_turn", False)
    if require_today_red_turn:
        cond &= df["today_red_turn"]

    brick_ratio_threshold = signal_params.get("brick_ratio_threshold", None)
    ratio_ref = signal_params.get("brick_ratio_ref", "prev_bar")
    if brick_ratio_threshold is not None:
        if ratio_ref == "prev_green":
            cond &= (df["brick_ratio_vs_prev_green"] >= brick_ratio_threshold)
        else:
            cond &= (df["brick_ratio_vs_prev_bar"] >= brick_ratio_threshold)

    brick_abs_threshold = signal_params.get("brick_abs_threshold", None)
    if brick_abs_threshold is not None:
        cond &= (df["brick"] >= brick_abs_threshold)

    brick_delta_threshold = signal_params.get("brick_delta_threshold", None)
    if brick_delta_threshold is not None:
        cond &= (df["brick_delta"] >= brick_delta_threshold)

    require_trend_gt_bullbear = signal_params.get("require_trend_gt_bullbear", False)
    if require_trend_gt_bullbear:
        cond &= df["trend_gt_bullbear"]

    require_yang = signal_params.get("require_yang", False)
    if require_yang:
        cond &= (df["close"] > df["open"])

    close_pos_threshold = signal_params.get("close_pos_threshold", None)
    if close_pos_threshold is not None:
        cond &= (df["close_pos"] >= close_pos_threshold)

    body_pct_range_threshold = signal_params.get("body_pct_range_threshold", None)
    if body_pct_range_threshold is not None:
        cond &= (df["body_pct_range"] >= body_pct_range_threshold)

    prev_structure = signal_params.get("prev_structure", None)
    cond &= get_prev_structure_mask(df, prev_structure)

    min_prev_green_streak = signal_params.get("min_prev_green_streak", None)
    if min_prev_green_streak is not None:
        cond &= (df["prev_green_streak"] >= min_prev_green_streak)

    vol_ratio_5_20_threshold = signal_params.get("vol_ratio_5_20_threshold", None)
    if vol_ratio_5_20_threshold is not None:
        cond &= (df["vol_ratio_5_20"] >= vol_ratio_5_20_threshold)

    vol_today_ratio_5_threshold = signal_params.get("vol_today_ratio_5_threshold", None)
    if vol_today_ratio_5_threshold is not None:
        cond &= (df["vol_today_ratio_5"] >= vol_today_ratio_5_threshold)

    amount_threshold = signal_params.get("amount_threshold", None)
    if amount_threshold is not None:
        cond &= (df["amount"] >= amount_threshold)

    ma_filter = signal_params.get("ma_filter", None)
    cond &= get_ma_filter_mask(df, ma_filter)

    require_ma20_slope_up = signal_params.get("require_ma20_slope_up", False)
    if require_ma20_slope_up:
        cond &= df["ma20_slope_up"]

    return cond.fillna(False)


def build_signal_df(
    feature_map: Dict[str, pd.DataFrame],
    signal_params: Dict[str, Any],
    signal_cache: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    cache_key = make_cache_key(signal_params)
    if cache_key in signal_cache:
        return signal_cache[cache_key].copy()

    frames = []
    total = len(feature_map)

    use_cols = [
        "date", "code", "open", "high", "low", "close", "volume",
        "amount", "atr14",
        "brick", "brick_prev", "brick_prev2", "brick_delta",
        "brick_bar_len", "brick_red_len", "brick_green_len",
        "brick_ratio_vs_prev_bar", "brick_ratio_vs_prev_green",
        "today_red", "yesterday_green", "today_red_turn",
        "prev_green_streak",
        "trend_line", "bull_bear_line",
        "close_pos", "body_pct_range",
        "vol_ratio_5_20", "vol_today_ratio_5",
        "prev_5day_drawdown",
        "entry_low_for_trade", "prev_day_low_for_trade", "prev2_day_low_for_trade",
    ]

    for i, (code, df) in enumerate(feature_map.items(), 1):
        mask = build_signal_mask(df, signal_params)
        if mask.any():
            sig = df.loc[mask, use_cols].copy()
            sig["code"] = code
            frames.append(sig)

        if i % 500 == 0 or i == total:
            print(f"信号构建进度: {i}/{total}")

    if not frames:
        out = pd.DataFrame()
    else:
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["date", "code"]).reset_index(drop=True)

    signal_cache[cache_key] = out
    return out.copy()


# =========================================================
# 排序与每天取前N
# =========================================================
def build_sort_score(df: pd.DataFrame, sort_by: Optional[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    if sort_by is None or sort_by == "none":
        return pd.Series(np.arange(len(df), dtype=float), index=df.index)

    if sort_by == "brick_value":
        return df["brick"].fillna(-1e18)

    if sort_by == "brick_delta":
        return df["brick_delta"].fillna(-1e18)

    if sort_by == "vol_ratio":
        return df["vol_ratio_5_20"].fillna(-1e18)

    if sort_by == "trend_strength":
        return (df["trend_line"] - df["bull_bear_line"]).fillna(-1e18)

    if sort_by == "prev_drawdown":
        return (-df["prev_5day_drawdown"]).fillna(-1e18)

    if sort_by == "composite":
        f1 = df["brick_delta"]
        f2 = df["vol_ratio_5_20"]
        f3 = df["trend_line"] - df["bull_bear_line"]
        f4 = df["close_pos"]

        z1 = (f1 - f1.mean()) / (f1.std() + EPS)
        z2 = (f2 - f2.mean()) / (f2.std() + EPS)
        z3 = (f3 - f3.mean()) / (f3.std() + EPS)
        z4 = (f4 - f4.mean()) / (f4.std() + EPS)
        return (0.35 * z1 + 0.25 * z2 + 0.20 * z3 + 0.20 * z4).fillna(-1e18)

    return pd.Series(np.arange(len(df), dtype=float), index=df.index)


def apply_daily_selection(signal_df: pd.DataFrame, select_params: Dict[str, Any]) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df

    top_n = select_params.get("top_n", None)
    sort_by = select_params.get("sort_by", None)

    df = signal_df.copy()
    df["sort_score"] = build_sort_score(df, sort_by)

    df = df.sort_values(["date", "sort_score", "code"], ascending=[True, False, True])

    if top_n is None:
        return df.reset_index(drop=True)

    selected = df.groupby("date", group_keys=False).head(top_n).reset_index(drop=True)
    return selected


# =========================================================
# 单笔交易回测
# =========================================================
def parse_stop_mode(stop_mode: str) -> Tuple[str, Optional[float]]:
    if stop_mode.startswith("fixed_pct_"):
        try:
            return "fixed_pct", float(stop_mode.replace("fixed_pct_", ""))
        except Exception:
            return stop_mode, None

    if stop_mode.startswith("atr_"):
        try:
            return "atr", float(stop_mode.replace("atr_", ""))
        except Exception:
            return stop_mode, None

    return stop_mode, None


def get_stop_price_from_arrays(
    entry_close: float,
    entry_low: float,
    prev_day_low: float,
    prev2_day_low: float,
    atr14: float,
    stop_mode: str
) -> Optional[float]:
    mode, value = parse_stop_mode(stop_mode)

    if not np.isfinite(entry_close) or entry_close <= 0:
        return None

    if mode == "none":
        return None
    if mode == "prev_day_low":
        return prev_day_low if np.isfinite(prev_day_low) and prev_day_low > 0 else None
    if mode == "entry_low":
        return entry_low if np.isfinite(entry_low) and entry_low > 0 else None
    if mode == "prev2_day_low":
        return prev2_day_low if np.isfinite(prev2_day_low) and prev2_day_low > 0 else None

    if mode == "tighter_of_two":
        vals = [v for v in [entry_low, prev_day_low] if np.isfinite(v) and v > 0]
        return max(vals) if vals else None

    if mode == "looser_of_two":
        vals = [v for v in [entry_low, prev_day_low] if np.isfinite(v) and v > 0]
        return min(vals) if vals else None

    if mode == "fixed_pct" and value is not None:
        return entry_close * (1 - value)

    if mode == "atr" and value is not None and np.isfinite(atr14) and atr14 > 0:
        return entry_close - value * atr14

    return None


def safe_return(exit_price: float, entry_price: float) -> float:
    if not np.isfinite(exit_price) or not np.isfinite(entry_price) or entry_price <= 0:
        return np.nan
    return exit_price / entry_price - 1


def simulate_one_trade(
    full_df: pd.DataFrame,
    entry_idx: int,
    max_hold_days: int,
    stop_mode: str,
    stop_trigger_mode: str,
    take_profit_mode: str,
    take_profit_threshold: float,
    take_profit_sell_timing: str
) -> Tuple[float, int, str, Optional[pd.Timestamp], Optional[float]]:
    n = len(full_df)
    if entry_idx < 0 or entry_idx >= n:
        return np.nan, 0, "索引越界", None, np.nan

    entry_row = full_df.iloc[entry_idx]
    entry_close = entry_row["close"]
    entry_low = entry_row["entry_low_for_trade"]
    prev_day_low = entry_row["prev_day_low_for_trade"]
    prev2_day_low = entry_row["prev2_day_low_for_trade"]
    atr14 = entry_row["atr14"]

    if not np.isfinite(entry_close) or entry_close <= 0:
        return np.nan, 0, "无效入场价", None, np.nan

    stop_price = get_stop_price_from_arrays(
        entry_close=entry_close,
        entry_low=entry_low,
        prev_day_low=prev_day_low,
        prev2_day_low=prev2_day_low,
        atr14=atr14,
        stop_mode=stop_mode
    )

    last_j = min(entry_idx + max_hold_days, n - 1)

    for j in range(entry_idx + 1, last_j + 1):
        row = full_df.iloc[j]
        high_j = row["high"]
        low_j = row["low"]
        close_j = row["close"]

        # 止损
        if stop_price is not None and np.isfinite(stop_price) and stop_price > 0:
            if stop_trigger_mode == "intraday":
                if np.isfinite(low_j) and low_j <= stop_price:
                    return safe_return(stop_price, entry_close), j - entry_idx, "止损", row["date"], stop_price
            elif stop_trigger_mode == "close":
                if np.isfinite(close_j) and close_j <= stop_price:
                    return safe_return(close_j, entry_close), j - entry_idx, "收盘止损", row["date"], close_j

        # 止盈
        if take_profit_mode != "none" and take_profit_threshold is not None and take_profit_threshold > 0:
            target_price = entry_close * (1 + take_profit_threshold)

            if take_profit_mode == "high_touch_same_day":
                if np.isfinite(high_j) and high_j >= target_price:
                    return take_profit_threshold, j - entry_idx, "触及止盈", row["date"], target_price

            elif take_profit_mode in ("close_profit_next_close", "close_profit_next_open"):
                if np.isfinite(close_j) and close_j >= target_price:
                    sell_idx = min(j + 1, n - 1)
                    sell_row = full_df.iloc[sell_idx]

                    if take_profit_sell_timing == "next_open":
                        px = sell_row["open"]
                        reason = "收盘达标次日开盘止盈"
                    else:
                        px = sell_row["close"]
                        reason = "收盘达标次日收盘止盈"

                    if np.isfinite(px) and px > 0:
                        return safe_return(px, entry_close), sell_idx - entry_idx, reason, sell_row["date"], px

    close_last = full_df.iloc[last_j]["close"]
    return safe_return(close_last, entry_close), last_j - entry_idx, "到期卖出", full_df.iloc[last_j]["date"], close_last


def backtest_trades(
    feature_map: Dict[str, pd.DataFrame],
    signal_df: pd.DataFrame,
    exit_params: Dict[str, Any]
) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame()

    rows = []
    total_signals = len(signal_df)
    processed = 0
    
    print(f"  开始回测计算，总信号数: {total_signals}")

    for code, sig in signal_df.groupby("code", sort=False):
        if code not in feature_map:
            processed += len(sig)
            continue

        full_df = feature_map[code]
        date_to_idx = full_df.attrs["date_to_idx"]

        for row in sig.itertuples(index=False):
            d = row.date
            entry_idx = date_to_idx.get(d, None)
            if entry_idx is None:
                processed += 1
                continue

            ret, hold_days, exit_reason, exit_date, exit_price = simulate_one_trade(
                full_df=full_df,
                entry_idx=int(entry_idx),
                max_hold_days=exit_params.get("max_hold_days", 3),
                stop_mode=exit_params.get("stop_mode", "none"),
                stop_trigger_mode=exit_params.get("stop_trigger_mode", "intraday"),
                take_profit_mode=exit_params.get("take_profit_mode", "none"),
                take_profit_threshold=exit_params.get("take_profit_threshold", 0.0),
                take_profit_sell_timing=exit_params.get("take_profit_sell_timing", "next_close"),
            )

            processed += 1
            
            # 每处理1000个信号显示一次进度
            if processed % 1000 == 0 or processed == total_signals:
                progress = (processed / total_signals) * 100
                print(f"    回测进度: {processed}/{total_signals} ({progress:.1f}%)")

            rows.append({
                "date": d,
                "code": code,
                "entry_price": row.close,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "ret": ret,
                "hold_days": hold_days,
                "exit_reason": exit_reason,
                "brick": getattr(row, "brick", np.nan),
                "brick_delta": getattr(row, "brick_delta", np.nan),
                "brick_red_len": getattr(row, "brick_red_len", np.nan),
                "brick_ratio_vs_prev_bar": getattr(row, "brick_ratio_vs_prev_bar", np.nan),
                "brick_ratio_vs_prev_green": getattr(row, "brick_ratio_vs_prev_green", np.nan),
                "vol_ratio_5_20": getattr(row, "vol_ratio_5_20", np.nan),
                "trend_strength": (getattr(row, "trend_line", np.nan) - getattr(row, "bull_bear_line", np.nan)),
                "close_pos": getattr(row, "close_pos", np.nan),
                "sort_score": getattr(row, "sort_score", np.nan),
            })

    trade_df = pd.DataFrame(rows)
    if trade_df.empty:
        print(f"  回测完成: 生成0笔有效交易")
        return trade_df

    trade_df = trade_df.dropna(subset=["ret"])
    trade_df = trade_df[np.isfinite(trade_df["ret"])].copy()
    trade_df = trade_df.sort_values(["date", "code"]).reset_index(drop=True)
    
    valid_trades = len(trade_df)
    print(f"  回测完成: 生成{valid_trades}笔有效交易 (有效率: {valid_trades/total_signals*100:.1f}%)")
    return trade_df


# =========================================================
# 统计
# =========================================================
def calc_profit_factor(ret_series: pd.Series) -> float:
    pos = ret_series[ret_series > 0].sum()
    neg = ret_series[ret_series < 0].sum()
    if abs(neg) < EPS:
        return np.nan
    return pos / abs(neg)


def calc_equity_and_mdd(ret_series: pd.Series) -> Tuple[float, float]:
    if len(ret_series) == 0:
        return np.nan, np.nan

    r = ret_series.clip(lower=-0.999999)
    log_eq = np.log1p(r).cumsum()

    final_log = float(log_eq.iloc[-1])
    capped_final_log = min(final_log, 700)
    final_equity = float(np.exp(capped_final_log))

    running_max_log = log_eq.cummax()
    dd = np.exp(log_eq - running_max_log) - 1.0
    max_dd = float(dd.min()) if len(dd) > 0 else np.nan

    return final_equity, max_dd


def summarize_trades(trade_df: pd.DataFrame, extra_cols: Dict[str, Any]) -> pd.DataFrame:
    if trade_df.empty:
        row = {
            "样本数": 0,
            "平均每笔收益": np.nan,
            "胜率": np.nan,
            "收益标准差": np.nan,
            "平均持有天数": np.nan,
            "盈亏比": np.nan,
            "近似夏普": np.nan,
            "累计收益(逐笔净值)": np.nan,
            "最大回撤(逐笔净值)": np.nan,
        }
        row.update(extra_cols)
        return pd.DataFrame([row])

    ret = trade_df["ret"]
    mean_ret = ret.mean()
    std_ret = ret.std()
    win_rate = (ret > 0).mean()
    hold_days = trade_df["hold_days"].mean()
    profit_factor = calc_profit_factor(ret)
    sharpe_like = mean_ret / std_ret if pd.notna(std_ret) and std_ret > 0 else np.nan
    final_equity, max_dd = calc_equity_and_mdd(ret)

    row = {
        "样本数": int(len(trade_df)),
        "平均每笔收益": mean_ret,
        "胜率": win_rate,
        "收益标准差": std_ret,
        "平均持有天数": hold_days,
        "盈亏比": profit_factor,
        "近似夏普": sharpe_like,
        "累计收益(逐笔净值)": final_equity,
        "最大回撤(逐笔净值)": max_dd,
    }
    row.update(extra_cols)
    return pd.DataFrame([row])


def build_daily_basket(trade_df: pd.DataFrame, basket_weight: str = "equal") -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    rows = []
    for d, g in trade_df.groupby("date", sort=True):
        g = g.copy()

        if basket_weight == "score" and "sort_score" in g.columns:
            score = g["sort_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            score = score - score.min() + 1e-6
            total_score = score.sum()
            w = score / total_score if total_score > 0 else np.repeat(1 / len(g), len(g))
        else:
            w = np.repeat(1 / len(g), len(g))

        basket_ret = float(np.sum(g["ret"].values * w))
        rows.append({
            "date": d,
            "basket_size": len(g),
            "basket_ret": basket_ret
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_basket(basket_df: pd.DataFrame, extra_cols: Dict[str, Any], exp_name: str) -> pd.DataFrame:
    if basket_df.empty:
        row = {
            "实验名称": exp_name,
            "交易日数": 0,
            "日篮子平均收益": np.nan,
            "日篮子胜率": np.nan,
            "日篮子收益标准差": np.nan,
            "日篮子近似夏普": np.nan,
            "日篮子累计净值": np.nan,
            "日篮子最大回撤": np.nan,
        }
        row.update(extra_cols)
        return pd.DataFrame([row])

    ret = basket_df["basket_ret"]
    final_equity, max_dd = calc_equity_and_mdd(ret)

    row = {
        "实验名称": exp_name,
        "交易日数": int(len(basket_df)),
        "日篮子平均收益": ret.mean(),
        "日篮子胜率": (ret > 0).mean(),
        "日篮子收益标准差": ret.std(),
        "日篮子近似夏普": ret.mean() / ret.std() if ret.std() > 0 else np.nan,
        "日篮子累计净值": final_equity,
        "日篮子最大回撤": max_dd,
    }
    row.update(extra_cols)
    return pd.DataFrame([row])


# =========================================================
# 实验执行器 + 缓存
# =========================================================
def run_trade_experiment(
    feature_map: Dict[str, pd.DataFrame],
    signal_params: Dict[str, Any],
    exit_params: Dict[str, Any],
    select_params: Dict[str, Any],
    exp_name: str,
    signal_cache: Dict[str, pd.DataFrame],
    trade_cache: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_signal_df = build_signal_df(feature_map, signal_params, signal_cache)
    signal_df = apply_daily_selection(raw_signal_df, select_params)

    trade_key = make_cache_key(signal_params, exit_params, select_params)
    if trade_key in trade_cache:
        trade_df = trade_cache[trade_key].copy()
    else:
        trade_df = backtest_trades(feature_map, signal_df, exit_params)
        trade_cache[trade_key] = trade_df

    extra = {}
    extra.update(signal_params)
    extra.update(exit_params)
    extra.update(select_params)

    summary_df = summarize_trades(trade_df, extra)
    summary_df.insert(0, "实验名称", exp_name)

    basket_df = build_daily_basket(trade_df, basket_weight=select_params.get("basket_weight", "equal"))
    basket_summary_df = summarize_basket(basket_df, extra, exp_name)

    return summary_df, trade_df, basket_summary_df, signal_df


def best_row_by_mean_then_sharpe(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    sort_cols = []
    ascending = []
    for c in ["平均每笔收益", "近似夏普", "胜率", "样本数"]:
        if c in df.columns:
            sort_cols.append(c)
            ascending.append(False)
    if not sort_cols:
        return None
    return df.sort_values(sort_cols, ascending=ascending).iloc[0]


# =========================================================
# 自动结论
# =========================================================
def generate_conclusion(report_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for k, df in report_map.items():
        if df.empty:
            continue
        best = best_row_by_mean_then_sharpe(df)
        if best is None:
            continue

        rows.append({
            "结论项": k,
            "结果": best["实验名称"] if "实验名称" in best.index else "见表格",
            "原因": f"平均每笔收益={best.get('平均每笔收益', np.nan):.4f}, 胜率={best.get('胜率', np.nan):.2%}, 近似夏普={best.get('近似夏普', np.nan):.4f}, 样本数={int(best.get('样本数', 0))}"
        })

    return pd.DataFrame(rows)


# =========================================================
# 运行单组实验并可选保存
# =========================================================
def run_and_collect(
    exp_name: str,
    report_key: str,
    signal_params: Dict[str, Any],
    exit_params: Dict[str, Any],
    select_params: Dict[str, Any],
    feature_map: Dict[str, pd.DataFrame],
    signal_cache: Dict[str, pd.DataFrame],
    trade_cache: Dict[str, pd.DataFrame],
    report_map: Dict[str, List[pd.DataFrame]],
    basket_report_map: Dict[str, List[pd.DataFrame]],
):
    summary, trades, basket_summary, signal_df = run_trade_experiment(
        feature_map=feature_map,
        signal_params=signal_params,
        exit_params=exit_params,
        select_params=select_params,
        exp_name=exp_name,
        signal_cache=signal_cache,
        trade_cache=trade_cache,
    )

    report_map.setdefault(report_key, []).append(summary)
    basket_report_map.setdefault(report_key + "_日篮子", []).append(basket_summary)

    if SAVE_TRADE_DETAIL and not trades.empty:
        trades.to_csv(os.path.join(OUTPUT_DIR, f"{exp_name}_逐笔交易.csv"), index=False, encoding="utf-8-sig")

    if SAVE_SIGNAL_DETAIL and not signal_df.empty:
        signal_df.to_csv(os.path.join(OUTPUT_DIR, f"{exp_name}_信号池.csv"), index=False, encoding="utf-8-sig")


# =========================================================
# 主流程
# =========================================================
def main():
    print("开始加载数据...")
    data_map = load_all_data(DATA_DIR)
    print(f"有效股票数: {len(data_map)}")

    print("开始构建特征...")
    feature_map = preprocess_all_data(data_map)

    signal_cache: Dict[str, pd.DataFrame] = {}
    trade_cache: Dict[str, pd.DataFrame] = {}

    # 基准参数
    base_signal = {
        "require_today_red": True,
        "require_yesterday_green": True,
        "require_today_red_turn": False,
        "brick_ratio_threshold": 1.0,
        "brick_ratio_ref": "prev_green",
        "brick_abs_threshold": None,
        "brick_delta_threshold": None,
        "require_trend_gt_bullbear": False,
        "require_yang": False,
        "close_pos_threshold": None,
        "body_pct_range_threshold": None,
        "prev_structure": None,
        "min_prev_green_streak": None,
        "vol_ratio_5_20_threshold": None,
        "vol_today_ratio_5_threshold": None,
        "amount_threshold": None,
        "ma_filter": None,
        "require_ma20_slope_up": False,
    }

    base_exit = {
        "max_hold_days": 3,
        "stop_mode": "none",
        "stop_trigger_mode": "intraday",
        "take_profit_mode": "none",
        "take_profit_threshold": 0.0,
        "take_profit_sell_timing": "next_close",
    }

    base_select = {
        "sort_by": "none",
        "top_n": None,
        "basket_weight": "equal",
    }

    report_map: Dict[str, List[pd.DataFrame]] = {}
    basket_report_map: Dict[str, List[pd.DataFrame]] = {}

    # 1
    print("\n开始实验1：砖长比例阈值对比...")
    for v in [0.66, 0.8, 1.0, 1.2, 1.5]:
        print(f"  正在处理砖长比例阈值: {v}")
        sp = dict(base_signal, brick_ratio_threshold=v, brick_ratio_ref="prev_green", require_yesterday_green=True)
        
        # 添加调试信息
        print(f"  开始构建信号...")
        try:
            signal_df = build_signal_df(feature_map, sp, signal_cache)
            print(f"  信号数量: {len(signal_df)}")
            
            if signal_df.empty:
                print(f"  警告: 没有生成有效信号，跳过阈值 {v}")
                continue
                
            print(f"  开始运行回测...")
            print(f"  信号数据预览:")
            print(f"    日期范围: {signal_df['date'].min()} 到 {signal_df['date'].max()}")
            print(f"    股票数量: {signal_df['code'].nunique()}")
            print(f"    每日平均信号数: {len(signal_df) / signal_df['date'].nunique():.1f}")
            print(f"    总信号数: {len(signal_df)}")
            
            run_and_collect(
                exp_name=f"实验1_砖长比例_{v:.2f}",
                report_key="实验1_砖长比例阈值对比",
                signal_params=sp, exit_params=base_exit, select_params=base_select,
                feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
                report_map=report_map, basket_report_map=basket_report_map
            )
            print(f"  砖长比例阈值 {v} 处理完成")
        except Exception as e:
            print(f"  错误: 处理阈值 {v} 时发生异常: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 2
    print("\n开始实验2：昨天是否必须绿柱...")
    for v in [True, False]:
        sp = dict(base_signal, require_yesterday_green=v, require_today_red=True, require_today_red_turn=False,
                  brick_ratio_ref="prev_bar", brick_ratio_threshold=1.0)
        run_and_collect(
            exp_name=f"实验2_昨天绿柱_{v}",
            report_key="实验2_昨天是否必须绿柱",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 3
    print("\n开始实验3：今天是否必须红柱翻转...")
    for v in [True, False]:
        sp = dict(base_signal, require_today_red_turn=v, require_yesterday_green=False, require_today_red=True,
                  brick_ratio_ref="prev_bar", brick_ratio_threshold=1.0)
        run_and_collect(
            exp_name=f"实验3_今天必须翻转_{v}",
            report_key="实验3_今天是否必须红柱翻转",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 4
    print("\n开始实验4：砖型图绝对值门槛...")
    for v in [None, 10, 20, 30]:
        sp = dict(base_signal, brick_abs_threshold=v)
        run_and_collect(
            exp_name=f"实验4_砖型图绝对值_{'None' if v is None else v}",
            report_key="实验4_砖型图绝对值门槛",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 5
    print("\n开始实验5：砖型图增量门槛...")
    for v in [None, 3, 5, 10, 15]:
        sp = dict(base_signal, brick_delta_threshold=v)
        run_and_collect(
            exp_name=f"实验5_砖型图增量_{'None' if v is None else v}",
            report_key="实验5_砖型图增量门槛",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 6
    print("\n开始实验6：趋势线>多空线是否必须...")
    for v in [False, True]:
        sp = dict(base_signal, require_trend_gt_bullbear=v)
        run_and_collect(
            exp_name=f"实验6_趋势线大于多空线_{v}",
            report_key="实验6_趋势线>多空线是否必须",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 7
    print("\n开始实验7：K线是否必须阳线...")
    for v in [False, True]:
        sp = dict(base_signal, require_yang=v)
        run_and_collect(
            exp_name=f"实验7_必须阳线_{v}",
            report_key="实验7_K线必须为阳线吗",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 8
    print("\n开始实验8：收盘位置限制...")
    for v in [None, 0.6, 0.7, 0.8]:
        sp = dict(base_signal, close_pos_threshold=v)
        run_and_collect(
            exp_name=f"实验8_收盘位置_{'None' if v is None else v}",
            report_key="实验8_收盘位置限制",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 9
    print("\n开始实验9：实体长度限制...")
    for v in [None, 0.3, 0.5, 0.7]:
        sp = dict(base_signal, body_pct_range_threshold=v)
        run_and_collect(
            exp_name=f"实验9_实体占比_{'None' if v is None else v}",
            report_key="实验9_实体长度限制",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 10
    print("\n开始实验10：前几日涨跌结构...")
    for v in [None, "3down", "2down1up", "5day_drawdown_gt3", "5day_drawdown_gt5"]:
        sp = dict(base_signal, prev_structure=v)
        run_and_collect(
            exp_name=f"实验10_前序结构_{'None' if v is None else v}",
            report_key="实验10_前几日涨跌结构",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 11
    print("\n开始实验11：5日均量/20日均量阈值...")
    for v in [None, 1.0, 1.1, 1.2, 1.5]:
        sp = dict(base_signal, vol_ratio_5_20_threshold=v)
        run_and_collect(
            exp_name=f"实验11_5日量20日量_{'None' if v is None else v}",
            report_key="实验11_5日均量除20日均量阈值",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 12
    print("\n开始实验12：当天量/5日均量阈值...")
    for v in [None, 1.0, 1.2, 1.5]:
        sp = dict(base_signal, vol_today_ratio_5_threshold=v)
        run_and_collect(
            exp_name=f"实验12_当天量5日量_{'None' if v is None else v}",
            report_key="实验12_当天成交量除5日均量阈值",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 13
    print("\n开始实验13：成交额门槛...")
    for v in [None, 5e7, 1e8, 3e8]:
        sp = dict(base_signal, amount_threshold=v)
        run_and_collect(
            exp_name=f"实验13_成交额门槛_{'None' if v is None else int(v)}",
            report_key="实验13_成交额门槛",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 14
    print("\n开始实验14：均线过滤...")
    for ma_filter, slope_up in itertools.product(
        [None, "close_above_ma20", "ma5>ma10>ma20", "close_above_ma60"],
        [False, True]
    ):
        sp = dict(base_signal, ma_filter=ma_filter, require_ma20_slope_up=slope_up)
        run_and_collect(
            exp_name=f"实验14_MA过滤_{ma_filter}_MA20斜率上升_{slope_up}",
            report_key="实验14_均线与斜率过滤",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 15
    print("\n开始实验15：止损方式...")
    for v in ["none", "prev_day_low", "entry_low", "prev2_day_low", "tighter_of_two", "looser_of_two", "fixed_pct_0.03", "atr_1.0", "atr_1.5"]:
        ep = dict(base_exit, stop_mode=v)
        run_and_collect(
            exp_name=f"实验15_止损方式_{v}",
            report_key="实验15_止损方式",
            signal_params=base_signal, exit_params=ep, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 16
    print("\n开始实验16：止损触发方式...")
    for v in ["intraday", "close"]:
        ep = dict(base_exit, stop_mode="prev_day_low", stop_trigger_mode=v)
        run_and_collect(
            exp_name=f"实验16_止损触发_{v}",
            report_key="实验16_盘中止损还是收盘止损",
            signal_params=base_signal, exit_params=ep, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 17
    print("\n开始实验17：止盈方式...")
    tp_cases = [
        ("none", 0.0, "next_close"),
        ("close_profit_next_close", 0.025, "next_close"),
        ("close_profit_next_close", 0.03, "next_close"),
        ("close_profit_next_close", 0.035, "next_close"),
        ("close_profit_next_close", 0.04, "next_close"),
        ("close_profit_next_open", 0.03, "next_open"),
        ("close_profit_next_open", 0.035, "next_open"),
        ("high_touch_same_day", 0.03, "next_close"),
        ("high_touch_same_day", 0.04, "next_close"),
        ("high_touch_same_day", 0.05, "next_close"),
    ]
    for mode, th, timing in tp_cases:
        ep = dict(base_exit, take_profit_mode=mode, take_profit_threshold=th, take_profit_sell_timing=timing)
        run_and_collect(
            exp_name=f"实验17_止盈_{mode}_{th}_{timing}",
            report_key="实验17_止盈方式与阈值",
            signal_params=base_signal, exit_params=ep, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 18
    print("\n开始实验18：最长持有天数...")
    for v in [1, 2, 3, 5]:
        ep = dict(base_exit, max_hold_days=v)
        run_and_collect(
            exp_name=f"实验18_最长持有_{v}天",
            report_key="实验18_最长持有天数",
            signal_params=base_signal, exit_params=ep, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 19
    print("\n开始实验19：排序与每天取前N...")
    for sort_by, top_n in itertools.product(
        ["brick_value", "brick_delta", "vol_ratio", "trend_strength", "prev_drawdown", "composite"],
        [5, 10, 20]
    ):
        sel = dict(base_select, sort_by=sort_by, top_n=top_n, basket_weight="equal")
        run_and_collect(
            exp_name=f"实验19_排序_{sort_by}_前{top_n}",
            report_key="实验19_排序与每天取前N",
            signal_params=base_signal, exit_params=base_exit, select_params=sel,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 20
    print("\n开始实验20：组合层权重近似对比...")
    for basket_weight in ["equal", "score"]:
        sel = dict(base_select, sort_by="composite", top_n=10, basket_weight=basket_weight)
        run_and_collect(
            exp_name=f"实验20_权重_{basket_weight}",
            report_key="实验20_组合层权重近似对比",
            signal_params=base_signal, exit_params=base_exit, select_params=sel,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 21
    print("\n开始实验21：连续绿几天后翻红 + 红柱长度比例...")
    for streak, ratio in itertools.product([1, 2, 3, 4], [0.66, 1.0, 1.2, 1.5]):
        sp = dict(
            base_signal,
            require_today_red=True,
            require_yesterday_green=True,
            require_today_red_turn=True,
            min_prev_green_streak=streak,
            brick_ratio_threshold=ratio,
            brick_ratio_ref="prev_green",
        )
        run_and_collect(
            exp_name=f"实验21_连续绿{streak}天后翻红_比例{ratio}",
            report_key="实验21_连续绿几天后翻红且红柱长度比例",
            signal_params=sp, exit_params=base_exit, select_params=base_select,
            feature_map=feature_map, signal_cache=signal_cache, trade_cache=trade_cache,
            report_map=report_map, basket_report_map=basket_report_map
        )

    # 汇总保存
    final_report_map: Dict[str, pd.DataFrame] = {}
    final_basket_report_map: Dict[str, pd.DataFrame] = {}

    for name, rows in report_map.items():
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        final_report_map[name] = df
        df.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False, encoding="utf-8-sig")

    for name, rows in basket_report_map.items():
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        final_basket_report_map[name] = df
        df.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False, encoding="utf-8-sig")

    conclusion_df = generate_conclusion(final_report_map)
    conclusion_df.to_csv(os.path.join(OUTPUT_DIR, "自动结论.csv"), index=False, encoding="utf-8-sig")

    print("\n================= 自动结论 =================")
    if conclusion_df.empty:
        print("暂无结论，请检查样本数是否不足。")
    else:
        print(conclusion_df.to_string(index=False))

    print("\n信号缓存数量:", len(signal_cache))
    print("交易缓存数量:", len(trade_cache))

    print("\n结果文件已保存到：")
    print(OUTPUT_DIR)
    print("\n重点查看：")
    for k in final_report_map.keys():
        print(f"- {k}.csv")
    for k in final_basket_report_map.keys():
        print(f"- {k}.csv")
    print("- 自动结论.csv")


if __name__ == "__main__":
    main()