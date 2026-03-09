# -*- coding: utf-8 -*-
import os
import math
import itertools
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# =========================================================
# 配置区
# =========================================================
DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
OUTPUT_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/brick_strategy_full_experiment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_BARS = 160

DATE_COL_CANDIDATES = ["date", "Date", "trade_date", "日期", "DATE"]
OPEN_COL_CANDIDATES = ["open", "Open", "开盘", "OPEN"]
HIGH_COL_CANDIDATES = ["high", "High", "最高", "HIGH"]
LOW_COL_CANDIDATES = ["low", "Low", "最低", "LOW"]
CLOSE_COL_CANDIDATES = ["close", "Close", "收盘", "CLOSE"]
VOL_COL_CANDIDATES = ["volume", "vol", "Volume", "成交量", "VOL"]
CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码", "CODE"]

EPS = 1e-12


# =========================================================
# 通用工具
# =========================================================
def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full_like(a_arr, default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > EPS)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"缺少字段，候选字段={candidates}")
    return None


def read_csv_auto(path: str) -> pd.DataFrame:
    # 先正常读逗号分隔
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # 再尝试空白/制表分隔
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

    # 基础过滤，防止0价/负价引发回测异常
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
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

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

    # 如果没有成交额字段，这里用 close * volume 近似
    x["amount"] = x["close"] * x["volume"]

    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - x["close"].shift(1)).abs(),
        (x["low"] - x["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean()

    # 你的知行趋势线 / 多空线
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean()
    x["trend_line"] = x["trend_line"].ewm(span=10, adjust=False).mean()

    x["MA14"] = x["close"].rolling(14).mean()
    x["MA28"] = x["close"].rolling(28).mean()
    x["MA57"] = x["close"].rolling(57).mean()
    x["MA114"] = x["close"].rolling(114).mean()
    x["bull_bear_line"] = (x["MA14"] + x["MA28"] + x["MA57"] + x["MA114"]) / 4.0

    x["trend_gt_bullbear"] = (x["trend_line"] > x["bull_bear_line"]).astype(int)
    x["close_above_ma20"] = (x["close"] > x["ma20"]).astype(int)
    x["ma5_gt_ma10_gt_ma20"] = ((x["ma5"] > x["ma10"]) & (x["ma10"] > x["ma20"])).astype(int)
    x["ma20_slope_up"] = (x["ma20"] > x["ma20"].shift(5)).astype(int)

    # 前序走势结构
    x["prev_1_down"] = (x["close"].shift(1) < x["close"].shift(2)).astype(int)
    x["prev_2_down"] = (x["close"].shift(2) < x["close"].shift(3)).astype(int)
    x["prev_3_down"] = (x["close"].shift(3) < x["close"].shift(4)).astype(int)
    x["prev_3day_all_down"] = ((x["prev_1_down"] == 1) & (x["prev_2_down"] == 1) & (x["prev_3_down"] == 1)).astype(int)

    x["prev_1_up"] = (x["close"].shift(1) > x["close"].shift(2)).astype(int)
    x["prev_2day_1up1down"] = (((x["prev_1_down"] == 1) & (x["prev_2_down"] == 0)) | ((x["prev_1_down"] == 0) & (x["prev_2_down"] == 1))).astype(int)

    # 前5日回撤
    rolling_max_5 = x["close"].shift(1).rolling(5).max()
    x["prev_5day_drawdown"] = safe_div(x["close"].shift(1) - rolling_max_5, rolling_max_5)

    # ========== 通达信砖型图 ==========
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

    # 砖柱颜色定义
    x["brick_red"] = (x["brick"] > x["brick_prev"]).astype(int)
    x["brick_green"] = (x["brick"] < x["brick_prev"]).astype(int)

    # 砖柱长度定义：真正的柱体长度 = |今天砖值 - 昨天砖值|
    x["brick_bar_len"] = (x["brick"] - x["brick_prev"]).abs()
    x["brick_red_len"] = np.where(x["brick"] > x["brick_prev"], x["brick"] - x["brick_prev"], 0.0)
    x["brick_green_len"] = np.where(x["brick"] < x["brick_prev"], x["brick_prev"] - x["brick"], 0.0)

    # 今天红 / 昨天绿 / 翻转
    x["today_red"] = (x["brick_red_len"] > 0).astype(int)
    x["yesterday_green"] = (x["brick_green_len"].shift(1) > 0).astype(int)
    x["today_red_turn"] = ((x["today_red"] == 1) & (x["yesterday_green"] == 1)).astype(int)

    # 关键比例：今天红柱长度 / 昨天柱体长度；如果昨天必须是绿柱，则昨天柱体就是昨天绿柱长度
    prev_bar_len = x["brick_bar_len"].shift(1)
    prev_green_len = x["brick_green_len"].shift(1)

    x["brick_ratio_vs_prev_bar"] = safe_div(x["brick_red_len"], prev_bar_len)
    x["brick_ratio_vs_prev_green"] = safe_div(x["brick_red_len"], prev_green_len)

    # 绝对增量
    x["brick_delta"] = x["brick"] - x["brick_prev"]

    # 连续绿柱天数（往前数，到昨天为止）
    green_flag = (x["brick_green_len"] > 0).astype(int)
    streak = np.zeros(len(x), dtype=int)
    for i in range(1, len(x)):
        if green_flag.iloc[i]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 0
    x["green_streak"] = streak
    x["prev_green_streak"] = x["green_streak"].shift(1)

    return x


def preprocess_all_data(data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    feature_map = {}
    total = len(data_map)
    for i, (code, df) in enumerate(data_map.items(), 1):
        try:
            feature_map[code] = add_features(df)
        except Exception as e:
            print(f"特征构建失败: {code} -> {e}")

        if i % 500 == 0 or i == total:
            print(f"特征构建进度: {i}/{total}")
    return feature_map


# =========================================================
# 信号条件
# =========================================================
def apply_prev_structure_filter(x: pd.DataFrame, prev_structure: Optional[str]) -> pd.Series:
    if prev_structure is None:
        return pd.Series(True, index=x.index)

    if prev_structure == "3down":
        return x["prev_3day_all_down"] == 1

    if prev_structure == "2down1up":
        # 一个简单近似：最近两天中至少1天跌，且不要求三连跌
        return x["prev_2day_1up1down"] == 1

    if prev_structure == "5day_drawdown_gt3":
        return x["prev_5day_drawdown"] <= -0.03

    if prev_structure == "5day_drawdown_gt5":
        return x["prev_5day_drawdown"] <= -0.05

    return pd.Series(True, index=x.index)


def apply_ma_filter(x: pd.DataFrame, ma_filter: Optional[str]) -> pd.Series:
    if ma_filter is None:
        return pd.Series(True, index=x.index)
    if ma_filter == "close_above_ma20":
        return x["close_above_ma20"] == 1
    if ma_filter == "ma5>ma10>ma20":
        return x["ma5_gt_ma10_gt_ma20"] == 1
    if ma_filter == "close_above_ma60":
        return x["close"] > x["ma60"]
    return pd.Series(True, index=x.index)


def build_signal_df(feature_map: Dict[str, pd.DataFrame], signal_params: Dict[str, Any]) -> pd.DataFrame:
    frames = []
    total = len(feature_map)

    for i, (code, x) in enumerate(feature_map.items(), 1):
        df = x.copy()
        cond = pd.Series(True, index=df.index)

        # 基础要求：数据有效
        cond &= df["brick"].notna()
        cond &= df["brick_prev"].notna()
        cond &= df["close"].notna()
        cond &= (df["close"] > 0)

        # 1) 今天是否必须红柱
        require_today_red = signal_params.get("require_today_red", True)
        if require_today_red:
            cond &= (df["today_red"] == 1)

        # 2) 昨天是否必须绿柱
        require_yesterday_green = signal_params.get("require_yesterday_green", False)
        if require_yesterday_green:
            cond &= (df["yesterday_green"] == 1)

        # 3) 今天是否必须是翻转红柱（今天红且昨天绿）
        require_today_red_turn = signal_params.get("require_today_red_turn", False)
        if require_today_red_turn:
            cond &= (df["today_red_turn"] == 1)

        # 4) 砖长比例阈值
        brick_ratio_threshold = signal_params.get("brick_ratio_threshold", None)
        ratio_ref = signal_params.get("brick_ratio_ref", "prev_bar")  # prev_bar / prev_green

        if brick_ratio_threshold is not None:
            if ratio_ref == "prev_green":
                cond &= (df["brick_ratio_vs_prev_green"] >= brick_ratio_threshold)
            else:
                cond &= (df["brick_ratio_vs_prev_bar"] >= brick_ratio_threshold)

        # 5) 砖型图绝对值门槛
        brick_abs_threshold = signal_params.get("brick_abs_threshold", None)
        if brick_abs_threshold is not None:
            cond &= (df["brick"] >= brick_abs_threshold)

        # 6) 砖型图增量门槛
        brick_delta_threshold = signal_params.get("brick_delta_threshold", None)
        if brick_delta_threshold is not None:
            cond &= (df["brick_delta"] >= brick_delta_threshold)

        # 7) 趋势线 > 多空线
        require_trend_gt_bullbear = signal_params.get("require_trend_gt_bullbear", False)
        if require_trend_gt_bullbear:
            cond &= (df["trend_gt_bullbear"] == 1)

        # 8) K线必须阳线吗
        require_yang = signal_params.get("require_yang", False)
        if require_yang:
            cond &= (df["close"] > df["open"])

        # 9) 收盘位置限制
        close_pos_threshold = signal_params.get("close_pos_threshold", None)
        if close_pos_threshold is not None:
            cond &= (df["close_pos"] >= close_pos_threshold)

        # 10) 实体占振幅比例
        body_pct_range_threshold = signal_params.get("body_pct_range_threshold", None)
        if body_pct_range_threshold is not None:
            cond &= (df["body_pct_range"] >= body_pct_range_threshold)

        # 11) 前几日涨跌结构
        prev_structure = signal_params.get("prev_structure", None)
        cond &= apply_prev_structure_filter(df, prev_structure)

        # 12) 连续绿柱天数
        min_prev_green_streak = signal_params.get("min_prev_green_streak", None)
        if min_prev_green_streak is not None:
            cond &= (df["prev_green_streak"] >= min_prev_green_streak)

        # 13) 量能
        vol_ratio_5_20_threshold = signal_params.get("vol_ratio_5_20_threshold", None)
        if vol_ratio_5_20_threshold is not None:
            cond &= (df["vol_ratio_5_20"] >= vol_ratio_5_20_threshold)

        vol_today_ratio_5_threshold = signal_params.get("vol_today_ratio_5_threshold", None)
        if vol_today_ratio_5_threshold is not None:
            cond &= (df["vol_today_ratio_5"] >= vol_today_ratio_5_threshold)

        amount_threshold = signal_params.get("amount_threshold", None)
        if amount_threshold is not None:
            cond &= (df["amount"] >= amount_threshold)

        # 14) MA过滤
        ma_filter = signal_params.get("ma_filter", None)
        cond &= apply_ma_filter(df, ma_filter)

        require_ma20_slope_up = signal_params.get("require_ma20_slope_up", False)
        if require_ma20_slope_up:
            cond &= (df["ma20_slope_up"] == 1)

        # 最终有效性
        cond &= df["date"].notna()
        cond &= np.isfinite(df["close"])
        cond &= np.isfinite(df["low"])
        cond &= np.isfinite(df["high"])

        sig = df.loc[cond].copy()
        if not sig.empty:
            sig["code"] = code
            frames.append(sig)

        if i % 500 == 0 or i == total:
            print(f"信号构建进度: {i}/{total}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "code"]).reset_index(drop=True)
    return out


# =========================================================
# 排序与每天取前N
# =========================================================
def build_sort_score(df: pd.DataFrame, sort_by: Optional[str]) -> pd.Series:
    if sort_by is None or sort_by == "none":
        return pd.Series(np.arange(len(df)), index=df.index, dtype=float)

    if sort_by == "brick_value":
        return df["brick"].fillna(-1e18)

    if sort_by == "brick_delta":
        return df["brick_delta"].fillna(-1e18)

    if sort_by == "vol_ratio":
        return df["vol_ratio_5_20"].fillna(-1e18)

    if sort_by == "trend_strength":
        return (df["trend_line"] - df["bull_bear_line"]).fillna(-1e18)

    if sort_by == "prev_drawdown":
        # 跌得越多，值越小，这里取负号让“回撤更深”排前
        return (-df["prev_5day_drawdown"]).fillna(-1e18)

    if sort_by == "composite":
        z1 = (df["brick_delta"] - df["brick_delta"].mean()) / (df["brick_delta"].std() + EPS)
        z2 = (df["vol_ratio_5_20"] - df["vol_ratio_5_20"].mean()) / (df["vol_ratio_5_20"].std() + EPS)
        z3 = ((df["trend_line"] - df["bull_bear_line"]) - (df["trend_line"] - df["bull_bear_line"]).mean()) / ((df["trend_line"] - df["bull_bear_line"]).std() + EPS)
        z4 = (df["close_pos"] - df["close_pos"].mean()) / (df["close_pos"].std() + EPS)
        return (0.35 * z1 + 0.25 * z2 + 0.20 * z3 + 0.20 * z4).fillna(-1e18)

    return pd.Series(np.arange(len(df)), index=df.index, dtype=float)


def apply_daily_selection(signal_df: pd.DataFrame, select_params: Dict[str, Any]) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df

    top_n = select_params.get("top_n", None)
    sort_by = select_params.get("sort_by", None)

    df = signal_df.copy()
    df["sort_score"] = build_sort_score(df, sort_by)

    if top_n is None:
        return df.sort_values(["date", "sort_score"], ascending=[True, False]).reset_index(drop=True)

    selected = (
        df.sort_values(["date", "sort_score"], ascending=[True, False])
          .groupby("date", group_keys=False)
          .head(top_n)
          .reset_index(drop=True)
    )
    return selected


# =========================================================
# 单笔交易回测
# =========================================================
def parse_stop_mode(stop_mode: str) -> Tuple[str, Optional[float]]:
    if stop_mode.startswith("fixed_pct_"):
        try:
            pct = float(stop_mode.replace("fixed_pct_", ""))
            return "fixed_pct", pct
        except Exception:
            return stop_mode, None

    if stop_mode.startswith("atr_"):
        try:
            multiple = float(stop_mode.replace("atr_", ""))
            return "atr", multiple
        except Exception:
            return stop_mode, None

    return stop_mode, None


def get_stop_price(row: pd.Series, stop_mode: str) -> Optional[float]:
    mode, value = parse_stop_mode(stop_mode)
    entry_close = row["close"]

    if not np.isfinite(entry_close) or entry_close <= 0:
        return None

    if mode == "none":
        return None
    if mode == "prev_day_low":
        return row.get("low", np.nan) if False else row.get("prev_day_low_for_trade", np.nan)
    if mode == "entry_low":
        return row.get("entry_low_for_trade", np.nan)
    if mode == "prev2_day_low":
        return row.get("prev2_day_low_for_trade", np.nan)

    if mode == "tighter_of_two":
        a = row.get("entry_low_for_trade", np.nan)
        b = row.get("prev_day_low_for_trade", np.nan)
        vals = [v for v in [a, b] if np.isfinite(v) and v > 0]
        return max(vals) if vals else None

    if mode == "looser_of_two":
        a = row.get("entry_low_for_trade", np.nan)
        b = row.get("prev_day_low_for_trade", np.nan)
        vals = [v for v in [a, b] if np.isfinite(v) and v > 0]
        return min(vals) if vals else None

    if mode == "fixed_pct" and value is not None:
        return entry_close * (1 - value)

    if mode == "atr" and value is not None:
        atr = row.get("atr14", np.nan)
        if np.isfinite(atr) and atr > 0:
            return entry_close - value * atr
        return None

    return None


def safe_return(exit_price: float, entry_price: float) -> float:
    if not np.isfinite(exit_price) or not np.isfinite(entry_price) or entry_price <= 0:
        return np.nan
    return exit_price / entry_price - 1


def simulate_one_trade(full_df: pd.DataFrame,
                       entry_idx: int,
                       max_hold_days: int,
                       stop_mode: str,
                       stop_trigger_mode: str,
                       take_profit_mode: str,
                       take_profit_threshold: float,
                       take_profit_sell_timing: str) -> Tuple[float, int, str, Optional[pd.Timestamp], Optional[float]]:
    """
    入场：信号当日收盘
    出场：
      - 止损：盘中/收盘
      - 止盈：触及或收盘达到阈值
      - 到期：max_hold_days 后收盘
    """
    n = len(full_df)
    entry_row = full_df.iloc[entry_idx]
    entry_close = entry_row["close"]

    if not np.isfinite(entry_close) or entry_close <= 0:
        return np.nan, 0, "无效入场价", None, np.nan

    tmp_row = entry_row.copy()
    tmp_row["entry_low_for_trade"] = entry_row["low"]
    tmp_row["prev_day_low_for_trade"] = full_df.iloc[entry_idx - 1]["low"] if entry_idx - 1 >= 0 else entry_row["low"]
    tmp_row["prev2_day_low_for_trade"] = full_df.iloc[entry_idx - 2]["low"] if entry_idx - 2 >= 0 else tmp_row["prev_day_low_for_trade"]

    stop_price = get_stop_price(tmp_row, stop_mode)
    last_j = min(entry_idx + max_hold_days, n - 1)

    for j in range(entry_idx + 1, last_j + 1):
        row = full_df.iloc[j]
        high_j = row["high"]
        low_j = row["low"]
        close_j = row["close"]
        open_j = row["open"]

        # 1) 止损
        if stop_price is not None and np.isfinite(stop_price) and stop_price > 0:
            if stop_trigger_mode == "intraday":
                if np.isfinite(low_j) and low_j <= stop_price:
                    return safe_return(stop_price, entry_close), j - entry_idx, "止损", row["date"], stop_price
            elif stop_trigger_mode == "close":
                if np.isfinite(close_j) and close_j <= stop_price:
                    return safe_return(close_j, entry_close), j - entry_idx, "收盘止损", row["date"], close_j

        # 2) 止盈
        if take_profit_mode != "none" and take_profit_threshold is not None and take_profit_threshold > 0:
            target_price = entry_close * (1 + take_profit_threshold)

            hit_take_profit = False
            if take_profit_mode == "high_touch_same_day":
                hit_take_profit = np.isfinite(high_j) and high_j >= target_price
                if hit_take_profit:
                    # 当天触及就按目标价卖
                    return take_profit_threshold, j - entry_idx, "触及止盈", row["date"], target_price

            elif take_profit_mode in ("close_profit_next_close", "close_profit_next_open"):
                hit_take_profit = np.isfinite(close_j) and close_j >= target_price
                if hit_take_profit:
                    sell_idx = min(j + 1, n - 1)

                    if take_profit_sell_timing == "next_open":
                        px = full_df.iloc[sell_idx]["open"]
                        reason = "收盘达标次日开盘止盈"
                    else:
                        px = full_df.iloc[sell_idx]["close"]
                        reason = "收盘达标次日收盘止盈"

                    if np.isfinite(px) and px > 0:
                        return safe_return(px, entry_close), sell_idx - entry_idx, reason, full_df.iloc[sell_idx]["date"], px

        # 否则继续持有

    # 3) 到期卖出
    close_last = full_df.iloc[last_j]["close"]
    return safe_return(close_last, entry_close), last_j - entry_idx, "到期卖出", full_df.iloc[last_j]["date"], close_last


def backtest_trades(feature_map: Dict[str, pd.DataFrame],
                    signal_df: pd.DataFrame,
                    exit_params: Dict[str, Any]) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame()

    rows = []

    grouped_signals = signal_df.groupby("code")
    for code, sig in grouped_signals:
        if code not in feature_map:
            continue

        full_df = feature_map[code].reset_index(drop=True)
        # 建立日期->索引映射
        date_to_idx = pd.Series(full_df.index.values, index=full_df["date"]).to_dict()

        for _, row in sig.iterrows():
            d = row["date"]
            if d not in date_to_idx:
                continue

            entry_idx = int(date_to_idx[d])

            ret, hold_days, exit_reason, exit_date, exit_price = simulate_one_trade(
                full_df=full_df,
                entry_idx=entry_idx,
                max_hold_days=exit_params.get("max_hold_days", 3),
                stop_mode=exit_params.get("stop_mode", "none"),
                stop_trigger_mode=exit_params.get("stop_trigger_mode", "intraday"),
                take_profit_mode=exit_params.get("take_profit_mode", "none"),
                take_profit_threshold=exit_params.get("take_profit_threshold", 0.0),
                take_profit_sell_timing=exit_params.get("take_profit_sell_timing", "next_close"),
            )

            rows.append({
                "date": d,
                "code": code,
                "entry_price": row["close"],
                "exit_date": exit_date,
                "exit_price": exit_price,
                "ret": ret,
                "hold_days": hold_days,
                "exit_reason": exit_reason,
                "brick": row.get("brick", np.nan),
                "brick_delta": row.get("brick_delta", np.nan),
                "brick_red_len": row.get("brick_red_len", np.nan),
                "brick_ratio_vs_prev_bar": row.get("brick_ratio_vs_prev_bar", np.nan),
                "brick_ratio_vs_prev_green": row.get("brick_ratio_vs_prev_green", np.nan),
                "vol_ratio_5_20": row.get("vol_ratio_5_20", np.nan),
                "trend_strength": row.get("trend_line", np.nan) - row.get("bull_bear_line", np.nan),
                "close_pos": row.get("close_pos", np.nan),
                "sort_score": row.get("sort_score", np.nan),
            })

    trade_df = pd.DataFrame(rows)
    if trade_df.empty:
        return trade_df

    trade_df = trade_df.dropna(subset=["ret"])
    trade_df = trade_df[np.isfinite(trade_df["ret"])].copy()
    trade_df = trade_df.sort_values(["date", "code"]).reset_index(drop=True)
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
    """
    用 log1p 防止累计收益爆炸到 inf。
    """
    if len(ret_series) == 0:
        return np.nan, np.nan

    r = ret_series.clip(lower=-0.999999)
    log_eq = np.log1p(r).cumsum()

    # 最终净值
    final_log = float(log_eq.iloc[-1])
    capped_final_log = min(final_log, 700)  # 防止exp溢出
    final_equity = float(np.exp(capped_final_log))

    # 最大回撤：用log空间算更稳
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
    for d, g in trade_df.groupby("date"):
        g = g.copy()

        if basket_weight == "score" and "sort_score" in g.columns:
            score = g["sort_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            score = score - score.min() + 1e-6
            w = score / score.sum() if score.sum() > 0 else np.repeat(1 / len(g), len(g))
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
# 实验执行器
# =========================================================
def run_trade_experiment(feature_map: Dict[str, pd.DataFrame],
                         signal_params: Dict[str, Any],
                         exit_params: Dict[str, Any],
                         select_params: Dict[str, Any],
                         exp_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_df = build_signal_df(feature_map, signal_params)
    signal_df = apply_daily_selection(signal_df, select_params)

    trade_df = backtest_trades(feature_map, signal_df, exit_params)

    extra = {}
    extra.update(signal_params)
    extra.update(exit_params)
    extra.update(select_params)

    summary_df = summarize_trades(trade_df, extra)
    summary_df.insert(0, "实验名称", exp_name)

    basket_df = build_daily_basket(trade_df, basket_weight=select_params.get("basket_weight", "equal"))
    basket_summary_df = summarize_basket(basket_df, extra, exp_name)

    return summary_df, trade_df, basket_summary_df


def best_row_by_mean_then_sharpe(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    cols = [c for c in ["平均每笔收益", "近似夏普", "胜率", "样本数"] if c in df.columns]
    return df.sort_values(cols, ascending=[False, False, False, False]).iloc[0]


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
# 主流程实验设计
# 除“样本过滤类变量”不做，其他都做
# =========================================================
def main():
    print("开始加载数据...")
    data_map = load_all_data(DATA_DIR)
    print(f"有效股票数: {len(data_map)}")

    print("开始构建特征...")
    feature_map = preprocess_all_data(data_map)

    # -----------------------------
    # 基准参数
    # -----------------------------
    base_signal = {
        "require_today_red": True,
        "require_yesterday_green": True,
        "require_today_red_turn": False,     # 注意：和昨天绿柱分开
        "brick_ratio_threshold": 1.0,
        "brick_ratio_ref": "prev_green",     # 更符合“红柱长度 > 前一天绿柱长度 * 比例”
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

    report_map = {}
    basket_report_map = {}

    # =====================================================
    # 实验1：砖长比例阈值
    # =====================================================
    print("\n开始实验1：砖长比例阈值对比...")
    exp1_rows = []
    for v in [0.66, 0.8, 1.0, 1.2, 1.5]:
        sp = dict(base_signal, brick_ratio_threshold=v, brick_ratio_ref="prev_green", require_yesterday_green=True)
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, f"实验1_砖长比例_{v:.2f}")
        exp1_rows.append(summary)
        trades.to_csv(os.path.join(OUTPUT_DIR, f"实验1_砖长比例_{v:.2f}_逐笔交易.csv"), index=False, encoding="utf-8-sig")
    exp1 = pd.concat(exp1_rows, ignore_index=True)
    report_map["实验1_砖长比例阈值对比"] = exp1

    # =====================================================
    # 实验2：昨天是否必须绿柱
    # =====================================================
    print("\n开始实验2：昨天是否必须绿柱...")
    exp2_rows = []
    for v in [True, False]:
        sp = dict(base_signal, require_yesterday_green=v, require_today_red=True, require_today_red_turn=False, brick_ratio_ref="prev_bar", brick_ratio_threshold=1.0)
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, f"实验2_昨天绿柱_{v}")
        exp2_rows.append(summary)
    exp2 = pd.concat(exp2_rows, ignore_index=True)
    report_map["实验2_昨天是否必须绿柱"] = exp2

    # =====================================================
    # 实验3：今天是否必须红柱翻转
    # =====================================================
    print("\n开始实验3：今天是否必须红柱翻转...")
    exp3_rows = []
    for v in [True, False]:
        sp = dict(base_signal, require_today_red_turn=v, require_yesterday_green=False, require_today_red=True, brick_ratio_ref="prev_bar", brick_ratio_threshold=1.0)
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, f"实验3_今天必须翻转_{v}")
        exp3_rows.append(summary)
    exp3 = pd.concat(exp3_rows, ignore_index=True)
    report_map["实验3_今天是否必须红柱翻转"] = exp3

    # =====================================================
    # 实验4：砖型图绝对值门槛
    # =====================================================
    print("\n开始实验4：砖型图绝对值门槛...")
    exp4_rows = []
    for v in [None, 10, 20, 30]:
        sp = dict(base_signal, brick_abs_threshold=v)
        name = f"实验4_砖型图绝对值_{'None' if v is None else v}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, name)
        exp4_rows.append(summary)
    exp4 = pd.concat(exp4_rows, ignore_index=True)
    report_map["实验4_砖型图绝对值门槛"] = exp4

    # =====================================================
    # 实验5：砖型图增量门槛
    # =====================================================
    print("\n开始实验5：砖型图增量门槛...")
    exp5_rows = []
    for v in [None, 3, 5, 10, 15]:
        sp = dict(base_signal, brick_delta_threshold=v)
        name = f"实验5_砖型图增量_{'None' if v is None else v}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, name)
        exp5_rows.append(summary)
    exp5 = pd.concat(exp5_rows, ignore_index=True)
    report_map["实验5_砖型图增量门槛"] = exp5

    # =====================================================
    # 实验6：趋势线 > 多空线
    # =====================================================
    print("\n开始实验6：趋势线>多空线是否必须...")
    exp6_rows = []
    for v in [False, True]:
        sp = dict(base_signal, require_trend_gt_bullbear=v)
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, f"实验6_趋势线大于多空线_{v}")
        exp6_rows.append(summary)
    exp6 = pd.concat(exp6_rows, ignore_index=True)
    report_map["实验6_趋势线>多空线是否必须"] = exp6

    # =====================================================
    # 实验7：K线必须为阳线吗
    # =====================================================
    print("\n开始实验7：K线是否必须阳线...")
    exp7_rows = []
    for v in [False, True]:
        sp = dict(base_signal, require_yang=v)
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, f"实验7_必须阳线_{v}")
        exp7_rows.append(summary)
    exp7 = pd.concat(exp7_rows, ignore_index=True)
    report_map["实验7_K线必须为阳线吗"] = exp7

    # =====================================================
    # 实验8：收盘位置限制
    # =====================================================
    print("\n开始实验8：收盘位置限制...")
    exp8_rows = []
    for v in [None, 0.6, 0.7, 0.8]:
        sp = dict(base_signal, close_pos_threshold=v)
        name = f"实验8_收盘位置_{'None' if v is None else v}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, name)
        exp8_rows.append(summary)
    exp8 = pd.concat(exp8_rows, ignore_index=True)
    report_map["实验8_收盘位置限制"] = exp8

    # =====================================================
    # 实验9：实体长度限制
    # =====================================================
    print("\n开始实验9：实体长度限制...")
    exp9_rows = []
    for v in [None, 0.3, 0.5, 0.7]:
        sp = dict(base_signal, body_pct_range_threshold=v)
        name = f"实验9_实体占比_{'None' if v is None else v}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, name)
        exp9_rows.append(summary)
    exp9 = pd.concat(exp9_rows, ignore_index=True)
    report_map["实验9_实体长度限制"] = exp9

    # =====================================================
    # 实验10：前几日涨跌结构
    # =====================================================
    print("\n开始实验10：前几日涨跌结构...")
    exp10_rows = []
    for v in [None, "3down", "2down1up", "5day_drawdown_gt3", "5day_drawdown_gt5"]:
        sp = dict(base_signal, prev_structure=v)
        name = f"实验10_前序结构_{'None' if v is None else v}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, name)
        exp10_rows.append(summary)
    exp10 = pd.concat(exp10_rows, ignore_index=True)
    report_map["实验10_前几日涨跌结构"] = exp10

    # =====================================================
    # 实验11：5日均量 / 20日均量阈值
    # =====================================================
    print("\n开始实验11：5日均量/20日均量阈值...")
    exp11_rows = []
    for v in [None, 1.0, 1.1, 1.2, 1.5]:
        sp = dict(base_signal, vol_ratio_5_20_threshold=v)
        name = f"实验11_5日量20日量_{'None' if v is None else v}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, name)
        exp11_rows.append(summary)
    exp11 = pd.concat(exp11_rows, ignore_index=True)
    report_map["实验11_5日均量除20日均量阈值"] = exp11

    # =====================================================
    # 实验12：当天成交量 / 5日均量阈值
    # =====================================================
    print("\n开始实验12：当天量/5日均量阈值...")
    exp12_rows = []
    for v in [None, 1.0, 1.2, 1.5]:
        sp = dict(base_signal, vol_today_ratio_5_threshold=v)
        name = f"实验12_当天量5日量_{'None' if v is None else v}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, name)
        exp12_rows.append(summary)
    exp12 = pd.concat(exp12_rows, ignore_index=True)
    report_map["实验12_当天成交量除5日均量阈值"] = exp12

    # =====================================================
    # 实验13：成交额门槛
    # =====================================================
    print("\n开始实验13：成交额门槛...")
    exp13_rows = []
    for v in [None, 5e7, 1e8, 3e8]:
        sp = dict(base_signal, amount_threshold=v)
        name = f"实验13_成交额门槛_{'None' if v is None else int(v)}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, name)
        exp13_rows.append(summary)
    exp13 = pd.concat(exp13_rows, ignore_index=True)
    report_map["实验13_成交额门槛"] = exp13

    # =====================================================
    # 实验14：均线过滤
    # =====================================================
    print("\n开始实验14：均线过滤...")
    exp14_rows = []
    for ma_filter, slope_up in itertools.product([None, "close_above_ma20", "ma5>ma10>ma20", "close_above_ma60"], [False, True]):
        sp = dict(base_signal, ma_filter=ma_filter, require_ma20_slope_up=slope_up)
        name = f"实验14_MA过滤_{ma_filter}_MA20斜率上升_{slope_up}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, name)
        exp14_rows.append(summary)
    exp14 = pd.concat(exp14_rows, ignore_index=True)
    report_map["实验14_均线与斜率过滤"] = exp14

    # =====================================================
    # 实验15：止损方式
    # =====================================================
    print("\n开始实验15：止损方式...")
    exp15_rows = []
    stop_modes = ["none", "prev_day_low", "entry_low", "prev2_day_low", "tighter_of_two", "looser_of_two", "fixed_pct_0.03", "atr_1.0", "atr_1.5"]
    for v in stop_modes:
        ep = dict(base_exit, stop_mode=v)
        name = f"实验15_止损方式_{v}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, base_signal, ep, base_select, name)
        exp15_rows.append(summary)
    exp15 = pd.concat(exp15_rows, ignore_index=True)
    report_map["实验15_止损方式"] = exp15

    # =====================================================
    # 实验16：盘中止损 vs 收盘止损
    # =====================================================
    print("\n开始实验16：止损触发方式...")
    exp16_rows = []
    for v in ["intraday", "close"]:
        ep = dict(base_exit, stop_mode="prev_day_low", stop_trigger_mode=v)
        name = f"实验16_止损触发_{v}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, base_signal, ep, base_select, name)
        exp16_rows.append(summary)
    exp16 = pd.concat(exp16_rows, ignore_index=True)
    report_map["实验16_盘中止损还是收盘止损"] = exp16

    # =====================================================
    # 实验17：止盈方式
    # =====================================================
    print("\n开始实验17：止盈方式...")
    exp17_rows = []
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
        name = f"实验17_止盈_{mode}_{th}_{timing}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, base_signal, ep, base_select, name)
        exp17_rows.append(summary)
    exp17 = pd.concat(exp17_rows, ignore_index=True)
    report_map["实验17_止盈方式与阈值"] = exp17

    # =====================================================
    # 实验18：持有期
    # =====================================================
    print("\n开始实验18：最长持有天数...")
    exp18_rows = []
    for v in [1, 2, 3, 5]:
        ep = dict(base_exit, max_hold_days=v)
        name = f"实验18_最长持有_{v}天"
        summary, trades, basket_summary = run_trade_experiment(feature_map, base_signal, ep, base_select, name)
        exp18_rows.append(summary)
    exp18 = pd.concat(exp18_rows, ignore_index=True)
    report_map["实验18_最长持有天数"] = exp18

    # =====================================================
    # 实验19：排序与每天取前N
    # =====================================================
    print("\n开始实验19：排序与每天取前N...")
    exp19_rows = []
    for sort_by, top_n in itertools.product(["brick_value", "brick_delta", "vol_ratio", "trend_strength", "prev_drawdown", "composite"], [5, 10, 20]):
        sel = dict(base_select, sort_by=sort_by, top_n=top_n, basket_weight="equal")
        name = f"实验19_排序_{sort_by}_前{top_n}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, base_signal, base_exit, sel, name)
        exp19_rows.append(summary)
    exp19 = pd.concat(exp19_rows, ignore_index=True)
    report_map["实验19_排序与每天取前N"] = exp19

    # =====================================================
    # 实验20：组合层权重（同日篮子）
    # 这里用“同日选出后按单笔收益聚合”为简化近似，不做真实持仓资金占用模拟
    # =====================================================
    print("\n开始实验20：组合层权重近似对比...")
    exp20_trade_rows = []
    exp20_basket_rows = []
    for basket_weight in ["equal", "score"]:
        sel = dict(base_select, sort_by="composite", top_n=10, basket_weight=basket_weight)
        name = f"实验20_权重_{basket_weight}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, base_signal, base_exit, sel, name)
        exp20_trade_rows.append(summary)
        exp20_basket_rows.append(basket_summary)
    exp20 = pd.concat(exp20_trade_rows, ignore_index=True)
    exp20_basket = pd.concat(exp20_basket_rows, ignore_index=True)
    report_map["实验20_组合层权重近似对比_单笔"] = exp20
    basket_report_map["实验20_组合层权重近似对比_日篮子"] = exp20_basket

    # =====================================================
    # 实验21：连续绿几天后翻红，且红柱长度 > 前一天绿柱长度 * 比例
    # =====================================================
    print("\n开始实验21：连续绿几天后翻红 + 红柱长度比例...")
    exp21_rows = []
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
        name = f"实验21_连续绿{streak}天后翻红_比例{ratio}"
        summary, trades, basket_summary = run_trade_experiment(feature_map, sp, base_exit, base_select, name)
        exp21_rows.append(summary)
    exp21 = pd.concat(exp21_rows, ignore_index=True)
    report_map["实验21_连续绿几天后翻红且红柱长度比例"] = exp21

    # =====================================================
    # 保存结果
    # =====================================================
    for name, df in report_map.items():
        safe_name = name.replace("/", "_")
        df.to_csv(os.path.join(OUTPUT_DIR, f"{safe_name}.csv"), index=False, encoding="utf-8-sig")

    for name, df in basket_report_map.items():
        safe_name = name.replace("/", "_")
        df.to_csv(os.path.join(OUTPUT_DIR, f"{safe_name}.csv"), index=False, encoding="utf-8-sig")

    conclusion_df = generate_conclusion(report_map)
    conclusion_df.to_csv(os.path.join(OUTPUT_DIR, "自动结论.csv"), index=False, encoding="utf-8-sig")

    print("\n================= 自动结论 =================")
    if conclusion_df.empty:
        print("暂无结论，请检查样本数是否不足。")
    else:
        print(conclusion_df.to_string(index=False))

    print("\n结果文件已保存到：")
    print(OUTPUT_DIR)
    print("\n重点查看：")
    for k in report_map.keys():
        print(f"- {k}.csv")
    for k in basket_report_map.keys():
        print(f"- {k}.csv")
    print("- 自动结论.csv")


if __name__ == "__main__":
    main()