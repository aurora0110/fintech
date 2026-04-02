#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
B3 自定义 K 线组合回测（严格修正版）

========================
一、买入信号定义
========================
在第 N 日收盘后检查，满足以下全部条件则发出买入信号，
并在 N+1 日开盘买入：

1) N 日（B3）：
   - N 日是红砖（brick_red = True）
   - N 日涨幅 < 2%
   - N 日振幅 < 5%
   - N 日成交量 < N-1 日成交量 * 0.9

2) N-1 日（B2）：
   - N-1 日是红砖（brick_red = True）
   - N-1 日涨幅 > 4%
   - N-1 日上影线长度 < 实体长度 / 4
   - N-1 日 J < 80
   - N-1 日成交量 > N-2 日成交量 * 2

3) N-2 日：
   - N-2 日是绿砖（brick_green = True）
   - N-2 日是“最近几块绿砖中最低的位置”

   这里“最近几块绿砖中最低的位置”被严格定义为：
   以 N-2 日为结尾，向前回溯直到遇到第一个非绿砖为止，
   得到“最近一段连续绿砖区间”，要求 N-2 日 LOW
   等于这段连续绿砖区间中的最低 LOW。

4) N-5 到 N-3 日：
   - N-5, N-4, N-3 都是绿砖
   - 且 brick 值逐渐下降：
     brick[N-5] >= brick[N-4] >= brick[N-3] >= brick[N-2]

========================
二、执行规则
========================
- 信号：N 日收盘后产生
- 买入：N+1 日开盘价
- 止损价：min(买入日 LOW, 信号日 LOW) * (1 - stop_loss_pct)

========================
三、回测策略
========================
策略 1：固定持有天数 + 止损
- 收盘价 <= 止损价：当日收盘触发，次日开盘卖出
- 持有天数 >= hold_days：当日收盘触发，次日开盘卖出
- 不包含止盈

策略 2：止盈止损
- 收盘价 <= 止损价：当日收盘触发，次日开盘卖出
- 收盘价 >= 买入价 * (1 + take_profit_pct)：当日收盘触发，次日开盘卖出

========================
四、重要说明
========================
- 所有“触发卖出”都统一为：收盘判定，次日开盘执行
- 若次日开盘涨跌停，则按涨跌停价近似成交
- 所有交易统计口径统一为：一笔完整开仓到完整平仓记为 1 次交易
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================================================
# 基础参数
# =========================================================
INITIAL_CAPITAL = 100000000.0
FEE_RATE = 0.0003
SLIPPAGE = 0.001
MAX_POSITIONS = 10

BASE_DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data"

HOLD_DAYS_LIST = [2, 3, 4, 5, 10, 15, 20, 30, 60]
TAKE_PROFIT_LIST = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
STOP_LOSS_PCT = 0.05


# =========================================================
# 通用函数
# =========================================================
def find_latest_available_data_dir(root_dir: str, today_str: str = None) -> Tuple[Optional[str], Optional[str]]:
    root = Path(root_dir)
    candidates = []
    for date_dir in root.glob("20*"):
        if not date_dir.is_dir():
            continue
        # 跳过所有带 min 后缀的目录（1min、5min等）
        if "min" in date_dir.name:
            continue
        txt_count = len(list(date_dir.glob("*.txt")))
        if txt_count == 0:
            continue
        date_str = date_dir.name
        candidates.append((date_str, str(date_dir)))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0])
    if today_str:
        for date_str, data_dir in reversed(candidates):
            if date_str <= today_str:
                return date_str, data_dir
    return candidates[-1]


def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > 1e-12)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    return series.ewm(alpha=m / n, adjust=False).mean()


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


def get_limit_pct_by_code(code: str) -> float:
    """
    主板 10%，创业板/科创板 20%，北交所 30%
    """
    pure = code.split("#")[-1].upper()

    if pure.startswith(("300", "301", "688", "689")):
        return 0.20
    if pure.startswith(("8", "83", "87", "43", "92")):
        return 0.30
    return 0.10


def get_board_filtered_files(data_dir: str) -> List[str]:
    files = []
    for f in os.listdir(data_dir):
        if not f.endswith(".txt"):
            continue

        stock_code = f.replace(".txt", "")
        pure = stock_code.split("#")[-1].upper()

        # 跳过北交所/BJ
        if stock_code.upper().startswith("BJ"):
            continue
        if pure.startswith(("8", "83", "87", "43", "92")):
            continue

        files.append(f)
    return files


# =========================================================
# 指标计算
# =========================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    close = x["CLOSE"]
    high = x["HIGH"]
    low = x["LOW"]

    x["涨跌幅"] = close.pct_change() * 100.0

    x["MA5"] = close.rolling(5).mean()
    x["MA10"] = close.rolling(10).mean()
    x["MA20"] = close.rolling(20).mean()
    x["MA30"] = close.rolling(30).mean()

    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-6) * 100.0
    x["K"] = rsv.ewm(com=2, adjust=False).mean()
    x["D"] = x["K"].ewm(com=2, adjust=False).mean()
    x["J"] = 3 * x["K"] - 2 * x["D"]

    return x


def add_brick_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=False)

    hhv4 = x["HIGH"].rolling(4).max()
    llv4 = x["LOW"].rolling(4).min()
    den4 = (hhv4 - llv4).replace(0, np.nan)

    var1a = safe_div((hhv4 - x["CLOSE"]), den4) * 100.0 - 90.0
    var2a = tdx_sma(pd.Series(var1a, index=x.index), 4, 1) + 100.0
    var3a = safe_div((x["CLOSE"] - llv4), den4) * 100.0
    var4a = tdx_sma(pd.Series(var3a, index=x.index), 6, 1)
    var5a = tdx_sma(var4a, 6, 1) + 100.0
    var6a = var5a - var2a

    x["brick"] = np.where(var6a > 4, var6a - 4, 0.0)
    x["brick_prev"] = x["brick"].shift(1)

    x["brick_red_len"] = np.where(
        x["brick"] > x["brick_prev"],
        x["brick"] - x["brick_prev"],
        0.0,
    )
    x["brick_green_len"] = np.where(
        x["brick"] < x["brick_prev"],
        x["brick_prev"] - x["brick"],
        0.0,
    )

    x["brick_red"] = x["brick_red_len"] > 0
    x["brick_green"] = x["brick_green_len"] > 0
    x["prev_green_streak"] = (
        pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(1)
    )

    x = x.set_index("日期")
    return x


# =========================================================
# 数据加载
# =========================================================
def load_one_stock(path: str, stock_code: str) -> Optional[pd.DataFrame]:
    df = None

    for encoding in ["gbk", "utf-8", "gb2312", "latin1"]:
        try:
            tmp = pd.read_csv(path, sep=r"\s+", encoding=encoding, skiprows=1)
            tmp = tmp[tmp["日期"].astype(str).str.match(r"^\d{4}")]
            if len(tmp.columns) >= 6:
                df = tmp
                break
        except Exception:
            continue

    if df is None or len(df.columns) < 6:
        return None

    df.columns = ["日期", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"] + list(df.columns[6:])
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"]).copy()

    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]).copy()
    df = df[(df["OPEN"] > 0) & (df["HIGH"] > 0) & (df["LOW"] > 0) & (df["CLOSE"] > 0)].copy()

    if df.empty:
        return None

    df = df.sort_values("日期").drop_duplicates(subset=["日期"], keep="last").set_index("日期")
    df["PREV_CLOSE"] = df["CLOSE"].shift(1)
    df["limit_pct"] = get_limit_pct_by_code(stock_code)
    df["stock_code"] = stock_code

    df = calculate_indicators(df)
    df = add_brick_features(df)

    return df


def load_all_stock_data(data_dir: str) -> Tuple[Dict[str, pd.DataFrame], Dict[pd.Timestamp, List[Tuple[str, dict, float]]]]:
    stock_data: Dict[str, pd.DataFrame] = {}
    daily_signals: Dict[pd.Timestamp, List[Tuple[str, dict, float]]] = {}

    files = get_board_filtered_files(data_dir)

    loaded_count = 0
    signal_count = 0

    for file in files:
        stock_code = file.replace(".txt", "")
        path = os.path.join(data_dir, file)

        try:
            df = load_one_stock(path, stock_code)
            if df is None or len(df) < 20:
                continue

            stock_data[stock_code] = df
            loaded_count += 1

            for i in range(5, len(df)):
                is_signal, details = check_custom_pattern(df, i)
                if is_signal:
                    signal_date = df.index[i]
                    signal_low = float(df.iloc[i]["LOW"])
                    daily_signals.setdefault(signal_date, []).append((stock_code, details, signal_low))
                    signal_count += 1

        except Exception:
            continue

    print(f"成功加载 {loaded_count} 只股票")
    print(f"共生成 {signal_count} 个买入信号")

    return stock_data, daily_signals


# =========================================================
# 买点定义
# =========================================================
def get_latest_continuous_green_segment(df: pd.DataFrame, end_idx: int) -> List[int]:
    """
    返回以 end_idx 为结尾的最近一段连续绿砖区间索引列表。
    若 end_idx 不是绿砖，返回空列表。
    """
    if end_idx < 0:
        return []

    if not bool(df.iloc[end_idx]["brick_green"]):
        return []

    seg = [end_idx]
    j = end_idx - 1
    while j >= 0 and bool(df.iloc[j]["brick_green"]):
        seg.append(j)
        j -= 1

    seg.reverse()
    return seg


def check_custom_pattern(df: pd.DataFrame, idx: int) -> Tuple[bool, Dict]:
    """
    严格按定义检查买点信号。
    idx 对应 N 日。
    """
    if idx < 5:
        return False, {}

    n = df.iloc[idx]
    n1 = df.iloc[idx - 1]
    n2 = df.iloc[idx - 2]
    n3 = df.iloc[idx - 3]
    n4 = df.iloc[idx - 4]
    n5 = df.iloc[idx - 5]

    needed_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "J", "brick", "brick_red", "brick_green"]
    for row in [n, n1, n2, n3, n4, n5]:
        if row[needed_cols].isna().any():
            return False, {}

    # -------------------------
    # N 日（B3）
    # -------------------------
    cond_n_red = bool(n["brick_red"])

    n_change_pct = ((n["CLOSE"] - n1["CLOSE"]) / n1["CLOSE"] * 100.0) if n1["CLOSE"] > 0 else np.nan
    cond_n_change = pd.notna(n_change_pct) and (n_change_pct < 2.0)

    n_amplitude = ((n["HIGH"] - n["LOW"]) / n["LOW"] * 100.0) if n["LOW"] > 0 else np.nan
    cond_n_amplitude = pd.notna(n_amplitude) and (n_amplitude < 5.0)

    cond_n_volume = n["VOLUME"] < n1["VOLUME"] * 0.9

    # -------------------------
    # N-1 日（B2）
    # -------------------------
    cond_n1_red = bool(n1["brick_red"])

    n1_change_pct = ((n1["CLOSE"] - n2["CLOSE"]) / n2["CLOSE"] * 100.0) if n2["CLOSE"] > 0 else np.nan
    cond_n1_change = pd.notna(n1_change_pct) and (n1_change_pct > 4.0)

    n1_body = n1["CLOSE"] - n1["OPEN"]
    n1_upper_shadow = n1["HIGH"] - max(n1["OPEN"], n1["CLOSE"])
    cond_n1_upper_shadow = (n1_body > 0) and (n1_upper_shadow < n1_body / 4.0)

    cond_n1_j = n1["J"] < 80.0
    cond_n1_volume = n1["VOLUME"] > n2["VOLUME"] * 2.0

    # -------------------------
    # N-2 日
    # -------------------------
    cond_n2_green = bool(n2["brick_green"])

    # 最近一段连续绿砖区间，以 N-2 结尾
    latest_green_seg = get_latest_continuous_green_segment(df, idx - 2)
    if not latest_green_seg:
        return False, {}

    seg_lows = df.iloc[latest_green_seg]["LOW"].to_numpy(dtype=float)
    seg_low_min = np.nanmin(seg_lows) if len(seg_lows) > 0 else np.nan
    cond_n2_is_lowest_in_latest_green_seg = pd.notna(seg_low_min) and np.isclose(n2["LOW"], seg_low_min)

    # -------------------------
    # N-5 ~ N-3 都是绿砖
    # -------------------------
    cond_n3_green = bool(n3["brick_green"])
    cond_n4_green = bool(n4["brick_green"])
    cond_n5_green = bool(n5["brick_green"])

    # brick 逐渐下降：N-5 >= N-4 >= N-3 >= N-2
    cond_brick_down_54 = n5["brick"] >= n4["brick"]
    cond_brick_down_43 = n4["brick"] >= n3["brick"]
    cond_brick_down_32 = n3["brick"] >= n2["brick"]

    all_conds = [
        cond_n_red,
        cond_n_change,
        cond_n_amplitude,
        cond_n_volume,
        cond_n1_red,
        cond_n1_change,
        cond_n1_upper_shadow,
        cond_n1_j,
        cond_n1_volume,
        cond_n2_green,
        cond_n2_is_lowest_in_latest_green_seg,
        cond_n3_green,
        cond_n4_green,
        cond_n5_green,
        cond_brick_down_54,
        cond_brick_down_43,
        cond_brick_down_32,
    ]

    if not all(all_conds):
        return False, {}

    details = {
        "n_change_pct": float(n_change_pct),
        "n_amplitude": float(n_amplitude),
        "n_vol_ratio": float(n["VOLUME"] / n1["VOLUME"]) if n1["VOLUME"] > 0 else np.nan,
        "n1_change_pct": float(n1_change_pct),
        "n1_body": float(n1_body),
        "n1_upper_shadow": float(n1_upper_shadow),
        "n1_vol_ratio": float(n1["VOLUME"] / n2["VOLUME"]) if n2["VOLUME"] > 0 else np.nan,
        "J_n": float(n["J"]),
        "J_n1": float(n1["J"]),
        "latest_green_seg_len": len(latest_green_seg),
        "n2_low": float(n2["LOW"]),
        "latest_green_seg_low_min": float(seg_low_min),
    }
    return True, details


# =========================================================
# 交易执行辅助
# =========================================================
def build_trading_calendar(stock_data: Dict[str, pd.DataFrame]) -> Tuple[List[pd.Timestamp], Dict[pd.Timestamp, int]]:
    all_dates = []
    for df in stock_data.values():
        all_dates.extend(df.index.tolist())
    all_dates = sorted(set(all_dates))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    return all_dates, date_to_idx


def build_pending_buy_signals(
    daily_signals: Dict[pd.Timestamp, List[Tuple[str, dict, float]]],
    all_dates: List[pd.Timestamp],
    date_to_idx: Dict[pd.Timestamp, int],
) -> Dict[pd.Timestamp, List[Tuple[str, dict, float]]]:
    pending = {}
    n = len(all_dates)

    for signal_date, sigs in daily_signals.items():
        idx = date_to_idx.get(signal_date)
        if idx is None:
            continue
        if idx + 1 >= n:
            continue
        next_date = all_dates[idx + 1]
        pending.setdefault(next_date, []).extend(sigs)

    return pending


def clamp_sell_price_for_limit_down(row: pd.Series, sell_price: float) -> float:
    prev_close = row.get("PREV_CLOSE", np.nan)
    limit_pct = row.get("limit_pct", 0.10)

    if pd.notna(prev_close) and prev_close > 0:
        limit_down = prev_close * (1 - limit_pct)
        sell_price = max(sell_price, limit_down)

    return sell_price


def clamp_buy_price_for_limit_up(row: pd.Series, buy_price: float) -> Optional[float]:
    prev_close = row.get("PREV_CLOSE", np.nan)
    limit_pct = row.get("limit_pct", 0.10)

    if pd.notna(prev_close) and prev_close > 0:
        limit_up = prev_close * (1 + limit_pct)
        if buy_price >= limit_up:
            return None

    return buy_price


def get_mark_to_market_price(df: pd.DataFrame, current_date: pd.Timestamp) -> float:
    if current_date in df.index:
        p = df.loc[current_date, "CLOSE"]
        if pd.notna(p) and p > 0:
            return float(p)

    hist = df.loc[:current_date, "CLOSE"]
    hist = hist[pd.notna(hist) & (hist > 0)]
    if len(hist) == 0:
        return 0.0
    return float(hist.iloc[-1])


# =========================================================
# 回测核心：固定持有天数 + 止损
# =========================================================
def run_backtest_fixed_hold(
    stock_data: Dict[str, pd.DataFrame],
    daily_signals: Dict[pd.Timestamp, List[Tuple[str, dict, float]]],
    hold_days: int,
    stop_loss_pct: float = 0.05,
) -> Optional[Dict]:
    all_dates, date_to_idx = build_trading_calendar(stock_data)
    pending_buy_signals = build_pending_buy_signals(daily_signals, all_dates, date_to_idx)

    cash = float(INITIAL_CAPITAL)
    positions: List[Dict] = []
    equity_curve: List[float] = []

    trade_count = 0
    win_count = 0
    holding_returns: List[float] = []
    holding_days_list: List[int] = []
    max_consecutive_losses = 0
    current_consecutive_losses = 0

    # key: execution_date, value: {stock_code: exit_reason}
    pending_exit_signals: Dict[pd.Timestamp, Dict[str, Dict]] = {}

    first_trade_date = None
    last_trade_date = None

    for current_date in all_dates:
        current_idx = date_to_idx[current_date]

        # -------------------------
        # 1) 先执行今天开盘的卖出单
        # -------------------------
        new_positions = []
        for pos in positions:
            stock = pos["stock"]
            df = stock_data[stock]

            exit_today = current_date in pending_exit_signals and stock in pending_exit_signals[current_date]
            if not exit_today:
                new_positions.append(pos)
                continue

            if current_date not in df.index:
                new_positions.append(pos)
                continue

            row = df.loc[current_date]
            exit_price = float(row["OPEN"])
            exit_price = clamp_sell_price_for_limit_down(row, exit_price)

            sell_value = pos["shares"] * exit_price * (1 - FEE_RATE - SLIPPAGE)
            cash += sell_value

            pnl = (exit_price - pos["entry_price"]) / pos["entry_price"]
            trade_count += 1
            holding_returns.append(pnl)
            holding_days_list.append(current_idx - pos["entry_idx"])

            if pnl > 0:
                win_count += 1
                current_consecutive_losses = 0
            else:
                current_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)

        positions = new_positions

        # -------------------------
        # 2) 再检查今天收盘后是否触发新的卖出信号
        #    固定持有策略：只包含止损 + 到期
        # -------------------------
        for pos in positions:
            stock = pos["stock"]
            df = stock_data[stock]

            if current_date not in df.index:
                continue

            row = df.loc[current_date]
            close = float(row["CLOSE"])

            if not np.isfinite(close) or close <= 0:
                continue

            entry_idx = pos["entry_idx"]
            if current_idx < entry_idx + 1:
                continue

            holding_days_now = current_idx - entry_idx
            exit_reason = None

            if close <= pos["stop_price"]:
                exit_reason = "止损"
            elif holding_days_now >= hold_days:
                exit_reason = f"持有{hold_days}天"

            if exit_reason is not None and not pos["exit_marked"]:
                if current_idx + 1 < len(all_dates):
                    next_date = all_dates[current_idx + 1]
                    pending_exit_signals.setdefault(next_date, {})[stock] = {
                        "exit_reason": exit_reason
                    }
                    pos["exit_marked"] = True

        # -------------------------
        # 3) 最后执行今天开盘的买入单
        # -------------------------
        if current_date in pending_buy_signals:
            candidates = pending_buy_signals[current_date]

            for stock, details, signal_low in candidates:
                df = stock_data.get(stock)
                if df is None or current_date not in df.index:
                    continue

                row = df.loc[current_date]
                open_price = float(row["OPEN"])
                open_price = clamp_buy_price_for_limit_up(row, open_price)
                if open_price is None or open_price <= 0:
                    continue

                available_slots = MAX_POSITIONS - len(positions)
                if available_slots <= 0:
                    continue

                entry_low = float(row["LOW"])
                stop_price = min(entry_low, float(signal_low)) * (1 - stop_loss_pct)

                allocation = cash / available_slots
                shares = int(allocation / open_price / 100) * 100
                if shares <= 0:
                    continue

                cost = shares * open_price * (1 + FEE_RATE + SLIPPAGE)
                if cost > cash:
                    continue

                cash -= cost
                positions.append({
                    "stock": stock,
                    "shares": shares,
                    "entry_price": open_price,
                    "entry_idx": current_idx,
                    "entry_date": current_date,
                    "stop_price": stop_price,
                    "exit_marked": False,
                })

                if first_trade_date is None:
                    first_trade_date = current_date

        # -------------------------
        # 4) 盘后估值
        # -------------------------
        total_value = cash
        for pos in positions:
            df = stock_data[pos["stock"]]
            mtm = get_mark_to_market_price(df, current_date)
            total_value += pos["shares"] * mtm

        equity_curve.append(total_value)
        last_trade_date = current_date

    if len(equity_curve) < 2:
        return None

    return summarize_result(
        equity_curve=equity_curve,
        trade_count=trade_count,
        win_count=win_count,
        holding_returns=holding_returns,
        holding_days_list=holding_days_list,
        max_consecutive_losses=max_consecutive_losses,
        first_trade_date=first_trade_date,
        last_trade_date=last_trade_date,
        extra={"hold_days": hold_days},
    )


# =========================================================
# 回测核心：止盈止损
# =========================================================
def run_backtest_take_profit(
    stock_data: Dict[str, pd.DataFrame],
    daily_signals: Dict[pd.Timestamp, List[Tuple[str, dict, float]]],
    take_profit_pct: float,
    stop_loss_pct: float = 0.05,
) -> Optional[Dict]:
    all_dates, date_to_idx = build_trading_calendar(stock_data)
    pending_buy_signals = build_pending_buy_signals(daily_signals, all_dates, date_to_idx)

    cash = float(INITIAL_CAPITAL)
    positions: List[Dict] = []
    equity_curve: List[float] = []

    trade_count = 0
    win_count = 0
    holding_returns: List[float] = []
    holding_days_list: List[int] = []
    max_consecutive_losses = 0
    current_consecutive_losses = 0

    pending_exit_signals: Dict[pd.Timestamp, Dict[str, Dict]] = {}

    first_trade_date = None
    last_trade_date = None

    for current_date in all_dates:
        current_idx = date_to_idx[current_date]

        # -------------------------
        # 1) 执行今天开盘卖出
        # -------------------------
        new_positions = []
        for pos in positions:
            stock = pos["stock"]
            df = stock_data[stock]

            exit_today = current_date in pending_exit_signals and stock in pending_exit_signals[current_date]
            if not exit_today:
                new_positions.append(pos)
                continue

            if current_date not in df.index:
                new_positions.append(pos)
                continue

            row = df.loc[current_date]
            exit_price = float(row["OPEN"])
            exit_price = clamp_sell_price_for_limit_down(row, exit_price)

            sell_value = pos["shares"] * exit_price * (1 - FEE_RATE - SLIPPAGE)
            cash += sell_value

            pnl = (exit_price - pos["entry_price"]) / pos["entry_price"]
            trade_count += 1
            holding_returns.append(pnl)
            holding_days_list.append(current_idx - pos["entry_idx"])

            if pnl > 0:
                win_count += 1
                current_consecutive_losses = 0
            else:
                current_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)

        positions = new_positions

        # -------------------------
        # 2) 收盘后触发新的止盈/止损
        # -------------------------
        for pos in positions:
            stock = pos["stock"]
            df = stock_data[stock]

            if current_date not in df.index:
                continue

            row = df.loc[current_date]
            close = float(row["CLOSE"])

            if not np.isfinite(close) or close <= 0:
                continue

            entry_idx = pos["entry_idx"]
            if current_idx < entry_idx + 1:
                continue

            stop_price = pos["stop_price"]
            target_price = pos["entry_price"] * (1 + take_profit_pct)

            exit_reason = None
            if close <= stop_price:
                exit_reason = "止损"
            elif close >= target_price:
                exit_reason = f"{take_profit_pct * 100:.0f}%止盈"

            if exit_reason is not None and not pos["exit_marked"]:
                if current_idx + 1 < len(all_dates):
                    next_date = all_dates[current_idx + 1]
                    pending_exit_signals.setdefault(next_date, {})[stock] = {
                        "exit_reason": exit_reason
                    }
                    pos["exit_marked"] = True

        # -------------------------
        # 3) 执行今天开盘买入
        # -------------------------
        if current_date in pending_buy_signals:
            candidates = pending_buy_signals[current_date]

            for stock, details, signal_low in candidates:
                df = stock_data.get(stock)
                if df is None or current_date not in df.index:
                    continue

                row = df.loc[current_date]
                open_price = float(row["OPEN"])
                open_price = clamp_buy_price_for_limit_up(row, open_price)
                if open_price is None or open_price <= 0:
                    continue

                available_slots = MAX_POSITIONS - len(positions)
                if available_slots <= 0:
                    continue

                entry_low = float(row["LOW"])
                stop_price = min(entry_low, float(signal_low)) * (1 - stop_loss_pct)

                allocation = cash / available_slots
                shares = int(allocation / open_price / 100) * 100
                if shares <= 0:
                    continue

                cost = shares * open_price * (1 + FEE_RATE + SLIPPAGE)
                if cost > cash:
                    continue

                cash -= cost
                positions.append({
                    "stock": stock,
                    "shares": shares,
                    "entry_price": open_price,
                    "entry_idx": current_idx,
                    "entry_date": current_date,
                    "stop_price": stop_price,
                    "exit_marked": False,
                })

                if first_trade_date is None:
                    first_trade_date = current_date

        # -------------------------
        # 4) 盘后估值
        # -------------------------
        total_value = cash
        for pos in positions:
            df = stock_data[pos["stock"]]
            mtm = get_mark_to_market_price(df, current_date)
            total_value += pos["shares"] * mtm

        equity_curve.append(total_value)
        last_trade_date = current_date

    if len(equity_curve) < 2:
        return None

    return summarize_result(
        equity_curve=equity_curve,
        trade_count=trade_count,
        win_count=win_count,
        holding_returns=holding_returns,
        holding_days_list=holding_days_list,
        max_consecutive_losses=max_consecutive_losses,
        first_trade_date=first_trade_date,
        last_trade_date=last_trade_date,
        extra={"take_profit_pct": take_profit_pct},
    )


# =========================================================
# 结果汇总
# =========================================================
def summarize_result(
    equity_curve: List[float],
    trade_count: int,
    win_count: int,
    holding_returns: List[float],
    holding_days_list: List[int],
    max_consecutive_losses: int,
    first_trade_date: Optional[pd.Timestamp],
    last_trade_date: Optional[pd.Timestamp],
    extra: Dict,
) -> Dict:
    final_capital = equity_curve[-1]
    final_multiple = final_capital / INITIAL_CAPITAL

    if first_trade_date is not None and last_trade_date is not None:
        years = max((last_trade_date - first_trade_date).days / 365.25, 0.01)
    else:
        years = max(len(equity_curve) / 252.0, 0.01)

    cagr = (final_multiple ** (1 / years) - 1) if years > 0 else 0.0

    equity_curve_arr = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity_curve_arr)
    drawdowns = (equity_curve_arr - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    avg_dd = float(np.mean(drawdowns))

    success_rate = win_count / trade_count if trade_count > 0 else 0.0
    avg_holding_return = float(np.mean(holding_returns) * 100.0) if holding_returns else 0.0
    max_holding_return = float(np.max(holding_returns) * 100.0) if holding_returns else 0.0
    avg_holding_days = float(np.mean(holding_days_list)) if holding_days_list else 0.0

    out = {
        "final_multiple": final_multiple,
        "CAGR": cagr * 100.0,
        "trade_count": trade_count,
        "success_rate": success_rate,
        "max_dd": max_dd * 100.0,
        "avg_dd": avg_dd * 100.0,
        "max_consecutive_losses": max_consecutive_losses,
        "avg_holding_return": avg_holding_return,
        "max_holding_return": max_holding_return,
        "avg_holding_days": avg_holding_days,
    }
    out.update(extra)
    return out


# =========================================================
# 打印结果
# =========================================================
def print_header():
    print("=" * 90)
    print("B3自定义K线组合回测（严格修正版）")
    print("=" * 90)
    print("""
================================================================================
                           B3自定义K线组合详细方案
================================================================================

【买入条件】(N日满足以下所有条件)

N日（B3）:
  1. N日是红砖(brick_red)
  2. N日涨幅 < 2%
  3. N日振幅 < 5%
  4. N日成交量 < N-1日成交量 * 0.9

N-1日（B2）:
  5. N-1日是红砖(brick_red)
  6. N-1日涨幅 > 4%
  7. N-1日上影线长度 < 实体长度 / 4
  8. N-1日 J < 80
  9. N-1日成交量 > N-2日成交量 * 2

N-2日:
  10. N-2日是绿砖(brick_green)
  11. N-2日是最近一段连续绿砖区间里 LOW 最低的位置

N-5日到N-3日:
  12. N-3日是绿砖
  13. N-4日是绿砖
  14. N-5日是绿砖
  15. brick[N-5] >= brick[N-4] >= brick[N-3] >= brick[N-2]

【信号发出】
  N日收盘后发出买入信号

【买入执行】
  N+1日开盘价买入

【止损价】
  min(买入日LOW, 信号日LOW) * (1 - stop_loss_pct)

【策略1：固定持有天数 + 止损】
  卖出条件:
    - 收盘价 <= 止损价：次日开盘卖出
    - 持有天数 >= hold_days：次日开盘卖出

【策略2：止盈止损】
  卖出条件:
    - 收盘价 <= 止损价：次日开盘卖出
    - 收盘价 >= 买入价 * (1 + 止盈比例)：次日开盘卖出
================================================================================
""")


def print_fixed_hold_results(results: List[Dict]):
    print("\n" + "=" * 80)
    print("策略1：固定持有天数 + 止损")
    print("=" * 80)
    print(
        f"\n{'持有天数':<10} {'最终倍数':<12} {'年化收益率':<12} {'交易次数':<10} "
        f"{'成功率':<10} {'最大回撤':<12} {'平均回撤':<12} {'最大连亏':<10} "
        f"{'平均持有收益':<14} {'最大持有收益':<14} {'平均持有天数':<14}"
    )
    print("-" * 140)

    for r in results:
        print(
            f"{r['hold_days']:<10} "
            f"{r['final_multiple']:<12.4f} "
            f"{r['CAGR']:<12.2f}% "
            f"{r['trade_count']:<10} "
            f"{r['success_rate']:<10.2%} "
            f"{r['max_dd']:<12.2f}% "
            f"{r['avg_dd']:<12.2f}% "
            f"{r['max_consecutive_losses']:<10} "
            f"{r['avg_holding_return']:<14.2f}% "
            f"{r['max_holding_return']:<14.2f}% "
            f"{r['avg_holding_days']:<14.1f}"
        )


def print_take_profit_results(results: List[Dict]):
    print("\n" + "=" * 80)
    print("策略2：止盈止损")
    print("=" * 80)
    print(
        f"\n{'止盈比例':<10} {'最终倍数':<12} {'年化收益率':<12} {'交易次数':<10} "
        f"{'成功率':<10} {'最大回撤':<12} {'平均回撤':<12} {'最大连亏':<10} "
        f"{'平均持有收益':<14} {'最大持有收益':<14} {'平均持有天数':<14}"
    )
    print("-" * 140)

    for r in results:
        print(
            f"{r['take_profit_pct'] * 100:<10.0f}% "
            f"{r['final_multiple']:<12.4f} "
            f"{r['CAGR']:<12.2f}% "
            f"{r['trade_count']:<10} "
            f"{r['success_rate']:<10.2%} "
            f"{r['max_dd']:<12.2f}% "
            f"{r['avg_dd']:<12.2f}% "
            f"{r['max_consecutive_losses']:<10} "
            f"{r['avg_holding_return']:<14.2f}% "
            f"{r['max_holding_return']:<14.2f}% "
            f"{r['avg_holding_days']:<14.1f}"
        )


# =========================================================
# 主函数
# =========================================================
def main():
    print_header()

    from datetime import datetime
    today_str = datetime.today().strftime("%Y%m%d")
    fallback_date_str, DATA_DIR = find_latest_available_data_dir(BASE_DATA_DIR, today_str)
    
    if DATA_DIR is None:
        print("未找到任何数据目录")
        return

    print(f"使用数据目录: {DATA_DIR}")
    print(f"数据日期: {fallback_date_str}")

    stock_data, daily_signals = load_all_stock_data(DATA_DIR)
    if not stock_data:
        print("没有成功加载任何股票数据。")
        return

    fixed_hold_results = []
    for hold_days in HOLD_DAYS_LIST:
        r = run_backtest_fixed_hold(
            stock_data=stock_data,
            daily_signals=daily_signals,
            hold_days=hold_days,
            stop_loss_pct=STOP_LOSS_PCT,
        )
        if r is not None:
            fixed_hold_results.append(r)

    take_profit_results = []
    for tp in TAKE_PROFIT_LIST:
        r = run_backtest_take_profit(
            stock_data=stock_data,
            daily_signals=daily_signals,
            take_profit_pct=tp,
            stop_loss_pct=STOP_LOSS_PCT,
        )
        if r is not None:
            take_profit_results.append(r)

    print_fixed_hold_results(fixed_hold_results)
    print_take_profit_results(take_profit_results)

    print("\n" + "=" * 80)
    print("对比汇总")
    print("=" * 80)

    if fixed_hold_results:
        best_hold = max(fixed_hold_results, key=lambda x: x["CAGR"])
        print(
            f"\n最佳固定持有: {best_hold['hold_days']}天, "
            f"年化{best_hold['CAGR']:.2f}%, 倍数{best_hold['final_multiple']:.4f}x"
        )

    if take_profit_results:
        best_tp = max(take_profit_results, key=lambda x: x["CAGR"])
        print(
            f"最佳止盈: {best_tp['take_profit_pct'] * 100:.0f}%, "
            f"年化{best_tp['CAGR']:.2f}%, 倍数{best_tp['final_multiple']:.4f}x"
        )


if __name__ == "__main__":
    main()