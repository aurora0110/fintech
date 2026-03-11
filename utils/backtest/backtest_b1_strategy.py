# -*- coding: utf-8 -*-
import os
import math
import itertools
import numpy as np
import pandas as pd


# =========================================================
# 配置
# =========================================================
DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
OUTPUT_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/b1_full_risk_experiment_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL_CAPITAL = 1_000_000
FEE_RATE = 0.0003
SLIPPAGE = 0.001

LOAD_PROGRESS_STEP = 500
SIGNAL_PROGRESS_STEP = 500
BACKTEST_PROGRESS_STEP = 500


# =========================================================
# 基础工具
# =========================================================
def get_stock_loc(df: pd.DataFrame, date) -> int:
    if date not in df.index:
        return -1
    loc = df.index.get_loc(date)
    if isinstance(loc, slice):
        return int(loc.start)
    if isinstance(loc, np.ndarray):
        return int(loc[0]) if len(loc) > 0 else -1
    return int(loc)


def safe_pct_change(s: pd.Series) -> pd.Series:
    return s.pct_change().replace([np.inf, -np.inf], np.nan)


def calc_profit_factor(ret_series: pd.Series) -> float:
    pos = ret_series[ret_series > 0].sum()
    neg = ret_series[ret_series < 0].sum()
    if abs(neg) < 1e-12:
        return np.nan
    return pos / abs(neg)


def calc_sharpe(daily_ret: np.ndarray) -> float:
    if len(daily_ret) <= 1:
        return np.nan
    std = np.std(daily_ret, ddof=1)
    if std <= 1e-12:
        return np.nan
    return np.mean(daily_ret) / std * np.sqrt(252)


def calc_max_drawdown(equity_arr: np.ndarray) -> float:
    if len(equity_arr) == 0:
        return np.nan
    running_max = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - running_max) / running_max
    return float(dd.min())


def calc_cagr(final_multiple: float, n_days: int) -> float:
    years = max(n_days / 252.0, 1e-9)
    if final_multiple <= 0:
        return np.nan
    return final_multiple ** (1 / years) - 1


# =========================================================
# 数据清洗与指标
# =========================================================
def check_data_anomaly(df: pd.DataFrame) -> set:
    anomaly_dates = set()
    if len(df) < 2:
        return anomaly_dates

    for i in range(len(df)):
        row = df.iloc[i]
        open_p = row["OPEN"]
        high = row["HIGH"]
        low = row["LOW"]
        close = row["CLOSE"]
        volume = row["VOLUME"]

        if pd.isna(open_p) or pd.isna(high) or pd.isna(low) or pd.isna(close):
            anomaly_dates.add(df.index[i])
            continue

        if min(open_p, high, low, close) <= 0:
            anomaly_dates.add(df.index[i])
            continue

        if high < max(open_p, close) or low > min(open_p, close):
            anomaly_dates.add(df.index[i])
            continue

        if pd.isna(volume) or volume <= 0:
            anomaly_dates.add(df.index[i])
            continue

        if i > 0:
            prev_close = df.iloc[i - 1]["CLOSE"]
            if pd.notna(prev_close) and prev_close > 0:
                change_pct = (close - prev_close) / prev_close * 100
                if change_pct > 25 or change_pct < -25:
                    anomaly_dates.add(df.index[i])

    return anomaly_dates


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["CLOSE"]
    high = df["HIGH"]
    low = df["LOW"]
    volume = df["VOLUME"]

    # 均线
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA30"] = close.rolling(30).mean()
    df["MA60"] = close.rolling(60).mean()

    # KDJ
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-9) * 100
    df["K"] = rsv.ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    df["J_Q05_30"] = df["J"].rolling(30).quantile(0.05)

    # 趋势线 / 多空线
    df["trend_line"] = close.ewm(span=10, adjust=False).mean()
    df["trend_line"] = df["trend_line"].ewm(span=10, adjust=False).mean()
    df["MA14"] = close.rolling(14).mean()
    df["MA28"] = close.rolling(28).mean()
    df["MA57"] = close.rolling(57).mean()
    df["MA114"] = close.rolling(114).mean()
    df["bull_bear_line"] = (df["MA14"] + df["MA28"] + df["MA57"] + df["MA114"]) / 4.0

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # 主策略条件
    df["trend_ok"] = df["trend_line"] > df["bull_bear_line"]
    df["bullish_filter"] = (
        (df["CLOSE"] > df["MA30"]) &
        (df["MA10"] > df["MA10"].shift(5)) &
        (df["MA30"] > df["MA30"].shift(5))
    )

    # 市场状态代理原子字段
    df["ret1"] = safe_pct_change(close)
    df["vol_ma20"] = volume.rolling(20).mean()
    df["vol_expand"] = volume > df["vol_ma20"]

    return df


def load_all_data(data_dir: str):
    stock_data = {}
    all_dates = []

    print("加载数据...")
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    total_files = len(files)
    loaded_count = 0

    for i, file in enumerate(files, 1):
        stock_code = file.replace(".txt", "")

        if stock_code.startswith("BJ"):
            if i % LOAD_PROGRESS_STEP == 0 or i == total_files:
                print(f"加载进度: {i}/{total_files}")
            continue

        code_part = stock_code.split("#")[1] if "#" in stock_code else stock_code
        if code_part.startswith("8") and len(code_part) >= 3:
            if code_part[1] == "3" or code_part.startswith("83") or code_part.startswith("87"):
                if i % LOAD_PROGRESS_STEP == 0 or i == total_files:
                    print(f"加载进度: {i}/{total_files}")
                continue

        path = os.path.join(data_dir, file)

        try:
            df = pd.read_csv(path, sep="\t", encoding="utf-8")
            if df.shape[1] < 7:
                if i % LOAD_PROGRESS_STEP == 0 or i == total_files:
                    print(f"加载进度: {i}/{total_files}")
                continue

            df = df.iloc[:, :7].copy()
            df.columns = ["日期", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMOUNT"]
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
            df = df.dropna(subset=["日期"])
            df = df.set_index("日期").sort_index()
            df = df[~df.index.duplicated(keep="last")]

            if len(df) < 150:
                if i % LOAD_PROGRESS_STEP == 0 or i == total_files:
                    print(f"加载进度: {i}/{total_files}")
                continue

            anomaly_dates = check_data_anomaly(df)
            if len(anomaly_dates) > len(df) * 0.1:
                if i % LOAD_PROGRESS_STEP == 0 or i == total_files:
                    print(f"加载进度: {i}/{total_files}")
                continue

            df = calculate_indicators(df)
            stock_data[stock_code] = df
            loaded_count += 1
            all_dates.extend(df.index.tolist())

        except Exception as e:
            print(f"加载失败 {file}: {e}")

        if i % LOAD_PROGRESS_STEP == 0 or i == total_files:
            print(f"加载进度: {i}/{total_files}")

    all_dates = sorted(set(all_dates))
    print(f"成功加载 {loaded_count} 只股票")
    print(f"总交易日: {len(all_dates)}")
    return stock_data, all_dates


# =========================================================
# 市场状态开关（无行业风格数据版）
# =========================================================
def build_market_regime(stock_data: dict, all_dates: list) -> pd.DataFrame:
    rows = []
    items = list(stock_data.items())
    total = len(items)

    # 聚合横截面数据
    date_bucket = {}
    for i, (code, df) in enumerate(items, 1):
        tmp = df[["ret1", "vol_expand", "CLOSE", "MA20"]].copy()
        tmp["close_above_ma20"] = tmp["CLOSE"] > tmp["MA20"]
        for dt, row in tmp.iterrows():
            if dt not in date_bucket:
                date_bucket[dt] = {"rets": [], "vol_expand": [], "close_above_ma20": []}
            if pd.notna(row["ret1"]):
                date_bucket[dt]["rets"].append(float(row["ret1"]))
            if pd.notna(row["vol_expand"]):
                date_bucket[dt]["vol_expand"].append(bool(row["vol_expand"]))
            if pd.notna(row["close_above_ma20"]):
                date_bucket[dt]["close_above_ma20"].append(bool(row["close_above_ma20"]))

        if i % SIGNAL_PROGRESS_STEP == 0 or i == total:
            print(f"市场状态构建进度: {i}/{total}")

    for dt in all_dates:
        bucket = date_bucket.get(dt, None)
        if bucket is None or len(bucket["rets"]) == 0:
            rows.append({
                "date": dt,
                "up_ratio": np.nan,
                "strong_ratio": np.nan,
                "vol_expand_ratio": np.nan,
                "above_ma20_ratio": np.nan,
            })
            continue

        rets = np.array(bucket["rets"], dtype=float)
        vol_expand = np.array(bucket["vol_expand"], dtype=bool)
        above_ma20 = np.array(bucket["close_above_ma20"], dtype=bool)

        rows.append({
            "date": dt,
            "up_ratio": np.mean(rets > 0),
            "strong_ratio": np.mean(rets > 0.03),
            "vol_expand_ratio": np.mean(vol_expand),
            "above_ma20_ratio": np.mean(above_ma20),
        })

    regime_df = pd.DataFrame(rows).set_index("date").sort_index()

    # 开关条件可实验时调用
    return regime_df


def regime_pass(regime_row: pd.Series, regime_mode: str) -> bool:
    if regime_mode == "none":
        return True
    if regime_row is None or regime_row.isna().all():
        return False

    if regime_mode == "basic":
        return (
            regime_row["up_ratio"] >= 0.45 and
            regime_row["above_ma20_ratio"] >= 0.45
        )
    if regime_mode == "strong":
        return (
            regime_row["up_ratio"] >= 0.50 and
            regime_row["strong_ratio"] >= 0.08 and
            regime_row["above_ma20_ratio"] >= 0.50
        )
    return True


# =========================================================
# 主策略信号
# =========================================================
def build_main_strategy_signals_for_stock(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    cond = (
        x["J"].notna() &
        x["J_Q05_30"].notna() &
        (x["J"] <= x["J_Q05_30"]) &
        x["trend_ok"] &
        x["bullish_filter"]
    )
    out = x.loc[cond, ["OPEN", "HIGH", "LOW", "CLOSE", "J", "J_Q05_30", "ATR14"]].copy()
    out["score"] = (-out["J"]).fillna(0.0)  # J越低分越高
    return out


def build_daily_signals(stock_data: dict):
    daily_scores = {}
    items = list(stock_data.items())
    total = len(items)

    print("开始构建主策略信号...")
    for i, (stock_code, df) in enumerate(items, 1):
        sig_df = build_main_strategy_signals_for_stock(df)
        if not sig_df.empty:
            for dt, row in sig_df.iterrows():
                daily_scores.setdefault(dt, []).append(
                    {
                        "stock": stock_code,
                        "score": float(row["score"]),
                        "J": float(row["J"]) if pd.notna(row["J"]) else np.nan,
                        "J_Q05_30": float(row["J_Q05_30"]) if pd.notna(row["J_Q05_30"]) else np.nan,
                        "ATR14": float(row["ATR14"]) if pd.notna(row["ATR14"]) else np.nan,
                    }
                )

        if i % SIGNAL_PROGRESS_STEP == 0 or i == total:
            print(f"信号构建进度: {i}/{total}")

    print(f"主策略有信号的天数: {len(daily_scores)}")
    return daily_scores


def generate_pending_buy_signals(daily_scores: dict, all_dates: list):
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    pending_buy = {}
    for signal_date, items in daily_scores.items():
        i = date_to_idx.get(signal_date, None)
        if i is None or i + 1 >= len(all_dates):
            continue
        next_date = all_dates[i + 1]
        pending_buy.setdefault(next_date, []).extend(items)
    return pending_buy


def get_mark_price(df: pd.DataFrame, current_date):
    idx = get_stock_loc(df, current_date)
    if idx >= 0:
        px = df.iloc[idx]["CLOSE"]
        if pd.notna(px) and px > 0:
            return float(px)
    hist = df.loc[:current_date, "CLOSE"]
    hist = hist[pd.notna(hist) & (hist > 0)]
    return float(hist.iloc[-1]) if len(hist) > 0 else np.nan


# =========================================================
# 止损/止盈
# =========================================================
def get_stop_price(stop_mode: str, entry_low: float, signal_low: float, entry_price: float, atr14: float):
    if stop_mode == "entry_low_093":
        return entry_low * 0.93
    if stop_mode == "entry_low_095":
        return entry_low * 0.95
    if stop_mode == "signal_or_entry_tighter_095":
        vals = [v for v in [entry_low, signal_low] if pd.notna(v) and v > 0]
        return max(vals) * 0.95 if vals else np.nan
    if stop_mode == "atr_1.0":
        return entry_price - atr14 * 1.0 if pd.notna(atr14) and atr14 > 0 else np.nan
    if stop_mode == "atr_1.5":
        return entry_price - atr14 * 1.5 if pd.notna(atr14) and atr14 > 0 else np.nan
    return np.nan


def should_schedule_take_profit(pos: dict, high_p: float, close_p: float):
    tp_mode = pos["take_profit_mode"]
    if tp_mode == "none":
        return False, None

    entry_price = pos["entry_price"]

    if tp_mode == "fixed_2":
        target = entry_price * 1.02
        return pd.notna(high_p) and high_p >= target, "止盈2%"
    if tp_mode == "fixed_3":
        target = entry_price * 1.03
        return pd.notna(high_p) and high_p >= target, "止盈3%"
    if tp_mode == "fixed_4":
        target = entry_price * 1.04
        return pd.notna(high_p) and high_p >= target, "止盈4%"
    if tp_mode == "fixed_5":
        target = entry_price * 1.05
        return pd.notna(high_p) and high_p >= target, "止盈5%"
    if tp_mode == "fixed_6":
        target = entry_price * 1.06
        return pd.notna(high_p) and high_p >= target, "止盈6%"

    # 分批：这里只做简化版，达到3%后卖一半，达到5%后卖剩余
    if tp_mode == "ladder_3_5":
        if (not pos.get("tp1_done", False)) and pd.notna(high_p) and high_p >= entry_price * 1.03:
            return True, "分批止盈1"
        if pos.get("tp1_done", False) and pd.notna(high_p) and high_p >= entry_price * 1.05:
            return True, "分批止盈2"
        return False, None

    # 达标后跟踪止盈：先达到3%，后续跌破前一日低点退出
    if tp_mode == "trail_after_3":
        if not pos.get("trail_armed", False):
            if pd.notna(high_p) and high_p >= entry_price * 1.03:
                pos["trail_armed"] = True
        return False, None

    return False, None


# =========================================================
# 停机机制
# =========================================================
def should_pause_by_trade_window(trade_pnls: list, pause_rule: str):
    if pause_rule == "loss_streak_3_pause_5":
        return False  # 由主流程连亏逻辑处理
    if pause_rule == "loss_streak_3_pause_7":
        return False
    if pause_rule == "loss_streak_4_pause_5":
        return False
    if pause_rule == "avg10_lt0_pause5":
        if len(trade_pnls) >= 10 and np.mean(trade_pnls[-10:]) < 0:
            return True
    if pause_rule == "winrate10_lt40_pause5":
        if len(trade_pnls) >= 10:
            last10 = trade_pnls[-10:]
            wr = np.mean(np.array(last10) > 0)
            if wr < 0.40:
                return True
    return False


def get_loss_streak_params(pause_rule: str):
    if pause_rule == "loss_streak_3_pause_5":
        return 3, 5
    if pause_rule == "loss_streak_3_pause_7":
        return 3, 7
    if pause_rule == "loss_streak_4_pause_5":
        return 4, 5
    return None, None


# =========================================================
# 回测
# =========================================================
def run_backtest(
    stock_data: dict,
    all_dates: list,
    pending_buy_signals: dict,
    regime_df: pd.DataFrame,
    params: dict,
    exp_name: str,
):
    max_positions = params["max_positions"]
    max_new_buys_per_day = params["max_new_buys_per_day"]
    max_hold_days = params["max_hold_days"]
    day_cash_cap = params["day_cash_cap"]
    single_pos_cap = params["single_pos_cap"]
    take_profit_mode = params["take_profit_mode"]
    stop_mode = params["stop_mode"]
    pause_rule = params["pause_rule"]
    regime_mode = params["regime_mode"]
    score_bucket = params["score_bucket"]

    cash = float(INITIAL_CAPITAL)
    positions = []
    equity_curve = []
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    trade_count = 0
    win_count = 0
    loss_count = 0
    current_consecutive_losses = 0
    max_consecutive_losses = 0
    pause_trading_days = 0
    trade_pnls = []
    holding_returns = []
    exit_records = []

    total_days = len(all_dates)

    print(f"\n开始回测：{exp_name}")
    print(f"参数：{params}")

    for day_i, current_date in enumerate(all_dates, 1):
        if pause_trading_days > 0:
            pause_trading_days -= 1

        current_global_idx = date_to_idx[current_date]

        # =========================
        # 1) 卖出
        # =========================
        new_positions = []
        for pos in positions:
            stock = pos["stock"]
            df = stock_data[stock]
            stock_idx = get_stock_loc(df, current_date)

            if stock_idx < 0:
                new_positions.append(pos)
                continue

            row = df.iloc[stock_idx]
            open_p = row["OPEN"]
            high_p = row["HIGH"]
            low_p = row["LOW"]
            close_p = row["CLOSE"]

            exit_flag = False
            exit_reason = ""
            exit_price = np.nan

            if pos.get("scheduled_exit_date") == current_date:
                exit_flag = True
                exit_reason = pos.get("scheduled_exit_reason", "计划卖出")
                exit_price = open_p

            if not exit_flag and not pos.get("exit_marked", False):
                hold_days = current_global_idx - pos["entry_global_idx"]

                # 分批止盈逻辑
                if take_profit_mode == "ladder_3_5":
                    if (not pos.get("tp1_done", False)) and pd.notna(high_p) and high_p >= pos["entry_price"] * 1.03:
                        # 次日开盘卖一半
                        if current_global_idx + 1 < len(all_dates):
                            pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                            pos["scheduled_exit_reason"] = "分批止盈1"
                            pos["partial_exit_ratio"] = 0.5
                            pos["exit_marked"] = True
                    elif pos.get("tp1_done", False) and pd.notna(high_p) and high_p >= pos["entry_price"] * 1.05:
                        if current_global_idx + 1 < len(all_dates):
                            pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                            pos["scheduled_exit_reason"] = "分批止盈2"
                            pos["partial_exit_ratio"] = 1.0
                            pos["exit_marked"] = True

                # 达标后跟踪止盈
                if take_profit_mode == "trail_after_3":
                    if not pos.get("trail_armed", False):
                        if pd.notna(high_p) and high_p >= pos["entry_price"] * 1.03:
                            pos["trail_armed"] = True
                    else:
                        prev_idx = stock_idx - 1
                        if prev_idx >= 0:
                            prev_low = df.iloc[prev_idx]["LOW"]
                            if pd.notna(close_p) and pd.notna(prev_low) and close_p < prev_low:
                                if current_global_idx + 1 < len(all_dates):
                                    pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                                    pos["scheduled_exit_reason"] = "跟踪止盈退出"
                                    pos["partial_exit_ratio"] = 1.0
                                    pos["exit_marked"] = True

                # 固定止盈
                hit_tp, tp_reason = should_schedule_take_profit(pos, high_p, close_p)
                if hit_tp and (not pos.get("exit_marked", False)):
                    if current_global_idx + 1 < len(all_dates):
                        pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                        pos["scheduled_exit_reason"] = tp_reason
                        pos["partial_exit_ratio"] = 1.0
                        pos["exit_marked"] = True

                # 止损
                if (not pos.get("exit_marked", False)) and pd.notna(close_p) and pd.notna(pos["stop_price"]) and close_p < pos["stop_price"]:
                    if current_global_idx + 1 < len(all_dates):
                        pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                        pos["scheduled_exit_reason"] = "止损"
                        pos["partial_exit_ratio"] = 1.0
                        pos["exit_marked"] = True

                # 状态止损：跌破趋势线
                if (not pos.get("exit_marked", False)) and pos.get("use_trend_stop", False):
                    trend_line = row["trend_line"]
                    if pd.notna(close_p) and pd.notna(trend_line) and close_p < trend_line:
                        if current_global_idx + 1 < len(all_dates):
                            pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                            pos["scheduled_exit_reason"] = "跌破趋势线"
                            pos["partial_exit_ratio"] = 1.0
                            pos["exit_marked"] = True

                # 最长持有
                if (not pos.get("exit_marked", False)) and hold_days >= max_hold_days:
                    if current_global_idx + 1 < len(all_dates):
                        pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                        pos["scheduled_exit_reason"] = f"到期{max_hold_days}日卖出"
                        pos["partial_exit_ratio"] = 1.0
                        pos["exit_marked"] = True

            if exit_flag:
                if pd.isna(exit_price) or exit_price <= 0:
                    new_positions.append(pos)
                    continue

                exit_ratio = pos.get("partial_exit_ratio", 1.0)
                exit_shares = int(pos["shares"] * exit_ratio / 100) * 100 if exit_ratio < 1 else pos["shares"]
                if exit_shares <= 0:
                    exit_shares = pos["shares"]

                sell_value = exit_shares * exit_price * (1 - FEE_RATE - SLIPPAGE)
                cash += sell_value

                pnl = (exit_price - pos["entry_price"]) / pos["entry_price"]
                trade_count += 1
                holding_returns.append(pnl)
                trade_pnls.append(pnl)

                if pnl > 0:
                    win_count += 1
                    current_consecutive_losses = 0
                else:
                    loss_count += 1
                    current_consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)

                loss_streak_trigger, loss_pause_days = get_loss_streak_params(pause_rule)
                if loss_streak_trigger is not None and current_consecutive_losses >= loss_streak_trigger:
                    pause_trading_days = loss_pause_days

                if should_pause_by_trade_window(trade_pnls, pause_rule):
                    pause_trading_days = 5

                exit_records.append({
                    "stock": stock,
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "ret": pnl,
                    "hold_days": current_global_idx - pos["entry_global_idx"],
                    "exit_reason": exit_reason,
                })

                remain_shares = pos["shares"] - exit_shares
                if remain_shares > 0 and exit_ratio < 1.0:
                    pos["shares"] = remain_shares
                    pos["tp1_done"] = True
                    pos["exit_marked"] = False
                    pos["scheduled_exit_date"] = None
                    pos["scheduled_exit_reason"] = None
                    pos["partial_exit_ratio"] = None
                    new_positions.append(pos)
            else:
                new_positions.append(pos)

        positions = new_positions

        # =========================
        # 2) 买入
        # =========================
        if pause_trading_days == 0 and current_date in pending_buy_signals:
            regime_row = regime_df.loc[current_date] if current_date in regime_df.index else None
            if regime_pass(regime_row, regime_mode):
                available_slots = max_positions - len(positions)
                if available_slots > 0 and cash > 0:
                    candidates = pending_buy_signals[current_date]
                    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

                    # 信号分层过滤
                    if len(candidates) > 0 and score_bucket != "all":
                        scores = np.array([x["score"] for x in candidates], dtype=float)
                        if score_bucket == "top50":
                            thresh = np.quantile(scores, 0.50)
                            candidates = [x for x in candidates if x["score"] >= thresh]
                        elif score_bucket == "top30":
                            thresh = np.quantile(scores, 0.70)
                            candidates = [x for x in candidates if x["score"] >= thresh]
                        elif score_bucket == "top10":
                            thresh = np.quantile(scores, 0.90)
                            candidates = [x for x in candidates if x["score"] >= thresh]

                    already_holding = {p["stock"] for p in positions}
                    bought_today = 0
                    day_used_cash = 0.0
                    day_cash_limit = (cash + sum(
                        p["shares"] * get_mark_price(stock_data[p["stock"]], current_date)
                        for p in positions if pd.notna(get_mark_price(stock_data[p["stock"]], current_date))
                    )) * day_cash_cap

                    for item in candidates:
                        if available_slots <= 0 or cash <= 0:
                            break
                        if bought_today >= max_new_buys_per_day:
                            break

                        stock = item["stock"]
                        if stock in already_holding:
                            continue

                        df = stock_data[stock]
                        stock_idx = get_stock_loc(df, current_date)
                        if stock_idx < 0:
                            continue

                        row = df.iloc[stock_idx]
                        open_price = row["OPEN"]
                        low_price = row["LOW"]
                        atr14 = row["ATR14"]

                        if pd.isna(open_price) or open_price <= 0 or pd.isna(low_price) or low_price <= 0:
                            continue

                        # 单票上限 + 当日资金预算 + 剩余仓位均分 三者取最小
                        total_equity_now = cash + sum(
                            p["shares"] * get_mark_price(stock_data[p["stock"]], current_date)
                            for p in positions if pd.notna(get_mark_price(stock_data[p["stock"]], current_date))
                        )
                        alloc_by_slot = cash / available_slots
                        alloc_by_single_cap = total_equity_now * single_pos_cap
                        alloc_by_day_cap = max(day_cash_limit - day_used_cash, 0.0)
                        allocation = min(alloc_by_slot, alloc_by_single_cap, alloc_by_day_cap)

                        if allocation <= 0:
                            continue

                        shares = int(allocation / (open_price * (1 + FEE_RATE + SLIPPAGE)) / 100) * 100
                        if shares <= 0:
                            continue

                        cost = shares * open_price * (1 + FEE_RATE + SLIPPAGE)
                        if cost > cash:
                            continue

                        signal_idx = stock_idx - 1
                        signal_low = df.iloc[signal_idx]["LOW"] if signal_idx >= 0 else np.nan
                        stop_price = get_stop_price(stop_mode, low_price, signal_low, open_price, atr14)

                        cash -= cost
                        day_used_cash += cost

                        positions.append({
                            "stock": stock,
                            "shares": shares,
                            "entry_price": open_price,
                            "entry_low": low_price,
                            "signal_low": signal_low,
                            "entry_date": current_date,
                            "entry_global_idx": current_global_idx,
                            "stop_price": stop_price,
                            "score": item["score"],
                            "exit_marked": False,
                            "scheduled_exit_date": None,
                            "scheduled_exit_reason": None,
                            "partial_exit_ratio": None,
                            "tp1_done": False,
                            "trail_armed": False,
                            "take_profit_mode": take_profit_mode,
                            "use_trend_stop": True,
                        })
                        already_holding.add(stock)
                        available_slots -= 1
                        bought_today += 1

        # =========================
        # 3) 估值
        # =========================
        position_value = 0.0
        for pos in positions:
            stock = pos["stock"]
            mark_price = get_mark_price(stock_data[stock], current_date)
            if pd.notna(mark_price) and mark_price > 0:
                position_value += pos["shares"] * mark_price

        total_value = cash + position_value

        # 防止估值异常为0
        if total_value <= 0 or not np.isfinite(total_value):
            total_value = equity_curve[-1] if len(equity_curve) > 0 else INITIAL_CAPITAL

        equity_curve.append(total_value)

        if day_i % BACKTEST_PROGRESS_STEP == 0 or day_i == total_days:
            print(f"回测进度: {day_i}/{total_days}")

    # 统计
    equity_arr = np.array(equity_curve, dtype=float)
    daily_ret = np.array([
        (equity_arr[i + 1] - equity_arr[i]) / equity_arr[i]
        for i in range(len(equity_arr) - 1)
        if equity_arr[i] > 0 and np.isfinite(equity_arr[i + 1])
    ])

    final_capital = float(equity_arr[-1]) if len(equity_arr) > 0 else INITIAL_CAPITAL
    final_multiple = final_capital / INITIAL_CAPITAL
    max_dd = calc_max_drawdown(equity_arr)
    sharpe = calc_sharpe(daily_ret)
    cagr = calc_cagr(final_multiple, len(all_dates))

    success_rate = win_count / trade_count * 100 if trade_count > 0 else np.nan
    avg_holding_return = np.mean(holding_returns) * 100 if holding_returns else np.nan
    max_holding_return = np.max(holding_returns) * 100 if holding_returns else np.nan
    profit_factor = calc_profit_factor(pd.Series(holding_returns)) if holding_returns else np.nan
    calmar = (cagr * 100) / abs(max_dd * 100) if pd.notna(cagr) and pd.notna(max_dd) and abs(max_dd) > 1e-12 else np.nan

    print(f"\n{'=' * 100}")
    print(exp_name)
    print(f"{'=' * 100}")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}")
    print(f"最终资金: {final_capital:,.2f}")
    print(f"最终倍数: {final_multiple:.2f}")
    print(f"年化收益率(CAGR): {cagr * 100:.2f}%")
    print(f"最大回撤: {max_dd * 100:.2f}%")
    print(f"Calmar: {calmar:.2f}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"总交易次数: {trade_count}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"盈亏比: {profit_factor:.2f}")
    print(f"平均持有期间收益率: {avg_holding_return:.2f}%")
    print(f"最大持有期间收益率: {max_holding_return:.2f}%")
    print(f"最大连续失败次数: {max_consecutive_losses}")
    print(f"当前未平仓数: {len(positions)}")

    return {
        "实验名称": exp_name,
        "最终资金": final_capital,
        "最终倍数": final_multiple,
        "年化收益率": cagr * 100,
        "最大回撤": max_dd * 100,
        "Calmar": calmar,
        "夏普比率": sharpe,
        "总交易次数": trade_count,
        "成功率": success_rate,
        "盈亏比": profit_factor,
        "平均持有期间收益率": avg_holding_return,
        "最大持有期间收益率": max_holding_return,
        "最大连续失败次数": max_consecutive_losses,
        "当前未平仓数": len(positions),
        **params,
    }


# =========================================================
# 主程序
# =========================================================
def main():
    stock_data, all_dates = load_all_data(DATA_DIR)
    daily_scores = build_daily_signals(stock_data)
    pending_buy_signals = generate_pending_buy_signals(daily_scores, all_dates)
    regime_df = build_market_regime(stock_data, all_dates)

    # 这里只给一版“专业但不爆炸”的实验矩阵
    param_grid = {
        "max_positions": [10, 5],
        "max_new_buys_per_day": [1, 2, 3],
        "max_hold_days": [2, 3],
        "day_cash_cap": [0.30, 0.50],          # 单日最多动用30%/50%资金
        "single_pos_cap": [0.10, 0.15],        # 单票仓位上限10%/15%
        "take_profit_mode": [
            "none", "fixed_2", "fixed_3", "fixed_4", "fixed_5", "ladder_3_5", "trail_after_3"
        ],
        "stop_mode": [
            "entry_low_093", "entry_low_095", "signal_or_entry_tighter_095", "atr_1.0"
        ],
        "pause_rule": [
            "loss_streak_3_pause_5",
            "loss_streak_3_pause_7",
            "avg10_lt0_pause5",
            "winrate10_lt40_pause5"
        ],
        "regime_mode": [
            "none", "basic"
        ],
        "score_bucket": [
            "all", "top50", "top30"
        ],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    print("\n" + "=" * 140)
    print("开始 B1 主策略专业风控/止盈/状态开关实验")
    print("=" * 140)

    total_exps = len(all_param_combos)
    for i, params in enumerate(all_param_combos, 1):
        exp_name = (
            f"实验{i}_{total_exps}"
            f"_持仓{params['max_positions']}"
            f"_单日新开{params['max_new_buys_per_day']}"
            f"_持有{params['max_hold_days']}"
            f"_日资金{int(params['day_cash_cap']*100)}%"
            f"_单票{int(params['single_pos_cap']*100)}%"
            f"_止盈{params['take_profit_mode']}"
            f"_止损{params['stop_mode']}"
            f"_停机{params['pause_rule']}"
            f"_市场{params['regime_mode']}"
            f"_分层{params['score_bucket']}"
        )

        res = run_backtest(
            stock_data=stock_data,
            all_dates=all_dates,
            pending_buy_signals=pending_buy_signals,
            regime_df=regime_df,
            params=params,
            exp_name=exp_name,
        )
        results.append(res)

    result_df = pd.DataFrame(results)
    result_df["回撤绝对值"] = result_df["最大回撤"].abs()

    out_csv = os.path.join(OUTPUT_DIR, "b1_professional_risk_experiment_results.csv")
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 180)
    print("全部实验结果完整汇总")
    print("=" * 180)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 400):
        print(result_df.to_string(index=False))

    # 压回撤优先
    best_by_drawdown = result_df.sort_values(
        by=["回撤绝对值", "Calmar", "夏普比率", "年化收益率"],
        ascending=[True, False, False, False]
    ).iloc[0]

    print("\n" + "=" * 140)
    print("压回撤优先的最优结果")
    print("=" * 140)
    print(best_by_drawdown.to_string())

    # 收益优先
    best_by_return = result_df.sort_values(
        by=["年化收益率", "Calmar", "夏普比率", "回撤绝对值"],
        ascending=[False, False, False, True]
    ).iloc[0]

    print("\n" + "=" * 140)
    print("收益优先的最优结果")
    print("=" * 140)
    print(best_by_return.to_string())

    # 风险收益平衡优先
    best_balanced = result_df.sort_values(
        by=["Calmar", "夏普比率", "回撤绝对值", "年化收益率"],
        ascending=[False, False, True, False]
    ).iloc[0]

    print("\n" + "=" * 140)
    print("风险收益平衡优先的最优结果")
    print("=" * 140)
    print(best_balanced.to_string())

    pd.DataFrame([best_by_drawdown]).to_csv(
        os.path.join(OUTPUT_DIR, "b1_best_by_drawdown.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([best_by_return]).to_csv(
        os.path.join(OUTPUT_DIR, "b1_best_by_return.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([best_balanced]).to_csv(
        os.path.join(OUTPUT_DIR, "b1_best_balanced.csv"), index=False, encoding="utf-8-sig"
    )

    print("\n结果已保存到：")
    print(out_csv)
    print(os.path.join(OUTPUT_DIR, "b1_best_by_drawdown.csv"))
    print(os.path.join(OUTPUT_DIR, "b1_best_by_return.csv"))
    print(os.path.join(OUTPUT_DIR, "b1_best_balanced.csv"))


if __name__ == "__main__":
    main()