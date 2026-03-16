from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.backtest import backtest_b1_strategy as b1bt  # type: ignore


DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
RESULT_DIR = ROOT / "results/b1_distribution_exit_ab_20260314"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

BASE_PARAMS = {
    "max_positions": 10,
    "max_new_buys_per_day": 1,
    "day_cash_cap": 0.30,
    "single_pos_cap": 0.10,
    "pause_rule": "loss_streak_3_pause_5",
    "regime_mode": "none",
}


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def add_distribution_labels_upper(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["vol_ma20"] = x["VOLUME"].rolling(20).mean()
    x["vol_rank30"] = x["VOLUME"].rolling(30, min_periods=30).apply(
        lambda v: pd.Series(v).rank(pct=True).iloc[-1], raw=False
    )
    x["vol_rank60"] = x["VOLUME"].rolling(60, min_periods=60).apply(
        lambda v: pd.Series(v).rank(pct=True).iloc[-1], raw=False
    )
    x["high30_prev"] = x["HIGH"].shift(1).rolling(30).max()
    x["high60_prev"] = x["HIGH"].shift(1).rolling(60).max()
    x["close20_max_prev"] = x["CLOSE"].shift(1).rolling(20).max()
    x["close60_prev"] = x["CLOSE"].shift(1).rolling(60).max()
    x["ret5_prev"] = x["CLOSE"].shift(1) / x["CLOSE"].shift(6) - 1.0
    x["ret10_prev"] = x["CLOSE"].shift(1) / x["CLOSE"].shift(11) - 1.0
    x["ret20_prev"] = x["CLOSE"].shift(1) / x["CLOSE"].shift(21) - 1.0
    x["trend_slope_5_prev"] = x["trend_line"].shift(1) / x["trend_line"].shift(6) - 1.0
    x["trend_slope_10_prev"] = x["trend_line"].shift(1) / x["trend_line"].shift(11) - 1.0
    x["is_bear"] = x["CLOSE"] < x["OPEN"]
    x["is_bull"] = x["CLOSE"] > x["OPEN"]
    x["bear_vol"] = np.where(x["is_bear"], x["VOLUME"], 0.0)
    x["bull_vol"] = np.where(x["is_bull"], x["VOLUME"], 0.0)
    x["upper_shadow"] = np.maximum(x["HIGH"] - x[["OPEN", "CLOSE"]].max(axis=1), 0.0)
    x["close_position"] = pd.Series(
        _safe_div(x["CLOSE"] - x["LOW"], (x["HIGH"] - x["LOW"]).replace(0.0, np.nan)),
        index=x.index,
    ).clip(lower=0.0, upper=1.0)
    x["upper_range_ratio"] = pd.Series(
        _safe_div(x["upper_shadow"], (x["HIGH"] - x["LOW"]).replace(0.0, np.nan)),
        index=x.index,
    ).clip(lower=0.0)
    x["body_ratio"] = pd.Series(
        _safe_div((x["CLOSE"] - x["OPEN"]).abs(), (x["HIGH"] - x["LOW"]).replace(0.0, np.nan)),
        index=x.index,
    ).clip(lower=0.0)
    x["top_zone_now"] = (
        (x["HIGH"] >= x["high30_prev"] * 0.97)
        | (x["HIGH"] >= x["high60_prev"] * 0.95)
        | (x["CLOSE"] >= x["close20_max_prev"] * 0.94)
    )
    x["top_zone_recent"] = pd.Series(x["top_zone_now"].fillna(False)).rolling(12, min_periods=1).max().astype(bool)

    point_accel_heavy_bear = (
        x["is_bear"]
        & (x["body_ratio"] >= 0.45)
        & (x["close_position"] <= 0.36)
        & ((x["vol_rank30"] >= 0.90) | (x["vol_rank60"] >= 0.93) | (x["VOLUME"] >= x["vol_ma20"] * 1.15))
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
        (x["HIGH"] >= x["high30_prev"] * 0.995)
        & (x["CLOSE"] <= x["high30_prev"] * 0.985)
        & (x["upper_range_ratio"] >= 0.28)
        & (x["VOLUME"] >= x["vol_ma20"] * 1.2)
        & x["top_zone_now"]
    )

    x["point_any"] = (point_accel_heavy_bear | point_failed_breakout).fillna(False)

    x["bear_vol_sum10"] = pd.Series(x["bear_vol"]).rolling(10).sum()
    x["bull_vol_sum10"] = pd.Series(x["bull_vol"]).rolling(10).sum()
    x["bear_days_10"] = pd.Series(x["is_bear"].astype(int)).rolling(10).sum()
    zone_top_distribution = (
        (x["bear_days_10"] >= 5)
        & (x["bear_vol_sum10"] >= x["bull_vol_sum10"] * 1.20)
        & x["top_zone_recent"]
    )

    had_new_high_recent = pd.Series(x["top_zone_now"].shift(1).fillna(False)).rolling(8, min_periods=1).max().astype(bool)
    recent_bear_4 = pd.Series(x["is_bear"].astype(int)).rolling(4).sum() >= 2
    zone_post_new_high_selloff = (
        had_new_high_recent.fillna(False)
        & recent_bear_4.fillna(False)
        & (x["CLOSE"] <= x["close20_max_prev"] * 0.95)
        & ((x["VOLUME"] >= x["vol_ma20"] * 1.00) | (x["vol_rank30"] >= 0.85))
    )

    stair_active = pd.Series(False, index=x.index)
    point_arr = x["point_any"].fillna(False).to_numpy(dtype=bool)
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
        if not bool((sub["CLOSE"] < sub["OPEN"]).all()):
            continue
        if not bool((sub["VOLUME"].diff().fillna(0) <= 0).iloc[1:].all()):
            continue
        if not bool((sub["CLOSE"].diff().fillna(0) <= 0).iloc[1:].all()):
            continue
        stair_active.iat[i] = True

    x["zone_any"] = (zone_top_distribution | zone_post_new_high_selloff | stair_active).fillna(False)
    x["zone_end"] = x["zone_any"] & (~x["zone_any"].shift(-1).fillna(False))
    x["distribution_sell"] = x["point_any"] | x["zone_end"]
    return x


def build_signals(stock_data: Dict[str, pd.DataFrame]) -> dict:
    daily_scores: dict[pd.Timestamp, list[dict]] = {}
    items = list(stock_data.items())
    total = len(items)
    print("构建 B1(J20分位10%) 信号...")
    for i, (stock_code, df) in enumerate(items, 1):
        x = df.copy()
        x["J_Q10_20"] = x["J"].rolling(20).quantile(0.10)
        cond = x["J"].notna() & x["J_Q10_20"].notna() & (x["J"] <= x["J_Q10_20"])
        sig_df = x.loc[cond, ["OPEN", "HIGH", "LOW", "CLOSE", "J", "J_Q10_20", "ATR14"]].copy()
        sig_df["score"] = (-sig_df["J"]).fillna(0.0)
        if not sig_df.empty:
            for dt, row in sig_df.iterrows():
                if EXCLUDE_START <= dt <= EXCLUDE_END:
                    continue
                daily_scores.setdefault(dt, []).append(
                    {
                        "stock": stock_code,
                        "score": float(row["score"]),
                        "J": float(row["J"]),
                        "J_Q10_20": float(row["J_Q10_20"]),
                        "ATR14": float(row["ATR14"]) if pd.notna(row["ATR14"]) else np.nan,
                    }
                )
        if i % 500 == 0 or i == total:
            print(f"信号构建进度: {i}/{total}")
    return daily_scores


def generate_pending_buy_signals_filtered(daily_scores: dict, all_dates_full: list[pd.Timestamp]) -> dict:
    date_to_idx = {d: i for i, d in enumerate(all_dates_full)}
    pending_buy: dict[pd.Timestamp, list[dict]] = {}
    for signal_date, items in daily_scores.items():
        i = date_to_idx.get(signal_date)
        if i is None or i + 1 >= len(all_dates_full):
            continue
        next_date = all_dates_full[i + 1]
        if EXCLUDE_START <= next_date <= EXCLUDE_END:
            continue
        if next_date - signal_date > pd.Timedelta(days=15):
            continue
        pending_buy.setdefault(next_date, []).extend(items)
    return pending_buy


def run_backtest_custom(
    stock_data: dict,
    all_dates: list,
    pending_buy_signals: dict,
    regime_df: pd.DataFrame,
    params: dict,
    exp_name: str,
    use_distribution_exit: bool,
):
    max_positions = params["max_positions"]
    max_new_buys_per_day = params["max_new_buys_per_day"]
    max_hold_days = params["max_hold_days"]
    day_cash_cap = params["day_cash_cap"]
    single_pos_cap = params["single_pos_cap"]
    pause_rule = params["pause_rule"]
    regime_mode = params["regime_mode"]

    cash = float(b1bt.INITIAL_CAPITAL)
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

    total_days = len(all_dates)
    print(f"\n开始回测：{exp_name}")

    for day_i, current_date in enumerate(all_dates, 1):
        if pause_trading_days > 0:
            pause_trading_days -= 1

        current_global_idx = date_to_idx[current_date]

        new_positions = []
        for pos in positions:
            stock = pos["stock"]
            df = stock_data[stock]
            stock_idx = b1bt.get_stock_loc(df, current_date)
            if stock_idx < 0:
                new_positions.append(pos)
                continue

            row = df.iloc[stock_idx]
            open_p = row["OPEN"]
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

                if pd.notna(close_p) and pd.notna(pos["stop_price"]) and close_p < pos["stop_price"]:
                    if current_global_idx + 1 < len(all_dates):
                        pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                        pos["scheduled_exit_reason"] = "止损"
                        pos["partial_exit_ratio"] = 1.0
                        pos["exit_marked"] = True

                if (not pos.get("exit_marked", False)) and use_distribution_exit and bool(row.get("distribution_sell", False)):
                    if current_global_idx + 1 < len(all_dates):
                        pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                        pos["scheduled_exit_reason"] = "出货标签"
                        pos["partial_exit_ratio"] = 1.0
                        pos["exit_marked"] = True

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
                sell_value = pos["shares"] * exit_price * (1 - b1bt.FEE_RATE - b1bt.SLIPPAGE)
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

                loss_streak_trigger, loss_pause_days = b1bt.get_loss_streak_params(pause_rule)
                if loss_streak_trigger is not None and current_consecutive_losses >= loss_streak_trigger:
                    pause_trading_days = loss_pause_days
                if b1bt.should_pause_by_trade_window(trade_pnls, pause_rule):
                    pause_trading_days = 5
            else:
                new_positions.append(pos)
        positions = new_positions

        if pause_trading_days == 0 and current_date in pending_buy_signals:
            regime_row = regime_df.loc[current_date] if current_date in regime_df.index else None
            if b1bt.regime_pass(regime_row, regime_mode):
                available_slots = max_positions - len(positions)
                if available_slots > 0 and cash > 0:
                    candidates = sorted(pending_buy_signals[current_date], key=lambda x: x["score"], reverse=True)
                    already_holding = {p["stock"] for p in positions}
                    bought_today = 0
                    day_used_cash = 0.0
                    day_cash_limit = (cash + sum(
                        p["shares"] * b1bt.get_mark_price(stock_data[p["stock"]], current_date)
                        for p in positions if pd.notna(b1bt.get_mark_price(stock_data[p["stock"]], current_date))
                    )) * day_cash_cap

                    for item in candidates:
                        if available_slots <= 0 or cash <= 0 or bought_today >= max_new_buys_per_day:
                            break
                        stock = item["stock"]
                        if stock in already_holding:
                            continue
                        df = stock_data[stock]
                        stock_idx = b1bt.get_stock_loc(df, current_date)
                        if stock_idx < 0:
                            continue
                        row = df.iloc[stock_idx]
                        open_price = row["OPEN"]
                        low_price = row["LOW"]
                        if pd.isna(open_price) or open_price <= 0 or pd.isna(low_price) or low_price <= 0:
                            continue
                        total_equity_now = cash + sum(
                            p["shares"] * b1bt.get_mark_price(stock_data[p["stock"]], current_date)
                            for p in positions if pd.notna(b1bt.get_mark_price(stock_data[p["stock"]], current_date))
                        )
                        alloc_by_slot = cash / available_slots
                        alloc_by_single_cap = total_equity_now * single_pos_cap
                        alloc_by_day_cap = max(day_cash_limit - day_used_cash, 0.0)
                        allocation = min(alloc_by_slot, alloc_by_single_cap, alloc_by_day_cap)
                        if allocation <= 0:
                            continue
                        shares = int(allocation / (open_price * (1 + b1bt.FEE_RATE + b1bt.SLIPPAGE)) / 100) * 100
                        if shares <= 0:
                            continue
                        cost = shares * open_price * (1 + b1bt.FEE_RATE + b1bt.SLIPPAGE)
                        if cost > cash:
                            continue
                        cash -= cost
                        day_used_cash += cost
                        positions.append(
                            {
                                "stock": stock,
                                "shares": shares,
                                "entry_price": open_price,
                                "entry_low": low_price,
                                "entry_date": current_date,
                                "entry_global_idx": current_global_idx,
                                "stop_price": low_price * 0.95,
                                "score": item["score"],
                                "exit_marked": False,
                                "scheduled_exit_date": None,
                                "scheduled_exit_reason": None,
                                "partial_exit_ratio": None,
                            }
                        )
                        already_holding.add(stock)
                        available_slots -= 1
                        bought_today += 1

        position_value = 0.0
        for pos in positions:
            stock = pos["stock"]
            mark_price = b1bt.get_mark_price(stock_data[stock], current_date)
            if pd.notna(mark_price) and mark_price > 0:
                position_value += pos["shares"] * mark_price

        total_value = cash + position_value
        if total_value <= 0 or not np.isfinite(total_value):
            total_value = equity_curve[-1] if equity_curve else b1bt.INITIAL_CAPITAL
        equity_curve.append(total_value)

        if day_i % b1bt.BACKTEST_PROGRESS_STEP == 0 or day_i == total_days:
            print(f"回测进度: {day_i}/{total_days}")

    equity_arr = np.array(equity_curve, dtype=float)
    daily_ret = np.array([
        (equity_arr[i + 1] - equity_arr[i]) / equity_arr[i]
        for i in range(len(equity_arr) - 1)
        if equity_arr[i] > 0 and np.isfinite(equity_arr[i + 1])
    ])
    final_capital = float(equity_arr[-1]) if len(equity_arr) > 0 else b1bt.INITIAL_CAPITAL
    final_multiple = final_capital / b1bt.INITIAL_CAPITAL
    max_dd = b1bt.calc_max_drawdown(equity_arr)
    sharpe = b1bt.calc_sharpe(daily_ret)
    cagr = b1bt.calc_cagr(final_multiple, len(all_dates))

    success_rate = win_count / trade_count * 100 if trade_count > 0 else np.nan
    avg_holding_return = np.mean(holding_returns) * 100 if holding_returns else np.nan
    max_holding_return = np.max(holding_returns) * 100 if holding_returns else np.nan
    profit_factor = b1bt.calc_profit_factor(pd.Series(holding_returns)) if holding_returns else np.nan
    calmar = (cagr * 100) / abs(max_dd * 100) if pd.notna(cagr) and pd.notna(max_dd) and abs(max_dd) > 1e-12 else np.nan

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
        "use_distribution_exit": use_distribution_exit,
    }


def main():
    stock_data, all_dates_full = b1bt.load_all_data(DATA_DIR)
    for code in list(stock_data.keys()):
        stock_data[code] = add_distribution_labels_upper(stock_data[code])

    all_dates = [d for d in all_dates_full if not (EXCLUDE_START <= d <= EXCLUDE_END)]
    regime_df = b1bt.build_market_regime(stock_data, all_dates)
    daily_scores = build_signals(stock_data)
    pending_buy = generate_pending_buy_signals_filtered(daily_scores, all_dates_full)

    rows = []
    for max_hold_days in [2, 5, 10, 20, 30]:
        for use_distribution_exit in [False, True]:
            params = {
                **BASE_PARAMS,
                "max_hold_days": max_hold_days,
            }
            name = f"B1_J20Q10_hold{max_hold_days}_{'stop_plus_dist' if use_distribution_exit else 'stop_only'}"
            res = run_backtest_custom(
                stock_data=stock_data,
                all_dates=all_dates,
                pending_buy_signals=pending_buy,
                regime_df=regime_df,
                params=params,
                exp_name=name,
                use_distribution_exit=use_distribution_exit,
            )
            rows.append(res)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    summary = {
        "signal_days": int(len(daily_scores)),
        "buy_signal_count": int(sum(len(v) for v in pending_buy.values())),
        "comparison_rows": int(len(result_df)),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
