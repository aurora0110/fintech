from __future__ import annotations

import argparse
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
DEFAULT_RESULT_DIR = ROOT / "results/b1_line_slope_compare_20260315"

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

INITIAL_CAPITAL = float(b1bt.INITIAL_CAPITAL)
FEE_RATE = float(b1bt.FEE_RATE)
SLIPPAGE = float(b1bt.SLIPPAGE)

J_RANK_WINDOW = 20
J_RANK_MAX = 0.10
TP_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
SLOPE_WINDOWS = [5, 10, 20, 30]
MAX_HOLD_DAYS = 60
SINGLE_POS_CAP = 0.10
DAY_CASH_CAP = 1.00
MAX_POSITIONS = 999
PAUSE_AFTER_STOPS = 3
PAUSE_DAYS = 5

LINE_DEFS = {
    "trend_line": "趋势线",
    "bull_bear_line": "多空线",
    "MA5": "5日均线",
    "MA10": "10日均线",
    "MA20": "20日均线",
    "MA30": "30日均线",
    "MA60": "60日均线",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare B1 line slope variants.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory for this run's outputs.",
    )
    return parser.parse_args()


def rolling_last_percentile(series: pd.Series, window: int) -> pd.Series:
    values = series.astype(float)

    def _pct_last(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0 or not np.isfinite(arr[-1]):
            return np.nan
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid <= arr[-1]) / len(valid))

    return values.rolling(window, min_periods=window).apply(_pct_last, raw=True)


def prepare_stock(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["J_rank20"] = rolling_last_percentile(x["J"], J_RANK_WINDOW)
    x["base_ok"] = (
        x["trend_line"].notna()
        & x["bull_bear_line"].notna()
        & x["J_rank20"].notna()
        & (x["trend_line"] > x["bull_bear_line"])
        & (x["J_rank20"] < J_RANK_MAX)
    )
    for line_key in LINE_DEFS:
        for window in SLOPE_WINDOWS:
            x[f"slope_{line_key}_{window}"] = x[line_key] / x[line_key].shift(window) - 1.0
    return x


def build_variants() -> List[dict]:
    variants = [
        {
            "variant_id": "base_j20q10",
            "line_key": "",
            "line_name": "无附加斜率",
            "window": 0,
            "description": "仅J<20日历史10%分位且趋势线>多空线",
        }
    ]
    for line_key, line_name in LINE_DEFS.items():
        for window in SLOPE_WINDOWS:
            variants.append(
                {
                    "variant_id": f"{line_key}_slope{window}",
                    "line_key": line_key,
                    "line_name": line_name,
                    "window": window,
                    "description": f"{line_name}{window}日斜率>0 + J<20日历史10%分位 + 趋势线>多空线",
                }
            )
    return variants


def build_signal_sets(
    stock_data: Dict[str, pd.DataFrame],
    all_dates_full: List[pd.Timestamp],
    variants: List[dict],
):
    date_to_idx = {d: i for i, d in enumerate(all_dates_full)}
    pending_by_variant = {v["variant_id"]: {} for v in variants}
    signal_rows = []

    items = list(stock_data.items())
    total = len(items)
    print("构建 B1 线斜率对比信号...")
    for i, (stock_code, df) in enumerate(items, 1):
        x = prepare_stock(df)
        for variant in variants:
            cond = x["base_ok"].copy()
            if variant["line_key"]:
                cond &= x[f"slope_{variant['line_key']}_{variant['window']}"] > 0

            sig_df = x.loc[cond, ["J", "J_rank20"]].copy()
            if sig_df.empty:
                continue

            for signal_date, row in sig_df.iterrows():
                signal_idx = date_to_idx.get(signal_date)
                if signal_idx is None or signal_idx + 1 >= len(all_dates_full):
                    continue
                entry_date = all_dates_full[signal_idx + 1]
                if EXCLUDE_START <= signal_date <= EXCLUDE_END:
                    continue
                if EXCLUDE_START <= entry_date <= EXCLUDE_END:
                    continue
                if entry_date - signal_date > pd.Timedelta(days=15):
                    continue

                pending_by_variant[variant["variant_id"]].setdefault(entry_date, []).append(
                    {
                        "stock": stock_code,
                        "signal_date": signal_date,
                        "j_value": float(row["J"]) if pd.notna(row["J"]) else np.nan,
                        "j_rank20": float(row["J_rank20"]) if pd.notna(row["J_rank20"]) else np.nan,
                    }
                )
                signal_rows.append(
                    {
                        "variant_id": variant["variant_id"],
                        "line_name": variant["line_name"],
                        "window": variant["window"],
                        "stock": stock_code,
                        "signal_date": signal_date,
                        "entry_date": entry_date,
                        "J": float(row["J"]) if pd.notna(row["J"]) else np.nan,
                        "J_rank20": float(row["J_rank20"]) if pd.notna(row["J_rank20"]) else np.nan,
                    }
                )

        if i % 500 == 0 or i == total:
            print(f"信号构建进度: {i}/{total}")

    signal_df = pd.DataFrame(signal_rows)
    return pending_by_variant, signal_df


def get_mark_price(df: pd.DataFrame, current_date: pd.Timestamp) -> float:
    idx = b1bt.get_stock_loc(df, current_date)
    if idx >= 0:
        px = df.iloc[idx]["CLOSE"]
        if pd.notna(px) and px > 0:
            return float(px)
    hist = df.loc[:current_date, "CLOSE"]
    hist = hist[pd.notna(hist) & (hist > 0)]
    return float(hist.iloc[-1]) if len(hist) > 0 else np.nan


def run_account_backtest(
    stock_data: Dict[str, pd.DataFrame],
    all_dates: List[pd.Timestamp],
    pending_buy_signals: dict,
    variant_meta: dict,
    take_profit_pct: float,
):
    cash = INITIAL_CAPITAL
    positions: list[dict] = []
    equity_rows = []
    trade_rows = []
    trade_returns = []
    consecutive_stop_losses = 0
    max_consecutive_stop_losses = 0
    pause_days_left = 0
    pause_event_count = 0
    stop_loss_exit_count = 0
    closed_trade_count = 0
    win_trade_count = 0

    total_days = len(all_dates)

    for day_i, current_date in enumerate(all_dates, 1):
        if pause_days_left > 0:
            pause_days_left -= 1

        current_global_idx = day_i - 1

        # 先执行已经计划好的次日开盘卖出
        remaining_positions = []
        for pos in positions:
            if pos.get("scheduled_exit_date") == current_date:
                df = stock_data[pos["stock"]]
                stock_idx = b1bt.get_stock_loc(df, current_date)
                if stock_idx < 0:
                    remaining_positions.append(pos)
                    continue
                row = df.iloc[stock_idx]
                exit_price = float(row["OPEN"])
                if not np.isfinite(exit_price) or exit_price <= 0:
                    remaining_positions.append(pos)
                    continue

                proceeds = pos["shares"] * exit_price * (1 - FEE_RATE - SLIPPAGE)
                cash += proceeds
                pnl = exit_price / pos["entry_price"] - 1.0
                hold_days = current_global_idx - pos["entry_global_idx"]

                reason = pos.get("scheduled_exit_reason", "unknown")
                is_stop_exit = reason == "stop_loss"
                if is_stop_exit:
                    stop_loss_exit_count += 1
                    consecutive_stop_losses += 1
                    max_consecutive_stop_losses = max(max_consecutive_stop_losses, consecutive_stop_losses)
                    if consecutive_stop_losses >= PAUSE_AFTER_STOPS:
                        pause_days_left = PAUSE_DAYS
                        pause_event_count += 1
                        consecutive_stop_losses = 0
                else:
                    consecutive_stop_losses = 0

                closed_trade_count += 1
                if pnl > 0:
                    win_trade_count += 1
                trade_returns.append(pnl)
                trade_rows.append(
                    {
                        "variant_id": variant_meta["variant_id"],
                        "line_name": variant_meta["line_name"],
                        "window": variant_meta["window"],
                        "take_profit_pct": take_profit_pct,
                        "stock": pos["stock"],
                        "signal_date": pos["signal_date"],
                        "entry_date": pos["entry_date"],
                        "exit_date": current_date,
                        "exit_reason": reason,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "hold_days": hold_days,
                        "return_pct": pnl * 100,
                    }
                )
            else:
                remaining_positions.append(pos)
        positions = remaining_positions

        # 再检查持仓中的新触发，买入当日不允许卖出，所有触发次日开盘执行
        for pos in positions:
            if pos.get("scheduled_exit_date") is not None:
                continue
            if current_global_idx <= pos["entry_global_idx"]:
                continue

            df = stock_data[pos["stock"]]
            stock_idx = b1bt.get_stock_loc(df, current_date)
            if stock_idx < 0:
                continue

            row = df.iloc[stock_idx]
            high_p = float(row["HIGH"]) if pd.notna(row["HIGH"]) else np.nan
            low_p = float(row["LOW"]) if pd.notna(row["LOW"]) else np.nan
            hold_days = current_global_idx - pos["entry_global_idx"]
            tp_price = pos["entry_price"] * (1.0 + take_profit_pct)

            stop_hit = np.isfinite(low_p) and low_p <= pos["stop_price"]
            take_profit_hit = np.isfinite(high_p) and high_p >= tp_price

            if current_global_idx + 1 >= total_days:
                continue

            if stop_hit:
                pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                pos["scheduled_exit_reason"] = "stop_loss"
            elif take_profit_hit:
                pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                pos["scheduled_exit_reason"] = f"tp_{int(take_profit_pct * 100)}"
            elif hold_days >= MAX_HOLD_DAYS:
                pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                pos["scheduled_exit_reason"] = "time_exit_60"

        # 最后处理买入
        if pause_days_left == 0 and current_date in pending_buy_signals:
            already_holding = {p["stock"] for p in positions}
            candidates = sorted(
                pending_buy_signals[current_date],
                key=lambda x: (
                    np.inf if pd.isna(x["j_rank20"]) else x["j_rank20"],
                    np.inf if pd.isna(x["j_value"]) else x["j_value"],
                    x["stock"],
                ),
            )
            position_value_now = 0.0
            for pos in positions:
                mark_price = get_mark_price(stock_data[pos["stock"]], current_date)
                if pd.notna(mark_price) and mark_price > 0:
                    position_value_now += pos["shares"] * mark_price
            total_equity_now = cash + position_value_now
            day_cash_limit = total_equity_now * DAY_CASH_CAP
            day_used_cash = 0.0

            for item in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break
                if cash <= 0:
                    break
                if day_used_cash >= day_cash_limit:
                    break

                stock = item["stock"]
                if stock in already_holding:
                    continue
                df = stock_data[stock]
                stock_idx = b1bt.get_stock_loc(df, current_date)
                if stock_idx < 0:
                    continue
                row = df.iloc[stock_idx]
                entry_price = float(row["OPEN"]) if pd.notna(row["OPEN"]) else np.nan
                entry_low = float(row["LOW"]) if pd.notna(row["LOW"]) else np.nan
                if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(entry_low) or entry_low <= 0:
                    continue

                position_value_now = 0.0
                for pos in positions:
                    mark_price = get_mark_price(stock_data[pos["stock"]], current_date)
                    if pd.notna(mark_price) and mark_price > 0:
                        position_value_now += pos["shares"] * mark_price
                total_equity_now = cash + position_value_now
                alloc_cap = total_equity_now * SINGLE_POS_CAP
                alloc_day = max(day_cash_limit - day_used_cash, 0.0)
                allocation = min(alloc_cap, alloc_day, cash)
                if allocation <= 0:
                    continue

                shares = int(allocation / (entry_price * (1 + FEE_RATE + SLIPPAGE)) / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * entry_price * (1 + FEE_RATE + SLIPPAGE)
                if cost > cash or cost > alloc_day + 1e-9:
                    continue

                cash -= cost
                day_used_cash += cost
                positions.append(
                    {
                        "stock": stock,
                        "shares": shares,
                        "entry_price": entry_price,
                        "entry_date": current_date,
                        "entry_global_idx": current_global_idx,
                        "signal_date": item["signal_date"],
                        "stop_price": entry_low * 0.90,
                        "scheduled_exit_date": None,
                        "scheduled_exit_reason": None,
                    }
                )
                already_holding.add(stock)

        # 记录净值
        position_value = 0.0
        for pos in positions:
            mark_price = get_mark_price(stock_data[pos["stock"]], current_date)
            if pd.notna(mark_price) and mark_price > 0:
                position_value += pos["shares"] * mark_price
        total_value = cash + position_value
        if not np.isfinite(total_value) or total_value <= 0:
            total_value = equity_rows[-1]["equity"] if equity_rows else INITIAL_CAPITAL
        equity_rows.append(
            {
                "variant_id": variant_meta["variant_id"],
                "line_name": variant_meta["line_name"],
                "window": variant_meta["window"],
                "take_profit_pct": take_profit_pct,
                "date": current_date,
                "equity": total_value,
                "cash": cash,
                "position_value": position_value,
                "open_positions": len(positions),
            }
        )

        if day_i % 100 == 0 or day_i == total_days:
            print(
                f"账户回测进度 {variant_meta['variant_id']} TP{int(take_profit_pct * 100)}: "
                f"{day_i}/{total_days}"
            )

    equity_df = pd.DataFrame(equity_rows)
    equity_arr = equity_df["equity"].to_numpy(dtype=float)
    daily_ret = equity_df["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    final_capital = float(equity_arr[-1]) if len(equity_arr) > 0 else INITIAL_CAPITAL
    final_multiple = final_capital / INITIAL_CAPITAL
    max_dd = b1bt.calc_max_drawdown(equity_arr)
    sharpe = b1bt.calc_sharpe(daily_ret)
    cagr = b1bt.calc_cagr(final_multiple, len(all_dates))
    trade_win_rate = win_trade_count / closed_trade_count * 100 if closed_trade_count > 0 else np.nan
    avg_trade_return = np.mean(trade_returns) * 100 if trade_returns else np.nan

    result_row = {
        "variant_id": variant_meta["variant_id"],
        "line_name": variant_meta["line_name"],
        "window": variant_meta["window"],
        "description": variant_meta["description"],
        "take_profit_pct": take_profit_pct,
        "initial_capital": INITIAL_CAPITAL,
        "single_pos_cap": SINGLE_POS_CAP,
        "day_cash_cap": DAY_CASH_CAP,
        "max_positions": MAX_POSITIONS,
        "max_hold_days": MAX_HOLD_DAYS,
        "stop_price_rule": "买入K线最低价*0.9",
        "execution_rule": "买入当日不能卖出，止盈止损次日开盘成交",
        "closed_trade_count": closed_trade_count,
        "stop_loss_exit_count": stop_loss_exit_count,
        "trade_win_rate_pct": trade_win_rate,
        "avg_trade_return_pct": avg_trade_return,
        "final_capital": final_capital,
        "final_multiple": final_multiple,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "pause_event_count": pause_event_count,
        "max_consecutive_stop_losses": max_consecutive_stop_losses,
        "open_positions_end": len(positions),
    }
    return result_row, trade_rows, equity_rows


def main(result_dir: Path | None = None):
    result_dir = (result_dir or DEFAULT_RESULT_DIR).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    stock_data_raw, all_dates_full = b1bt.load_all_data(DATA_DIR)
    stock_data = {code: df for code, df in stock_data_raw.items()}
    all_dates = [d for d in all_dates_full if not (EXCLUDE_START <= d <= EXCLUDE_END)]
    variants = build_variants()

    pending_by_variant, signal_df = build_signal_sets(stock_data, all_dates_full, variants)
    signal_df.to_csv(result_dir / "signals.csv", index=False, encoding="utf-8-sig")

    signal_summary_rows = []
    for variant in variants:
        var_id = variant["variant_id"]
        subset = signal_df[signal_df["variant_id"] == var_id]
        signal_summary_rows.append(
            {
                "variant_id": var_id,
                "line_name": variant["line_name"],
                "window": variant["window"],
                "description": variant["description"],
                "signal_count": int(len(subset)),
                "signal_days": int(subset["signal_date"].nunique()),
                "signal_stocks": int(subset["stock"].nunique()),
            }
        )
    signal_summary_df = pd.DataFrame(signal_summary_rows).sort_values(["signal_count", "variant_id"], ascending=[False, True])
    signal_summary_df.to_csv(result_dir / "signal_summary.csv", index=False, encoding="utf-8-sig")

    result_rows = []
    all_trade_rows = []
    all_equity_rows = []
    partial_summary = {"completed_combos": 0, "total_combos": len(variants) * len(TP_LEVELS)}

    for variant in variants:
        pending = pending_by_variant[variant["variant_id"]]
        for tp in TP_LEVELS:
            result_row, trade_rows, equity_rows = run_account_backtest(
                stock_data=stock_data,
                all_dates=all_dates,
                pending_buy_signals=pending,
                variant_meta=variant,
                take_profit_pct=tp,
            )
            result_rows.append(result_row)
            all_trade_rows.extend(trade_rows)
            all_equity_rows.extend(equity_rows)

            pd.DataFrame(result_rows).to_csv(
                result_dir / "account_results_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(all_equity_rows).to_csv(
                result_dir / "equity_curve_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )
            partial_summary["completed_combos"] += 1
            partial_summary["last_variant_id"] = variant["variant_id"]
            partial_summary["last_take_profit_pct"] = tp
            (result_dir / "progress.json").write_text(
                json.dumps(partial_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    results_df = pd.DataFrame(result_rows).sort_values(
        ["trade_win_rate_pct", "avg_trade_return_pct", "cagr_pct"],
        ascending=[False, False, False],
    )
    results_df.to_csv(result_dir / "account_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_trade_rows).to_csv(result_dir / "trade_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_equity_rows).to_csv(result_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")

    best_trade_win = results_df.sort_values(
        ["trade_win_rate_pct", "avg_trade_return_pct", "cagr_pct"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    best_trade_return = results_df.sort_values(
        ["avg_trade_return_pct", "trade_win_rate_pct", "cagr_pct"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    best_account = results_df.sort_values(
        ["cagr_pct", "final_multiple", "trade_win_rate_pct"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()

    summary = {
        "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "signal_time_axis": {
            "signal_date": "T",
            "entry_date": "T+1开盘",
            "first_eligible_exit_date": "T+2起",
            "execution_rule": "止盈止损触发后次日开盘成交",
            "same_day_tp_sl_priority": "止损优先",
        },
        "buy_condition": "J<20日历史10%分位 且 趋势线>多空线 且 附加线斜率>0",
        "tp_levels": TP_LEVELS,
        "stop_rule": "买入K线最低价*0.9",
        "max_hold_days": MAX_HOLD_DAYS,
        "capital_rule": {
            "initial_capital": INITIAL_CAPITAL,
            "single_pos_cap": SINGLE_POS_CAP,
            "day_cash_cap": DAY_CASH_CAP,
            "max_positions": MAX_POSITIONS,
            "max_new_buys_per_day": "无限制",
            "pause_rule": "连续止损3次暂停5天",
        },
        "variant_count": len(variants),
        "combo_count": len(results_df),
        "best_trade_win": best_trade_win,
        "best_trade_return": best_trade_return,
        "best_account": best_account,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(result_dir=args.result_dir)
