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


SIGNAL_DIR = ROOT / "results/b1_similarity_ml_signal_v4_20260320_220031"
SELECTED_ROWS_PATH = SIGNAL_DIR / "final_test_selected_rows.csv"
FINAL_REPORT_PATH = SIGNAL_DIR / "final_test_report.csv"
DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
DEFAULT_RESULT_DIR = ROOT / "results/b1_similarity_account_v1_20260320"

INITIAL_CAPITAL = float(b1bt.INITIAL_CAPITAL)
FEE_RATE = float(b1bt.FEE_RATE)
SLIPPAGE = float(b1bt.SLIPPAGE)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

TP_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
MAX_HOLD_DAYS = 60
STOP_MULTIPLIER = 0.90
SINGLE_POS_CAP = 0.10
DAY_CASH_CAP = 1.00
MAX_POSITIONS = 999
PAUSE_AFTER_STOPS = 3
PAUSE_DAYS = 5
PROGRESS_STEP = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 相似度/机器学习策略账户层回测")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="本轮结果目录",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, extra: dict | None = None) -> None:
    payload = {"stage": stage}
    if extra:
        payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def load_selected_rows() -> pd.DataFrame:
    df = pd.read_csv(SELECTED_ROWS_PATH)
    for col in ["signal_date", "entry_date"]:
        df[col] = pd.to_datetime(df[col])
    df["strategy_tag"] = df["strategy_tag"].astype(str)
    df["code"] = df["code"].astype(str)
    return df


def load_strategy_meta() -> pd.DataFrame:
    df = pd.read_csv(FINAL_REPORT_PATH)
    df["strategy_tag"] = (
        df["family"].astype(str)
        + "_"
        + df["variant"].astype(str)
        + "_"
        + df["pool"].astype(str)
        + "_top"
        + df["topn"].astype(str)
    )
    return df


def get_mark_price(df: pd.DataFrame, current_date: pd.Timestamp) -> float:
    idx = b1bt.get_stock_loc(df, current_date)
    if idx >= 0:
        px = df.iloc[idx]["CLOSE"]
        if pd.notna(px) and px > 0:
            return float(px)
    hist = df.loc[:current_date, "CLOSE"]
    hist = hist[pd.notna(hist) & (hist > 0)]
    return float(hist.iloc[-1]) if len(hist) > 0 else np.nan


def build_pending_signals(
    signal_df: pd.DataFrame,
    all_dates_full: List[pd.Timestamp],
) -> Dict[pd.Timestamp, List[dict]]:
    date_to_idx = {d: i for i, d in enumerate(all_dates_full)}
    pending: Dict[pd.Timestamp, List[dict]] = {}
    items = signal_df.sort_values(["entry_date", "signal_date", "code"]).to_dict("records")
    total = len(items)
    for i, row in enumerate(items, 1):
        signal_date = pd.Timestamp(row["signal_date"])
        entry_date = pd.Timestamp(row["entry_date"])
        if EXCLUDE_START <= signal_date <= EXCLUDE_END:
            continue
        if EXCLUDE_START <= entry_date <= EXCLUDE_END:
            continue
        signal_idx = date_to_idx.get(signal_date)
        entry_idx = date_to_idx.get(entry_date)
        if signal_idx is None or entry_idx is None:
            continue
        if entry_idx != signal_idx + 1:
            continue
        pending.setdefault(entry_date, []).append(
            {
                "stock": str(row["code"]),
                "signal_date": signal_date,
                "entry_date": entry_date,
            }
        )
        if i % PROGRESS_STEP == 0 or i == total:
            print(f"信号整理进度: {i}/{total}")
    return pending


def summarize_account(
    equity_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    stop_loss_exit_count: int,
    pause_event_count: int,
    max_consecutive_stop_losses: int,
) -> dict:
    equity_arr = equity_df["equity"].to_numpy(dtype=float)
    daily_ret = equity_df["daily_return"].fillna(0.0).to_numpy(dtype=float)
    final_multiple = float(equity_arr[-1] / INITIAL_CAPITAL) if len(equity_arr) else np.nan
    total_days = int(len(equity_df))
    closed_trade_count = int(len(trade_df))
    win_trade_count = int((trade_df["return_pct"] > 0).sum()) if closed_trade_count else 0
    win_rate = float(win_trade_count / closed_trade_count) if closed_trade_count else np.nan
    avg_trade_return = float(trade_df["return_pct"].mean() / 100.0) if closed_trade_count else np.nan
    profit_factor = float(b1bt.calc_profit_factor(trade_df["return_pct"] / 100.0)) if closed_trade_count else np.nan
    return {
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": float(equity_arr[-1]) if len(equity_arr) else np.nan,
        "final_multiple": final_multiple,
        "cagr": float(b1bt.calc_cagr(final_multiple, total_days)) if len(equity_arr) else np.nan,
        "max_drawdown": float(b1bt.calc_max_drawdown(equity_arr)) if len(equity_arr) else np.nan,
        "sharpe": float(b1bt.calc_sharpe(daily_ret)) if len(equity_arr) > 1 else np.nan,
        "trade_count": closed_trade_count,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "profit_factor": profit_factor,
        "stop_loss_exit_count": int(stop_loss_exit_count),
        "pause_event_count": int(pause_event_count),
        "max_consecutive_stop_losses": int(max_consecutive_stop_losses),
    }


def run_account_backtest(
    stock_data: Dict[str, pd.DataFrame],
    all_dates: List[pd.Timestamp],
    pending_buy_signals: Dict[pd.Timestamp, List[dict]],
    strategy_tag: str,
    take_profit_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cash = INITIAL_CAPITAL
    positions: list[dict] = []
    equity_rows = []
    trade_rows = []
    consecutive_stop_losses = 0
    max_consecutive_stop_losses = 0
    pause_days_left = 0
    pause_event_count = 0
    stop_loss_exit_count = 0

    total_days = len(all_dates)

    for day_i, current_date in enumerate(all_dates, 1):
        current_global_idx = day_i - 1
        if pause_days_left > 0:
            pause_days_left -= 1

        # 先执行次日开盘卖出
        remaining_positions = []
        for pos in positions:
            if pos.get("scheduled_exit_date") != current_date:
                remaining_positions.append(pos)
                continue
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

            if reason == "stop_loss":
                stop_loss_exit_count += 1
                consecutive_stop_losses += 1
                max_consecutive_stop_losses = max(max_consecutive_stop_losses, consecutive_stop_losses)
                if consecutive_stop_losses >= PAUSE_AFTER_STOPS:
                    pause_days_left = PAUSE_DAYS
                    pause_event_count += 1
                    consecutive_stop_losses = 0
            else:
                consecutive_stop_losses = 0

            trade_rows.append(
                {
                    "strategy_tag": strategy_tag,
                    "take_profit_pct": take_profit_pct,
                    "stock": pos["stock"],
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "exit_reason": reason,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "hold_days": hold_days,
                    "return_pct": pnl * 100.0,
                }
            )

        positions = remaining_positions

        # 买入当日不能卖，持仓触发后次日开盘执行；止损优先
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
            tp_hit = np.isfinite(high_p) and high_p >= tp_price
            if current_global_idx + 1 >= total_days:
                continue
            if stop_hit:
                pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                pos["scheduled_exit_reason"] = "stop_loss"
            elif tp_hit:
                pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                pos["scheduled_exit_reason"] = f"tp_{int(take_profit_pct * 100)}"
            elif hold_days >= MAX_HOLD_DAYS:
                pos["scheduled_exit_date"] = all_dates[current_global_idx + 1]
                pos["scheduled_exit_reason"] = "time_exit_60"

        # 再执行买入
        if pause_days_left == 0 and current_date in pending_buy_signals:
            already_holding = {p["stock"] for p in positions}
            candidates = sorted(
                pending_buy_signals[current_date],
                key=lambda x: (x["signal_date"], x["stock"]),
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
                if item["stock"] in already_holding:
                    continue
                df = stock_data.get(item["stock"])
                if df is None:
                    continue
                stock_idx = b1bt.get_stock_loc(df, current_date)
                if stock_idx < 0:
                    continue
                row = df.iloc[stock_idx]
                entry_price = float(row["OPEN"])
                entry_low = float(row["LOW"])
                if (
                    not np.isfinite(entry_price)
                    or entry_price <= 0
                    or not np.isfinite(entry_low)
                    or entry_low <= 0
                ):
                    continue

                position_cap_cash = total_equity_now * SINGLE_POS_CAP
                budget = min(cash, position_cap_cash, day_cash_limit - day_used_cash)
                if budget <= 0:
                    continue
                shares = int(budget // (entry_price * (1 + FEE_RATE + SLIPPAGE)))
                if shares <= 0:
                    continue
                cost = shares * entry_price * (1 + FEE_RATE + SLIPPAGE)
                if cost > cash + 1e-9:
                    continue

                cash -= cost
                day_used_cash += cost
                positions.append(
                    {
                        "stock": item["stock"],
                        "signal_date": item["signal_date"],
                        "entry_date": current_date,
                        "entry_price": entry_price,
                        "shares": shares,
                        "stop_price": entry_low * STOP_MULTIPLIER,
                        "entry_global_idx": current_global_idx,
                        "scheduled_exit_date": None,
                        "scheduled_exit_reason": None,
                    }
                )
                already_holding.add(item["stock"])

        # 记录净值
        holdings_value = 0.0
        for pos in positions:
            mark_price = get_mark_price(stock_data[pos["stock"]], current_date)
            if pd.notna(mark_price) and mark_price > 0:
                holdings_value += pos["shares"] * mark_price
        equity = cash + holdings_value
        prev_equity = equity_rows[-1]["equity"] if equity_rows else INITIAL_CAPITAL
        daily_return = equity / prev_equity - 1.0 if prev_equity > 0 else np.nan
        equity_rows.append(
            {
                "date": current_date,
                "equity": equity,
                "cash": cash,
                "holdings_value": holdings_value,
                "position_count": len(positions),
                "daily_return": daily_return,
            }
        )

        if day_i % PROGRESS_STEP == 0 or day_i == total_days:
            print(f"{strategy_tag} TP{int(take_profit_pct * 100)} 账户层进度: {day_i}/{total_days}")

    equity_df = pd.DataFrame(equity_rows)
    trade_df = pd.DataFrame(trade_rows)
    summary = summarize_account(
        equity_df,
        trade_df,
        stop_loss_exit_count=stop_loss_exit_count,
        pause_event_count=pause_event_count,
        max_consecutive_stop_losses=max_consecutive_stop_losses,
    )
    return equity_df, trade_df, summary


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "loading_inputs")

    selected_df = load_selected_rows()
    meta_df = load_strategy_meta()
    selected_df.to_csv(result_dir / "signals_for_backtest.csv", index=False, encoding="utf-8-sig")

    strategy_tags = meta_df["strategy_tag"].tolist()
    chosen_meta = meta_df[["strategy_tag", "family", "variant", "pool", "topn"]].copy()
    chosen_meta.to_csv(result_dir / "strategy_meta.csv", index=False, encoding="utf-8-sig")

    print("加载 forward_data...")
    stock_data, all_dates = b1bt.load_all_data(DATA_DIR)
    all_dates = pd.to_datetime(all_dates)
    update_progress(result_dir, "signals_ready", {"strategy_count": len(strategy_tags), "date_count": len(all_dates)})

    account_rows = []
    all_trade_rows = []

    for s_idx, strategy_tag in enumerate(strategy_tags, 1):
        strategy_rows = selected_df[selected_df["strategy_tag"] == strategy_tag].copy()
        pending = build_pending_signals(strategy_rows, list(all_dates))
        strategy_rows.to_csv(
            result_dir / f"selected_{strategy_tag}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        for tp in TP_LEVELS:
            update_progress(
                result_dir,
                "backtesting",
                {
                    "strategy_idx": s_idx,
                    "strategy_count": len(strategy_tags),
                    "strategy_tag": strategy_tag,
                    "take_profit_pct": tp,
                },
            )
            equity_df, trade_df, summary = run_account_backtest(
                stock_data=stock_data,
                all_dates=list(all_dates),
                pending_buy_signals=pending,
                strategy_tag=strategy_tag,
                take_profit_pct=tp,
            )
            equity_name = f"equity_{strategy_tag}_tp{int(tp * 100)}.csv"
            trade_name = f"trades_{strategy_tag}_tp{int(tp * 100)}.csv"
            equity_df.to_csv(result_dir / equity_name, index=False, encoding="utf-8-sig")
            trade_df.to_csv(result_dir / trade_name, index=False, encoding="utf-8-sig")

            row = {"strategy_tag": strategy_tag, "take_profit_pct": tp}
            row.update(summary)
            account_rows.append(row)
            if not trade_df.empty:
                x = trade_df.copy()
                x["take_profit_pct"] = tp
                x["strategy_tag"] = strategy_tag
                all_trade_rows.append(x)

            pd.DataFrame(account_rows).to_csv(
                result_dir / "account_results_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    result_df = pd.DataFrame(account_rows).sort_values(
        ["final_multiple", "cagr", "win_rate"],
        ascending=[False, False, False],
    )
    result_df.to_csv(result_dir / "account_results.csv", index=False, encoding="utf-8-sig")
    if all_trade_rows:
        pd.concat(all_trade_rows, ignore_index=True).to_csv(
            result_dir / "all_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )

    best_row = result_df.iloc[0].to_dict() if not result_df.empty else {}
    summary = {
        "result_dir": str(result_dir),
        "signal_source": str(SELECTED_ROWS_PATH),
        "strategy_count": int(len(strategy_tags)),
        "take_profit_levels": TP_LEVELS,
        "stop_multiplier": STOP_MULTIPLIER,
        "max_hold_days": MAX_HOLD_DAYS,
        "best_account_row": best_row,
    }
    write_json(result_dir / "summary.json", summary)
    update_progress(result_dir, "finished", {"result_rows": int(len(result_df))})


if __name__ == "__main__":
    main()
