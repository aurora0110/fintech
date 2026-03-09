from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

from core.metrics import compute_metrics
from utils.multi_factor_research.data_processor import load_stock_directory
from utils.multi_factor_research.research_metrics import summarize_trade_metrics
from strategies.common import base_prepare


def prepare_stock(df: pd.DataFrame, first_cross_window: int) -> pd.DataFrame:
    out = base_prepare(df.copy())
    out["cross_up"] = out["short_trend"].gt(out["long_trend"]) & out["short_trend"].shift(1).le(out["long_trend"].shift(1))
    out["cross_down"] = out["short_trend"].lt(out["long_trend"]) & out["short_trend"].shift(1).ge(out["long_trend"].shift(1))
    prior_cross_up = out["cross_up"].shift(1).rolling(window=first_cross_window, min_periods=1).max().fillna(False).astype(bool)
    out["first_cross_up"] = out["cross_up"] & ~prior_cross_up
    return out


def build_prepared(stock_data: Dict[str, pd.DataFrame], first_cross_window: int) -> Dict[str, pd.DataFrame]:
    return {code: prepare_stock(df.reset_index(drop=True), first_cross_window) for code, df in stock_data.items()}


def simulate_signal_trades(prepared: Dict[str, pd.DataFrame], use_three_day_bull_bear_stop: bool = False) -> pd.DataFrame:
    records: List[dict] = []
    for code, df in prepared.items():
        idx = 0
        while idx < len(df) - 1:
            if not bool(df.at[idx, "first_cross_up"]):
                idx += 1
                continue
            entry_idx = idx + 1
            entry_price = float(df.at[entry_idx, "open"])
            if pd.isna(entry_price) or entry_price <= 0:
                idx += 1
                continue

            sell_signal_idx = None
            sell_reason = "cross_down"
            below_long = df["close"] < df["long_trend"]
            stop_watch_active = False
            stop_floor = None
            for j in range(entry_idx, len(df)):
                if use_three_day_bull_bear_stop:
                    if j >= 2:
                        triple_break = bool(below_long.iloc[j] and below_long.iloc[j - 1] and below_long.iloc[j - 2])
                        if triple_break:
                            stop_watch_active = True
                            stop_floor = float(df.loc[j - 2 : j, "close"].min())
                    if stop_watch_active and stop_floor is not None and float(df.at[j, "close"]) < stop_floor:
                        sell_signal_idx = j
                        sell_reason = "three_day_break_stop"
                        break
                if bool(df.at[j, "cross_down"]):
                    sell_signal_idx = j
                    break

            if sell_signal_idx is None:
                exit_idx = len(df) - 1
                exit_price = float(df.at[exit_idx, "close"])
                exit_reason = "end_of_data"
            else:
                if sell_signal_idx + 1 < len(df):
                    exit_idx = sell_signal_idx + 1
                    exit_price = float(df.at[exit_idx, "open"])
                else:
                    exit_idx = sell_signal_idx
                    exit_price = float(df.at[exit_idx, "close"])
                exit_reason = sell_reason

            if pd.isna(exit_price) or exit_price <= 0:
                idx += 1
                continue

            records.append(
                {
                    "code": code,
                    "signal_date": df.at[idx, "date"],
                    "entry_date": df.at[entry_idx, "date"],
                    "exit_date": df.at[exit_idx, "date"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": exit_price / entry_price - 1.0,
                    "holding_days": int(exit_idx - entry_idx + 1),
                    "exit_reason": exit_reason,
                    "success": float(exit_price > entry_price),
                }
            )
            idx = exit_idx + 1
    return pd.DataFrame(records).sort_values(["entry_date", "code"]).reset_index(drop=True) if records else pd.DataFrame()


def _collect_trading_dates(prepared: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    dates = set()
    for df in prepared.values():
        dates.update(pd.to_datetime(df["date"]).tolist())
    return sorted(dates)


def run_portfolio_backtest(
    prepared: Dict[str, pd.DataFrame],
    max_positions: int,
    initial_cash: float = 1_000_000.0,
    use_three_day_bull_bear_stop: bool = False,
) -> dict:
    all_dates = _collect_trading_dates(prepared)
    code_frames = {code: df.set_index("date") for code, df in prepared.items()}
    pending_entries: Dict[pd.Timestamp, List[str]] = {}
    pending_exits: Dict[pd.Timestamp, List[str]] = {}

    for code, df in prepared.items():
        entry_dates = df.loc[df["first_cross_up"], "date"]
        for signal_date in pd.to_datetime(entry_dates):
            loc = df.index[df["date"] == signal_date]
            if len(loc) == 0:
                continue
            idx = int(loc[0])
            if idx + 1 < len(df):
                exec_date = pd.Timestamp(df.at[idx + 1, "date"])
                pending_entries.setdefault(exec_date, []).append(code)

        below_long = df["close"] < df["long_trend"]
        stop_watch_active = False
        stop_floor = None
        for idx in range(len(df)):
            reason = None
            if use_three_day_bull_bear_stop and idx >= 2:
                triple_break = bool(below_long.iloc[idx] and below_long.iloc[idx - 1] and below_long.iloc[idx - 2])
                if triple_break:
                    stop_watch_active = True
                    stop_floor = float(df.loc[idx - 2 : idx, "close"].min())
            if use_three_day_bull_bear_stop and stop_watch_active and stop_floor is not None and float(df.at[idx, "close"]) < stop_floor:
                reason = "three_day_break_stop"
            elif bool(df.at[idx, "cross_down"]):
                reason = "cross_down"
            if reason is None:
                continue
            exec_idx = idx + 1 if idx + 1 < len(df) else idx
            exec_date = pd.Timestamp(df.at[exec_idx, "date"])
            pending_exits.setdefault(exec_date, []).append(f"{code}|{reason}")

    cash = initial_cash
    positions: Dict[str, dict] = {}
    equity_rows: List[dict] = []
    trade_rows: List[dict] = []

    for current_date in all_dates:
        for payload in pending_exits.get(current_date, []):
            code, exit_reason = payload.split("|", 1)
            if code not in positions:
                continue
            frame = code_frames.get(code)
            if frame is None or current_date not in frame.index:
                continue
            open_px = float(frame.at[current_date, "open"])
            if pd.isna(open_px) or open_px <= 0:
                continue
            pos = positions.pop(code)
            proceeds = pos["shares"] * open_px
            cash += proceeds
            trade_rows.append(
                {
                    "code": code,
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "entry_price": pos["entry_price"],
                    "exit_price": open_px,
                    "shares": pos["shares"],
                    "return_pct": open_px / pos["entry_price"] - 1.0,
                    "holding_days": int((pd.Timestamp(current_date) - pd.Timestamp(pos["entry_date"])).days),
                    "success": float(open_px > pos["entry_price"]),
                    "exit_reason": exit_reason,
                }
            )

        slots = max_positions - len(positions)
        today_entries = sorted(set(pending_entries.get(current_date, [])))
        if slots > 0 and today_entries:
            candidates = [code for code in today_entries if code not in positions]
            selected = candidates[:slots]
            allocation = cash / max(len(selected), 1) if selected else 0.0
            for code in selected:
                frame = code_frames.get(code)
                if frame is None or current_date not in frame.index:
                    continue
                open_px = float(frame.at[current_date, "open"])
                if pd.isna(open_px) or open_px <= 0:
                    continue
                shares = math.floor(allocation / open_px)
                if shares <= 0:
                    continue
                cost = shares * open_px
                if cost > cash:
                    continue
                cash -= cost
                positions[code] = {
                    "entry_date": current_date,
                    "entry_price": open_px,
                    "shares": shares,
                    "last_price": open_px,
                }

        market_value = 0.0
        for code, pos in positions.items():
            frame = code_frames.get(code)
            if frame is not None and current_date in frame.index:
                close_px = float(frame.at[current_date, "close"])
                if not pd.isna(close_px) and close_px > 0:
                    pos["last_price"] = close_px
            market_value += pos["shares"] * float(pos["last_price"])
        equity_rows.append({"date": current_date, "equity": cash + market_value, "cash": cash, "positions": len(positions)})

    if all_dates:
        last_date = all_dates[-1]
        for code, pos in list(positions.items()):
            frame = code_frames.get(code)
            if frame is None or last_date not in frame.index:
                continue
            close_px = float(frame.at[last_date, "close"])
            if pd.isna(close_px) or close_px <= 0:
                continue
            trade_rows.append(
                {
                    "code": code,
                    "entry_date": pos["entry_date"],
                    "exit_date": last_date,
                    "entry_price": pos["entry_price"],
                    "exit_price": close_px,
                    "shares": pos["shares"],
                    "return_pct": close_px / pos["entry_price"] - 1.0,
                    "holding_days": int((pd.Timestamp(last_date) - pd.Timestamp(pos["entry_date"])).days),
                    "success": float(close_px > pos["entry_price"]),
                    "exit_reason": "end_of_data",
                }
            )

    trades = pd.DataFrame(trade_rows).sort_values(["entry_date", "code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    equity_curve = pd.DataFrame(equity_rows)
    metrics = compute_metrics(equity_curve["equity"]) if not equity_curve.empty else {}
    trade_metrics = summarize_trade_metrics(trades) if not trades.empty else {}
    return {
        "max_positions": max_positions,
        "trade_count": int(len(trades)),
        "trade_metrics": trade_metrics,
        "equity_metrics": metrics,
        "equity_curve": equity_curve,
        "trades": trades,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend cross portfolio backtest and first-cross window study")
    parser.add_argument("data_dir", nargs="?", default="/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
    parser.add_argument("--output-dir", default="results/trend_cross_portfolio_and_window_study")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_data = load_stock_directory(args.data_dir)

    window_results = []
    portfolio_results = []
    portfolio_windows = {30}

    for window in [10, 20, 30, 40]:
        prepared = build_prepared(stock_data, first_cross_window=window)
        trades = simulate_signal_trades(prepared)
        trade_metrics = summarize_trade_metrics(trades) if not trades.empty else {}
        summary = {
            "first_cross_window": window,
            "trade_count": int(len(trades)),
            "trade_metrics": trade_metrics,
        }
        trades.to_csv(output_dir / f"window_{window}_trades.csv", index=False, encoding="utf-8-sig")
        window_results.append(summary)

        if window in portfolio_windows:
            for max_positions in [1, 3, 5]:
                result = run_portfolio_backtest(prepared, max_positions=max_positions)
                result["equity_curve"].to_csv(
                    output_dir / f"portfolio_window_{window}_maxpos_{max_positions}_equity.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                result["trades"].to_csv(
                    output_dir / f"portfolio_window_{window}_maxpos_{max_positions}_trades.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                portfolio_results.append(
                    {
                        "first_cross_window": window,
                        "max_positions": max_positions,
                        "trade_count": result["trade_count"],
                        "trade_metrics": result["trade_metrics"],
                        "equity_metrics": result["equity_metrics"],
                    }
                )

    stop_prepared = build_prepared(stock_data, first_cross_window=30)
    stop_trades = simulate_signal_trades(stop_prepared, use_three_day_bull_bear_stop=True)
    stop_trade_metrics = summarize_trade_metrics(stop_trades) if not stop_trades.empty else {}
    stop_portfolios = []
    for max_positions in [1, 3, 5]:
        result = run_portfolio_backtest(
            stop_prepared,
            max_positions=max_positions,
            use_three_day_bull_bear_stop=True,
        )
        result["equity_curve"].to_csv(
            output_dir / f"portfolio_window_30_three_day_stop_maxpos_{max_positions}_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )
        result["trades"].to_csv(
            output_dir / f"portfolio_window_30_three_day_stop_maxpos_{max_positions}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        stop_portfolios.append(
            {
                "first_cross_window": 30,
                "max_positions": max_positions,
                "trade_count": result["trade_count"],
                "trade_metrics": result["trade_metrics"],
                "equity_metrics": result["equity_metrics"],
            }
        )
    stop_trades.to_csv(output_dir / "window_30_three_day_stop_trades.csv", index=False, encoding="utf-8-sig")

    summary = {
        "strategy": "首次趋势线上穿多空线买入，首次趋势线下穿多空线卖出",
        "window_study": window_results,
        "portfolio_backtest": portfolio_results,
        "three_day_bull_bear_stop": {
            "rule": "收盘价连续三天跌破多空线后，后续任意一天收盘价低于这三天最低收盘价则触发止损",
            "trade_count": int(len(stop_trades)),
            "trade_metrics": stop_trade_metrics,
            "portfolio_backtest": stop_portfolios,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
