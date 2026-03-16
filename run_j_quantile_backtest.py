from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from core.data_loader import load_price_directory
from core.market_rules import is_limit_down, is_limit_up
from core.metrics import compute_metrics
from strategies.common import base_prepare


@dataclass
class JQuantileConfig:
    initial_capital: float = 10_000_000.0
    max_positions: int = 20
    single_position_cap_ratio: float = 0.10
    quantile_window: int = 30
    buy_quantile: float = 0.10
    sell_quantile: float = 0.90
    win_rate_window: int = 15
    half_position_threshold: float = 0.50
    half_position_ratio: float = 0.50
    stop_loss_multiplier: float = 0.98
    commission_rate: float = 0.0003
    slippage_rate: float = 0.001
    stamp_duty_rate: float = 0.001
    min_lot: int = 100
    market_index_path: Optional[str] = None
    market_ma_window: int = 20
    market_slope_window: int = 5
    market_long_ma_window: Optional[int] = None


@dataclass
class PositionState:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    signal_date: pd.Timestamp
    signal_low: float
    last_close: float


def load_index_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python", skiprows=1, header=None, encoding="utf-8")
    if df.shape[1] < 6:
        raise ValueError(f"Index file has insufficient columns: {path}")
    cols = ["date", "open", "high", "low", "close", "volume", "amount"][: df.shape[1]]
    df = df.iloc[:, : len(cols)]
    df.columns = cols
    if "amount" not in df.columns:
        df["amount"] = 0.0
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    return df.set_index("date")


def build_market_filter(index_df: pd.DataFrame, cfg: JQuantileConfig, all_dates: List[pd.Timestamp]) -> pd.Series:
    out = index_df.copy()
    out["ma_short"] = out["close"].rolling(cfg.market_ma_window, min_periods=cfg.market_ma_window).mean()
    out["slope_up"] = out["ma_short"] > out["ma_short"].shift(cfg.market_slope_window)
    condition = out["slope_up"].fillna(False)
    if cfg.market_long_ma_window is not None:
        out["ma_long"] = out["close"].rolling(cfg.market_long_ma_window, min_periods=cfg.market_long_ma_window).mean()
        condition = condition & out["ma_short"].gt(out["ma_long"]).fillna(False)
    condition = condition.reindex(pd.DatetimeIndex(all_dates)).fillna(False)
    return condition


def prepare_stock(df: pd.DataFrame, cfg: JQuantileConfig) -> pd.DataFrame:
    out = base_prepare(df.copy())
    out["buy_threshold"] = out["J"].shift(1).rolling(cfg.quantile_window, min_periods=cfg.quantile_window).quantile(cfg.buy_quantile)
    out["sell_threshold"] = out["J"].shift(1).rolling(cfg.quantile_window, min_periods=cfg.quantile_window).quantile(cfg.sell_quantile)
    out["buy_signal"] = out["J"].lt(out["buy_threshold"]) & out["short_trend"].gt(out["long_trend"])
    out["sell_signal"] = out["J"].gt(out["sell_threshold"])
    return out


def build_prepared(stock_data: Dict[str, pd.DataFrame], cfg: JQuantileConfig) -> Dict[str, pd.DataFrame]:
    return {code: prepare_stock(df, cfg) for code, df in stock_data.items()}


def win_rate_factor(
    closed_trades: List[dict],
    current_date: pd.Timestamp,
    cfg: JQuantileConfig,
    date_to_idx: Dict[pd.Timestamp, int],
    all_dates: List[pd.Timestamp],
) -> float:
    if not closed_trades:
        return 1.0
    current_idx = date_to_idx.get(current_date)
    if current_idx is None:
        return 1.0
    window_start_idx = max(0, current_idx - cfg.win_rate_window)
    start_date = all_dates[window_start_idx]
    eligible = [
        trade
        for trade in closed_trades
        if start_date <= trade["exit_date"] < current_date
    ]
    if not eligible:
        return 1.0
    win_rate = sum(1 for trade in eligible if trade["return_pct"] > 0) / len(eligible)
    return 1.0 if win_rate >= cfg.half_position_threshold else cfg.half_position_ratio


def summarize_signals(prepared: Dict[str, pd.DataFrame]) -> List[dict]:
    rows: List[dict] = []
    for code, df in prepared.items():
        for dt, row in df.loc[df["buy_signal"].fillna(False)].iterrows():
            rows.append(
                {
                    "signal_date": pd.Timestamp(dt),
                    "code": code,
                    "J": float(row["J"]),
                    "buy_threshold": float(row["buy_threshold"]),
                    "sell_threshold": float(row["sell_threshold"]) if pd.notna(row["sell_threshold"]) else None,
                    "short_trend": float(row["short_trend"]),
                    "long_trend": float(row["long_trend"]),
                    "signal_low": float(row["low"]),
                }
            )
    return sorted(rows, key=lambda item: (item["signal_date"], item["code"]))


def run_backtest_with_prepared(
    prepared: Dict[str, pd.DataFrame],
    all_dates: List[pd.Timestamp],
    cfg: JQuantileConfig,
    flat_signals: Optional[List[dict]] = None,
    market_filter: Optional[pd.Series] = None,
) -> dict:
    if flat_signals is None:
        flat_signals = summarize_signals(prepared)
    date_to_idx = {dt: idx for idx, dt in enumerate(all_dates)}
    pending_entries: Dict[pd.Timestamp, List[dict]] = {}
    pending_exits: Dict[pd.Timestamp, Dict[str, str]] = {}
    rejected_entry_no_next_day = 0

    for signal in flat_signals:
        idx = date_to_idx.get(signal["signal_date"])
        if idx is None or idx + 1 >= len(all_dates):
            rejected_entry_no_next_day += 1
            continue
        exec_date = all_dates[idx + 1]
        pending_entries.setdefault(exec_date, []).append(signal)

    cash = float(cfg.initial_capital)
    positions: Dict[str, PositionState] = {}
    trades: List[dict] = []
    equity_rows: List[dict] = []
    diagnostics = {
        "signal_count": float(len(flat_signals)),
        "rejected_entry_no_next_day": float(rejected_entry_no_next_day),
        "blocked_suspended_buys": 0.0,
        "blocked_suspended_sells": 0.0,
        "rejected_buys_limit_up": 0.0,
        "rejected_buys_cash_or_lot": 0.0,
        "rejected_buys_market_filter": 0.0,
        "rejected_sells_limit_down": 0.0,
        "filled_buys": 0.0,
        "filled_sells": 0.0,
    }

    for current_date in all_dates:
        next_positions: Dict[str, PositionState] = {}
        for code, position in positions.items():
            df = prepared[code]
            if current_date not in df.index:
                next_positions[code] = position
                continue

            row = df.loc[current_date]
            position.last_close = float(row["close"])
            if bool(row.get("is_suspended", False)):
                diagnostics["blocked_suspended_sells"] += 1
                next_positions[code] = position
                continue

            scheduled_reason = pending_exits.get(current_date, {}).get(code)
            if scheduled_reason is not None:
                prev_close = float(row.get("prev_close", row["close"]))
                limit_pct = float(row.get("limit_pct", 0.10))
                if is_limit_down(float(row["open"]), prev_close, limit_pct):
                    diagnostics["rejected_sells_limit_down"] += 1
                    idx = date_to_idx.get(current_date)
                    if idx is not None and idx + 1 < len(all_dates):
                        pending_exits.setdefault(all_dates[idx + 1], {})[code] = scheduled_reason
                    next_positions[code] = position
                    continue

                sell_price = float(row["open"]) * (1.0 - cfg.slippage_rate)
                gross_cash = position.shares * sell_price
                fee = gross_cash * cfg.commission_rate
                tax = gross_cash * cfg.stamp_duty_rate
                cash += gross_cash - fee - tax
                pnl = (sell_price - position.entry_price) * position.shares - fee - tax
                trade = {
                    "code": code,
                    "signal_date": position.signal_date,
                    "entry_date": position.entry_date,
                    "exit_date": current_date,
                    "entry_price": position.entry_price,
                    "exit_price": sell_price,
                    "shares": position.shares,
                    "pnl": pnl,
                    "return_pct": (sell_price - position.entry_price) / position.entry_price,
                    "reason": scheduled_reason,
                    "signal_low": position.signal_low,
                    "stop_price": position.stop_price,
                }
                trades.append(trade)
                diagnostics["filled_sells"] += 1
                continue

            exit_reason = None
            if float(row["close"]) < position.stop_price:
                exit_reason = "stop_loss"
            elif bool(row["sell_signal"]):
                exit_reason = "sell_signal"

            if exit_reason is not None:
                idx = date_to_idx.get(current_date)
                if idx is not None and idx + 1 < len(all_dates):
                    pending_exits.setdefault(all_dates[idx + 1], {})[code] = exit_reason
            next_positions[code] = position

        positions = next_positions

        available_slots = max(cfg.max_positions - len(positions), 0)
        raw_candidates = pending_entries.get(current_date, [])
        executable_candidates: List[dict] = []
        market_buy_allowed = True if market_filter is None else bool(market_filter.get(current_date, False))
        for signal in sorted(raw_candidates, key=lambda item: (item["J"], item["code"])):
            if available_slots <= 0:
                break
            if signal["code"] in positions or any(item["code"] == signal["code"] for item in executable_candidates):
                continue
            if not market_buy_allowed:
                diagnostics["rejected_buys_market_filter"] += 1
                continue

            df = prepared[signal["code"]]
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            if bool(row.get("is_suspended", False)):
                diagnostics["blocked_suspended_buys"] += 1
                continue
            prev_close = float(row.get("prev_close", row["close"]))
            limit_pct = float(row.get("limit_pct", 0.10))
            if is_limit_up(float(row["open"]), prev_close, limit_pct):
                diagnostics["rejected_buys_limit_up"] += 1
                continue
            executable_candidates.append(signal)
            available_slots -= 1

        if executable_candidates:
            equity = cash
            for code, position in positions.items():
                df = prepared[code]
                if current_date in df.index:
                    equity += position.shares * float(df.loc[current_date, "close"])
                else:
                    equity += position.shares * position.last_close

            risk_factor = win_rate_factor(trades, current_date, cfg, date_to_idx, all_dates)
            buy_budget = cash * risk_factor
            candidate_count = len(executable_candidates)
            single_cap = equity * cfg.single_position_cap_ratio

            for signal in executable_candidates:
                df = prepared[signal["code"]]
                row = df.loc[current_date]
                entry_price = float(row["open"]) * (1.0 + cfg.slippage_rate)
                target_capital = min(single_cap, buy_budget / candidate_count)
                shares = int(target_capital / entry_price / cfg.min_lot) * cfg.min_lot
                if shares <= 0:
                    diagnostics["rejected_buys_cash_or_lot"] += 1
                    continue
                gross_cost = shares * entry_price
                fee = gross_cost * cfg.commission_rate
                total_cost = gross_cost + fee
                if total_cost > cash:
                    diagnostics["rejected_buys_cash_or_lot"] += 1
                    continue

                cash -= total_cost
                positions[signal["code"]] = PositionState(
                    code=signal["code"],
                    entry_date=current_date,
                    entry_price=entry_price,
                    shares=shares,
                    stop_price=float(signal["signal_low"]) * cfg.stop_loss_multiplier,
                    signal_date=signal["signal_date"],
                    signal_low=float(signal["signal_low"]),
                    last_close=float(row["close"]),
                )
                diagnostics["filled_buys"] += 1

        equity_value = cash
        for code, position in positions.items():
            df = prepared[code]
            if current_date in df.index:
                mark_price = float(df.loc[current_date, "close"])
                position.last_close = mark_price
            else:
                mark_price = position.last_close
            equity_value += position.shares * mark_price
        equity_rows.append({"date": current_date, "equity": equity_value, "cash": cash, "positions": len(positions)})

    if all_dates:
        last_date = all_dates[-1]
        for code, position in list(positions.items()):
            close_price = position.last_close
            pnl = (close_price - position.entry_price) * position.shares
            trades.append(
                {
                    "code": code,
                    "signal_date": position.signal_date,
                    "entry_date": position.entry_date,
                    "exit_date": last_date,
                    "entry_price": position.entry_price,
                    "exit_price": close_price,
                    "shares": position.shares,
                    "pnl": pnl,
                    "return_pct": (close_price - position.entry_price) / position.entry_price,
                    "reason": "end_of_data",
                    "signal_low": position.signal_low,
                    "stop_price": position.stop_price,
                }
            )

    trades_df = pd.DataFrame(trades).sort_values(["entry_date", "code"]).reset_index(drop=True) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_rows)
    equity_curve = pd.Series(dtype=float) if equity_df.empty else pd.Series(equity_df["equity"].to_numpy(), index=pd.DatetimeIndex(equity_df["date"]), dtype=float)
    metrics = compute_metrics(equity_curve)
    if not trades_df.empty:
        metrics["win_rate"] = float((trades_df["return_pct"] > 0).mean())
        metrics["avg_trade_return"] = float(trades_df["return_pct"].mean())
    else:
        metrics["win_rate"] = 0.0
        metrics["avg_trade_return"] = 0.0

    diagnostics["open_positions_end"] = float(len(positions))
    diagnostics["trade_count"] = float(len(trades_df))
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "trades": trades_df,
        "equity_curve": equity_df,
        "signals": pd.DataFrame(flat_signals),
        "config": cfg,
    }


def run_backtest(stock_data: Dict[str, pd.DataFrame], all_dates: List[pd.Timestamp], cfg: JQuantileConfig) -> dict:
    prepared = build_prepared(stock_data, cfg)
    flat_signals = summarize_signals(prepared)
    market_filter = None
    if cfg.market_index_path:
        market_filter = build_market_filter(load_index_data(cfg.market_index_path), cfg, all_dates)
    return run_backtest_with_prepared(prepared, all_dates, cfg, flat_signals=flat_signals, market_filter=market_filter)


def write_trades(path: Path, trades: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if trades.empty:
        trades.to_csv(path, index=False, encoding="utf-8")
        return
    trades = trades.copy()
    for column in ["signal_date", "entry_date", "exit_date"]:
        trades[column] = pd.to_datetime(trades[column]).dt.strftime("%Y-%m-%d")
    trades.to_csv(path, index=False, encoding="utf-8")


def write_signals(path: Path, signals: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if signals.empty:
        signals.to_csv(path, index=False, encoding="utf-8")
        return
    out = signals.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8")


def write_equity(path: Path, equity_curve: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = equity_curve.copy()
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="J quantile trend backtest")
    parser.add_argument("data_dir")
    parser.add_argument("--output", default="results/j_quantile_backtest_summary.json")
    parser.add_argument("--trades-output", default="")
    parser.add_argument("--signals-output", default="")
    parser.add_argument("--equity-output", default="")
    parser.add_argument("--initial-capital", type=float, default=10_000_000)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--stop-loss-multiplier", type=float, default=0.98)
    parser.add_argument("--market-index-path", default="")
    parser.add_argument("--market-ma-window", type=int, default=20)
    parser.add_argument("--market-slope-window", type=int, default=5)
    parser.add_argument("--market-long-ma-window", type=int, default=0)
    args = parser.parse_args()

    cfg = JQuantileConfig(
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        stop_loss_multiplier=args.stop_loss_multiplier,
        market_index_path=args.market_index_path or None,
        market_ma_window=args.market_ma_window,
        market_slope_window=args.market_slope_window,
        market_long_ma_window=args.market_long_ma_window or None,
    )
    stock_data, all_dates = load_price_directory(args.data_dir)
    result = run_backtest(stock_data, all_dates, cfg)
    payload = {
        "strategy": "J_QUANTILE_TREND",
        "metrics": result["metrics"],
        "diagnostics": result["diagnostics"],
        "config": {
            "initial_capital": cfg.initial_capital,
            "max_positions": cfg.max_positions,
            "single_position_cap_ratio": cfg.single_position_cap_ratio,
            "quantile_window": cfg.quantile_window,
            "buy_quantile": cfg.buy_quantile,
            "sell_quantile": cfg.sell_quantile,
            "win_rate_window": cfg.win_rate_window,
            "half_position_threshold": cfg.half_position_threshold,
            "half_position_ratio": cfg.half_position_ratio,
            "stop_loss_multiplier": cfg.stop_loss_multiplier,
            "commission_rate": cfg.commission_rate,
            "slippage_rate": cfg.slippage_rate,
            "stamp_duty_rate": cfg.stamp_duty_rate,
            "min_lot": cfg.min_lot,
            "market_index_path": cfg.market_index_path,
            "market_ma_window": cfg.market_ma_window,
            "market_slope_window": cfg.market_slope_window,
            "market_long_ma_window": cfg.market_long_ma_window,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.trades_output:
        write_trades(Path(args.trades_output), result["trades"])
    if args.signals_output:
        write_signals(Path(args.signals_output), result["signals"])
    if args.equity_output:
        write_equity(Path(args.equity_output), result["equity_curve"])

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
