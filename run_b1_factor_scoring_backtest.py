from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.data_loader import load_price_directory
from core.engine import BacktestEngine, EngineConfig
from core.market_rules import is_limit_down, is_limit_up
from core.metrics import compute_metrics
from strategies.common import calc_trend_cn
from strategies.unified_strategies import B1Strategy
from utils.multi_factor_research.research_metrics import summarize_trade_metrics


DISASTER_START = pd.Timestamp("2015-06-12")
DISASTER_END = pd.Timestamp("2024-09-18")
MIN_FACTOR_SAMPLES = 200


@dataclass
class WeightedB1Config:
    initial_capital: float = 1_000_000.0
    max_positions: int = 10
    single_position_cap_ratio: float = 0.10
    stop_loss_multiplier: float = 0.95
    win_rate_window: int = 15
    half_position_threshold: float = 0.50
    half_position_ratio: float = 0.50
    commission_rate: float = 0.0003
    slippage_rate: float = 0.001
    stamp_duty_rate: float = 0.001
    min_lot: int = 100


FACTOR_COLUMNS = [
    "shrink_volume_factor",
    "expand_volume_factor",
    "board_range_shrink_factor",
    "pullback_trend_confirm_factor",
    "pullback_bull_bear_confirm_factor",
    "long_bear_short_volume_factor",
    "j_turn_up_factor",
    "daily_bull_factor",
    "weekly_bull_factor",
    "j_n_rise_factor",
]

FACTOR_NAME_MAP = {
    "shrink_volume_factor": "缩量",
    "expand_volume_factor": "放量",
    "board_range_shrink_factor": "板块差异化涨幅区间且缩量",
    "pullback_trend_confirm_factor": "回踩趋势线确认",
    "pullback_bull_bear_confirm_factor": "回踩多空线确认",
    "long_bear_short_volume_factor": "长阴短柱",
    "j_turn_up_factor": "J值拐头确认",
    "daily_bull_factor": "日线多头",
    "weekly_bull_factor": "周线多头",
    "j_n_rise_factor": "J值N型上升",
}


def _board_range(change_pct: pd.Series, limit_pct: pd.Series) -> pd.Series:
    multiplier = limit_pct / 0.10
    lower = -0.035 * multiplier
    upper = 0.02 * multiplier
    return change_pct.between(lower, upper, inclusive="both")


def _calc_weekly_bull(df: pd.DataFrame) -> pd.Series:
    weekly = (
        df.reset_index()
        .groupby(pd.Grouper(key="date", freq="W-FRI"))
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .dropna(subset=["open", "close"])
    )
    short, long_line = calc_trend_cn(weekly["close"])
    weekly = weekly.assign(weekly_short=short, weekly_long=long_line).reset_index()
    daily = df.reset_index()[["date"]].sort_values("date")
    merged = pd.merge_asof(daily, weekly[["date", "weekly_short", "weekly_long"]], on="date", direction="backward")
    return merged["weekly_short"].gt(merged["weekly_long"]).fillna(False).to_numpy()


def _calc_j_n_rise(df: pd.DataFrame, lookback: int = 20, j_tol: float = 5.0, min_gap: int = 3) -> pd.Series:
    j_vals = df["J"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=bool)
    for idx in range(len(df)):
        start = max(0, idx - lookback)
        current_j = j_vals[idx]
        current_close = closes[idx]
        if not np.isfinite(current_j) or not np.isfinite(current_close):
            continue
        for prev in range(start, idx - min_gap + 1):
            if not np.isfinite(j_vals[prev]) or not np.isfinite(closes[prev]):
                continue
            if abs(current_j - j_vals[prev]) <= j_tol and current_close > closes[prev]:
                out[idx] = True
                break
    return pd.Series(out, index=df.index)


def prepare_with_factors(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    strategy = B1Strategy()
    prepared = strategy.prepare(stock_data)
    for code, df in prepared.items():
        out = df.copy()
        prev_close = out["close"].shift(1)
        prev_volume = out["volume"].shift(1)
        daily_change = out["close"] / prev_close - 1.0
        bullish = out["close"] > out["open"]
        bearish = out["close"] < out["open"]
        shrink_volume = out["volume"] < prev_volume
        expand_volume = out["volume"] > prev_volume

        recent_abs_change = daily_change.shift(1).abs().rolling(5, min_periods=5).max()
        near_trend_yesterday = (out["low"].shift(1) / out["short_trend"].shift(1) - 1.0).abs() <= 0.01
        near_bull_bear_yesterday = (out["low"].shift(1) / out["long_trend"].shift(1) - 1.0).abs() <= 0.01

        out["shrink_volume_factor"] = shrink_volume.fillna(False)
        out["expand_volume_factor"] = expand_volume.fillna(False)
        out["board_range_shrink_factor"] = (_board_range(daily_change, out["limit_pct"]) & shrink_volume).fillna(False)
        out["pullback_trend_confirm_factor"] = (
            near_trend_yesterday
            & bullish
            & shrink_volume
            & out["close"].ge(out["short_trend"])
        ).fillna(False)
        out["pullback_bull_bear_confirm_factor"] = (
            near_bull_bear_yesterday
            & bullish
            & shrink_volume
            & out["close"].ge(out["long_trend"])
        ).fillna(False)
        out["long_bear_short_volume_factor"] = (
            bearish
            & daily_change.lt(0)
            & daily_change.abs().gt(recent_abs_change)
            & shrink_volume
        ).fillna(False)
        out["j_turn_up_factor"] = (
            out["J"].gt(out["J"].shift(1))
            & out["J"].shift(1).le(out["J"].shift(2))
        ).fillna(False)
        out["daily_bull_factor"] = out["short_trend"].gt(out["long_trend"]).fillna(False)
        out["weekly_bull_factor"] = _calc_weekly_bull(out)
        out["j_n_rise_factor"] = _calc_j_n_rise(out).fillna(False)
        prepared[code] = out
    return prepared


def build_signal_records(prepared: Dict[str, pd.DataFrame], all_dates: List[pd.Timestamp]) -> pd.DataFrame:
    rows: List[dict] = []
    date_to_idx = {dt: idx for idx, dt in enumerate(all_dates)}
    for code, df in prepared.items():
        signal_idx = np.flatnonzero(df["buy_signal"].fillna(False).to_numpy())
        for idx in signal_idx:
            signal_date = df.index[idx]
            date_idx = date_to_idx.get(signal_date)
            if date_idx is None or date_idx + 1 >= len(all_dates):
                continue
            entry_date = all_dates[date_idx + 1]
            signal_row = df.iloc[idx]
            rows.append(
                {
                    "code": code,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "signal_low": float(signal_row["low"]),
                    **{factor: bool(signal_row[factor]) for factor in FACTOR_COLUMNS},
                }
            )
    return pd.DataFrame(rows).sort_values(["entry_date", "code"]).reset_index(drop=True) if rows else pd.DataFrame()


def run_base_b1_portfolio(stock_data: Dict[str, pd.DataFrame], all_dates: List[pd.Timestamp]) -> pd.DataFrame:
    engine = BacktestEngine(EngineConfig(initial_capital=1_000_000, max_positions=10))
    result = engine.run(B1Strategy(), stock_data, all_dates)
    rows = [
        {
            "code": trade.code,
            "entry_date": trade.entry_date,
            "exit_date": trade.exit_date,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "shares": trade.shares,
            "pnl": trade.pnl,
            "return_pct": trade.return_pct,
            "exit_reason": trade.reason,
        }
        for trade in result.trades
    ]
    return pd.DataFrame(rows).sort_values(["entry_date", "code"]).reset_index(drop=True) if rows else pd.DataFrame()


def audit_factors(signal_trades: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
    base_metrics = summarize_trade_metrics(signal_trades)
    outside = signal_trades[
        ((signal_trades["entry_date"] < DISASTER_START) & (signal_trades["exit_date"] < DISASTER_START))
        | ((signal_trades["entry_date"] > DISASTER_END) & (signal_trades["exit_date"] > DISASTER_END))
    ].copy()
    base_outside = summarize_trade_metrics(outside)
    rows: List[dict] = []
    for factor in FACTOR_COLUMNS:
        hit = signal_trades[signal_trades[factor]].copy()
        miss = signal_trades[~signal_trades[factor]].copy()
        hit_outside = outside[outside[factor]].copy()
        hit_metrics = summarize_trade_metrics(hit)
        miss_metrics = summarize_trade_metrics(miss)
        hit_outside_metrics = summarize_trade_metrics(hit_outside)
        avg_lift = hit_metrics["avg_return"] - base_metrics["avg_return"]
        win_lift = hit_metrics["positive_return_rate"] - base_metrics["positive_return_rate"]
        outside_lift = hit_outside_metrics["avg_return"] - base_outside["avg_return"]
        positive = (
            hit_metrics["sample_count"] >= MIN_FACTOR_SAMPLES
            and hit_outside_metrics["sample_count"] >= MIN_FACTOR_SAMPLES
            and avg_lift > 0
            and win_lift >= 0
            and outside_lift > 0
        )
        strength = max(avg_lift, 0.0) * 100.0 + max(win_lift, 0.0) * 10.0 + max(outside_lift, 0.0) * 100.0
        rows.append(
            {
                "factor": factor,
                "factor_cn": FACTOR_NAME_MAP[factor],
                "sample_count": hit_metrics["sample_count"],
                "avg_return_hit": hit_metrics["avg_return"],
                "avg_return_base": base_metrics["avg_return"],
                "avg_return_lift": avg_lift,
                "win_rate_hit": hit_metrics["positive_return_rate"],
                "win_rate_base": base_metrics["positive_return_rate"],
                "win_rate_lift": win_lift,
                "outside_sample_count": hit_outside_metrics["sample_count"],
                "outside_avg_return_hit": hit_outside_metrics["avg_return"],
                "outside_avg_return_base": base_outside["avg_return"],
                "outside_avg_return_lift": outside_lift,
                "miss_avg_return": miss_metrics["avg_return"],
                "is_positive": bool(positive),
                "strength": strength,
            }
        )
    result = pd.DataFrame(rows).sort_values(["is_positive", "strength"], ascending=[False, False]).reset_index(drop=True)
    positive = result[result["is_positive"]].copy()
    weights: Dict[str, float] = {}
    if not positive.empty and positive["strength"].sum() > 0:
        total_strength = float(positive["strength"].sum())
        for row in positive.itertuples(index=False):
            weights[row.factor] = float(row.strength / total_strength)
    return result, weights


def _risk_factor(trades: List[dict], current_date: pd.Timestamp, all_dates: List[pd.Timestamp], cfg: WeightedB1Config) -> float:
    date_to_idx = {dt: idx for idx, dt in enumerate(all_dates)}
    current_idx = date_to_idx.get(current_date)
    if current_idx is None:
        return 1.0
    start_idx = max(0, current_idx - cfg.win_rate_window)
    start_date = all_dates[start_idx]
    recent = [trade for trade in trades if start_date <= trade["exit_date"] < current_date]
    if not recent:
        return 1.0
    win_rate = sum(1 for trade in recent if trade["return_pct"] > 0) / len(recent)
    return 1.0 if win_rate >= cfg.half_position_threshold else cfg.half_position_ratio


def run_weighted_portfolio(
    prepared: Dict[str, pd.DataFrame],
    factor_weights: Dict[str, float],
    all_dates: List[pd.Timestamp],
    cfg: WeightedB1Config,
) -> dict:
    signal_records: List[dict] = []
    date_to_idx = {dt: idx for idx, dt in enumerate(all_dates)}
    pending_entries: Dict[pd.Timestamp, List[dict]] = {}
    pending_exits: Dict[pd.Timestamp, Dict[str, str]] = {}

    for code, df in prepared.items():
        for dt in df.index[df["buy_signal"].fillna(False)]:
            idx = date_to_idx.get(dt)
            if idx is None or idx + 1 >= len(all_dates):
                continue
            row = df.loc[dt]
            score_add = sum(weight for factor, weight in factor_weights.items() if bool(row[factor]))
            score = 1.0 + score_add
            record = {
                "signal_date": dt,
                "code": code,
                "score": score,
                "signal_low": float(row["low"]),
                "factors_hit": [FACTOR_NAME_MAP[f] for f in factor_weights if bool(row[f])],
            }
            signal_records.append(record)
            pending_entries.setdefault(all_dates[idx + 1], []).append(record)

    cash = cfg.initial_capital
    positions: Dict[str, dict] = {}
    trades: List[dict] = []
    equity_rows: List[dict] = []

    for current_date in all_dates:
        next_positions: Dict[str, dict] = {}
        for code, pos in positions.items():
            df = prepared[code]
            if current_date not in df.index:
                next_positions[code] = pos
                continue
            row = df.loc[current_date]
            pos["last_close"] = float(row["close"])
            scheduled = pending_exits.get(current_date, {}).get(code)
            if scheduled is not None:
                prev_close = float(row.get("prev_close", row["close"]))
                limit_pct = float(row.get("limit_pct", 0.10))
                if bool(row.get("is_suspended", False)) or is_limit_down(float(row["open"]), prev_close, limit_pct):
                    idx = date_to_idx.get(current_date)
                    if idx is not None and idx + 1 < len(all_dates):
                        pending_exits.setdefault(all_dates[idx + 1], {})[code] = scheduled
                    next_positions[code] = pos
                    continue
                sell_price = float(row["open"]) * (1.0 - cfg.slippage_rate)
                gross = pos["shares"] * sell_price
                fee = gross * cfg.commission_rate
                tax = gross * cfg.stamp_duty_rate
                cash += gross - fee - tax
                trades.append(
                    {
                        "code": code,
                        "entry_date": pos["entry_date"],
                        "exit_date": current_date,
                        "entry_price": pos["entry_price"],
                        "exit_price": sell_price,
                        "shares": pos["shares"],
                        "pnl": (sell_price - pos["entry_price"]) * pos["shares"] - fee - tax,
                        "return_pct": sell_price / pos["entry_price"] - 1.0,
                        "reason": scheduled,
                        "score": pos["score"],
                    }
                )
                continue

            if float(row["close"]) < pos["stop_price"]:
                reason = "stop_loss"
            elif bool(row.get("sell_signal", False)):
                reason = "sell_signal"
            else:
                reason = None
            if reason is not None:
                idx = date_to_idx.get(current_date)
                if idx is not None and idx + 1 < len(all_dates):
                    pending_exits.setdefault(all_dates[idx + 1], {})[code] = reason
            next_positions[code] = pos
        positions = next_positions

        available_slots = max(cfg.max_positions - len(positions), 0)
        existing = set(positions)
        executable: List[tuple[dict, pd.Series]] = []
        for signal in sorted(pending_entries.get(current_date, []), key=lambda item: item["score"], reverse=True):
            if available_slots <= 0:
                break
            if signal["code"] in existing:
                continue
            if current_date not in prepared[signal["code"]].index:
                continue
            row = prepared[signal["code"]].loc[current_date]
            prev_close = float(row.get("prev_close", row["close"]))
            limit_pct = float(row.get("limit_pct", 0.10))
            if bool(row.get("is_suspended", False)) or is_limit_up(float(row["open"]), prev_close, limit_pct):
                continue
            executable.append((signal, row))
            existing.add(signal["code"])
            available_slots -= 1

        if executable:
            equity_before = cash + sum(pos["shares"] * pos["last_close"] for pos in positions.values())
            risk_factor = _risk_factor(trades, current_date, all_dates, cfg)
            buy_budget = cash * risk_factor
            total_scores = sum(signal["score"] for signal, _ in executable)
            for signal, row in executable:
                entry_price = float(row["open"]) * (1.0 + cfg.slippage_rate)
                target_capital = min(
                    equity_before * cfg.single_position_cap_ratio,
                    buy_budget * signal["score"] / total_scores if total_scores > 0 else 0.0,
                )
                shares = int(target_capital / entry_price / cfg.min_lot) * cfg.min_lot
                if shares <= 0:
                    continue
                gross = shares * entry_price
                fee = gross * cfg.commission_rate
                total_cost = gross + fee
                if total_cost > cash:
                    continue
                cash -= total_cost
                positions[signal["code"]] = {
                    "entry_date": current_date,
                    "entry_price": entry_price,
                    "shares": shares,
                    "stop_price": signal["signal_low"] * cfg.stop_loss_multiplier,
                    "last_close": float(row["close"]),
                    "score": signal["score"],
                }

        equity_value = cash
        for code, pos in positions.items():
            if current_date in prepared[code].index:
                pos["last_close"] = float(prepared[code].loc[current_date, "close"])
            equity_value += pos["shares"] * pos["last_close"]
        equity_rows.append({"date": current_date, "equity": equity_value, "cash": cash, "positions": len(positions)})

    if all_dates:
        last_date = all_dates[-1]
        for code, pos in positions.items():
            close_px = pos["last_close"]
            trades.append(
                {
                    "code": code,
                    "entry_date": pos["entry_date"],
                    "exit_date": last_date,
                    "entry_price": pos["entry_price"],
                    "exit_price": close_px,
                    "shares": pos["shares"],
                    "pnl": (close_px - pos["entry_price"]) * pos["shares"],
                    "return_pct": close_px / pos["entry_price"] - 1.0,
                    "reason": "end_of_data",
                    "score": pos["score"],
                }
            )

    trades_df = pd.DataFrame(trades).sort_values(["entry_date", "code"]).reset_index(drop=True) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_rows)
    equity_curve = pd.Series(dtype=float) if equity_df.empty else pd.Series(equity_df["equity"].to_numpy(), index=pd.DatetimeIndex(equity_df["date"]), dtype=float)
    metrics = compute_metrics(equity_curve)
    if not trades_df.empty:
        metrics["win_rate"] = float((trades_df["return_pct"] > 0).mean())
        metrics["avg_trade_return"] = float(trades_df["return_pct"].mean())
    return {
        "metrics": metrics,
        "equity_curve": equity_df,
        "trades": trades_df,
        "signals": pd.DataFrame(signal_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B1 factor audit and weighted capital backtest")
    parser.add_argument("data_dir")
    parser.add_argument("--output-dir", default="results/b1_factor_scoring")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_data, all_dates = load_price_directory(args.data_dir)
    prepared = prepare_with_factors(stock_data)
    signal_records = build_signal_records(prepared, all_dates)
    base_trades = run_base_b1_portfolio(stock_data, all_dates)
    signal_trades = base_trades.merge(signal_records, on=["code", "entry_date"], how="left")
    for factor in FACTOR_COLUMNS:
        signal_trades[factor] = signal_trades[factor].fillna(False).astype(bool)
    signal_trades.to_csv(output_dir / "signal_trade_audit.csv", index=False, encoding="utf-8-sig")

    factor_audit, factor_weights = audit_factors(signal_trades)
    factor_audit.to_csv(output_dir / "factor_audit.csv", index=False, encoding="utf-8-sig")

    cfg = WeightedB1Config()
    portfolio_result = run_weighted_portfolio(prepared, factor_weights, all_dates, cfg)
    portfolio_result["trades"].to_csv(output_dir / "weighted_trades.csv", index=False, encoding="utf-8-sig")
    portfolio_result["equity_curve"].to_csv(output_dir / "weighted_equity.csv", index=False, encoding="utf-8-sig")
    portfolio_result["signals"].to_csv(output_dir / "weighted_signals.csv", index=False, encoding="utf-8-sig")

    payload = {
        "base_signal_trade_metrics": summarize_trade_metrics(signal_trades),
        "positive_factors": factor_audit.loc[factor_audit["is_positive"], ["factor_cn", "factor", "strength"]].to_dict(orient="records"),
        "factor_weights": {FACTOR_NAME_MAP.get(k, k): v for k, v in factor_weights.items()},
        "portfolio_metrics": portfolio_result["metrics"],
        "trade_count": int(len(portfolio_result["trades"])),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
