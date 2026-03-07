from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from core.data_loader import load_price_directory
from core.market_rules import is_limit_down, is_limit_up
from core.metrics import compute_metrics
from utils.multi_factor_research.factor_calculator import FACTOR_COLUMNS, PENALTY_COLUMNS, prepare_factor_frame
from utils.multi_factor_research.weight_optimizer import PENALTY_WEIGHTS

持仓高位钝化扣分 = 0.08


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 1_000_000.0
    max_positions: int = 10
    commission_rate: float = 0.0003
    slippage_rate: float = 0.001
    stamp_duty_rate: float = 0.001
    min_lot: int = 100
    top_quantile: float = 0.30
    replacement_threshold: float = 0.03
    min_hold_days_for_replace: int = 5
    max_daily_replacements: int = 1


def _parse_weight_spec(spec: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for part in spec.split(";"):
        text = part.strip()
        if not text:
            continue
        name, value = text.split("=")
        weights[name.strip()] = float(value.strip())
    return weights


def _load_refined_weights(refined_root: Path) -> dict[str, dict]:
    mapping = {
        "fixed_take_profit": refined_root / "固定涨幅止盈_40pct" / "summary.json",
        "fixed_days": refined_root / "固定持有_30天" / "summary.json",
        "tiered": refined_root / "分批顺序止盈" / "summary.json",
    }
    result: dict[str, dict] = {}
    for key, path in mapping.items():
        payload = json.loads(path.read_text())
        result[key] = payload["best_weighted_combos"]["best_by_score"]
    return result


def _prepare_stock_frames(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    prepared: Dict[str, pd.DataFrame] = {}
    for code, df in stock_data.items():
        raw = df.reset_index().copy()
        factor_df = prepare_factor_frame(raw, burst_window=20)
        factor_df["limit_pct"] = raw["limit_pct"].to_numpy()
        factor_df["is_suspended"] = raw["is_suspended"].to_numpy()
        factor_df = factor_df.set_index("date")
        prepared[code] = factor_df
    return prepared


def _calc_net_score_series(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    factor_score = sum(df[col].fillna(0.0) * weights[col] for col in FACTOR_COLUMNS)
    penalty_score = sum(df[col].fillna(0.0) * PENALTY_WEIGHTS.get(col, 0.0) for col in PENALTY_COLUMNS)
    return factor_score - penalty_score


def _build_signal_map(
    prepared: Dict[str, pd.DataFrame],
    all_dates: List[pd.Timestamp],
    weights: Dict[str, float],
    top_quantile: float,
) -> Dict[pd.Timestamp, List[dict]]:
    rows: List[dict] = []
    date_to_idx = {dt: idx for idx, dt in enumerate(all_dates)}
    for code, df in prepared.items():
        net_score = _calc_net_score_series(df, weights)
        signal_mask = df["J"].lt(13) & df["trend_line"].gt(df["bull_bear_line"])
        for dt in df.index[signal_mask]:
            idx = date_to_idx.get(dt)
            if idx is None or idx + 1 >= len(all_dates):
                continue
            rows.append(
                {
                    "signal_date": dt,
                    "exec_date": all_dates[idx + 1],
                    "code": code,
                    "score": float(net_score.loc[dt]),
                }
            )
    if not rows:
        return {}
    signal_df = pd.DataFrame(rows).sort_values(["signal_date", "score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    cutoff = signal_df["score"].quantile(max(0.0, min(1.0, 1.0 - top_quantile)))
    signal_df = signal_df[signal_df["score"] >= cutoff].copy()
    signal_map: Dict[pd.Timestamp, List[dict]] = {}
    for row in signal_df.itertuples(index=False):
        signal_map.setdefault(row.exec_date, []).append({"code": row.code, "score": float(row.score), "signal_date": row.signal_date})
    for dt in signal_map:
        signal_map[dt].sort(key=lambda item: (-item["score"], item["code"]))
    return signal_map


def _sell_position(cash: float, shares: int, raw_price: float, config: PortfolioConfig) -> tuple[float, float]:
    exec_price = raw_price * (1.0 - config.slippage_rate)
    gross = shares * exec_price
    fee = gross * config.commission_rate
    tax = gross * config.stamp_duty_rate
    return cash + gross - fee - tax, exec_price


def _buy_shares(cash: float, entry_price: float, available_slots: int, config: PortfolioConfig) -> int:
    if available_slots <= 0 or entry_price <= 0:
        return 0
    allocation = cash / available_slots
    shares = int(allocation / entry_price / config.min_lot) * config.min_lot
    if shares <= 0:
        return 0
    total_cost = shares * entry_price * (1.0 + config.commission_rate)
    return shares if total_cost <= cash else 0


def _buy_shares_for_value(cash: float, entry_price: float, target_value: float, config: PortfolioConfig) -> int:
    if entry_price <= 0 or target_value <= 0:
        return 0
    value = min(cash, target_value)
    shares = int(value / entry_price / config.min_lot) * config.min_lot
    if shares <= 0:
        return 0
    total_cost = shares * entry_price * (1.0 + config.commission_rate)
    return shares if total_cost <= cash else 0


def _position_current_score(code: str, current_date: pd.Timestamp, prepared: Dict[str, pd.DataFrame], weights: Dict[str, float], fallback: float) -> float:
    df = prepared[code]
    if current_date not in df.index:
        return fallback
    row = df.loc[current_date]
    reward = sum(float(row[col]) * weights[col] for col in FACTOR_COLUMNS)
    penalty = sum(float(row[col]) * PENALTY_WEIGHTS.get(col, 0.0) for col in PENALTY_COLUMNS)
    return reward - penalty


def _position_dynamic_penalty(position: dict, row: pd.Series) -> float:
    penalty = 0.0
    if float(row.get("J", 0.0)) > 80 and float(row.get("close", 0.0)) <= float(position["entry_price"]):
        penalty += 持仓高位钝化扣分
    return penalty


def _position_replace_eligible(label: str, position: dict, df: pd.DataFrame, idx_now: int, idx_entry: int, hold_bars: int) -> bool:
    if hold_bars < position.get("最短换仓持有天数", 5):
        return False
    if label in {"fixed_take_profit", "fixed_days"}:
        check_bars = min(5, hold_bars)
        early_window = df.iloc[idx_entry:idx_entry + check_bars]
        if early_window.empty:
            return False
        early_high = float(early_window["high"].max())
        if early_high >= position["entry_price"] * 1.02:
            return False
        return hold_bars >= 5
    if label == "tiered" and not position.get("step1", False):
        return hold_bars >= 5
    return True


def _can_add_tranche(row: pd.Series) -> bool:
    if bool(row.get("is_suspended", False)):
        return False
    if float(row.get("trend_line", 0.0)) <= float(row.get("bull_bear_line", 0.0)):
        return False
    key_close = row.get("关键K收盘价")
    if pd.notna(key_close) and float(row["close"]) < float(key_close):
        return False
    return float(row["close"]) >= float(row["trend_line"])


def _close_position(
    cash: float,
    positions: Dict[str, dict],
    trades: List[dict],
    code: str,
    current_date: pd.Timestamp,
    raw_price: float,
    reason: str,
    config: PortfolioConfig,
) -> float:
    pos = positions[code]
    cash, exec_price = _sell_position(cash, pos["shares"], raw_price, config)
    trades.append(
        {
            "code": code,
            "entry_date": pos["entry_date"],
            "exit_date": current_date,
            "entry_price": pos["entry_price"],
            "exit_price": exec_price,
            "shares": pos["shares"],
            "reason": reason,
        }
    )
    del positions[code]
    return cash


def _can_open_position(row: pd.Series) -> bool:
    if bool(row.get("is_suspended", False)):
        return False
    if float(row.get("bull_bear_line", 0.0)) > float(row.get("trend_line", 0.0)):
        return False
    prev_close = float(row.get("prev_close", row["close"]))
    return not is_limit_up(float(row["open"]), prev_close, float(row.get("limit_pct", 0.10)))


def _can_close_position(row: pd.Series) -> bool:
    if bool(row.get("is_suspended", False)):
        return False
    prev_close = float(row.get("prev_close", row["close"]))
    return not is_limit_down(float(row["open"]), prev_close, float(row.get("limit_pct", 0.10)))


def _run_model(
    label: str,
    prepared: Dict[str, pd.DataFrame],
    all_dates: List[pd.Timestamp],
    weights: Dict[str, float],
    config: PortfolioConfig,
) -> dict:
    signal_map = _build_signal_map(prepared, all_dates, weights, config.top_quantile)
    cash = float(config.initial_capital)
    positions: Dict[str, dict] = {}
    trades: List[dict] = []
    equity_points: List[float] = []
    equity_index: List[pd.Timestamp] = []

    for current_date in all_dates:
        opened_today: set[str] = set()
        daily_replacements = 0

        for code in list(positions.keys()):
            pos = positions[code]
            df = prepared[code]
            if current_date not in df.index or code in opened_today:
                continue
            row = df.loc[current_date]
            if bool(row.get("is_suspended", False)):
                continue
            idx_now = df.index.get_loc(current_date)
            idx_entry = df.index.get_loc(pos["entry_date"])
            hold_bars = idx_now - idx_entry + 1

            if label == "fixed_take_profit":
                if float(row["bull_bear_line"]) > float(row["trend_line"]):
                    if float(row["close"]) < float(row["bull_bear_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "多空线压制卖出", config)
                        continue
                    if float(row["close"]) < float(row["trend_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势线破位卖出", config)
                        continue
                if float(row["low"]) <= pos["stop_price"]:
                    cash = _close_position(cash, positions, trades, code, current_date, pos["stop_price"], "止损卖出", config)
                    continue
                tp_price = pos["entry_price"] * 1.40
                if float(row["high"]) >= tp_price:
                    cash = _close_position(cash, positions, trades, code, current_date, tp_price, "固定涨幅止盈", config)
                    continue
                if hold_bars >= 30:
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "持有到期卖出", config)
                    continue

            elif label == "fixed_days":
                if float(row["bull_bear_line"]) > float(row["trend_line"]):
                    if float(row["close"]) < float(row["bull_bear_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "多空线压制卖出", config)
                        continue
                    if float(row["close"]) < float(row["trend_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势线破位卖出", config)
                        continue
                if float(row["low"]) <= pos["stop_price"]:
                    cash = _close_position(cash, positions, trades, code, current_date, pos["stop_price"], "止损卖出", config)
                    continue
                if hold_bars >= 30:
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "固定天数卖出", config)
                    continue

            elif label == "tiered":
                if float(row["bull_bear_line"]) > float(row["trend_line"]):
                    if float(row["close"]) < float(row["bull_bear_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "多空线压制卖出", config)
                        continue
                    if float(row["close"]) < float(row["trend_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势线破位卖出", config)
                        continue
                if float(row["low"]) <= pos["stop_price"]:
                    cash = _close_position(cash, positions, trades, code, current_date, pos["stop_price"], "止损卖出", config)
                    continue

                base_chunk = int((pos["initial_shares"] * 0.2) / config.min_lot) * config.min_lot
                prev_low = float(df.iloc[idx_now - 1]["low"]) if idx_now > 0 else float(row["low"])
                trend_line = float(row["trend_line"])
                j_value = float(row["J"])

                def sell_chunk(raw_price: float, reason: str, sell_all: bool = False) -> bool:
                    nonlocal cash
                    shares_to_sell = pos["shares"] if sell_all else min(pos["shares"], max(base_chunk, config.min_lot))
                    if shares_to_sell <= 0:
                        return False
                    cash, exec_price_local = _sell_position(cash, shares_to_sell, raw_price, config)
                    trades.append({"code": code, "entry_date": pos["entry_date"], "exit_date": current_date, "entry_price": pos["entry_price"], "exit_price": exec_price_local, "shares": shares_to_sell, "reason": reason})
                    pos["shares"] -= shares_to_sell
                    return True

                take_profit_stop = float(row["low"]) * 0.95

                if not pos["step1"] and j_value > 100:
                    if sell_chunk(float(row["close"]), "第一段止盈"):
                        pos["step1"] = True
                        pos["stop_price"] = max(pos["stop_price"], take_profit_stop)

                if code not in positions or pos["shares"] <= 0:
                    if code in positions and pos["shares"] <= 0:
                        del positions[code]
                    continue

                if not pos["step2"] and float(row["high"]) >= pos["entry_price"] * 1.08:
                    if sell_chunk(pos["entry_price"] * 1.08, "第二段止盈"):
                        pos["step2"] = True
                        pos["stop_price"] = max(pos["stop_price"], take_profit_stop)

                if code not in positions or pos["shares"] <= 0:
                    if code in positions and pos["shares"] <= 0:
                        del positions[code]
                    continue

                if pos["step1"] and pos["step2"] and not pos["step3"] and float(row["high"]) > trend_line * 1.15:
                    if sell_chunk(float(row["close"]), "第三段止盈"):
                        pos["step3"] = True
                        pos["stop_price"] = max(pos["stop_price"], take_profit_stop)
                elif pos["step1"] and pos["step2"] and pos["step3"] and not pos["step4"] and float(row["high"]) > trend_line * 1.20:
                    if sell_chunk(float(row["close"]), "第四段止盈"):
                        pos["step4"] = True
                        pos["stop_price"] = max(pos["stop_price"], take_profit_stop)
                elif pos["step1"] and pos["step2"] and pos["step4"] and not pos["step5"] and float(row["high"]) > trend_line * 1.25:
                    if sell_chunk(float(row["close"]), "第五段止盈", sell_all=True):
                        pos["step5"] = True

                if code not in positions:
                    continue
                if pos["shares"] <= 0:
                    del positions[code]
                    continue

                if pos["step2"]:
                    pos["stop_price"] = max(pos["stop_price"], trend_line * 0.98)
                    pos["dd_count"] = pos["dd_count"] + 1 if float(row["close"]) < prev_low else 0
                    if pos["dd_count"] >= 3:
                        sell_chunk(float(row["close"]), "滴滴止损", sell_all=True)
                        del positions[code]
                        continue

                if hold_bars >= 60:
                    sell_chunk(float(row["close"]), "持有60天卖出", sell_all=True)
                    del positions[code]
                    continue

            if code not in positions:
                continue
            if pos.get("买入批次", 1) < 3 and _can_add_tranche(row):
                next_batch = pos["买入批次"] + 1
                tranche_triggered = (
                    (next_batch == 2 and float(row["J"]) < 0)
                    or (next_batch == 3 and float(row["J"]) < -5)
                )
                if tranche_triggered:
                    tranche_ratio = 0.3 if next_batch == 2 else 0.4
                    add_price = float(row["open"]) * (1.0 + config.slippage_rate)
                    add_target_value = pos["计划仓位金额"] * tranche_ratio
                    add_shares = _buy_shares_for_value(cash, add_price, add_target_value, config)
                    if add_shares > 0:
                        gross = add_shares * add_price
                        fee = gross * config.commission_rate
                        cash -= gross + fee
                        total_shares = pos["shares"] + add_shares
                        pos["entry_price"] = (pos["entry_price"] * pos["shares"] + add_price * add_shares) / total_shares
                        pos["shares"] = total_shares
                        pos["initial_shares"] += add_shares
                        pos["买入批次"] = next_batch

        candidates = signal_map.get(current_date, [])
        for signal in candidates:
            if signal["code"] in positions:
                continue
            df = prepared[signal["code"]]
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            if not _can_open_position(row):
                continue

            available_slots = max(config.max_positions - len(positions), 0)
            if available_slots <= 0 and positions and daily_replacements < config.max_daily_replacements:
                replace_candidates = []
                for replace_code, replace_pos in positions.items():
                    replace_df = prepared[replace_code]
                    if current_date not in replace_df.index:
                        continue
                    replace_row = replace_df.loc[current_date]
                    if not _can_close_position(replace_row):
                        continue
                    idx_now = replace_df.index.get_loc(current_date)
                    idx_entry = replace_df.index.get_loc(replace_pos["entry_date"])
                    hold_bars = idx_now - idx_entry + 1
                    if not _position_replace_eligible(label, replace_pos, replace_df, idx_now, idx_entry, hold_bars):
                        continue
                    replace_score = replace_pos.get("prior_close_score", replace_pos["score"])
                    replace_candidates.append((replace_score, replace_code, replace_row))

                if replace_candidates:
                    replace_score, replace_code, replace_row = min(replace_candidates, key=lambda item: (item[0], item[1]))
                    if signal["score"] > replace_score + config.replacement_threshold:
                        cash = _close_position(cash, positions, trades, replace_code, current_date, float(replace_row["open"]), "换仓卖出", config)
                        available_slots = 1
                        daily_replacements += 1

            if available_slots <= 0:
                continue

            entry_price = float(row["open"]) * (1.0 + config.slippage_rate)
            planned_value = cash / available_slots if available_slots > 0 else 0.0
            shares = _buy_shares_for_value(cash, entry_price, planned_value * 0.3, config)
            if shares <= 0:
                continue
            gross = shares * entry_price
            fee = gross * config.commission_rate
            cash -= gross + fee
            positions[signal["code"]] = {
                "entry_date": current_date,
                "entry_price": entry_price,
                "shares": shares,
                "initial_shares": shares,
                "stop_price": float(row["low"]) * 0.9,
                "score": signal["score"],
                "prior_close_score": signal["score"],
                "计划仓位金额": planned_value,
                "买入批次": 1,
                "最短换仓持有天数": config.min_hold_days_for_replace,
                "step1": False,
                "step2": False,
                "step3": False,
                "step4": False,
                "step5": False,
                "dd_count": 0,
            }
            opened_today.add(signal["code"])

        for code, pos in positions.items():
            df = prepared[code]
            if current_date in df.index:
                base_score = _position_current_score(code, current_date, prepared, weights, pos.get("prior_close_score", pos["score"]))
                pos["prior_close_score"] = base_score - _position_dynamic_penalty(pos, df.loc[current_date])

        equity = cash
        for code, pos in positions.items():
            df = prepared[code]
            if current_date in df.index:
                equity += pos["shares"] * float(df.loc[current_date, "close"])
            else:
                equity += pos["shares"] * pos["entry_price"]
        equity_index.append(current_date)
        equity_points.append(equity)

    equity_curve = pd.Series(equity_points, index=pd.DatetimeIndex(equity_index), dtype=float)
    metrics = compute_metrics(equity_curve)
    payload = {
        "label": label,
        "metrics": metrics,
        "trade_count": len(trades),
        "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else config.initial_capital,
        "equity_curve": equity_curve,
        "trades": trades,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weighted-factor portfolio backtests under capital constraints")
    parser.add_argument("data_dir")
    parser.add_argument("--refined-root", default="results/multi_factor_research_v4_refined")
    parser.add_argument("--output-root", default="results/weighted_portfolio_backtest")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--replacement-threshold", type=float, default=0.03)
    parser.add_argument("--min-hold-days-for-replace", type=int, default=5)
    parser.add_argument("--max-daily-replacements", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    weights_payload = _load_refined_weights(Path(args.refined_root))
    stock_data, all_dates = load_price_directory(args.data_dir)
    prepared = _prepare_stock_frames(stock_data)
    config = PortfolioConfig(
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        replacement_threshold=args.replacement_threshold,
        min_hold_days_for_replace=args.min_hold_days_for_replace,
        max_daily_replacements=args.max_daily_replacements,
    )

    models = {
        "fixed_take_profit": _parse_weight_spec(weights_payload["fixed_take_profit"]["combo"]),
        "fixed_days": _parse_weight_spec(weights_payload["fixed_days"]["combo"]),
        "tiered": _parse_weight_spec(weights_payload["tiered"]["combo"]),
    }

    summary: dict[str, dict] = {}
    for label, weights in models.items():
        result = _run_model(label, prepared, all_dates, weights, config)
        summary[label] = {
            "weights": weights,
            "metrics": result["metrics"],
            "trade_count": result["trade_count"],
            "final_equity": result["final_equity"],
            "换仓分差阈值": config.replacement_threshold,
            "最短换仓持有天数": config.min_hold_days_for_replace,
            "每日最大换仓数": config.max_daily_replacements,
        }
        run_dir = output_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary[label], ensure_ascii=False, indent=2), encoding="utf-8")
        result["equity_curve"].rename("equity").to_csv(run_dir / "equity_curve.csv", encoding="utf-8")
        with (run_dir / "trades.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["code", "entry_date", "exit_date", "entry_price", "exit_price", "shares", "reason"])
            writer.writeheader()
            for trade in result["trades"]:
                writer.writerow(
                    {
                        "code": trade["code"],
                        "entry_date": pd.Timestamp(trade["entry_date"]).strftime("%Y-%m-%d"),
                        "exit_date": pd.Timestamp(trade["exit_date"]).strftime("%Y-%m-%d"),
                        "entry_price": trade["entry_price"],
                        "exit_price": trade["exit_price"],
                        "shares": trade["shares"],
                        "reason": trade["reason"],
                    }
                )

    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
