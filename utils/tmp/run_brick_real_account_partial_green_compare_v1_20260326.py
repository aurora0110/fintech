from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
DAILY_DIR = ROOT / "data" / "20260324"
SIGNAL_SOURCE = RESULT_ROOT / "brick_minute_execution_compare_v1_full_day5_parallel_20260325_r4" / "selected_signals.csv"
OLD_RESULT_DIR = RESULT_ROOT / "brick_hybrid_local_search_minoc_full_20260326_r2"
BASE_SCRIPT_PATH = ROOT / "utils" / "tmp" / "run_brick_intraday_minute_compare_v1_20260325.py"
REAL_COMPARE_SOURCE = RESULT_ROOT / "brick_real_account_compare_v1_full_20260326_r2"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metrics import compute_metrics
from core.data_loader import _read_txt

INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
DAILY_NEW_LIMIT = 10
DAILY_BUDGET_FRAC = 1.0
POSITION_CAP_FRAC = 0.10
ALLOCATION_MODE = "equal"
COMMISSION_RATE = 0.0003
SLIPPAGE_RATE = 0.001
STAMP_DUTY_RATE = 0.001
MIN_LOT = 100
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT_PATH, "brick_partial_green_compare_base_v1")


@dataclass(frozen=True)
class AccountConfig:
    initial_capital: float = INITIAL_CAPITAL
    max_positions: int = MAX_POSITIONS
    daily_new_limit: int = DAILY_NEW_LIMIT
    daily_budget_frac: float = DAILY_BUDGET_FRAC
    position_cap_frac: float = POSITION_CAP_FRAC
    allocation_mode: str = ALLOCATION_MODE
    commission_rate: float = COMMISSION_RATE
    slippage_rate: float = SLIPPAGE_RATE
    stamp_duty_rate: float = STAMP_DUTY_RATE
    min_lot: int = MIN_LOT


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_error(result_dir: Path, exc: BaseException) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (result_dir / "error.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))


def load_old_champion_trades(file_limit: int) -> pd.DataFrame:
    summary = json.loads((OLD_RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    strategy_key = str(summary["strategy_best"])
    trades = pd.read_csv(
        OLD_RESULT_DIR / "hybrid_local_trades.csv",
        parse_dates=["signal_date", "entry_date", "exit_date"],
    )
    trades = trades[trades["strategy_key"] == strategy_key].copy()
    if trades.empty:
        raise RuntimeError(f"未找到旧冠军交易: {strategy_key}")
    trades = trades.sort_values(["signal_date", "code", "signal_idx"]).reset_index(drop=True)
    if file_limit > 0:
        keep_codes = sorted(trades["code"].astype(str).unique())[:file_limit]
        trades = trades[trades["code"].astype(str).isin(keep_codes)].copy()
    return trades.reset_index(drop=True)


def load_signal_scores(file_limit: int) -> pd.DataFrame:
    df = pd.read_csv(SIGNAL_SOURCE, parse_dates=["signal_date", "entry_date"])
    if file_limit > 0:
        keep_codes = sorted(df["code"].astype(str).unique())[:file_limit]
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    df["sort_score"] = pd.to_numeric(df.get("base_score", 0.0), errors="coerce").fillna(0.0)
    return df[["code", "signal_idx", "signal_date", "entry_date", "sort_score", "base_score"]].drop_duplicates(
        ["code", "signal_idx", "signal_date", "entry_date"]
    )


def attach_sort_scores(trades: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    out = trades.merge(score_df, on=["code", "signal_idx", "signal_date", "entry_date"], how="left")
    out["sort_score"] = pd.to_numeric(out["sort_score"], errors="coerce").fillna(0.0)
    return out


def _load_feature_subset_for_code(code: str) -> tuple[str, pd.DataFrame | None]:
    path = DAILY_DIR / f"{code}.txt"
    if not path.exists():
        return code, None
    raw = _read_txt(str(path))
    if raw is None or raw.empty:
        return code, None
    daily_df = raw[(raw["date"] < base.EXCLUDE_START) | (raw["date"] > base.EXCLUDE_END)].copy()
    daily_df = daily_df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["date", "open", "high", "low", "close"])
    if daily_df.empty:
        return code, None
    feat = base.brick_base.build_feature_df(daily_df)
    keep = feat[["date", "close", "brick_green"]].copy()
    keep["brick_green"] = keep["brick_green"].fillna(False).astype(bool)
    return code, keep


def build_feature_registry(codes: list[str], max_workers: int, progress_cb: Any | None = None) -> dict[str, pd.DataFrame]:
    codes = sorted(set(codes))
    out: dict[str, pd.DataFrame] = {}
    total = len(codes)
    if total == 0:
        return out
    if max_workers <= 1:
        for idx, code in enumerate(codes, start=1):
            _, feat = _load_feature_subset_for_code(code)
            if feat is not None and not feat.empty:
                out[code] = feat
            if progress_cb is not None and (idx == 1 or idx % 100 == 0 or idx == total):
                progress_cb(idx, total)
        return out
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_feature_subset_for_code, code): code for code in codes}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                code, feat = future.result()
                if feat is not None and not feat.empty:
                    out[code] = feat
                if progress_cb is not None and (completed == 1 or completed % 100 == 0 or completed == total):
                    progress_cb(completed, total)
    except Exception:
        out = {}
        for idx, code in enumerate(codes, start=1):
            _, feat = _load_feature_subset_for_code(code)
            if feat is not None and not feat.empty:
                out[code] = feat
            if progress_cb is not None and (idx == 1 or idx % 100 == 0 or idx == total):
                progress_cb(idx, total, fallback="serial")
    return out


def infer_exit_session(exit_reason: str) -> str:
    reason = str(exit_reason)
    if "next_open" in reason:
        return "open"
    if "same_day_hybrid" in reason or "same_day" in reason:
        return "intraday"
    if "close" in reason:
        return "close"
    return "close"


def find_green_exit(code: str, partial_exit_date: pd.Timestamp, partial_session: str, feature_registry: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, float, str]:
    feat = feature_registry.get(code)
    if feat is None or feat.empty:
        raise RuntimeError(f"缺少 {code} 的特征表，无法搜索绿砖退出")
    if partial_session == "open":
        mask = feat["date"] >= partial_exit_date
    else:
        mask = feat["date"] > partial_exit_date
    future = feat[mask].copy()
    if future.empty:
        last = feat.iloc[-1]
        return pd.Timestamp(last["date"]), float(last["close"]), "end_of_data_close_after_partial"
    green = future[future["brick_green"]]
    if not green.empty:
        row = green.iloc[0]
        return pd.Timestamp(row["date"]), float(row["close"]), "green_close_after_partial"
    last = future.iloc[-1]
    return pd.Timestamp(last["date"]), float(last["close"]), "end_of_window_close_after_partial"


def build_strategy_events(trades: pd.DataFrame, feature_registry: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []

    for row in trades.itertuples(index=False):
        position_id = f"{row.code}|{int(row.signal_idx)}|{pd.Timestamp(row.entry_date).date()}"
        base_payload = {
            "position_id": position_id,
            "code": row.code,
            "signal_idx": int(row.signal_idx),
            "signal_date": pd.Timestamp(row.signal_date),
            "entry_date": pd.Timestamp(row.entry_date),
            "entry_price": float(row.entry_price),
            "sort_score": float(row.sort_score),
            "strategy_key": str(row.strategy_key),
        }
        old_rows.append(
            {
                **base_payload,
                "strategy_label": "old_strategy",
                "leg_name": "full_exit",
                "exit_date": pd.Timestamp(row.exit_date),
                "exit_session": infer_exit_session(str(row.exit_reason)),
                "exit_price": float(row.exit_price),
                "exit_ratio": 1.0,
                "exit_reason": str(row.exit_reason),
            }
        )

        reason = str(row.exit_reason)
        if reason.startswith("tp_"):
            partial_session = infer_exit_session(reason)
            partial_date = pd.Timestamp(row.exit_date)
            partial_price = float(row.exit_price)
            green_date, green_price, green_reason = find_green_exit(
                code=str(row.code),
                partial_exit_date=partial_date,
                partial_session=partial_session,
                feature_registry=feature_registry,
            )
            new_rows.append(
                {
                    **base_payload,
                    "strategy_label": "partial_80_green_20",
                    "leg_name": "tp_partial_80",
                    "exit_date": partial_date,
                    "exit_session": partial_session,
                    "exit_price": partial_price,
                    "exit_ratio": 0.8,
                    "exit_reason": f"{reason}_80pct",
                }
            )
            new_rows.append(
                {
                    **base_payload,
                    "strategy_label": "partial_80_green_20",
                    "leg_name": "green_tail_20",
                    "exit_date": green_date,
                    "exit_session": "close",
                    "exit_price": green_price,
                    "exit_ratio": 0.2,
                    "exit_reason": green_reason,
                }
            )
        else:
            new_rows.append(
                {
                    **base_payload,
                    "strategy_label": "partial_80_green_20",
                    "leg_name": "full_exit",
                    "exit_date": pd.Timestamp(row.exit_date),
                    "exit_session": infer_exit_session(reason),
                    "exit_price": float(row.exit_price),
                    "exit_ratio": 1.0,
                    "exit_reason": reason,
                }
            )

    old_df = pd.DataFrame(old_rows).sort_values(["entry_date", "code", "signal_idx"]).reset_index(drop=True)
    new_df = pd.DataFrame(new_rows).sort_values(["entry_date", "code", "signal_idx", "exit_date", "leg_name"]).reset_index(drop=True)
    return old_df, new_df


def _fast_load_close_series(path: Path) -> pd.Series | None:
    df = _read_txt(str(path))
    if df is None or df.empty:
        return None
    df = df[(df["date"] < base.EXCLUDE_START) | (df["date"] > base.EXCLUDE_END)].copy()
    if df.empty:
        return None
    return df[["date", "close"]].dropna(subset=["date", "close"]).set_index("date")["close"].astype(float)


def build_close_map(codes: list[str], progress_cb: Any | None = None) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    relevant: dict[str, pd.Series] = {}
    all_dates: set[pd.Timestamp] = set()
    unique_codes = sorted(set(codes))
    total = len(unique_codes)
    for idx, code in enumerate(unique_codes, start=1):
        path = DAILY_DIR / f"{code}.txt"
        if not path.exists():
            continue
        s = _fast_load_close_series(path)
        if s is None or s.empty:
            continue
        relevant[code] = s
        all_dates.update(s.index.tolist())
        if progress_cb is not None and (idx == 1 or idx % 100 == 0 or idx == total):
            progress_cb(idx, total)
    market_dates = pd.DatetimeIndex(sorted(all_dates))
    close_map: dict[str, pd.Series] = {}
    for code, s in relevant.items():
        close_map[code] = s.reindex(market_dates).ffill()
    return market_dates, close_map


def _compute_calmar(annual_return: float, max_drawdown_abs: float) -> float:
    if not np.isfinite(annual_return):
        return float("nan")
    if not np.isfinite(max_drawdown_abs) or max_drawdown_abs <= 0:
        return float("nan")
    return float(annual_return / max_drawdown_abs)


def _max_losing_streak(returns: list[float]) -> int:
    streak = 0
    max_streak = 0
    for r in returns:
        if r <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def simulate_real_account_event_based(
    events: pd.DataFrame,
    close_map: dict[str, pd.Series],
    market_dates: pd.DatetimeIndex,
    config: AccountConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if events.empty:
        raise RuntimeError("事件表为空，无法回测")

    plans = events[
        ["position_id", "code", "signal_idx", "signal_date", "entry_date", "entry_price", "sort_score", "strategy_label", "strategy_key"]
    ].drop_duplicates(["position_id"]).copy()
    plans = plans.sort_values(["entry_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    entries_by_date = {
        d: g.sort_values(["sort_score", "code"], ascending=[False, True]).to_dict("records")
        for d, g in plans.groupby("entry_date")
    }

    exits_by_date_session: dict[tuple[pd.Timestamp, str], list[dict[str, Any]]] = {}
    for rec in events.sort_values(["exit_date", "exit_session", "code"]).to_dict("records"):
        key = (pd.Timestamp(rec["exit_date"]), str(rec["exit_session"]))
        exits_by_date_session.setdefault(key, []).append(rec)

    cash = float(config.initial_capital)
    positions: dict[str, dict[str, Any]] = {}
    executed_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    def process_exit_event(current_date: pd.Timestamp, session: str) -> None:
        nonlocal cash
        for ev in exits_by_date_session.get((current_date, session), []):
            position_id = str(ev["position_id"])
            if position_id not in positions:
                continue
            pos = positions[position_id]
            total_shares = int(pos["shares_total"])
            remaining_shares = int(pos["shares_remaining"])
            if remaining_shares <= 0:
                positions.pop(position_id, None)
                continue

            if str(ev["leg_name"]) == "tp_partial_80":
                target = int((total_shares * 0.8) // config.min_lot) * config.min_lot
                if target <= 0 or target >= total_shares:
                    shares_to_sell = remaining_shares
                else:
                    shares_to_sell = min(target, remaining_shares)
            else:
                shares_to_sell = remaining_shares

            raw_exit_price = float(ev["exit_price"])
            exit_price = raw_exit_price * (1.0 - config.slippage_rate)
            gross_cash = shares_to_sell * exit_price
            fee = gross_cash * config.commission_rate
            tax = gross_cash * config.stamp_duty_rate
            cash += gross_cash - fee - tax

            entry_fee_alloc = pos["entry_fee_total"] * (shares_to_sell / total_shares)
            pnl = (exit_price - pos["entry_price_exec"]) * shares_to_sell - entry_fee_alloc - fee - tax
            cost_base = pos["entry_price_exec"] * shares_to_sell + entry_fee_alloc
            realized_return = pnl / cost_base if cost_base > 0 else float("nan")
            pos["shares_remaining"] = remaining_shares - shares_to_sell

            executed_rows.append(
                {
                    "strategy_label": pos["strategy_label"],
                    "strategy_key": pos["strategy_key"],
                    "position_id": position_id,
                    "code": pos["code"],
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "exit_session": session,
                    "leg_name": ev["leg_name"],
                    "entry_price_raw": pos["entry_price_raw"],
                    "entry_price_exec": pos["entry_price_exec"],
                    "exit_price_raw": raw_exit_price,
                    "exit_price_exec": exit_price,
                    "shares_sold": shares_to_sell,
                    "shares_remaining_after": pos["shares_remaining"],
                    "gross_entry_cost_alloc": pos["entry_price_exec"] * shares_to_sell,
                    "entry_fee_alloc": entry_fee_alloc,
                    "exit_fee_tax": fee + tax,
                    "pnl": pnl,
                    "return_pct_net": realized_return,
                    "exit_reason": ev["exit_reason"],
                    "sort_score": pos["sort_score"],
                }
            )

            if pos["shares_remaining"] <= 0:
                positions.pop(position_id, None)

    for current_date in market_dates:
        process_exit_event(current_date, "open")

        equity_before_entry = cash
        for _, pos in positions.items():
            mark_price = float(close_map[pos["code"]].get(current_date, pos["entry_price_exec"]))
            equity_before_entry += pos["shares_remaining"] * mark_price

        entry_candidates = entries_by_date.get(current_date, [])
        available_slots = max(config.max_positions - len(positions), 0)
        if entry_candidates and available_slots > 0:
            to_add: list[dict[str, Any]] = []
            for rec in entry_candidates:
                position_id = str(rec["position_id"])
                if position_id in positions:
                    continue
                to_add.append(rec)
                if len(to_add) >= min(available_slots, config.daily_new_limit):
                    break
            if to_add:
                investable = min(cash, equity_before_entry * config.daily_budget_frac)
                if investable > 0:
                    if config.allocation_mode == "equal":
                        weights = np.full(len(to_add), 1.0 / len(to_add), dtype=float)
                    else:
                        scores = np.array([max(float(rec["sort_score"]), 0.01) for rec in to_add], dtype=float)
                        weights = scores / scores.sum()
                    per_pos_cap = equity_before_entry * config.position_cap_frac
                    for rec, weight in zip(to_add, weights):
                        raw_entry_price = float(rec["entry_price"])
                        entry_price = raw_entry_price * (1.0 + config.slippage_rate)
                        alloc = min(investable * float(weight), per_pos_cap, cash)
                        if alloc <= 0 or entry_price <= 0:
                            continue
                        shares = int(alloc / entry_price / config.min_lot) * config.min_lot
                        if shares <= 0:
                            continue
                        gross_cost = shares * entry_price
                        fee = gross_cost * config.commission_rate
                        total_cost = gross_cost + fee
                        if total_cost > cash:
                            continue
                        cash -= total_cost
                        positions[str(rec["position_id"])] = {
                            "code": str(rec["code"]),
                            "signal_date": pd.Timestamp(rec["signal_date"]),
                            "entry_date": current_date,
                            "shares_total": shares,
                            "shares_remaining": shares,
                            "entry_price_raw": raw_entry_price,
                            "entry_price_exec": entry_price,
                            "entry_fee_total": fee,
                            "sort_score": float(rec["sort_score"]),
                            "strategy_label": str(rec["strategy_label"]),
                            "strategy_key": str(rec["strategy_key"]),
                        }

        process_exit_event(current_date, "intraday")
        process_exit_event(current_date, "close")

        equity = cash
        for _, pos in positions.items():
            mark_price = float(close_map[pos["code"]].get(current_date, pos["entry_price_exec"]))
            equity += pos["shares_remaining"] * mark_price
        equity_rows.append({"date": current_date, "equity": equity, "cash": cash, "position_count": len(positions)})

    equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    equity_curve = pd.Series(
        equity_df["equity"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(equity_df["date"]),
        dtype=float,
    )
    metrics = compute_metrics(equity_curve)
    max_drawdown_abs = float(metrics["max_drawdown"])
    annual_return = float(metrics["annual_return"])
    sharpe = float(metrics["sharpe"])
    calmar = _compute_calmar(annual_return, max_drawdown_abs)

    executed_df = pd.DataFrame(executed_rows).sort_values(["exit_date", "entry_date", "code", "leg_name"]).reset_index(drop=True)
    if executed_df.empty:
        raise RuntimeError("账户回放没有任何成交")

    position_summary = (
        executed_df.groupby(["strategy_label", "position_id"], as_index=False)
        .agg(
            code=("code", "first"),
            signal_date=("signal_date", "first"),
            entry_date=("entry_date", "first"),
            exit_date=("exit_date", "max"),
            total_pnl=("pnl", "sum"),
            gross_entry_cost=("gross_entry_cost_alloc", "sum"),
            entry_fee=("entry_fee_alloc", "sum"),
            strategy_key=("strategy_key", "first"),
        )
    )
    position_summary["trade_return_net"] = position_summary["total_pnl"] / (position_summary["gross_entry_cost"] + position_summary["entry_fee"])

    avg_trade_return = float(position_summary["trade_return_net"].mean())
    success_rate = float((position_summary["trade_return_net"] > 0).mean())
    hold_return = float(equity_df.iloc[-1]["equity"] / config.initial_capital - 1.0)
    payoff_wins = position_summary.loc[position_summary["trade_return_net"] > 0, "trade_return_net"]
    payoff_losses = position_summary.loc[position_summary["trade_return_net"] < 0, "trade_return_net"]
    payoff_ratio = float(payoff_wins.mean() / abs(payoff_losses.mean())) if not payoff_losses.empty else float("inf")
    profit_factor = float(payoff_wins.sum() / abs(payoff_losses.sum())) if not payoff_losses.empty else float("inf")

    summary = {
        "final_multiple": float(metrics["final_multiple"]),
        "annual_return": annual_return,
        "holding_return": hold_return,
        "max_drawdown": -max_drawdown_abs,
        "sharpe": sharpe,
        "calmar": calmar,
        "trade_count": int(len(position_summary)),
        "success_rate": success_rate,
        "avg_trade_return": avg_trade_return,
        "avg_win_return": float(payoff_wins.mean()) if not payoff_wins.empty else float("nan"),
        "avg_loss_return": float(payoff_losses.mean()) if not payoff_losses.empty else float("nan"),
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "max_losing_streak": int(_max_losing_streak(position_summary["trade_return_net"].tolist())),
        "equity_days": int(metrics["days"]),
        "final_equity": float(equity_df.iloc[-1]["equity"]),
    }
    return equity_df, executed_df, summary


def run_chain(result_dir: Path, file_limit: int, max_workers: int) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "loading_old_trades", file_limit=file_limit, max_workers=max_workers)

    old_trades = load_old_champion_trades(file_limit=file_limit)
    score_df = load_signal_scores(file_limit=file_limit)
    old_trades = attach_sort_scores(old_trades, score_df)
    old_trades.to_csv(result_dir / "old_champion_source_trades.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "old_trades_ready", trade_rows=int(len(old_trades)))

    codes = sorted(old_trades["code"].astype(str).unique())
    def feature_progress(done: int, total: int, **extra: Any) -> None:
        update_progress(result_dir, "loading_features", done_codes=done, total_codes=total, **extra)

    feature_registry = build_feature_registry(
        codes,
        max_workers=max_workers,
        progress_cb=feature_progress,
    )
    feature_coverage = pd.DataFrame(
        [{"code": code, "has_feature": bool(code in feature_registry)} for code in codes]
    )
    feature_coverage.to_csv(result_dir / "feature_coverage.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "features_ready", feature_codes=int(len(feature_registry)))

    old_events, new_events = build_strategy_events(old_trades, feature_registry)
    old_events.to_csv(result_dir / "old_strategy_events.csv", index=False, encoding="utf-8-sig")
    new_events.to_csv(result_dir / "partial_green_events.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "events_ready", old_events=int(len(old_events)), new_events=int(len(new_events)))

    market_dates, close_map = build_close_map(
        codes,
        progress_cb=lambda done, total: update_progress(result_dir, "building_close_map", done_codes=done, total_codes=total),
    )
    update_progress(result_dir, "close_map_ready", market_days=int(len(market_dates)), close_codes=int(len(close_map)))

    config = AccountConfig()
    old_equity, old_exec, old_summary = simulate_real_account_event_based(old_events, close_map, market_dates, config)
    old_equity.to_csv(result_dir / "old_strategy_equity.csv", index=False, encoding="utf-8-sig")
    old_exec.to_csv(result_dir / "old_strategy_executed_trades.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "old_strategy_ready", trade_count=int(old_summary["trade_count"]))

    new_equity, new_exec, new_summary = simulate_real_account_event_based(new_events, close_map, market_dates, config)
    new_equity.to_csv(result_dir / "partial_green_equity.csv", index=False, encoding="utf-8-sig")
    new_exec.to_csv(result_dir / "partial_green_executed_trades.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "new_strategy_ready", trade_count=int(new_summary["trade_count"]))

    comparison = {
        "annual_return_diff": float(new_summary["annual_return"] - old_summary["annual_return"]),
        "holding_return_diff": float(new_summary["holding_return"] - old_summary["holding_return"]),
        "avg_trade_return_diff": float(new_summary["avg_trade_return"] - old_summary["avg_trade_return"]),
        "success_rate_diff": float(new_summary["success_rate"] - old_summary["success_rate"]),
        "max_drawdown_diff": float(new_summary["max_drawdown"] - old_summary["max_drawdown"]),
        "sharpe_diff": float(new_summary["sharpe"] - old_summary["sharpe"]),
        "calmar_diff": float(new_summary["calmar"] - old_summary["calmar"]),
        "payoff_ratio_diff": float(new_summary["payoff_ratio"] - old_summary["payoff_ratio"]),
        "profit_factor_diff": float(new_summary["profit_factor"] - old_summary["profit_factor"]),
    }
    pd.DataFrame(
        [
            {"label": "old_strategy", **old_summary},
            {"label": "partial_80_green_20", **new_summary},
        ]
    ).to_csv(result_dir / "real_account_summary.csv", index=False, encoding="utf-8-sig")

    assumptions = {
        "initial_capital": config.initial_capital,
        "max_positions": config.max_positions,
        "daily_new_limit": config.daily_new_limit,
        "daily_budget_frac": config.daily_budget_frac,
        "position_cap_frac": config.position_cap_frac,
        "allocation_mode": config.allocation_mode,
        "commission_rate": config.commission_rate,
        "slippage_rate": config.slippage_rate,
        "stamp_duty_rate": config.stamp_duty_rate,
        "min_lot": config.min_lot,
        "signal_pool": "brick.formal_best",
        "account_type": "real_account_engine_like_with_partial_exits",
        "old_strategy": "formal_best + 当日止损次日止盈 + min(open,close)止损 + 5.5%止盈",
        "new_strategy": "old_strategy 基础上，盈利触发时按原时点先卖80%，剩余20%等到第一次绿砖当日收盘卖出",
        "assumption_on_tail": "尾仓20%不再受3天持有上限限制，若直到样本结束都未转绿，则最后一天收盘卖出",
    }
    summary = {
        "assumptions": assumptions,
        "old_strategy": {"label": "old_strategy", **old_summary},
        "partial_green_20": {"label": "partial_80_green_20", **new_summary},
        "comparison": comparison,
        "reference_previous_real_compare": str(REAL_COMPARE_SOURCE),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--file-limit", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_limit = int(args.file_limit) if args.mode == "smoke" else 0
    result_dir = Path(args.output_dir)
    try:
        run_chain(result_dir=result_dir, file_limit=file_limit, max_workers=int(args.max_workers))
    except Exception as exc:
        write_error(result_dir, exc)
        raise


if __name__ == "__main__":
    main()
