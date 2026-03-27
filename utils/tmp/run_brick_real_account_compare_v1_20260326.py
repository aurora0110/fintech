from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
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
BASELINE_RESULT_DIR = RESULT_ROOT / "brick_minute_execution_compare_v1_full_day5_parallel_20260325_r4"
OLD_RESULT_DIR = RESULT_ROOT / "brick_hybrid_local_search_v1_full_20260326_r3"
MINOC_RESULT_DIR = RESULT_ROOT / "brick_hybrid_local_search_minoc_full_20260326_r1"
MINUTE_COMPARE_BASE_PATH = ROOT / "utils" / "tmp" / "run_brick_intraday_minute_compare_v1_20260325.py"
BASKET_COMPARE_R3 = RESULT_ROOT / "brick_hybrid_local_search_v1_full_20260326_r3" / "summary.json"
BASKET_COMPARE_MINOC = RESULT_ROOT / "brick_hybrid_local_search_minoc_full_20260326_r1" / "summary.json"

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
TRADING_DAYS_PER_YEAR = 252


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


minute_base = load_module(MINUTE_COMPARE_BASE_PATH, "brick_real_account_compare_base_v1")


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


def load_best_strategy(result_dir: Path) -> tuple[str, pd.DataFrame]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    strategy = str(summary["strategy_best"])
    trades = pd.read_csv(result_dir / "hybrid_local_trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    subset = trades[trades["strategy_key"] == strategy].copy()
    if subset.empty:
        raise RuntimeError(f"{result_dir} 未找到冠军策略交易: {strategy}")
    return strategy, subset.reset_index(drop=True)


def load_formal_baseline(result_dir: Path) -> tuple[str, pd.DataFrame]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    rows = summary.get("summary_rows", [])
    baseline_row = next((r for r in rows if r.get("mode") == "nextday_open"), None)
    if baseline_row is None:
        raise RuntimeError(f"{result_dir} 未找到 nextday_open baseline")
    strategy = str(baseline_row["strategy"])
    trades = pd.read_csv(result_dir / "execution_compare_trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    subset = trades[trades["execution_mode"] == "nextday_open"].copy()
    if subset.empty:
        raise RuntimeError(f"{result_dir} 未找到 nextday_open 交易明细")
    subset = subset.rename(columns={"execution_mode": "mode", "strategy": "strategy_key"})
    subset["strategy_key"] = strategy
    return strategy, subset.reset_index(drop=True)


def load_signal_scores() -> pd.DataFrame:
    df = pd.read_csv(SIGNAL_SOURCE, parse_dates=["signal_date", "entry_date"])
    df["sort_score"] = pd.to_numeric(df.get("base_score", 0.0), errors="coerce").fillna(0.0)
    return df[["code", "signal_idx", "signal_date", "entry_date", "sort_score", "base_score"]].drop_duplicates(
        ["code", "signal_idx", "signal_date", "entry_date"]
    )


def attach_sort_scores(trades: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    out = trades.merge(score_df, on=["code", "signal_idx", "signal_date", "entry_date"], how="left")
    out["sort_score"] = pd.to_numeric(out["sort_score"], errors="coerce").fillna(0.0)
    return out


def _fast_load_close_series(path: Path) -> pd.Series | None:
    df = _read_txt(str(path))
    if df is None or df.empty:
        return None
    df = df[(df["date"] < minute_base.EXCLUDE_START) | (df["date"] > minute_base.EXCLUDE_END)].copy()
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


def simulate_real_account(trades: pd.DataFrame, close_map: dict[str, pd.Series], market_dates: pd.DatetimeIndex, config: AccountConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if trades.empty:
        raise RuntimeError("交易表为空，无法做真实账户回测")

    trades = trades.sort_values(["entry_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    entries_by_date = {
        d: g.sort_values(["sort_score", "code"], ascending=[False, True]).to_dict("records")
        for d, g in trades.groupby("entry_date")
    }
    exits_by_date = {d: g.to_dict("records") for d, g in trades.groupby("exit_date")}

    cash = float(config.initial_capital)
    positions: dict[str, dict[str, Any]] = {}
    executed_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    closed_returns: list[float] = []

    for current_date in market_dates:
        for tr in exits_by_date.get(current_date, []):
            code = str(tr["code"])
            if code not in positions:
                continue
            pos = positions.pop(code)
            raw_exit_price = float(tr["exit_price"])
            exit_price = raw_exit_price * (1.0 - config.slippage_rate)
            gross_cash = pos["shares"] * exit_price
            fee = gross_cash * config.commission_rate
            tax = gross_cash * config.stamp_duty_rate
            cash += gross_cash - fee - tax
            pnl = (exit_price - pos["entry_price"]) * pos["shares"] - pos["entry_fee"] - fee - tax
            cost_base = pos["entry_price"] * pos["shares"] + pos["entry_fee"]
            realized_return = pnl / cost_base if cost_base > 0 else float("nan")
            closed_returns.append(realized_return)
            executed_rows.append(
                {
                    "strategy_key": tr["strategy_key"],
                    "code": code,
                    "signal_date": tr["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "entry_price_raw": pos["entry_price_raw"],
                    "entry_price_exec": pos["entry_price"],
                    "exit_price_raw": raw_exit_price,
                    "exit_price_exec": exit_price,
                    "shares": pos["shares"],
                    "gross_entry_cost": pos["entry_price"] * pos["shares"],
                    "entry_fee": pos["entry_fee"],
                    "exit_fee_tax": fee + tax,
                    "pnl": pnl,
                    "return_pct_net": realized_return,
                    "exit_reason": tr["exit_reason"],
                    "sort_score": tr["sort_score"],
                }
            )

        equity_before_entry = cash
        for code, pos in positions.items():
            mark_price = float(close_map[code].get(current_date, pos["entry_price"]))
            equity_before_entry += pos["shares"] * mark_price

        entry_candidates = entries_by_date.get(current_date, [])
        available_slots = max(config.max_positions - len(positions), 0)
        if entry_candidates and available_slots > 0:
            to_add: list[dict[str, Any]] = []
            for tr in entry_candidates:
                code = str(tr["code"])
                if code in positions:
                    continue
                to_add.append(tr)
                if len(to_add) >= min(available_slots, config.daily_new_limit):
                    break
            if to_add:
                investable = min(cash, equity_before_entry * config.daily_budget_frac)
                if investable > 0:
                    if config.allocation_mode == "equal":
                        weights = np.full(len(to_add), 1.0 / len(to_add), dtype=float)
                    else:
                        scores = np.array([max(float(tr["sort_score"]), 0.01) for tr in to_add], dtype=float)
                        weights = scores / scores.sum()
                    per_pos_cap = equity_before_entry * config.position_cap_frac
                    for tr, weight in zip(to_add, weights):
                        code = str(tr["code"])
                        raw_entry_price = float(tr["entry_price"])
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
                        positions[code] = {
                            "shares": shares,
                            "entry_price": entry_price,
                            "entry_price_raw": raw_entry_price,
                            "entry_fee": fee,
                            "entry_date": current_date,
                        }

        equity = cash
        for code, pos in positions.items():
            mark_price = float(close_map[code].get(current_date, pos["entry_price"]))
            equity += pos["shares"] * mark_price
        equity_rows.append({"date": current_date, "equity": equity, "cash": cash, "position_count": len(positions)})

    equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    equity_curve = pd.Series(equity_df["equity"].to_numpy(dtype=float), index=pd.DatetimeIndex(equity_df["date"]), dtype=float)
    metrics = compute_metrics(equity_curve)
    max_drawdown_abs = float(metrics["max_drawdown"])
    annual_return = float(metrics["annual_return"])
    sharpe = float(metrics["sharpe"])
    calmar = _compute_calmar(annual_return, max_drawdown_abs)
    executed_df = pd.DataFrame(executed_rows).sort_values(["exit_date", "entry_date", "code"]).reset_index(drop=True) if executed_rows else pd.DataFrame()
    avg_trade_return = float(executed_df["return_pct_net"].mean()) if not executed_df.empty else float("nan")
    success_rate = float((executed_df["return_pct_net"] > 0).mean()) if not executed_df.empty else float("nan")
    hold_return = float(equity_df.iloc[-1]["equity"] / config.initial_capital - 1.0) if not equity_df.empty else float("nan")
    summary = {
        "final_multiple": float(metrics["final_multiple"]),
        "annual_return": annual_return,
        "holding_return": hold_return,
        "max_drawdown": -max_drawdown_abs,
        "sharpe": sharpe,
        "calmar": calmar,
        "trade_count": int(len(executed_df)),
        "success_rate": success_rate,
        "avg_trade_return": avg_trade_return,
        "max_losing_streak": int(_max_losing_streak(executed_df["return_pct_net"].tolist() if not executed_df.empty else [])),
        "equity_days": int(metrics["days"]),
        "final_equity": float(equity_df.iloc[-1]["equity"]) if not equity_df.empty else float("nan"),
    }
    return equity_df, executed_df, summary


def run_compare(file_limit_codes: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(output_dir, "loading_sources", file_limit_codes=file_limit_codes)

    baseline_key, baseline_trades = load_formal_baseline(BASELINE_RESULT_DIR)
    old_key, old_trades = load_best_strategy(OLD_RESULT_DIR)
    minoc_key, minoc_trades = load_best_strategy(MINOC_RESULT_DIR)
    scores = load_signal_scores()
    baseline_trades = attach_sort_scores(baseline_trades, scores)
    old_trades = attach_sort_scores(old_trades, scores)
    minoc_trades = attach_sort_scores(minoc_trades, scores)

    if file_limit_codes > 0:
        keep_codes = sorted(set(baseline_trades["code"].astype(str)) | set(old_trades["code"].astype(str)) | set(minoc_trades["code"].astype(str)))[:file_limit_codes]
        baseline_trades = baseline_trades[baseline_trades["code"].astype(str).isin(keep_codes)].copy()
        old_trades = old_trades[old_trades["code"].astype(str).isin(keep_codes)].copy()
        minoc_trades = minoc_trades[minoc_trades["code"].astype(str).isin(keep_codes)].copy()

    all_codes = sorted(set(baseline_trades["code"].astype(str)) | set(old_trades["code"].astype(str)) | set(minoc_trades["code"].astype(str)))
    market_dates, close_map = build_close_map(
        all_codes,
        progress_cb=lambda done, total: update_progress(
            output_dir,
            "loading_sources",
            file_limit_codes=file_limit_codes,
            close_codes_done=done,
            close_codes_total=total,
        ),
    )
    if len(market_dates) == 0:
        raise RuntimeError("无法构建账户层 close_map")

    baseline_trades.to_csv(output_dir / "candidate_trades_baseline.csv", index=False, encoding="utf-8-sig")
    old_trades.to_csv(output_dir / "candidate_trades_old.csv", index=False, encoding="utf-8-sig")
    minoc_trades.to_csv(output_dir / "candidate_trades_minoc.csv", index=False, encoding="utf-8-sig")
    update_progress(
        output_dir,
        "data_ready",
        baseline_trade_count=len(baseline_trades),
        old_trade_count=len(old_trades),
        minoc_trade_count=len(minoc_trades),
        code_count=len(all_codes),
    )

    config = AccountConfig()
    summaries: list[dict[str, Any]] = []
    for label, strategy_key, trades in [
        ("formal_baseline", baseline_key, baseline_trades),
        ("old_champion", old_key, old_trades),
        ("minoc_champion", minoc_key, minoc_trades),
    ]:
        equity_df, executed_df, summary = simulate_real_account(trades, close_map, market_dates, config)
        equity_df.to_csv(output_dir / f"{label}_equity.csv", index=False, encoding="utf-8-sig")
        executed_df.to_csv(output_dir / f"{label}_executed_trades.csv", index=False, encoding="utf-8-sig")
        row = {"label": label, "strategy_key": strategy_key, **summary}
        summaries.append(row)
        update_progress(output_dir, "simulated_one", last_label=label)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "real_account_summary.csv", index=False, encoding="utf-8-sig")

    baseline_row = summary_df[summary_df["label"] == "formal_baseline"].iloc[0].to_dict()
    old_row = summary_df[summary_df["label"] == "old_champion"].iloc[0].to_dict()
    minoc_row = summary_df[summary_df["label"] == "minoc_champion"].iloc[0].to_dict()

    def build_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
        return {
            "annual_return_diff": float(b["annual_return"] - a["annual_return"]),
            "holding_return_diff": float(b["holding_return"] - a["holding_return"]),
            "avg_trade_return_diff": float(b["avg_trade_return"] - a["avg_trade_return"]),
            "success_rate_diff": float(b["success_rate"] - a["success_rate"]),
            "max_drawdown_diff": float(b["max_drawdown"] - a["max_drawdown"]),
            "sharpe_diff": float(b["sharpe"] - a["sharpe"]),
            "calmar_diff": float(b["calmar"] - a["calmar"]) if np.isfinite(b["calmar"]) and np.isfinite(a["calmar"]) else float("nan"),
        }

    summary_json = {
        "assumptions": {
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
            "account_type": "real_account_engine_like",
        "note": "基于已生成逐笔交易表做真实资金占用与持仓重叠回放，不再使用信号日篮子近似。",
        },
        "formal_baseline": baseline_row,
        "old_champion": old_row,
        "minoc_champion": minoc_row,
        "comparison": {
            "baseline_vs_old": build_diff(baseline_row, old_row),
            "baseline_vs_minoc": build_diff(baseline_row, minoc_row),
            "old_vs_minoc": build_diff(old_row, minoc_row),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(output_dir, "finished", output_dir=str(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK 两条执行冠军的真实账户引擎对比")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--file-limit-codes", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_real_account_compare_v1_{args.mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_limit_codes = int(args.file_limit_codes)
    if args.mode == "full":
        file_limit_codes = 0
    try:
        run_compare(file_limit_codes=file_limit_codes, output_dir=output_dir)
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
