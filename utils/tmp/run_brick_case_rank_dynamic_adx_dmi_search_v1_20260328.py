from __future__ import annotations

import argparse
import json
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
BASE_SCRIPT = ROOT / "utils" / "tmp" / "run_brick_case_rank_final_spec_search_v1_20260327.py"
BASE_RESULT_DIR = RESULT_ROOT / "brick_case_rank_h3_fixed_tp_extend_v1_full_20260327_r1"
SOURCE_CANDIDATES = RESULT_ROOT / "brick_case_rank_model_search_v1_full_20260327_r1" / "best_model_top20_candidates.csv"
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
ADX_PERIOD = 14
STRONG_ADX_VALUES = [20.0, 25.0, 30.0]
WEAK_ADX_VALUES = [15.0, 20.0, 25.0]
DMI_RULES = ["gt", "gt_1p1x"]
SMOKE_TP_GRID = [0.05, 0.07, 0.085, 0.10]
SMOKE_HOLD_GRID = [2, 3, 4]
FULL_TP_GRID = [0.05, 0.07, 0.085, 0.10]
FULL_HOLD_GRID = [2, 3, 4]
FIXED_STOP_BASE = "signal_low"
FIXED_STOP_EXEC = "same_day_close"
FIXED_EXIT_MODE = "next_day_open"
BUY_GAP_LIMIT = 0.04


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "brick_case_rank_dynamic_adx_dmi_base")


@dataclass
class SimResult:
    trade: dict[str, Any] | None
    skipped: bool = False
    skip_reason: str = ""


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_error(result_dir: Path, exc: BaseException) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (result_dir / "error.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))


def _compute_adx_dmi(daily_df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    df = daily_df.sort_values("date").copy()
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_sm = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_sm = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_dm_sm / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm_sm / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    out = df.copy()
    out["plus_di"] = plus_di.replace([np.inf, -np.inf], np.nan)
    out["minus_di"] = minus_di.replace([np.inf, -np.inf], np.nan)
    out["adx"] = adx.replace([np.inf, -np.inf], np.nan)
    return out


def _is_bullish_dmi(row: pd.Series, rule: str) -> bool:
    plus_di = float(row.get("plus_di", np.nan))
    minus_di = float(row.get("minus_di", np.nan))
    if not np.isfinite(plus_di) or not np.isfinite(minus_di):
        return False
    if rule == "gt":
        return plus_di > minus_di
    if rule == "gt_1p1x":
        return plus_di >= minus_di * 1.1
    raise ValueError(rule)


def _classify_regime(row: pd.Series, cfg: dict[str, Any]) -> str:
    adx = float(row.get("adx", np.nan))
    bullish = _is_bullish_dmi(row, str(cfg["dmi_rule"]))
    if np.isfinite(adx) and adx >= float(cfg["strong_adx"]) and bullish:
        return "strong"
    if (np.isfinite(adx) and adx <= float(cfg["weak_adx"])) or (not bullish):
        return "weak"
    return "neutral"


def _monotonic_triplets(values: list[float | int]) -> list[tuple[float | int, float | int, float | int]]:
    triplets: list[tuple[float | int, float | int, float | int]] = []
    vals = sorted(values)
    for i, weak in enumerate(vals):
        for j in range(i, len(vals)):
            neutral = vals[j]
            for k in range(j, len(vals)):
                strong = vals[k]
                triplets.append((weak, neutral, strong))
    return triplets


def _dynamic_configs(tp_grid: list[float], hold_grid: list[int]) -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = []
    tp_triplets = _monotonic_triplets(tp_grid)
    hold_triplets = _monotonic_triplets(hold_grid)
    for strong_adx in STRONG_ADX_VALUES:
        for weak_adx in WEAK_ADX_VALUES:
            if weak_adx >= strong_adx:
                continue
            for dmi_rule in DMI_RULES:
                for weak_tp, neutral_tp, strong_tp in tp_triplets:
                    for weak_hold, neutral_hold, strong_hold in hold_triplets:
                        cfgs.append(
                            {
                                "profile_name": (
                                    f"dynfix_adx{int(strong_adx)}_weak{int(weak_adx)}_{dmi_rule}"
                                    f"_tp{weak_tp:.3f}-{neutral_tp:.3f}-{strong_tp:.3f}"
                                    f"_h{weak_hold}-{neutral_hold}-{strong_hold}"
                                ),
                                "profile_family": "dynamic_fixed_tp",
                                "strong_adx": float(strong_adx),
                                "weak_adx": float(weak_adx),
                                "dmi_rule": str(dmi_rule),
                                "tp_map": {"weak": float(weak_tp), "neutral": float(neutral_tp), "strong": float(strong_tp)},
                                "hold_map": {"weak": int(weak_hold), "neutral": int(neutral_hold), "strong": int(strong_hold)},
                            }
                        )
    return cfgs


def _load_baseline_summary() -> dict[str, Any]:
    summary_path = BASE_RESULT_DIR / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8")).get("best_account_profile", {})


def _summarize_signal_basket(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["profile_name", "stop_base", "stop_exec_mode"]
    for keys, g in trades.groupby(group_cols, sort=True):
        profile_name, stop_base, stop_exec_mode = keys
        strategy = f"case_rank_dynamic|{stop_base}|{stop_exec_mode}|{profile_name}"
        row = base.tp_mod.hybrid.base.summarize_trades(g, strategy)
        first = g.iloc[0]
        row.update(
            {
                "profile_name": profile_name,
                "profile_family": str(first["profile_family"]),
                "strong_adx": float(first["strong_adx"]),
                "weak_adx": float(first["weak_adx"]),
                "dmi_rule": str(first["dmi_rule"]),
                "weak_tp": float(first["weak_tp"]),
                "neutral_tp": float(first["neutral_tp"]),
                "strong_tp": float(first["strong_tp"]),
                "weak_hold": int(first["weak_hold"]),
                "neutral_hold": int(first["neutral_hold"]),
                "strong_hold": int(first["strong_hold"]),
                "stop_base": stop_base,
                "stop_exec_mode": stop_exec_mode,
                "avg_realized_hold_days": float(pd.to_numeric(g["real_hold_days"], errors="coerce").mean()),
                "regime_weak_ratio": float((g["entry_regime"] == "weak").mean()),
                "regime_neutral_ratio": float((g["entry_regime"] == "neutral").mean()),
                "regime_strong_ratio": float((g["entry_regime"] == "strong").mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["annual_return_signal_basket", "final_equity_signal_basket", "profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _summarize_account(trades: pd.DataFrame, result_dir: Path, max_workers: int) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    all_codes = sorted(trades["code"].astype(str).unique())
    market_dates, close_map = base._build_close_map_parallel(all_codes, result_dir, max_workers=max_workers)
    if len(market_dates) == 0:
        raise RuntimeError("无法构建账户层 close_map")

    rows: list[dict[str, Any]] = []
    config = base.real_account.AccountConfig()

    def _is_open_exit(reason: str) -> bool:
        return "daily_open" in str(reason)

    group_cols = ["profile_name", "stop_base", "stop_exec_mode"]
    for keys, g in trades.groupby(group_cols, sort=True):
        profile_name, stop_base, stop_exec_mode = keys
        trades_one = g.sort_values(["entry_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
        entries_by_date = {
            d: gg.sort_values(["sort_score", "code"], ascending=[False, True]).to_dict("records")
            for d, gg in trades_one.groupby("entry_date")
        }
        exits_by_date = {d: gg.to_dict("records") for d, gg in trades_one.groupby("exit_date")}

        cash = float(config.initial_capital)
        positions: dict[str, dict[str, Any]] = {}
        executed_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []

        for current_date in market_dates:
            todays_exits = exits_by_date.get(current_date, [])
            open_exits = [tr for tr in todays_exits if _is_open_exit(str(tr["exit_reason"]))]
            later_exits = [tr for tr in todays_exits if not _is_open_exit(str(tr["exit_reason"]))]

            for tr in open_exits:
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
                executed_rows.append(
                    {
                        "strategy_key": f"case_rank_dynamic|{stop_base}|{stop_exec_mode}|{profile_name}",
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
                        weights = np.full(len(to_add), 1.0 / len(to_add), dtype=float)
                        per_pos_cap = equity_before_entry * config.position_cap_frac
                        for tr, weight in zip(to_add, weights):
                            code = str(tr["code"])
                            raw_entry_price = float(tr["entry_price"])
                            entry_price = raw_entry_price * (1.0 + config.slippage_rate)
                            alloc = min(investable * float(weight), per_pos_cap, cash)
                            shares = int(alloc / entry_price / config.min_lot) * config.min_lot if alloc > 0 and entry_price > 0 else 0
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

            for tr in later_exits:
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
                executed_rows.append(
                    {
                        "strategy_key": f"case_rank_dynamic|{stop_base}|{stop_exec_mode}|{profile_name}",
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

            equity = cash
            for code, pos in positions.items():
                mark_price = float(close_map[code].get(current_date, pos["entry_price"]))
                equity += pos["shares"] * mark_price
            equity_rows.append({"date": current_date, "equity": equity, "cash": cash, "position_count": len(positions)})

        equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
        equity_curve = pd.Series(equity_df["equity"].to_numpy(dtype=float), index=pd.DatetimeIndex(equity_df["date"]), dtype=float)
        metrics = base.real_account.compute_metrics(equity_curve)
        max_drawdown_abs = float(metrics["max_drawdown"])
        annual_return = float(metrics["annual_return"])
        sharpe = float(metrics["sharpe"])
        calmar = base.real_account._compute_calmar(annual_return, max_drawdown_abs)
        executed_df = pd.DataFrame(executed_rows).sort_values(["exit_date", "entry_date", "code"]).reset_index(drop=True) if executed_rows else pd.DataFrame()
        avg_trade_return = float(executed_df["return_pct_net"].mean()) if not executed_df.empty else float("nan")
        success_rate = float((executed_df["return_pct_net"] > 0).mean()) if not executed_df.empty else float("nan")
        hold_return = float(equity_df.iloc[-1]["equity"] / config.initial_capital - 1.0) if not equity_df.empty else float("nan")
        first = g.iloc[0]
        rows.append(
            {
                "profile_name": profile_name,
                "profile_family": str(first["profile_family"]),
                "strong_adx": float(first["strong_adx"]),
                "weak_adx": float(first["weak_adx"]),
                "dmi_rule": str(first["dmi_rule"]),
                "weak_tp": float(first["weak_tp"]),
                "neutral_tp": float(first["neutral_tp"]),
                "strong_tp": float(first["strong_tp"]),
                "weak_hold": int(first["weak_hold"]),
                "neutral_hold": int(first["neutral_hold"]),
                "strong_hold": int(first["strong_hold"]),
                "stop_base": stop_base,
                "stop_exec_mode": stop_exec_mode,
                "final_multiple": float(metrics["final_multiple"]),
                "annual_return": annual_return,
                "holding_return": hold_return,
                "max_drawdown": -max_drawdown_abs,
                "sharpe": sharpe,
                "calmar": calmar,
                "trade_count": int(len(executed_df)),
                "success_rate": success_rate,
                "avg_trade_return": avg_trade_return,
                "max_losing_streak": int(base.real_account._max_losing_streak(executed_df["return_pct_net"].tolist() if not executed_df.empty else [])),
                "equity_days": int(metrics["days"]),
                "final_equity": float(equity_df.iloc[-1]["equity"]) if not equity_df.empty else float("nan"),
                "avg_realized_hold_days": float(pd.to_numeric(g["real_hold_days"], errors="coerce").mean()),
                "regime_weak_ratio": float((g["entry_regime"] == "weak").mean()),
                "regime_neutral_ratio": float((g["entry_regime"] == "neutral").mean()),
                "regime_strong_ratio": float((g["entry_regime"] == "strong").mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["annual_return", "final_equity", "max_drawdown", "sharpe", "calmar"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    daily_path = base.DAILY_DIR / f"{code}.txt"
    if not daily_path.exists():
        return [], [{"code": code, "reason": "missing_daily_series"}]

    daily_df = base.tp_mod.hybrid.base.load_daily_df(daily_path)
    if daily_df is None or daily_df.empty:
        return [], [{"code": code, "reason": "empty_daily_df"}]
    daily_df = daily_df[(daily_df["date"] < EXCLUDE_START) | (daily_df["date"] > EXCLUDE_END)].copy()
    daily_df = _compute_adx_dmi(daily_df, period=ADX_PERIOD)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    min5_df = None
    min5_path = base.MIN5_DIR / f"{code}.txt"
    if min5_path.exists():
        loaded = base.tp_mod.hybrid.base.load_minute_df(min5_path)
        min5_df = base.tp_mod._prepare_min5_indicators(loaded) if loaded is not None and not loaded.empty else None

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    stop_cfg = {"stop_base": FIXED_STOP_BASE, "stop_exec_mode": FIXED_STOP_EXEC}

    daily_indexed = daily_df.set_index("date")
    for item in rows:
        signal_date = pd.Timestamp(item["signal_date"])
        if signal_date not in daily_indexed.index:
            skipped.append({"code": code, "signal_date": item["signal_date"], "reason": "missing_signal_date_row"})
            continue
        signal_row = daily_indexed.loc[signal_date]
        for cfg in payload["dynamic_configs"]:
            regime = _classify_regime(signal_row, cfg)
            tp_pct = float(cfg["tp_map"][regime])
            hold_days = int(cfg["hold_map"][regime])
            profile = {
                "name": cfg["profile_name"],
                "family": "fixed_tp",
                "tp_pct": tp_pct,
                "hold_days": hold_days,
                "exit_mode": FIXED_EXIT_MODE,
            }
            sim = base.simulate_one_trade_profile(
                code=code,
                signal_date=signal_date,
                entry_date=pd.Timestamp(item["entry_date"]),
                signal_idx=int(item.get("signal_idx", -1)),
                sort_score=float(item["sort_score"]),
                signal_open=float(item["signal_open"]),
                signal_close=float(item["signal_close"]),
                signal_low=float(item["signal_low"]),
                daily_df=daily_df,
                min5_df=min5_df,
                profile=profile,
                stop_cfg=stop_cfg,
                buy_gap_limit=float(payload["buy_gap_limit"]),
            )
            if sim.skipped or sim.trade is None:
                skipped.append(
                    {
                        "code": code,
                        "signal_date": item["signal_date"],
                        "entry_date": item["entry_date"],
                        "profile_name": cfg["profile_name"],
                        "reason": sim.skip_reason,
                    }
                )
                continue
            trade = sim.trade
            trade.update(
                {
                    "profile_name": cfg["profile_name"],
                    "profile_family": "dynamic_fixed_tp",
                    "strong_adx": float(cfg["strong_adx"]),
                    "weak_adx": float(cfg["weak_adx"]),
                    "dmi_rule": str(cfg["dmi_rule"]),
                    "weak_tp": float(cfg["tp_map"]["weak"]),
                    "neutral_tp": float(cfg["tp_map"]["neutral"]),
                    "strong_tp": float(cfg["tp_map"]["strong"]),
                    "weak_hold": int(cfg["hold_map"]["weak"]),
                    "neutral_hold": int(cfg["hold_map"]["neutral"]),
                    "strong_hold": int(cfg["hold_map"]["strong"]),
                    "entry_regime": regime,
                    "selected_tp_pct": tp_pct,
                    "selected_hold_days": hold_days,
                    "real_hold_days": int((pd.Timestamp(trade["exit_date"]) - pd.Timestamp(trade["entry_date"])).days),
                    "stop_base": FIXED_STOP_BASE,
                    "stop_exec_mode": FIXED_STOP_EXEC,
                }
            )
            trades.append(trade)
    return trades, skipped


def run_search(result_dir: Path, source_csv: Path, file_limit_codes: int, date_limit: int, max_workers: int, buy_gap_limit: float, tp_grid: list[float], hold_grid: list[int]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    dynamic_configs = _dynamic_configs(tp_grid=tp_grid, hold_grid=hold_grid)
    update_progress(
        result_dir,
        "loading_source",
        source_csv=str(source_csv),
        file_limit_codes=file_limit_codes,
        date_limit=date_limit,
        max_workers=max_workers,
        buy_gap_limit=buy_gap_limit,
        dynamic_config_count=len(dynamic_configs),
    )
    candidates = base.load_source_candidates(source_csv, file_limit_codes, date_limit)
    candidates.to_csv(result_dir / "source_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dynamic_configs).to_csv(result_dir / "dynamic_configs.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        raise RuntimeError("源候选为空")
    if "signal_idx" not in candidates.columns:
        candidates["signal_idx"] = -1

    grouped_payloads = []
    for code, g in candidates.groupby("code", sort=True):
        grouped_payloads.append(
            {
                "code": str(code),
                "rows": g[["signal_date", "entry_date", "signal_idx", "sort_score", "signal_open", "signal_close", "signal_low"]].to_dict("records"),
                "buy_gap_limit": float(buy_gap_limit),
                "dynamic_configs": dynamic_configs,
            }
        )

    total_codes = len(grouped_payloads)
    total_jobs = len(candidates) * len(dynamic_configs)
    trade_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(simulate_code_bundle, payload): payload["code"] for payload in grouped_payloads}
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            code = future_map[future]
            trades_part, skipped_part = future.result()
            trade_rows.extend(trades_part)
            skipped_rows.extend(skipped_part)
            if completed == 1 or completed % 25 == 0 or completed == total_codes:
                update_progress(
                    result_dir,
                    "simulating_trades",
                    done_codes=completed,
                    total_codes=total_codes,
                    done_jobs=len(trade_rows) + len(skipped_rows),
                    total_jobs=total_jobs,
                    last_code=code,
                )

    trades = pd.DataFrame(trade_rows).sort_values(["profile_name", "signal_date", "code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows).sort_values(["profile_name", "signal_date", "code"], na_position="last").reset_index(drop=True) if skipped_rows else pd.DataFrame()
    trades.to_csv(result_dir / "trades.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(result_dir / "skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "trades_ready", trade_count=int(len(trades)), skipped_count=int(len(skipped)))

    signal_summary = _summarize_signal_basket(trades)
    signal_summary.to_csv(result_dir / "signal_basket_summary.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "signal_basket_ready", profile_count=int(len(signal_summary)))

    account_summary = _summarize_account(trades, result_dir, max_workers=max_workers)
    account_summary.to_csv(result_dir / "account_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "source_candidates": str(source_csv),
            "fixed_buy_model": "case_rank_lgbm_top20",
            "exclude_window": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
            "dynamic_stage": "fixed_tp_only",
            "adx_period": ADX_PERIOD,
            "strong_adx_values": STRONG_ADX_VALUES,
            "weak_adx_values": WEAK_ADX_VALUES,
            "dmi_rules": DMI_RULES,
            "tp_grid": tp_grid,
            "hold_grid": hold_grid,
            "entry_time_regime_only": True,
            "fixed_stop_base": FIXED_STOP_BASE,
            "fixed_stop_exec_mode": FIXED_STOP_EXEC,
            "fixed_exit_mode": FIXED_EXIT_MODE,
        },
        "baseline_best_fixed_tp": _load_baseline_summary(),
        "best_signal_basket_profile": signal_summary.iloc[0].to_dict() if not signal_summary.empty else {},
        "best_account_profile": account_summary.iloc[0].to_dict() if not account_summary.empty else {},
        "signal_profile_count": int(len(signal_summary)),
        "account_profile_count": int(len(account_summary)),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="搜索 case_rank_lgbm_top20 的 ADX/DMI 动态固定止盈与动态持有天数")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--source-csv", type=str, default=str(SOURCE_CANDIDATES))
    parser.add_argument("--buy-gap-limit", type=float, default=BUY_GAP_LIMIT)
    parser.add_argument("--file-limit-codes", type=int, default=30)
    parser.add_argument("--date-limit", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_case_rank_dynamic_adx_dmi_search_v1_{args.mode}_{timestamp}"
    file_limit_codes = int(args.file_limit_codes)
    date_limit = int(args.date_limit)
    if args.mode == "full":
        file_limit_codes = 0
        date_limit = 0
    tp_grid = FULL_TP_GRID if args.mode == "full" else SMOKE_TP_GRID
    hold_grid = FULL_HOLD_GRID if args.mode == "full" else SMOKE_HOLD_GRID
    try:
        run_search(
            result_dir=output_dir,
            source_csv=Path(args.source_csv),
            file_limit_codes=file_limit_codes,
            date_limit=date_limit,
            max_workers=int(args.max_workers),
            buy_gap_limit=float(args.buy_gap_limit),
            tp_grid=tp_grid,
            hold_grid=hold_grid,
        )
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
