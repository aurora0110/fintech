from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
PHASE0_RESULT_DIR = RESULT_ROOT / "brick_case_rank_daily_stream_v2_full_20260328_r1"
SOURCE_CANDIDATES = PHASE0_RESULT_DIR / "daily_top20_candidates.csv"
DAILY_DIR = ROOT / "data" / "20260324"
MIN5_DIR = ROOT / "data" / "202603245min"
PHASE1_V1_PATH = ROOT / "utils" / "brick_optimize" / "run_brick_case_rank_phase1_fixed_exit_search_v1_20260328.py"
BASE_SCRIPT_PATH = ROOT / "utils" / "brick_optimize" / "run_brick_case_rank_final_spec_search_v1_20260327.py"
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUY_GAP_LIMIT = 0.04
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
PROFILE_BATCH_SIZE = 18

# Narrowed baseline grid based on prior relative results.
MAX_HOLD_DAYS_LIST = [3, 4, 5]
TP_LEVELS = [round(x, 4) for x in np.arange(0.07, 0.1101, 0.005)]
STOP_CONFIGS = [
    {"stop_base": "entry_low", "stop_exec_mode": "next_5min_close"},
    {"stop_base": "entry_low", "stop_exec_mode": "same_day_close"},
    {"stop_base": "signal_low", "stop_exec_mode": "next_5min_close"},
    {"stop_base": "signal_low", "stop_exec_mode": "same_day_close"},
]


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


phase1_v1 = load_module(PHASE1_V1_PATH, "brick_case_rank_phase1_v2_prev_mod")
base = load_module(BASE_SCRIPT_PATH, "brick_case_rank_phase1_v2_base_mod")


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


def _profile_defs() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for hold_days in MAX_HOLD_DAYS_LIST:
        for tp_pct in TP_LEVELS:
            tag = f"{tp_pct:.4f}"
            profiles.extend(
                [
                    {"name": f"tp_next5close_{tag}_h{hold_days}", "family": "fixed_tp", "tp_pct": tp_pct, "hold_days": hold_days, "exit_mode": "next_5min_close"},
                    {"name": f"tp_close_{tag}_h{hold_days}", "family": "fixed_tp", "tp_pct": tp_pct, "hold_days": hold_days, "exit_mode": "same_day_close"},
                    {"name": f"tp_next_open_{tag}_h{hold_days}", "family": "fixed_tp", "tp_pct": tp_pct, "hold_days": hold_days, "exit_mode": "next_day_open"},
                    {"name": f"tp_limit_same_day_{tag}_h{hold_days}", "family": "fixed_tp", "tp_pct": tp_pct, "hold_days": hold_days, "exit_mode": "limit_same_day"},
                ]
            )
    return profiles


PROFILE_DEFS = _profile_defs()


def _chunks(seq: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _pick_smoke_dates(dates: list[str], count: int) -> list[str]:
    if count <= 0 or len(dates) <= count:
        return dates
    idxs = np.linspace(0, len(dates) - 1, num=count, dtype=int)
    return [dates[i] for i in sorted(set(idxs.tolist()))]


def load_source_candidates(source_csv: Path, date_limit: int) -> pd.DataFrame:
    parse_dates = ["signal_date", "entry_date", "exit_date"]
    cols = pd.read_csv(source_csv, nrows=0).columns.tolist()
    df = pd.read_csv(source_csv, parse_dates=[c for c in parse_dates if c in cols])
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df[(df["signal_date"] < EXCLUDE_START) | (df["signal_date"] > EXCLUDE_END)].copy()
    df["code"] = df["code"].map(base._resolve_daily_stem)
    if "sort_score" not in df.columns:
        df["sort_score"] = pd.to_numeric(df.get("model_score", 0.0), errors="coerce").fillna(0.0)
    df = df.sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    if date_limit > 0:
        keep_dates = _pick_smoke_dates(sorted(df["signal_date"].dt.strftime("%Y-%m-%d").unique()), date_limit)
        df = df[df["signal_date"].dt.strftime("%Y-%m-%d").isin(keep_dates)].copy()
    return df.reset_index(drop=True)


def _resolve_exit(
    exit_mode: str,
    trigger_date: pd.Timestamp,
    trigger_dt: pd.Timestamp | None,
    day_daily_row: pd.Series,
    daily_df: pd.DataFrame,
    day_min: pd.DataFrame,
    tp_price: float,
) -> tuple[pd.Timestamp | None, float | None, str]:
    if exit_mode == "limit_same_day":
        return pd.Timestamp(trigger_date), float(tp_price), "limit_same_day"
    return base._resolve_exit(exit_mode, trigger_date, trigger_dt, day_daily_row, daily_df, day_min)


def simulate_one_trade_profile(
    code: str,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    signal_idx: int,
    sort_score: float,
    signal_open: float,
    signal_close: float,
    signal_low: float,
    daily_df: pd.DataFrame,
    min5_df: pd.DataFrame | None,
    profile: dict[str, Any],
    stop_cfg: dict[str, Any],
    buy_gap_limit: float,
) -> SimResult:
    entry_rows = daily_df[daily_df["date"] == entry_date]
    signal_rows = daily_df[daily_df["date"] == signal_date]
    if entry_rows.empty or signal_rows.empty:
        return SimResult(trade=None, skipped=True, skip_reason="missing_daily_row")
    entry_row = entry_rows.iloc[0]
    entry_open = float(entry_row["open"])
    entry_close = float(entry_row["close"])
    entry_low = float(entry_row["low"])
    if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(signal_close) or signal_close <= 0:
        return SimResult(trade=None, skipped=True, skip_reason="invalid_entry_or_signal")
    if entry_open / signal_close - 1.0 >= buy_gap_limit:
        return SimResult(trade=None, skipped=True, skip_reason="gap_gte_4pct")

    stop_price = base._resolve_stop_base(stop_cfg["stop_base"], signal_open, signal_close, signal_low, entry_open, entry_close, entry_low)
    tp_price = entry_open * (1.0 + float(profile["tp_pct"]))
    trade_window_dates, eligible_exit_dates = base._build_trade_window(daily_df, entry_date, int(profile["hold_days"]))
    if not trade_window_dates:
        return SimResult(trade=None, skipped=True, skip_reason="no_trade_window")

    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    trigger_source = "max_hold"

    for d in eligible_exit_dates:
        day_daily = daily_df[daily_df["date"] == d]
        if day_daily.empty:
            continue
        day_daily_row = day_daily.iloc[0]
        day_min = pd.DataFrame()
        if min5_df is not None:
            day_min = min5_df[min5_df["date"] == d].copy().sort_values("datetime").reset_index(drop=True)

        if day_min.empty:
            day_low = float(day_daily_row["low"])
            day_high = float(day_daily_row["high"])
            if day_low <= stop_price:
                exit_date, exit_price, mode_reason = _resolve_exit(stop_cfg["stop_exec_mode"], pd.Timestamp(d), None, day_daily_row, daily_df, day_min, tp_price)
                exit_reason = f'{stop_cfg["stop_base"]}|{stop_cfg["stop_exec_mode"]}|{mode_reason}'
                trigger_source = "daily_fallback"
                break
            if day_high >= tp_price:
                exit_date, exit_price, mode_reason = _resolve_exit(profile["exit_mode"], pd.Timestamp(d), None, day_daily_row, daily_df, day_min, tp_price)
                exit_reason = f'{profile["name"]}|{mode_reason}'
                trigger_source = "daily_fallback"
                break
            continue

        for _, bar in day_min.iterrows():
            bar_dt = pd.Timestamp(bar["datetime"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            if bar_low <= stop_price:
                exit_date, exit_price, mode_reason = _resolve_exit(stop_cfg["stop_exec_mode"], pd.Timestamp(d), bar_dt, day_daily_row, daily_df, day_min, tp_price)
                exit_reason = f'{stop_cfg["stop_base"]}|{stop_cfg["stop_exec_mode"]}|{mode_reason}'
                trigger_source = "5min"
                break
            if bar_high >= tp_price:
                exit_date, exit_price, mode_reason = _resolve_exit(profile["exit_mode"], pd.Timestamp(d), bar_dt, day_daily_row, daily_df, day_min, tp_price)
                exit_reason = f'{profile["name"]}|{mode_reason}'
                trigger_source = "5min"
                break
        if exit_reason is not None:
            break

    if exit_reason is None or exit_date is None or exit_price is None:
        final_date = pd.Timestamp(trade_window_dates[-1])
        final_row = daily_df[daily_df["date"] == final_date]
        if final_row.empty:
            return SimResult(trade=None, skipped=True, skip_reason="missing_forced_exit_day")
        exit_date = final_date
        exit_price = float(final_row.iloc[0]["close"])
        exit_reason = "max_hold_close"

    return_pct = float(exit_price) / entry_open - 1.0
    return SimResult(
        trade={
            "code": code,
            "signal_idx": int(signal_idx),
            "signal_date": pd.Timestamp(signal_date),
            "entry_date": pd.Timestamp(entry_date),
            "exit_date": pd.Timestamp(exit_date),
            "entry_price": float(entry_open),
            "exit_price": float(exit_price),
            "return_pct": float(return_pct),
            "hold_days": int(profile["hold_days"]),
            "exit_reason": str(exit_reason),
            "trigger_take_profit": float(tp_price),
            "trigger_stop_loss": float(stop_price),
            "stop_base_price": float(stop_price),
            "minute_source": "5min" if min5_df is not None else "missing_5min",
            "trigger_source": trigger_source,
            "profile_name": profile["name"],
            "profile_family": profile["family"],
            "tp_pct": float(profile["tp_pct"]),
            "weak_n": 0,
            "dd_pct": 0.0,
            "tp_arm_date": pd.NaT,
            "tp_arm_dt": pd.NaT,
            "stop_base": str(stop_cfg["stop_base"]),
            "stop_exec_mode": str(stop_cfg["stop_exec_mode"]),
            "sort_score": float(sort_score),
        }
    )


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    profiles = payload["profiles"]
    daily_path = DAILY_DIR / f"{code}.txt"
    if not daily_path.exists():
        skipped = []
        for item in rows:
            for stop_cfg in STOP_CONFIGS:
                for profile in profiles:
                    skipped.append(
                        {
                            "code": code,
                            "signal_date": item["signal_date"],
                            "entry_date": item["entry_date"],
                            "profile_name": profile["name"],
                            "stop_base": stop_cfg["stop_base"],
                            "stop_exec_mode": stop_cfg["stop_exec_mode"],
                            "reason": "missing_daily_series",
                        }
                    )
        return [], skipped

    daily_df = base.tp_mod.hybrid.base.load_daily_df(daily_path)
    if daily_df is None or daily_df.empty:
        return [], [{"code": code, "reason": "empty_daily_df"}]
    daily_df = daily_df[(daily_df["date"] < EXCLUDE_START) | (daily_df["date"] > EXCLUDE_END)].copy()

    min5_df = None
    min5_path = MIN5_DIR / f"{code}.txt"
    if min5_path.exists():
        loaded = base.tp_mod.hybrid.base.load_minute_df(min5_path)
        min5_df = base.tp_mod._prepare_min5_indicators(loaded) if loaded is not None and not loaded.empty else None

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in rows:
        for stop_cfg in STOP_CONFIGS:
            for profile in profiles:
                sim = simulate_one_trade_profile(
                    code=code,
                    signal_date=pd.Timestamp(item["signal_date"]),
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
                            "profile_name": profile["name"],
                            "stop_base": stop_cfg["stop_base"],
                            "stop_exec_mode": stop_cfg["stop_exec_mode"],
                            "reason": sim.skip_reason,
                        }
                    )
                    continue
                trades.append(sim.trade)
    return trades, skipped


def _summarize_signal_basket(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["profile_name", "stop_base", "stop_exec_mode", "hold_days"]
    for keys, g in trades.groupby(group_cols, sort=True):
        profile_name, stop_base, stop_exec_mode, hold_days = keys
        strategy = f"case_rank_phase1_v2|{stop_base}|{stop_exec_mode}|{profile_name}"
        row = base.tp_mod.hybrid.base.summarize_trades(g, strategy)
        row["profile_name"] = profile_name
        row["profile_family"] = str(g.iloc[0]["profile_family"])
        row["tp_pct"] = float(g.iloc[0]["tp_pct"])
        row["hold_days"] = int(hold_days)
        row["stop_base"] = stop_base
        row["stop_exec_mode"] = stop_exec_mode
        row["trigger_source_5min_ratio"] = float((g["trigger_source"] == "5min").mean())
        row["trigger_source_daily_ratio"] = float((g["trigger_source"] == "daily_fallback").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["annual_return_signal_basket", "final_equity_signal_basket", "profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    header = not path.exists()
    df.to_csv(path, index=False, encoding="utf-8-sig", mode="a", header=header)


def _load_close_series_payload(code: str) -> tuple[str, pd.Series | None]:
    path = DAILY_DIR / f"{code}.txt"
    if not path.exists():
        return code, None
    s = base.real_account._fast_load_close_series(path)
    if s is None or s.empty:
        return code, None
    return code, s


def _build_close_map_parallel_mp(codes: list[str], result_dir: Path, max_workers: int) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    relevant: dict[str, pd.Series] = {}
    all_dates: set[pd.Timestamp] = set()
    unique_codes = sorted(set(map(str, codes)))
    total = len(unique_codes)
    if total == 0:
        return pd.DatetimeIndex([]), {}
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max_workers) as pool:
        completed = 0
        for code, s in pool.imap_unordered(_load_close_series_payload, unique_codes, chunksize=32):
            completed += 1
            if s is not None and not s.empty:
                relevant[code] = s
                all_dates.update(pd.DatetimeIndex(s.index).tolist())
            if completed == 1 or completed % 100 == 0 or completed == total:
                update_progress(result_dir, "building_close_map", done_codes=int(completed), total_codes=int(total), loaded_codes=int(len(relevant)))
    market_dates = pd.DatetimeIndex(sorted(all_dates))
    close_map: dict[str, pd.Series] = {}
    total_series = len(relevant)
    for idx, (code, s) in enumerate(relevant.items(), start=1):
        close_map[code] = s.reindex(market_dates).ffill()
        if idx == 1 or idx % 100 == 0 or idx == total_series:
            update_progress(result_dir, "reindexing_close_map", done_series=int(idx), total_series=int(total_series), market_days=int(len(market_dates)))
    return market_dates, close_map


def run_search(result_dir: Path, source_csv: Path, date_limit: int, max_workers: int, buy_gap_limit: float, profile_batch_size: int) -> None:
    base._build_close_map_parallel = _build_close_map_parallel_mp
    result_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = result_dir / "batch_outputs"
    batch_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "loading_source", source_csv=str(source_csv), date_limit=date_limit, max_workers=max_workers, buy_gap_limit=buy_gap_limit, profile_batch_size=profile_batch_size)
    candidates = load_source_candidates(source_csv, date_limit)
    candidates.to_csv(result_dir / "source_candidates.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        raise RuntimeError("源候选为空")
    if "signal_idx" not in candidates.columns:
        candidates["signal_idx"] = -1

    grouped_payloads_template = []
    for code, g in candidates.groupby("code", sort=True):
        grouped_payloads_template.append(
            {
                "code": str(code),
                "rows": g[["signal_date", "entry_date", "signal_idx", "sort_score", "signal_open", "signal_close", "signal_low"]].to_dict("records"),
                "buy_gap_limit": float(buy_gap_limit),
            }
        )

    profile_batches = _chunks(PROFILE_DEFS, profile_batch_size)
    all_signal_summaries: list[pd.DataFrame] = []
    all_account_summaries: list[pd.DataFrame] = []

    total_jobs = len(candidates) * len(STOP_CONFIGS) * len(PROFILE_DEFS)
    done_jobs_cum = 0
    total_batches = len(profile_batches)

    for batch_idx, profile_batch in enumerate(profile_batches, start=1):
        batch_name = f"batch_{batch_idx:03d}"
        batch_result_dir = result_dir / batch_name
        batch_result_dir.mkdir(parents=True, exist_ok=True)
        batch_trades_path = batch_dir / f"{batch_name}_trades.csv"
        batch_skipped_path = batch_dir / f"{batch_name}_skipped.csv"
        payloads = [{**item, "profiles": profile_batch} for item in grouped_payloads_template]
        trades_rows: list[dict[str, Any]] = []
        skipped_rows: list[dict[str, Any]] = []
        ctx = mp.get_context("fork")
        progress_every = max(1, len(payloads) // 10)
        with ctx.Pool(processes=max_workers, maxtasksperchild=64) as pool:
            completed = 0
            for code, trades_part, skipped_part in pool.imap_unordered(
                _simulate_payload_wrapper,
                payloads,
                chunksize=1,
            ):
                completed += 1
                trades_rows.extend(trades_part)
                skipped_rows.extend(skipped_part)
                if completed == 1 or completed % progress_every == 0 or completed == len(payloads):
                    update_progress(
                        result_dir,
                        "simulating_trades",
                        batch=batch_idx,
                        total_batches=total_batches,
                        done_codes=completed,
                        total_codes=len(payloads),
                        done_jobs=done_jobs_cum + len(trades_rows) + len(skipped_rows),
                        total_jobs=total_jobs,
                        last_code=code,
                    )

        trades = pd.DataFrame(trades_rows)
        skipped = pd.DataFrame(skipped_rows)
        if not trades.empty:
            trades = trades.sort_values(["profile_name", "stop_base", "stop_exec_mode", "signal_date", "code"]).reset_index(drop=True)
            trades.to_csv(batch_trades_path, index=False, encoding="utf-8-sig")
        if not skipped.empty:
            skipped = skipped.sort_values(["profile_name", "stop_base", "stop_exec_mode", "signal_date", "code"], na_position="last").reset_index(drop=True)
            skipped.to_csv(batch_skipped_path, index=False, encoding="utf-8-sig")

        done_jobs_cum += len(trades_rows) + len(skipped_rows)
        signal_summary = _summarize_signal_basket(trades)
        if not signal_summary.empty:
            signal_summary["batch"] = batch_name
            all_signal_summaries.append(signal_summary)
            _append_csv(signal_summary, result_dir / "signal_basket_summary.csv")

        account_summary = base._summarize_account(trades, batch_result_dir, max_workers=max_workers) if not trades.empty else pd.DataFrame()
        if not account_summary.empty:
            account_summary["batch"] = batch_name
            all_account_summaries.append(account_summary)
            _append_csv(account_summary, result_dir / "account_summary.csv")

        update_progress(
            result_dir,
            "batch_finished",
            batch=batch_idx,
            total_batches=total_batches,
            batch_profiles=len(profile_batch),
            batch_trade_count=int(len(trades)),
            batch_skipped_count=int(len(skipped)),
            done_jobs=done_jobs_cum,
            total_jobs=total_jobs,
        )

    signal_summary_full = pd.concat(all_signal_summaries, ignore_index=True) if all_signal_summaries else pd.DataFrame()
    if not signal_summary_full.empty:
        signal_summary_full = signal_summary_full.sort_values(
            ["annual_return_signal_basket", "final_equity_signal_basket", "profit_factor"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        signal_summary_full.to_csv(result_dir / "signal_basket_summary.csv", index=False, encoding="utf-8-sig")

    account_summary_full = pd.concat(all_account_summaries, ignore_index=True) if all_account_summaries else pd.DataFrame()
    if not account_summary_full.empty:
        account_summary_full = account_summary_full.sort_values(
            ["annual_return", "final_equity", "max_drawdown", "sharpe", "calmar"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        account_summary_full.to_csv(result_dir / "account_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "phase": "phase1_fixed_exit_baseline_v2",
            "source_candidates": str(source_csv),
            "fixed_buy_model": "case_rank_lgbm_top20_daily_stream",
            "buy_gap_limit": buy_gap_limit,
            "exclude_window": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
            "max_hold_days_list": MAX_HOLD_DAYS_LIST,
            "tp_levels": TP_LEVELS,
            "stop_configs": STOP_CONFIGS,
            "tp_modes": ["next_5min_close", "same_day_close", "next_day_open", "limit_same_day"],
            "same_bar_priority": "hard_stop_then_fixed_tp",
            "profile_batch_size": profile_batch_size,
            "search_note": "narrowed_grid_using_prior_relative_results",
        },
        "best_signal_basket_profile": signal_summary_full.iloc[0].to_dict() if not signal_summary_full.empty else {},
        "best_account_profile": account_summary_full.iloc[0].to_dict() if not account_summary_full.empty else {},
        "signal_profile_count": int(len(signal_summary_full)),
        "account_profile_count": int(len(account_summary_full)),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished")


def _simulate_payload_wrapper(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    trades, skipped = simulate_code_bundle(payload)
    return str(payload["code"]), trades, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 v2: 缩窄固定退出基线搜索，批次并行落盘")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--source-csv", type=str, default=str(SOURCE_CANDIDATES))
    parser.add_argument("--buy-gap-limit", type=float, default=BUY_GAP_LIMIT)
    parser.add_argument("--date-limit", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--profile-batch-size", type=int, default=PROFILE_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_case_rank_phase1_fixed_exit_search_v2_{args.mode}_{timestamp}"
    date_limit = int(args.date_limit)
    if args.mode == "full":
        date_limit = 0
    try:
        run_search(
            result_dir=output_dir,
            source_csv=Path(args.source_csv),
            date_limit=date_limit,
            max_workers=int(args.max_workers),
            buy_gap_limit=float(args.buy_gap_limit),
            profile_batch_size=int(args.profile_batch_size),
        )
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
