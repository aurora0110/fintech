from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
PHASE0_RESULT_DIR = RESULT_ROOT / "brick_case_rank_daily_stream_v2_full_20260328_r1"
SOURCE_CANDIDATES = PHASE0_RESULT_DIR / "daily_top20_candidates.csv"
BASE_SCRIPT_PATH = ROOT / "utils" / "brick_optimize" / "run_brick_case_rank_final_spec_search_v1_20260327.py"
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
BUY_GAP_LIMIT = 0.04
PROFILE_BATCH_SIZE = 18
ATR_PERIODS = [10, 14, 20]
HOLD_DAYS = [3, 4, 5]
EXIT_MODES = ["next_day_open", "limit_same_day"]
STOP_CFG = {"stop_base": "entry_low", "stop_exec_mode": "same_day_close"}
K_MIN = 1.0
K_MAX = 4.0
K_STEP = 0.5
LOWER_CAP = 0.25
UPPER_CAP = 8.0


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT_PATH, "brick_case_rank_phase2_atr_v2_base_mod")


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


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")


def _normalize_skipped_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required_cols = ["profile_name", "signal_date", "entry_date", "code", "reason"]
    out = df.copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _chunks(seq: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _pick_smoke_dates(dates: list[str], count: int) -> list[str]:
    if count <= 0 or len(dates) <= count:
        return dates
    idxs = np.linspace(0, len(dates) - 1, num=count, dtype=int)
    return [dates[i] for i in sorted(set(idxs.tolist()))]


def load_source_candidates(source_csv: Path, date_limit: int) -> pd.DataFrame:
    cols = pd.read_csv(source_csv, nrows=0).columns.tolist()
    parse_dates = [c for c in ["signal_date", "entry_date", "exit_date"] if c in cols]
    df = pd.read_csv(source_csv, parse_dates=parse_dates)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df[(df["signal_date"] < EXCLUDE_START) | (df["signal_date"] > EXCLUDE_END)].copy()
    df["code"] = df["code"].map(base._resolve_daily_stem)
    if "sort_score" not in df.columns:
        df["sort_score"] = pd.to_numeric(df.get("model_score", 0.0), errors="coerce").fillna(0.0)
    if "signal_idx" not in df.columns:
        df["signal_idx"] = -1
    df = df.sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    if date_limit > 0:
        keep_dates = _pick_smoke_dates(sorted(df["signal_date"].dt.strftime("%Y-%m-%d").unique()), date_limit)
        df = df[df["signal_date"].dt.strftime("%Y-%m-%d").isin(keep_dates)].copy()
    return df.reset_index(drop=True)


def _compute_atr_columns(daily_df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    df = daily_df.sort_values("date").copy()
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    for period in periods:
        atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        df[f"atr_{period}"] = atr
    return df


def _build_k_values(k_min: float, k_max: float, step: float) -> list[float]:
    values: list[float] = []
    x = k_min
    while x <= k_max + 1e-12:
        values.append(round(x, 4))
        x += step
    return values


def _build_profiles(k_values: list[float]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for atr_period in ATR_PERIODS:
        for k in k_values:
            for hold_days in HOLD_DAYS:
                for exit_mode in EXIT_MODES:
                    tag = "nextopen" if exit_mode == "next_day_open" else "limitsameday"
                    profiles.append(
                        {
                            "name": f"atr_tp_p{atr_period}_k{k:.4f}_h{hold_days}_{tag}",
                            "family": "atr_fixed_tp",
                            "atr_period": int(atr_period),
                            "k_value": float(k),
                            "hold_days": int(hold_days),
                            "exit_mode": str(exit_mode),
                        }
                    )
    return profiles


def _atr_pct_for_signal(signal_row: pd.Series, atr_period: int) -> float | None:
    atr_value = float(signal_row.get(f"atr_{atr_period}", np.nan))
    signal_close = float(signal_row.get("close", np.nan))
    if not np.isfinite(atr_value) or not np.isfinite(signal_close) or atr_value <= 0 or signal_close <= 0:
        return None
    return float(atr_value / signal_close)


def _resolve_tp_exit(exit_mode: str, trigger_date: pd.Timestamp, day_daily_row: pd.Series, daily_df: pd.DataFrame, tp_price: float) -> tuple[pd.Timestamp, float, str]:
    if exit_mode == "limit_same_day":
        return pd.Timestamp(trigger_date), float(tp_price), "limit_same_day"
    if exit_mode == "next_day_open":
        next_rows = daily_df[daily_df["date"] > trigger_date]
        if next_rows.empty:
            return pd.Timestamp(trigger_date), float(day_daily_row["close"]), "fallback_same_day_close"
        next_row = next_rows.iloc[0]
        return pd.Timestamp(next_row["date"]), float(next_row["open"]), "next_day_open"
    raise ValueError(f"未知 exit_mode: {exit_mode}")


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    profiles = payload["profiles"]
    daily_path = base.DAILY_DIR / f"{code}.txt"
    if not daily_path.exists():
        return code, [], [{"code": code, "reason": "missing_daily_series"}]

    daily_df = base.tp_mod.hybrid.base.load_daily_df(daily_path)
    if daily_df is None or daily_df.empty:
        return code, [], [{"code": code, "reason": "empty_daily_df"}]
    daily_df = daily_df[(daily_df["date"] < EXCLUDE_START) | (daily_df["date"] > EXCLUDE_END)].copy()
    daily_df = _compute_atr_columns(daily_df, ATR_PERIODS)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_indexed = daily_df.set_index("date")

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for item in rows:
        signal_date = pd.Timestamp(item["signal_date"])
        entry_date = pd.Timestamp(item["entry_date"])
        signal_rows = daily_indexed.loc[[signal_date]] if signal_date in daily_indexed.index else pd.DataFrame()
        entry_rows = daily_indexed.loc[[entry_date]] if entry_date in daily_indexed.index else pd.DataFrame()
        if len(signal_rows) == 0 or len(entry_rows) == 0:
            skipped.append({"code": code, "signal_date": item["signal_date"], "entry_date": item["entry_date"], "reason": "missing_daily_row"})
            continue
        signal_row = signal_rows.iloc[0]
        entry_row = entry_rows.iloc[0]
        entry_open = float(entry_row["open"])
        entry_close = float(entry_row["close"])
        entry_low = float(entry_row["low"])
        signal_open = float(item["signal_open"])
        signal_close = float(item["signal_close"])
        signal_low = float(item["signal_low"])
        if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(signal_close) or signal_close <= 0:
            skipped.append({"code": code, "signal_date": item["signal_date"], "entry_date": item["entry_date"], "reason": "invalid_entry_or_signal"})
            continue
        if entry_open / signal_close - 1.0 >= float(payload["buy_gap_limit"]):
            skipped.append({"code": code, "signal_date": item["signal_date"], "entry_date": item["entry_date"], "reason": "gap_gte_4pct"})
            continue

        stop_price = base._resolve_stop_base(STOP_CFG["stop_base"], signal_open, signal_close, signal_low, entry_open, entry_close, entry_low)
        trade_window_dates, eligible_exit_dates = base._build_trade_window(daily_df, entry_date, int(max(HOLD_DAYS)))
        if not trade_window_dates:
            skipped.append({"code": code, "signal_date": item["signal_date"], "entry_date": item["entry_date"], "reason": "no_trade_window"})
            continue

        for profile in profiles:
            atr_pct = _atr_pct_for_signal(signal_row, int(profile["atr_period"]))
            if atr_pct is None:
                skipped.append(
                    {
                        "code": code,
                        "signal_date": item["signal_date"],
                        "entry_date": item["entry_date"],
                        "profile_name": profile["name"],
                        "reason": "invalid_atr_pct",
                    }
                )
                continue
            tp_pct = float(profile["k_value"]) * atr_pct
            tp_price = entry_open * (1.0 + tp_pct)
            _, eligible_exit_dates = base._build_trade_window(daily_df, entry_date, int(profile["hold_days"]))
            exit_date: pd.Timestamp | None = None
            exit_price: float | None = None
            exit_reason: str | None = None
            trigger_source = "max_hold"

            for d in eligible_exit_dates:
                day_rows = daily_indexed.loc[[d]] if d in daily_indexed.index else pd.DataFrame()
                if len(day_rows) == 0:
                    continue
                day_row = day_rows.iloc[0]
                day_low = float(day_row["low"])
                day_high = float(day_row["high"])

                if day_low <= stop_price:
                    exit_date = pd.Timestamp(d)
                    exit_price = float(day_row["close"])
                    exit_reason = f'{STOP_CFG["stop_base"]}|{STOP_CFG["stop_exec_mode"]}|same_day_close'
                    trigger_source = "daily"
                    break
                if day_high >= tp_price:
                    exit_date, exit_price, mode_reason = _resolve_tp_exit(str(profile["exit_mode"]), pd.Timestamp(d), day_row, daily_df, tp_price)
                    exit_reason = f'{profile["name"]}|{mode_reason}'
                    trigger_source = "daily"
                    break

            if exit_reason is None:
                final_date = pd.Timestamp(eligible_exit_dates[-1] if eligible_exit_dates else trade_window_dates[-1])
                final_row = daily_indexed.loc[[final_date]] if final_date in daily_indexed.index else pd.DataFrame()
                if len(final_row) == 0:
                    skipped.append(
                        {
                            "code": code,
                            "signal_date": item["signal_date"],
                            "entry_date": item["entry_date"],
                            "profile_name": profile["name"],
                            "reason": "missing_forced_exit_day",
                        }
                    )
                    continue
                exit_date = final_date
                exit_price = float(final_row.iloc[0]["close"])
                exit_reason = "max_hold_close"

            return_pct = float(exit_price) / entry_open - 1.0
            trades.append(
                {
                    "code": code,
                    "signal_idx": int(item.get("signal_idx", -1)),
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
                    "minute_source": "daily_only",
                    "trigger_source": trigger_source,
                    "profile_name": profile["name"],
                    "profile_family": "atr_fixed_tp",
                    "tp_pct": float(tp_pct),
                    "weak_n": 0,
                    "dd_pct": 0.0,
                    "tp_arm_date": pd.NaT,
                    "tp_arm_dt": pd.NaT,
                    "stop_base": STOP_CFG["stop_base"],
                    "stop_exec_mode": STOP_CFG["stop_exec_mode"],
                    "sort_score": float(item["sort_score"]),
                    "atr_period": int(profile["atr_period"]),
                    "k_value": float(profile["k_value"]),
                    "atr_pct": float(atr_pct),
                    "selected_tp_pct": float(tp_pct),
                    "selected_hold_days": int(profile["hold_days"]),
                    "exit_mode": str(profile["exit_mode"]),
                }
            )

    return code, trades, skipped


def _summarize_signal_basket(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["profile_name", "stop_base", "stop_exec_mode"]
    for keys, g in trades.groupby(group_cols, sort=True):
        profile_name, stop_base, stop_exec_mode = keys
        strategy = f"case_rank_phase2_atr_v2|{stop_base}|{stop_exec_mode}|{profile_name}"
        row = base.tp_mod.hybrid.base.summarize_trades(g, strategy)
        first = g.iloc[0]
        row.update(
            {
                "profile_name": profile_name,
                "profile_family": "atr_fixed_tp",
                "atr_period": int(first["atr_period"]),
                "k_value": float(first["k_value"]),
                "hold_days": int(first["selected_hold_days"]),
                "exit_mode": str(first["exit_mode"]),
                "avg_tp_pct": float(pd.to_numeric(g["selected_tp_pct"], errors="coerce").mean()),
                "median_tp_pct": float(pd.to_numeric(g["selected_tp_pct"], errors="coerce").median()),
                "avg_atr_pct": float(pd.to_numeric(g["atr_pct"], errors="coerce").mean()),
                "stop_base": stop_base,
                "stop_exec_mode": stop_exec_mode,
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["annual_return_signal_basket", "final_equity_signal_basket", "profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _load_close_series_payload(code: str) -> tuple[str, pd.Series | None]:
    path = base.DAILY_DIR / f"{code}.txt"
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
    with ctx.Pool(processes=max_workers, maxtasksperchild=128) as pool:
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


def _summarize_account(trades: pd.DataFrame, result_dir: Path, max_workers: int) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base._build_close_map_parallel = _build_close_map_parallel_mp
    account_summary = base._summarize_account(trades, result_dir, max_workers=max_workers)
    extra = (
        trades.groupby(["profile_name", "stop_base", "stop_exec_mode"], sort=True)
        .agg(
            atr_period=("atr_period", "first"),
            k_value=("k_value", "first"),
            hold_days=("selected_hold_days", "first"),
            exit_mode=("exit_mode", "first"),
            avg_tp_pct=("selected_tp_pct", "mean"),
            median_tp_pct=("selected_tp_pct", "median"),
            avg_atr_pct=("atr_pct", "mean"),
        )
        .reset_index()
    )
    return account_summary.merge(extra, on=["profile_name", "stop_base", "stop_exec_mode"], how="left")


def _best_k_edge(summary: dict[str, Any], k_min: float, k_max: float) -> str:
    best = summary.get("best_account_profile", {}) or {}
    if not best:
        return "unknown"
    best_k = round(float(best.get("k_value", np.nan)), 4)
    if np.isfinite(best_k):
        if abs(best_k - round(k_min, 4)) < 1e-9:
            return "lower"
        if abs(best_k - round(k_max, 4)) < 1e-9:
            return "upper"
    return "inner"


def run_round(result_dir: Path, source_csv: Path, k_min: float, k_max: float, k_step: float, date_limit: int, max_workers: int, profile_batch_size: int) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = result_dir / "batch_outputs"
    batch_dir.mkdir(parents=True, exist_ok=True)

    k_values = _build_k_values(k_min, k_max, k_step)
    profiles = _build_profiles(k_values)
    update_progress(
        result_dir,
        "loading_source",
        source_csv=str(source_csv),
        k_min=k_min,
        k_max=k_max,
        k_step=k_step,
        k_count=len(k_values),
        profile_count=len(profiles),
        date_limit=date_limit,
        max_workers=max_workers,
        profile_batch_size=profile_batch_size,
    )

    candidates = load_source_candidates(source_csv, date_limit)
    candidates.to_csv(result_dir / "source_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"k_value": k_values}).to_csv(result_dir / "k_grid.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        raise RuntimeError("源候选为空")

    grouped_payloads_template = []
    for code, g in candidates.groupby("code", sort=True):
        grouped_payloads_template.append(
            {
                "code": str(code),
                "rows": g[["signal_date", "entry_date", "signal_idx", "sort_score", "signal_open", "signal_close", "signal_low"]].to_dict("records"),
                "buy_gap_limit": float(BUY_GAP_LIMIT),
            }
        )

    profile_batches = _chunks(profiles, profile_batch_size)
    total_batches = len(profile_batches)
    total_jobs = len(candidates) * len(profiles)
    done_jobs_cum = 0
    all_signal_summaries: list[pd.DataFrame] = []
    all_account_summaries: list[pd.DataFrame] = []

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
        with ctx.Pool(processes=max_workers, maxtasksperchild=128) as pool:
            completed = 0
            for code, trades_part, skipped_part in pool.imap_unordered(simulate_code_bundle, payloads, chunksize=1):
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
            trades = trades.sort_values(["profile_name", "signal_date", "code"]).reset_index(drop=True)
            trades.to_csv(batch_trades_path, index=False, encoding="utf-8-sig")
        if not skipped.empty:
            skipped = _normalize_skipped_df(skipped)
            skipped = skipped.sort_values(["profile_name", "signal_date", "code"], na_position="last").reset_index(drop=True)
            skipped.to_csv(batch_skipped_path, index=False, encoding="utf-8-sig")

        done_jobs_cum += len(trades_rows) + len(skipped_rows)
        signal_summary = _summarize_signal_basket(trades)
        if not signal_summary.empty:
            signal_summary["batch"] = batch_name
            all_signal_summaries.append(signal_summary)
            _append_csv(signal_summary, result_dir / "signal_basket_summary.csv")

        account_summary = _summarize_account(trades, batch_result_dir, max_workers=max_workers) if not trades.empty else pd.DataFrame()
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
            "phase": "phase2_atr_v2",
            "source_candidates": str(source_csv),
            "fixed_buy_model": "case_rank_lgbm_top20_daily_stream",
            "buy_gap_limit": BUY_GAP_LIMIT,
            "exclude_window": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
            "fixed_stop_cfg": STOP_CFG,
            "exit_modes": EXIT_MODES,
            "atr_periods": ATR_PERIODS,
            "hold_days": HOLD_DAYS,
            "k_min": k_min,
            "k_max": k_max,
            "k_step": k_step,
            "profile_batch_size": profile_batch_size,
            "search_note": "daily_only_trigger_to_replace_slow_5min_path",
        },
        "best_signal_basket_profile": signal_summary_full.iloc[0].to_dict() if not signal_summary_full.empty else {},
        "best_account_profile": account_summary_full.iloc[0].to_dict() if not account_summary_full.empty else {},
        "signal_profile_count": int(len(signal_summary_full)),
        "account_profile_count": int(len(account_summary_full)),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished")
    return summary


def run_search(output_root: Path, source_csv: Path, date_limit: int, max_workers: int, profile_batch_size: int, k_min: float, k_max: float, k_step: float, lower_cap: float, upper_cap: float) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rounds: list[dict[str, Any]] = []
    cur_min = k_min
    cur_max = k_max
    round_idx = 1
    while True:
        round_dir = output_root / f"round_{round_idx:02d}_k_{cur_min:.4f}_{cur_max:.4f}"
        summary = run_round(round_dir, source_csv, cur_min, cur_max, k_step, date_limit, max_workers, profile_batch_size)
        edge = _best_k_edge(summary, cur_min, cur_max)
        rounds.append({"round": round_idx, "k_min": cur_min, "k_max": cur_max, "edge": edge, "best_account_profile": summary.get("best_account_profile", {})})
        if edge == "lower" and cur_min > lower_cap + 1e-12:
            cur_min = max(lower_cap, round(cur_min - k_step, 4))
            round_idx += 1
            continue
        if edge == "upper" and cur_max < upper_cap - 1e-12:
            cur_max = min(upper_cap, round(cur_max + k_step, 4))
            round_idx += 1
            continue
        final = {
            "mode": "phase2_atr_v2",
            "rounds": rounds,
            "final_round": summary,
        }
        (output_root / "summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 ATR v2：日线级动态止盈搜索，避免慢速 5min 路径")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--source-csv", type=str, default=str(SOURCE_CANDIDATES))
    parser.add_argument("--date-limit", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--profile-batch-size", type=int, default=PROFILE_BATCH_SIZE)
    parser.add_argument("--k-min", type=float, default=K_MIN)
    parser.add_argument("--k-max", type=float, default=K_MAX)
    parser.add_argument("--k-step", type=float, default=K_STEP)
    parser.add_argument("--lower-cap", type=float, default=LOWER_CAP)
    parser.add_argument("--upper-cap", type=float, default=UPPER_CAP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_case_rank_phase2_atr_search_v2_{args.mode}_{timestamp}_r1"
    date_limit = args.date_limit if args.mode == "smoke" else 0
    try:
        run_search(
            output_root=output_dir,
            source_csv=Path(args.source_csv),
            date_limit=date_limit,
            max_workers=int(args.max_workers),
            profile_batch_size=int(args.profile_batch_size),
            k_min=float(args.k_min),
            k_max=float(args.k_max),
            k_step=float(args.k_step),
            lower_cap=float(args.lower_cap),
            upper_cap=float(args.upper_cap),
        )
    except BaseException as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
