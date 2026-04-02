from __future__ import annotations

import argparse
import importlib.util
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
SOURCE_CANDIDATES = ROOT / "results" / "brick_case_rank_model_search_v1_full_20260327_r1" / "best_model_top20_candidates.csv"


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "brick_case_rank_h3_half_profit_protect_base")
real_account = base.real_account

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
STOP_CONFIGS = list(base.STOP_CONFIGS)
WEAK_COUNTS = list(base.WEAK_COUNTS)
PULLBACK_PCTS = list(base.PULLBACK_PCTS)


@dataclass
class SimBundle:
    trades: list[dict[str, Any]]
    skipped: list[dict[str, Any]]


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


def build_tp_levels(max_tp: float) -> list[float]:
    levels: list[float] = []
    value = 0.03
    while value <= max_tp + 1e-12:
        levels.append(round(value, 4))
        value += 0.005
    return levels


def build_profiles(max_tp: float) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for tp_pct in build_tp_levels(max_tp):
        tag = f"{tp_pct:.4f}"
        profiles.extend(
            [
                {
                    "name": f"tp_next5close_{tag}_h3",
                    "family": "fixed_tp",
                    "tp_pct": tp_pct,
                    "hold_days": 3,
                    "exit_mode": "next_5min_close",
                },
                {
                    "name": f"tp_close_{tag}_h3",
                    "family": "fixed_tp",
                    "tp_pct": tp_pct,
                    "hold_days": 3,
                    "exit_mode": "same_day_close",
                },
                {
                    "name": f"tp_next_open_{tag}_h3",
                    "family": "fixed_tp",
                    "tp_pct": tp_pct,
                    "hold_days": 3,
                    "exit_mode": "next_day_open",
                },
            ]
        )
        for weak_n in WEAK_COUNTS:
            for exit_mode, exit_tag in [
                ("next_5min_open", "next5open"),
                ("next_5min_close", "next5close"),
                ("next_day_open", "nextopen"),
            ]:
                profiles.append(
                    {
                        "name": f"pp_weak_{exit_tag}_{tag}_n{weak_n}_h3",
                        "family": "profit_protect_weak",
                        "tp_pct": tp_pct,
                        "hold_days": 3,
                        "weak_n": weak_n,
                        "exit_mode": exit_mode,
                    }
                )
                profiles.append(
                    {
                        "name": f"pp_half_{exit_tag}_{tag}_n{weak_n}_h3",
                        "family": "profit_protect_half_nextopen",
                        "tp_pct": tp_pct,
                        "hold_days": 3,
                        "weak_n": weak_n,
                        "exit_mode": exit_mode,
                    }
                )
        for dd_pct in PULLBACK_PCTS:
            dd_tag = f"{dd_pct * 100:.1f}"
            for exit_mode, exit_tag in [
                ("next_5min_open", "next5open"),
                ("next_5min_close", "next5close"),
            ]:
                profiles.append(
                    {
                        "name": f"pp_dd_{exit_tag}_{tag}_m{dd_tag}_h3",
                        "family": "profit_protect_pullback",
                        "tp_pct": tp_pct,
                        "hold_days": 3,
                        "dd_pct": dd_pct,
                        "exit_mode": exit_mode,
                    }
                )
    return profiles


def _build_trade_window(daily_df: pd.DataFrame, entry_date: pd.Timestamp, max_hold_days: int) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    window_dates = [pd.Timestamp(x) for x in daily_df[daily_df["date"] >= entry_date]["date"].head(max_hold_days).tolist()]
    eligible_exit_dates = [d for d in window_dates if d > entry_date]
    return window_dates, eligible_exit_dates


def _resolve_stop_base(stop_base: str, signal_open: float, signal_close: float, signal_low: float, entry_open: float, entry_close: float, entry_low: float) -> float:
    if stop_base == "entry_min_oc":
        return min(entry_open, entry_close)
    if stop_base == "entry_low":
        return entry_low
    if stop_base == "signal_min_oc":
        return min(signal_open, signal_close)
    if stop_base == "signal_low":
        return signal_low
    raise ValueError(stop_base)


def _resolve_exit(exit_mode: str, trigger_date: pd.Timestamp, trigger_dt: pd.Timestamp | None, day_daily_row: pd.Series, daily_df: pd.DataFrame, day_min: pd.DataFrame) -> tuple[pd.Timestamp, float, str]:
    return base._resolve_exit(exit_mode, trigger_date, trigger_dt, day_daily_row, daily_df, day_min)


def _new_group_id(code: str, signal_date: pd.Timestamp, profile_name: str, stop_base: str, stop_exec_mode: str) -> str:
    return f"{code}|{pd.Timestamp(signal_date).date()}|{profile_name}|{stop_base}|{stop_exec_mode}"


def _trade_common_fields(
    code: str,
    signal_idx: int,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    entry_open: float,
    sort_score: float,
    profile: dict[str, Any],
    stop_cfg: dict[str, Any],
    stop_price: float,
    tp_price: float,
    tp_arm_date: pd.Timestamp | None,
    tp_arm_dt: pd.Timestamp | None,
    group_id: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "position_group": group_id,
        "signal_idx": int(signal_idx),
        "signal_date": pd.Timestamp(signal_date),
        "entry_date": pd.Timestamp(entry_date),
        "entry_price": float(entry_open),
        "hold_days": int(profile["hold_days"]),
        "trigger_take_profit": float(tp_price),
        "trigger_stop_loss": float(stop_price),
        "stop_base_price": float(stop_price),
        "profile_name": profile["name"],
        "profile_family": profile["family"],
        "tp_pct": float(profile["tp_pct"]),
        "weak_n": int(profile.get("weak_n", 0)),
        "dd_pct": float(profile.get("dd_pct", 0.0)),
        "tp_arm_date": pd.Timestamp(tp_arm_date) if tp_arm_date is not None else pd.NaT,
        "tp_arm_dt": pd.Timestamp(tp_arm_dt) if tp_arm_dt is not None else pd.NaT,
        "stop_base": str(stop_cfg["stop_base"]),
        "stop_exec_mode": str(stop_cfg["stop_exec_mode"]),
        "sort_score": float(sort_score),
    }


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
) -> SimBundle:
    entry_rows = daily_df[daily_df["date"] == entry_date]
    signal_rows = daily_df[daily_df["date"] == signal_date]
    if entry_rows.empty or signal_rows.empty:
        return SimBundle(trades=[], skipped=[{"code": code, "reason": "missing_daily_row"}])
    entry_row = entry_rows.iloc[0]
    entry_open = float(entry_row["open"])
    entry_close = float(entry_row["close"])
    entry_low = float(entry_row["low"])
    if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(signal_close) or signal_close <= 0:
        return SimBundle(trades=[], skipped=[{"code": code, "reason": "invalid_entry_or_signal"}])
    if entry_open / signal_close - 1.0 >= buy_gap_limit:
        return SimBundle(trades=[], skipped=[{"code": code, "reason": "gap_gte_4pct"}])

    stop_price = _resolve_stop_base(stop_cfg["stop_base"], signal_open, signal_close, signal_low, entry_open, entry_close, entry_low)
    tp_price = entry_open * (1.0 + float(profile["tp_pct"]))
    trade_window_dates, eligible_exit_dates = _build_trade_window(daily_df, entry_date, int(profile["hold_days"]))
    if not trade_window_dates:
        return SimBundle(trades=[], skipped=[{"code": code, "reason": "no_trade_window"}])

    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    trigger_source = "max_hold"
    armed = False
    tp_arm_date: pd.Timestamp | None = None
    tp_arm_dt: pd.Timestamp | None = None
    peak_price: float | None = None
    weak_count = 0
    prev_ref_low: float | None = None
    protect_trigger_date: pd.Timestamp | None = None
    protect_trigger_dt: pd.Timestamp | None = None
    protect_trigger_reason: str | None = None

    group_id = _new_group_id(code, signal_date, profile["name"], stop_cfg["stop_base"], stop_cfg["stop_exec_mode"])

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
            day_close = float(day_daily_row["close"])
            if day_low <= stop_price:
                exit_date, exit_price, mode_reason = _resolve_exit(
                    stop_cfg["stop_exec_mode"], pd.Timestamp(d), None, day_daily_row, daily_df, day_min
                )
                exit_reason = f'{stop_cfg["stop_base"]}|{stop_cfg["stop_exec_mode"]}|{mode_reason}'
                trigger_source = "daily_fallback"
                break

            if profile["family"] == "fixed_tp":
                if day_high >= tp_price:
                    exit_date, exit_price, mode_reason = _resolve_exit(
                        profile["exit_mode"], pd.Timestamp(d), None, day_daily_row, daily_df, day_min
                    )
                    exit_reason = f'{profile["name"]}|{mode_reason}'
                    trigger_source = "daily_fallback"
                    break
            else:
                if (not armed) and day_high >= tp_price:
                    armed = True
                    tp_arm_date = pd.Timestamp(d)
                    peak_price = day_high
                    prev_ref_low = day_low
                    trigger_source = "daily_fallback"
                elif armed:
                    peak_price = max(float(peak_price or -np.inf), day_high)
                    if profile["family"] in {"profit_protect_weak", "profit_protect_half_nextopen"}:
                        is_weak = bool(prev_ref_low is not None and np.isfinite(prev_ref_low) and day_close < prev_ref_low)
                        weak_count = weak_count + 1 if is_weak else 0
                        prev_ref_low = day_low
                        if weak_count >= int(profile["weak_n"]):
                            protect_trigger_date = pd.Timestamp(d)
                            protect_trigger_dt = None
                            protect_trigger_reason = "daily_fallback"
                            if profile["family"] == "profit_protect_half_nextopen":
                                break
                            exit_date, exit_price, mode_reason = _resolve_exit(
                                profile["exit_mode"], pd.Timestamp(d), None, day_daily_row, daily_df, day_min
                            )
                            exit_reason = f'{profile["name"]}|{mode_reason}'
                            trigger_source = "daily_fallback"
                            break
                    else:
                        dd_pct = float(profile["dd_pct"])
                        if peak_price is not None and day_low <= peak_price * (1.0 - dd_pct):
                            exit_date, exit_price, mode_reason = _resolve_exit(
                                profile["exit_mode"], pd.Timestamp(d), None, day_daily_row, daily_df, day_min
                            )
                            exit_reason = f'{profile["name"]}|{mode_reason}'
                            trigger_source = "daily_fallback"
                            break
            if protect_trigger_reason is not None:
                break
            continue

        for _, bar in day_min.iterrows():
            bar_dt = pd.Timestamp(bar["datetime"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            if bar_low <= stop_price:
                exit_date, exit_price, mode_reason = _resolve_exit(
                    stop_cfg["stop_exec_mode"], pd.Timestamp(d), bar_dt, day_daily_row, daily_df, day_min
                )
                exit_reason = f'{stop_cfg["stop_base"]}|{stop_cfg["stop_exec_mode"]}|{mode_reason}'
                trigger_source = "5min"
                break

            if profile["family"] == "fixed_tp":
                if bar_high >= tp_price:
                    exit_date, exit_price, mode_reason = _resolve_exit(
                        profile["exit_mode"], pd.Timestamp(d), bar_dt, day_daily_row, daily_df, day_min
                    )
                    exit_reason = f'{profile["name"]}|{mode_reason}'
                    trigger_source = "5min"
                    break
                continue

            if (not armed) and bar_high >= tp_price:
                armed = True
                tp_arm_date = pd.Timestamp(d)
                tp_arm_dt = bar_dt
                peak_price = bar_high
                prev_ref_low = bar_low
                trigger_source = "5min"
                continue

            if armed:
                peak_price = max(float(peak_price or -np.inf), bar_high)
                if profile["family"] in {"profit_protect_weak", "profit_protect_half_nextopen"}:
                    is_weak = bool(prev_ref_low is not None and np.isfinite(prev_ref_low) and bar_close < prev_ref_low)
                    weak_count = weak_count + 1 if is_weak else 0
                    prev_ref_low = bar_low
                    if weak_count >= int(profile["weak_n"]):
                        protect_trigger_date = pd.Timestamp(d)
                        protect_trigger_dt = bar_dt
                        protect_trigger_reason = "5min"
                        if profile["family"] == "profit_protect_half_nextopen":
                            break
                        exit_date, exit_price, mode_reason = _resolve_exit(
                            profile["exit_mode"], pd.Timestamp(d), bar_dt, day_daily_row, daily_df, day_min
                        )
                        exit_reason = f'{profile["name"]}|{mode_reason}'
                        trigger_source = "5min"
                        break
                else:
                    dd_pct = float(profile["dd_pct"])
                    if peak_price is not None and bar_low <= peak_price * (1.0 - dd_pct):
                        exit_date, exit_price, mode_reason = _resolve_exit(
                            profile["exit_mode"], pd.Timestamp(d), bar_dt, day_daily_row, daily_df, day_min
                        )
                        exit_reason = f'{profile["name"]}|{mode_reason}'
                        trigger_source = "5min"
                        break

        if exit_reason is not None or protect_trigger_reason is not None:
            break

    if profile["family"] == "profit_protect_half_nextopen" and protect_trigger_reason is not None and protect_trigger_date is not None:
        trigger_day = daily_df[daily_df["date"] == protect_trigger_date]
        if trigger_day.empty:
            return SimBundle(trades=[], skipped=[{"code": code, "reason": "missing_protect_trigger_day"}])
        trigger_day_row = trigger_day.iloc[0]
        trigger_day_min = pd.DataFrame()
        if min5_df is not None:
            trigger_day_min = min5_df[min5_df["date"] == protect_trigger_date].copy().sort_values("datetime").reset_index(drop=True)

        half1_exit_date, half1_exit_price, half1_mode = _resolve_exit(
            profile["exit_mode"], protect_trigger_date, protect_trigger_dt, trigger_day_row, daily_df, trigger_day_min
        )
        half2_exit_date, half2_exit_price, half2_mode = _resolve_exit(
            "next_day_open", protect_trigger_date, protect_trigger_dt, trigger_day_row, daily_df, trigger_day_min
        )

        common = _trade_common_fields(
            code,
            signal_idx,
            signal_date,
            entry_date,
            entry_open,
            sort_score,
            profile,
            stop_cfg,
            stop_price,
            tp_price,
            tp_arm_date,
            tp_arm_dt,
            group_id,
        )
        trades = []
        for leg_suffix, leg_frac, leg_exit_date, leg_exit_price, leg_reason, leg_source in [
            ("a", 0.5, half1_exit_date, half1_exit_price, f'{profile["name"]}|{half1_mode}', protect_trigger_reason),
            ("b", 0.5, half2_exit_date, half2_exit_price, f'{profile["name"]}|half_nextopen_{half2_mode}', "daily_followup"),
        ]:
            trade = dict(common)
            trade.update(
                {
                    "position_leg": leg_suffix,
                    "position_frac": leg_frac,
                    "exit_date": pd.Timestamp(leg_exit_date),
                    "exit_price": float(leg_exit_price),
                    "return_pct": float(leg_exit_price) / entry_open - 1.0,
                    "exit_reason": leg_reason,
                    "trigger_source": leg_source,
                }
            )
            trades.append(trade)
        return SimBundle(trades=trades, skipped=[])

    if exit_reason is None or exit_date is None or exit_price is None:
        final_date = pd.Timestamp(trade_window_dates[-1])
        final_row = daily_df[daily_df["date"] == final_date]
        if final_row.empty:
            return SimBundle(trades=[], skipped=[{"code": code, "reason": "missing_forced_exit_day"}])
        exit_date = final_date
        exit_price = float(final_row.iloc[0]["close"])
        exit_reason = "max_hold_close"

    common = _trade_common_fields(
        code,
        signal_idx,
        signal_date,
        entry_date,
        entry_open,
        sort_score,
        profile,
        stop_cfg,
        stop_price,
        tp_price,
        tp_arm_date,
        tp_arm_dt,
        group_id,
    )
    trade = dict(common)
    trade.update(
        {
            "position_leg": "full",
            "position_frac": 1.0,
            "exit_date": pd.Timestamp(exit_date),
            "exit_price": float(exit_price),
            "return_pct": float(exit_price) / entry_open - 1.0,
            "exit_reason": str(exit_reason),
            "trigger_source": trigger_source,
        }
    )
    return SimBundle(trades=[trade], skipped=[])


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    profiles = payload["profiles"]
    daily_path = base.DAILY_DIR / f"{code}.txt"
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
    daily_df = daily_df[(daily_df["date"] < base.EXCLUDE_START) | (daily_df["date"] > base.EXCLUDE_END)].copy()

    min5_df = None
    min5_path = base.MIN5_DIR / f"{code}.txt"
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
                if sim.skipped:
                    for row in sim.skipped:
                        skipped.append(
                            {
                                "code": code,
                                "signal_date": item["signal_date"],
                                "entry_date": item["entry_date"],
                                "profile_name": profile["name"],
                                "stop_base": stop_cfg["stop_base"],
                                "stop_exec_mode": stop_cfg["stop_exec_mode"],
                                "reason": row.get("reason", "skipped"),
                            }
                        )
                    continue
                trades.extend(sim.trades)
    return trades, skipped


def _summarize_signal_basket(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["profile_name", "stop_base", "stop_exec_mode", "hold_days"]
    for keys, g in trades.groupby(group_cols, sort=True):
        profile_name, stop_base, stop_exec_mode, hold_days = keys
        weight = g["position_frac"].astype(float)
        avg_trade_return = float(np.average(g["return_pct"].astype(float), weights=weight))
        success_rate = float(np.average((g["return_pct"].astype(float) > 0).astype(float), weights=weight))
        rows.append(
            {
                "strategy": f"case_rank|{stop_base}|{stop_exec_mode}|{profile_name}",
                "trade_count": int(g["position_group"].nunique()),
                "avg_trade_return": avg_trade_return,
                "success_rate": success_rate,
                "avg_holding_days": float((pd.to_datetime(g["exit_date"]) - pd.to_datetime(g["entry_date"])).dt.days.mean() + 1.0),
                "profit_factor": float("nan"),
                "annual_return_signal_basket": float("nan"),
                "max_drawdown_signal_basket": float("nan"),
                "final_equity_signal_basket": float("nan"),
                "profile_name": profile_name,
                "profile_family": str(g.iloc[0]["profile_family"]),
                "tp_pct": float(g.iloc[0]["tp_pct"]),
                "weak_n": int(g.iloc[0]["weak_n"]),
                "dd_pct": float(g.iloc[0]["dd_pct"]),
                "hold_days": int(hold_days),
                "stop_base": stop_base,
                "stop_exec_mode": stop_exec_mode,
                "trigger_source_5min_ratio": float(np.average((g["trigger_source"] == "5min").astype(float), weights=weight)),
                "trigger_source_daily_ratio": float(np.average((g["trigger_source"] == "daily_fallback").astype(float), weights=weight)),
                "tp_armed_ratio": float(np.average(g["tp_arm_date"].notna().astype(float), weights=weight)),
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_trade_return", "success_rate"], ascending=[False, False]).reset_index(drop=True)


def _load_close_series_payload(code: str) -> tuple[str, pd.Series | None]:
    path = base.DAILY_DIR / f"{code}.txt"
    if not path.exists():
        return code, None
    s = real_account._fast_load_close_series(path)
    if s is None or s.empty:
        return code, None
    return code, s


def _build_close_map_parallel(codes: list[str], result_dir: Path, max_workers: int) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    return base._build_close_map_parallel(codes, result_dir, max_workers=max_workers)


def _summarize_account(trades: pd.DataFrame, result_dir: Path, max_workers: int) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    all_codes = sorted(trades["code"].astype(str).unique())
    market_dates, close_map = _build_close_map_parallel(all_codes, result_dir, max_workers=max_workers)
    if len(market_dates) == 0:
        raise RuntimeError("无法构建账户层 close_map")

    rows: list[dict[str, Any]] = []
    config = real_account.AccountConfig()

    def _is_open_exit(reason: str) -> bool:
        return "daily_open" in str(reason)

    group_cols = ["profile_name", "stop_base", "stop_exec_mode", "hold_days"]
    for keys, g in trades.groupby(group_cols, sort=True):
        profile_name, stop_base, stop_exec_mode, hold_days = keys
        trades_one = g.sort_values(["entry_date", "sort_score", "code", "position_leg"], ascending=[True, False, True, True]).reset_index(drop=True)
        groups_meta = (
            trades_one.groupby("position_group", sort=False)
            .agg(code=("code", "first"), entry_date=("entry_date", "first"), sort_score=("sort_score", "first"), entry_price=("entry_price", "first"))
            .reset_index()
        )
        entries_by_date = {
            d: gg.sort_values(["sort_score", "code"], ascending=[False, True]).to_dict("records")
            for d, gg in groups_meta.groupby("entry_date")
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
                pos_key = f'{tr["position_group"]}|{tr["position_leg"]}'
                if pos_key not in positions:
                    continue
                pos = positions.pop(pos_key)
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
                        "strategy_key": f"case_rank|{stop_base}|{stop_exec_mode}|{profile_name}",
                        "position_group": tr["position_group"],
                        "position_leg": tr["position_leg"],
                        "code": tr["code"],
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
                        "position_frac": tr["position_frac"],
                    }
                )

            equity_before_entry = cash
            for pos in positions.values():
                mark_price = float(close_map[pos["code"]].get(current_date, pos["entry_price"]))
                equity_before_entry += pos["shares"] * mark_price

            entry_candidates = entries_by_date.get(current_date, [])
            open_groups = {k.split("|", 1)[0] for k in positions.keys()}
            available_slots = max(config.max_positions - len(open_groups), 0)
            if entry_candidates and available_slots > 0:
                to_add: list[dict[str, Any]] = []
                for tr in entry_candidates:
                    group_id = str(tr["position_group"])
                    if group_id in open_groups:
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
                            group_id = str(tr["position_group"])
                            group_trades = trades_one[trades_one["position_group"] == group_id].copy()
                            if group_trades.empty:
                                continue
                            raw_entry_price = float(tr["entry_price"])
                            entry_price = raw_entry_price * (1.0 + config.slippage_rate)
                            alloc = min(investable * float(weight), per_pos_cap, cash)
                            gross_target = max(0.0, alloc / (1.0 + config.commission_rate))
                            total_shares = int(gross_target / entry_price / config.min_lot) * config.min_lot if gross_target > 0 and entry_price > 0 else 0
                            if total_shares <= 0:
                                continue
                            leg_plan = group_trades.sort_values("position_leg")[["position_leg", "position_frac", "code", "signal_date", "sort_score"]].to_dict("records")
                            leg_allocs: list[tuple[dict[str, Any], int]] = []
                            allocated = 0
                            for idx, leg in enumerate(leg_plan):
                                if idx < len(leg_plan) - 1:
                                    leg_shares = int(total_shares * float(leg["position_frac"]) / config.min_lot) * config.min_lot
                                    allocated += leg_shares
                                else:
                                    leg_shares = total_shares - allocated
                                if leg_shares > 0:
                                    leg_allocs.append((leg, leg_shares))
                            total_cost = 0.0
                            total_entry_fee = 0.0
                            temp_positions: list[tuple[str, dict[str, Any]]] = []
                            for leg, shares in leg_allocs:
                                gross_cost = shares * entry_price
                                fee = gross_cost * config.commission_rate
                                total_cost += gross_cost + fee
                                total_entry_fee += fee
                                pos_key = f'{group_id}|{leg["position_leg"]}'
                                temp_positions.append(
                                    (
                                        pos_key,
                                        {
                                            "group_id": group_id,
                                            "code": str(leg["code"]),
                                            "shares": shares,
                                            "entry_price": entry_price,
                                            "entry_price_raw": raw_entry_price,
                                            "entry_fee": fee,
                                            "entry_date": current_date,
                                            "position_frac": float(leg["position_frac"]),
                                        },
                                    )
                                )
                            if total_cost > cash:
                                continue
                            cash -= total_cost
                            for pos_key, pos in temp_positions:
                                positions[pos_key] = pos
                            open_groups.add(group_id)

            for tr in later_exits:
                pos_key = f'{tr["position_group"]}|{tr["position_leg"]}'
                if pos_key not in positions:
                    continue
                pos = positions.pop(pos_key)
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
                        "strategy_key": f"case_rank|{stop_base}|{stop_exec_mode}|{profile_name}",
                        "position_group": tr["position_group"],
                        "position_leg": tr["position_leg"],
                        "code": tr["code"],
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
                        "position_frac": tr["position_frac"],
                    }
                )

            equity = cash
            for pos in positions.values():
                mark_price = float(close_map[pos["code"]].get(current_date, pos["entry_price"]))
                equity += pos["shares"] * mark_price
            equity_rows.append({"date": current_date, "equity": equity, "cash": cash, "position_count": len({k.split('|',1)[0] for k in positions.keys()})})

        equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
        equity_curve = pd.Series(equity_df["equity"].to_numpy(dtype=float), index=pd.DatetimeIndex(equity_df["date"]), dtype=float)
        metrics = real_account.compute_metrics(equity_curve)
        max_drawdown_abs = float(metrics["max_drawdown"])
        annual_return = float(metrics["annual_return"])
        sharpe = float(metrics["sharpe"])
        calmar = real_account._compute_calmar(annual_return, max_drawdown_abs)
        executed_df = pd.DataFrame(executed_rows).sort_values(["exit_date", "entry_date", "code", "position_leg"]).reset_index(drop=True) if executed_rows else pd.DataFrame()
        group_returns = executed_df.groupby("position_group")["pnl"].sum() / executed_df.groupby("position_group")["gross_entry_cost"].sum() if not executed_df.empty else pd.Series(dtype=float)
        avg_trade_return = float(group_returns.mean()) if not group_returns.empty else float("nan")
        success_rate = float((group_returns > 0).mean()) if not group_returns.empty else float("nan")
        hold_return = float(equity_df.iloc[-1]["equity"] / config.initial_capital - 1.0) if not equity_df.empty else float("nan")
        summary = {
            "profile_name": profile_name,
            "profile_family": str(g.iloc[0]["profile_family"]),
            "tp_pct": float(g.iloc[0]["tp_pct"]),
            "weak_n": int(g.iloc[0]["weak_n"]),
            "dd_pct": float(g.iloc[0]["dd_pct"]),
            "hold_days": int(hold_days),
            "stop_base": stop_base,
            "stop_exec_mode": stop_exec_mode,
            "final_multiple": float(metrics["final_multiple"]),
            "annual_return": annual_return,
            "holding_return": hold_return,
            "max_drawdown": -max_drawdown_abs,
            "sharpe": sharpe,
            "calmar": calmar,
            "trade_count": int(len(group_returns)),
            "success_rate": success_rate,
            "avg_trade_return": avg_trade_return,
            "max_losing_streak": int(real_account._max_losing_streak(group_returns.tolist() if not group_returns.empty else [])),
            "equity_days": int(metrics["days"]),
            "final_equity": float(equity_df.iloc[-1]["equity"]) if not equity_df.empty else float("nan"),
        }
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(
        ["annual_return", "final_equity", "max_drawdown", "sharpe", "calmar"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def best_profit_protect(summary_df: pd.DataFrame) -> dict[str, Any]:
    if summary_df.empty:
        return {}
    sub = summary_df[summary_df["profile_family"].astype(str).str.startswith("profit_protect")].copy()
    if sub.empty:
        return {}
    sub = sub.sort_values(
        ["annual_return", "final_equity", "max_drawdown", "sharpe", "calmar"],
        ascending=[False, False, False, False, False],
    )
    return sub.iloc[0].to_dict()


def run_search_round(
    result_dir: Path,
    source_csv: Path,
    max_tp: float,
    file_limit_codes: int,
    date_limit: int,
    max_workers: int,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    profiles = build_profiles(max_tp)
    update_progress(
        result_dir,
        "loading_source",
        source_csv=str(source_csv),
        max_tp=max_tp,
        profile_count=len(profiles),
        file_limit_codes=file_limit_codes,
        date_limit=date_limit,
        max_workers=max_workers,
    )
    candidates = base.load_source_candidates(source_csv, file_limit_codes, date_limit)
    candidates.to_csv(result_dir / "source_candidates.csv", index=False, encoding="utf-8-sig")
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
                "buy_gap_limit": float(base.BUY_GAP_LIMIT),
                "profiles": profiles,
            }
        )

    total_codes = len(grouped_payloads)
    total_jobs = len(candidates) * len(STOP_CONFIGS) * len(profiles)
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

    trades = pd.DataFrame(trade_rows).sort_values(["profile_name", "stop_base", "stop_exec_mode", "signal_date", "code", "position_leg"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows).sort_values(["profile_name", "stop_base", "stop_exec_mode", "signal_date", "code"], na_position="last").reset_index(drop=True) if skipped_rows else pd.DataFrame()
    trades.to_csv(result_dir / "trades.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(result_dir / "skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "trades_ready", trade_count=int(len(trades)), skipped_count=int(len(skipped)))

    signal_summary = _summarize_signal_basket(trades)
    account_summary = _summarize_account(trades, result_dir, max_workers=max_workers)
    signal_summary.to_csv(result_dir / "signal_basket_summary.csv", index=False, encoding="utf-8-sig")
    account_summary.to_csv(result_dir / "account_summary.csv", index=False, encoding="utf-8-sig")

    best_overall = account_summary.iloc[0].to_dict() if not account_summary.empty else {}
    best_profit = best_profit_protect(account_summary)
    best_fixed = (
        account_summary[account_summary["profile_family"] == "fixed_tp"]
        .sort_values(["annual_return", "final_equity", "max_drawdown", "sharpe", "calmar"], ascending=[False, False, False, False, False])
        .iloc[0]
        .to_dict()
        if not account_summary[account_summary["profile_family"] == "fixed_tp"].empty
        else {}
    )

    summary = {
        "mode": "h3_with_half_profit_protect",
        "max_tp": max_tp,
        "profile_count": len(profiles),
        "best_signal_basket_profile": signal_summary.iloc[0].to_dict() if not signal_summary.empty else {},
        "best_account_profile": best_overall,
        "best_profit_protect_profile": best_profit,
        "best_fixed_tp_profile": best_fixed,
        "signal_profile_count": int(len(signal_summary)),
        "account_profile_count": int(len(account_summary)),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3日窗口 + 固定止盈/利润保护/半仓利润保护联合搜索")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--source-csv", type=str, default=str(SOURCE_CANDIDATES))
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--file-limit-codes", type=int, default=80)
    parser.add_argument("--date-limit", type=int, default=4)
    parser.add_argument("--start-max-tp", type=float, default=0.085)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--max-cap", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else RESULT_ROOT / f"brick_case_rank_h3_with_half_profit_protect_search_v1_{args.mode}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    rounds: list[dict[str, Any]] = []
    current_max_tp = round(float(args.start_max_tp), 4)
    step = round(float(args.step), 4)
    max_cap = round(float(args.max_cap), 4)
    round_idx = 1

    try:
        while True:
            round_dir = output_root / f"round_{round_idx:02d}_maxtp_{current_max_tp:.4f}"
            actual_file_limit = 0 if args.mode == "full" else int(args.file_limit_codes)
            actual_date_limit = 0 if args.mode == "full" else int(args.date_limit)
            summary = run_search_round(
                result_dir=round_dir,
                source_csv=Path(args.source_csv),
                max_tp=current_max_tp,
                file_limit_codes=actual_file_limit,
                date_limit=actual_date_limit,
                max_workers=int(args.max_workers),
            )
            best_profit = summary.get("best_profit_protect_profile", {}) or {}
            best_profit_tp = float(best_profit.get("tp_pct", -1)) if best_profit else -1.0
            profit_edge = abs(best_profit_tp - current_max_tp) < 1e-12 if best_profit else False
            rounds.append(
                {
                    "result_dir": str(round_dir),
                    "max_tp": current_max_tp,
                    "best_account_profile": summary.get("best_account_profile", {}),
                    "best_profit_protect_profile": best_profit,
                    "profit_protect_tp_on_upper_edge": profit_edge,
                }
            )
            if not profit_edge:
                break
            next_tp = round(current_max_tp + step, 4)
            if next_tp > max_cap + 1e-12:
                break
            current_max_tp = next_tp
            round_idx += 1

        final = {
            "mode": args.mode,
            "source_csv": str(args.source_csv),
            "rounds": rounds,
            "final_round": rounds[-1] if rounds else {},
            "stopped_because": "profit_protect_interior_best_found" if rounds and not rounds[-1]["profit_protect_tp_on_upper_edge"] else "hit_max_cap_or_no_rounds",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        (output_root / "summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        write_error(output_root, exc)
        raise


if __name__ == "__main__":
    main()
