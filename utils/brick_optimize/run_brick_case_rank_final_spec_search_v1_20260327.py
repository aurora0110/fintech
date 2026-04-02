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
BUY_MODEL_RESULT_DIR = RESULT_ROOT / "brick_case_rank_model_search_v1_full_20260327_r1"
SOURCE_CANDIDATES = BUY_MODEL_RESULT_DIR / "best_model_top20_candidates.csv"
DAILY_DIR = ROOT / "data" / "20260324"
MIN5_DIR = ROOT / "data" / "202603245min"
TP_SEARCH_PATH = ROOT / "utils" / "tmp" / "run_brick_r3_minute_tp_search_v1_20260327.py"
REAL_ACCOUNT_PATH = ROOT / "utils" / "tmp" / "run_brick_real_account_compare_v1_20260326.py"
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUY_GAP_LIMIT = 0.04
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
MAX_HOLD_DAYS_LIST = [2, 3, 4, 5, 6]
TP_LEVELS = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085]
WEAK_COUNTS = [2, 3, 4]
PULLBACK_PCTS = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
STOP_CONFIGS = [
    {"stop_base": "entry_min_oc", "stop_exec_mode": "next_5min_close"},
    {"stop_base": "entry_min_oc", "stop_exec_mode": "same_day_close"},
    {"stop_base": "entry_low", "stop_exec_mode": "next_5min_close"},
    {"stop_base": "entry_low", "stop_exec_mode": "same_day_close"},
    {"stop_base": "signal_min_oc", "stop_exec_mode": "next_5min_close"},
    {"stop_base": "signal_min_oc", "stop_exec_mode": "same_day_close"},
    {"stop_base": "signal_low", "stop_exec_mode": "next_5min_close"},
    {"stop_base": "signal_low", "stop_exec_mode": "same_day_close"},
]
_DAILY_STEM_MAP: dict[str, str] | None = None


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tp_mod = load_module(TP_SEARCH_PATH, "brick_case_rank_final_spec_tp_mod")
real_account = load_module(REAL_ACCOUNT_PATH, "brick_case_rank_final_spec_real_account")


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


def _code_key(value: Any) -> str:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return digits
    return digits[-6:] if len(digits) >= 6 else digits.zfill(6)


def _build_stem_map(directory: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(directory.glob("*.txt")):
        key = _code_key(path.stem)
        if key:
            mapping[key] = path.stem
    return mapping


def _resolve_daily_stem(code: Any) -> str:
    global _DAILY_STEM_MAP
    if _DAILY_STEM_MAP is None:
        _DAILY_STEM_MAP = _build_stem_map(DAILY_DIR)
    text = str(code)
    key = _code_key(text)
    return _DAILY_STEM_MAP.get(key, text)


def _profile_defs() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for hold_days in MAX_HOLD_DAYS_LIST:
        for tp_pct in TP_LEVELS:
            tag = f"{tp_pct:.4f}"
            profiles.extend(
                [
                    {
                        "name": f"tp_next5close_{tag}_h{hold_days}",
                        "family": "fixed_tp",
                        "tp_pct": tp_pct,
                        "hold_days": hold_days,
                        "exit_mode": "next_5min_close",
                    },
                    {
                        "name": f"tp_close_{tag}_h{hold_days}",
                        "family": "fixed_tp",
                        "tp_pct": tp_pct,
                        "hold_days": hold_days,
                        "exit_mode": "same_day_close",
                    },
                    {
                        "name": f"tp_next_open_{tag}_h{hold_days}",
                        "family": "fixed_tp",
                        "tp_pct": tp_pct,
                        "hold_days": hold_days,
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
                            "name": f"pp_weak_{exit_tag}_{tag}_n{weak_n}_h{hold_days}",
                            "family": "profit_protect_weak",
                            "tp_pct": tp_pct,
                            "hold_days": hold_days,
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
                            "name": f"pp_dd_{exit_tag}_{tag}_m{dd_tag}_h{hold_days}",
                            "family": "profit_protect_pullback",
                            "tp_pct": tp_pct,
                            "hold_days": hold_days,
                            "dd_pct": dd_pct,
                            "exit_mode": exit_mode,
                        }
                    )
    return profiles


PROFILE_DEFS = _profile_defs()


def _sort_source(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    brick_height = pd.to_numeric(out.get("brick_red_len", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    signal_ret = pd.to_numeric(out.get("signal_ret", 0.0), errors="coerce").fillna(0.0)
    out["rank_ratio"] = brick_height / np.maximum(signal_ret.abs().to_numpy(dtype=float), 0.01)
    out["rank_rebound"] = pd.to_numeric(out.get("rebound_ratio", 0.0), errors="coerce").fillna(0.0)
    out["rank_close_loc"] = pd.to_numeric(out.get("close_location", 0.0), errors="coerce").fillna(0.0)
    out["rank_upper_shadow"] = pd.to_numeric(out.get("upper_shadow_pct", 999.0), errors="coerce").fillna(999.0)
    out = out.sort_values(
        ["signal_date", "rank_ratio", "rank_rebound", "rank_close_loc", "rank_upper_shadow", "code"],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)
    out["sort_score"] = out["rank_ratio"] + out["rank_rebound"] * 1e-3 + out["rank_close_loc"] * 1e-4 - out["rank_upper_shadow"] * 1e-5
    return out


def load_source_candidates(source_csv: Path, file_limit_codes: int, date_limit: int) -> pd.DataFrame:
    sample_cols = pd.read_csv(source_csv, nrows=0).columns.tolist()
    parse_dates = [col for col in ["signal_date", "entry_date", "exit_date"] if col in sample_cols]
    df = pd.read_csv(source_csv, parse_dates=parse_dates)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df[(df["signal_date"] < EXCLUDE_START) | (df["signal_date"] > EXCLUDE_END)].copy()
    df["code"] = df["code"].map(_resolve_daily_stem)
    df = _sort_source(df)
    if file_limit_codes > 0:
        keep_codes = sorted(df["code"].astype(str).unique())[:file_limit_codes]
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    if date_limit > 0:
        keep_dates = sorted(df["signal_date"].dt.strftime("%Y-%m-%d").unique())[:date_limit]
        df = df[df["signal_date"].dt.strftime("%Y-%m-%d").isin(keep_dates)].copy()
    return df.reset_index(drop=True)


def _next_trade_date(daily_df: pd.DataFrame, current_date: pd.Timestamp) -> pd.Timestamp | None:
    later = daily_df[daily_df["date"] > current_date]["date"]
    if later.empty:
        return None
    return pd.Timestamp(later.iloc[0])


def _next_bar_open(min5_df: pd.DataFrame, current_dt: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    later = min5_df[min5_df["datetime"] > current_dt]
    if later.empty:
        return None, None
    row = later.iloc[0]
    return pd.Timestamp(row["date"]), float(row["open"])


def _next_bar_close(min5_df: pd.DataFrame, current_dt: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    later = min5_df[min5_df["datetime"] > current_dt]
    if later.empty:
        return None, None
    row = later.iloc[0]
    return pd.Timestamp(row["date"]), float(row["close"])


def _daily_exit(daily_df: pd.DataFrame, target_date: pd.Timestamp, mode: str) -> tuple[pd.Timestamp | None, float | None, str]:
    row = daily_df[daily_df["date"] == target_date]
    if row.empty:
        return None, None, "missing_target_daily_row"
    if mode == "close":
        return pd.Timestamp(target_date), float(row.iloc[0]["close"]), "daily_close"
    if mode == "open":
        return pd.Timestamp(target_date), float(row.iloc[0]["open"]), "daily_open"
    raise ValueError(mode)


def _resolve_exit(
    exit_mode: str,
    trigger_date: pd.Timestamp,
    trigger_dt: pd.Timestamp | None,
    day_daily_row: pd.Series,
    daily_df: pd.DataFrame,
    day_min: pd.DataFrame,
) -> tuple[pd.Timestamp | None, float | None, str]:
    if exit_mode == "same_day_close":
        return pd.Timestamp(trigger_date), float(day_daily_row["close"]), "same_day_close"
    if exit_mode == "next_day_open":
        next_date = _next_trade_date(daily_df, pd.Timestamp(trigger_date))
        if next_date is None:
            return pd.Timestamp(trigger_date), float(day_daily_row["close"]), "fallback_same_day_close"
        return _daily_exit(daily_df, next_date, "open")
    if exit_mode == "next_5min_open":
        if trigger_dt is not None and not day_min.empty:
            next_date, next_open = _next_bar_open(day_min, trigger_dt)
            if next_date is not None and next_open is not None:
                return next_date, next_open, "next_5min_open"
        return pd.Timestamp(trigger_date), float(day_daily_row["close"]), "fallback_same_day_close"
    if exit_mode == "next_5min_close":
        if trigger_dt is not None and not day_min.empty:
            next_date, next_close = _next_bar_close(day_min, trigger_dt)
            if next_date is not None and next_close is not None:
                return next_date, next_close, "next_5min_close"
        return pd.Timestamp(trigger_date), float(day_daily_row["close"]), "fallback_same_day_close"
    raise ValueError(exit_mode)


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

    stop_price = _resolve_stop_base(stop_cfg["stop_base"], signal_open, signal_close, signal_low, entry_open, entry_close, entry_low)
    tp_price = entry_open * (1.0 + float(profile["tp_pct"]))
    trade_window_dates, eligible_exit_dates = _build_trade_window(daily_df, entry_date, int(profile["hold_days"]))
    if not trade_window_dates:
        return SimResult(trade=None, skipped=True, skip_reason="no_trade_window")

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
            # hard stop has highest priority
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
                    if profile["family"] == "profit_protect_weak":
                        is_weak = bool(prev_ref_low is not None and np.isfinite(prev_ref_low) and day_close < prev_ref_low)
                        weak_count = weak_count + 1 if is_weak else 0
                        prev_ref_low = day_low
                        if weak_count >= int(profile["weak_n"]):
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
                if profile["family"] == "profit_protect_weak":
                    is_weak = bool(prev_ref_low is not None and np.isfinite(prev_ref_low) and bar_close < prev_ref_low)
                    weak_count = weak_count + 1 if is_weak else 0
                    prev_ref_low = bar_low
                    if weak_count >= int(profile["weak_n"]):
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
            "weak_n": int(profile.get("weak_n", 0)),
            "dd_pct": float(profile.get("dd_pct", 0.0)),
            "tp_arm_date": pd.Timestamp(tp_arm_date) if tp_arm_date is not None else pd.NaT,
            "tp_arm_dt": pd.Timestamp(tp_arm_dt) if tp_arm_dt is not None else pd.NaT,
            "stop_base": str(stop_cfg["stop_base"]),
            "stop_exec_mode": str(stop_cfg["stop_exec_mode"]),
            "sort_score": float(sort_score),
        }
    )


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    daily_path = DAILY_DIR / f"{code}.txt"
    if not daily_path.exists():
        skipped = []
        for item in rows:
            for stop_cfg in STOP_CONFIGS:
                for profile in PROFILE_DEFS:
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

    daily_df = tp_mod.hybrid.base.load_daily_df(daily_path)
    if daily_df is None or daily_df.empty:
        return [], [{"code": code, "reason": "empty_daily_df"}]
    daily_df = daily_df[(daily_df["date"] < EXCLUDE_START) | (daily_df["date"] > EXCLUDE_END)].copy()

    min5_df = None
    min5_path = MIN5_DIR / f"{code}.txt"
    if min5_path.exists():
        loaded = tp_mod.hybrid.base.load_minute_df(min5_path)
        min5_df = tp_mod._prepare_min5_indicators(loaded) if loaded is not None and not loaded.empty else None

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in rows:
        for stop_cfg in STOP_CONFIGS:
            for profile in PROFILE_DEFS:
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
    rows: list[dict[str, Any]] = []
    group_cols = ["profile_name", "stop_base", "stop_exec_mode", "hold_days"]
    for keys, g in trades.groupby(group_cols, sort=True):
        profile_name, stop_base, stop_exec_mode, hold_days = keys
        strategy = f"case_rank|{stop_base}|{stop_exec_mode}|{profile_name}"
        row = tp_mod.hybrid.base.summarize_trades(g, strategy)
        row["profile_name"] = profile_name
        row["profile_family"] = str(g.iloc[0]["profile_family"])
        row["tp_pct"] = float(g.iloc[0]["tp_pct"])
        row["weak_n"] = int(g.iloc[0]["weak_n"])
        row["dd_pct"] = float(g.iloc[0]["dd_pct"])
        row["hold_days"] = int(hold_days)
        row["stop_base"] = stop_base
        row["stop_exec_mode"] = stop_exec_mode
        row["trigger_source_5min_ratio"] = float((g["trigger_source"] == "5min").mean())
        row["trigger_source_daily_ratio"] = float((g["trigger_source"] == "daily_fallback").mean())
        row["tp_armed_ratio"] = float(g["tp_arm_date"].notna().mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["annual_return_signal_basket", "final_equity_signal_basket", "profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _load_close_series_payload(code: str) -> tuple[str, pd.Series | None]:
    path = DAILY_DIR / f"{code}.txt"
    if not path.exists():
        return code, None
    s = real_account._fast_load_close_series(path)
    if s is None or s.empty:
        return code, None
    return code, s


def _build_close_map_parallel(codes: list[str], result_dir: Path, max_workers: int) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    relevant: dict[str, pd.Series] = {}
    all_dates: set[pd.Timestamp] = set()
    unique_codes = sorted(set(map(str, codes)))
    total = len(unique_codes)
    if total == 0:
        return pd.DatetimeIndex([]), {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_load_close_series_payload, code): code for code in unique_codes}
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            code, s = future.result()
            if s is not None and not s.empty:
                relevant[code] = s
                all_dates.update(pd.DatetimeIndex(s.index).tolist())
            if completed == 1 or completed % 50 == 0 or completed == total:
                update_progress(result_dir, "building_close_map", done_codes=int(completed), total_codes=int(total), loaded_codes=int(len(relevant)))
    market_dates = pd.DatetimeIndex(sorted(all_dates))
    close_map: dict[str, pd.Series] = {}
    total_series = len(relevant)
    for idx, (code, s) in enumerate(relevant.items(), start=1):
        close_map[code] = s.reindex(market_dates).ffill()
        if idx == 1 or idx % 50 == 0 or idx == total_series:
            update_progress(result_dir, "reindexing_close_map", done_series=int(idx), total_series=int(total_series), market_days=int(len(market_dates)))
    return market_dates, close_map


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
                        "strategy_key": f"case_rank|{stop_base}|{stop_exec_mode}|{profile_name}",
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
                        "strategy_key": f"case_rank|{stop_base}|{stop_exec_mode}|{profile_name}",
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
        metrics = real_account.compute_metrics(equity_curve)
        max_drawdown_abs = float(metrics["max_drawdown"])
        annual_return = float(metrics["annual_return"])
        sharpe = float(metrics["sharpe"])
        calmar = real_account._compute_calmar(annual_return, max_drawdown_abs)
        executed_df = pd.DataFrame(executed_rows).sort_values(["exit_date", "entry_date", "code"]).reset_index(drop=True) if executed_rows else pd.DataFrame()
        avg_trade_return = float(executed_df["return_pct_net"].mean()) if not executed_df.empty else float("nan")
        success_rate = float((executed_df["return_pct_net"] > 0).mean()) if not executed_df.empty else float("nan")
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
            "trade_count": int(len(executed_df)),
            "success_rate": success_rate,
            "avg_trade_return": avg_trade_return,
            "max_losing_streak": int(real_account._max_losing_streak(executed_df["return_pct_net"].tolist() if not executed_df.empty else [])),
            "equity_days": int(metrics["days"]),
            "final_equity": float(equity_df.iloc[-1]["equity"]) if not equity_df.empty else float("nan"),
        }
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(
        ["annual_return", "final_equity", "max_drawdown", "sharpe", "calmar"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def run_search(
    result_dir: Path,
    source_csv: Path,
    file_limit_codes: int,
    date_limit: int,
    max_workers: int,
    buy_gap_limit: float,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "loading_source", source_csv=str(source_csv), file_limit_codes=file_limit_codes, date_limit=date_limit, max_workers=max_workers, buy_gap_limit=buy_gap_limit)
    candidates = load_source_candidates(source_csv, file_limit_codes, date_limit)
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
                "buy_gap_limit": float(buy_gap_limit),
            }
        )

    total_codes = len(grouped_payloads)
    total_jobs = len(candidates) * len(STOP_CONFIGS) * len(PROFILE_DEFS)
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
                update_progress(result_dir, "simulating_trades", done_codes=completed, total_codes=total_codes, done_jobs=len(trade_rows) + len(skipped_rows), total_jobs=total_jobs, last_code=code)

    trades = pd.DataFrame(trade_rows).sort_values(["profile_name", "stop_base", "stop_exec_mode", "signal_date", "code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows).sort_values(["profile_name", "stop_base", "stop_exec_mode", "signal_date", "code"], na_position="last").reset_index(drop=True) if skipped_rows else pd.DataFrame()
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
            "buy_gap_limit": buy_gap_limit,
            "exclude_window": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
            "max_hold_days_list": MAX_HOLD_DAYS_LIST,
            "minute_priority": "5min_then_daily_fallback",
            "same_bar_priority": "hard_stop_then_fixed_tp_then_profit_protect",
            "same_type_tie_break": "worse_price_for_account",
            "trigger_fill_rule": "next_5min_open_or_close_by_profile",
            "stop_configs": STOP_CONFIGS,
            "tp_levels": TP_LEVELS,
            "weak_counts": WEAK_COUNTS,
            "pullback_pcts": PULLBACK_PCTS,
            "daily_fallback_rule": "5min_trigger_uses_daily_high_low_and_5min_exec_falls_back_to_same_day_close",
        },
        "best_signal_basket_profile": signal_summary.iloc[0].to_dict() if not signal_summary.empty else {},
        "best_account_profile": account_summary.iloc[0].to_dict() if not account_summary.empty else {},
        "signal_profile_count": int(len(signal_summary)),
        "account_profile_count": int(len(account_summary)),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按最新最终版实验方案搜索 case_rank_lgbm_top20 的退出矩阵")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--source-csv", type=str, default=str(SOURCE_CANDIDATES))
    parser.add_argument("--buy-gap-limit", type=float, default=BUY_GAP_LIMIT)
    parser.add_argument("--file-limit-codes", type=int, default=60)
    parser.add_argument("--date-limit", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_case_rank_final_spec_search_v1_{args.mode}_{timestamp}"
    file_limit_codes = int(args.file_limit_codes)
    date_limit = int(args.date_limit)
    if args.mode == "full":
        file_limit_codes = 0
        date_limit = 0
    try:
        run_search(
            result_dir=output_dir,
            source_csv=Path(args.source_csv),
            file_limit_codes=file_limit_codes,
            date_limit=date_limit,
            max_workers=int(args.max_workers),
            buy_gap_limit=float(args.buy_gap_limit),
        )
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
