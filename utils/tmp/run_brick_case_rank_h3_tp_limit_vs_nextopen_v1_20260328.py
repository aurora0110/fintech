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


base = load_module(BASE_SCRIPT, "brick_case_rank_h3_tp_limit_vs_nextopen_base")

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
STOP_CONFIGS = list(base.STOP_CONFIGS)


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
                    "name": f"tp_limit_same_day_{tag}_h3",
                    "family": "fixed_tp_limit",
                    "tp_pct": tp_pct,
                    "hold_days": 3,
                    "exit_mode": "tp_limit_same_day",
                },
                {
                    "name": f"tp_next_open_{tag}_h3",
                    "family": "fixed_tp_nextopen",
                    "tp_pct": tp_pct,
                    "hold_days": 3,
                    "exit_mode": "next_day_open",
                },
            ]
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


def _resolve_exit(exit_mode: str, trigger_date: pd.Timestamp, trigger_dt: pd.Timestamp | None, day_daily_row: pd.Series, daily_df: pd.DataFrame, day_min: pd.DataFrame, tp_price: float) -> tuple[pd.Timestamp, float, str]:
    if exit_mode == "tp_limit_same_day":
        return pd.Timestamp(trigger_date), float(tp_price), "tp_limit_same_day"
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

    stop_price = _resolve_stop_base(stop_cfg["stop_base"], signal_open, signal_close, signal_low, entry_open, entry_close, entry_low)
    tp_price = entry_open * (1.0 + float(profile["tp_pct"]))
    trade_window_dates, eligible_exit_dates = _build_trade_window(daily_df, entry_date, int(profile["hold_days"]))
    if not trade_window_dates:
        return SimResult(trade=None, skipped=True, skip_reason="no_trade_window")

    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    trigger_source = "max_hold"
    tp_arm_date: pd.Timestamp | None = None
    tp_arm_dt: pd.Timestamp | None = None

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
                exit_date, exit_price, mode_reason = base._resolve_exit(
                    stop_cfg["stop_exec_mode"], pd.Timestamp(d), None, day_daily_row, daily_df, day_min
                )
                exit_reason = f'{stop_cfg["stop_base"]}|{stop_cfg["stop_exec_mode"]}|{mode_reason}'
                trigger_source = "daily_fallback"
                break
            if day_high >= tp_price:
                tp_arm_date = pd.Timestamp(d)
                exit_date, exit_price, mode_reason = _resolve_exit(
                    profile["exit_mode"], pd.Timestamp(d), None, day_daily_row, daily_df, day_min, tp_price
                )
                exit_reason = f'{profile["name"]}|{mode_reason}'
                trigger_source = "daily_fallback"
                break
            continue

        for _, bar in day_min.iterrows():
            bar_dt = pd.Timestamp(bar["datetime"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])

            if bar_low <= stop_price:
                exit_date, exit_price, mode_reason = base._resolve_exit(
                    stop_cfg["stop_exec_mode"], pd.Timestamp(d), bar_dt, day_daily_row, daily_df, day_min
                )
                exit_reason = f'{stop_cfg["stop_base"]}|{stop_cfg["stop_exec_mode"]}|{mode_reason}'
                trigger_source = "5min"
                break

            if bar_high >= tp_price:
                tp_arm_date = pd.Timestamp(d)
                tp_arm_dt = bar_dt
                exit_date, exit_price, mode_reason = _resolve_exit(
                    profile["exit_mode"], pd.Timestamp(d), bar_dt, day_daily_row, daily_df, day_min, tp_price
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
            "weak_n": 0,
            "dd_pct": 0.0,
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


def run_search_round(result_dir: Path, source_csv: Path, max_tp: float, file_limit_codes: int, date_limit: int, max_workers: int) -> dict[str, Any]:
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

    trades = pd.DataFrame(trade_rows).sort_values(["profile_name", "stop_base", "stop_exec_mode", "signal_date", "code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows).sort_values(["profile_name", "stop_base", "stop_exec_mode", "signal_date", "code"], na_position="last").reset_index(drop=True) if skipped_rows else pd.DataFrame()
    trades.to_csv(result_dir / "trades.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(result_dir / "skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "trades_ready", trade_count=int(len(trades)), skipped_count=int(len(skipped)))

    signal_summary = base._summarize_signal_basket(trades)
    account_summary = base._summarize_account(trades, result_dir, max_workers=max_workers)
    signal_summary.to_csv(result_dir / "signal_basket_summary.csv", index=False, encoding="utf-8-sig")
    account_summary.to_csv(result_dir / "account_summary.csv", index=False, encoding="utf-8-sig")

    def _best_family(df: pd.DataFrame, family: str) -> dict[str, Any]:
        sub = df[df["profile_family"] == family].copy()
        if sub.empty:
            return {}
        sub = sub.sort_values(["annual_return", "final_equity", "max_drawdown", "sharpe", "calmar"], ascending=[False, False, False, False, False])
        return sub.iloc[0].to_dict()

    best_overall = account_summary.iloc[0].to_dict() if not account_summary.empty else {}
    best_limit = _best_family(account_summary, "fixed_tp_limit")
    best_nextopen = _best_family(account_summary, "fixed_tp_nextopen")
    summary = {
        "mode": "h3_tp_limit_vs_nextopen",
        "max_tp": max_tp,
        "profile_count": len(profiles),
        "best_signal_basket_profile": signal_summary.iloc[0].to_dict() if not signal_summary.empty else {},
        "best_account_profile": best_overall,
        "best_limit_profile": best_limit,
        "best_nextopen_profile": best_nextopen,
        "signal_profile_count": int(len(signal_summary)),
        "account_profile_count": int(len(account_summary)),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished")
    return summary


def _edge_state(summary: dict[str, Any], current_max_tp: float) -> dict[str, bool]:
    best_limit = summary.get("best_limit_profile", {}) or {}
    best_nextopen = summary.get("best_nextopen_profile", {}) or {}
    limit_edge = abs(float(best_limit.get("tp_pct", -1)) - current_max_tp) < 1e-12 if best_limit else False
    nextopen_edge = abs(float(best_nextopen.get("tp_pct", -1)) - current_max_tp) < 1e-12 if best_nextopen else False
    return {"limit_edge": limit_edge, "nextopen_edge": nextopen_edge}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3日窗口 fixed_tp：当日限价止盈 vs 次日开盘止盈 对比，并自动扩边")
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
    output_root = Path(args.output_root) if args.output_root else RESULT_ROOT / f"brick_case_rank_h3_tp_limit_vs_nextopen_v1_{args.mode}_{timestamp}"
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
            edge = _edge_state(summary, current_max_tp)
            rounds.append(
                {
                    "result_dir": str(round_dir),
                    "max_tp": current_max_tp,
                    "best_account_profile": summary.get("best_account_profile", {}),
                    "best_limit_profile": summary.get("best_limit_profile", {}),
                    "best_nextopen_profile": summary.get("best_nextopen_profile", {}),
                    **edge,
                }
            )
            if not edge["limit_edge"] and not edge["nextopen_edge"]:
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
            "stopped_because": "both_interior_best_found" if rounds and (not rounds[-1]["limit_edge"]) and (not rounds[-1]["nextopen_edge"]) else "hit_max_cap_or_no_rounds",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        (output_root / "summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        write_error(output_root, exc)
        raise


if __name__ == "__main__":
    main()
