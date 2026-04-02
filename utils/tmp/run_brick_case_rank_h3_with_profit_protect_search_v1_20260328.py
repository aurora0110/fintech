from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

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


base = load_module(BASE_SCRIPT, "brick_case_rank_h3_profit_protect_base")

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
STOP_CONFIGS = list(base.STOP_CONFIGS)
WEAK_COUNTS = list(base.WEAK_COUNTS)
PULLBACK_PCTS = list(base.PULLBACK_PCTS)


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
                sim = base.simulate_one_trade_profile(
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


def summarize_profiles(trades: pd.DataFrame, result_dir: Path, max_workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_summary = base._summarize_signal_basket(trades)
    account_summary = base._summarize_account(trades, result_dir, max_workers=max_workers)
    return signal_summary, account_summary


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

    trades = (
        pd.DataFrame(trade_rows)
        .sort_values(["profile_name", "stop_base", "stop_exec_mode", "signal_date", "code"])
        .reset_index(drop=True)
        if trade_rows
        else pd.DataFrame()
    )
    skipped = (
        pd.DataFrame(skipped_rows)
        .sort_values(["profile_name", "stop_base", "stop_exec_mode", "signal_date", "code"], na_position="last")
        .reset_index(drop=True)
        if skipped_rows
        else pd.DataFrame()
    )
    trades.to_csv(result_dir / "trades.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(result_dir / "skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "trades_ready", trade_count=int(len(trades)), skipped_count=int(len(skipped)))

    signal_summary, account_summary = summarize_profiles(trades, result_dir, max_workers=max_workers)
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
        "mode": "h3_with_profit_protect",
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
    parser = argparse.ArgumentParser(description="3日窗口 + 固定止盈/利润保护联合搜索，并跟踪利润保护策略最优参数")
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
    output_root = Path(args.output_root) if args.output_root else RESULT_ROOT / f"brick_case_rank_h3_with_profit_protect_search_v1_{args.mode}_{timestamp}"
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
