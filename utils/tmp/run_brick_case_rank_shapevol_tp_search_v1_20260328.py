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

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
BASE_SCRIPT = ROOT / "utils" / "tmp" / "run_brick_case_rank_final_spec_search_v1_20260327.py"
SOURCE_CANDIDATES = ROOT / "results" / "brick_case_rank_model_search_v1_full_20260327_r1" / "best_model_top20_candidates.csv"
BASELINE_SUMMARY = ROOT / "results" / "brick_case_rank_histvol_tp_search_v1_full_20260328_r1" / "summary.json"


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "brick_case_rank_shapevol_tp_base")

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
HOLD_DAYS = [2, 3, 4, 5, 6]
STOP_CFG = {"stop_base": "signal_low", "stop_exec_mode": "same_day_close"}
EXIT_MODE = "next_day_open"
BUY_GAP_LIMIT = 0.04


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


def _load_baseline_best() -> dict[str, Any]:
    if not BASELINE_SUMMARY.exists():
        return {}
    payload = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    return payload.get("final_round", {}).get("best_account_profile", {})


def _build_k_values(k_min: float, k_max: float, step: float) -> list[float]:
    values: list[float] = []
    x = k_min
    while x <= k_max + 1e-12:
        values.append(round(x, 4))
        x += step
    return values


def _build_profiles(k_values: list[float]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for k in k_values:
        for hold_days in HOLD_DAYS:
            profiles.append(
                {
                    "name": f"shapevol_tp_k{k:.4f}_h{hold_days}",
                    "family": "shapevol_fixed_tp",
                    "k_value": float(k),
                    "hold_days": int(hold_days),
                    "exit_mode": EXIT_MODE,
                }
            )
    return profiles


def _recent_range_pct(daily_df: pd.DataFrame, signal_date: pd.Timestamp, lookback: int = 3) -> float | None:
    hist = daily_df[daily_df["date"] <= signal_date].sort_values("date").tail(lookback)
    if hist.empty:
        return None
    denom = pd.to_numeric(hist["close"], errors="coerce").replace(0, np.nan)
    range_pct = (pd.to_numeric(hist["high"], errors="coerce") - pd.to_numeric(hist["low"], errors="coerce")) / denom
    value = float(range_pct.mean())
    return value if np.isfinite(value) and value > 0 else None


def _shape_vol_pct(signal_row: pd.Series, daily_df: pd.DataFrame, signal_date: pd.Timestamp, body_ratio: float, rebound_ratio: float) -> float | None:
    signal_high = float(signal_row.get("high", np.nan))
    signal_low = float(signal_row.get("low", np.nan))
    signal_close = float(signal_row.get("close", np.nan))
    if not np.isfinite(signal_high) or not np.isfinite(signal_low) or not np.isfinite(signal_close) or signal_close <= 0:
        return None
    signal_range_pct = (signal_high - signal_low) / signal_close
    recent3_range_pct = _recent_range_pct(daily_df, signal_date, lookback=3)
    if recent3_range_pct is None:
        recent3_range_pct = signal_range_pct
    body_adj = float(np.clip(0.5 + 0.5 * max(body_ratio, 0.0), 0.5, 1.0))
    rebound_adj = float(np.clip(max(rebound_ratio, 0.0) / 2.0, 0.75, 1.5))
    vol_pct = (0.6 * signal_range_pct + 0.4 * recent3_range_pct) * body_adj * rebound_adj
    vol_pct = float(np.clip(vol_pct, 0.005, 0.30))
    return vol_pct if np.isfinite(vol_pct) and vol_pct > 0 else None


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    profiles = payload["profiles"]
    daily_path = base.DAILY_DIR / f"{code}.txt"
    if not daily_path.exists():
        return [], [{"code": code, "reason": "missing_daily_series"}]

    daily_df = base.tp_mod.hybrid.base.load_daily_df(daily_path)
    if daily_df is None or daily_df.empty:
        return [], [{"code": code, "reason": "empty_daily_df"}]
    daily_df = daily_df[(daily_df["date"] < base.EXCLUDE_START) | (daily_df["date"] > base.EXCLUDE_END)].copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_indexed = daily_df.set_index("date")

    min5_df = None
    min5_path = base.MIN5_DIR / f"{code}.txt"
    if min5_path.exists():
        loaded = base.tp_mod.hybrid.base.load_minute_df(min5_path)
        min5_df = base.tp_mod._prepare_min5_indicators(loaded) if loaded is not None and not loaded.empty else None

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in rows:
        signal_date = pd.Timestamp(item["signal_date"])
        if signal_date not in daily_indexed.index:
            skipped.append({"code": code, "signal_date": item["signal_date"], "reason": "missing_signal_row"})
            continue
        signal_row = daily_indexed.loc[signal_date]
        body_ratio = float(item.get("body_ratio", np.nan))
        rebound_ratio = float(item.get("rebound_ratio", np.nan))
        vol_pct = _shape_vol_pct(signal_row, daily_df, signal_date, body_ratio=body_ratio, rebound_ratio=rebound_ratio)
        if vol_pct is None:
            skipped.append({"code": code, "signal_date": item["signal_date"], "reason": "invalid_shapevol_pct"})
            continue
        for profile in profiles:
            tp_pct = float(profile["k_value"]) * vol_pct
            sim_profile = {
                "name": profile["name"],
                "family": "fixed_tp",
                "tp_pct": tp_pct,
                "hold_days": int(profile["hold_days"]),
                "exit_mode": EXIT_MODE,
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
                profile=sim_profile,
                stop_cfg=STOP_CFG,
                buy_gap_limit=float(payload["buy_gap_limit"]),
            )
            if sim.skipped or sim.trade is None:
                skipped.append(
                    {
                        "code": code,
                        "signal_date": item["signal_date"],
                        "entry_date": item["entry_date"],
                        "profile_name": profile["name"],
                        "reason": sim.skip_reason,
                    }
                )
                continue
            trade = sim.trade
            trade.update(
                {
                    "profile_name": profile["name"],
                    "profile_family": "shapevol_fixed_tp",
                    "k_value": float(profile["k_value"]),
                    "shapevol_pct": float(vol_pct),
                    "body_ratio_input": float(body_ratio),
                    "rebound_ratio_input": float(rebound_ratio),
                    "selected_tp_pct": float(tp_pct),
                    "selected_hold_days": int(profile["hold_days"]),
                    "stop_base": STOP_CFG["stop_base"],
                    "stop_exec_mode": STOP_CFG["stop_exec_mode"],
                }
            )
            trades.append(trade)
    return trades, skipped


def _summarize_signal_basket(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["profile_name", "stop_base", "stop_exec_mode"]
    for keys, g in trades.groupby(group_cols, sort=True):
        profile_name, stop_base, stop_exec_mode = keys
        strategy = f"case_rank_shapevol|{stop_base}|{stop_exec_mode}|{profile_name}"
        row = base.tp_mod.hybrid.base.summarize_trades(g, strategy)
        first = g.iloc[0]
        row.update(
            {
                "profile_name": profile_name,
                "profile_family": "shapevol_fixed_tp",
                "k_value": float(first["k_value"]),
                "hold_days": int(first["selected_hold_days"]),
                "avg_tp_pct": float(pd.to_numeric(g["selected_tp_pct"], errors="coerce").mean()),
                "median_tp_pct": float(pd.to_numeric(g["selected_tp_pct"], errors="coerce").median()),
                "avg_shapevol_pct": float(pd.to_numeric(g["shapevol_pct"], errors="coerce").mean()),
                "avg_body_ratio_input": float(pd.to_numeric(g["body_ratio_input"], errors="coerce").mean()),
                "avg_rebound_ratio_input": float(pd.to_numeric(g["rebound_ratio_input"], errors="coerce").mean()),
                "stop_base": stop_base,
                "stop_exec_mode": stop_exec_mode,
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
    base_summary = base._summarize_account(trades, result_dir, max_workers=max_workers)
    extra = (
        trades.groupby(["profile_name", "stop_base", "stop_exec_mode"], sort=True)
        .agg(
            k_value=("k_value", "first"),
            hold_days=("selected_hold_days", "first"),
            avg_tp_pct=("selected_tp_pct", "mean"),
            median_tp_pct=("selected_tp_pct", "median"),
            avg_shapevol_pct=("shapevol_pct", "mean"),
            avg_body_ratio_input=("body_ratio_input", "mean"),
            avg_rebound_ratio_input=("rebound_ratio_input", "mean"),
        )
        .reset_index()
    )
    return base_summary.merge(extra, on=["profile_name", "stop_base", "stop_exec_mode"], how="left")


def run_round(result_dir: Path, source_csv: Path, k_min: float, k_max: float, k_step: float, file_limit_codes: int, date_limit: int, max_workers: int) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
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
        file_limit_codes=file_limit_codes,
        date_limit=date_limit,
        max_workers=max_workers,
    )
    candidates = base.load_source_candidates(source_csv, file_limit_codes, date_limit)
    candidates.to_csv(result_dir / "source_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"k_value": k_values}).to_csv(result_dir / "k_grid.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        raise RuntimeError("源候选为空")
    if "signal_idx" not in candidates.columns:
        candidates["signal_idx"] = -1

    grouped_payloads = []
    cols = ["signal_date", "entry_date", "signal_idx", "sort_score", "signal_open", "signal_close", "signal_low", "body_ratio", "rebound_ratio"]
    for code, g in candidates.groupby("code", sort=True):
        grouped_payloads.append(
            {
                "code": str(code),
                "rows": g[cols].to_dict("records"),
                "buy_gap_limit": float(BUY_GAP_LIMIT),
                "profiles": profiles,
            }
        )

    total_codes = len(grouped_payloads)
    total_jobs = len(candidates) * len(profiles)
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
        "mode": "shapevol_fixed_tp_search",
        "k_min": k_min,
        "k_max": k_max,
        "k_step": k_step,
        "hold_days": HOLD_DAYS,
        "shape_proxy": "0.6*signal_range_pct + 0.4*recent3_range_pct, then * body_adj * rebound_adj",
        "fixed_stop_base": STOP_CFG["stop_base"],
        "fixed_stop_exec_mode": STOP_CFG["stop_exec_mode"],
        "fixed_exit_mode": EXIT_MODE,
        "best_signal_basket_profile": signal_summary.iloc[0].to_dict() if not signal_summary.empty else {},
        "best_account_profile": account_summary.iloc[0].to_dict() if not account_summary.empty else {},
        "signal_profile_count": int(len(signal_summary)),
        "account_profile_count": int(len(account_summary)),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished")
    return summary


def _best_k_edge(summary: dict[str, Any], k_min: float, k_max: float) -> tuple[str, dict[str, Any]]:
    best = summary.get("best_account_profile", {}) or {}
    k_value = float(best.get("k_value", np.nan))
    if not np.isfinite(k_value):
        return "none", best
    if abs(k_value - k_min) < 1e-12:
        return "lower", best
    if abs(k_value - k_max) < 1e-12:
        return "upper", best
    return "interior", best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="砖块/形态波动率动态止盈搜索：固定买点/止损/执行方式，只搜索 K × 持有天数，K 自动扩边直到不贴边")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--source-csv", type=str, default=str(SOURCE_CANDIDATES))
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--file-limit-codes", type=int, default=80)
    parser.add_argument("--date-limit", type=int, default=4)
    parser.add_argument("--start-k-min", type=float, default=0.25)
    parser.add_argument("--start-k-max", type=float, default=4.0)
    parser.add_argument("--k-step", type=float, default=0.125)
    parser.add_argument("--lower-cap", type=float, default=0.125)
    parser.add_argument("--upper-cap", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else RESULT_ROOT / f"brick_case_rank_shapevol_tp_search_v1_{args.mode}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    k_min = round(float(args.start_k_min), 4)
    k_max = round(float(args.start_k_max), 4)
    k_step = round(float(args.k_step), 4)
    lower_cap = round(float(args.lower_cap), 4)
    upper_cap = round(float(args.upper_cap), 4)

    file_limit_codes = 0 if args.mode == "full" else int(args.file_limit_codes)
    date_limit = 0 if args.mode == "full" else int(args.date_limit)

    rounds: list[dict[str, Any]] = []
    round_idx = 1

    try:
        while True:
            round_dir = output_root / f"round_{round_idx:02d}_k_{k_min:.4f}_{k_max:.4f}"
            summary = run_round(
                result_dir=round_dir,
                source_csv=Path(args.source_csv),
                k_min=k_min,
                k_max=k_max,
                k_step=k_step,
                file_limit_codes=file_limit_codes,
                date_limit=date_limit,
                max_workers=int(args.max_workers),
            )
            edge, best = _best_k_edge(summary, k_min, k_max)
            rounds.append(
                {
                    "result_dir": str(round_dir),
                    "k_min": k_min,
                    "k_max": k_max,
                    "best_account_profile": best,
                    "best_k_edge": edge,
                }
            )
            if edge == "interior":
                stopped = "interior_best_found"
                break
            if edge == "lower":
                next_min = round(k_min - k_step, 4)
                if next_min < lower_cap - 1e-12:
                    stopped = "hit_lower_cap"
                    break
                k_min = next_min
            elif edge == "upper":
                next_max = round(k_max + k_step, 4)
                if next_max > upper_cap + 1e-12:
                    stopped = "hit_upper_cap"
                    break
                k_max = next_max
            else:
                stopped = "missing_best_k"
                break
            round_idx += 1

        final = {
            "mode": args.mode,
            "source_csv": str(args.source_csv),
            "baseline_best_histvol_tp": _load_baseline_best(),
            "rounds": rounds,
            "final_round": rounds[-1] if rounds else {},
            "stopped_because": stopped,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        (output_root / "summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        write_error(output_root, exc)
        raise


if __name__ == "__main__":
    main()
