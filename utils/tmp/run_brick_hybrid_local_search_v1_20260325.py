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
SOURCE_RESULT_DIR = RESULT_ROOT / "brick_minute_execution_compare_v1_full_day5_parallel_20260325_r4"
SOURCE_CANDIDATES = SOURCE_RESULT_DIR / "selected_signals.csv"
SOURCE_COVERAGE = SOURCE_RESULT_DIR / "minute_files_coverage.csv"
DAILY_DIR = ROOT / "data" / "20260324"
MIN5_DIR = ROOT / "data" / "202603245min"
BASE_SCRIPT_PATH = ROOT / "utils" / "tmp" / "run_brick_intraday_minute_compare_v1_20260325.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INITIAL_CAPITAL = 1_000_000.0
BUY_GAP_LIMIT = 0.04
MAX_HOLD_DAYS = 3
TP_GRID = [0.055, 0.0575, 0.06, 0.0625, 0.065]
STOP_MULTIPLIERS = [1.0, 0.9975, 0.995, 0.9925, 0.99]
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))
STOP_BASE_CHOICES = {"entry_low", "min_oc"}


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT_PATH, "brick_hybrid_local_search_base_v1")


def resolve_stop_base(entry_open: float, entry_close: float, entry_low: float, stop_base: str) -> float:
    if stop_base == "entry_low":
        return entry_low
    if stop_base == "min_oc":
        return min(entry_open, entry_close)
    raise ValueError(f"未知止损基准: {stop_base}")


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


def load_candidates(file_limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(SOURCE_CANDIDATES, parse_dates=["signal_date", "entry_date"])
    coverage = pd.read_csv(SOURCE_COVERAGE)
    candidates = candidates.sort_values(["signal_date", "code", "signal_idx"]).reset_index(drop=True)
    if file_limit > 0:
        keep_codes = sorted(candidates["code"].astype(str).unique())[:file_limit]
        candidates = candidates[candidates["code"].astype(str).isin(keep_codes)].copy()
        coverage = coverage[coverage["code"].astype(str).isin(keep_codes)].copy()
    return candidates.reset_index(drop=True), coverage.reset_index(drop=True)


@dataclass
class SimResult:
    trade: dict[str, Any] | None
    skipped: bool = False
    skip_reason: str = ""


def simulate_one_trade(
    code: str,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    signal_idx: int,
    daily_df: pd.DataFrame,
    min5_df: pd.DataFrame | None,
    tp_pct: float,
    stop_multiplier: float,
    stop_base: str,
) -> SimResult:
    entry_row = daily_df[daily_df["date"] == entry_date]
    signal_row = daily_df[daily_df["date"] == signal_date]
    if entry_row.empty or signal_row.empty:
        return SimResult(trade=None, skipped=True, skip_reason="missing_daily_row")

    entry_row = entry_row.iloc[0]
    signal_row = signal_row.iloc[0]
    entry_open = float(entry_row["open"])
    entry_close = float(entry_row["close"])
    entry_low = float(entry_row["low"])
    signal_close = float(signal_row["close"])
    if not np.isfinite(signal_close) or signal_close <= 0:
        return SimResult(trade=None, skipped=True, skip_reason="invalid_signal_close")
    if not np.isfinite(entry_open) or entry_open <= 0:
        return SimResult(trade=None, skipped=True, skip_reason="invalid_entry_open")
    if not np.isfinite(entry_close) or entry_close <= 0:
        return SimResult(trade=None, skipped=True, skip_reason="invalid_entry_close")
    if not np.isfinite(entry_low) or entry_low <= 0:
        return SimResult(trade=None, skipped=True, skip_reason="invalid_entry_low")
    if entry_open / signal_close - 1.0 > BUY_GAP_LIMIT:
        return SimResult(trade=None, skipped=True, skip_reason="gap_gt_4pct")

    stop_base_price = resolve_stop_base(entry_open, entry_close, entry_low, stop_base)
    if not np.isfinite(stop_base_price) or stop_base_price <= 0:
        return SimResult(trade=None, skipped=True, skip_reason="invalid_stop_base_price")
    stop_price = stop_base_price * stop_multiplier
    tp_price = entry_open * (1.0 + tp_pct)
    gap_group = base.classify_gap(signal_close, entry_open)

    eligible_dates = daily_df[daily_df["date"] > entry_date]["date"].head(MAX_HOLD_DAYS).tolist()
    if not eligible_dates:
        return SimResult(trade=None, skipped=True, skip_reason="no_exit_window")

    trigger_date: pd.Timestamp | None = None
    trigger_reason: str | None = None
    trigger_price: float | None = None
    trigger_source: str | None = None

    for d in eligible_dates:
        day_min = pd.DataFrame()
        if min5_df is not None:
            day_min = min5_df[min5_df["date"] == d]
        if day_min.empty:
            day_daily = daily_df[daily_df["date"] == d]
            if day_daily.empty:
                continue
            day_high = float(day_daily.iloc[0]["high"])
            day_low = float(day_daily.iloc[0]["low"])
            if day_high >= tp_price:
                trigger_date = d
                trigger_reason = "tp"
                trigger_price = tp_price
                trigger_source = "daily_fallback"
                break
            if day_low <= stop_price:
                trigger_date = d
                trigger_reason = "sl"
                trigger_price = stop_price
                trigger_source = "daily_fallback"
                break
            continue
        for row in day_min.itertuples(index=False):
            day_high = float(row.high)
            day_low = float(row.low)
            if day_high >= tp_price:
                trigger_date = d
                trigger_reason = "tp"
                trigger_price = tp_price
                trigger_source = "5min"
                break
            if day_low <= stop_price:
                trigger_date = d
                trigger_reason = "sl"
                trigger_price = stop_price
                trigger_source = "5min"
                break
        if trigger_reason is not None:
            break

    if trigger_reason is None:
        exit_date = pd.Timestamp(eligible_dates[-1])
        exit_row = daily_df[daily_df["date"] == exit_date]
        if exit_row.empty:
            return SimResult(trade=None, skipped=True, skip_reason="missing_forced_exit_day")
        exit_price = float(exit_row.iloc[0]["close"])
        if not np.isfinite(exit_price) or exit_price <= 0:
            return SimResult(trade=None, skipped=True, skip_reason="invalid_forced_exit_price")
        exit_reason = "max_hold_close"
    else:
        if trigger_reason == "sl":
            exit_date = pd.Timestamp(trigger_date)
            exit_price = float(trigger_price)
            exit_reason = "sl_same_day_hybrid"
        else:
            later = daily_df[daily_df["date"] > trigger_date]["date"]
            if later.empty:
                exit_date = pd.Timestamp(trigger_date)
                exit_row = daily_df[daily_df["date"] == exit_date]
                exit_price = float(exit_row.iloc[0]["close"])
                if not np.isfinite(exit_price) or exit_price <= 0:
                    return SimResult(trade=None, skipped=True, skip_reason="invalid_same_day_fallback_close")
                exit_reason = "tp_same_day_hybrid_fallback_close"
            else:
                exit_date = pd.Timestamp(later.iloc[0])
                exit_row = daily_df[daily_df["date"] == exit_date]
                if exit_row.empty:
                    return SimResult(trade=None, skipped=True, skip_reason="missing_next_exec_day")
                exit_price = float(exit_row.iloc[0]["open"])
                if not np.isfinite(exit_price) or exit_price <= 0:
                    return SimResult(trade=None, skipped=True, skip_reason="invalid_next_open")
                exit_reason = "tp_next_open_hybrid"

    hold_days = int((pd.Timestamp(exit_date) - entry_date).days)
    return_pct = exit_price / entry_open - 1.0
    return SimResult(
        trade={
            "code": code,
            "signal_idx": signal_idx,
            "signal_date": signal_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_open,
            "exit_price": exit_price,
            "return_pct": return_pct,
            "hold_days": hold_days,
            "exit_reason": exit_reason,
            "trigger_take_profit": tp_price,
            "trigger_stop_loss": stop_price,
            "stop_base_price": stop_base_price,
            "gap_group": gap_group,
            "entry_gap_pct": entry_open / signal_close - 1.0,
            "minute_source": "5min" if min5_df is not None else "missing_5min",
            "trigger_source": trigger_source or "max_hold",
            "mode": "same_day_sl_nextday_tp",
            "tp_pct": tp_pct,
            "stop_multiplier": stop_multiplier,
            "stop_base": stop_base,
            "strategy_key": f"same_day_sl_nextday_tp|{stop_base}_x_{stop_multiplier:.4f}|tp_{tp_pct:.4f}",
        }
    )


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    has_5min = bool(payload["has_5min"])
    stop_base = str(payload["stop_base"])
    daily_path = DAILY_DIR / f"{code}.txt"
    if not daily_path.exists():
        skipped = []
        for item in rows:
            for tp_pct in TP_GRID:
                for stop_multiplier in STOP_MULTIPLIERS:
                    skipped.append(
                        {
                            "code": code,
                            "signal_date": item["signal_date"],
                            "entry_date": item["entry_date"],
                            "tp_pct": tp_pct,
                            "stop_multiplier": stop_multiplier,
                            "reason": "missing_daily_series",
                        }
                    )
        return [], skipped

    daily_df = base.load_daily_df(daily_path)
    min5_df = None
    if has_5min:
        min5_path = MIN5_DIR / f"{code}.txt"
        if min5_path.exists():
            min5_df = base.load_minute_df(min5_path)
            if min5_df.empty:
                min5_df = None

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in rows:
        signal_date = pd.Timestamp(item["signal_date"])
        entry_date = pd.Timestamp(item["entry_date"])
        signal_idx = int(item["signal_idx"])
        for tp_pct in TP_GRID:
            for stop_multiplier in STOP_MULTIPLIERS:
                result = simulate_one_trade(
                    code=code,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    signal_idx=signal_idx,
                    daily_df=daily_df,
                    min5_df=min5_df,
                    tp_pct=tp_pct,
                    stop_multiplier=stop_multiplier,
                    stop_base=stop_base,
                )
                if result.skipped or result.trade is None:
                    skipped.append(
                        {
                            "code": code,
                            "signal_date": signal_date,
                            "entry_date": entry_date,
                            "tp_pct": tp_pct,
                            "stop_multiplier": stop_multiplier,
                            "stop_base": stop_base,
                            "reason": result.skip_reason,
                        }
                    )
                else:
                    trades.append(result.trade)
    return trades, skipped


def run_compare(file_limit: int, output_dir: Path, max_workers: int, stop_base: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(output_dir, "loading_cached_inputs", file_limit=file_limit, max_workers=max_workers, stop_base=stop_base)
    candidates, coverage = load_candidates(file_limit)
    candidates.to_csv(output_dir / "selected_signals.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / "minute_files_coverage.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        raise RuntimeError("候选为空，无法比较")

    min5_codes = set(coverage.loc[coverage["has_5min"].astype(bool), "code"].astype(str))
    grouped_payloads = []
    for code, g in candidates.sort_values(["code", "signal_date", "signal_idx"]).groupby("code", sort=True):
        grouped_payloads.append(
            {
                "code": code,
                "has_5min": bool(code in min5_codes),
                "rows": [
                    {
                        "signal_idx": int(r.signal_idx),
                        "signal_date": str(pd.Timestamp(r.signal_date)),
                        "entry_date": str(pd.Timestamp(r.entry_date)),
                    }
                    for r in g.itertuples(index=False)
                ],
                "stop_base": stop_base,
            }
        )

    update_progress(
        output_dir,
        "data_ready",
        candidate_count=len(candidates),
        code_count=len(grouped_payloads),
        strategy_count=len(TP_GRID) * len(STOP_MULTIPLIERS),
        stop_base=stop_base,
    )

    trade_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    total_codes = len(grouped_payloads)
    total_jobs = len(candidates) * len(TP_GRID) * len(STOP_MULTIPLIERS)
    completed_codes = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(simulate_code_bundle, payload): payload["code"] for payload in grouped_payloads}
        for future in as_completed(future_map):
            completed_codes += 1
            code = future_map[future]
            trades_part, skipped_part = future.result()
            trade_rows.extend(trades_part)
            skipped_rows.extend(skipped_part)
            if completed_codes == 1 or completed_codes % 25 == 0 or completed_codes == total_codes:
                update_progress(
                    output_dir,
                    "simulating_trades",
                    done_codes=completed_codes,
                    total_codes=total_codes,
                    done_jobs=len(trade_rows) + len(skipped_rows),
                    total_jobs=total_jobs,
                    last_code=code,
                )

    trades = pd.DataFrame(trade_rows).sort_values(["stop_multiplier", "tp_pct", "signal_date", "code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows).sort_values(["stop_multiplier", "tp_pct", "signal_date", "code"], na_position="last").reset_index(drop=True) if skipped_rows else pd.DataFrame()
    trades.to_csv(output_dir / "hybrid_local_trades.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(output_dir / "hybrid_local_skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(output_dir, "trades_ready", trade_count=len(trades), skipped_count=len(skipped))

    summary_rows = []
    for (stop_multiplier, tp_pct), g in trades.groupby(["stop_multiplier", "tp_pct"], sort=True):
        strategy = f"same_day_sl_nextday_tp|{stop_base}_x_{stop_multiplier:.4f}|tp_{tp_pct:.4f}"
        row = base.summarize_trades(g, strategy)
        row["mode"] = "same_day_sl_nextday_tp"
        row["stop_base"] = stop_base
        row["stop_multiplier"] = stop_multiplier
        row["tp_pct"] = tp_pct
        row["buy_gap_limit"] = BUY_GAP_LIMIT
        row["max_hold_days"] = MAX_HOLD_DAYS
        row["avg_entry_gap_pct"] = float(g["entry_gap_pct"].mean())
        row["trigger_source_5min_ratio"] = float((g["trigger_source"] == "5min").mean())
        row["trigger_source_daily_ratio"] = float((g["trigger_source"] == "daily_fallback").mean())
        summary_rows.append(row)
        slug = strategy.replace("|", "__")
        base._portfolio_curve(g).to_csv(output_dir / f"equity_{slug}.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["annual_return_signal_basket", "avg_trade_return", "success_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary_df.to_csv(output_dir / "hybrid_local_summary.csv", index=False, encoding="utf-8-sig")

    best_by_stop = []
    for stop_multiplier, g in summary_df.groupby("stop_multiplier", sort=True):
        best_by_stop.append(g.iloc[0].to_dict())
    pd.DataFrame(best_by_stop).to_csv(output_dir / "best_by_stop_multiplier.csv", index=False, encoding="utf-8-sig")

    best_by_tp = []
    for tp_pct, g in summary_df.groupby("tp_pct", sort=True):
        best_by_tp.append(g.iloc[0].to_dict())
    pd.DataFrame(best_by_tp).to_csv(output_dir / "best_by_tp.csv", index=False, encoding="utf-8-sig")

    summary_json = {
        "candidate_count": int(len(candidates)),
        "trade_count": int(len(trades)),
        "skipped_count": int(len(skipped)),
        "strategy_best": summary_df.iloc[0]["strategy"] if not summary_df.empty else None,
        "stop_multiplier_best": float(summary_df.iloc[0]["stop_multiplier"]) if not summary_df.empty else None,
        "tp_best": float(summary_df.iloc[0]["tp_pct"]) if not summary_df.empty else None,
        "summary_rows": summary_rows,
        "best_by_stop_multiplier": best_by_stop,
        "best_by_tp": best_by_tp,
        "assumptions": {
            "signal_pool": "brick.formal_best",
            "mode": "same_day_sl_nextday_tp",
            "stop_base": stop_base,
            "stop_multipliers": STOP_MULTIPLIERS,
            "tp_grid": TP_GRID,
            "buy_gap_limit_pct": BUY_GAP_LIMIT,
            "max_hold_days": MAX_HOLD_DAYS,
            "buy_day_cannot_sell": True,
            "same_bar_priority": "take_profit_first",
            "minute_priority": "5min_then_daily_fallback",
            "candidate_source": str(SOURCE_CANDIDATES),
            "coverage_source": str(SOURCE_COVERAGE),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(output_dir, "finished", output_dir=str(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK 当前最优附近的局部参数搜索")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--file-limit", type=int, default=120)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--stop-base", choices=sorted(STOP_BASE_CHOICES), default="entry_low")
    parser.add_argument("--tp-grid", type=str, default="")
    parser.add_argument("--stop-multipliers", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global TP_GRID, STOP_MULTIPLIERS
    if args.tp_grid:
        TP_GRID = [float(x.strip()) for x in str(args.tp_grid).split(",") if x.strip()]
    if args.stop_multipliers:
        STOP_MULTIPLIERS = [float(x.strip()) for x in str(args.stop_multipliers).split(",") if x.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_hybrid_local_search_v1_{args.mode}_{timestamp}"
    file_limit = int(args.file_limit)
    if args.mode == "full":
        file_limit = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_compare(
            file_limit=file_limit,
            output_dir=output_dir,
            max_workers=max(1, int(args.max_workers)),
            stop_base=str(args.stop_base),
        )
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
