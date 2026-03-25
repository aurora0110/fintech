from __future__ import annotations

import argparse
import os
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
DAILY_DIR = ROOT / "data" / "20260324"
MIN5_DIR = ROOT / "data" / "202603245min"
RESULT_ROOT = ROOT / "results"

BRICK_BASE_PATH = ROOT / "utils" / "backtest" / "run_momentum_tail_experiment.py"
BRICK_RANKING_PATH = ROOT / "utils" / "backtest" / "compare_momentum_tail_ranking_models.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
INITIAL_CAPITAL = 1_000_000.0
TOP_N = 10
BUY_GAP_LIMIT = 0.04
TAKE_PROFIT = 0.03
MAX_HOLD_DAYS = 3
EXECUTION_MODES = ["same_day_exit", "nextday_open", "same_day_sl_nextday_tp"]
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


brick_base = load_module(BRICK_BASE_PATH, "brick_minute_compare_base_day5")
brick_ranking = load_module(BRICK_RANKING_PATH, "brick_minute_compare_ranking_day5")


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


def read_text_auto(path: Path) -> list[str]:
    for enc in ("gbk", "utf-8", "latin1"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except Exception:
            pass
    raise RuntimeError(f"无法读取文件: {path}")


def load_daily_df(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in read_text_auto(path)[2:]:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        rows.append(
            {
                "date": pd.to_datetime(parts[0], errors="coerce"),
                "open": pd.to_numeric(parts[1], errors="coerce"),
                "high": pd.to_numeric(parts[2], errors="coerce"),
                "low": pd.to_numeric(parts[3], errors="coerce"),
                "close": pd.to_numeric(parts[4], errors="coerce"),
                "volume": pd.to_numeric(parts[5], errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return df[(df["date"] < EXCLUDE_START) | (df["date"] > EXCLUDE_END)].copy()


def load_minute_df(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in read_text_auto(path)[2:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        date_text = parts[0]
        time_text = str(parts[1]).zfill(4)
        rows.append(
            {
                "datetime": pd.to_datetime(f"{date_text} {time_text}", format="%Y/%m/%d %H%M", errors="coerce"),
                "date": pd.to_datetime(date_text, errors="coerce"),
                "time": time_text,
                "open": pd.to_numeric(parts[2], errors="coerce"),
                "high": pd.to_numeric(parts[3], errors="coerce"),
                "low": pd.to_numeric(parts[4], errors="coerce"),
                "close": pd.to_numeric(parts[5], errors="coerce"),
                "volume": pd.to_numeric(parts[6], errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["datetime", "date", "time", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    return df.dropna(subset=["datetime", "date", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)


def build_daily_map(data_dir: Path, file_limit: int = 0, codes: set[str] | None = None) -> dict[str, pd.DataFrame]:
    files = sorted(data_dir.glob("*.txt"))
    if codes is not None:
        files = [p for p in files if p.stem in codes]
    elif file_limit > 0:
        files = files[:file_limit]
    out: dict[str, pd.DataFrame] = {}
    for path in files:
        df = load_daily_df(path)
        if df.empty:
            continue
        out[path.stem] = df
    return out


def extract_formal_best_rows_for_file(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    combo = brick_base.Combo(
        rebound_threshold=1.2,
        gain_limit=0.08,
        take_profit=0.03,
        stop_mode="entry_low_x_0.99",
    )
    raw_df = load_daily_df(path)
    if raw_df.empty or len(raw_df) < 120:
        return []
    raw_df["code"] = path.stem
    df = brick_base.build_feature_df(raw_df)
    df = brick_ranking.add_long_line(df)
    mask_a = df["pattern_a"] & (df["rebound_ratio"] >= combo.rebound_threshold)
    mask_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
    mask = (
        df["signal_base"]
        & (df["ret1"] <= combo.gain_limit)
        & (mask_a | mask_b)
        & (df["trend_line"] > df["long_line"])
    )
    rows: list[dict[str, Any]] = []
    signal_idxs = np.flatnonzero(mask.to_numpy())
    for signal_idx in signal_idxs:
        signal_idx = int(signal_idx)
        entry_idx = signal_idx + 1
        if entry_idx >= len(df):
            continue
        feat = brick_ranking.compute_signal_features(df, signal_idx)
        rows.append(
            {
                "code": path.stem,
                "signal_idx": signal_idx,
                "signal_date": pd.Timestamp(df.at[signal_idx, "date"]),
                "entry_date": pd.Timestamp(df.at[entry_idx, "date"]),
                "candidate_pool": "brick.formal_best",
                "strategy_family": "brick",
                "signal_type": "brick_formal_best",
                "pattern_a": bool(df.at[signal_idx, "pattern_a"]),
                "pattern_b": bool(df.at[signal_idx, "pattern_b"]),
                "ret1": float(df.at[signal_idx, "ret1"]),
                "rebound_ratio": float(df.at[signal_idx, "rebound_ratio"]),
                "brick_red_len": float(df.at[signal_idx, "brick_red_len"]),
                "signal_vs_ma5": float(df.at[signal_idx, "signal_vs_ma5"]),
                "trend_line": float(df.at[signal_idx, "trend_line"]),
                "long_line": float(df.at[signal_idx, "long_line"]),
                "signal_low": float(df.at[signal_idx, "low"]),
                "base_score": 0.0,
                **feat,
            }
        )
    return rows


def build_min5_registry(codes: set[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for code in sorted(codes):
        path = MIN5_DIR / f"{code}.txt"
        if path.exists():
            out[code] = path
    return out


class Minute5Store:
    def __init__(self, registry: dict[str, Path]) -> None:
        self.registry = registry
        self.cache: dict[str, pd.DataFrame | None] = {}

    def has(self, code: str) -> bool:
        return code in self.registry

    def get(self, code: str) -> pd.DataFrame | None:
        if code in self.cache:
            return self.cache[code]
        path = self.registry.get(code)
        if path is None:
            self.cache[code] = None
            return None
        df = load_minute_df(path)
        self.cache[code] = df if not df.empty else None
        return self.cache[code]


def daily_path_for_code(code: str) -> Path:
    return DAILY_DIR / f"{code}.txt"


def classify_gap(signal_close: float, entry_open: float) -> str:
    if not np.isfinite(signal_close) or signal_close <= 0 or not np.isfinite(entry_open) or entry_open <= 0:
        return "gap_invalid"
    gap = entry_open / signal_close - 1.0
    if gap > 1e-12:
        return "gap_up"
    if gap < -1e-12:
        return "gap_down"
    return "gap_flat"


def build_formal_best_candidates(
    file_limit: int = 0,
    progress_cb: Callable[[int, int], None] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    files = sorted(DAILY_DIR.glob("*.txt"))
    if file_limit > 0:
        files = files[:file_limit]
    total = len(files)
    path_strs = [str(path) for path in files]
    if total == 0:
        return pd.DataFrame()

    if max_workers <= 1:
        for idx, path_str in enumerate(path_strs, start=1):
            if progress_cb is not None and (idx == 1 or idx % 20 == 0 or idx == total):
                progress_cb(idx, total)
            rows.extend(extract_formal_best_rows_for_file(path_str))
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(extract_formal_best_rows_for_file, path_str): path_str for path_str in path_strs}
                completed = 0
                for future in as_completed(future_map):
                    completed += 1
                    if progress_cb is not None and (completed == 1 or completed % 20 == 0 or completed == total):
                        progress_cb(completed, total)
                    rows.extend(future.result())
        except Exception as exc:
            # 并行失败时回退串行，确保 full 不因为环境限制直接失败。
            if progress_cb is not None:
                progress_cb(0, total, parallel_error=type(exc).__name__, parallel_error_message=str(exc))
            rows = []
            for idx, path_str in enumerate(path_strs, start=1):
                if progress_cb is not None and (idx == 1 or idx % 20 == 0 or idx == total):
                    progress_cb(idx, total, fallback="serial", parallel_error=type(exc).__name__, parallel_error_message=str(exc))
                rows.extend(extract_formal_best_rows_for_file(path_str))

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)
    out = brick_ranking.assign_rank_scores(out)
    out["base_score"] = brick_ranking.build_sort_score(out, "shrink_focus")
    out["daily_pct_rank"] = out.groupby("signal_date")["base_score"].rank(pct=True, method="first")
    out = out[out["daily_pct_rank"] >= 0.50].copy()
    out = out.sort_values(["signal_date", "base_score", "code"], ascending=[True, False, True])
    out = out.groupby("signal_date", group_keys=False).head(TOP_N).reset_index(drop=True)
    return out


def _portfolio_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "daily_ret", "equity"])
    x = trades.copy()
    x["signal_date"] = pd.to_datetime(x["signal_date"])
    daily = x.groupby("signal_date", as_index=False)["return_pct"].mean()
    daily = daily.sort_values("signal_date").reset_index(drop=True)
    daily["equity"] = INITIAL_CAPITAL * (1.0 + daily["return_pct"]).cumprod()
    daily["date"] = daily["signal_date"]
    daily["daily_ret"] = daily["return_pct"]
    return daily[["date", "daily_ret", "equity"]]


def summarize_trades(trades: pd.DataFrame, strategy: str) -> dict[str, Any]:
    if trades.empty:
        return {
            "strategy": strategy,
            "trade_count": 0,
            "avg_trade_return": 0.0,
            "success_rate": 0.0,
            "avg_holding_days": 0.0,
            "profit_factor": 0.0,
            "annual_return_signal_basket": 0.0,
            "max_drawdown_signal_basket": 0.0,
            "final_equity_signal_basket": INITIAL_CAPITAL,
        }
    equity = _portfolio_curve(trades)
    rets = trades["return_pct"].astype(float)
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if not losses.empty and abs(losses.sum()) > 1e-12 else float("inf")
    dd = equity["equity"] / equity["equity"].cummax() - 1.0
    years = max((equity["date"].iloc[-1] - equity["date"].iloc[0]).days / 365.25, 1 / 365.25)
    annual = float((equity["equity"].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) if len(equity) > 1 else float(rets.mean() * 252)
    return {
        "strategy": strategy,
        "trade_count": int(len(trades)),
        "avg_trade_return": float(rets.mean()),
        "success_rate": float((rets > 0).mean()),
        "avg_holding_days": float(trades["hold_days"].mean()),
        "profit_factor": profit_factor,
        "annual_return_signal_basket": annual,
        "max_drawdown_signal_basket": float(dd.min()) if not dd.empty else 0.0,
        "final_equity_signal_basket": float(equity["equity"].iloc[-1]),
    }


@dataclass
class SimResult:
    trade: dict[str, Any] | None
    skipped: bool = False
    skip_reason: str = ""


def minute_trigger_trade(
    code: str,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    signal_idx: int,
    daily_df: pd.DataFrame,
    min5_df: pd.DataFrame | None,
    execution_mode: str,
) -> SimResult:
    entry_row = daily_df[daily_df["date"] == entry_date]
    signal_row = daily_df[daily_df["date"] == signal_date]
    if entry_row.empty or signal_row.empty:
        return SimResult(trade=None, skipped=True, skip_reason="missing_daily_row")

    entry_open = float(entry_row.iloc[0]["open"])
    signal_close = float(signal_row.iloc[0]["close"])
    signal_low = float(signal_row.iloc[0]["low"])
    if not np.isfinite(signal_close) or signal_close <= 0:
        return SimResult(trade=None, skipped=True, skip_reason="invalid_signal_close")
    if not np.isfinite(entry_open) or entry_open <= 0:
        return SimResult(trade=None, skipped=True, skip_reason="invalid_entry_open")
    if not np.isfinite(signal_low) or signal_low <= 0:
        return SimResult(trade=None, skipped=True, skip_reason="invalid_signal_low")
    if entry_open / signal_close - 1.0 > BUY_GAP_LIMIT:
        return SimResult(trade=None, skipped=True, skip_reason="gap_gt_4pct")

    stop_price = signal_low * 0.99
    tp_price = entry_open * (1.0 + TAKE_PROFIT)
    gap_group = classify_gap(signal_close, entry_open)

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
        if execution_mode == "same_day_exit" or (execution_mode == "same_day_sl_nextday_tp" and trigger_reason == "sl"):
            exit_date = pd.Timestamp(trigger_date)
            exit_price = float(trigger_price)
            suffix = "same_day" if execution_mode == "same_day_exit" else "same_day_hybrid"
            exit_reason = f"{trigger_reason}_{suffix}"
        else:
            later = daily_df[daily_df["date"] > trigger_date]["date"]
            if later.empty:
                exit_date = pd.Timestamp(trigger_date)
                exit_row = daily_df[daily_df["date"] == exit_date]
                exit_price = float(exit_row.iloc[0]["close"])
                if not np.isfinite(exit_price) or exit_price <= 0:
                    return SimResult(trade=None, skipped=True, skip_reason="invalid_same_day_fallback_close")
                suffix = "same_day_fallback_close" if execution_mode == "nextday_open" else "same_day_hybrid_fallback_close"
                exit_reason = f"{trigger_reason}_{suffix}"
            else:
                exit_date = pd.Timestamp(later.iloc[0])
                exit_row = daily_df[daily_df["date"] == exit_date]
                if exit_row.empty:
                    return SimResult(trade=None, skipped=True, skip_reason="missing_next_exec_day")
                exit_price = float(exit_row.iloc[0]["open"])
                if not np.isfinite(exit_price) or exit_price <= 0:
                    return SimResult(trade=None, skipped=True, skip_reason="invalid_next_open")
                suffix = "next_open" if execution_mode == "nextday_open" else "next_open_hybrid"
                exit_reason = f"{trigger_reason}_{suffix}"

    hold_days = int((pd.Timestamp(exit_date) - entry_date).days)
    return_pct = exit_price / entry_open - 1.0
    return SimResult(
        trade={
            "code": code,
            "signal_idx": signal_idx,
            "signal_date": signal_date,
            "entry_date": entry_date,
            "exit_date": pd.Timestamp(exit_date),
            "entry_price": entry_open,
            "exit_price": exit_price,
            "return_pct": return_pct,
            "hold_days": hold_days,
            "exit_reason": exit_reason,
            "trigger_take_profit": tp_price,
            "trigger_stop_loss": stop_price,
            "gap_group": gap_group,
            "entry_gap_pct": entry_open / signal_close - 1.0,
            "minute_source": "5min" if min5_df is not None else "missing_5min",
            "trigger_source": trigger_source or "max_hold",
            "execution_mode": execution_mode,
        }
    )


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    min5_available = bool(payload["has_5min"])
    daily_path = daily_path_for_code(code)
    if not daily_path.exists():
        skipped = []
        for item in rows:
            for mode in EXECUTION_MODES:
                skipped.append(
                    {
                        "code": code,
                        "signal_date": item["signal_date"],
                        "entry_date": item["entry_date"],
                        "mode": mode,
                        "reason": "missing_daily_series",
                    }
                )
        return [], skipped

    daily_df = load_daily_df(daily_path)
    min5_df = None
    if min5_available:
        min5_path = MIN5_DIR / f"{code}.txt"
        if min5_path.exists():
            min5_df = load_minute_df(min5_path)
            if min5_df.empty:
                min5_df = None

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in rows:
        signal_date = pd.Timestamp(item["signal_date"])
        entry_date = pd.Timestamp(item["entry_date"])
        signal_idx = int(item["signal_idx"])
        for mode in EXECUTION_MODES:
            result = minute_trigger_trade(
                code=code,
                signal_date=signal_date,
                entry_date=entry_date,
                signal_idx=signal_idx,
                daily_df=daily_df,
                min5_df=min5_df,
                execution_mode=mode,
            )
            if result.skipped or result.trade is None:
                skipped.append(
                    {
                        "code": code,
                        "signal_date": signal_date,
                        "entry_date": entry_date,
                        "mode": mode,
                        "reason": result.skip_reason,
                    }
                )
            else:
                trades.append(result.trade)
    return trades, skipped


def build_minute_coverage(candidates: pd.DataFrame, min5_store: Minute5Store) -> pd.DataFrame:
    rows = []
    for code, g in candidates.groupby("code"):
        rows.append(
            {
                "code": code,
                "signal_count": int(len(g)),
                "has_5min": bool(min5_store.has(code)),
                "minute_source": "5min" if min5_store.has(code) else "daily_only",
            }
        )
    return pd.DataFrame(rows).sort_values(["has_5min", "signal_count", "code"], ascending=[False, False, True]).reset_index(drop=True)


def run_compare(file_limit: int, output_dir: Path, max_workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(output_dir, "loading_candidates", file_limit=file_limit, daily_dir=str(DAILY_DIR), minute_dir=str(MIN5_DIR), max_workers=max_workers)

    def candidate_progress(done: int, total: int, **extra: Any) -> None:
        update_progress(output_dir, "loading_candidates", file_limit=file_limit, scanned_files=done, total_files=total, max_workers=max_workers, **extra)

    candidates = build_formal_best_candidates(file_limit=file_limit, progress_cb=candidate_progress, max_workers=max_workers)
    if not candidates.empty:
        invalid_mask = ~pd.to_numeric(candidates["signal_low"], errors="coerce").gt(0)
        candidates = candidates[~invalid_mask].copy()
    candidates.to_csv(output_dir / "selected_signals.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        empty = pd.DataFrame()
        for name in [
            "execution_compare_trades.csv",
            "execution_compare_skipped.csv",
            "execution_compare_summary.csv",
            "gap_group_summary.csv",
            "minute_files_coverage.csv",
            "equity_same_day.csv",
            "equity_nextday.csv",
        ]:
            empty.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
        summary_json = {
            "candidate_count": 0,
            "trade_count": 0,
            "skipped_count": 0,
            "mode_best": None,
            "summary_rows": [],
            "gap_group_rows": [],
            "comparison": {},
            "assumptions": {
                "signal_pool": "brick.formal_best",
                "buy_gap_limit_pct": BUY_GAP_LIMIT,
                "take_profit_pct": TAKE_PROFIT,
                "stop_loss_rule": "signal_low_x_0.99",
                "max_hold_days": MAX_HOLD_DAYS,
                "buy_day_cannot_sell": True,
                "same_bar_priority": "take_profit_first",
                "minute_priority": "5min_then_daily_fallback",
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        update_progress(output_dir, "finished", candidate_count=0)
        return

    candidate_codes = set(candidates["code"].astype(str))
    min5_registry = build_min5_registry(candidate_codes)
    min5_store = Minute5Store(min5_registry)
    coverage_df = build_minute_coverage(candidates, min5_store)
    coverage_df.to_csv(output_dir / "minute_files_coverage.csv", index=False, encoding="utf-8-sig")
    update_progress(
        output_dir,
        "data_ready",
        candidate_count=len(candidates),
        daily_count=int(candidates["code"].nunique()),
        min5_count=len(min5_registry),
        min5_coverage_ratio=float(coverage_df["has_5min"].mean()) if not coverage_df.empty else 0.0,
    )

    grouped_payloads = []
    grouped = candidates.sort_values(["code", "signal_date", "signal_idx"]).groupby("code", sort=True)
    for code, g in grouped:
        grouped_payloads.append(
            {
                "code": code,
                "has_5min": bool(code in min5_registry),
                "rows": [
                    {
                        "signal_idx": int(r.signal_idx),
                        "signal_date": str(pd.Timestamp(r.signal_date)),
                        "entry_date": str(pd.Timestamp(r.entry_date)),
                    }
                    for r in g.itertuples(index=False)
                ],
            }
        )

    trade_rows = []
    skipped_rows = []
    total_codes = len(grouped_payloads)
    total_jobs = len(candidates) * len(EXECUTION_MODES)
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
                done_jobs = len(trade_rows) + len(skipped_rows)
                update_progress(
                    output_dir,
                    "simulating_trades",
                    done_codes=completed_codes,
                    total_codes=total_codes,
                    done_jobs=done_jobs,
                    total_jobs=total_jobs,
                    last_code=code,
                )

    trades = pd.DataFrame(trade_rows).sort_values(["execution_mode", "signal_date", "code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows).sort_values(["mode", "signal_date", "code"], na_position="last").reset_index(drop=True) if skipped_rows else pd.DataFrame()
    trades.to_csv(output_dir / "execution_compare_trades.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(output_dir / "execution_compare_skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(output_dir, "trades_ready", trade_count=len(trades), skipped_count=len(skipped))

    summary_rows = []
    gap_rows = []
    equity_by_mode: dict[str, pd.DataFrame] = {}
    for mode, mode_df in trades.groupby("execution_mode"):
        summary = summarize_trades(mode_df, mode)
        summary["mode"] = mode
        summary["buy_gap_limit"] = BUY_GAP_LIMIT
        summary["take_profit_pct"] = TAKE_PROFIT
        summary["stop_loss_rule"] = "signal_low_x_0.99"
        summary["max_hold_days"] = MAX_HOLD_DAYS
        summary["avg_entry_gap_pct"] = float(mode_df["entry_gap_pct"].mean())
        summary["trigger_source_5min_ratio"] = float((mode_df["trigger_source"] == "5min").mean())
        summary["trigger_source_daily_ratio"] = float((mode_df["trigger_source"] == "daily_fallback").mean())
        summary_rows.append(summary)
        equity_by_mode[mode] = _portfolio_curve(mode_df)
        for gap_group, gap_df in mode_df.groupby("gap_group"):
            row = summarize_trades(gap_df, f"{mode}|{gap_group}")
            row["mode"] = mode
            row["gap_group"] = gap_group
            row["mean_entry_gap_pct"] = float(gap_df["entry_gap_pct"].mean())
            gap_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("final_equity_signal_basket", ascending=False)
    gap_df = pd.DataFrame(gap_rows).sort_values(["mode", "gap_group"]) if gap_rows else pd.DataFrame()
    summary_df.to_csv(output_dir / "execution_compare_summary.csv", index=False, encoding="utf-8-sig")
    gap_df.to_csv(output_dir / "gap_group_summary.csv", index=False, encoding="utf-8-sig")
    equity_by_mode.get("same_day_exit", pd.DataFrame()).to_csv(output_dir / "equity_same_day.csv", index=False, encoding="utf-8-sig")
    equity_by_mode.get("nextday_open", pd.DataFrame()).to_csv(output_dir / "equity_nextday.csv", index=False, encoding="utf-8-sig")
    equity_by_mode.get("same_day_sl_nextday_tp", pd.DataFrame()).to_csv(output_dir / "equity_hybrid.csv", index=False, encoding="utf-8-sig")

    comparison = {}
    mode_set = set(summary_df["mode"]) if not summary_df.empty else set()
    if {"same_day_exit", "nextday_open"} <= mode_set:
        same_day = summary_df[summary_df["mode"] == "same_day_exit"].iloc[0].to_dict()
        nextday = summary_df[summary_df["mode"] == "nextday_open"].iloc[0].to_dict()
        comparison = {
            "nextday_minus_same_day_total_return": float(nextday["final_equity_signal_basket"] / INITIAL_CAPITAL - same_day["final_equity_signal_basket"] / INITIAL_CAPITAL),
            "nextday_minus_same_day_avg_trade_return": float(nextday["avg_trade_return"] - same_day["avg_trade_return"]),
            "nextday_minus_same_day_success_rate": float(nextday["success_rate"] - same_day["success_rate"]),
            "nextday_minus_same_day_max_drawdown": float(nextday["max_drawdown_signal_basket"] - same_day["max_drawdown_signal_basket"]),
        }
        if "same_day_sl_nextday_tp" in mode_set:
            hybrid = summary_df[summary_df["mode"] == "same_day_sl_nextday_tp"].iloc[0].to_dict()
            comparison.update(
                {
                    "hybrid_minus_same_day_total_return": float(hybrid["final_equity_signal_basket"] / INITIAL_CAPITAL - same_day["final_equity_signal_basket"] / INITIAL_CAPITAL),
                    "hybrid_minus_same_day_avg_trade_return": float(hybrid["avg_trade_return"] - same_day["avg_trade_return"]),
                    "hybrid_minus_same_day_success_rate": float(hybrid["success_rate"] - same_day["success_rate"]),
                    "hybrid_minus_same_day_max_drawdown": float(hybrid["max_drawdown_signal_basket"] - same_day["max_drawdown_signal_basket"]),
                    "hybrid_minus_nextday_total_return": float(hybrid["final_equity_signal_basket"] / INITIAL_CAPITAL - nextday["final_equity_signal_basket"] / INITIAL_CAPITAL),
                    "hybrid_minus_nextday_avg_trade_return": float(hybrid["avg_trade_return"] - nextday["avg_trade_return"]),
                    "hybrid_minus_nextday_success_rate": float(hybrid["success_rate"] - nextday["success_rate"]),
                    "hybrid_minus_nextday_max_drawdown": float(hybrid["max_drawdown_signal_basket"] - nextday["max_drawdown_signal_basket"]),
                }
            )

    summary_json = {
        "candidate_count": int(len(candidates)),
        "trade_count": int(len(trades)),
        "skipped_count": int(len(skipped)),
        "mode_best": summary_df.iloc[0]["mode"] if not summary_df.empty else None,
        "summary_rows": summary_rows,
        "gap_group_rows": gap_rows,
        "comparison": comparison,
        "assumptions": {
            "signal_pool": "brick.formal_best",
            "buy_gap_limit_pct": BUY_GAP_LIMIT,
            "take_profit_pct": TAKE_PROFIT,
            "stop_loss_rule": "signal_low_x_0.99",
            "max_hold_days": MAX_HOLD_DAYS,
            "buy_day_cannot_sell": True,
            "same_bar_priority": "take_profit_first",
            "minute_priority": "5min_then_daily_fallback",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(output_dir, "finished", output_dir=str(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK 日线+5分钟线 当日卖出/次日开盘卖出 对比实验")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--file-limit", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_minute_execution_compare_v1_{args.mode}_{timestamp}"
    file_limit = int(args.file_limit)
    if args.mode == "full":
        file_limit = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_compare(file_limit=file_limit, output_dir=output_dir, max_workers=max(1, int(args.max_workers)))
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
