from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
DAILY_DIR = ROOT / "data" / "20260324"
CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
RAW_CASE_DIR = ROOT / "data" / "20260312"
SIGNAL_SOURCE = RESULT_ROOT / "brick_minute_execution_compare_v1_full_day5_parallel_20260325_r4" / "selected_signals.csv"
CHAMPION_RESULT_DIR = RESULT_ROOT / "brick_hybrid_local_search_minoc_full_20260326_r2"
REAL_ACCOUNT_COMPARE_PATH = ROOT / "utils" / "tmp" / "run_brick_real_account_compare_v1_20260326.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import _read_txt


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


real_account = load_module(REAL_ACCOUNT_COMPARE_PATH, "brick_signal_constraints_real_account_v1")

DEF1_THRESHOLD = 1.0 / 3.0


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_error(result_dir: Path, exc: BaseException) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (result_dir / "error.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))


def load_champion_trades(file_limit_codes: int) -> pd.DataFrame:
    summary = json.loads((CHAMPION_RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    strategy_key = str(summary["strategy_best"])
    trades = pd.read_csv(
        CHAMPION_RESULT_DIR / "hybrid_local_trades.csv",
        parse_dates=["signal_date", "entry_date", "exit_date"],
    )
    trades = trades[trades["strategy_key"] == strategy_key].copy()
    if trades.empty:
        raise RuntimeError(f"未找到当前冠军交易: {strategy_key}")
    scores = real_account.load_signal_scores()
    trades = real_account.attach_sort_scores(trades, scores)
    if file_limit_codes > 0:
        keep_codes = sorted(trades["code"].astype(str).unique())[:file_limit_codes]
        trades = trades[trades["code"].astype(str).isin(keep_codes)].copy()
    return trades.sort_values(["signal_date", "code", "signal_idx"]).reset_index(drop=True)


def load_daily_map(codes: list[str], progress_cb: Any | None = None) -> dict[str, pd.DataFrame]:
    daily_map: dict[str, pd.DataFrame] = {}
    total = len(codes)
    for idx, code in enumerate(sorted(set(codes)), start=1):
        path = DAILY_DIR / f"{code}.txt"
        if not path.exists():
            continue
        df = _read_txt(str(path))
        if df is None or df.empty:
            continue
        df = df[(df["date"] < real_account.minute_base.EXCLUDE_START) | (df["date"] > real_account.minute_base.EXCLUDE_END)].copy()
        df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["date", "open", "high", "low", "close"])
        if df.empty:
            continue
        daily_map[code] = df.sort_values("date").reset_index(drop=True)
        if progress_cb is not None and (idx == 1 or idx % 100 == 0 or idx == total):
            progress_cb(idx, total)
    return daily_map


def upper_shadow_body_ratio(open_price: float, high_price: float, close_price: float) -> float:
    if not all(np.isfinite([open_price, high_price, close_price])):
        return float("nan")
    body = abs(close_price - open_price)
    upper_shadow = max(0.0, high_price - max(open_price, close_price))
    if body <= 1e-12:
        return float("inf") if upper_shadow > 1e-12 else 0.0
    return float(upper_shadow / body)


def attach_signal_shadow_features(trades: pd.DataFrame, daily_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in trades.itertuples(index=False):
        code = str(row.code)
        daily_df = daily_map.get(code)
        signal_row = None
        if daily_df is not None and not daily_df.empty:
            m = daily_df.loc[daily_df["date"] == pd.Timestamp(row.signal_date)]
            if not m.empty:
                signal_row = m.iloc[-1]
        if signal_row is None:
            signal_open = signal_high = signal_low = signal_close = float("nan")
            ratio = float("nan")
        else:
            signal_open = float(signal_row["open"])
            signal_high = float(signal_row["high"])
            signal_low = float(signal_row["low"])
            signal_close = float(signal_row["close"])
            ratio = upper_shadow_body_ratio(signal_open, signal_high, signal_close)
        rows.append(
            {
                "code": code,
                "signal_idx": int(row.signal_idx),
                "signal_date": pd.Timestamp(row.signal_date),
                "signal_open": signal_open,
                "signal_high": signal_high,
                "signal_low_daily": signal_low,
                "signal_close": signal_close,
                "signal_upper_body_ratio": ratio,
            }
        )
    feat_df = pd.DataFrame(rows)
    out = trades.merge(feat_df, on=["code", "signal_idx", "signal_date"], how="left")
    out["entry_gap_pct"] = pd.to_numeric(out["entry_gap_pct"], errors="coerce")
    out["signal_upper_body_ratio"] = pd.to_numeric(out["signal_upper_body_ratio"], errors="coerce")
    return out


def parse_case_files() -> pd.DataFrame:
    import re

    rows: list[dict[str, Any]] = []
    pat = re.compile(r"(.+?)(\d{8})\.png$")
    for path in sorted(CASE_DIR.glob("*.png")):
        m = pat.match(path.name)
        if not m:
            continue
        rows.append({"stock_name": m.group(1), "signal_date": pd.to_datetime(m.group(2), format="%Y%m%d"), "case_file": str(path)})
    return pd.DataFrame(rows)


def build_name_code_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    valid_codes = {p.stem for p in DAILY_DIR.glob("*.txt")}
    for path in RAW_CASE_DIR.glob("*.txt"):
        try:
            first_line = path.read_text(encoding="gbk", errors="ignore").splitlines()[0].strip()
        except Exception:
            continue
        parts = first_line.split()
        if len(parts) >= 2 and parts[0].isdigit() and path.stem in valid_codes:
            mapping[parts[1]] = path.stem
    return mapping


def derive_case_shadow_threshold(result_dir: Path, progress_cb: Any | None = None) -> dict[str, Any]:
    cases = parse_case_files()
    name_code = build_name_code_map()
    rows: list[dict[str, Any]] = []
    total = len(cases)
    for idx, case in enumerate(cases.itertuples(index=False), start=1):
        code = name_code.get(str(case.stock_name))
        if not code:
            if progress_cb is not None and (idx == 1 or idx % 10 == 0 or idx == total):
                progress_cb(idx, total, usable_cases=len(rows), skipped_reason="missing_code")
            continue
        path = DAILY_DIR / f"{code}.txt"
        df = _read_txt(str(path))
        if df is None or df.empty:
            if progress_cb is not None and (idx == 1 or idx % 10 == 0 or idx == total):
                progress_cb(idx, total, usable_cases=len(rows), skipped_reason="missing_daily")
            continue
        df = df[(df["date"] < real_account.minute_base.EXCLUDE_START) | (df["date"] > real_account.minute_base.EXCLUDE_END)].copy()
        m = df.loc[df["date"] == pd.Timestamp(case.signal_date)]
        if m.empty:
            if progress_cb is not None and (idx == 1 or idx % 10 == 0 or idx == total):
                progress_cb(idx, total, usable_cases=len(rows), skipped_reason="missing_signal")
            continue
        r = m.iloc[-1]
        ratio = upper_shadow_body_ratio(float(r["open"]), float(r["high"]), float(r["close"]))
        rows.append({"stock_name": case.stock_name, "code": code, "signal_date": pd.Timestamp(case.signal_date), "upper_body_ratio": ratio})
        if progress_cb is not None and (idx == 1 or idx % 10 == 0 or idx == total):
            progress_cb(idx, total, usable_cases=len(rows))

    case_df = pd.DataFrame(rows)
    case_df.to_csv(result_dir / "perfect_case_shadow_analysis.csv", index=False, encoding="utf-8-sig")
    finite = case_df["upper_body_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    threshold = float(finite.max()) if not finite.empty else float("nan")
    stats = {
        "usable_case_count": int(len(case_df)),
        "case_upper_body_ratio_max": threshold,
        "case_upper_body_ratio_median": float(finite.median()) if not finite.empty else float("nan"),
    }
    (result_dir / "case_shadow_threshold.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def filter_variants(trades: pd.DataFrame, case_threshold: float) -> dict[str, pd.DataFrame]:
    gap_mask = trades["entry_gap_pct"].fillna(0.0) < 0.04
    def1_mask = trades["signal_upper_body_ratio"].fillna(float("inf")) < DEF1_THRESHOLD
    if np.isfinite(case_threshold):
        def2_mask = trades["signal_upper_body_ratio"].fillna(float("inf")) <= case_threshold
    else:
        def2_mask = pd.Series(True, index=trades.index)
    return {
        "champion_current": trades.copy(),
        "gap4_filter_only": trades[gap_mask].copy(),
        "shadow_def1_filter": trades[gap_mask & def1_mask].copy(),
        "shadow_def2_filter": trades[gap_mask & def2_mask].copy(),
    }


def build_filter_audit(trades: pd.DataFrame, case_threshold: float) -> dict[str, Any]:
    gap_removed = trades[trades["entry_gap_pct"].fillna(0.0) >= 0.04]
    def1_removed = trades[trades["signal_upper_body_ratio"].fillna(float("inf")) >= DEF1_THRESHOLD]
    if np.isfinite(case_threshold):
        def2_removed = trades[trades["signal_upper_body_ratio"].fillna(float("inf")) > case_threshold]
    else:
        def2_removed = trades.iloc[0:0].copy()
    return {
        "source_trade_count": int(len(trades)),
        "gap_ge_4pct_removed_count": int(len(gap_removed)),
        "shadow_def1_removed_count": int(len(def1_removed)),
        "shadow_def2_removed_count": int(len(def2_removed)),
        "gap_ge_4pct_removed_ratio": float(len(gap_removed) / len(trades)) if len(trades) else 0.0,
        "shadow_def1_removed_ratio": float(len(def1_removed) / len(trades)) if len(trades) else 0.0,
        "shadow_def2_removed_ratio": float(len(def2_removed) / len(trades)) if len(trades) else 0.0,
    }


def build_comparison(base_row: dict[str, Any], other_row: dict[str, Any]) -> dict[str, float]:
    metrics = [
        "annual_return",
        "holding_return",
        "avg_trade_return",
        "success_rate",
        "max_drawdown",
        "sharpe",
        "calmar",
        "final_multiple",
        "final_equity",
    ]
    out: dict[str, float] = {}
    for key in metrics:
        a = float(base_row[key])
        b = float(other_row[key])
        out[f"{key}_diff"] = b - a
    return out


def run_compare(result_dir: Path, file_limit_codes: int) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "loading_champion", file_limit_codes=file_limit_codes)
    trades = load_champion_trades(file_limit_codes=file_limit_codes)
    trades.to_csv(result_dir / "champion_source_trades.csv", index=False, encoding="utf-8-sig")

    codes = sorted(trades["code"].astype(str).unique())
    daily_map = load_daily_map(
        codes,
        progress_cb=lambda done, total: update_progress(result_dir, "loading_signal_bars", done_codes=done, total_codes=total),
    )
    trades = attach_signal_shadow_features(trades, daily_map)
    trades.to_csv(result_dir / "champion_source_trades_with_signal_features.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "signal_features_ready", trade_count=int(len(trades)), feature_codes=int(len(daily_map)))

    case_stats = derive_case_shadow_threshold(
        result_dir,
        progress_cb=lambda done, total, **extra: update_progress(result_dir, "deriving_case_threshold", done_cases=done, total_cases=total, **extra),
    )
    case_threshold = float(case_stats["case_upper_body_ratio_max"]) if np.isfinite(case_stats["case_upper_body_ratio_max"]) else float("nan")
    update_progress(result_dir, "case_threshold_ready", **case_stats)

    variants = filter_variants(trades, case_threshold=case_threshold)
    audit = build_filter_audit(trades, case_threshold=case_threshold)
    for label, subset in variants.items():
        subset.to_csv(result_dir / f"{label}_candidate_trades.csv", index=False, encoding="utf-8-sig")
    (result_dir / "filter_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "filters_ready", **audit)

    market_dates, close_map = real_account.build_close_map(
        list(variants["champion_current"]["code"].astype(str).unique()),
        progress_cb=lambda done, total: update_progress(result_dir, "building_close_map", done_codes=done, total_codes=total),
    )
    if len(market_dates) == 0:
        raise RuntimeError("无法构建账户层 close_map")
    update_progress(result_dir, "close_map_ready", market_days=int(len(market_dates)), close_codes=int(len(close_map)))

    config = real_account.AccountConfig()
    summary_rows: list[dict[str, Any]] = []
    summary_lookup: dict[str, dict[str, Any]] = {}
    for label, subset in variants.items():
        equity_df, executed_df, summary = real_account.simulate_real_account(subset, close_map, market_dates, config)
        equity_df.to_csv(result_dir / f"{label}_equity.csv", index=False, encoding="utf-8-sig")
        executed_df.to_csv(result_dir / f"{label}_executed_trades.csv", index=False, encoding="utf-8-sig")
        row = {"label": label, **summary}
        summary_rows.append(row)
        summary_lookup[label] = row
        update_progress(result_dir, "simulated_one", last_label=label)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(result_dir / "real_account_summary.csv", index=False, encoding="utf-8-sig")

    base_row = summary_lookup["champion_current"]
    comparisons = {
        "champion_vs_gap4": build_comparison(base_row, summary_lookup["gap4_filter_only"]),
        "champion_vs_shadow_def1": build_comparison(base_row, summary_lookup["shadow_def1_filter"]),
        "champion_vs_shadow_def2": build_comparison(base_row, summary_lookup["shadow_def2_filter"]),
    }
    summary = {
        "assumptions": {
            "signal_pool": "brick.formal_best",
            "champion_strategy": "formal_best + 当日止损次日止盈 + min(open,close)止损 + 5.5%止盈",
            "shadow_definition_1": "上影线/实体长度 >= 1/3 视为长上影",
            "shadow_definition_2": "超过成功案例中最大上影/实体比例视为长上影",
            "gap_definition": "信号日次日高开 >= 4% 不买",
            "account_type": "real_account_engine_like",
        },
        "case_stats": case_stats,
        "filter_audit": audit,
        "summary_rows": summary_rows,
        "comparison": comparisons,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK 冠军增加高开与长上影过滤的真实账户层对比")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--file-limit-codes", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_real_account_signal_constraints_compare_v1_{args.mode}_{timestamp}"
    file_limit_codes = int(args.file_limit_codes)
    if args.mode == "full":
        file_limit_codes = 0
    try:
        run_compare(result_dir=output_dir, file_limit_codes=file_limit_codes)
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
