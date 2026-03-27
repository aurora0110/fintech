from __future__ import annotations

import argparse
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
DAILY_DIR = ROOT / "data" / "20260324"
MIN5_DIR = ROOT / "data" / "202603245min"
FORMAL_RESULT_DIR = RESULT_ROOT / "brick_hybrid_local_search_minoc_full_20260326_r2"
RELAXED_RESULT_DIR = RESULT_ROOT / "brick_comprehensive_lab_full_20260325_r1"
REAL_ACCOUNT_COMPARE_PATH = ROOT / "utils" / "tmp" / "run_brick_real_account_compare_v1_20260326.py"
HYBRID_PATH = ROOT / "utils" / "tmp" / "run_brick_hybrid_local_search_v1_20260325.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import _read_txt

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


real_account = load_module(REAL_ACCOUNT_COMPARE_PATH, "brick_buypoint_compare_real_account_v1")
hybrid = load_module(HYBRID_PATH, "brick_buypoint_compare_hybrid_v1")


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


def build_daily_code_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in DAILY_DIR.glob("*.txt"):
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            continue
        suffix6 = digits[-6:].zfill(6)
        mapping[suffix6] = stem
    return mapping


def load_formal_champion(file_limit_codes: int) -> tuple[str, pd.DataFrame]:
    summary = json.loads((FORMAL_RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    strategy_key = str(summary["strategy_best"])
    trades = pd.read_csv(FORMAL_RESULT_DIR / "hybrid_local_trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    trades = trades[trades["strategy_key"] == strategy_key].copy()
    if trades.empty:
        raise RuntimeError(f"未找到 formal 冠军交易: {strategy_key}")
    scores = real_account.load_signal_scores()
    trades = real_account.attach_sort_scores(trades, scores)
    if file_limit_codes > 0:
        keep_codes = sorted(trades["code"].astype(str).unique())[:file_limit_codes]
        trades = trades[trades["code"].astype(str).isin(keep_codes)].copy()
    return strategy_key, trades.sort_values(["signal_date", "code", "signal_idx"]).reset_index(drop=True)


def load_relaxed_candidates(file_limit_codes: int) -> tuple[dict[str, Any], pd.DataFrame]:
    best = json.loads((RELAXED_RESULT_DIR / "best_config.json").read_text(encoding="utf-8"))
    scored = pd.read_csv(RELAXED_RESULT_DIR / "candidate_scored.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    code_map = build_daily_code_map()
    pool = str(best["candidate_pool"])
    sim_gate = float(best["sim_gate"])
    daily_topn = int(best["daily_topn"])
    scored = scored[scored["candidate_pool"].astype(str) == pool].copy()
    scored["code"] = scored["code"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    scored["code"] = scored["code"].map(code_map)
    scored = scored.dropna(subset=["code"]).copy()
    scored["sim_score"] = pd.to_numeric(scored["sim_score"], errors="coerce")
    scored["rank_score"] = pd.to_numeric(scored["rank_score"], errors="coerce")
    scored = scored[scored["sim_score"] >= sim_gate].copy()
    scored = scored.sort_values(["signal_date", "rank_score", "code"], ascending=[True, False, True]).groupby("signal_date", group_keys=False).head(daily_topn).reset_index(drop=True)
    scored["strategy_key"] = str(best["strategy"])
    if file_limit_codes > 0:
        keep_codes = sorted(scored["code"].astype(str).unique())[:file_limit_codes]
        scored = scored[scored["code"].astype(str).isin(keep_codes)].copy()
    keep_cols = [
        "code",
        "signal_idx",
        "signal_date",
        "entry_date",
        "entry_price",
        "signal_low",
        "rank_score",
        "sim_score",
        "factor_score",
        "ml_score",
        "strategy_key",
    ]
    return best, scored[keep_cols].reset_index(drop=True)


def load_daily_df(path: Path) -> pd.DataFrame:
    df = _read_txt(str(path))
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = df[(df["date"] < real_account.minute_base.EXCLUDE_START) | (df["date"] > real_account.minute_base.EXCLUDE_END)].copy()
    return df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)


def load_min5_fast(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            skiprows=2,
            header=None,
            names=["date_text", "time", "open", "high", "low", "close", "volume", "amount"],
            encoding="gbk",
            engine="c",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            skiprows=2,
            header=None,
            names=["date_text", "time", "open", "high", "low", "close", "volume", "amount"],
            encoding="utf-8",
            engine="c",
        )
    if df.empty:
        return pd.DataFrame(columns=["datetime", "date", "time", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date_text"], format="%Y/%m/%d", errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce").round().astype("Int64").astype(str).str.replace("<NA>", "", regex=False).str.zfill(4)
    df["datetime"] = pd.to_datetime(df["date_text"].astype(str) + " " + df["time"], format="%Y/%m/%d %H%M", errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "date", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "date", "time", "open", "high", "low", "close", "volume"]]


def simulate_relaxed_for_code(code: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    daily_path = DAILY_DIR / f"{code}.txt"
    daily_df = load_daily_df(daily_path)
    if daily_df.empty:
        return []
    min5_path = MIN5_DIR / f"{code}.txt"
    min5_df = load_min5_fast(min5_path) if min5_path.exists() else None
    out: list[dict[str, Any]] = []
    for row in rows:
        sim = hybrid.simulate_one_trade(
            code=code,
            signal_date=pd.Timestamp(row["signal_date"]),
            entry_date=pd.Timestamp(row["entry_date"]),
            signal_idx=int(row["signal_idx"]),
            daily_df=daily_df,
            min5_df=min5_df,
            tp_pct=0.055,
            stop_multiplier=1.0,
            stop_base="min_oc",
        )
        if sim.trade is None or sim.skipped:
            continue
        tr = sim.trade
        tr["sort_score"] = float(row["rank_score"])
        tr["strategy_key"] = str(row["strategy_key"])
        tr["signal_date"] = pd.Timestamp(tr["signal_date"])
        tr["entry_date"] = pd.Timestamp(tr["entry_date"])
        tr["exit_date"] = pd.Timestamp(tr["exit_date"])
        out.append(tr)
    return out


def build_relaxed_trades(candidates: pd.DataFrame, result_dir: Path, max_workers: int) -> pd.DataFrame:
    grouped = {
        str(code): group.to_dict("records")
        for code, group in candidates.sort_values(["signal_date", "rank_score", "code"], ascending=[True, False, True]).groupby("code")
    }
    codes = sorted(grouped)
    total = len(codes)
    rows: list[dict[str, Any]] = []
    if max_workers <= 1:
        for idx, code in enumerate(codes, start=1):
            rows.extend(simulate_relaxed_for_code(code, grouped[code]))
            if idx == 1 or idx % 50 == 0 or idx == total:
                update_progress(result_dir, "simulating_relaxed", done_codes=idx, total_codes=total, fallback="serial")
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(simulate_relaxed_for_code, code, grouped[code]): code for code in codes}
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    rows.extend(future.result())
                    if completed == 1 or completed % 50 == 0 or completed == total:
                        update_progress(result_dir, "simulating_relaxed", done_codes=completed, total_codes=total)
        except Exception as exc:
            rows = []
            for idx, code in enumerate(codes, start=1):
                rows.extend(simulate_relaxed_for_code(code, grouped[code]))
                if idx == 1 or idx % 50 == 0 or idx == total:
                    update_progress(
                        result_dir,
                        "simulating_relaxed",
                        done_codes=idx,
                        total_codes=total,
                        fallback="serial",
                        parallel_error=type(exc).__name__,
                    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_date", "code", "signal_idx"]).reset_index(drop=True)


def restrict_common_window(formal_trades: pd.DataFrame, relaxed_trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    start_date = max(pd.Timestamp(formal_trades["signal_date"].min()), pd.Timestamp(relaxed_trades["signal_date"].min()))
    end_date = min(pd.Timestamp(formal_trades["signal_date"].max()), pd.Timestamp(relaxed_trades["signal_date"].max()))
    formal = formal_trades[(formal_trades["signal_date"] >= start_date) & (formal_trades["signal_date"] <= end_date)].copy()
    relaxed = relaxed_trades[(relaxed_trades["signal_date"] >= start_date) & (relaxed_trades["signal_date"] <= end_date)].copy()
    return formal.reset_index(drop=True), relaxed.reset_index(drop=True), {
        "compare_start": start_date.strftime("%Y-%m-%d"),
        "compare_end": end_date.strftime("%Y-%m-%d"),
    }


def build_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
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
    return {f"{key}_diff": float(b[key] - a[key]) for key in metrics}


def run_compare(result_dir: Path, file_limit_codes: int, max_workers: int) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "loading_formal", file_limit_codes=file_limit_codes, max_workers=max_workers)
    formal_key, formal_trades = load_formal_champion(file_limit_codes=file_limit_codes)
    best_cfg, relaxed_candidates = load_relaxed_candidates(file_limit_codes=file_limit_codes)
    formal_trades.to_csv(result_dir / "formal_champion_source_trades.csv", index=False, encoding="utf-8-sig")
    relaxed_candidates.to_csv(result_dir / "relaxed_fusion_candidates.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "data_ready", formal_trade_count=int(len(formal_trades)), relaxed_candidate_count=int(len(relaxed_candidates)))

    relaxed_trades = build_relaxed_trades(relaxed_candidates, result_dir=result_dir, max_workers=max_workers)
    if relaxed_trades.empty:
        raise RuntimeError("relaxed_fusion 在当前冠军卖法下没有生成交易")
    relaxed_trades.to_csv(result_dir / "relaxed_fusion_simulated_trades.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "relaxed_ready", relaxed_trade_count=int(len(relaxed_trades)))

    formal_cmp, relaxed_cmp, window = restrict_common_window(formal_trades, relaxed_trades)
    formal_cmp.to_csv(result_dir / "formal_champion_compare_window_trades.csv", index=False, encoding="utf-8-sig")
    relaxed_cmp.to_csv(result_dir / "relaxed_fusion_compare_window_trades.csv", index=False, encoding="utf-8-sig")
    update_progress(
        result_dir,
        "window_ready",
        compare_start=window["compare_start"],
        compare_end=window["compare_end"],
        formal_compare_trades=int(len(formal_cmp)),
        relaxed_compare_trades=int(len(relaxed_cmp)),
    )

    all_codes = sorted(set(formal_cmp["code"].astype(str)) | set(relaxed_cmp["code"].astype(str)))
    market_dates, close_map = real_account.build_close_map(
        all_codes,
        progress_cb=lambda done, total: update_progress(result_dir, "building_close_map", done_codes=done, total_codes=total),
    )
    if len(market_dates) == 0:
        raise RuntimeError("无法构建账户层 close_map")
    update_progress(result_dir, "close_map_ready", market_days=int(len(market_dates)), close_codes=int(len(close_map)))

    config = real_account.AccountConfig()
    summary_rows: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    for label, strategy_key, trades in [
        ("formal_best", formal_key, formal_cmp),
        ("relaxed_fusion", str(best_cfg["strategy"]), relaxed_cmp),
    ]:
        equity_df, executed_df, summary = real_account.simulate_real_account(trades, close_map, market_dates, config)
        equity_df.to_csv(result_dir / f"{label}_equity.csv", index=False, encoding="utf-8-sig")
        executed_df.to_csv(result_dir / f"{label}_executed_trades.csv", index=False, encoding="utf-8-sig")
        row = {"label": label, "strategy_key": strategy_key, **summary}
        summary_rows.append(row)
        lookup[label] = row
        update_progress(result_dir, "simulated_one", last_label=label)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(result_dir / "real_account_summary.csv", index=False, encoding="utf-8-sig")
    summary = {
        "assumptions": {
            "signal_pool_compare": "formal_best vs relaxed_fusion",
            "fixed_exit_strategy": "当日止损次日止盈 + min(open,close)止损 + 5.5%止盈",
            "stop_base": "min_oc",
            "stop_multiplier": 1.0,
            "tp_pct": 0.055,
            "compare_start": window["compare_start"],
            "compare_end": window["compare_end"],
            "account_type": "real_account_engine_like",
        },
        "formal_best": lookup["formal_best"],
        "relaxed_fusion": lookup["relaxed_fusion"],
        "comparison": {
            "formal_vs_relaxed": build_diff(lookup["formal_best"], lookup["relaxed_fusion"]),
        },
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定冠军卖法下比较 formal_best 与 relaxed_fusion 买点")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--file-limit-codes", type=int, default=200)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_buypoint_real_account_compare_v1_{args.mode}_{timestamp}"
    file_limit_codes = int(args.file_limit_codes)
    if args.mode == "full":
        file_limit_codes = 0
    try:
        run_compare(result_dir=output_dir, file_limit_codes=file_limit_codes, max_workers=int(args.max_workers))
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
