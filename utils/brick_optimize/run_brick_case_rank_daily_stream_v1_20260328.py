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
DATA_DIR = ROOT / "data" / "20260324"
MODEL_SEARCH_SCRIPT = ROOT / "utils" / "brick_optimize" / "run_brick_case_rank_model_search_v1_20260327.py"
CASE_FIRST_SCRIPT = ROOT / "utils" / "brick_optimize" / "brickfilter_case_first_v1_20260326.py"
CASE_RECALL_SCRIPT = ROOT / "utils" / "brick_optimize" / "brickfilter_case_recall_v1_20260327.py"
MODEL_SEARCH_RESULT = RESULT_ROOT / "brick_case_rank_model_search_v1_full_20260327_r1"
TRAIN_DATASET_CSV = MODEL_SEARCH_RESULT / "candidate_dataset.csv"
TRAIN_SUMMARY_JSON = MODEL_SEARCH_RESULT / "summary.json"

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
ALL_TOPNS = [20, 50, 100]


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


rank_model = load_module(MODEL_SEARCH_SCRIPT, "brick_case_rank_daily_stream_rank_model")
case_first = load_module(CASE_FIRST_SCRIPT, "brick_case_rank_daily_stream_case_first")
case_recall = load_module(CASE_RECALL_SCRIPT, "brick_case_rank_daily_stream_case_recall")


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


def _read_dates_for_file(file_path: str) -> list[pd.Timestamp]:
    df = case_first.sim.load_stock_data(file_path)
    if df is None or df.empty:
        return []
    dates = pd.to_datetime(df["date"], errors="coerce").dropna().tolist()
    return [pd.Timestamp(x) for x in dates]


def collect_trading_dates(data_dir: Path, max_workers: int) -> list[pd.Timestamp]:
    daily_dir = case_first._resolve_daily_dir(data_dir)
    file_paths = sorted(str(path) for path in daily_dir.glob("*.txt"))
    if not file_paths:
        raise RuntimeError(f"未找到日线文件: {daily_dir}")
    rows: list[pd.Timestamp] = []
    if max_workers <= 1:
        for file_path in file_paths:
            rows.extend(_read_dates_for_file(file_path))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for dates in executor.map(_read_dates_for_file, file_paths, chunksize=16):
                if dates:
                    rows.extend(dates)
    if not rows:
        raise RuntimeError("未收集到任何交易日")
    all_dates = pd.Series(pd.to_datetime(rows)).drop_duplicates().sort_values()
    mask = (all_dates < case_first.relaxed.EXCLUDE_START) | (all_dates > case_first.relaxed.EXCLUDE_END)
    return [pd.Timestamp(x) for x in all_dates[mask].tolist()]


def pick_smoke_dates(all_dates: list[pd.Timestamp], smoke_dates: int) -> list[pd.Timestamp]:
    if smoke_dates <= 0 or len(all_dates) <= smoke_dates:
        return list(all_dates)
    idxs = np.linspace(0, len(all_dates) - 1, smoke_dates, dtype=int)
    return [all_dates[int(i)] for i in idxs]


def build_next_date_map(dates: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.Timestamp]:
    out: dict[pd.Timestamp, pd.Timestamp] = {}
    for idx in range(len(dates) - 1):
        out[pd.Timestamp(dates[idx])] = pd.Timestamp(dates[idx + 1])
    return out


def _load_frozen_best_model():
    if not TRAIN_SUMMARY_JSON.exists():
        raise RuntimeError(f"缺少冠军模型摘要: {TRAIN_SUMMARY_JSON}")
    payload = json.loads(TRAIN_SUMMARY_JSON.read_text(encoding="utf-8"))
    best_name = str(payload["best_model_name"])
    best_params = dict(payload["best_model_params"])
    dataset = pd.read_csv(TRAIN_DATASET_CSV)
    dataset["signal_date"] = pd.to_datetime(dataset["signal_date"])
    dataset = rank_model._prepare_features(dataset)
    x_train = dataset[rank_model.FEATURE_COLS].to_numpy(dtype=float)
    y_train = dataset["label"].astype(int).to_numpy()
    model = rank_model._build_model(best_name, best_params)
    model.fit(x_train, y_train)
    return best_name, best_params, model


def _score_with_model(model: Any, score_df: pd.DataFrame) -> pd.Series:
    score_df = rank_model._prepare_features(score_df)
    x_score = score_df[rank_model.FEATURE_COLS].to_numpy(dtype=float)
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(x_score)[:, 1], index=score_df.index, dtype=float)
    return pd.Series(model.predict(x_score), index=score_df.index, dtype=float)


def _has_required_seq_map(seq_map: Any) -> bool:
    if not isinstance(seq_map, dict):
        return False
    for seq_len in case_recall.CASE_SEQ_LENS:
        seq_item = seq_map.get(int(seq_len))
        if not isinstance(seq_item, dict):
            return False
        if "close_norm" not in seq_item:
            return False
    return True


def _enrich_one_date(task: tuple[pd.Timestamp, pd.DataFrame, str, pd.Timestamp | None]) -> pd.DataFrame:
    date_key, cand_df, data_dir_str, entry_date = task
    current = cand_df.copy().reset_index(drop=True)
    if "seq_map" in current.columns:
        current = current[current["seq_map"].apply(_has_required_seq_map)].copy().reset_index(drop=True)
    if current.empty:
        return pd.DataFrame()
    enriched = case_recall.enrich_candidates_for_date(pd.Timestamp(date_key), current, Path(data_dir_str))
    if enriched.empty:
        return pd.DataFrame()
    enriched = rank_model._prepare_features(enriched)
    enriched["signal_date"] = pd.to_datetime(enriched["signal_date"])
    enriched["entry_date"] = pd.Timestamp(entry_date) if entry_date is not None else pd.NaT
    enriched["exit_date"] = pd.Timestamp(entry_date) if entry_date is not None else pd.NaT
    return enriched


def _enrich_daily_candidates(
    cache_df: pd.DataFrame,
    result_dir: Path,
    data_dir: Path,
    next_date_map: dict[pd.Timestamp, pd.Timestamp],
    max_workers: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    unique_dates = sorted(pd.to_datetime(cache_df["signal_date"]).drop_duplicates().tolist())
    total = len(unique_dates)
    tasks: list[tuple[pd.Timestamp, pd.DataFrame, str, pd.Timestamp | None]] = []
    for date_ts in unique_dates:
        date_key = pd.Timestamp(date_ts)
        cand_df = cache_df[cache_df["signal_date"] == date_key].copy().reset_index(drop=True)
        tasks.append((date_key, cand_df, str(data_dir), next_date_map.get(date_key)))

    done = 0
    if max_workers <= 1:
        for date_key, cand_df, data_dir_str, entry_date in tasks:
            enriched = _enrich_one_date((date_key, cand_df, data_dir_str, entry_date))
            if not enriched.empty:
                rows.append(enriched)
            done += 1
            if done == 1 or done % 25 == 0 or done == total:
                update_progress(
                    result_dir,
                    "enriching_candidates",
                    done_dates=done,
                    total_dates=total,
                    latest_date=pd.Timestamp(date_key).strftime("%Y-%m-%d"),
                )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_enrich_one_date, task): task[0] for task in tasks}
            for future in as_completed(future_map):
                date_key = future_map[future]
                enriched = future.result()
                if not enriched.empty:
                    rows.append(enriched)
                done += 1
                if done == 1 or done % 25 == 0 or done == total:
                    update_progress(
                        result_dir,
                        "enriching_candidates",
                        done_dates=done,
                        total_dates=total,
                        latest_date=pd.Timestamp(date_key).strftime("%Y-%m-%d"),
                    )
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["entry_date"]).reset_index(drop=True)
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    return out


def _select_topn(scored_df: pd.DataFrame, topn: int) -> pd.DataFrame:
    selected = rank_model._select_topn_per_date(scored_df, topn)
    if selected.empty:
        return selected
    selected = selected.copy()
    selected["sort_score"] = pd.to_numeric(selected["model_score"], errors="coerce").fillna(0.0)
    selected["strategy_key"] = "case_rank|lgbm_daily_stream"
    return selected


def _build_date_coverage_summary(
    calendar_dates: list[pd.Timestamp],
    scored_df: pd.DataFrame,
    top_tables: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    cal_df = pd.DataFrame({"signal_date": pd.to_datetime(calendar_dates)})
    daily_rows = (
        scored_df.groupby("signal_date", sort=True)
        .size()
        .rename("candidate_count")
        .reset_index()
    )
    out = cal_df.merge(daily_rows, on="signal_date", how="left")
    out["candidate_count"] = pd.to_numeric(out["candidate_count"], errors="coerce").fillna(0).astype(int)
    for topn, table in top_tables.items():
        counts = table.groupby("signal_date", sort=True).size().rename(f"top{topn}_count").reset_index()
        out = out.merge(counts, on="signal_date", how="left")
        out[f"top{topn}_count"] = pd.to_numeric(out[f"top{topn}_count"], errors="coerce").fillna(0).astype(int)
    out["has_candidates"] = out["candidate_count"] > 0
    out["has_top20"] = out["top20_count"] > 0
    out["year"] = out["signal_date"].dt.year.astype(int)
    return out.sort_values("signal_date").reset_index(drop=True)


def _empty_day_summary(coverage_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_trading_days": int(len(coverage_df)),
        "zero_candidate_days": int((coverage_df["candidate_count"] == 0).sum()),
        "zero_top20_days": int((coverage_df["top20_count"] == 0).sum()),
    }
    yearly: dict[str, Any] = {}
    for year in [2025, 2026]:
        sub = coverage_df[coverage_df["year"] == year].copy()
        yearly[str(year)] = {
            "trading_days": int(len(sub)),
            "zero_candidate_days": int((sub["candidate_count"] == 0).sum()),
            "zero_top20_days": int((sub["top20_count"] == 0).sum()),
            "candidate_days": int((sub["candidate_count"] > 0).sum()),
            "top20_days": int((sub["top20_count"] > 0).sum()),
        }
    summary["yearly"] = yearly
    return summary


def run_phase0(result_dir: Path, mode: str, smoke_dates: int, max_workers: int) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "collecting_trading_dates", mode=mode, max_workers=max_workers)
    all_dates = collect_trading_dates(DATA_DIR, max_workers=max_workers)
    target_dates = pick_smoke_dates(all_dates, smoke_dates) if mode == "smoke" else all_dates
    next_date_map = build_next_date_map(all_dates)
    update_progress(
        result_dir,
        "building_candidate_cache",
        total_calendar_dates=len(all_dates),
        target_dates=len(target_dates),
        first_date=pd.Timestamp(target_dates[0]).strftime("%Y-%m-%d") if target_dates else None,
        last_date=pd.Timestamp(target_dates[-1]).strftime("%Y-%m-%d") if target_dates else None,
    )
    cache_df = case_first.build_candidate_cache_for_dates(
        DATA_DIR,
        target_dates,
        max_workers=max_workers,
        required_lens=case_recall.CASE_SEQ_LENS,
    )
    if cache_df.empty:
        raise RuntimeError("全日级候选缓存为空，无法继续 Phase 0")
    cache_df["signal_date"] = pd.to_datetime(cache_df["signal_date"])
    cache_df.to_csv(result_dir / "candidate_cache.csv", index=False, encoding="utf-8-sig")

    enriched_df = _enrich_daily_candidates(cache_df, result_dir, DATA_DIR, next_date_map, max_workers=max_workers)
    if enriched_df.empty:
        raise RuntimeError("候选 enrich 后为空，无法继续 Phase 0")
    enriched_df.to_csv(result_dir / "enriched_candidates.csv", index=False, encoding="utf-8-sig")

    update_progress(result_dir, "training_frozen_model", train_rows=int(len(enriched_df)))
    best_name, best_params, model = _load_frozen_best_model()

    update_progress(result_dir, "scoring_candidates", total_rows=int(len(enriched_df)))
    scored_df = enriched_df.copy()
    scored_df["model_score"] = _score_with_model(model, scored_df)
    scored_df["sort_score"] = pd.to_numeric(scored_df["model_score"], errors="coerce").fillna(0.0)
    scored_df["strategy_key"] = "case_rank|lgbm_daily_stream"
    scored_df = scored_df.sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    scored_df.to_csv(result_dir / "daily_scored_candidates.csv", index=False, encoding="utf-8-sig")

    top_tables: dict[int, pd.DataFrame] = {}
    for topn in ALL_TOPNS:
        table = _select_topn(scored_df, topn)
        top_tables[topn] = table
        table.to_csv(result_dir / f"daily_top{topn}_candidates.csv", index=False, encoding="utf-8-sig")

    coverage_df = _build_date_coverage_summary(target_dates, scored_df, top_tables)
    coverage_df.to_csv(result_dir / "date_coverage_summary.csv", index=False, encoding="utf-8-sig")
    coverage_df[coverage_df["year"].isin([2025, 2026])].to_csv(
        result_dir / "date_coverage_summary_2025_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )

    empty_summary = _empty_day_summary(coverage_df)
    (result_dir / "empty_day_precheck_summary.json").write_text(
        json.dumps(empty_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "assumptions": {
            "mode": mode,
            "goal": "case_rank_daily_stream_phase0",
            "data_dir": str(DATA_DIR),
            "exclude_range": {
                "start": str(case_first.relaxed.EXCLUDE_START.date()),
                "end": str(case_first.relaxed.EXCLUDE_END.date()),
            },
            "model_name": best_name,
            "model_params": best_params,
            "feature_cols": rank_model.FEATURE_COLS,
        },
        "calendar_days_total": int(len(all_dates)),
        "calendar_days_used": int(len(target_dates)),
        "candidate_rows": int(len(scored_df)),
        "candidate_signal_dates": int(scored_df["signal_date"].nunique()),
        "min_signal_date": pd.Timestamp(scored_df["signal_date"].min()).strftime("%Y-%m-%d"),
        "max_signal_date": pd.Timestamp(scored_df["signal_date"].max()).strftime("%Y-%m-%d"),
        "empty_day_precheck": empty_summary,
        "daily_candidate_count_stats": {
            "mean": float(pd.to_numeric(coverage_df["candidate_count"], errors="coerce").mean()),
            "median": float(pd.to_numeric(coverage_df["candidate_count"], errors="coerce").median()),
            "max": int(pd.to_numeric(coverage_df["candidate_count"], errors="coerce").max()),
            "days_with_candidates": int((coverage_df["candidate_count"] > 0).sum()),
        },
        "top20_days": int((coverage_df["top20_count"] > 0).sum()),
        "top50_days": int((coverage_df["top50_count"] > 0).sum()),
        "top100_days": int((coverage_df["top100_count"] > 0).sum()),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished", calendar_days_used=len(target_dates), candidate_rows=len(scored_df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0: case_rank 全市场全交易日日级出票流构建")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--smoke-dates", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_case_rank_daily_stream_v1_{args.mode}_{timestamp}"
    try:
        run_phase0(
            result_dir=output_dir,
            mode=str(args.mode),
            smoke_dates=int(args.smoke_dates),
            max_workers=int(args.max_workers),
        )
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
