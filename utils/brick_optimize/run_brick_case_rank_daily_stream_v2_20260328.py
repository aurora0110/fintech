from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
BASE_SCRIPT = ROOT / "utils" / "brick_optimize" / "run_brick_case_rank_daily_stream_v1_20260328.py"

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


base = load_module(BASE_SCRIPT, "brick_case_rank_daily_stream_v2_base")
rank_model = base.rank_model
case_first = base.case_first
case_recall = base.case_recall


def _bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[unknown]"
    ratio = max(0.0, min(1.0, done / total))
    filled = int(round(width * ratio))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total} ({ratio:.1%})"


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


def _chunk_files(file_paths: list[str], max_workers: int) -> list[list[str]]:
    if not file_paths:
        return []
    chunk_size = max(8, math.ceil(len(file_paths) / max(max_workers * 4, 1)))
    return [file_paths[i : i + chunk_size] for i in range(0, len(file_paths), chunk_size)]


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


def _cache_chunk_worker(task: tuple[list[str], list[str] | None, list[int], str]) -> dict[str, Any]:
    file_paths, target_date_list, required_lens, part_path_str = task
    target_date_set = set(target_date_list) if target_date_list is not None else None
    rows: list[dict[str, Any]] = []
    for file_path in file_paths:
        rows.extend(case_first._records_for_file(file_path, target_date_set, required_lens))
    part_path = Path(part_path_str)
    if rows:
        pd.DataFrame(rows).to_pickle(part_path)
    return {
        "part_path": str(part_path),
        "rows": int(len(rows)),
        "files": int(len(file_paths)),
        "written": bool(rows),
    }


def _enrich_date_worker(task: tuple[pd.Timestamp, str, pd.Timestamp | None, str, str]) -> dict[str, Any]:
    date_key, cache_part_path_str, entry_date, enriched_part_path_str, data_dir_str = task
    cache_part_path = Path(cache_part_path_str)
    if not cache_part_path.exists():
        return {"date": str(pd.Timestamp(date_key).date()), "rows": 0, "written": False}
    df = pd.read_pickle(cache_part_path)
    if df.empty:
        return {"date": str(pd.Timestamp(date_key).date()), "rows": 0, "written": False}
    if "seq_map" in df.columns:
        df = df[df["seq_map"].apply(_has_required_seq_map)].copy().reset_index(drop=True)
    if df.empty:
        return {"date": str(pd.Timestamp(date_key).date()), "rows": 0, "written": False}
    enriched = case_recall.enrich_candidates_for_date(pd.Timestamp(date_key), df, Path(data_dir_str))
    if enriched.empty:
        return {"date": str(pd.Timestamp(date_key).date()), "rows": 0, "written": False}
    enriched = rank_model._prepare_features(enriched)
    enriched["signal_date"] = pd.to_datetime(enriched["signal_date"])
    enriched["entry_date"] = pd.Timestamp(entry_date) if entry_date is not None else pd.NaT
    enriched["exit_date"] = pd.Timestamp(entry_date) if entry_date is not None else pd.NaT
    enriched = enriched.dropna(subset=["entry_date"]).reset_index(drop=True)
    if enriched.empty:
        return {"date": str(pd.Timestamp(date_key).date()), "rows": 0, "written": False}
    enriched.to_pickle(enriched_part_path_str)
    return {"date": str(pd.Timestamp(date_key).date()), "rows": int(len(enriched)), "written": True}


def _load_candidate_cache(parts_dir: Path) -> pd.DataFrame:
    part_files = sorted(parts_dir.glob("candidate_cache_part_*.pkl"))
    if not part_files:
        return pd.DataFrame()
    frames = [pd.read_pickle(path) for path in part_files if path.exists()]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    return out.sort_values(["signal_date", "code"]).reset_index(drop=True)


def _load_enriched_candidates(parts_dir: Path) -> pd.DataFrame:
    part_files = sorted(parts_dir.glob("enriched_part_*.pkl"))
    if not part_files:
        return pd.DataFrame()
    frames = [pd.read_pickle(path) for path in part_files if path.exists()]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    return out.sort_values(["signal_date", "code"]).reset_index(drop=True)


def _select_topn(scored_df: pd.DataFrame, topn: int) -> pd.DataFrame:
    selected = rank_model._select_topn_per_date(scored_df, topn)
    if selected.empty:
        return selected
    selected = selected.copy()
    selected["sort_score"] = pd.to_numeric(selected["model_score"], errors="coerce").fillna(0.0)
    selected["strategy_key"] = "case_rank|lgbm_daily_stream"
    return selected


def run_phase0(result_dir: Path, mode: str, smoke_dates: int, max_workers: int) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_parts_dir = result_dir / "candidate_cache_parts"
    enriched_parts_dir = result_dir / "enriched_parts"
    cache_parts_dir.mkdir(parents=True, exist_ok=True)
    enriched_parts_dir.mkdir(parents=True, exist_ok=True)

    update_progress(result_dir, "collecting_trading_dates", mode=mode, max_workers=max_workers)
    all_dates = base.collect_trading_dates(DATA_DIR, max_workers=max_workers)
    target_dates = base.pick_smoke_dates(all_dates, smoke_dates) if mode == "smoke" else all_dates
    next_date_map = base.build_next_date_map(all_dates)

    daily_dir = case_first._resolve_daily_dir(DATA_DIR)
    file_paths = sorted(str(path) for path in daily_dir.glob("*.txt"))
    file_chunks = _chunk_files(file_paths, max_workers)
    target_date_list = None if mode == "full" else sorted(pd.Timestamp(x).strftime("%Y-%m-%d") for x in target_dates)

    update_progress(
        result_dir,
        "building_candidate_cache",
        total_calendar_dates=len(all_dates),
        target_dates=len(target_dates),
        total_files=len(file_paths),
        total_chunks=len(file_chunks),
        progress_bar=_bar(0, len(file_chunks)),
        first_date=pd.Timestamp(target_dates[0]).strftime("%Y-%m-%d") if target_dates else None,
        last_date=pd.Timestamp(target_dates[-1]).strftime("%Y-%m-%d") if target_dates else None,
    )

    cache_tasks = []
    for idx, chunk in enumerate(file_chunks, start=1):
        part_path = cache_parts_dir / f"candidate_cache_part_{idx:04d}.pkl"
        cache_tasks.append((chunk, target_date_list, case_recall.CASE_SEQ_LENS, str(part_path)))

    done_chunks = 0
    done_files = 0
    written_parts = 0
    if max_workers <= 1:
        for task in cache_tasks:
            info = _cache_chunk_worker(task)
            done_chunks += 1
            done_files += int(info["files"])
            written_parts += int(bool(info["written"]))
            if done_chunks == 1 or done_chunks % 10 == 0 or done_chunks == len(cache_tasks):
                update_progress(
                    result_dir,
                    "building_candidate_cache",
                    total_calendar_dates=len(all_dates),
                    target_dates=len(target_dates),
                    total_files=len(file_paths),
                    done_files=done_files,
                    total_chunks=len(file_chunks),
                    done_chunks=done_chunks,
                    written_parts=written_parts,
                    progress_bar=_bar(done_chunks, len(file_chunks)),
                )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_cache_chunk_worker, task): idx for idx, task in enumerate(cache_tasks, start=1)}
            for future in as_completed(future_map):
                info = future.result()
                done_chunks += 1
                done_files += int(info["files"])
                written_parts += int(bool(info["written"]))
                if done_chunks == 1 or done_chunks % 10 == 0 or done_chunks == len(cache_tasks):
                    update_progress(
                        result_dir,
                        "building_candidate_cache",
                        total_calendar_dates=len(all_dates),
                        target_dates=len(target_dates),
                        total_files=len(file_paths),
                        done_files=done_files,
                        total_chunks=len(file_chunks),
                        done_chunks=done_chunks,
                        written_parts=written_parts,
                        progress_bar=_bar(done_chunks, len(file_chunks)),
                    )

    update_progress(result_dir, "combining_candidate_cache_parts", written_parts=written_parts)
    cache_df = _load_candidate_cache(cache_parts_dir)
    if cache_df.empty:
        raise RuntimeError("全日级候选缓存为空，无法继续 Phase 0")
    cache_df.to_csv(result_dir / "candidate_cache.csv", index=False, encoding="utf-8-sig")

    unique_dates = sorted(pd.to_datetime(cache_df["signal_date"]).drop_duplicates().tolist())
    update_progress(
        result_dir,
        "enriching_candidates",
        total_dates=len(unique_dates),
        written_parts=0,
        progress_bar=_bar(0, len(unique_dates)),
    )

    enrich_tasks = []
    for idx, date_key in enumerate(unique_dates, start=1):
        date_df = cache_df[cache_df["signal_date"] == pd.Timestamp(date_key)].copy().reset_index(drop=True)
        cache_part_path = cache_parts_dir / f"date_cache_{idx:05d}.pkl"
        date_df.to_pickle(cache_part_path)
        enriched_part_path = enriched_parts_dir / f"enriched_part_{idx:05d}.pkl"
        enrich_tasks.append((pd.Timestamp(date_key), str(cache_part_path), next_date_map.get(pd.Timestamp(date_key)), str(enriched_part_path), str(DATA_DIR)))

    done_dates = 0
    written_enriched = 0
    if max_workers <= 1:
        for task in enrich_tasks:
            info = _enrich_date_worker(task)
            done_dates += 1
            written_enriched += int(bool(info["written"]))
            if done_dates == 1 or done_dates % 25 == 0 or done_dates == len(enrich_tasks):
                update_progress(
                    result_dir,
                    "enriching_candidates",
                    total_dates=len(unique_dates),
                    done_dates=done_dates,
                    written_parts=written_enriched,
                    latest_date=str(info["date"]),
                    progress_bar=_bar(done_dates, len(unique_dates)),
                )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_enrich_date_worker, task): task[0] for task in enrich_tasks}
            for future in as_completed(future_map):
                info = future.result()
                done_dates += 1
                written_enriched += int(bool(info["written"]))
                if done_dates == 1 or done_dates % 25 == 0 or done_dates == len(enrich_tasks):
                    update_progress(
                        result_dir,
                        "enriching_candidates",
                        total_dates=len(unique_dates),
                        done_dates=done_dates,
                        written_parts=written_enriched,
                        latest_date=str(info["date"]),
                        progress_bar=_bar(done_dates, len(unique_dates)),
                    )

    update_progress(result_dir, "combining_enriched_parts", written_parts=written_enriched)
    enriched_df = _load_enriched_candidates(enriched_parts_dir)
    if enriched_df.empty:
        raise RuntimeError("候选 enrich 后为空，无法继续 Phase 0")
    enriched_df.to_csv(result_dir / "enriched_candidates.csv", index=False, encoding="utf-8-sig")

    update_progress(result_dir, "training_frozen_model", train_rows=int(len(enriched_df)))
    best_name, best_params, model = base._load_frozen_best_model()

    update_progress(result_dir, "scoring_candidates", total_rows=int(len(enriched_df)))
    scored_df = enriched_df.copy()
    scored_df["model_score"] = base._score_with_model(model, scored_df)
    scored_df["sort_score"] = pd.to_numeric(scored_df["model_score"], errors="coerce").fillna(0.0)
    scored_df["strategy_key"] = "case_rank|lgbm_daily_stream"
    scored_df = scored_df.sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    scored_df.to_csv(result_dir / "daily_scored_candidates.csv", index=False, encoding="utf-8-sig")

    top_tables: dict[int, pd.DataFrame] = {}
    for topn in ALL_TOPNS:
        table = _select_topn(scored_df, topn)
        top_tables[topn] = table
        table.to_csv(result_dir / f"daily_top{topn}_candidates.csv", index=False, encoding="utf-8-sig")

    coverage_df = base._build_date_coverage_summary(target_dates, scored_df, top_tables)
    coverage_df.to_csv(result_dir / "date_coverage_summary.csv", index=False, encoding="utf-8-sig")
    coverage_df[coverage_df["year"].isin([2025, 2026])].to_csv(
        result_dir / "date_coverage_summary_2025_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )

    empty_summary = base._empty_day_summary(coverage_df)
    (result_dir / "empty_day_precheck_summary.json").write_text(
        json.dumps(empty_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "assumptions": {
            "mode": mode,
            "goal": "case_rank_daily_stream_phase0_optimized",
            "data_dir": str(DATA_DIR),
            "exclude_range": {
                "start": str(case_first.relaxed.EXCLUDE_START.date()),
                "end": str(case_first.relaxed.EXCLUDE_END.date()),
            },
            "model_name": best_name,
            "model_params": best_params,
            "feature_cols": rank_model.FEATURE_COLS,
            "max_workers": max_workers,
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
    update_progress(result_dir, "finished", calendar_days_used=len(target_dates), candidate_rows=len(scored_df), progress_bar=_bar(len(target_dates), len(target_dates)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 优化版：case_rank 全市场全交易日日级出票流构建")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--smoke-dates", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_case_rank_daily_stream_v2_{args.mode}_{timestamp}"
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
