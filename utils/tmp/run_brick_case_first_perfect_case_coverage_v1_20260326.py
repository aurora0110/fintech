from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tmp import brick_case_semantics_v1_20260326 as case_semantics
from utils.tmp import brickfilter_case_first_v1_20260326 as case_first

CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
RESULT_ROOT = ROOT / "results"
DEFAULT_DATA_DIR = ROOT / "data" / "20260324"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def write_error(result_dir: Path, exc: BaseException) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(result_dir / "error.json", payload)


def resolve_cases(data_dir: Path) -> pd.DataFrame:
    daily_dir = case_first._resolve_daily_dir(data_dir)
    cases = case_semantics.parse_case_images(CASE_DIR, skip_counter_examples=True)
    name_map = case_semantics.build_name_code_map(daily_dir)
    cases["code"] = cases["stock_name"].map(name_map)
    cases = cases.dropna(subset=["code"]).copy()
    cases["code_key"] = cases["code"].map(case_semantics.code_key)
    type_rows = []
    feat_df = case_semantics.load_case_day_features(CASE_DIR, daily_dir)
    if not feat_df.empty:
        type_rows = feat_df[["stock_name", "signal_date", "brick_case_type_name"]].copy()
    if len(type_rows):
        cases = cases.merge(type_rows, on=["stock_name", "signal_date"], how="left")
    else:
        cases["brick_case_type_name"] = "其他"
    return cases.reset_index(drop=True)


def _score_single_date(
    target_date: pd.Timestamp,
    data_dir: Path,
    topn: int,
    daily_cases: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    selected_df = case_first.score_candidates_for_date(target_date, cand_df, data_dir, topn=topn)
    cand_keys = set(cand_df["code"].map(case_semantics.code_key)) if not cand_df.empty else set()
    selected_keys = set(selected_df["code"].map(case_semantics.code_key)) if not selected_df.empty else set()
    top50_keys = set(selected_df["code"].map(case_semantics.code_key).head(50)) if not selected_df.empty else set()
    rows: list[dict[str, Any]] = []
    for row in daily_cases.itertuples(index=False):
        key = case_semantics.code_key(row.code)
        if key in selected_keys:
            stage = "selected"
        elif key in cand_keys:
            stage = "candidate_pool"
        else:
            stage = "missed"
        rows.append(
            {
                "stock_name": row.stock_name,
                "code": row.code,
                "signal_date": pd.Timestamp(row.signal_date),
                "brick_case_type_name": row.brick_case_type_name,
                "stage": stage,
                "in_top20": key in selected_keys,
                "in_top50": key in top50_keys,
            }
        )
    return rows


def run_case_coverage(data_dir: Path, output_dir: Path, topn: int, max_workers: int, max_dates: int = 0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = resolve_cases(data_dir)
    if cases.empty:
        write_json(output_dir / "summary.json", {"error": "未解析到完美案例"})
        return
    dates = sorted(pd.to_datetime(cases["signal_date"]).dropna().unique())
    if max_dates > 0:
        dates = dates[:max_dates]
    update_progress(output_dir, "building_candidate_cache", done_dates=0, total_dates=len(dates))
    all_candidates = case_first.build_candidate_cache_for_dates(data_dir, dates, max_workers=max_workers)
    rows: list[dict[str, Any]] = []
    update_progress(output_dir, "processing_date", done_dates=0, total_dates=len(dates))
    for idx, target_date in enumerate(dates, start=1):
        date_ts = pd.Timestamp(target_date)
        date_key = str(date_ts.date())
        daily_cases = cases[cases["signal_date"] == date_ts].copy().reset_index(drop=True)
        cand_df = all_candidates[all_candidates["signal_date"] == date_ts].copy().reset_index(drop=True)
        update_progress(
            output_dir,
            "processing_date",
            done_dates=idx - 1,
            total_dates=len(dates),
            current_date=date_key,
            candidate_count=int(len(cand_df)),
            case_count=int(len(daily_cases)),
        )
        rows.extend(_score_single_date(date_ts, data_dir, topn, daily_cases, cand_df))
    result_df = pd.DataFrame(rows).sort_values(["signal_date", "stock_name"]).reset_index(drop=True)
    result_df.to_csv(output_dir / "perfect_case_casefirst_results.csv", index=False)
    summary = {
        "total_cases": int(len(result_df)),
        "selected_count": int((result_df["stage"] == "selected").sum()),
        "candidate_pool_count": int((result_df["stage"] == "candidate_pool").sum()),
        "missed_count": int((result_df["stage"] == "missed").sum()),
        "top20_count": int(result_df["in_top20"].sum()),
        "top50_count": int(result_df["in_top50"].sum()),
        "type_counts": result_df["brick_case_type_name"].value_counts().to_dict(),
        "stage_counts": result_df["stage"].value_counts().to_dict(),
        "topn": int(topn),
    }
    write_json(output_dir / "summary.json", summary)
    update_progress(output_dir, "finished", total_dates=len(dates))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(RESULT_ROOT / "brick_case_first_perfect_case_coverage_v1_20260326_r1"))
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-dates", type=int, default=0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    try:
        run_case_coverage(
            Path(args.data_dir),
            output_dir,
            topn=int(args.topn),
            max_workers=int(args.max_workers),
            max_dates=int(args.max_dates),
        )
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_error(output_dir, exc)
        update_progress(output_dir, "error")
        raise


if __name__ == "__main__":
    main()
