from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import traceback
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
DEFAULT_DATA_DIR = ROOT / "data" / "20260324"
POS_CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
NEG_CASE_DIR = ROOT / "data" / "完美图" / "砖型图反例"
PHASE0_RESULT_DIR = RESULT_ROOT / "brick_case_rank_daily_stream_v2_full_20260328_r1"
CANDIDATE_CACHE_DIR = PHASE0_RESULT_DIR / "candidate_cache_parts"
CASE_RECALL_PATH = ROOT / "utils" / "brick_optimize" / "brickfilter_case_recall_v1_20260327.py"
CASE_SEMANTICS_PATH = ROOT / "utils" / "brick_optimize" / "brick_case_semantics_v1_20260326.py"
MODEL_SEARCH_PATH = ROOT / "utils" / "brick_optimize" / "run_brick_case_rank_model_search_v1_20260327.py"
DAILY_STREAM_BASE_PATH = ROOT / "utils" / "brick_optimize" / "run_brick_case_rank_daily_stream_v1_20260328.py"
REAL_ACCOUNT_PATH = ROOT / "utils" / "tmp" / "run_brick_real_account_compare_v1_20260326.py"

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
BUY_TP_LEVELS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
MAX_POSITIONS = 10
MAX_HOLD_DAYS = 3
INITIAL_CAPITAL = 1_000_000.0
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4), 10))
THRESHOLD_QUANTILE = 0.10
MIN_TRAIN_POS = 10
MIN_TRAIN_NEG = 5
TOPN = 20
_DAILY_STEM_MAP: dict[str, str] | None = None

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metrics import compute_metrics


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


case_recall = load_module(CASE_RECALL_PATH, "brick_wf_case_recall")
case_semantics = load_module(CASE_SEMANTICS_PATH, "brick_wf_case_semantics")
model_search = load_module(MODEL_SEARCH_PATH, "brick_wf_model_search")
daily_stream_base = load_module(DAILY_STREAM_BASE_PATH, "brick_wf_daily_stream_base")
real_account = load_module(REAL_ACCOUNT_PATH, "brick_wf_real_account")
DATA_DIR = DEFAULT_DATA_DIR
GLOBAL_CFG: "WalkforwardConfig | None" = None


@dataclass
class WalkforwardConfig:
    mode: str
    output_dir: Path
    data_dir: Path
    max_workers: int
    labeled_eval_date_limit: int
    backtest_date_limit: int
    use_default_exclude: bool
    backtest_start_date: pd.Timestamp | None
    backtest_end_date: pd.Timestamp | None


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


def _is_allowed_date(date_value: pd.Timestamp) -> bool:
    date_ts = pd.Timestamp(date_value)
    return bool((date_ts < EXCLUDE_START) or (date_ts > EXCLUDE_END))


def _load_case_pairs(case_dir: Path, label: int) -> pd.DataFrame:
    df = case_semantics.load_case_day_features(case_dir, DATA_DIR)
    if df.empty:
        return df
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["code"] = df["code"].map(case_semantics.code_key).astype(str)
    df["label"] = int(label)
    df["case_label"] = "positive" if int(label) == 1 else "negative"
    return df[["code", "signal_date", "stock_name", "case_file", "label", "case_label"]].drop_duplicates(["code", "signal_date"]).reset_index(drop=True)


def _is_allowed_date_series(series: pd.Series) -> pd.Series:
    x = pd.to_datetime(series)
    return (x < EXCLUDE_START) | (x > EXCLUDE_END)


def _apply_optional_exclude(df: pd.DataFrame, date_col: str, use_default_exclude: bool) -> pd.DataFrame:
    if df.empty or not use_default_exclude:
        return df
    return df[_is_allowed_date_series(df[date_col])].copy()


def _apply_date_window(df: pd.DataFrame, date_col: str, start_date: pd.Timestamp | None, end_date: pd.Timestamp | None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    if start_date is not None:
        out = out[out[date_col] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        out = out[out[date_col] <= pd.Timestamp(end_date)].copy()
    return out


def _build_cache_index(cache_dir: Path, cache_index_csv: Path) -> pd.DataFrame:
    if cache_index_csv.exists():
        df = pd.read_csv(cache_index_csv, parse_dates=["signal_date"])
        if not df.empty:
            return df.sort_values("signal_date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    paths = sorted(cache_dir.glob("date_cache_*.pkl"))
    total = len(paths)
    if total == 0:
        raise RuntimeError(f"未找到 date_cache_*.pkl: {cache_dir}")
    for idx, path in enumerate(paths, start=1):
        obj = pickle.loads(path.read_bytes())
        if not isinstance(obj, pd.DataFrame) or obj.empty or "signal_date" not in obj.columns:
            continue
        rows.append(
            {
                "signal_date": pd.Timestamp(obj["signal_date"].iloc[0]),
                "cache_file": str(path),
                "candidate_count": int(len(obj)),
            }
        )
    if not rows:
        raise RuntimeError("未构建出候选缓存索引")
    df = pd.DataFrame(rows).drop_duplicates(["signal_date"], keep="last").sort_values("signal_date").reset_index(drop=True)
    df.to_csv(cache_index_csv, index=False, encoding="utf-8-sig")
    return df


def _build_target_cache_index(
    result_dir: Path,
    data_dir: Path,
    target_dates: list[pd.Timestamp],
    max_workers: int,
) -> pd.DataFrame:
    cache_dir = result_dir / "target_candidate_cache_parts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing_index = result_dir / "date_cache_index.csv"
    if existing_index.exists():
        df = pd.read_csv(existing_index, parse_dates=["signal_date"])
        if not df.empty and df["cache_file"].map(lambda x: Path(str(x)).exists()).all():
            update_progress(
                result_dir,
                "reusing_target_candidate_cache",
                cached_dates=int(len(df)),
                first_date=pd.Timestamp(df["signal_date"].min()).strftime("%Y-%m-%d"),
                last_date=pd.Timestamp(df["signal_date"].max()).strftime("%Y-%m-%d"),
            )
            return df.sort_values("signal_date").reset_index(drop=True)
    daily_dir = model_search.case_first._resolve_daily_dir(data_dir)
    file_paths = sorted(daily_dir.glob("*.txt"))
    file_path_strs = [str(path) for path in file_paths]
    chunk_size = max(32, math.ceil(len(file_path_strs) / max(1, max_workers)))
    file_chunks = [file_path_strs[i : i + chunk_size] for i in range(0, len(file_path_strs), chunk_size)]
    target_date_list = sorted(pd.Timestamp(x).strftime("%Y-%m-%d") for x in target_dates)
    payloads = [(chunk, target_date_list, case_recall.CASE_SEQ_LENS) for chunk in file_chunks]

    chunk_rows = _pool_map(_records_for_file_chunk_local, payloads, max_workers=max_workers)
    merged_rows: list[dict[str, Any]] = []
    for idx, items in enumerate(chunk_rows, start=1):
        if items:
            merged_rows.extend(items)
        if idx == 1 or idx % 5 == 0 or idx == len(chunk_rows):
            update_progress(
                result_dir,
                "building_target_candidate_cache",
                done_chunks=int(idx),
                total_chunks=int(len(chunk_rows)),
                scanned_files=int(min(idx * chunk_size, len(file_path_strs))),
                total_files=int(len(file_path_strs)),
                target_dates=int(len(target_dates)),
            )

    if not merged_rows:
        raise RuntimeError("目标日期候选缓存为空")
    cache_df = pd.DataFrame(merged_rows).sort_values(["signal_date", "code"]).reset_index(drop=True)
    cache_df["signal_date"] = pd.to_datetime(cache_df["signal_date"])

    rows: list[dict[str, Any]] = []
    grouped = list(cache_df.groupby("signal_date", sort=True))
    for idx, (date_value, g) in enumerate(grouped, start=1):
        path = cache_dir / f"date_cache_{idx:05d}.pkl"
        g.to_pickle(path)
        rows.append(
            {
                "signal_date": pd.Timestamp(date_value),
                "cache_file": str(path),
                "candidate_count": int(len(g)),
            }
        )
        if idx == 1 or idx % 100 == 0 or idx == len(grouped):
            update_progress(
                result_dir,
                "writing_target_candidate_cache",
                done_dates=int(idx),
                total_dates=int(len(grouped)),
                latest_date=pd.Timestamp(date_value).strftime("%Y-%m-%d"),
            )

    out = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    out.to_csv(result_dir / "date_cache_index.csv", index=False, encoding="utf-8-sig")
    return out


def _load_date_cache(path: str | Path) -> pd.DataFrame:
    obj = pickle.loads(Path(path).read_bytes())
    if not isinstance(obj, pd.DataFrame):
        raise RuntimeError(f"缓存文件不是 DataFrame: {path}")
    df = obj.copy()
    if "signal_date" in df.columns:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df


def _build_stem_map(directory: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(directory.glob("*.txt")):
        key = case_semantics.code_key(path.stem)
        if key:
            mapping[key] = path.stem
    return mapping


def _resolve_daily_stem(code: Any) -> str:
    global _DAILY_STEM_MAP
    if _DAILY_STEM_MAP is None:
        _DAILY_STEM_MAP = _build_stem_map(DATA_DIR)
    text = str(code)
    key = case_semantics.code_key(text)
    if key.isdigit() and len(key) < 6:
        key = key.zfill(6)
    return _DAILY_STEM_MAP.get(key, text)


def _commonality_score(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    close_col = pd.to_numeric(df["close_location"], errors="coerce").fillna(0.0) if "close_location" in df.columns else pd.Series(0.0, index=df.index, dtype=float)
    rebound_col = pd.to_numeric(df["rebound_ratio"], errors="coerce").fillna(0.0) if "rebound_ratio" in df.columns else pd.Series(0.0, index=df.index, dtype=float)
    upper_col = pd.to_numeric(df["upper_shadow_pct"], errors="coerce").fillna(0.0) if "upper_shadow_pct" in df.columns else pd.Series(0.0, index=df.index, dtype=float)
    close_score = close_col.rank(method="average", pct=True, ascending=True)
    rebound_score = rebound_col.rank(method="average", pct=True, ascending=True)
    upper_penalty = upper_col.rank(method="average", pct=True, ascending=True)
    return close_score + rebound_score - upper_penalty


def _nearer_line(df: pd.DataFrame) -> pd.Series:
    close_to_trend = pd.to_numeric(df.get("close_to_trend", np.nan), errors="coerce").fillna(np.nan).abs()
    close_to_long = pd.to_numeric(df.get("close_to_long", np.nan), errors="coerce").fillna(np.nan).abs()
    out = np.where(close_to_trend <= close_to_long, "trend", "long")
    return pd.Series(out, index=df.index, dtype="object")


def _prepare_enriched_for_date(
    target_date: pd.Timestamp,
    cache_file: str,
    labeled_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    cand_df = _load_date_cache(cache_file)
    if cand_df.empty:
        return cand_df
    if "seq_map" in cand_df.columns:
        required_lens = tuple(int(x) for x in case_recall.CASE_SEQ_LENS)
        seq_ok = cand_df["seq_map"].map(
            lambda x: isinstance(x, dict) and all(int(k) in x for k in required_lens)
        )
        cand_df = cand_df[seq_ok].copy()
    if cand_df.empty:
        return cand_df
    enriched = case_recall.enrich_candidates_for_date(pd.Timestamp(target_date), cand_df, DATA_DIR)
    if enriched.empty:
        return enriched
    out = enriched.copy()
    date_key = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["case_key"] = out["code"].astype(str) + "|" + out["signal_date"].dt.strftime("%Y-%m-%d")
    out["explicit_case"] = out["case_key"].isin(labeled_pairs)
    out["commonality_score"] = _commonality_score(out)
    out["nearer_line"] = _nearer_line(out)
    return out


def _enrich_date_worker(payload: tuple[pd.Timestamp, str, set[tuple[str, str]]]) -> pd.DataFrame:
    target_date, cache_file, labeled_pairs = payload
    return _prepare_enriched_for_date(target_date, cache_file, labeled_pairs)


def _build_target_cache_worker(payload: tuple[Path, pd.Timestamp, list[int] | tuple[int, ...]]) -> pd.DataFrame:
    data_dir, target_date, required_lens = payload
    return model_search.case_first.build_candidates_for_date(
        pd.Timestamp(target_date),
        data_dir,
        max_workers=1,
        required_lens=required_lens,
    )


def _records_for_file_chunk_local(payload: tuple[list[str], list[str], list[int] | tuple[int, ...]]) -> list[dict[str, Any]]:
    file_path_list, target_date_list, required_lens = payload
    target_date_set = set(target_date_list)
    rows: list[dict[str, Any]] = []
    for file_path_str in file_path_list:
        rows.extend(model_search.case_first._records_for_file(file_path_str, target_date_set, required_lens))
    return rows


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for col in model_search.FEATURE_COLS:
        if col not in x.columns:
            x[col] = 0.0
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    return x


def _fit_and_score(train_df: pd.DataFrame, score_df: pd.DataFrame, model_name: str, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train_x = _prepare_features(train_df)
    score_x = _prepare_features(score_df)
    if model_name == "heuristic":
        train_scores = pd.to_numeric(train_df["recall_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        score_scores = pd.to_numeric(score_df["recall_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return train_scores, score_scores
    model = model_search._build_model(model_name, params)
    model.fit(train_x[model_search.FEATURE_COLS], train_df["label"].astype(int))
    if hasattr(model, "predict_proba"):
        train_scores = model.predict_proba(train_x[model_search.FEATURE_COLS])[:, 1]
        score_scores = model.predict_proba(score_x[model_search.FEATURE_COLS])[:, 1]
    else:
        train_scores = model.predict(train_x[model_search.FEATURE_COLS]).astype(float)
        score_scores = model.predict(score_x[model_search.FEATURE_COLS]).astype(float)
    return np.asarray(train_scores, dtype=float), np.asarray(score_scores, dtype=float)


def _apply_filter_for_date(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df[pd.to_numeric(df["model_score"], errors="coerce").fillna(-1.0) >= float(threshold)].copy()
    if out.empty:
        return out
    out = out.sort_values(
        ["commonality_score", "model_score", "code"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(TOPN).reset_index(drop=True)
    return out


def _config_payloads() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for model_name, grids in model_search.MODEL_GRIDS.items():
        if model_name == "xgb" and not getattr(model_search, "XGB_OK", False):
            continue
        if model_name == "lgbm" and not getattr(model_search, "LGBM_OK", False):
            continue
        for params in grids:
            payloads.append({"model_name": model_name, "params": params})
    return payloads


def _eval_one_config(payload: dict[str, Any]) -> dict[str, Any]:
    model_name = str(payload["model_name"])
    params = payload["params"]
    date_frames: list[tuple[pd.Timestamp, pd.DataFrame]] = payload["date_frames"]
    min_train_pos = int(payload["min_train_pos"])
    min_train_neg = int(payload["min_train_neg"])

    selected_count = 0
    selected_positive = 0
    total_positive = 0
    eval_dates = 0
    positive_eval_dates = 0
    hit_dates = 0
    reciprocal_ranks: list[float] = []

    for idx, (eval_date, df_eval) in enumerate(date_frames):
        train_parts = [frame.assign(_date=d) for d, frame in date_frames[:idx] if not frame.empty]
        if not train_parts:
            continue
        train_df = pd.concat(train_parts, ignore_index=True)
        pos_train = int((train_df["label"] == 1).sum())
        neg_train = int((train_df["label"] == 0).sum())
        if pos_train < min_train_pos or neg_train < min_train_neg:
            continue

        eval_df = df_eval.copy()
        if eval_df.empty:
            continue
        train_scores, eval_scores = _fit_and_score(train_df, eval_df, model_name, params)
        train_df = train_df.copy()
        eval_df = eval_df.copy()
        train_df["model_score"] = train_scores
        eval_df["model_score"] = eval_scores
        threshold = float(pd.Series(train_df.loc[train_df["label"] == 1, "model_score"]).quantile(THRESHOLD_QUANTILE))

        filtered = _apply_filter_for_date(eval_df, threshold)
        eval_positive = int((eval_df["label"] == 1).sum())
        total_positive += eval_positive
        eval_dates += 1
        if eval_positive > 0:
            positive_eval_dates += 1

        if eval_positive > 0:
            ranked = eval_df.sort_values(["model_score", "commonality_score", "code"], ascending=[False, False, True]).reset_index(drop=True)
            pos_idx = ranked.index[ranked["label"] == 1].tolist()
            if pos_idx:
                reciprocal_ranks.append(1.0 / float(pos_idx[0] + 1))

        selected_count += int(len(filtered))
        if not filtered.empty:
            hits = int((filtered["label"] == 1).sum())
            selected_positive += hits
            if hits > 0:
                hit_dates += 1

    precision = float(selected_positive / selected_count) if selected_count > 0 else float("nan")
    case_recall = float(selected_positive / total_positive) if total_positive > 0 else float("nan")
    date_recall = float(hit_dates / positive_eval_dates) if positive_eval_dates > 0 else float("nan")
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
    return {
        "model_name": model_name,
        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "case_recall": case_recall,
        "date_recall": date_recall,
        "precision": precision,
        "mrr": mrr,
        "selected_count": int(selected_count),
        "selected_positive": int(selected_positive),
        "total_positive": int(total_positive),
        "eval_dates": int(eval_dates),
        "positive_eval_dates": int(positive_eval_dates),
        "hit_dates": int(hit_dates),
    }


def _pool_map(func, payloads: list[Any], max_workers: int) -> list[Any]:
    if not payloads:
        return []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max_workers) as pool:
        return pool.map(func, payloads, chunksize=1)


def _build_labeled_dataset(
    result_dir: Path,
    cache_index: pd.DataFrame,
    pos_cases: pd.DataFrame,
    neg_cases: pd.DataFrame,
    max_workers: int,
    date_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos_map = {(str(r.code), pd.Timestamp(r.signal_date).strftime("%Y-%m-%d")) for r in pos_cases.itertuples(index=False)}
    neg_map = {(str(r.code), pd.Timestamp(r.signal_date).strftime("%Y-%m-%d")) for r in neg_cases.itertuples(index=False)}
    labeled_pairs = pos_map | neg_map
    labeled_dates = sorted(set(pd.to_datetime(pos_cases["signal_date"]).tolist()) | set(pd.to_datetime(neg_cases["signal_date"]).tolist()))
    available = cache_index[cache_index["signal_date"].isin(labeled_dates)].copy().sort_values("signal_date").reset_index(drop=True)
    if date_limit > 0:
        available = available.tail(date_limit).reset_index(drop=True)
    update_progress(result_dir, "enriching_labeled_dates", total_dates=int(len(available)))
    payloads = [
        (pd.Timestamp(row.signal_date), str(row.cache_file), labeled_pairs)
        for row in available.itertuples(index=False)
    ]

    date_frames = _pool_map(_enrich_date_worker, payloads, max_workers=max_workers)
    rows: list[pd.DataFrame] = []
    date_rows: list[dict[str, Any]] = []
    for idx, (row, df) in enumerate(zip(available.itertuples(index=False), date_frames), start=1):
        date_key = pd.Timestamp(row.signal_date).strftime("%Y-%m-%d")
        if df.empty:
            continue
        local = df.copy()
        local["date_str"] = local["signal_date"].dt.strftime("%Y-%m-%d")
        local["label"] = 0
        local["label_source"] = "other_candidate"
        pos_mask = local.apply(lambda r: (str(r["code"]), str(r["date_str"])) in pos_map, axis=1)
        neg_mask = local.apply(lambda r: (str(r["code"]), str(r["date_str"])) in neg_map, axis=1)
        local.loc[pos_mask, "label"] = 1
        local.loc[pos_mask, "label_source"] = "positive_case"
        local.loc[neg_mask, "label"] = 0
        local.loc[neg_mask, "label_source"] = "negative_case"
        rows.append(local)
        date_rows.append(
            {
                "signal_date": pd.Timestamp(row.signal_date),
                "candidate_count": int(len(local)),
                "positive_count": int(pos_mask.sum()),
                "negative_case_count": int(neg_mask.sum()),
                "other_negative_count": int((~pos_mask & ~neg_mask).sum()),
            }
        )
        if idx == 1 or idx % 10 == 0 or idx == len(available):
            update_progress(result_dir, "enriching_labeled_dates", done_dates=int(idx), total_dates=int(len(available)), latest_date=date_key)

    if not rows:
        raise RuntimeError("标注数据集为空，无法继续")
    labeled_dataset = pd.concat(rows, ignore_index=True)
    labeled_dataset["signal_date"] = pd.to_datetime(labeled_dataset["signal_date"])
    labeled_summary = pd.DataFrame(date_rows).sort_values("signal_date").reset_index(drop=True)
    labeled_dataset.to_pickle(result_dir / "labeled_dataset.pkl")
    labeled_summary.to_csv(result_dir / "labeled_date_summary.csv", index=False, encoding="utf-8-sig")
    return labeled_dataset, labeled_summary


def _search_best_model(
    result_dir: Path,
    labeled_dataset: pd.DataFrame,
    labeled_summary: pd.DataFrame,
    max_workers: int,
    date_limit: int,
) -> dict[str, Any]:
    date_frames: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for date_value in sorted(pd.to_datetime(labeled_summary["signal_date"]).tolist()):
        frame = labeled_dataset[labeled_dataset["signal_date"] == pd.Timestamp(date_value)].copy().reset_index(drop=True)
        date_frames.append((pd.Timestamp(date_value), frame))
    if date_limit > 0:
        date_frames = date_frames[-date_limit:]
    payloads = []
    for cfg in _config_payloads():
        payloads.append(
            {
                "model_name": cfg["model_name"],
                "params": cfg["params"],
                "date_frames": date_frames,
                "min_train_pos": MIN_TRAIN_POS,
                "min_train_neg": MIN_TRAIN_NEG,
            }
        )
    update_progress(result_dir, "evaluating_model_configs", total_configs=int(len(payloads)))
    rows = _pool_map(_eval_one_config, payloads, max_workers=max_workers)
    eval_df = pd.DataFrame(rows).sort_values(
        ["case_recall", "date_recall", "precision", "mrr", "selected_positive"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    eval_df.to_csv(result_dir / "model_search_summary.csv", index=False, encoding="utf-8-sig")
    if eval_df.empty:
        raise RuntimeError("模型搜索结果为空")
    best = eval_df.iloc[0].to_dict()
    best["params"] = json.loads(str(best.pop("params_json")))
    (result_dir / "best_model.json").write_text(json.dumps(best, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return best


def _load_daily_df(code: str) -> pd.DataFrame:
    stem = _resolve_daily_stem(code)
    path = DATA_DIR / f"{stem}.txt"
    if not path.exists():
        return pd.DataFrame()
    df = case_semantics.sim.load_stock_data(str(path))
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    cfg = GLOBAL_CFG
    if cfg is not None:
        out = _apply_optional_exclude(out, "date", cfg.use_default_exclude)
        out = _apply_date_window(out, "date", cfg.backtest_start_date, cfg.backtest_end_date)
    return out.sort_values("date").reset_index(drop=True)


def _next_trade_date(daily_df: pd.DataFrame, current_date: pd.Timestamp) -> pd.Timestamp | None:
    later = daily_df[daily_df["date"] > pd.Timestamp(current_date)]["date"]
    if later.empty:
        return None
    return pd.Timestamp(later.iloc[0])


def _simulate_trade_one_profile(row: dict[str, Any], tp_pct: float) -> dict[str, Any] | None:
    code = str(row["code"])
    signal_date = pd.Timestamp(row["signal_date"])
    daily_df = _load_daily_df(code)
    if daily_df.empty:
        return None
    entry_date = _next_trade_date(daily_df, signal_date)
    if entry_date is None:
        return None
    entry_match = daily_df[daily_df["date"] == entry_date]
    if entry_match.empty:
        return None
    entry_row = entry_match.iloc[0]
    entry_price = float(entry_row["open"])
    stop_price = float(entry_row["low"])
    tp_price = float(entry_price * (1.0 + tp_pct))

    hold_window = [pd.Timestamp(x) for x in daily_df[daily_df["date"] >= entry_date]["date"].head(MAX_HOLD_DAYS).tolist()]
    if len(hold_window) < MAX_HOLD_DAYS:
        return None

    pending_open_exit: pd.Timestamp | None = None
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason = ""

    for current_date in hold_window:
        day_row = daily_df[daily_df["date"] == current_date].iloc[0]
        if pending_open_exit is not None and current_date == pending_open_exit:
            exit_date = current_date
            exit_price = float(day_row["open"])
            exit_reason = f"tp_next_open_{tp_pct:.2%}"
            break

        if float(day_row["low"]) <= stop_price:
            exit_date = current_date
            exit_price = stop_price
            exit_reason = "stop_same_day_fixed"
            break

        if float(day_row["high"]) >= tp_price:
            pending_open_exit = _next_trade_date(daily_df, current_date)
            if pending_open_exit is None:
                exit_date = current_date
                exit_price = float(day_row["close"])
                exit_reason = f"tp_fallback_close_{tp_pct:.2%}"
                break

    if exit_date is None:
        force_exit_date = _next_trade_date(daily_df, hold_window[-1])
        if force_exit_date is not None:
            force_row = daily_df[daily_df["date"] == force_exit_date].iloc[0]
            exit_date = force_exit_date
            exit_price = float(force_row["open"])
            exit_reason = "hold_3_next_open"
        else:
            force_row = daily_df[daily_df["date"] == hold_window[-1]].iloc[0]
            exit_date = hold_window[-1]
            exit_price = float(force_row["close"])
            exit_reason = "hold_3_fallback_close"

    if exit_date is None or exit_price is None:
        return None
    holding_days = int(((daily_df["date"] >= entry_date) & (daily_df["date"] <= pd.Timestamp(exit_date))).sum())
    return {
        "profile_name": f"fixed_tp_{tp_pct:.2%}",
        "tp_pct": float(tp_pct),
        "code": code,
        "signal_date": signal_date,
        "entry_date": entry_date,
        "entry_price": float(entry_price),
        "entry_low": float(entry_row["low"]),
        "stop_price": float(stop_price),
        "exit_date": pd.Timestamp(exit_date),
        "exit_price": float(exit_price),
        "exit_reason": str(exit_reason),
        "commonality_score": float(row["commonality_score"]),
        "model_score": float(row["model_score"]),
        "close_location": float(row.get("close_location", np.nan)),
        "rebound_ratio": float(row.get("rebound_ratio", np.nan)),
        "upper_shadow_pct": float(row.get("upper_shadow_pct", np.nan)),
        "nearer_line": str(row.get("nearer_line", "")),
        "close_to_trend": float(row.get("close_to_trend", np.nan)),
        "close_to_long": float(row.get("close_to_long", np.nan)),
        "return_pct_raw": float(exit_price / entry_price - 1.0),
        "holding_days": holding_days,
    }


def _simulate_code_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload["rows"]
    tp_levels = payload["tp_levels"]
    trade_rows: list[dict[str, Any]] = []
    for row in rows:
        for tp_pct in tp_levels:
            sim = _simulate_trade_one_profile(row, float(tp_pct))
            if sim is not None:
                trade_rows.append(sim)
    return trade_rows


def _load_close_payload(code: str) -> tuple[str, pd.Series | None]:
    stem = _resolve_daily_stem(code)
    path = DATA_DIR / f"{stem}.txt"
    if not path.exists():
        return code, None
    df = case_semantics.sim.load_stock_data(str(path))
    if df is None or df.empty:
        return code, None
    cfg = GLOBAL_CFG
    if cfg is not None:
        df = _apply_optional_exclude(df, "date", cfg.use_default_exclude)
        df = _apply_date_window(df, "date", cfg.backtest_start_date, cfg.backtest_end_date)
    if df.empty:
        return code, None
    s = df[["date", "close"]].dropna(subset=["date", "close"]).set_index("date")["close"].astype(float)
    return code, s


def _build_close_map_parallel(codes: list[str], result_dir: Path, max_workers: int) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    unique_codes = sorted(set(map(str, codes)))
    if not unique_codes:
        return pd.DatetimeIndex([]), {}

    payloads = unique_codes
    results = _pool_map(_load_close_payload, payloads, max_workers=max_workers)
    relevant: dict[str, pd.Series] = {}
    all_dates: set[pd.Timestamp] = set()
    for idx, (code, series) in enumerate(results, start=1):
        if series is not None and not series.empty:
            relevant[code] = series
            all_dates.update(pd.DatetimeIndex(series.index).tolist())
        if idx == 1 or idx % 100 == 0 or idx == len(results):
            update_progress(result_dir, "building_close_map", done_codes=int(idx), total_codes=int(len(results)), loaded_codes=int(len(relevant)))

    market_dates = pd.DatetimeIndex(sorted(all_dates))
    close_map = {code: series.reindex(market_dates).ffill() for code, series in relevant.items()}
    return market_dates, close_map


def _simulate_account(trades: pd.DataFrame, result_dir: Path, max_workers: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if trades.empty:
        return pd.DataFrame(), {}
    market_dates, close_map = _build_close_map_parallel(trades["code"].astype(str).unique().tolist(), result_dir, max_workers=max_workers)
    if len(market_dates) == 0:
        raise RuntimeError("账户层 close_map 为空")

    rows: list[dict[str, Any]] = []
    equity_map: dict[str, pd.DataFrame] = {}
    config = replace(
        real_account.AccountConfig(),
        initial_capital=INITIAL_CAPITAL,
        max_positions=MAX_POSITIONS,
        daily_new_limit=MAX_POSITIONS,
        allocation_mode="equal",
    )

    for profile_name, g in trades.groupby("profile_name", sort=True):
        entries_by_date = {
            d: gg.sort_values(["commonality_score", "model_score", "code"], ascending=[False, False, True]).to_dict("records")
            for d, gg in g.groupby("entry_date")
        }
        open_exits_by_date = {
            d: gg[gg["exit_reason"] != "stop_same_day_fixed"].to_dict("records")
            for d, gg in g.groupby("exit_date")
        }
        same_day_stop_by_date = {
            d: gg[gg["exit_reason"] == "stop_same_day_fixed"].to_dict("records")
            for d, gg in g.groupby("exit_date")
        }

        cash = float(config.initial_capital)
        positions: dict[str, dict[str, Any]] = {}
        executed_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []

        for current_date in market_dates:
            todays_open_exits = open_exits_by_date.get(current_date, [])
            for tr in todays_open_exits:
                code = str(tr["code"])
                if code not in positions:
                    continue
                pos = positions.pop(code)
                raw_exit = float(tr["exit_price"])
                exit_price = raw_exit * (1.0 - config.slippage_rate)
                gross_cash = pos["shares"] * exit_price
                fee = gross_cash * config.commission_rate
                tax = gross_cash * config.stamp_duty_rate
                cash += gross_cash - fee - tax
                pnl = (exit_price - pos["entry_price"]) * pos["shares"] - pos["entry_fee"] - fee - tax
                base_cost = pos["entry_price"] * pos["shares"] + pos["entry_fee"]
                executed_rows.append({**tr, "shares": pos["shares"], "pnl": pnl, "return_pct_net": pnl / base_cost if base_cost > 0 else float("nan")})

            equity_before_entry = cash
            for code, pos in positions.items():
                mark_price = float(close_map[code].get(current_date, pos["entry_price"]))
                equity_before_entry += pos["shares"] * mark_price

            todays_entries = entries_by_date.get(current_date, [])
            available_slots = max(config.max_positions - len(positions), 0)
            if todays_entries and available_slots > 0:
                selected_entries = []
                for tr in todays_entries:
                    code = str(tr["code"])
                    if code in positions:
                        continue
                    selected_entries.append(tr)
                    if len(selected_entries) >= available_slots:
                        break
                if selected_entries:
                    investable = min(cash, equity_before_entry * config.daily_budget_frac)
                    if investable > 0:
                        weights = np.full(len(selected_entries), 1.0 / len(selected_entries), dtype=float)
                        per_pos_cap = equity_before_entry * config.position_cap_frac
                        for tr, weight in zip(selected_entries, weights):
                            code = str(tr["code"])
                            raw_entry = float(tr["entry_price"])
                            entry_price = raw_entry * (1.0 + config.slippage_rate)
                            alloc = min(investable * float(weight), per_pos_cap, cash)
                            shares = int(alloc / entry_price / config.min_lot) * config.min_lot if alloc > 0 and entry_price > 0 else 0
                            if shares <= 0:
                                continue
                            gross_cost = shares * entry_price
                            fee = gross_cost * config.commission_rate
                            total_cost = gross_cost + fee
                            if total_cost > cash:
                                continue
                            cash -= total_cost
                            positions[code] = {
                                "shares": shares,
                                "entry_price": entry_price,
                                "entry_fee": fee,
                                "entry_price_raw": raw_entry,
                                "entry_date": current_date,
                            }

            todays_stop_exits = same_day_stop_by_date.get(current_date, [])
            for tr in todays_stop_exits:
                code = str(tr["code"])
                if code not in positions:
                    continue
                pos = positions.pop(code)
                raw_exit = float(tr["exit_price"])
                exit_price = raw_exit
                gross_cash = pos["shares"] * exit_price
                fee = gross_cash * config.commission_rate
                tax = gross_cash * config.stamp_duty_rate
                cash += gross_cash - fee - tax
                pnl = (exit_price - pos["entry_price"]) * pos["shares"] - pos["entry_fee"] - fee - tax
                base_cost = pos["entry_price"] * pos["shares"] + pos["entry_fee"]
                executed_rows.append({**tr, "shares": pos["shares"], "pnl": pnl, "return_pct_net": pnl / base_cost if base_cost > 0 else float("nan")})

            equity = cash
            for code, pos in positions.items():
                mark_price = float(close_map[code].get(current_date, pos["entry_price"]))
                equity += pos["shares"] * mark_price
            equity_rows.append({"date": current_date, "equity": equity, "cash": cash, "position_count": len(positions)})

        equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
        executed_df = pd.DataFrame(executed_rows).sort_values(["exit_date", "entry_date", "code"]).reset_index(drop=True) if executed_rows else pd.DataFrame()
        equity_curve = pd.Series(equity_df["equity"].to_numpy(dtype=float), index=pd.DatetimeIndex(equity_df["date"]), dtype=float)
        metrics = compute_metrics(equity_curve)
        max_drawdown_abs = float(metrics["max_drawdown"])
        annual_return = float(metrics["annual_return"])
        sharpe = float(metrics["sharpe"])
        calmar = real_account._compute_calmar(annual_return, max_drawdown_abs)
        avg_trade_return = float(executed_df["return_pct_net"].mean()) if not executed_df.empty else float("nan")
        avg_holding_return = float(executed_df["return_pct_raw"].mean()) if not executed_df.empty else float("nan")
        avg_holding_days = float(pd.to_numeric(executed_df["holding_days"], errors="coerce").mean()) if not executed_df.empty and "holding_days" in executed_df.columns else float("nan")
        success_rate = float((executed_df["return_pct_net"] > 0).mean()) if not executed_df.empty else float("nan")
        hold_return = float(equity_df.iloc[-1]["equity"] / config.initial_capital - 1.0) if not equity_df.empty else float("nan")
        rows.append(
            {
                "profile_name": str(profile_name),
                "tp_pct": float(g["tp_pct"].iloc[0]),
                "final_multiple": float(metrics["final_multiple"]),
                "annual_return": annual_return,
                "holding_return": hold_return,
                "max_drawdown": -max_drawdown_abs,
                "sharpe": sharpe,
                "calmar": calmar,
                "trade_count": int(len(executed_df)),
                "success_rate": success_rate,
                "avg_trade_return": avg_trade_return,
                "avg_holding_return": avg_holding_return,
                "avg_holding_days": avg_holding_days,
                "max_losing_streak": int(real_account._max_losing_streak(executed_df["return_pct_net"].tolist() if not executed_df.empty else [])),
                "equity_days": int(metrics["days"]),
                "final_equity": float(equity_df.iloc[-1]["equity"]) if not equity_df.empty else float("nan"),
            }
        )
        equity_map[str(profile_name)] = equity_df

    summary_df = pd.DataFrame(rows).sort_values(
        ["annual_return", "holding_return", "max_drawdown", "sharpe", "calmar"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return summary_df, equity_map


def run_experiment(cfg: WalkforwardConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(cfg.output_dir, "loading_cases", mode=cfg.mode, max_workers=cfg.max_workers)

    pos_cases = _load_case_pairs(POS_CASE_DIR, 1)
    neg_cases = _load_case_pairs(NEG_CASE_DIR, 0)
    if cfg.use_default_exclude:
        pos_cases = _apply_optional_exclude(pos_cases, "signal_date", True)
        neg_cases = _apply_optional_exclude(neg_cases, "signal_date", True)
    pos_cases.to_csv(cfg.output_dir / "positive_cases.csv", index=False, encoding="utf-8-sig")
    neg_cases.to_csv(cfg.output_dir / "negative_cases.csv", index=False, encoding="utf-8-sig")
    if pos_cases.empty:
        raise RuntimeError("正例为空，无法训练")

    all_dates = daily_stream_base.collect_trading_dates(cfg.data_dir, max_workers=cfg.max_workers)
    if cfg.use_default_exclude:
        all_dates = [d for d in all_dates if _is_allowed_date(pd.Timestamp(d))]
    labeled_all_dates = sorted(
        set(pd.to_datetime(pos_cases["signal_date"]).tolist()) | set(pd.to_datetime(neg_cases["signal_date"]).tolist())
    )
    if cfg.labeled_eval_date_limit > 0:
        labeled_use_dates = labeled_all_dates[-cfg.labeled_eval_date_limit :]
    else:
        labeled_use_dates = labeled_all_dates
    labeled_use_date_strs = {pd.Timestamp(x).strftime("%Y-%m-%d") for x in labeled_use_dates}
    backtest_candidates = [d for d in all_dates if pd.Timestamp(d).strftime("%Y-%m-%d") not in labeled_use_date_strs]
    if cfg.backtest_start_date is not None:
        backtest_candidates = [d for d in backtest_candidates if pd.Timestamp(d) >= pd.Timestamp(cfg.backtest_start_date)]
    if cfg.backtest_end_date is not None:
        backtest_candidates = [d for d in backtest_candidates if pd.Timestamp(d) <= pd.Timestamp(cfg.backtest_end_date)]
    if cfg.backtest_date_limit > 0:
        backtest_use_dates = backtest_candidates[-cfg.backtest_date_limit :]
    else:
        backtest_use_dates = backtest_candidates
    target_cache_dates = sorted(set(map(pd.Timestamp, labeled_use_dates)) | set(map(pd.Timestamp, backtest_use_dates)))
    update_progress(
        cfg.output_dir,
        "building_target_candidate_cache",
        target_dates=int(len(target_cache_dates)),
        first_date=pd.Timestamp(target_cache_dates[0]).strftime("%Y-%m-%d") if target_cache_dates else None,
        last_date=pd.Timestamp(target_cache_dates[-1]).strftime("%Y-%m-%d") if target_cache_dates else None,
    )
    cache_index = _build_target_cache_index(cfg.output_dir, cfg.data_dir, target_cache_dates, cfg.max_workers)
    labeled_dataset, labeled_summary = _build_labeled_dataset(
        cfg.output_dir,
        cache_index[cache_index["signal_date"].isin(labeled_use_dates)].reset_index(drop=True),
        pos_cases,
        neg_cases,
        max_workers=cfg.max_workers,
        date_limit=cfg.labeled_eval_date_limit,
    )
    best_model = _search_best_model(
        cfg.output_dir,
        labeled_dataset,
        labeled_summary,
        max_workers=cfg.max_workers,
        date_limit=cfg.labeled_eval_date_limit,
    )

    labeled_dates = set(pd.to_datetime(labeled_summary["signal_date"]).dt.strftime("%Y-%m-%d"))
    backtest_dates = cache_index[
        cache_index["signal_date"].dt.strftime("%Y-%m-%d").isin({pd.Timestamp(x).strftime("%Y-%m-%d") for x in backtest_use_dates})
        & (~cache_index["signal_date"].dt.strftime("%Y-%m-%d").isin(labeled_dates))
    ].copy().reset_index(drop=True)
    update_progress(cfg.output_dir, "preparing_backtest_dates", total_dates=int(len(backtest_dates)))

    labeled_pairs = set(
        pos_cases.assign(date_str=pos_cases["signal_date"].dt.strftime("%Y-%m-%d"))[["code", "date_str"]].itertuples(index=False, name=None)
    ) | set(
        neg_cases.assign(date_str=neg_cases["signal_date"].dt.strftime("%Y-%m-%d"))[["code", "date_str"]].itertuples(index=False, name=None)
    )

    payloads = [(pd.Timestamp(r.signal_date), str(r.cache_file), labeled_pairs) for r in backtest_dates.itertuples(index=False)]
    backtest_date_frames = _pool_map(_enrich_date_worker, payloads, max_workers=cfg.max_workers)
    selected_rows: list[dict[str, Any]] = []

    model_name = str(best_model["model_name"])
    params = best_model["params"]
    labeled_dataset = labeled_dataset.copy()
    labeled_dataset["signal_date"] = pd.to_datetime(labeled_dataset["signal_date"])

    snapshot_train_cache: dict[str, dict[str, Any]] = {}
    total_bt = len(backtest_dates)
    for idx, (bt_row, frame) in enumerate(zip(backtest_dates.itertuples(index=False), backtest_date_frames), start=1):
        target_date = pd.Timestamp(bt_row.signal_date)
        if frame.empty:
            continue
        train_df = labeled_dataset[labeled_dataset["signal_date"] < target_date].copy()
        pos_train = int((train_df["label"] == 1).sum())
        neg_train = int((train_df["label"] == 0).sum())
        if pos_train < MIN_TRAIN_POS or neg_train < MIN_TRAIN_NEG:
            continue
        snapshot_key = f"{int((train_df['label'] == 1).sum())}_{int((train_df['label'] == 0).sum())}"
        if snapshot_key not in snapshot_train_cache:
            train_scores, _ = _fit_and_score(train_df, train_df.iloc[:1].copy(), model_name, params)
            threshold = float(pd.Series(train_scores[train_df["label"].to_numpy(dtype=int) == 1]).quantile(THRESHOLD_QUANTILE))
            if model_name == "heuristic":
                snapshot_train_cache[snapshot_key] = {"threshold": threshold, "train_df": None}
            else:
                model = model_search._build_model(model_name, params)
                train_x = _prepare_features(train_df)
                model.fit(train_x[model_search.FEATURE_COLS], train_df["label"].astype(int))
                snapshot_train_cache[snapshot_key] = {"threshold": threshold, "model": model}

        cached = snapshot_train_cache[snapshot_key]
        score_df = frame.copy()
        if model_name == "heuristic":
            score_df["model_score"] = pd.to_numeric(score_df["recall_score"], errors="coerce").fillna(0.0)
        else:
            score_x = _prepare_features(score_df)
            score_df["model_score"] = cached["model"].predict_proba(score_x[model_search.FEATURE_COLS])[:, 1]
        filtered = _apply_filter_for_date(score_df, float(cached["threshold"]))
        if not filtered.empty:
            selected_rows.extend(filtered.to_dict("records"))
        if idx == 1 or idx % 20 == 0 or idx == total_bt:
            update_progress(
                cfg.output_dir,
                "scoring_backtest_dates",
                done_dates=int(idx),
                total_dates=int(total_bt),
                latest_date=target_date.strftime("%Y-%m-%d"),
                selected_rows=int(len(selected_rows)),
            )

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(cfg.output_dir / "selected_signals.csv", index=False, encoding="utf-8-sig")
    if selected_df.empty:
        raise RuntimeError("回测信号为空，无法继续")

    payloads = []
    for code, g in selected_df.groupby("code", sort=True):
        payloads.append(
            {
                "code": str(code),
                "rows": g.sort_values(["signal_date", "commonality_score", "model_score"], ascending=[True, False, False]).to_dict("records"),
                "tp_levels": BUY_TP_LEVELS,
            }
        )
    trade_parts = _pool_map(_simulate_code_payload, payloads, max_workers=cfg.max_workers)
    trade_rows = [row for part in trade_parts for row in part]
    trades = pd.DataFrame(trade_rows).sort_values(["profile_name", "entry_date", "commonality_score", "code"], ascending=[True, True, False, True]).reset_index(drop=True)
    trades.to_csv(cfg.output_dir / "trades.csv", index=False, encoding="utf-8-sig")
    if trades.empty:
        raise RuntimeError("交易明细为空，无法做账户层回测")

    account_summary, equity_map = _simulate_account(trades, cfg.output_dir, max_workers=cfg.max_workers)
    account_summary.to_csv(cfg.output_dir / "account_summary.csv", index=False, encoding="utf-8-sig")
    for profile_name, equity_df in equity_map.items():
        equity_df.to_csv(cfg.output_dir / f"equity_{profile_name}.csv", index=False, encoding="utf-8-sig")

    signal_summary = trades.groupby("profile_name", as_index=False).agg(
        tp_pct=("tp_pct", "first"),
        trade_count=("code", "count"),
        avg_return=("return_pct_raw", "mean"),
        win_rate=("return_pct_raw", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).mean())),
        trend_count=("nearer_line", lambda s: int((s == "trend").sum())),
        long_count=("nearer_line", lambda s: int((s == "long").sum())),
    )
    signal_summary.to_csv(cfg.output_dir / "signal_summary.csv", index=False, encoding="utf-8-sig")

    best_profile = account_summary.iloc[0]["profile_name"]
    best_trades = trades[trades["profile_name"] == best_profile].copy()
    line_compare = best_trades.groupby("nearer_line", as_index=False).agg(
        trade_count=("code", "count"),
        avg_return=("return_pct_raw", "mean"),
        win_rate=("return_pct_raw", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).mean())),
    )
    line_compare.to_csv(cfg.output_dir / "best_profile_line_compare.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "walkforward": True,
            "train_dates_excluded_from_backtest": True,
            "positive_case_dir": str(POS_CASE_DIR),
            "negative_case_dir": str(NEG_CASE_DIR),
            "data_dir": str(cfg.data_dir),
            "candidate_cache_dir": str(cfg.output_dir / "target_candidate_cache_parts"),
            "use_default_exclude": bool(cfg.use_default_exclude),
            "exclude_window": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())] if cfg.use_default_exclude else None,
            "backtest_start_date": str(cfg.backtest_start_date.date()) if cfg.backtest_start_date is not None else None,
            "backtest_end_date": str(cfg.backtest_end_date.date()) if cfg.backtest_end_date is not None else None,
            "buy_rule": "signal_date_next_open",
            "stop_rule": "entry_day_low_same_day_fixed_price",
            "tp_levels": BUY_TP_LEVELS,
            "tp_exec": "next_day_open",
            "max_hold_days": MAX_HOLD_DAYS,
            "forced_exit": "hold_3_next_day_open",
            "same_day_priority": "stop_before_tp_if_ohlc_conflict",
            "ranking_rule": "commonality_score=rank(close_location)+rank(rebound_ratio)-rank(upper_shadow_pct)",
            "max_positions": MAX_POSITIONS,
            "initial_capital": INITIAL_CAPITAL,
            "allocation_mode": "equal",
            "min_lot": 100,
            "threshold_quantile": THRESHOLD_QUANTILE,
            "min_train_pos": MIN_TRAIN_POS,
            "min_train_neg": MIN_TRAIN_NEG,
        },
        "case_counts": {
            "positive_cases": int(len(pos_cases)),
            "negative_cases": int(len(neg_cases)),
            "labeled_dates": int(labeled_summary["signal_date"].nunique()),
            "backtest_dates": int(len(backtest_dates)),
        },
        "best_model": best_model,
        "best_account_profile": account_summary.iloc[0].to_dict() if not account_summary.empty else {},
        "best_profile_line_compare": line_compare.to_dict("records"),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(cfg.output_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于新增砖型图正反例做严格 walk-forward 训练与回测")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--labeled-eval-date-limit", type=int, default=0)
    parser.add_argument("--backtest-date-limit", type=int, default=0)
    parser.add_argument("--disable-default-exclude", action="store_true")
    parser.add_argument("--backtest-start-date", type=str, default="")
    parser.add_argument("--backtest-end-date", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DATA_DIR, _DAILY_STEM_MAP, GLOBAL_CFG
    DATA_DIR = Path(args.data_dir)
    _DAILY_STEM_MAP = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULT_ROOT / f"brick_case_rank_walkforward_counterexample_backtest_v1_{args.mode}_{timestamp}"

    if args.mode == "smoke":
        labeled_eval_date_limit = args.labeled_eval_date_limit or 18
        backtest_date_limit = args.backtest_date_limit or 40
        max_workers = max(1, min(args.max_workers, 10))
    else:
        labeled_eval_date_limit = args.labeled_eval_date_limit or 0
        backtest_date_limit = args.backtest_date_limit or 0
        max_workers = max(1, min(args.max_workers, 10))

    backtest_start_date = pd.Timestamp(args.backtest_start_date) if args.backtest_start_date else None
    backtest_end_date = pd.Timestamp(args.backtest_end_date) if args.backtest_end_date else None

    cfg = WalkforwardConfig(
        mode=args.mode,
        output_dir=output_dir,
        data_dir=DATA_DIR,
        max_workers=max_workers,
        labeled_eval_date_limit=labeled_eval_date_limit,
        backtest_date_limit=backtest_date_limit,
        use_default_exclude=not bool(args.disable_default_exclude),
        backtest_start_date=backtest_start_date,
        backtest_end_date=backtest_end_date,
    )
    GLOBAL_CFG = cfg
    try:
        run_experiment(cfg)
    except Exception as exc:  # noqa: BLE001
        output_dir.mkdir(parents=True, exist_ok=True)
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
