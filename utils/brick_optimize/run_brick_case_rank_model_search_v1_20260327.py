from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
DATA_DIR = ROOT / "data" / "20260324"
CASE_RECALL_PATH = ROOT / "utils" / "brick_optimize" / "brickfilter_case_recall_v1_20260327.py"
CASE_FIRST_PATH = ROOT / "utils" / "brick_optimize" / "brickfilter_case_first_v1_20260326.py"
CASE_SEMANTICS_PATH = ROOT / "utils" / "brick_optimize" / "brick_case_semantics_v1_20260326.py"
BASELINE_CASE_RECALL_SUMMARY = RESULT_ROOT / "brick_case_recall_perfect_case_coverage_v1_20260327_full_r3" / "summary.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from xgboost import XGBClassifier

    XGB_OK = True
except Exception:
    XGB_OK = False
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier

    LGBM_OK = True
except Exception:
    LGBM_OK = False
    LGBMClassifier = None


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


case_recall = load_module(CASE_RECALL_PATH, "brick_case_rank_case_recall")
case_first = load_module(CASE_FIRST_PATH, "brick_case_rank_case_first")
case_semantics = load_module(CASE_SEMANTICS_PATH, "brick_case_rank_case_semantics")

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
TOPN = 20
ALL_TOPNS = [20, 50, 100]

FEATURE_COLS = [
    "same_type_case_sim_score",
    "perfect_case_sim_score",
    "perfect_case_quality_score",
    "recall_score",
    "brick_case_type_score",
    "early_red_stage_flag_num",
    "risk_distribution_recent_20",
    "risk_distribution_recent_30",
    "risk_distribution_recent_60",
    "rebound_ratio",
    "close_location",
    "body_ratio",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "prev_green_streak",
    "prev_red_streak",
    "signal_ret",
    "signal_vs_ma5_proxy",
    "trend_spread",
    "close_to_trend",
    "close_to_long",
    "RSI14",
    "MACD_hist",
    "KDJ_J",
    "pattern_a_relaxed",
    "pattern_b_relaxed",
]

MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "heuristic": [{}],
    "logreg": [
        {"C": 0.5, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 2.0, "class_weight": "balanced"},
    ],
    "rf": [
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 10, "class_weight": "balanced_subsample"},
        {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 10, "class_weight": "balanced_subsample"},
        {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 5, "class_weight": "balanced_subsample"},
    ],
}
if XGB_OK:
    MODEL_GRIDS["xgb"] = [
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 3},
    ]
if LGBM_OK:
    MODEL_GRIDS["lgbm"] = [
        {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 20},
        {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 20},
        {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 10},
    ]


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


def _load_perfect_cases() -> pd.DataFrame:
    df = case_semantics.load_case_day_features(case_semantics.PERFECT_CASE_DIR, DATA_DIR)
    if df.empty:
        return df
    df = df[
        (pd.to_datetime(df["signal_date"]) < case_first.relaxed.EXCLUDE_START)
        | (pd.to_datetime(df["signal_date"]) > case_first.relaxed.EXCLUDE_END)
    ].copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["case_key"] = df["code"].astype(str) + "|" + df["signal_date"].dt.strftime("%Y-%m-%d")
    return df.drop_duplicates(["code", "signal_date"]).reset_index(drop=True)


def _select_dates(perfect_df: pd.DataFrame, date_limit: int) -> list[pd.Timestamp]:
    dates = sorted(pd.to_datetime(perfect_df["signal_date"]).drop_duplicates().tolist())
    if date_limit > 0:
        return dates[:date_limit]
    return dates


def _build_dataset(dates: list[pd.Timestamp], result_dir: Path, max_workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    update_progress(result_dir, "building_candidate_cache", total_dates=len(dates), max_workers=max_workers)
    cache = case_first.build_candidate_cache_for_dates(
        DATA_DIR,
        dates,
        max_workers=max_workers,
        required_lens=case_recall.CASE_SEQ_LENS,
    )
    perfect_df = _load_perfect_cases()
    case_map = {
        (case_semantics.code_key(r.code), pd.Timestamp(r.signal_date).strftime("%Y-%m-%d")): r
        for r in perfect_df.itertuples(index=False)
    }
    rows: list[pd.DataFrame] = []
    total = len(dates)
    for idx, date_ts in enumerate(dates, start=1):
        cand_df = cache[cache["signal_date"] == pd.Timestamp(date_ts)].copy().reset_index(drop=True)
        if cand_df.empty:
            continue
        enriched = case_recall.enrich_candidates_for_date(pd.Timestamp(date_ts), cand_df, DATA_DIR)
        if enriched.empty:
            continue
        enriched = enriched.copy()
        date_key = pd.Timestamp(date_ts).strftime("%Y-%m-%d")
        enriched["label"] = enriched.apply(lambda r: int((case_semantics.code_key(r["code"]), date_key) in case_map), axis=1)
        enriched["case_key"] = enriched["code"].map(case_semantics.code_key) + "|" + date_key
        rows.append(enriched)
        if idx == 1 or idx % 5 == 0 or idx == total:
            update_progress(result_dir, "enriching_candidates", done_dates=idx, total_dates=total, latest_date=date_key)
    if not rows:
        raise RuntimeError("未构建出任何完美案例排序样本")
    dataset = pd.concat(rows, ignore_index=True)
    dataset["signal_date"] = pd.to_datetime(dataset["signal_date"])
    return dataset, perfect_df


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for col in FEATURE_COLS:
        if col not in x.columns:
            x[col] = 0.0
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    return x


def _split_dates(dates: list[pd.Timestamp]) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    if len(dates) < 6:
        cut = max(1, len(dates) // 2)
        return dates[:cut], dates[cut:]
    cut = max(3, int(math.floor(len(dates) * 0.7)))
    return dates[:cut], dates[cut:]


def _build_model(model_name: str, params: dict[str, Any]):
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=float(params["C"]),
            class_weight=params.get("class_weight"),
            random_state=42,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight=params.get("class_weight"),
            random_state=42,
            n_jobs=1,
        )
    if model_name == "xgb":
        if not XGB_OK:
            raise RuntimeError("xgboost 不可用")
        return XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            min_child_weight=float(params["min_child_weight"]),
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=1,
        )
    if model_name == "lgbm":
        if not LGBM_OK:
            raise RuntimeError("lightgbm 不可用")
        return LGBMClassifier(
            n_estimators=int(params["n_estimators"]),
            num_leaves=int(params["num_leaves"]),
            learning_rate=float(params["learning_rate"]),
            min_child_samples=int(params["min_child_samples"]),
            objective="binary",
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
    raise ValueError(f"未知模型: {model_name}")


def _serialize_params(params: dict[str, Any]) -> str:
    if not params:
        return "heuristic"
    return "|".join(f"{k}={params[k]}" for k in sorted(params))


def _score_model(model_name: str, params: dict[str, Any], train_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.Series:
    score_df = _prepare_features(score_df)
    if model_name == "heuristic":
        return pd.to_numeric(score_df["recall_score"], errors="coerce").fillna(0.0)
    train_df = _prepare_features(train_df)
    x_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df["label"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        # 小样本日期下允许退回到案例召回启发式，避免单类训练直接中断整轮实验。
        return pd.to_numeric(score_df["recall_score"], errors="coerce").fillna(0.0)
    model = _build_model(model_name, params)
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(score_df[FEATURE_COLS].to_numpy(dtype=float))[:, 1], index=score_df.index, dtype=float)
    return pd.Series(model.predict(score_df[FEATURE_COLS].to_numpy(dtype=float)), index=score_df.index, dtype=float)


def _select_topn_per_date(scored_df: pd.DataFrame, topn: int) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df
    daily_dir = case_first._resolve_daily_dir(DATA_DIR)
    picked_parts: list[pd.DataFrame] = []
    for date_ts, date_df in scored_df.groupby("signal_date", sort=True):
        quotas = case_recall._build_type_quota_map(int(topn), pd.Timestamp(date_ts), daily_dir)
        chosen_parts: list[pd.DataFrame] = []
        chosen_keys: set[tuple[str, str]] = set()
        for case_type, quota in quotas.items():
            if quota <= 0:
                continue
            sub = date_df[pd.to_numeric(date_df["brick_case_type"], errors="coerce").fillna(4).astype(int) == int(case_type)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(
                ["model_score", "same_type_case_sim_score", "perfect_case_sim_score", "perfect_case_quality_score", "code"],
                ascending=[False, False, False, False, True],
                kind="mergesort",
            ).head(int(quota))
            chosen_parts.append(sub)
            chosen_keys.update((str(r.code), pd.Timestamp(r.signal_date).strftime("%Y-%m-%d")) for r in sub.itertuples(index=False))
        remain_n = max(0, int(topn) - sum(len(x) for x in chosen_parts))
        if remain_n > 0:
            remainder = date_df[
                ~date_df.apply(lambda r: (str(r["code"]), pd.Timestamp(r["signal_date"]).strftime("%Y-%m-%d")) in chosen_keys, axis=1)
            ].copy()
            if not remainder.empty:
                remainder = remainder.sort_values(
                    ["model_score", "same_type_case_sim_score", "perfect_case_sim_score", "perfect_case_quality_score", "code"],
                    ascending=[False, False, False, False, True],
                    kind="mergesort",
                ).head(remain_n)
                chosen_parts.append(remainder)
        if chosen_parts:
            picked_parts.append(pd.concat(chosen_parts, ignore_index=True))
    if not picked_parts:
        return pd.DataFrame()
    out = pd.concat(picked_parts, ignore_index=True)
    return out.sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True]).reset_index(drop=True)


def _evaluate_case_coverage(scored_df: pd.DataFrame, perfect_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    selected_map: dict[int, set[str]] = {}
    top_tables: dict[int, pd.DataFrame] = {}
    for topn in ALL_TOPNS:
        table = _select_topn_per_date(scored_df, topn)
        top_tables[topn] = table
        selected_map[topn] = set(table["code"].map(case_semantics.code_key) + "|" + pd.to_datetime(table["signal_date"]).dt.strftime("%Y-%m-%d"))

    candidate_pool_keys = set(scored_df["code"].map(case_semantics.code_key) + "|" + scored_df["signal_date"].dt.strftime("%Y-%m-%d"))

    result_rows = []
    for row in perfect_df.itertuples(index=False):
        case_key = case_semantics.code_key(row.code) + "|" + pd.Timestamp(row.signal_date).strftime("%Y-%m-%d")
        in20 = case_key in selected_map[20]
        in50 = case_key in selected_map[50]
        in100 = case_key in selected_map[100]
        if in20:
            stage = "selected"
        elif case_key in candidate_pool_keys:
            stage = "candidate_pool"
        else:
            stage = "missed"
        result_rows.append(
            {
                "stock_name": row.stock_name,
                "code": row.code,
                "signal_date": pd.Timestamp(row.signal_date),
                "brick_case_type_name": row.brick_case_type_name,
                "stage": stage,
                "in_top20": in20,
                "in_top50": in50,
                "in_top100": in100,
            }
        )
    result_df = pd.DataFrame(result_rows).sort_values(["signal_date", "code"]).reset_index(drop=True)

    rank_df = scored_df.sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True]).copy()
    rank_df["daily_rank"] = rank_df.groupby("signal_date").cumcount() + 1
    case_rank_map = {
        case_semantics.code_key(r.code) + "|" + pd.Timestamp(r.signal_date).strftime("%Y-%m-%d"): int(r.daily_rank)
        for r in rank_df.itertuples(index=False)
    }
    reciprocal_ranks = []
    for row in perfect_df.itertuples(index=False):
        case_key = case_semantics.code_key(row.code) + "|" + pd.Timestamp(row.signal_date).strftime("%Y-%m-%d")
        rank = case_rank_map.get(case_key)
        reciprocal_ranks.append(0.0 if rank is None or rank <= 0 else 1.0 / float(rank))

    summary = {
        "total_cases": int(len(perfect_df)),
        "selected_count": int(result_df["in_top20"].sum()),
        "candidate_pool_count": int((result_df["stage"] == "candidate_pool").sum()),
        "missed_count": int((result_df["stage"] == "missed").sum()),
        "top20_count": int(result_df["in_top20"].sum()),
        "top50_count": int(result_df["in_top50"].sum()),
        "top100_count": int(result_df["in_top100"].sum()),
        "recall_at_20": float(result_df["in_top20"].mean()) if not result_df.empty else 0.0,
        "recall_at_50": float(result_df["in_top50"].mean()) if not result_df.empty else 0.0,
        "recall_at_100": float(result_df["in_top100"].mean()) if not result_df.empty else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }
    return summary, result_df, top_tables[20]


def _compare_to_case_recall(summary: dict[str, Any]) -> dict[str, Any]:
    if not BASELINE_CASE_RECALL_SUMMARY.exists():
        return {}
    baseline = json.loads(BASELINE_CASE_RECALL_SUMMARY.read_text(encoding="utf-8"))
    return {
        "selected_diff": int(summary["selected_count"]) - int(baseline.get("selected_count", 0)),
        "top20_diff": int(summary["top20_count"]) - int(baseline.get("top20_count", 0)),
        "top50_diff": int(summary["top50_count"]) - int(baseline.get("top50_count", 0)),
        "top100_diff": int(summary["top100_count"]) - int(baseline.get("top100_count", 0)),
        "recall20_diff": float(summary["recall_at_20"]) - float(int(baseline.get("top20_count", 0)) / max(int(baseline.get("total_cases", 1)), 1)),
    }


def run_search(result_dir: Path, date_limit: int, max_workers: int, reuse_dataset_csv: str = "") -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    if reuse_dataset_csv:
        dataset = pd.read_csv(reuse_dataset_csv)
        dataset["signal_date"] = pd.to_datetime(dataset["signal_date"])
        dataset = _prepare_features(dataset)
        perfect_df = _load_perfect_cases()
        dataset_dates = set(pd.to_datetime(dataset["signal_date"]).dt.strftime("%Y-%m-%d"))
        perfect_df = perfect_df[perfect_df["signal_date"].dt.strftime("%Y-%m-%d").isin(dataset_dates)].copy().reset_index(drop=True)
        if dataset.empty or perfect_df.empty:
            raise RuntimeError("复用的数据集为空，无法继续完美案例排序模型搜索")
        case_map = {
            (case_semantics.code_key(r.code), pd.Timestamp(r.signal_date).strftime("%Y-%m-%d")): r
            for r in perfect_df.itertuples(index=False)
        }
        dataset["label"] = dataset.apply(
            lambda r: int((case_semantics.code_key(r["code"]), pd.Timestamp(r["signal_date"]).strftime("%Y-%m-%d")) in case_map),
            axis=1,
        )
        dataset["case_key"] = dataset["code"].map(case_semantics.code_key) + "|" + dataset["signal_date"].dt.strftime("%Y-%m-%d")
        update_progress(
            result_dir,
            "dataset_reused",
            total_rows=int(len(dataset)),
            total_dates=int(dataset["signal_date"].nunique()),
            reuse_dataset_csv=str(reuse_dataset_csv),
        )
        dataset.to_csv(result_dir / "candidate_dataset.csv", index=False, encoding="utf-8-sig")
    else:
        perfect_df = _load_perfect_cases()
        dates = _select_dates(perfect_df, date_limit=date_limit)
        if not dates:
            raise RuntimeError("没有可用完美案例日期")
        dataset, perfect_df = _build_dataset(dates, result_dir=result_dir, max_workers=max_workers)
        dataset = _prepare_features(dataset)
        dataset.to_csv(result_dir / "candidate_dataset.csv", index=False, encoding="utf-8-sig")

    unique_dates = sorted(pd.to_datetime(dataset["signal_date"]).drop_duplicates().tolist())
    train_dates, val_dates = _split_dates(unique_dates)
    train_df = dataset[dataset["signal_date"].isin(train_dates)].copy()
    val_df = dataset[dataset["signal_date"].isin(val_dates)].copy()
    if train_df.empty or val_df.empty:
        raise RuntimeError("训练或验证样本为空，无法搜索完美案例排序模型")
    update_progress(result_dir, "dataset_ready", total_rows=int(len(dataset)), train_rows=int(len(train_df)), val_rows=int(len(val_df)))

    model_rows: list[dict[str, Any]] = []
    best_by_model: dict[str, dict[str, Any]] = {}
    for model_name, grid in MODEL_GRIDS.items():
        best_row: dict[str, Any] | None = None
        for params in grid:
            scored_val = val_df.copy()
            scored_val["model_score"] = _score_model(model_name, params, train_df, scored_val)
            summary, _, _ = _evaluate_case_coverage(scored_val, perfect_df[perfect_df["signal_date"].isin(val_dates)].copy())
            row = {
                "model_name": model_name,
                "params": _serialize_params(params),
                **summary,
            }
            model_rows.append(row)
            if best_row is None or (
                row["recall_at_20"],
                row["mrr"],
                row["recall_at_50"],
                row["recall_at_100"],
            ) > (
                best_row["recall_at_20"],
                best_row["mrr"],
                best_row["recall_at_50"],
                best_row["recall_at_100"],
            ):
                best_row = row | {"raw_params": params}
        if best_row is not None:
            best_by_model[model_name] = best_row

    validation_df = pd.DataFrame(model_rows).sort_values(
        ["recall_at_20", "mrr", "recall_at_50", "recall_at_100"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    validation_df.to_csv(result_dir / "model_validation_summary.csv", index=False, encoding="utf-8-sig")
    best_cfg_df = pd.DataFrame(
        [
            {
                "model_name": name,
                "params": row["params"],
                "recall_at_20": row["recall_at_20"],
                "mrr": row["mrr"],
                "recall_at_50": row["recall_at_50"],
                "recall_at_100": row["recall_at_100"],
            }
            for name, row in best_by_model.items()
        ]
    ).sort_values(["recall_at_20", "mrr"], ascending=[False, False]).reset_index(drop=True)
    best_cfg_df.to_csv(result_dir / "best_config_by_model.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "validation_ready", model_count=int(len(best_cfg_df)))

    full_rows: list[dict[str, Any]] = []
    best_full_name: str | None = None
    best_full_summary: dict[str, Any] | None = None
    best_full_candidates: pd.DataFrame | None = None
    best_full_case_results: pd.DataFrame | None = None
    for model_name, row in best_by_model.items():
        scored_all = dataset.copy()
        scored_all["model_score"] = _score_model(model_name, row["raw_params"], dataset, scored_all)
        summary, case_results_df, top20_df = _evaluate_case_coverage(scored_all, perfect_df.copy())
        full_rows.append({"model_name": model_name, "params": row["params"], **summary})
        if best_full_summary is None or (
            summary["recall_at_20"],
            summary["mrr"],
            summary["recall_at_50"],
            summary["recall_at_100"],
        ) > (
            best_full_summary["recall_at_20"],
            best_full_summary["mrr"],
            best_full_summary["recall_at_50"],
            best_full_summary["recall_at_100"],
        ):
            best_full_name = model_name
            best_full_summary = summary
            best_full_candidates = top20_df.copy()
            best_full_case_results = case_results_df.copy()

    full_df = pd.DataFrame(full_rows).sort_values(
        ["recall_at_20", "mrr", "recall_at_50", "recall_at_100"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    full_df.to_csv(result_dir / "model_full_coverage_summary.csv", index=False, encoding="utf-8-sig")
    if best_full_candidates is None or best_full_case_results is None or best_full_name is None or best_full_summary is None:
        raise RuntimeError("未找到任何有效的完美案例排序模型")

    best_full_candidates = best_full_candidates.sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    best_full_candidates["sort_score"] = pd.to_numeric(best_full_candidates["model_score"], errors="coerce").fillna(0.0)
    best_full_candidates["strategy_key"] = f"case_rank|{best_full_name}"
    best_full_candidates.to_csv(result_dir / "best_model_top20_candidates.csv", index=False, encoding="utf-8-sig")
    best_full_case_results.to_csv(result_dir / "best_model_case_results.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "goal": "perfect_case_ranking_model",
            "topn_target": TOPN,
            "coverage_definition": "累计 top20 code+signal_date 命中率",
            "data_dir": str(DATA_DIR),
            "feature_cols": FEATURE_COLS,
        },
        "best_model_name": best_full_name,
        "best_model_params": best_by_model[best_full_name]["raw_params"],
        "best_model_summary": best_full_summary,
        "best_model_vs_case_recall": _compare_to_case_recall(best_full_summary),
        "validation_best_by_model": {
            name: {
                "params": row["params"],
                "recall_at_20": row["recall_at_20"],
                "mrr": row["mrr"],
                "recall_at_50": row["recall_at_50"],
                "recall_at_100": row["recall_at_100"],
            }
            for name, row in best_by_model.items()
        },
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK 完美案例排序模型搜索")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--date-limit", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--reuse-dataset-csv", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_case_rank_model_search_v1_{args.mode}_{timestamp}"
    date_limit = int(args.date_limit)
    if args.mode == "full":
        date_limit = 0
    try:
        run_search(
            result_dir=output_dir,
            date_limit=date_limit,
            max_workers=int(args.max_workers),
            reuse_dataset_csv=str(args.reuse_dataset_csv or ""),
        )
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
