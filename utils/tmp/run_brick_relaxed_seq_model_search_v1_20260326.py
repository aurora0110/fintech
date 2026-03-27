from __future__ import annotations

import argparse
import json
import math
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
from pandas.tseries.offsets import DateOffset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
RELAXED_RESULT_DIR = RESULT_ROOT / "brick_comprehensive_lab_full_20260325_r1"
BUYPOINT_COMPARE_PATH = ROOT / "utils" / "tmp" / "run_brick_buypoint_real_account_compare_v1_20260326.py"
CASE_SEMANTICS_PATH = ROOT / "utils" / "tmp" / "brick_case_semantics_v1_20260326.py"
DATA_SNAPSHOT_DIR = ROOT / "data" / "20260324"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brickfilter_relaxed_fusion as relaxed

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


compare_base = load_module(BUYPOINT_COMPARE_PATH, "brick_relaxed_seq_model_compare_base")
real_account = compare_base.real_account
case_semantics = load_module(CASE_SEMANTICS_PATH, "brick_relaxed_seq_case_semantics")

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))
TRAIN_MONTHS = 24
VAL_MONTHS = 6
TEST_MONTHS = 6
SMOKE_TRAIN_MONTHS = 12
SMOKE_VAL_MONTHS = 3
SMOKE_TEST_MONTHS = 3
DEFAULT_SEQ_LENS = [5, 8, 10, 13, 21]
REFINE_SEQ_LENS = [4, 5, 6, 7, 8]
FULL_WINDOW_CANDIDATES = [
    (24, 6, 6),
    (18, 6, 6),
    (15, 4, 4),
    (12, 3, 3),
    (11, 3, 3),
    (10, 3, 3),
    (9, 3, 3),
]

FACTOR_FEATURES = [
    "trend_spread",
    "close_to_trend",
    "close_to_long",
    "signal_vs_ma5_proxy",
    "ret1",
    "ret5",
    "brick_green_len_prev",
    "brick_red_len",
    "rebound_ratio",
    "RSI14",
    "MACD_hist",
    "body_ratio",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "green4_flag_num",
    "green4_low_flag_num",
    "red4_flag_num",
    "turn_strength_layer_num",
    "trend_layer_num",
    "brick_case_type_score",
    "early_red_stage_flag_num",
    "risk_distribution_recent_20",
    "risk_distribution_recent_30",
    "risk_distribution_recent_60",
]

BASE_MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logreg": [
        {"C": 0.2, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 5.0, "class_weight": "balanced"},
    ],
    "rf": [
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 10},
        {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 20},
    ],
}
if XGB_OK:
    BASE_MODEL_GRIDS["xgb"] = [
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 1},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 5},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.10, "min_child_weight": 1},
    ]
if LGBM_OK:
    BASE_MODEL_GRIDS["lgbm"] = [
        {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 20},
        {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
        {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.10, "min_child_samples": 10},
        {"n_estimators": 400, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 20},
    ]

REFINE_MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logreg": [
        {"C": 0.5, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 2.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 2.0, "class_weight": "balanced"},
    ],
    "rf": [
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 10},
        {"n_estimators": 400, "max_depth": 6, "min_samples_leaf": 8},
        {"n_estimators": 500, "max_depth": 6, "min_samples_leaf": 5},
        {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 10},
    ],
}
if XGB_OK:
    REFINE_MODEL_GRIDS["xgb"] = [
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
        {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 5},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.07, "min_child_weight": 5},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 3},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 7},
    ]
if LGBM_OK:
    REFINE_MODEL_GRIDS["lgbm"] = [
        {"n_estimators": 150, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 20},
        {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 20},
        {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 20},
        {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
        {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 20},
        {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.07, "min_child_samples": 20},
        {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 10},
    ]


@dataclass(frozen=True)
class RollingWindow:
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def write_error(result_dir: Path, exc: BaseException) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(result_dir / "error.json", payload)
    update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))


def normalize_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    return series.rank(method="average", pct=True).fillna(0.0)


def code_key(value: Any) -> str:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else text


def serialize_params(params: dict[str, Any]) -> str:
    return "|".join(f"{k}={params[k]}" for k in sorted(params))


def load_relaxed_raw(file_limit_codes: int) -> tuple[dict[str, Any], pd.DataFrame]:
    best = json.loads((RELAXED_RESULT_DIR / "best_config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(RELAXED_RESULT_DIR / "candidate_scored.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    code_map = compare_base.build_daily_code_map()
    df = df[df["candidate_pool"].astype(str) == str(best["candidate_pool"])].copy()
    df = df[(df["signal_date"] < relaxed.EXCLUDE_START) | (df["signal_date"] > relaxed.EXCLUDE_END)].copy()
    df["code"] = df["code"].astype(str).map(code_key).map(code_map)
    df = df.dropna(subset=["code"]).copy()
    if file_limit_codes > 0:
        keep_codes = sorted(df["code"].astype(str).unique())[:file_limit_codes]
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["sample_key"] = (
        df["code"].astype(str)
        + "|"
        + df["signal_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + pd.to_numeric(df["signal_idx"], errors="coerce").fillna(-1).astype(int).astype(str)
    )
    for col in FACTOR_FEATURES + ["pool_bonus"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return best, df.reset_index(drop=True)


def generate_windows(start_date: pd.Timestamp, end_date: pd.Timestamp, train_months: int, val_months: int, test_months: int) -> list[RollingWindow]:
    windows: list[RollingWindow] = []
    anchor = pd.Timestamp(start_date).normalize()
    idx = 1
    while True:
        train_start = anchor
        train_end = train_start + DateOffset(months=train_months) - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + DateOffset(months=val_months) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        if test_start > end_date:
            break
        test_end = test_start + DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > end_date:
            test_end = pd.Timestamp(end_date)
        windows.append(
            RollingWindow(
                idx=idx,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        idx += 1
        if test_end >= end_date:
            break
        anchor = anchor + DateOffset(months=test_months)
    return windows


def choose_window_plan(start_date: pd.Timestamp, end_date: pd.Timestamp, mode: str) -> tuple[int, int, int, list[RollingWindow], dict[str, Any]]:
    if mode != "full":
        windows = generate_windows(start_date, end_date, SMOKE_TRAIN_MONTHS, SMOKE_VAL_MONTHS, SMOKE_TEST_MONTHS)
        return (
            SMOKE_TRAIN_MONTHS,
            SMOKE_VAL_MONTHS,
            SMOKE_TEST_MONTHS,
            windows[:1],
            {
                "requested": {
                    "train_months": SMOKE_TRAIN_MONTHS,
                    "val_months": SMOKE_VAL_MONTHS,
                    "test_months": SMOKE_TEST_MONTHS,
                },
                "actual": {
                    "train_months": SMOKE_TRAIN_MONTHS,
                    "val_months": SMOKE_VAL_MONTHS,
                    "test_months": SMOKE_TEST_MONTHS,
                },
                "fallback_used": False,
            },
        )
    requested = {
        "train_months": TRAIN_MONTHS,
        "val_months": VAL_MONTHS,
        "test_months": TEST_MONTHS,
    }
    for train_months, val_months, test_months in FULL_WINDOW_CANDIDATES:
        windows = generate_windows(start_date, end_date, train_months, val_months, test_months)
        if windows:
            return (
                train_months,
                val_months,
                test_months,
                windows,
                {
                    "requested": requested,
                    "actual": {
                        "train_months": train_months,
                        "val_months": val_months,
                        "test_months": test_months,
                    },
                    "fallback_used": (train_months, val_months, test_months) != (
                        TRAIN_MONTHS,
                        VAL_MONTHS,
                        TEST_MONTHS,
                    ),
                },
            )
    raise RuntimeError("滚动窗口为空，无法搜索")


def parse_seq_lens(seq_lens_arg: str, profile: str) -> list[int]:
    if seq_lens_arg.strip():
        seq_lens = sorted({int(part.strip()) for part in seq_lens_arg.split(",") if part.strip()})
        if not seq_lens:
            raise RuntimeError("seq_lens 为空")
        return seq_lens
    if profile == "refine":
        return list(REFINE_SEQ_LENS)
    return list(DEFAULT_SEQ_LENS)


def get_model_grids(profile: str, selected_models: list[str]) -> dict[str, list[dict[str, Any]]]:
    base = REFINE_MODEL_GRIDS if profile == "refine" else BASE_MODEL_GRIDS
    grids = {name: base[name] for name in selected_models if name in base}
    if not grids:
        raise RuntimeError("没有可用的模型家族可搜索")
    return grids


def build_factor_model(train_df: pd.DataFrame) -> dict[str, Any]:
    pos = train_df[train_df["label"] == 1]
    neg = train_df[train_df["label"] == 0]
    features: dict[str, Any] = {}
    for feature in FACTOR_FEATURES:
        series = pd.to_numeric(train_df[feature], errors="coerce").fillna(0.0)
        mean = float(series.mean())
        std = float(series.std())
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        pos_mean = float(pd.to_numeric(pos[feature], errors="coerce").fillna(0.0).mean()) if not pos.empty else mean
        neg_mean = float(pd.to_numeric(neg[feature], errors="coerce").fillna(0.0).mean()) if not neg.empty else mean
        direction = 1.0 if pos_mean >= neg_mean else -1.0
        features[feature] = {"mean": mean, "std": std, "direction": direction}
    return {"features": features, "feature_order": list(FACTOR_FEATURES)}


def apply_factor_model(df: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    score = pd.Series(0.0, index=out.index, dtype=float)
    for feature in model["feature_order"]:
        state = model["features"][feature]
        series = pd.to_numeric(out[feature], errors="coerce").fillna(state["mean"])
        z = (series - state["mean"]) / state["std"]
        score = score + z * state["direction"]
    out["factor_score_raw"] = score
    out["factor_score"] = normalize_rank(score)
    return out


def ml_feature_columns() -> list[str]:
    return ["sim_score", "perfect_case_sim_score", "factor_score"] + FACTOR_FEATURES


def fit_model(model_name: str, params: dict[str, Any], train_df: pd.DataFrame):
    if train_df.empty or train_df["label"].nunique() < 2:
        return None
    X = train_df[ml_feature_columns()].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_df["label"].astype(int)
    if model_name == "logreg":
        model = LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=1000,
            solver="lbfgs",
            n_jobs=1,
            random_state=42,
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42,
            n_jobs=1,
        )
    elif model_name == "xgb" and XGB_OK:
        model = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
    elif model_name == "lgbm" and LGBM_OK:
        model = LGBMClassifier(
            n_estimators=int(params["n_estimators"]),
            num_leaves=int(params["num_leaves"]),
            learning_rate=float(params["learning_rate"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
    else:
        return None
    model.fit(X, y)
    return model


def predict_prob(df: pd.DataFrame, model) -> pd.Series:
    if model is None or df.empty:
        return pd.Series(np.nan, index=df.index, dtype=float)
    X = df[ml_feature_columns()].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index, dtype=float)


def evaluate_model(df: pd.DataFrame, raw_prob: pd.Series) -> dict[str, float]:
    if df.empty or raw_prob.empty or df["label"].nunique() < 2:
        return {"auc": float("nan"), "f1": float("nan"), "accuracy": float("nan")}
    y_true = df["label"].astype(int)
    y_prob = raw_prob.reindex(df.index).fillna(0.0)
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return {
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def build_stage_records(df: pd.DataFrame, seq_len: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        seq_map = getattr(row, "seq_map_obj")
        if seq_len not in seq_map:
            continue
        rows.append(
            {
                "code": getattr(row, "code"),
                "date": pd.Timestamp(getattr(row, "signal_date")),
                "label": int(getattr(row, "label")),
                "result": getattr(row, "result", "pending"),
                "ret": float(getattr(row, "ret", 0.0) or 0.0),
                "entry_date": pd.Timestamp(getattr(row, "entry_date")),
                "exit_date": pd.Timestamp(getattr(row, "exit_date")),
                "entry_price": float(getattr(row, "entry_price")),
                "ret1": float(getattr(row, "ret1")),
                "ret5": float(getattr(row, "ret5")),
                "ret10": float(getattr(row, "ret10")),
                "signal_ret": float(getattr(row, "signal_ret")),
                "trend_spread": float(getattr(row, "trend_spread")),
                "close_to_trend": float(getattr(row, "close_to_trend")),
                "close_to_long": float(getattr(row, "close_to_long")),
                "ma10_slope_5": float(getattr(row, "ma10_slope_5")),
                "ma20_slope_5": float(getattr(row, "ma20_slope_5")),
                "brick_red_len": float(getattr(row, "brick_red_len")),
                "brick_green_len_prev": float(getattr(row, "brick_green_len_prev")),
                "rebound_ratio": float(getattr(row, "rebound_ratio")),
                "RSI14": float(getattr(row, "RSI14")),
                "MACD_hist": float(getattr(row, "MACD_hist")),
                "KDJ_J": float(getattr(row, "KDJ_J")),
                "body_ratio": float(getattr(row, "body_ratio")),
                "upper_shadow_pct": float(getattr(row, "upper_shadow_pct")),
                "lower_shadow_pct": float(getattr(row, "lower_shadow_pct")),
                "brick_case_type_score": float(getattr(row, "brick_case_type_score", 0.45)),
                "early_red_stage_flag_num": float(getattr(row, "early_red_stage_flag_num", 0.0)),
                "risk_distribution_recent_20": float(getattr(row, "risk_distribution_recent_20", 0.0)),
                "risk_distribution_recent_30": float(getattr(row, "risk_distribution_recent_30", 0.0)),
                "risk_distribution_recent_60": float(getattr(row, "risk_distribution_recent_60", 0.0)),
                "seq_map": {seq_len: seq_map[seq_len]},
            }
        )
    return rows


def score_similarity(train_df: pd.DataFrame, target_df: pd.DataFrame, seq_len: int, best: dict[str, Any]) -> tuple[pd.DataFrame, list[np.ndarray]]:
    train_records = build_stage_records(train_df, seq_len)
    target_records = build_stage_records(target_df, seq_len)
    success_records = [r for r in train_records if int(r["label"]) == 1]
    if not success_records or not target_records:
        out = target_df.copy()
        out["sim_score"] = np.nan
        return out, []
    templates = relaxed.sim.build_templates(success_records, seq_len, str(best["rep"]), str(best["builder"]))
    cfg = relaxed.sim.BaseConfig(
        builder=str(best["builder"]),
        seq_len=int(seq_len),
        rep=str(best["rep"]),
        scorer=str(best["scorer"]),
    )
    sim_df = relaxed.sim.build_scored_df_normal(target_records, templates, cfg).rename(columns={"date": "signal_date", "score": "sim_score"})
    out = target_df.drop(columns=["sim_score"], errors="ignore").merge(
        sim_df[["code", "signal_date", "sim_score"]],
        on=["code", "signal_date"],
        how="left",
    )
    out["sim_score"] = pd.to_numeric(out["sim_score"], errors="coerce").fillna(-1.0)
    perfect_templates = relaxed._build_perfect_case_templates(
        str(DATA_SNAPSHOT_DIR),
        pd.Timestamp(train_df["signal_date"].max()) + pd.Timedelta(days=1),
        seq_len,
        str(best["rep"]),
    )
    if perfect_templates:
        perfect_sim_df = relaxed.sim.build_scored_df_normal(target_records, perfect_templates, cfg).rename(
            columns={"date": "signal_date", "score": "perfect_case_sim_score"}
        )
        out = out.drop(columns=["perfect_case_sim_score"], errors="ignore").merge(
            perfect_sim_df[["code", "signal_date", "perfect_case_sim_score"]],
            on=["code", "signal_date"],
            how="left",
        )
    out["perfect_case_sim_score"] = pd.to_numeric(out.get("perfect_case_sim_score"), errors="coerce").fillna(-1.0)
    return out, templates


def apply_rank_and_select(df: pd.DataFrame, best: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["ml_score"] = normalize_rank(pd.to_numeric(out["ml_score_raw"], errors="coerce").fillna(0.0))
    base_rank_score = (
        normalize_rank(out["sim_score"]) * float(best["sim_weight"])
        + pd.to_numeric(out["factor_score"], errors="coerce").fillna(0.0) * float(best["factor_weight"])
        + pd.to_numeric(out["ml_score"], errors="coerce").fillna(0.0) * float(best["ml_weight"])
        + pd.to_numeric(out["pool_bonus"], errors="coerce").fillna(0.0)
    )
    out["perfect_case_rank"] = normalize_rank(pd.to_numeric(out["perfect_case_sim_score"], errors="coerce").fillna(-1.0))
    out["rank_score"] = (
        (1.0 - relaxed.PERFECT_CASE_WEIGHT) * base_rank_score
        + relaxed.PERFECT_CASE_WEIGHT * out["perfect_case_rank"]
    )
    gated = out[out["sim_score"] >= float(best["sim_gate"])].copy()
    if gated.empty:
        return gated
    return (
        gated.sort_values(
            ["signal_date", "perfect_case_rank", "rank_score", "code"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .groupby("signal_date", group_keys=False)
        .head(int(best["daily_topn"]))
        .reset_index(drop=True)
    )


def build_close_map_for_codes(codes: list[str]) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    market_dates, close_map = real_account.build_close_map(codes)
    return market_dates, close_map


def simulate_account_for_subset(
    trade_universe: pd.DataFrame,
    selected_keys: pd.Series,
    market_dates: pd.DatetimeIndex,
    close_map: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trades = trade_universe[trade_universe["sample_key"].isin(set(selected_keys.astype(str)))].copy()
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "annual_return": float("nan"),
            "holding_return": float("nan"),
            "avg_trade_return": float("nan"),
            "success_rate": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe": float("nan"),
            "calmar": float("nan"),
            "final_equity": float("nan"),
            "trade_count": 0,
            "max_consecutive_failures": float("nan"),
            "profit_factor": float("nan"),
            "avg_profit_return": float("nan"),
            "avg_loss_return": float("nan"),
            "payoff_ratio": float("nan"),
        }
    start = pd.Timestamp(trades["entry_date"].min())
    end = pd.Timestamp(trades["exit_date"].max())
    dates = market_dates[(market_dates >= start) & (market_dates <= end)]
    subset_close = {code: series.reindex(dates).ffill() for code, series in close_map.items() if code in trades["code"].astype(str).unique()}
    return real_account.simulate_real_account(trades, subset_close, dates, real_account.AccountConfig())


def _load_seq_maps_for_code(args: tuple[str, list[str], str, list[int]]) -> list[dict[str, Any]]:
    code, signal_dates, data_dir, seq_lens = args
    file_path = Path(data_dir) / f"{code}.txt"
    if not file_path.exists():
        return []
    raw_df = relaxed.sim.load_stock_data(str(file_path))
    if raw_df is None or raw_df.empty:
        return []
    feat_df = relaxed.sim.compute_relaxed_brick_features(raw_df).reset_index(drop=True)
    if feat_df.empty:
        return []
    feat_df["signal_vs_ma5_proxy"] = relaxed._compute_signal_vs_ma5_proxy(feat_df)
    date_to_idx = {pd.Timestamp(d): idx for idx, d in enumerate(raw_df["date"])}
    max_len = max(seq_lens)
    risk_profile = case_semantics.build_risk_profile(DATA_SNAPSHOT_DIR)
    rows: list[dict[str, Any]] = []
    for ds in sorted(set(signal_dates)):
        target = pd.Timestamp(ds)
        idx = date_to_idx.get(target)
        if idx is None or idx < max_len - 1:
            continue
        feat_row = feat_df.iloc[idx]
        seq_map: dict[int, dict[str, np.ndarray]] = {}
        for seq_len in seq_lens:
            if idx >= seq_len - 1:
                seq_map[seq_len] = relaxed.sim.extract_sequence(raw_df.iloc[idx - seq_len + 1 : idx + 1], seq_len)
        extra = case_semantics.enrich_case_type_and_risk_from_values(
            prev_green_streak=float(feat_row.get("prev_green_streak", 0.0) or 0.0),
            prev_red_streak=float(feat_row.get("prev_red_streak", 0.0) or 0.0),
            rebound_ratio=float(feat_row.get("rebound_ratio", 0.0) or 0.0),
            signal_ret=float(feat_row.get("signal_ret", 0.0) or 0.0),
            upper_shadow_pct=float(feat_row.get("upper_shadow_pct", 0.0) or 0.0),
            body_ratio=float(feat_row.get("body_ratio", 0.0) or 0.0),
            close_to_trend=float(feat_row.get("close_to_trend", 0.0) or 0.0),
            close_to_long=float(feat_row.get("close_to_long", 0.0) or 0.0),
            feature_df=feat_df,
            signal_idx=idx,
            risk_profile=risk_profile,
        )
        rows.append({"code": code, "signal_date": target, "seq_map_obj": seq_map, **extra})
    return rows


def attach_seq_maps(df: pd.DataFrame, seq_lens: list[int], data_dir: Path, max_workers: int, result_dir: Path) -> pd.DataFrame:
    grouped = (
        df[["code", "signal_date"]]
        .drop_duplicates()
        .assign(signal_date=lambda x: x["signal_date"].dt.strftime("%Y-%m-%d"))
        .groupby("code")["signal_date"]
        .apply(list)
        .to_dict()
    )
    tasks = [(code, dates, str(data_dir), seq_lens) for code, dates in grouped.items()]
    rows: list[dict[str, Any]] = []
    total = len(tasks)
    if max_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            rows.extend(_load_seq_maps_for_code(task))
            if idx == 1 or idx % 100 == 0 or idx == total:
                update_progress(result_dir, "building_seq_maps", done_codes=idx, total_codes=total, fallback="serial")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_seq_maps_for_code, task): task[0] for task in tasks}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                rows.extend(future.result())
                if completed == 1 or completed % 100 == 0 or completed == total:
                    update_progress(result_dir, "building_seq_maps", done_codes=completed, total_codes=total)
    seq_df = pd.DataFrame(rows)
    if seq_df.empty:
        raise RuntimeError("未能为 relaxed 候选构建任何序列")
    seq_df["signal_date"] = pd.to_datetime(seq_df["signal_date"])
    out = df.merge(seq_df, on=["code", "signal_date"], how="left")
    out = out.dropna(subset=["seq_map_obj"]).copy()
    return out.reset_index(drop=True)


def precompute_trade_universe(relaxed_df: pd.DataFrame, result_dir: Path, max_workers: int) -> pd.DataFrame:
    candidates = relaxed_df[
        ["code", "signal_idx", "signal_date", "entry_date", "entry_price", "signal_low", "sample_key"]
    ].copy()
    candidates["rank_score"] = 0.0
    candidates["strategy_key"] = "relaxed_universe"
    trades = compare_base.build_relaxed_trades(candidates, result_dir, max_workers=max_workers)
    if trades.empty:
        raise RuntimeError("固定冠军卖法下 relaxed 候选全量交易预计算为空")
    trades["sample_key"] = (
        trades["code"].astype(str)
        + "|"
        + pd.to_datetime(trades["signal_date"]).dt.strftime("%Y-%m-%d")
        + "|"
        + pd.to_numeric(trades["signal_idx"], errors="coerce").fillna(-1).astype(int).astype(str)
    )
    return trades.sort_values(["signal_date", "code", "signal_idx"]).reset_index(drop=True)


def choose_best_config(search_df: pd.DataFrame) -> dict[str, Any]:
    ranked = (
        search_df.groupby(["seq_len", "model_name", "param_key"], as_index=False)
        .agg(
            mean_val_auc=("val_auc", "mean"),
            mean_val_f1=("val_f1", "mean"),
            mean_val_accuracy=("val_accuracy", "mean"),
        )
        .sort_values(
            ["mean_val_auc", "mean_val_f1", "mean_val_accuracy", "seq_len", "model_name", "param_key"],
            ascending=[False, False, False, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if ranked.empty:
        raise RuntimeError("模型搜索结果为空")
    return ranked.iloc[0].to_dict()


def choose_best_configs_by_model(search_df: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        search_df.groupby(["model_name", "seq_len", "param_key"], as_index=False)
        .agg(
            mean_val_auc=("val_auc", "mean"),
            mean_val_f1=("val_f1", "mean"),
            mean_val_accuracy=("val_accuracy", "mean"),
        )
        .sort_values(
            ["model_name", "mean_val_auc", "mean_val_f1", "mean_val_accuracy", "seq_len", "param_key"],
            ascending=[True, False, False, False, True, True],
            kind="mergesort",
        )
    )
    return ranked.groupby("model_name", group_keys=False).head(1).reset_index(drop=True)


def run(
    mode: str,
    output_dir: Path,
    file_limit_codes: int,
    max_workers: int,
    seq_lens: list[int],
    model_names: list[str],
    grid_profile: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_grids = get_model_grids(grid_profile, model_names)
    update_progress(
        output_dir,
        "loading_inputs",
        mode=mode,
        file_limit_codes=file_limit_codes,
        max_workers=max_workers,
        seq_lens=seq_lens,
        model_names=model_names,
        grid_profile=grid_profile,
    )

    best, relaxed_raw = load_relaxed_raw(file_limit_codes)
    formal_strategy_key, formal_trades_all = compare_base.load_formal_champion(file_limit_codes)
    formal_trades_all.to_csv(output_dir / "formal_source_trades.csv", index=False)

    relaxed_raw = attach_seq_maps(relaxed_raw, seq_lens, DATA_SNAPSHOT_DIR, max_workers, output_dir)
    relaxed_raw.to_pickle(output_dir / "relaxed_raw_with_seq.pkl")

    update_progress(output_dir, "precomputing_trade_universe", candidate_count=int(len(relaxed_raw)))
    trade_universe = precompute_trade_universe(relaxed_raw, output_dir, max_workers)
    trade_universe.to_csv(output_dir / "relaxed_trade_universe.csv", index=False)

    common_start = max(pd.Timestamp(relaxed_raw["signal_date"].min()), pd.Timestamp(formal_trades_all["signal_date"].min()))
    common_end = min(pd.Timestamp(relaxed_raw["signal_date"].max()), pd.Timestamp(formal_trades_all["signal_date"].max()))
    relaxed_raw = relaxed_raw[(relaxed_raw["signal_date"] >= common_start) & (relaxed_raw["signal_date"] <= common_end)].copy()
    formal_trades_all = formal_trades_all[(formal_trades_all["signal_date"] >= common_start) & (formal_trades_all["signal_date"] <= common_end)].copy()
    trade_universe = trade_universe[(trade_universe["signal_date"] >= common_start) & (trade_universe["signal_date"] <= common_end)].copy()

    train_months, val_months, test_months, windows, window_meta = choose_window_plan(common_start, common_end, mode)
    write_json(
        output_dir / "window_plan.json",
        {
            "common_start": common_start.strftime("%Y-%m-%d"),
            "common_end": common_end.strftime("%Y-%m-%d"),
            "train_months": train_months,
            "val_months": val_months,
            "test_months": test_months,
            "requested_window_plan": window_meta["requested"],
            "fallback_used": window_meta["fallback_used"],
            "window_count": len(windows),
            "windows": [window.__dict__ for window in windows],
            "model_families": list(model_grids.keys()),
            "seq_lens": seq_lens,
            "grid_profile": grid_profile,
            "base_config": best,
            "formal_strategy_key": formal_strategy_key,
        },
    )

    all_codes = sorted(set(formal_trades_all["code"].astype(str)).union(set(trade_universe["code"].astype(str))))
    update_progress(output_dir, "building_close_map", code_count=len(all_codes))
    market_dates, close_map = build_close_map_for_codes(all_codes)

    search_rows: list[dict[str, Any]] = []
    total_windows = len(windows)
    for idx, window in enumerate(windows, start=1):
        train_df = relaxed_raw[(relaxed_raw["signal_date"] >= window.train_start) & (relaxed_raw["signal_date"] <= window.train_end)].copy()
        val_df = relaxed_raw[(relaxed_raw["signal_date"] >= window.val_start) & (relaxed_raw["signal_date"] <= window.val_end)].copy()
        if train_df.empty or val_df.empty or train_df["label"].nunique() < 2 or val_df["label"].nunique() < 2:
            continue
        factor_model = build_factor_model(train_df)
        train_factor = apply_factor_model(train_df, factor_model)
        val_factor = apply_factor_model(val_df, factor_model)
        for seq_len in seq_lens:
            train_scored, _ = score_similarity(train_factor, train_factor, seq_len, best)
            val_scored, _ = score_similarity(train_factor, val_factor, seq_len, best)
            for model_name, grid in model_grids.items():
                for params in grid:
                    model = fit_model(model_name, params, train_scored)
                    val_prob = predict_prob(val_scored, model)
                    val_metrics = evaluate_model(val_scored, val_prob)
                    search_rows.append(
                        {
                            "window_idx": window.idx,
                            "seq_len": seq_len,
                            "model_name": model_name,
                            "param_key": serialize_params(params),
                            "val_auc": val_metrics["auc"],
                            "val_f1": val_metrics["f1"],
                            "val_accuracy": val_metrics["accuracy"],
                        }
                    )
        update_progress(output_dir, "searching_models", done_windows=idx, total_windows=total_windows)

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(output_dir / "search_validation_results.csv", index=False)
    best_cfg_row = choose_best_config(search_df)
    best_cfg_by_model = choose_best_configs_by_model(search_df)
    best_cfg_by_model.to_csv(output_dir / "best_config_by_model.csv", index=False)
    best_seq_len = int(best_cfg_row["seq_len"])
    best_model_name = str(best_cfg_row["model_name"])
    best_params = None
    for params in model_grids[best_model_name]:
        if serialize_params(params) == str(best_cfg_row["param_key"]):
            best_params = params
            break
    if best_params is None:
        raise RuntimeError("未找到最佳参数对应配置")
    write_json(
        output_dir / "best_config.json",
        {
            "best_seq_len": best_seq_len,
            "best_model_name": best_model_name,
            "best_params": best_params,
            "selection_metric": "mean_val_auc",
            "mean_val_auc": float(best_cfg_row["mean_val_auc"]),
            "mean_val_f1": float(best_cfg_row["mean_val_f1"]),
            "mean_val_accuracy": float(best_cfg_row["mean_val_accuracy"]),
            "grid_profile": grid_profile,
            "seq_lens": seq_lens,
            "model_names": model_names,
            "fixed_exit_strategy": "当日止损次日止盈 + min(open,close)止损 + 5.5%止盈",
        },
    )

    per_model_summaries: list[dict[str, Any]] = []
    per_model_metric_rows: list[dict[str, Any]] = []
    for cfg_row in best_cfg_by_model.itertuples(index=False):
        model_name = str(cfg_row.model_name)
        seq_len = int(cfg_row.seq_len)
        params = None
        for cand in model_grids[model_name]:
            if serialize_params(cand) == str(cfg_row.param_key):
                params = cand
                break
        if params is None:
            continue
        selected_frames_model: list[pd.DataFrame] = []
        model_window_metrics: list[dict[str, Any]] = []
        for window in windows:
            train_df = relaxed_raw[(relaxed_raw["signal_date"] >= window.train_start) & (relaxed_raw["signal_date"] <= window.train_end)].copy()
            val_df = relaxed_raw[(relaxed_raw["signal_date"] >= window.val_start) & (relaxed_raw["signal_date"] <= window.val_end)].copy()
            test_df = relaxed_raw[(relaxed_raw["signal_date"] >= window.test_start) & (relaxed_raw["signal_date"] <= window.test_end)].copy()
            if train_df.empty or val_df.empty or test_df.empty or train_df["label"].nunique() < 2:
                continue
            trainval_df = pd.concat([train_df, val_df], ignore_index=True)
            factor_model = build_factor_model(trainval_df)
            trainval_factor = apply_factor_model(trainval_df, factor_model)
            test_factor = apply_factor_model(test_df, factor_model)
            trainval_scored, _ = score_similarity(trainval_factor, trainval_factor, seq_len, best)
            test_scored, _ = score_similarity(trainval_factor, test_factor, seq_len, best)
            model = fit_model(model_name, params, trainval_scored)
            test_prob = predict_prob(test_scored, model)
            test_scored["ml_score_raw"] = test_prob
            selected = apply_rank_and_select(test_scored, best)
            if not selected.empty:
                selected["window_idx"] = window.idx
                selected_frames_model.append(selected)
            metrics = evaluate_model(test_scored, test_prob)
            model_window_metrics.append(
                {
                    "model_name": model_name,
                    "window_idx": window.idx,
                    "seq_len": seq_len,
                    "param_key": serialize_params(params),
                    "test_auc": metrics["auc"],
                    "test_f1": metrics["f1"],
                    "test_accuracy": metrics["accuracy"],
                    "selected_count": int(len(selected)),
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "val_start": window.val_start,
                    "val_end": window.val_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                }
            )
        if not selected_frames_model:
            continue
        selected_candidates_model = pd.concat(selected_frames_model, ignore_index=True)
        selected_candidates_model.to_csv(output_dir / f"{model_name}_selected_candidates.csv", index=False)
        relaxed_test_trades_model = trade_universe[trade_universe["sample_key"].isin(set(selected_candidates_model["sample_key"].astype(str)))].copy()
        if relaxed_test_trades_model.empty:
            continue
        relaxed_test_trades_model = relaxed_test_trades_model.merge(
            selected_candidates_model[
                ["sample_key", "window_idx", "rank_score", "sim_score", "perfect_case_sim_score", "factor_score", "ml_score"]
            ],
            on="sample_key",
            how="left",
        )
        relaxed_test_trades_model.to_csv(output_dir / f"{model_name}_test_trades.csv", index=False)
        relaxed_eq_model, relaxed_exec_model, relaxed_summary_model = simulate_account_for_subset(
            relaxed_test_trades_model, relaxed_test_trades_model["sample_key"], market_dates, close_map
        )
        relaxed_eq_model.to_csv(output_dir / f"{model_name}_overall_equity.csv", index=False)
        relaxed_exec_model.to_csv(output_dir / f"{model_name}_overall_executed_trades.csv", index=False)
        per_model_summaries.append(
            {
                "model_name": model_name,
                "seq_len": seq_len,
                "param_key": serialize_params(params),
                **relaxed_summary_model,
            }
        )
        per_model_metric_rows.extend(model_window_metrics)

    if per_model_summaries:
        pd.DataFrame(per_model_summaries).to_csv(output_dir / "model_account_summary.csv", index=False)
    if per_model_metric_rows:
        pd.DataFrame(per_model_metric_rows).to_csv(output_dir / "model_test_window_metrics.csv", index=False)

    selected_frames: list[pd.DataFrame] = []
    model_metric_rows: list[dict[str, Any]] = []
    for idx, window in enumerate(windows, start=1):
        train_df = relaxed_raw[(relaxed_raw["signal_date"] >= window.train_start) & (relaxed_raw["signal_date"] <= window.train_end)].copy()
        val_df = relaxed_raw[(relaxed_raw["signal_date"] >= window.val_start) & (relaxed_raw["signal_date"] <= window.val_end)].copy()
        test_df = relaxed_raw[(relaxed_raw["signal_date"] >= window.test_start) & (relaxed_raw["signal_date"] <= window.test_end)].copy()
        if train_df.empty or val_df.empty or test_df.empty or train_df["label"].nunique() < 2:
            continue
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)
        factor_model = build_factor_model(trainval_df)
        trainval_factor = apply_factor_model(trainval_df, factor_model)
        test_factor = apply_factor_model(test_df, factor_model)
        trainval_scored, _ = score_similarity(trainval_factor, trainval_factor, best_seq_len, best)
        test_scored, _ = score_similarity(trainval_factor, test_factor, best_seq_len, best)
        model = fit_model(best_model_name, best_params, trainval_scored)
        test_prob = predict_prob(test_scored, model)
        test_scored["ml_score_raw"] = test_prob
        selected = apply_rank_and_select(test_scored, best)
        if not selected.empty:
            selected["window_idx"] = window.idx
            selected_frames.append(selected)
        metrics = evaluate_model(test_scored, test_prob)
        model_metric_rows.append(
            {
                "window_idx": window.idx,
                "seq_len": best_seq_len,
                "model_name": best_model_name,
                "param_key": serialize_params(best_params),
                "test_auc": metrics["auc"],
                "test_f1": metrics["f1"],
                "test_accuracy": metrics["accuracy"],
                "selected_count": int(len(selected)),
                "train_start": window.train_start,
                "train_end": window.train_end,
                "val_start": window.val_start,
                "val_end": window.val_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
            }
        )
        update_progress(output_dir, "building_final_selection", done_windows=idx, total_windows=total_windows)

    if not selected_frames:
        raise RuntimeError("最佳配置在测试集未生成任何候选")
    selected_candidates = pd.concat(selected_frames, ignore_index=True)
    selected_candidates.to_csv(output_dir / "relaxed_selected_candidates.csv", index=False)
    pd.DataFrame(model_metric_rows).to_csv(output_dir / "rolling_model_metrics.csv", index=False)

    update_progress(output_dir, "building_test_trades", selected_count=int(len(selected_candidates)))
    relaxed_test_trades = trade_universe[trade_universe["sample_key"].isin(set(selected_candidates["sample_key"].astype(str)))].copy()
    if relaxed_test_trades.empty:
        raise RuntimeError("最佳配置未匹配到任何预计算交易")
    relaxed_test_trades = relaxed_test_trades.merge(
        selected_candidates[["sample_key", "window_idx", "rank_score", "sim_score", "factor_score", "ml_score"]],
        on="sample_key",
        how="left",
    )
    relaxed_test_trades.to_csv(output_dir / "relaxed_test_trades.csv", index=False)

    formal_test_frames: list[pd.DataFrame] = []
    for window in windows:
        part = formal_trades_all[(formal_trades_all["signal_date"] >= window.test_start) & (formal_trades_all["signal_date"] <= window.test_end)].copy()
        part["window_idx"] = window.idx
        part["sample_key"] = (
            part["code"].astype(str)
            + "|"
            + pd.to_datetime(part["signal_date"]).dt.strftime("%Y-%m-%d")
            + "|"
            + pd.to_numeric(part["signal_idx"], errors="coerce").fillna(-1).astype(int).astype(str)
        )
        formal_test_frames.append(part)
    formal_test_trades = pd.concat(formal_test_frames, ignore_index=True)
    formal_test_trades.to_csv(output_dir / "formal_test_trades.csv", index=False)

    update_progress(output_dir, "rolling_account_compare", total_windows=total_windows)
    window_rows: list[dict[str, Any]] = []
    all_formal_exec: list[pd.DataFrame] = []
    all_relaxed_exec: list[pd.DataFrame] = []
    for idx, window in enumerate(windows, start=1):
        formal_window = formal_test_trades[formal_test_trades["window_idx"] == window.idx].copy()
        relaxed_window = relaxed_test_trades[relaxed_test_trades["window_idx"] == window.idx].copy()
        if formal_window.empty or relaxed_window.empty:
            update_progress(output_dir, "rolling_account_compare", done_windows=idx, total_windows=total_windows)
            continue
        formal_eq, formal_exec, formal_summary = simulate_account_for_subset(formal_window, formal_window["sample_key"], market_dates, close_map)
        relaxed_eq, relaxed_exec, relaxed_summary = simulate_account_for_subset(relaxed_window, relaxed_window["sample_key"], market_dates, close_map)
        formal_exec["window_idx"] = window.idx
        relaxed_exec["window_idx"] = window.idx
        all_formal_exec.append(formal_exec)
        all_relaxed_exec.append(relaxed_exec)
        window_rows.append(
            {
                "window_idx": window.idx,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "formal_annual_return": formal_summary["annual_return"],
                "relaxed_annual_return": relaxed_summary["annual_return"],
                "annual_return_diff": relaxed_summary["annual_return"] - formal_summary["annual_return"],
                "formal_success_rate": formal_summary["success_rate"],
                "relaxed_success_rate": relaxed_summary["success_rate"],
                "formal_avg_trade_return": formal_summary["avg_trade_return"],
                "relaxed_avg_trade_return": relaxed_summary["avg_trade_return"],
                "formal_max_drawdown": formal_summary["max_drawdown"],
                "relaxed_max_drawdown": relaxed_summary["max_drawdown"],
            }
        )
        update_progress(output_dir, "rolling_account_compare", done_windows=idx, total_windows=total_windows)
    pd.DataFrame(window_rows).to_csv(output_dir / "rolling_window_results.csv", index=False)
    pd.concat(all_formal_exec, ignore_index=True).to_csv(output_dir / "formal_window_executed_trades.csv", index=False)
    pd.concat(all_relaxed_exec, ignore_index=True).to_csv(output_dir / "relaxed_window_executed_trades.csv", index=False)

    update_progress(output_dir, "overall_account_compare")
    formal_eq, formal_exec, formal_summary = simulate_account_for_subset(formal_test_trades, formal_test_trades["sample_key"], market_dates, close_map)
    relaxed_eq, relaxed_exec, relaxed_summary = simulate_account_for_subset(relaxed_test_trades, relaxed_test_trades["sample_key"], market_dates, close_map)
    formal_eq.to_csv(output_dir / "formal_overall_equity.csv", index=False)
    relaxed_eq.to_csv(output_dir / "relaxed_overall_equity.csv", index=False)
    formal_exec.to_csv(output_dir / "formal_overall_executed_trades.csv", index=False)
    relaxed_exec.to_csv(output_dir / "relaxed_overall_executed_trades.csv", index=False)
    pd.DataFrame(
        [
            {"strategy": "formal_best", **formal_summary},
            {"strategy": f"relaxed_{best_model_name}_len{best_seq_len}", **relaxed_summary},
        ]
    ).to_csv(output_dir / "real_account_summary.csv", index=False)

    summary = {
        "compare_start": min(pd.Timestamp(formal_test_trades["signal_date"].min()), pd.Timestamp(relaxed_test_trades["signal_date"].min())).strftime("%Y-%m-%d"),
        "compare_end": max(pd.Timestamp(formal_test_trades["signal_date"].max()), pd.Timestamp(relaxed_test_trades["signal_date"].max())).strftime("%Y-%m-%d"),
        "window_count": len(windows),
        "best_seq_len": best_seq_len,
        "best_model_name": best_model_name,
        "best_params": best_params,
        "best_config_by_model": pd.DataFrame(per_model_summaries).to_dict(orient="records") if per_model_summaries else [],
        "formal_overall": formal_summary,
        "relaxed_overall": relaxed_summary,
        "overall_diff": {
            "annual_return_diff": relaxed_summary["annual_return"] - formal_summary["annual_return"],
            "holding_return_diff": relaxed_summary["holding_return"] - formal_summary["holding_return"],
            "avg_trade_return_diff": relaxed_summary["avg_trade_return"] - formal_summary["avg_trade_return"],
            "success_rate_diff": relaxed_summary["success_rate"] - formal_summary["success_rate"],
            "max_drawdown_diff": relaxed_summary["max_drawdown"] - formal_summary["max_drawdown"],
            "sharpe_diff": relaxed_summary["sharpe"] - formal_summary["sharpe"],
            "calmar_diff": relaxed_summary["calmar"] - formal_summary["calmar"],
            "final_equity_diff": relaxed_summary["final_equity"] - formal_summary["final_equity"],
        },
        "fixed_exit_strategy": "当日止损次日止盈 + min(open,close)止损 + 5.5%止盈",
    }
    write_json(output_dir / "summary.json", summary)
    update_progress(output_dir, "finished", window_count=len(windows), best_seq_len=best_seq_len, best_model_name=best_model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="BRICK relaxed_fusion 短序列与模型公平调参比较")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--file-limit-codes", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--grid-profile", choices=["default", "refine"], default="default")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--seq-lens", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_relaxed_seq_model_search_v1_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_names = [part.strip() for part in args.models.split(",") if part.strip()] if args.models.strip() else list(get_model_grids(args.grid_profile, list(BASE_MODEL_GRIDS.keys())).keys())
    seq_lens = parse_seq_lens(args.seq_lens, args.grid_profile)
    try:
        run(args.mode, output_dir, args.file_limit_codes, args.max_workers, seq_lens, model_names, args.grid_profile)
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
