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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
RELAXED_RESULT_DIR = RESULT_ROOT / "brick_comprehensive_lab_full_20260325_r1"
BUYPOINT_COMPARE_PATH = ROOT / "utils" / "tmp" / "run_brick_buypoint_real_account_compare_v1_20260326.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


compare_base = load_module(BUYPOINT_COMPARE_PATH, "brick_buypoint_compare_rolling_v1_base")
real_account = compare_base.real_account

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))
TRAIN_MONTHS = 24
VAL_MONTHS = 6
TEST_MONTHS = 6

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


def load_relaxed_raw(file_limit_codes: int) -> tuple[dict[str, Any], pd.DataFrame]:
    best = json.loads((RELAXED_RESULT_DIR / "best_config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(RELAXED_RESULT_DIR / "candidate_scored.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    code_map = compare_base.build_daily_code_map()
    df = df[df["candidate_pool"].astype(str) == str(best["candidate_pool"])].copy()
    df["code"] = df["code"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    df["code"] = df["code"].map(code_map)
    df = df.dropna(subset=["code"]).copy()
    if file_limit_codes > 0:
        keep_codes = sorted(df["code"].astype(str).unique())[:file_limit_codes]
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    for col in ["sim_score", "pool_bonus"] + FACTOR_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return best, df.reset_index(drop=True)


def generate_windows(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[RollingWindow]:
    windows: list[RollingWindow] = []
    anchor = pd.Timestamp(start_date).normalize()
    idx = 1
    while True:
        train_start = anchor
        train_end = train_start + DateOffset(months=TRAIN_MONTHS) - pd.Timedelta(days=1)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + DateOffset(months=VAL_MONTHS) - pd.Timedelta(days=1)
        test_start = val_end + pd.Timedelta(days=1)
        if test_start > end_date:
            break
        test_end = test_start + DateOffset(months=TEST_MONTHS) - pd.Timedelta(days=1)
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
        anchor = anchor + DateOffset(months=TEST_MONTHS)
    return windows


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
    return ["sim_score", "factor_score"] + FACTOR_FEATURES


def fit_rf_model(train_df: pd.DataFrame) -> RandomForestClassifier | None:
    if train_df.empty or train_df["label"].nunique() < 2:
        return None
    X = train_df[ml_feature_columns()].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_df["label"].astype(int)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def predict_rf_prob(df: pd.DataFrame, model: RandomForestClassifier | None) -> pd.Series:
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


def build_relaxed_window_selection(all_df: pd.DataFrame, best: dict[str, Any], window: RollingWindow) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_df = all_df[(all_df["signal_date"] >= window.train_start) & (all_df["signal_date"] <= window.train_end)].copy()
    val_df = all_df[(all_df["signal_date"] >= window.val_start) & (all_df["signal_date"] <= window.val_end)].copy()
    test_df = all_df[(all_df["signal_date"] >= window.test_start) & (all_df["signal_date"] <= window.test_end)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        return pd.DataFrame(), {
            "window_idx": window.idx,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "selected_count": 0,
            "skip_reason": "empty_split",
        }
    if train_df["label"].nunique() < 2 or val_df["label"].nunique() < 2:
        return pd.DataFrame(), {
            "window_idx": window.idx,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "selected_count": 0,
            "skip_reason": "single_class",
        }

    factor_model_train = build_factor_model(train_df)
    train_with_factor = apply_factor_model(train_df, factor_model_train)
    val_with_factor = apply_factor_model(val_df, factor_model_train)
    val_model = fit_rf_model(train_with_factor)
    val_prob = predict_rf_prob(val_with_factor, val_model)
    val_metrics = evaluate_model(val_with_factor, val_prob)

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    factor_model_final = build_factor_model(trainval_df)
    trainval_with_factor = apply_factor_model(trainval_df, factor_model_final)
    test_with_factor = apply_factor_model(test_df, factor_model_final)
    final_model = fit_rf_model(trainval_with_factor)
    test_prob = predict_rf_prob(test_with_factor, final_model)
    test_with_factor["ml_score_raw"] = test_prob
    test_with_factor["ml_score"] = normalize_rank(test_prob)
    test_with_factor["rank_score"] = (
        normalize_rank(test_with_factor["sim_score"]) * float(best["sim_weight"])
        + pd.to_numeric(test_with_factor["factor_score"], errors="coerce").fillna(0.0) * float(best["factor_weight"])
        + pd.to_numeric(test_with_factor["ml_score"], errors="coerce").fillna(0.0) * float(best["ml_weight"])
        + pd.to_numeric(test_with_factor["pool_bonus"], errors="coerce").fillna(0.0)
    )
    selected = test_with_factor[test_with_factor["sim_score"] >= float(best["sim_gate"])].copy()
    selected = (
        selected.sort_values(["signal_date", "rank_score", "code"], ascending=[True, False, True], kind="mergesort")
        .groupby("signal_date", group_keys=False)
        .head(int(best["daily_topn"]))
        .reset_index(drop=True)
    )
    selected["strategy_key"] = str(best["strategy"])
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
        "label",
        "strategy_key",
    ]
    test_metrics = evaluate_model(test_with_factor, test_prob)
    info = {
        "window_idx": window.idx,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "selected_count": int(len(selected)),
        "val_auc": val_metrics["auc"],
        "val_f1": val_metrics["f1"],
        "val_accuracy": val_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
        "skip_reason": "",
    }
    return selected[keep_cols], info


def runs_test(return_series: pd.Series) -> dict[str, float]:
    seq = (pd.to_numeric(return_series, errors="coerce").fillna(0.0) > 0).astype(int).tolist()
    if len(seq) < 2:
        return {"n": len(seq), "runs": float("nan"), "expected_runs": float("nan"), "z": float("nan"), "p_value": float("nan")}
    runs = 1
    for idx in range(1, len(seq)):
        if seq[idx] != seq[idx - 1]:
            runs += 1
    n1 = sum(seq)
    n0 = len(seq) - n1
    if n1 == 0 or n0 == 0:
        return {"n": len(seq), "runs": runs, "expected_runs": float("nan"), "z": float("nan"), "p_value": float("nan")}
    expected = 1.0 + 2.0 * n1 * n0 / (n1 + n0)
    variance = (2.0 * n1 * n0 * (2.0 * n1 * n0 - n1 - n0)) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
    if variance <= 0:
        z = float("nan")
        p = float("nan")
    else:
        z = (runs - expected) / math.sqrt(variance)
        p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n": len(seq), "runs": float(runs), "expected_runs": float(expected), "z": float(z), "p_value": float(p)}


def markov_transition(return_series: pd.Series) -> dict[str, float]:
    seq = (pd.to_numeric(return_series, errors="coerce").fillna(0.0) > 0).astype(int).tolist()
    counts = {"ww": 0, "wl": 0, "lw": 0, "ll": 0}
    for prev, cur in zip(seq[:-1], seq[1:]):
        if prev == 1 and cur == 1:
            counts["ww"] += 1
        elif prev == 1 and cur == 0:
            counts["wl"] += 1
        elif prev == 0 and cur == 1:
            counts["lw"] += 1
        else:
            counts["ll"] += 1
    win_total = counts["ww"] + counts["wl"]
    lose_total = counts["lw"] + counts["ll"]
    return {
        **counts,
        "p_win_to_win": counts["ww"] / win_total if win_total else float("nan"),
        "p_win_to_lose": counts["wl"] / win_total if win_total else float("nan"),
        "p_lose_to_win": counts["lw"] / lose_total if lose_total else float("nan"),
        "p_lose_to_lose": counts["ll"] / lose_total if lose_total else float("nan"),
    }


def compare_window_account(
    result_dir: Path,
    window: RollingWindow,
    formal_window_trades: pd.DataFrame,
    relaxed_window_trades: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    codes = sorted(set(formal_window_trades["code"].astype(str)).union(set(relaxed_window_trades["code"].astype(str))))
    market_dates, close_map = real_account.build_close_map(codes)
    max_exit_date = max(
        pd.Timestamp(formal_window_trades["exit_date"].max()) if not formal_window_trades.empty else window.test_end,
        pd.Timestamp(relaxed_window_trades["exit_date"].max()) if not relaxed_window_trades.empty else window.test_end,
    )
    market_dates = market_dates[(market_dates >= window.test_start) & (market_dates <= max_exit_date)]
    close_map = {code: series.reindex(market_dates).ffill() for code, series in close_map.items() if code in codes}
    config = real_account.AccountConfig()
    formal_equity, formal_exec, formal_summary = real_account.simulate_real_account(formal_window_trades, close_map, market_dates, config)
    relaxed_equity, relaxed_exec, relaxed_summary = real_account.simulate_real_account(relaxed_window_trades, close_map, market_dates, config)
    row = {
        "window_idx": window.idx,
        "train_start": window.train_start,
        "train_end": window.train_end,
        "val_start": window.val_start,
        "val_end": window.val_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "formal_annual_return": formal_summary["annual_return"],
        "formal_holding_return": formal_summary["holding_return"],
        "formal_avg_trade_return": formal_summary["avg_trade_return"],
        "formal_success_rate": formal_summary["success_rate"],
        "formal_max_drawdown": formal_summary["max_drawdown"],
        "formal_sharpe": formal_summary["sharpe"],
        "formal_calmar": formal_summary["calmar"],
        "formal_final_equity": formal_summary["final_equity"],
        "formal_trade_count": formal_summary["trade_count"],
        "relaxed_annual_return": relaxed_summary["annual_return"],
        "relaxed_holding_return": relaxed_summary["holding_return"],
        "relaxed_avg_trade_return": relaxed_summary["avg_trade_return"],
        "relaxed_success_rate": relaxed_summary["success_rate"],
        "relaxed_max_drawdown": relaxed_summary["max_drawdown"],
        "relaxed_sharpe": relaxed_summary["sharpe"],
        "relaxed_calmar": relaxed_summary["calmar"],
        "relaxed_final_equity": relaxed_summary["final_equity"],
        "relaxed_trade_count": relaxed_summary["trade_count"],
        "annual_return_diff": relaxed_summary["annual_return"] - formal_summary["annual_return"],
        "holding_return_diff": relaxed_summary["holding_return"] - formal_summary["holding_return"],
        "avg_trade_return_diff": relaxed_summary["avg_trade_return"] - formal_summary["avg_trade_return"],
        "success_rate_diff": relaxed_summary["success_rate"] - formal_summary["success_rate"],
        "max_drawdown_diff": relaxed_summary["max_drawdown"] - formal_summary["max_drawdown"],
        "sharpe_diff": relaxed_summary["sharpe"] - formal_summary["sharpe"],
        "calmar_diff": relaxed_summary["calmar"] - formal_summary["calmar"],
        "final_equity_diff": relaxed_summary["final_equity"] - formal_summary["final_equity"],
    }
    return row, formal_exec, relaxed_exec


def build_overall_summary(window_df: pd.DataFrame, formal_summary: dict[str, Any], relaxed_summary: dict[str, Any]) -> dict[str, Any]:
    relaxed_win_rate = float((window_df["annual_return_diff"] > 0).mean()) if not window_df.empty else float("nan")
    return {
        "window_count": int(len(window_df)),
        "rolling_relaxed_win_rate": relaxed_win_rate,
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
    }


def run(mode: str, output_dir: Path, file_limit_codes: int, max_workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(output_dir, "loading_inputs", mode=mode, file_limit_codes=file_limit_codes, max_workers=max_workers)

    best, relaxed_raw = load_relaxed_raw(file_limit_codes)
    formal_strategy_key, formal_trades_all = compare_base.load_formal_champion(file_limit_codes)
    formal_trades_all.to_csv(output_dir / "formal_source_trades.csv", index=False)

    common_start = max(pd.Timestamp(relaxed_raw["signal_date"].min()), pd.Timestamp(formal_trades_all["signal_date"].min()))
    common_end = min(pd.Timestamp(relaxed_raw["signal_date"].max()), pd.Timestamp(formal_trades_all["signal_date"].max()))
    relaxed_raw = relaxed_raw[(relaxed_raw["signal_date"] >= common_start) & (relaxed_raw["signal_date"] <= common_end)].copy()
    formal_trades_all = formal_trades_all[(formal_trades_all["signal_date"] >= common_start) & (formal_trades_all["signal_date"] <= common_end)].copy()

    windows = generate_windows(common_start, common_end)
    if mode == "smoke":
        windows = windows[:1]
    if not windows:
        raise RuntimeError("滚动窗口为空，无法比较")
    write_json(
        output_dir / "window_plan.json",
        {
            "common_start": common_start.strftime("%Y-%m-%d"),
            "common_end": common_end.strftime("%Y-%m-%d"),
            "train_months": TRAIN_MONTHS,
            "val_months": VAL_MONTHS,
            "test_months": TEST_MONTHS,
            "window_count": len(windows),
            "windows": [window.__dict__ for window in windows],
            "best_config": best,
            "formal_strategy_key": formal_strategy_key,
        },
    )

    selected_frames: list[pd.DataFrame] = []
    window_model_rows: list[dict[str, Any]] = []
    total_windows = len(windows)
    for idx, window in enumerate(windows, start=1):
        selected_df, info = build_relaxed_window_selection(relaxed_raw, best, window)
        info.update(
            {
                "train_start": window.train_start,
                "train_end": window.train_end,
                "val_start": window.val_start,
                "val_end": window.val_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
            }
        )
        window_model_rows.append(info)
        if not selected_df.empty:
            selected_df["window_idx"] = window.idx
            selected_frames.append(selected_df)
        update_progress(output_dir, "rolling_selection_ready", done_windows=idx, total_windows=total_windows)

    window_model_df = pd.DataFrame(window_model_rows)
    window_model_df.to_csv(output_dir / "rolling_model_metrics.csv", index=False)
    if not selected_frames:
        raise RuntimeError("滚动重训后 relaxed_fusion 没有生成任何候选")
    selected_candidates = pd.concat(selected_frames, ignore_index=True)
    selected_candidates.to_csv(output_dir / "relaxed_selected_candidates.csv", index=False)

    update_progress(output_dir, "simulating_relaxed_trades", candidate_count=int(len(selected_candidates)))
    relaxed_trades = compare_base.build_relaxed_trades(selected_candidates, output_dir, max_workers=max_workers)
    if relaxed_trades.empty:
        raise RuntimeError("滚动重训后 relaxed_fusion 未生成有效交易")
    relaxed_trades["signal_date"] = pd.to_datetime(relaxed_trades["signal_date"])
    relaxed_trades["entry_date"] = pd.to_datetime(relaxed_trades["entry_date"])
    relaxed_trades["exit_date"] = pd.to_datetime(relaxed_trades["exit_date"])
    relaxed_trades = relaxed_trades.merge(
        selected_candidates[["code", "signal_idx", "signal_date", "entry_date", "window_idx"]],
        on=["code", "signal_idx", "signal_date", "entry_date"],
        how="left",
    )
    relaxed_trades.to_csv(output_dir / "relaxed_trades.csv", index=False)

    formal_test_frames: list[pd.DataFrame] = []
    for window in windows:
        part = formal_trades_all[(formal_trades_all["signal_date"] >= window.test_start) & (formal_trades_all["signal_date"] <= window.test_end)].copy()
        part["window_idx"] = window.idx
        formal_test_frames.append(part)
    formal_test_trades = pd.concat(formal_test_frames, ignore_index=True)
    formal_test_trades.to_csv(output_dir / "formal_test_trades.csv", index=False)

    update_progress(output_dir, "rolling_account_compare", total_windows=total_windows)
    window_compare_rows: list[dict[str, Any]] = []
    all_formal_exec: list[pd.DataFrame] = []
    all_relaxed_exec: list[pd.DataFrame] = []
    for idx, window in enumerate(windows, start=1):
        formal_window = formal_test_trades[formal_test_trades["window_idx"] == window.idx].copy()
        relaxed_window = relaxed_trades[relaxed_trades["window_idx"] == window.idx].copy()
        if formal_window.empty or relaxed_window.empty:
            window_compare_rows.append(
                {
                    "window_idx": window.idx,
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "val_start": window.val_start,
                    "val_end": window.val_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                    "skip_reason": "missing_strategy_trades",
                }
            )
            update_progress(output_dir, "rolling_account_compare", done_windows=idx, total_windows=total_windows)
            continue
        row, formal_exec, relaxed_exec = compare_window_account(output_dir, window, formal_window, relaxed_window)
        window_compare_rows.append(row)
        formal_exec["window_idx"] = window.idx
        relaxed_exec["window_idx"] = window.idx
        all_formal_exec.append(formal_exec)
        all_relaxed_exec.append(relaxed_exec)
        update_progress(output_dir, "rolling_account_compare", done_windows=idx, total_windows=total_windows)

    window_compare_df = pd.DataFrame(window_compare_rows)
    window_compare_df.to_csv(output_dir / "rolling_window_results.csv", index=False)
    formal_window_exec_df = pd.concat(all_formal_exec, ignore_index=True) if all_formal_exec else pd.DataFrame()
    relaxed_window_exec_df = pd.concat(all_relaxed_exec, ignore_index=True) if all_relaxed_exec else pd.DataFrame()
    formal_window_exec_df.to_csv(output_dir / "formal_window_executed_trades.csv", index=False)
    relaxed_window_exec_df.to_csv(output_dir / "relaxed_window_executed_trades.csv", index=False)

    update_progress(output_dir, "overall_account_compare")
    formal_overall = formal_test_trades.sort_values(["signal_date", "code", "signal_idx"]).reset_index(drop=True)
    relaxed_overall = relaxed_trades.sort_values(["signal_date", "code", "signal_idx"]).reset_index(drop=True)
    all_codes = sorted(set(formal_overall["code"].astype(str)).union(set(relaxed_overall["code"].astype(str))))
    market_dates, close_map = real_account.build_close_map(all_codes)
    overall_end = max(pd.Timestamp(formal_overall["exit_date"].max()), pd.Timestamp(relaxed_overall["exit_date"].max()))
    overall_start = min(pd.Timestamp(formal_overall["entry_date"].min()), pd.Timestamp(relaxed_overall["entry_date"].min()))
    market_dates = market_dates[(market_dates >= overall_start) & (market_dates <= overall_end)]
    close_map = {code: series.reindex(market_dates).ffill() for code, series in close_map.items() if code in all_codes}
    formal_equity, formal_exec, formal_summary = real_account.simulate_real_account(formal_overall, close_map, market_dates, real_account.AccountConfig())
    relaxed_equity, relaxed_exec, relaxed_summary = real_account.simulate_real_account(relaxed_overall, close_map, market_dates, real_account.AccountConfig())

    formal_equity.to_csv(output_dir / "formal_overall_equity.csv", index=False)
    relaxed_equity.to_csv(output_dir / "relaxed_overall_equity.csv", index=False)
    formal_exec.to_csv(output_dir / "formal_overall_executed_trades.csv", index=False)
    relaxed_exec.to_csv(output_dir / "relaxed_overall_executed_trades.csv", index=False)

    runs_rows = [
        {"strategy": "formal_best", **runs_test(formal_exec["return_pct_net"] if not formal_exec.empty else pd.Series(dtype=float))},
        {"strategy": "relaxed_fusion_rolling", **runs_test(relaxed_exec["return_pct_net"] if not relaxed_exec.empty else pd.Series(dtype=float))},
    ]
    pd.DataFrame(runs_rows).to_csv(output_dir / "runs_test_summary.csv", index=False)

    markov_rows = [
        {"strategy": "formal_best", **markov_transition(formal_exec["return_pct_net"] if not formal_exec.empty else pd.Series(dtype=float))},
        {"strategy": "relaxed_fusion_rolling", **markov_transition(relaxed_exec["return_pct_net"] if not relaxed_exec.empty else pd.Series(dtype=float))},
    ]
    pd.DataFrame(markov_rows).to_csv(output_dir / "markov_transition_summary.csv", index=False)

    real_account_summary = pd.DataFrame(
        [
            {"strategy": "formal_best", **formal_summary},
            {"strategy": "relaxed_fusion_rolling", **relaxed_summary},
        ]
    )
    real_account_summary.to_csv(output_dir / "real_account_summary.csv", index=False)

    summary = build_overall_summary(window_compare_df, formal_summary, relaxed_summary)
    summary.update(
        {
            "compare_start": overall_start.strftime("%Y-%m-%d"),
            "compare_end": overall_end.strftime("%Y-%m-%d"),
            "formal_strategy_key": formal_strategy_key,
            "relaxed_strategy_key": str(best["strategy"]),
            "rolling_train_months": TRAIN_MONTHS,
            "rolling_val_months": VAL_MONTHS,
            "rolling_test_months": TEST_MONTHS,
        }
    )
    write_json(output_dir / "summary.json", summary)
    update_progress(output_dir, "finished", window_count=len(windows), compare_start=summary["compare_start"], compare_end=summary["compare_end"])


def main() -> None:
    parser = argparse.ArgumentParser(description="BRICK 买点滚动重训对比 formal_best vs relaxed_fusion")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--file-limit-codes", type=int, default=0, help="仅保留前N个股票代码，0表示全量")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    try:
        run(args.mode, output_dir, args.file_limit_codes, args.max_workers)
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
