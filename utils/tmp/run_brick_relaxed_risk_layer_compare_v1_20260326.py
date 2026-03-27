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
SEQ_SEARCH_PATH = ROOT / "utils" / "tmp" / "run_brick_relaxed_seq_model_search_v1_20260326.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


seq = load_module(SEQ_SEARCH_PATH, "brick_relaxed_risk_compare_seq")

BASE_FACTOR_FEATURES = [
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
CASE_FEATURES = ["brick_case_type_score", "early_red_stage_flag_num"]
RISK_FEATURES = ["risk_distribution_recent_20", "risk_distribution_recent_30", "risk_distribution_recent_60"]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def write_error(result_dir: Path, exc: BaseException) -> None:
    write_json(
        result_dir / "error.json",
        {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
    update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))


def feature_columns(profile: str) -> tuple[list[str], list[str]]:
    if profile == "base":
        factor_cols = list(BASE_FACTOR_FEATURES)
        ml_cols = ["sim_score", "factor_score"] + factor_cols
        return factor_cols, ml_cols
    if profile == "risk_only":
        factor_cols = BASE_FACTOR_FEATURES + RISK_FEATURES
        ml_cols = ["sim_score", "factor_score"] + factor_cols
        return factor_cols, ml_cols
    if profile == "risk_case":
        factor_cols = BASE_FACTOR_FEATURES + CASE_FEATURES + RISK_FEATURES
        ml_cols = ["sim_score", "perfect_case_sim_score", "factor_score"] + factor_cols
        return factor_cols, ml_cols
    raise RuntimeError(f"未知 profile: {profile}")


def normalize_rank(series: pd.Series) -> pd.Series:
    return seq.normalize_rank(series)


def build_factor_model(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    state: dict[str, Any] = {"features": {}, "feature_order": list(features)}
    for feature in features:
        series = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
        mean = float(series.mean())
        std = float(series.std())
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        pos_mean = float(pd.to_numeric(pos[feature], errors="coerce").fillna(0.0).mean()) if not pos.empty else mean
        neg_mean = float(pd.to_numeric(neg[feature], errors="coerce").fillna(0.0).mean()) if not neg.empty else mean
        direction = 1.0 if pos_mean >= neg_mean else -1.0
        state["features"][feature] = {"mean": mean, "std": std, "direction": direction}
    return state


def apply_factor_model(df: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    score = pd.Series(0.0, index=out.index, dtype=float)
    for feature in model["feature_order"]:
        meta = model["features"][feature]
        series = pd.to_numeric(out[feature], errors="coerce").fillna(meta["mean"])
        z = (series - meta["mean"]) / meta["std"]
        score = score + z * meta["direction"]
    out["factor_score_raw"] = score
    out["factor_score"] = normalize_rank(score)
    return out


def fit_model(model_name: str, params: dict[str, Any], train_df: pd.DataFrame, ml_cols: list[str]):
    if train_df.empty or train_df["label"].nunique() < 2:
        return None
    X = train_df[ml_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_df["label"].astype(int)
    if model_name == "logreg":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=1000,
            solver="lbfgs",
            n_jobs=1,
            random_state=42,
        )
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            n_jobs=1,
            random_state=42,
        )
    elif model_name == "xgb" and seq.XGB_OK:
        model = seq.XGBClassifier(
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
    elif model_name == "lgbm" and seq.LGBM_OK:
        model = seq.LGBMClassifier(
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


def predict_prob(df: pd.DataFrame, model, ml_cols: list[str]) -> pd.Series:
    if model is None or df.empty:
        return pd.Series(np.nan, index=df.index, dtype=float)
    X = df[ml_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index, dtype=float)


def apply_rank_and_select(df: pd.DataFrame, best: dict[str, Any], profile: str) -> pd.DataFrame:
    out = df.copy()
    out["ml_score"] = normalize_rank(pd.to_numeric(out["ml_score_raw"], errors="coerce").fillna(0.0))
    risk_penalty = (
        pd.to_numeric(out["risk_distribution_recent_20"], errors="coerce").fillna(0.0) * 0.08
        + pd.to_numeric(out["risk_distribution_recent_30"], errors="coerce").fillna(0.0) * 0.05
        + pd.to_numeric(out["risk_distribution_recent_60"], errors="coerce").fillna(0.0) * 0.03
    )
    case_bonus = (
        (pd.to_numeric(out["brick_case_type_score"], errors="coerce").fillna(0.45) - 0.45) * 0.12
        + pd.to_numeric(out["early_red_stage_flag_num"], errors="coerce").fillna(0.0) * 0.04
    )
    if profile == "base":
        pool_bonus = pd.Series(0.0, index=out.index)
    elif profile == "risk_only":
        pool_bonus = -risk_penalty
    else:
        pool_bonus = case_bonus - risk_penalty
    base_rank = (
        normalize_rank(out["sim_score"]) * float(best["sim_weight"])
        + pd.to_numeric(out["factor_score"], errors="coerce").fillna(0.0) * float(best["factor_weight"])
        + pd.to_numeric(out["ml_score"], errors="coerce").fillna(0.0) * float(best["ml_weight"])
        + pool_bonus
    )
    if profile == "risk_case":
        out["perfect_case_rank"] = normalize_rank(pd.to_numeric(out["perfect_case_sim_score"], errors="coerce").fillna(-1.0))
        out["rank_score"] = (1.0 - seq.relaxed.PERFECT_CASE_WEIGHT) * base_rank + seq.relaxed.PERFECT_CASE_WEIGHT * out["perfect_case_rank"]
        sort_cols = ["signal_date", "perfect_case_rank", "rank_score", "code"]
        ascending = [True, False, False, True]
    else:
        out["rank_score"] = base_rank
        sort_cols = ["signal_date", "rank_score", "code"]
        ascending = [True, False, True]
    gated = out[out["sim_score"] >= float(best["sim_gate"])].copy()
    if gated.empty:
        return gated
    return (
        gated.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        .groupby("signal_date", group_keys=False)
        .head(int(best["daily_topn"]))
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BRICK relaxed_fusion 风险层对比")
    parser.add_argument("--source-result-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        best_cfg = json.loads((source_dir / "best_config.json").read_text(encoding="utf-8"))
        summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
        relaxed_raw = pd.read_pickle(source_dir / "relaxed_raw_with_seq.pkl")
        trade_universe = pd.read_csv(source_dir / "relaxed_trade_universe.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
        formal_source = pd.read_csv(source_dir / "formal_source_trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
        window_plan = json.loads((source_dir / "window_plan.json").read_text(encoding="utf-8"))
        windows = [seq.RollingWindow(**row) for row in window_plan["windows"]]
        model_name = str(best_cfg["best_model_name"])
        params = best_cfg["best_params"]
        seq_len = int(best_cfg["best_seq_len"])
        compare_start = pd.Timestamp(summary["compare_start"])
        compare_end = pd.Timestamp(summary["compare_end"])
        formal_source = formal_source[(formal_source["signal_date"] >= compare_start) & (formal_source["signal_date"] <= compare_end)].copy()
        trade_universe = trade_universe[(trade_universe["signal_date"] >= compare_start) & (trade_universe["signal_date"] <= compare_end)].copy()
        all_codes = sorted(set(formal_source["code"].astype(str)).union(set(trade_universe["code"].astype(str))))
        update_progress(output_dir, "building_close_map", code_count=len(all_codes))
        market_dates, close_map = seq.build_close_map_for_codes(all_codes)

        profile_summaries: list[dict[str, Any]] = []
        for profile in ["base", "risk_only", "risk_case"]:
            factor_cols, ml_cols = feature_columns(profile)
            selected_frames: list[pd.DataFrame] = []
            for idx, window in enumerate(windows, start=1):
                train_df = relaxed_raw[(relaxed_raw["signal_date"] >= pd.Timestamp(window.train_start)) & (relaxed_raw["signal_date"] <= pd.Timestamp(window.train_end))].copy()
                val_df = relaxed_raw[(relaxed_raw["signal_date"] >= pd.Timestamp(window.val_start)) & (relaxed_raw["signal_date"] <= pd.Timestamp(window.val_end))].copy()
                test_df = relaxed_raw[(relaxed_raw["signal_date"] >= pd.Timestamp(window.test_start)) & (relaxed_raw["signal_date"] <= pd.Timestamp(window.test_end))].copy()
                if train_df.empty or val_df.empty or test_df.empty or train_df["label"].nunique() < 2:
                    continue
                trainval_df = pd.concat([train_df, val_df], ignore_index=True)
                factor_model = build_factor_model(trainval_df, factor_cols)
                trainval_factor = apply_factor_model(trainval_df, factor_model)
                test_factor = apply_factor_model(test_df, factor_model)
                trainval_scored, _ = seq.score_similarity(trainval_factor, trainval_factor, seq_len, seq.relaxed._load_best_and_history()[0])
                test_scored, _ = seq.score_similarity(trainval_factor, test_factor, seq_len, seq.relaxed._load_best_and_history()[0])
                model = fit_model(model_name, params, trainval_scored, ml_cols)
                test_scored["ml_score_raw"] = predict_prob(test_scored, model, ml_cols)
                selected = apply_rank_and_select(test_scored, seq.relaxed._load_best_and_history()[0], profile)
                if not selected.empty:
                    selected["window_idx"] = window.idx
                    selected_frames.append(selected)
                update_progress(output_dir, "running_profile", profile=profile, done_windows=idx, total_windows=len(windows))
            if not selected_frames:
                continue
            selected_candidates = pd.concat(selected_frames, ignore_index=True)
            selected_candidates.to_csv(output_dir / f"{profile}_selected_candidates.csv", index=False)
            trades = trade_universe[trade_universe["sample_key"].isin(set(selected_candidates["sample_key"].astype(str)))].copy()
            eq, exec_df, metrics = seq.simulate_account_for_subset(trades, trades["sample_key"], market_dates, close_map)
            eq.to_csv(output_dir / f"{profile}_equity.csv", index=False)
            exec_df.to_csv(output_dir / f"{profile}_executed_trades.csv", index=False)
            profile_summaries.append({"profile": profile, "model_name": model_name, "seq_len": seq_len, **metrics})

        if not profile_summaries:
            raise RuntimeError("风险层对比没有产出任何候选")
        pd.DataFrame(profile_summaries).to_csv(output_dir / "risk_profile_account_summary.csv", index=False)
        write_json(
            output_dir / "summary.json",
            {
                "source_result_dir": str(source_dir),
                "model_name": model_name,
                "seq_len": seq_len,
                "params": params,
                "profiles": profile_summaries,
                "fixed_exit_strategy": "当日止损次日止盈 + min(open,close)止损 + 5.5%止盈",
            },
        )
        update_progress(output_dir, "finished", profile_count=len(profile_summaries))
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
