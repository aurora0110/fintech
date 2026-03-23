from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
BASE_SIGNAL_DIR = ROOT / "results" / "b1_full_factor_signal_v6_full_20260321_102049"
ACCOUNT_BASE_DIR = ROOT / "results" / "b1_buy_sell_model_account_v2_full_20260321_102049"
SELL_MODEL_DIR = ROOT / "results" / "b1_sell_habit_experiment_v2_20260320_233121"
ACCOUNT_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_buy_sell_model_account_v2_20260320.py"

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_RESULT_DIR = ROOT / "results" / f"b1_targeted_joint_signal_v1_{RUN_TS}"
DEFAULT_TOPN_LIST = [3, 5, 8]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


acc_mod = load_module(ACCOUNT_SCRIPT, "b1_targeted_joint_acc")

HAS_SKLEARN = False
HAS_LGB = False
HAS_XGB = False
try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except Exception:
    ExtraTreesClassifier = None  # type: ignore
    LogisticRegression = None  # type: ignore

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    lgb = None  # type: ignore

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    xgb = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 定向联合优化：冠军方案 + 相似度 + 机器学习")
    parser.add_argument("--base-signal-dir", type=Path, default=BASE_SIGNAL_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--file-limit", type=int, default=0)
    parser.add_argument("--topn-list", type=str, default="")
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **kwargs: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    payload.update(kwargs)
    write_json(result_dir / "progress.json", payload)


def parse_topn_list(raw: str) -> List[int]:
    if not raw.strip():
        return DEFAULT_TOPN_LIST
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    return sorted(set(v for v in vals if v > 0))


def load_candidate_df(base_signal_dir: Path, file_limit: int) -> pd.DataFrame:
    df = pd.read_csv(base_signal_dir / "candidate_enriched.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    if file_limit > 0:
        keep_codes = sorted(df["code"].astype(str).drop_duplicates().tolist())[:file_limit]
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    return df.reset_index(drop=True)


def ecdf_score(train_vals: pd.Series, values: pd.Series) -> pd.Series:
    arr = np.sort(train_vals.dropna().astype(float).to_numpy())
    if len(arr) == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    x = pd.to_numeric(values, errors="coerce").fillna(np.nan).to_numpy(dtype=float)
    pos = np.searchsorted(arr, x, side="right")
    score = pos / float(len(arr))
    score[~np.isfinite(x)] = 0.5
    return pd.Series(score, index=values.index, dtype=float)


def add_targeted_semantics(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for col in [
        "near_trend_pullback",
        "near_long_pullback",
        "semantic_uptrend_pullback",
        "semantic_low_cross_pullback",
        "key_k_support",
        "half_volume",
        "semi_shrink",
        "double_bull_exist_60",
        "risk_distribution_any_20",
        "pool_low_cross",
    ]:
        if col in x.columns:
            x[col] = x[col].fillna(False).astype(bool)
    x["target_no_risk"] = ~x["risk_distribution_any_20"]
    x["target_pullback_trend"] = (
        (x["J"] < 13)
        & x["target_no_risk"]
        & x["near_trend_pullback"]
        & x["semantic_uptrend_pullback"]
    )
    x["target_pullback_long"] = (
        (x["J"] < 13)
        & x["target_no_risk"]
        & x["near_long_pullback"]
    )
    x["target_dual_core"] = x["target_pullback_trend"] | x["target_pullback_long"]
    x["target_confirm_bonus"] = (
        x["key_k_support"].astype(int) * 1.0
        + x["half_volume"].astype(int) * 0.8
        + x["semi_shrink"].astype(int) * 0.5
        + x["double_bull_exist_60"].astype(int) * 0.3
        + x["semantic_low_cross_pullback"].astype(int) * 1.2
        + x["semantic_uptrend_pullback"].astype(int) * 0.6
    )
    x["target_dual_confirmed"] = x["target_dual_core"] & (x["target_confirm_bonus"] >= 1.0)
    x["target_dual_strict"] = x["target_dual_core"] & (
        x["key_k_support"] | x["half_volume"] | x["semantic_low_cross_pullback"]
    )
    x["pool_target_trend"] = x["target_pullback_trend"]
    x["pool_target_long"] = x["target_pullback_long"]
    x["pool_target_dual"] = x["target_dual_core"]
    x["pool_target_confirmed"] = x["target_dual_confirmed"]
    x["pool_target_strict"] = x["target_dual_strict"]
    return x


def fit_focused_models(df: pd.DataFrame, feature_cols: List[str]) -> tuple[Dict[str, Any], Dict[str, List[str]]]:
    research = df[df["split"] == "research"].copy()
    research = research[research["target_dual_core"]].copy()
    research["label_strong20"] = (pd.to_numeric(research["ret_20d"], errors="coerce").fillna(0.0) > 0.10).astype(int)
    research = research.dropna(subset=feature_cols).copy()
    if research.empty or research["label_strong20"].nunique() < 2:
        return {}, {}
    X = research[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = research["label_strong20"].to_numpy(dtype=int)
    model_map: Dict[str, Any] = {}
    feature_map: Dict[str, List[str]] = {}

    if HAS_SKLEARN:
        try:
            lr = LogisticRegression(max_iter=500, class_weight="balanced")
            lr.fit(X, y)
            model_map["focused_lr_score"] = lr
            feature_map["focused_lr_score"] = feature_cols
        except Exception:
            pass
        try:
            et = ExtraTreesClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            et.fit(X, y)
            model_map["focused_et_score"] = et
            feature_map["focused_et_score"] = feature_cols
        except Exception:
            pass
    if HAS_LGB:
        try:
            clf = lgb.LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                min_child_samples=10,
                random_state=42,
            )
            clf.fit(X, y)
            model_map["focused_lgb_score"] = clf
            feature_map["focused_lgb_score"] = feature_cols
        except Exception:
            pass
    if HAS_XGB:
        try:
            clf = xgb.XGBClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
            clf.fit(X, y)
            model_map["focused_xgb_score"] = clf
            feature_map["focused_xgb_score"] = feature_cols
        except Exception:
            pass
    return model_map, feature_map


def score_focused_models(df: pd.DataFrame, model_map: Dict[str, Any], feature_map: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    for score_col, model in model_map.items():
        cols = feature_map[score_col]
        X = out[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        out[score_col] = model.predict_proba(X)[:, 1]
    return out


def add_targeted_scores(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    research = x[x["split"] == "research"].copy()
    rank_cols = [
        "discovery_factor_score",
        "sim_corr_close_vol_concat",
        "sim_cosine_close_vol_concat",
        "sim_spearman_close_vol_concat_contrast",
        "xgb_full_score",
        "lgb_full_score",
        "et_full_score",
        "target_confirm_bonus",
    ]
    for col in [c for c in rank_cols if c in x.columns]:
        x[f"rank_{col}_target"] = ecdf_score(research[col], x[col])
    if "focused_xgb_score" in x.columns:
        x["rank_focused_xgb_score_target"] = ecdf_score(research["focused_xgb_score"], x["focused_xgb_score"])
    if "focused_lgb_score" in x.columns:
        x["rank_focused_lgb_score_target"] = ecdf_score(research["focused_lgb_score"], x["focused_lgb_score"])
    if "focused_et_score" in x.columns:
        x["rank_focused_et_score_target"] = ecdf_score(research["focused_et_score"], x["focused_et_score"])
    x["target_similarity_core_score"] = (
        x.get("rank_sim_corr_close_vol_concat_target", 0.0)
        + x.get("rank_sim_cosine_close_vol_concat_target", 0.0)
        + x.get("rank_sim_spearman_close_vol_concat_contrast_target", 0.0)
    ) / 3.0
    ml_parts = []
    for c in ["rank_focused_xgb_score_target", "rank_focused_lgb_score_target", "rank_focused_et_score_target"]:
        if c in x.columns:
            ml_parts.append(x[c])
    if not ml_parts:
        for c in ["rank_xgb_full_score_target", "rank_lgb_full_score_target", "rank_et_full_score_target"]:
            if c in x.columns:
                ml_parts.append(x[c])
    x["target_ml_core_score"] = np.mean(np.column_stack(ml_parts), axis=1) if ml_parts else 0.5
    x["target_aux_core_score"] = (
        x.get("rank_target_confirm_bonus_target", 0.0) * 0.7
        + x["semantic_low_cross_pullback"].astype(float) * 0.2
        + x["semantic_uptrend_pullback"].astype(float) * 0.1
    )
    x["target_joint_champion_similarity_score"] = (
        x.get("rank_discovery_factor_score_target", 0.0) * 0.55
        + x["target_similarity_core_score"] * 0.30
        + x["target_aux_core_score"] * 0.15
    )
    x["target_joint_champion_ml_score"] = (
        x.get("rank_discovery_factor_score_target", 0.0) * 0.55
        + x["target_ml_core_score"] * 0.30
        + x["target_aux_core_score"] * 0.15
    )
    x["target_joint_full_score"] = (
        x.get("rank_discovery_factor_score_target", 0.0) * 0.40
        + x["target_similarity_core_score"] * 0.25
        + x["target_ml_core_score"] * 0.20
        + x["target_aux_core_score"] * 0.15
    )
    x["target_joint_contrast_score"] = (
        x.get("rank_discovery_factor_score_target", 0.0) * 0.35
        + x.get("rank_sim_spearman_close_vol_concat_contrast_target", 0.0) * 0.30
        + x["target_ml_core_score"] * 0.20
        + x["target_aux_core_score"] * 0.15
    )
    x["target_joint_lowcross_bias_score"] = (
        x.get("rank_discovery_factor_score_target", 0.0) * 0.35
        + x["target_similarity_core_score"] * 0.20
        + x["target_ml_core_score"] * 0.20
        + x["target_aux_core_score"] * 0.10
        + x["semantic_low_cross_pullback"].astype(float) * 0.15
    )
    return x


def signal_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for horizon in [3, 5, 10, 20, 30]:
        out[f"ret_{horizon}d_mean"] = float(pd.to_numeric(df[f"ret_{horizon}d"], errors="coerce").mean()) if not df.empty else np.nan
        out[f"up_{horizon}d_rate"] = float(pd.to_numeric(df[f"up_{horizon}d"], errors="coerce").mean()) if not df.empty else np.nan
    out["sample_count"] = int(len(df))
    return out


def evaluate_strategies(df: pd.DataFrame, topn_list: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    strategy_specs = [
        ("baseline", "champion_discovery", "pool_low_cross", "discovery_factor_score"),
        ("similarity", "spearman_contrast", "pool_target_dual", "sim_spearman_close_vol_concat_contrast"),
        ("similarity", "corr_close_vol", "pool_target_dual", "sim_corr_close_vol_concat"),
        ("similarity", "cosine_close_vol", "pool_target_dual", "sim_cosine_close_vol_concat"),
        ("ml", "focused_xgb", "pool_target_confirmed", "focused_xgb_score"),
        ("ml", "focused_lgb", "pool_target_confirmed", "focused_lgb_score"),
        ("ml", "focused_et", "pool_target_confirmed", "focused_et_score"),
        ("fusion", "champion_similarity", "pool_target_confirmed", "target_joint_champion_similarity_score"),
        ("fusion", "champion_ml", "pool_target_confirmed", "target_joint_champion_ml_score"),
        ("fusion", "joint_full", "pool_target_confirmed", "target_joint_full_score"),
        ("fusion", "joint_contrast", "pool_target_confirmed", "target_joint_contrast_score"),
        ("fusion", "joint_lowcross_bias", "pool_target_strict", "target_joint_lowcross_bias_score"),
    ]
    for family, variant, pool_name, score_col in strategy_specs:
        if score_col not in df.columns or pool_name not in df.columns:
            continue
        for topn in topn_list:
            for split in ["validation", "final_test"]:
                part = df[(df["split"] == split) & (df[pool_name].fillna(False))].copy()
                if part.empty:
                    continue
                selected = (
                    part.sort_values(["signal_date", score_col], ascending=[True, False])
                    .groupby("signal_date")
                    .head(topn)
                    .copy()
                )
                rows.append(
                    {
                        "split": split,
                        "family": family,
                        "variant": variant,
                        "pool": pool_name,
                        "topn": topn,
                        "score_col": score_col,
                        **signal_metrics(selected),
                    }
                )
    return pd.DataFrame(rows)


def choose_validation_family_best(leader_df: pd.DataFrame) -> pd.DataFrame:
    val = leader_df[leader_df["split"] == "validation"].copy()
    if val.empty:
        return pd.DataFrame()
    val = val.sort_values(
        ["ret_20d_mean", "up_20d_rate", "sample_count"],
        ascending=[False, False, False],
    )
    return val.groupby("family", as_index=False).head(1).reset_index(drop=True)


def build_final_test_report(leader_df: pd.DataFrame, family_best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    final_df = leader_df[leader_df["split"] == "final_test"].copy()
    if final_df.empty or family_best.empty:
        return pd.DataFrame()
    for _, row in family_best.iterrows():
        mask = (
            (final_df["family"] == row["family"])
            & (final_df["variant"] == row["variant"])
            & (final_df["pool"] == row["pool"])
            & (final_df["topn"] == row["topn"])
        )
        hit = final_df.loc[mask]
        if hit.empty:
            continue
        rows.append(hit.iloc[0].to_dict())
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def build_selected_rows(df: pd.DataFrame, family_best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    final_df = df[df["split"] == "final_test"].copy()
    for _, row in family_best.iterrows():
        family = row["family"]
        variant = row["variant"]
        pool_name = row["pool"]
        topn = int(row["topn"])
        score_col = row["score_col"]
        part = final_df[final_df[pool_name].fillna(False)].copy()
        if part.empty:
            continue
        selected = (
            part.sort_values(["signal_date", score_col], ascending=[True, False])
            .groupby("signal_date")
            .head(topn)
            .copy()
        )
        selected["strategy_tag"] = f"{family}_{variant}_{pool_name}_top{topn}"
        rows.append(
            selected[
                [
                    "strategy_tag",
                    "code",
                    "signal_date",
                    "entry_date",
                    "entry_price",
                    "split",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["strategy_tag", "code", "signal_date", "entry_date", "entry_price", "split"])


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    topn_list = parse_topn_list(args.topn_list)

    update_progress(result_dir, "loading_candidates", file_limit=args.file_limit)
    candidate_df = load_candidate_df(args.base_signal_dir, args.file_limit)
    candidate_df = add_targeted_semantics(candidate_df)
    candidate_df.to_csv(result_dir / "candidate_rows.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "candidate_ready", candidate_count=int(len(candidate_df)))

    focused_feature_cols = [
        "discovery_factor_score",
        "sim_corr_close_vol_concat",
        "sim_cosine_close_vol_concat",
        "sim_spearman_close_vol_concat_contrast",
        "close_to_trend",
        "close_to_long",
        "trend_slope_5",
        "long_slope_5",
        "ma5_slope_5",
        "signal_vs_ma5",
        "vol_vs_prev",
        "upper_shadow_pct",
        "body_ratio",
        "dist_60d_high",
        "semantic_uptrend_pullback",
        "semantic_low_cross_pullback",
        "near_trend_pullback",
        "near_long_pullback",
        "key_k_support",
        "half_volume",
        "semi_shrink",
        "double_bull_exist_60",
        "target_confirm_bonus",
    ]
    feature_cols = [c for c in focused_feature_cols if c in candidate_df.columns]
    model_map, feature_map = fit_focused_models(candidate_df, feature_cols)
    candidate_df = score_focused_models(candidate_df, model_map, feature_map)
    update_progress(result_dir, "focused_models_ready", focused_models=list(model_map.keys()), feature_count=len(feature_cols))

    candidate_df = add_targeted_scores(candidate_df)
    candidate_df.to_csv(result_dir / "candidate_enriched.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "scores_ready", candidate_count=int(len(candidate_df)))

    leader_df = evaluate_strategies(candidate_df, topn_list)
    leader_df.to_csv(result_dir / "signal_layer_leaderboard.csv", index=False, encoding="utf-8-sig")
    family_best = choose_validation_family_best(leader_df)
    family_best.to_csv(result_dir / "validation_family_best.csv", index=False, encoding="utf-8-sig")
    final_report = build_final_test_report(leader_df, family_best)
    final_report.to_csv(result_dir / "final_test_report.csv", index=False, encoding="utf-8-sig")
    selected_rows = build_selected_rows(candidate_df, family_best)
    selected_rows.to_csv(result_dir / "final_test_selected_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "base_signal_dir": str(args.base_signal_dir),
        "result_dir": str(result_dir),
        "candidate_count": int(len(candidate_df)),
        "file_limit": int(args.file_limit),
        "topn_list": topn_list,
        "focused_models": list(model_map.keys()),
        "leaderboard_rows": int(len(leader_df)),
        "validation_family_best_rows": int(len(family_best)),
        "final_test_report_rows": int(len(final_report)),
        "selected_signal_count": int(len(selected_rows)),
        "best_validation_row": family_best.iloc[0].to_dict() if not family_best.empty else {},
        "best_final_row": final_report.iloc[0].to_dict() if not final_report.empty else {},
    }
    write_json(result_dir / "summary.json", summary)
    update_progress(result_dir, "finished", selected_signal_count=int(len(selected_rows)))


if __name__ == "__main__":
    main()
