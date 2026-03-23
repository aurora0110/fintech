from __future__ import annotations

import importlib.util
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
BASE_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
V2_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v2_20260320.py"
V4_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v4_20260320.py"
V5_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v5_20260320.py"
SEMANTIC_SCRIPT = ROOT / "utils" / "tmp" / "b1_semantic_shared_20260320.py"
DATA_DIR = ROOT / "data" / "20260315" / "normal"
FORWARD_DATA_DIR = ROOT / "data" / "forward_data"
DEFAULT_TOPN_LIST = [3, 5, 8, 10, 15, 20, 30, 50]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base_mod = load_module(BASE_SCRIPT, "b1_v6_base")
v2_mod = load_module(V2_SCRIPT, "b1_v6_v2")
v4_mod = load_module(V4_SCRIPT, "b1_v6_v4")
v5_mod = load_module(V5_SCRIPT, "b1_v6_v5")
sem_mod = load_module(SEMANTIC_SCRIPT, "b1_v6_sem")

HAS_SKLEARN = bool(getattr(v4_mod, "HAS_SKLEARN", False))
HAS_LGB = bool(getattr(v4_mod, "HAS_LGB", False))
HAS_XGB = bool(getattr(v4_mod, "HAS_XGB", False))

if HAS_SKLEARN:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 全覆盖因子/相似度/聚类/模型实验")
    parser.add_argument("--file-limit", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--topn-list", type=str, default="")
    return parser.parse_args()


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **kwargs: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    payload.update(kwargs)
    write_json(result_dir / "progress.json", payload)


def build_candidate_df(result_dir: Path, file_limit: int = 0) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    files = sorted(DATA_DIR.glob("*.txt"))
    if file_limit > 0:
        files = files[:file_limit]
    total = len(files)
    for i, path in enumerate(files, 1):
        stock_rows = sem_mod.build_semantic_candidates_for_one_stock(str(path))
        if stock_rows:
            rows.extend(stock_rows)
        if i % 100 == 0 or i == total:
            print(f"v6 候选池构建进度: {i}/{total}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["signal_date", "code"]).reset_index(drop=True)
    df.to_csv(result_dir / "candidate_rows.csv", index=False, encoding="utf-8-sig")
    df.to_pickle(result_dir / "candidate_rows.pkl")
    return df


def build_positive_df(result_dir: Path) -> pd.DataFrame:
    raw_cases = v2_mod.parse_b1_case_files()
    mapping = v2_mod.build_name_code_map()
    enriched = pd.DataFrame([sem_mod.enrich_case_with_semantics(r, mapping) for _, r in raw_cases.iterrows()])
    enriched.to_csv(result_dir / "perfect_positive_manifest.csv", index=False, encoding="utf-8-sig")
    return enriched[enriched["status"] == "ok"].copy().reset_index(drop=True)


def broad_feature_cols() -> List[str]:
    return [
        "J",
        "ret1",
        "ret3",
        "ret5",
        "ret10",
        "ret20",
        "ret30",
        "signal_ret",
        "trend_spread",
        "close_to_trend",
        "close_to_long",
        "signal_vs_ma5",
        "vol_vs_prev",
        "ret_std_5",
        "ret_std_10",
        "ret_std_20",
        "ret_skew_10",
        "ret_skew_20",
        "ret_kurt_20",
        "price_skew_20",
        "atr14_pct",
        "dist_20d_high",
        "dist_20d_low",
        "dist_60d_high",
        "dist_60d_low",
        "range_pos_20",
        "range_pos_60",
        "volume_z20",
        "volume_rank20",
        "up_count_5",
        "down_count_5",
        "up_count_10",
        "down_count_10",
        "ma_bull_alignment",
        "ma_bear_alignment",
        "rsi14",
        "body_ratio",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "close_location",
        "ma5_slope_5",
        "ma10_slope_5",
        "ma20_slope_5",
        "trend_slope_5",
        "long_slope_5",
        "near_trend_pullback",
        "near_long_pullback",
        "pullback_any",
        "recent_low_level_context_20",
        "cross_up_event",
        "bars_since_cross_up",
        "first_pullback_after_cross",
        "half_volume",
        "semi_shrink",
        "double_bull_exist_60",
        "above_double_bull_close",
        "above_double_bull_high",
        "key_k_support",
        "recent_heavy_bear_top_20",
        "recent_failed_breakout_20",
        "top_distribution_20",
        "recent_stair_bear_20",
        "risk_distribution_any_20",
        "semantic_uptrend_pullback",
        "semantic_low_cross_pullback",
        "semantic_confirmed",
        "semantic_strict",
        "semantic_trend_focus",
        "semantic_long_focus",
        "buy_semantic_score",
    ]


def available_feature_cols(*dfs: pd.DataFrame) -> List[str]:
    cols = broad_feature_cols()
    for df in dfs:
        cols = [c for c in cols if c in df.columns]
    return cols


def model_feature_cols(include_similarity: bool) -> List[str]:
    cols = broad_feature_cols().copy()
    if include_similarity:
        cols.extend(
            [
                "sim_corr_close_vol_concat",
                "sim_cosine_close_vol_concat",
                "sim_euclidean_close_vol_concat",
                "sim_spearman_close_vol_concat",
                "sim_weighted_corr_close_vol_concat",
                "sim_lag_corr_close_vol_concat",
                "sim_derivative_corr_close_norm",
                "sim_derivative_corr_returns",
                "sim_corr_close_norm",
                "sim_cosine_close_norm",
                "sim_corr_returns",
                "sim_cosine_returns",
                "sim_contrast_corr_close_vol_concat",
                "sim_contrast_cosine_close_vol_concat",
                "habit_profile_score",
                "habit_rule_score",
                "semantic_profile_score",
                "semantic_rule_score",
                "cluster_centroid_score",
                "cluster_knn_score",
                "cluster_gmm_score",
                "discovery_factor_score",
                "discovery_rank_score",
                "full_fusion_score",
            ]
        )
    return cols


def build_feature_matrix(
    df: pd.DataFrame, include_similarity: bool = False, force_cols: List[str] | None = None
) -> Tuple[np.ndarray, List[str]]:
    cols = force_cols if force_cols is not None else [c for c in model_feature_cols(include_similarity) if c in df.columns]
    arr = df[cols].fillna(0.0).to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, cols


def sanitize_numeric_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # 防止极端异常值把聚类和树模型打炸
    arr = np.clip(arr, -1e6, 1e6)
    return arr


def add_extended_similarity(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    candidate_df = v4_mod.add_similarity_family(candidate_df, pos_df, neg_df)
    extra_specs = [
        ("sim_spearman_close_vol_concat", "close_vol_concat", "spearman"),
        ("sim_corr_close_norm", "close_norm", "corr"),
        ("sim_cosine_close_norm", "close_norm", "cosine"),
        ("sim_corr_returns", "returns", "corr"),
        ("sim_cosine_returns", "returns", "cosine"),
        ("sim_derivative_corr_close_norm", "deriv_close_norm", "derivative_corr"),
        ("sim_derivative_corr_returns", "deriv_returns", "derivative_corr"),
    ]
    cand_maps = [v4_mod.derive_rep_map(r["seq_map"]) for _, r in candidate_df.iterrows()]
    pos_maps = [v4_mod.derive_rep_map(r["seq_map"]) for _, r in pos_df.iterrows()]
    neg_maps = [v4_mod.derive_rep_map(r["seq_map"]) for _, r in neg_df.iterrows()] if not neg_df.empty else []
    for out_col, repr_name, scorer_name in extra_specs:
        seqs = [m[repr_name] for m in cand_maps]
        pos_tpls = [m[repr_name] for m in pos_maps]
        candidate_df[out_col] = v4_mod.compute_similarity_column(seqs, pos_tpls, scorer_name)
        if neg_maps:
            neg_tpls = [m[repr_name] for m in neg_maps]
            candidate_df[f"{out_col}_contrast"] = candidate_df[out_col] - v4_mod.compute_similarity_column(seqs, neg_tpls, scorer_name)
    return candidate_df


def add_pool_flags(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if "semantic_candidate" not in candidate_df.columns:
        candidate_df["semantic_candidate"] = candidate_df.get("semantic_base", False)
    candidate_df["semantic_candidate"] = candidate_df["semantic_candidate"].fillna(False).astype(bool)
    candidate_df["pool_all"] = candidate_df["semantic_candidate"]
    candidate_df["pool_pullback"] = candidate_df["semantic_base"]
    candidate_df["pool_uptrend"] = candidate_df["semantic_uptrend_pullback"]
    candidate_df["pool_low_cross"] = candidate_df["semantic_low_cross_pullback"]
    candidate_df["pool_confirmed"] = candidate_df["semantic_confirmed"]
    candidate_df["pool_trend_focus"] = candidate_df["semantic_trend_focus"]
    candidate_df["pool_strict"] = candidate_df["semantic_strict"]
    candidate_df["pool_shrink"] = candidate_df["semantic_candidate"] & candidate_df["semi_shrink"]
    candidate_df["pool_near_trend"] = candidate_df["semantic_candidate"] & candidate_df["near_trend_pullback"]
    candidate_df["pool_near_long"] = candidate_df["semantic_candidate"] & candidate_df["near_long_pullback"]
    candidate_df["pool_no_risk"] = candidate_df["semantic_candidate"] & (~candidate_df["risk_distribution_any_20"])
    candidate_df["pool_core_plus"] = (
        candidate_df["semantic_candidate"]
        & candidate_df["pullback_any"]
        & candidate_df["semi_shrink"]
        & (~candidate_df["risk_distribution_any_20"])
    )
    return candidate_df


def filter_forward_replayable(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "code" not in df.columns:
        return df.copy()
    out = df.copy()
    out["has_forward_data"] = out["code"].astype(str).map(lambda c: (FORWARD_DATA_DIR / f"{c}.txt").exists())
    return out[out["has_forward_data"]].copy()


def add_semantic_scores(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    candidate_df = v4_mod.add_habit_scores(candidate_df, pos_df, neg_df)
    candidate_df = v5_mod.add_semantic_habit_scores(candidate_df, pos_df, neg_df)
    return candidate_df


def add_cluster_scores(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, result_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_SKLEARN or len(pos_df) < 8:
        candidate_df["cluster_centroid_score"] = 0.0
        candidate_df["cluster_knn_score"] = 0.0
        candidate_df["cluster_gmm_score"] = 0.0
        return candidate_df, pd.DataFrame()

    feature_cols = available_feature_cols(candidate_df, pos_df)
    pos_arr = sanitize_numeric_matrix(pos_df[feature_cols].fillna(0.0).to_numpy(dtype=float))
    cand_arr = sanitize_numeric_matrix(candidate_df[feature_cols].fillna(0.0).to_numpy(dtype=float))
    scaler = StandardScaler()
    pos_scaled = scaler.fit_transform(pos_arr)
    pos_scaled = sanitize_numeric_matrix(pos_scaled)
    cand_scaled = sanitize_numeric_matrix(scaler.transform(cand_arr))

    k_candidates = [k for k in [2, 3, 4, 5, 6] if k < len(pos_df)]
    best_k = k_candidates[0]
    best_inertia = None
    best_model = None
    for k in k_candidates:
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        model.fit(pos_scaled)
        if best_inertia is None or model.inertia_ < best_inertia:
            best_inertia = float(model.inertia_)
            best_k = k
            best_model = model
    assert best_model is not None

    centroid_dist = ((cand_scaled[:, None, :] - best_model.cluster_centers_[None, :, :]) ** 2).sum(axis=2) ** 0.5
    candidate_df["cluster_centroid_score"] = -centroid_dist.min(axis=1)

    k_knn = min(5, len(pos_scaled))
    knn_dist = np.sort(((cand_scaled[:, None, :] - pos_scaled[None, :, :]) ** 2).sum(axis=2) ** 0.5, axis=1)[:, :k_knn]
    candidate_df["cluster_knn_score"] = -knn_dist.mean(axis=1)

    gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)
    gmm.fit(pos_scaled)
    candidate_df["cluster_gmm_score"] = gmm.score_samples(cand_scaled)

    cluster_summary = (
        pd.DataFrame({"cluster": best_model.labels_})
        .groupby("cluster")
        .size()
        .rename("sample_count")
        .reset_index()
    )
    cluster_summary["best_k"] = best_k
    cluster_summary.to_csv(result_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    return candidate_df, cluster_summary


def add_feature_discovery_scores(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame, result_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = available_feature_cols(candidate_df, pos_df, neg_df)
    train_df = pd.concat([pos_df.assign(label=1), neg_df.assign(label=0)], ignore_index=True)
    X = sanitize_numeric_matrix(train_df[feature_cols].fillna(0.0).to_numpy(dtype=float))
    y = train_df["label"].to_numpy(dtype=int)
    X_all = sanitize_numeric_matrix(candidate_df[feature_cols].fillna(0.0).to_numpy(dtype=float))

    rows: List[Dict[str, Any]] = []
    signed_effects: Dict[str, float] = {}
    pos_med = pos_df[feature_cols].fillna(0.0).median()
    neg_med = neg_df[feature_cols].fillna(0.0).median()
    ref_std = train_df[feature_cols].fillna(0.0).std().replace(0, np.nan)

    for col in feature_cols:
        delta = float(pos_med[col] - neg_med[col])
        scaled_delta = delta / float(ref_std[col]) if pd.notna(ref_std[col]) and ref_std[col] != 0 else 0.0
        signed_effects[col] = scaled_delta
        rows.append({"feature": col, "median_delta": delta, "scaled_delta": scaled_delta})

    if HAS_SKLEARN:
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)
        et = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        et.fit(X, y)
        for i, col in enumerate(feature_cols):
            rows[i]["mutual_info"] = float(mi[i])
            rows[i]["rf_importance"] = float(rf.feature_importances_[i])
            rows[i]["et_importance"] = float(et.feature_importances_[i])

    if HAS_LGB:
        model = v4_mod.lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="binary",
        )
        model.fit(X, y)
        imp = model.feature_importances_.astype(float)
        for i, col in enumerate(feature_cols):
            rows[i]["lgb_importance"] = float(imp[i])

    if HAS_XGB:
        model = v4_mod.xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            eval_metric="logloss",
        )
        model.fit(X, y)
        imp = getattr(model, "feature_importances_", np.zeros(len(feature_cols), dtype=float))
        for i, col in enumerate(feature_cols):
            rows[i]["xgb_importance"] = float(imp[i])

    factor_df = pd.DataFrame(rows).fillna(0.0)
    rank_cols = [c for c in factor_df.columns if c.endswith("importance") or c == "mutual_info"]
    for col in rank_cols:
        factor_df[f"rank_{col}"] = factor_df[col].rank(ascending=False, method="average")
    factor_df["importance_rank_mean"] = factor_df[[f"rank_{c}" for c in rank_cols]].mean(axis=1) if rank_cols else 0.0
    factor_df["importance_strength"] = 1.0 / factor_df["importance_rank_mean"].replace(0, np.nan)
    factor_df["signed_strength"] = factor_df["importance_strength"].fillna(0.0) * factor_df["scaled_delta"]
    factor_df = factor_df.sort_values(["importance_rank_mean", "feature"]).reset_index(drop=True)
    factor_df.to_csv(result_dir / "feature_discovery_report.csv", index=False, encoding="utf-8-sig")

    top_factors = factor_df.head(20).copy()
    contribution = np.zeros(len(candidate_df), dtype=float)
    rank_contrib = np.zeros(len(candidate_df), dtype=float)
    for _, row in top_factors.iterrows():
        col = str(row["feature"])
        signed = float(row["signed_strength"])
        strength = float(row["importance_strength"]) if np.isfinite(row["importance_strength"]) else 0.0
        values = candidate_df[col].fillna(0.0).to_numpy(dtype=float)
        z = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-9)
        contribution += z * signed
        day_rank = candidate_df.groupby("signal_date")[col].rank(pct=True, ascending=(signed < 0))
        rank_contrib += day_rank.fillna(0.0).to_numpy(dtype=float) * strength
    candidate_df["discovery_factor_score"] = contribution
    candidate_df["discovery_rank_score"] = rank_contrib
    return candidate_df, factor_df


def add_rank_fusions(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rank_inputs = [
        "sim_corr_close_vol_concat",
        "sim_cosine_close_vol_concat",
        "sim_euclidean_close_vol_concat",
        "sim_spearman_close_vol_concat",
        "sim_weighted_corr_close_vol_concat",
        "sim_lag_corr_close_vol_concat",
        "sim_derivative_corr_close_norm",
        "sim_derivative_corr_returns",
        "sim_contrast_corr_close_vol_concat",
        "sim_contrast_cosine_close_vol_concat",
        "habit_profile_score",
        "habit_rule_score",
        "semantic_profile_score",
        "semantic_rule_score",
        "buy_semantic_score",
        "cluster_centroid_score",
        "cluster_knn_score",
        "cluster_gmm_score",
        "discovery_factor_score",
        "discovery_rank_score",
    ]
    for col in rank_inputs:
        if col in candidate_df.columns:
            candidate_df[f"rank_{col}"] = candidate_df.groupby("signal_date")[col].rank(pct=True, ascending=True)
    candidate_df["similarity_cluster_fusion_score"] = (
        0.28 * candidate_df.get("rank_sim_corr_close_vol_concat", 0)
        + 0.18 * candidate_df.get("rank_sim_cosine_close_vol_concat", 0)
        + 0.12 * candidate_df.get("rank_sim_spearman_close_vol_concat", 0)
        + 0.12 * candidate_df.get("rank_cluster_centroid_score", 0)
        + 0.15 * candidate_df.get("rank_cluster_knn_score", 0)
        + 0.15 * candidate_df.get("rank_cluster_gmm_score", 0)
    )
    candidate_df["semantic_discovery_fusion_score"] = (
        0.25 * candidate_df.get("rank_semantic_rule_score", 0)
        + 0.15 * candidate_df.get("rank_semantic_profile_score", 0)
        + 0.20 * candidate_df.get("rank_discovery_factor_score", 0)
        + 0.15 * candidate_df.get("rank_discovery_rank_score", 0)
        + 0.15 * candidate_df.get("rank_habit_profile_score", 0)
        + 0.10 * candidate_df.get("rank_buy_semantic_score", 0)
    )
    candidate_df["full_fusion_score"] = (
        0.35 * candidate_df.get("similarity_cluster_fusion_score", 0)
        + 0.25 * candidate_df.get("semantic_discovery_fusion_score", 0)
        + 0.20 * candidate_df.get("rank_sim_weighted_corr_close_vol_concat", 0)
        + 0.10 * candidate_df.get("rank_sim_lag_corr_close_vol_concat", 0)
        + 0.05 * candidate_df.get("rank_sim_derivative_corr_close_norm", 0)
        + 0.05 * candidate_df.get("rank_sim_derivative_corr_returns", 0)
    )
    return candidate_df


def build_ml_scores(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    train_df = pd.concat([pos_df.assign(label=1), neg_df.assign(label=0)], ignore_index=True)
    plain_cols = [c for c in model_feature_cols(False) if c in train_df.columns and c in candidate_df.columns]
    mix_cols = [c for c in model_feature_cols(True) if c in train_df.columns and c in candidate_df.columns]
    X_plain, _ = build_feature_matrix(train_df, include_similarity=False, force_cols=plain_cols)
    X_mix, _ = build_feature_matrix(train_df, include_similarity=True, force_cols=mix_cols)
    y = train_df["label"].to_numpy(dtype=int)
    Xp_all, _ = build_feature_matrix(candidate_df, include_similarity=False, force_cols=plain_cols)
    Xm_all, _ = build_feature_matrix(candidate_df, include_similarity=True, force_cols=mix_cols)
    trained: List[str] = []

    gnb = v4_mod.fit_gaussian_nb(X_plain, y)
    candidate_df["gnb_full_score"] = v4_mod.predict_gaussian_nb(gnb, Xp_all)
    trained.append("gnb_full_score")

    if HAS_SKLEARN:
        lr = v4_mod.LogisticRegression(max_iter=500, class_weight="balanced")
        lr.fit(X_mix, y)
        candidate_df["logistic_full_score"] = lr.predict_proba(Xm_all)[:, 1]
        trained.append("logistic_full_score")

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_mix, y)
        candidate_df["rf_full_score"] = rf.predict_proba(Xm_all)[:, 1]
        trained.append("rf_full_score")

        et = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        et.fit(X_mix, y)
        candidate_df["et_full_score"] = et.predict_proba(Xm_all)[:, 1]
        trained.append("et_full_score")

        hgb = v4_mod.HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=250,
            min_samples_leaf=20,
            random_state=42,
        )
        hgb.fit(X_mix, y)
        candidate_df["hgb_full_score"] = hgb.predict_proba(Xm_all)[:, 1]
        trained.append("hgb_full_score")

    if HAS_LGB:
        model = v4_mod.lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="binary",
        )
        model.fit(X_mix, y)
        candidate_df["lgb_full_score"] = model.predict_proba(Xm_all)[:, 1]
        trained.append("lgb_full_score")

    if HAS_XGB:
        model = v4_mod.xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            eval_metric="logloss",
        )
        model.fit(X_mix, y)
        candidate_df["xgb_full_score"] = model.predict_proba(Xm_all)[:, 1]
        trained.append("xgb_full_score")

    return candidate_df, trained


def summarize_factor_contribution(pos_df: pd.DataFrame, neg_df: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in available_feature_cols(pos_df, neg_df):
        pos_s = pd.to_numeric(pos_df[col], errors="coerce")
        neg_s = pd.to_numeric(neg_df[col], errors="coerce")
        rows.append(
            {
                "factor": col,
                "positive_mean": float(pos_s.mean()) if len(pos_s) else np.nan,
                "negative_mean": float(neg_s.mean()) if len(neg_s) else np.nan,
                "mean_delta": float(pos_s.mean() - neg_s.mean()) if len(pos_s) and len(neg_s) else np.nan,
                "positive_median": float(pos_s.median()) if len(pos_s) else np.nan,
                "negative_median": float(neg_s.median()) if len(neg_s) else np.nan,
                "median_delta": float(pos_s.median() - neg_s.median()) if len(pos_s) and len(neg_s) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("median_delta", ascending=False).reset_index(drop=True)
    out.to_csv(result_dir / "factor_contribution.csv", index=False, encoding="utf-8-sig")
    return out


def evaluate_daily_topn(part: pd.DataFrame, score_col: str, ascending: bool, topn: int) -> Dict[str, Any]:
    selected = v4_mod.select_daily_topn(part, score_col=score_col, topn=topn, ascending=ascending)
    out = {"sample_count": int(len(selected)), "date_count": int(selected["signal_date"].nunique()) if not selected.empty else 0}
    for horizon in [3, 5, 10, 20, 30]:
        out[f"ret_{horizon}d_mean"] = float(selected[f"ret_{horizon}d"].mean()) if not selected.empty else np.nan
        out[f"up_{horizon}d_rate"] = float(selected[f"up_{horizon}d"].mean()) if not selected.empty else np.nan
    return out


def main() -> None:
    args = parse_args()
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.result_dir or (ROOT / "results" / f"b1_full_factor_signal_v6_{run_ts}")
    result_dir.mkdir(parents=True, exist_ok=True)
    topn_list = DEFAULT_TOPN_LIST if not args.topn_list else [int(x) for x in args.topn_list.split(",") if x.strip()]

    update_progress(result_dir, "starting", file_limit=int(args.file_limit), topn_list=topn_list)
    candidate_df = build_candidate_df(result_dir, file_limit=int(args.file_limit))
    positive_df = build_positive_df(result_dir)
    if positive_df.empty:
        raise ValueError("完美图/B1 没有可用正样本")

    cutoffs = v5_mod.split_three_way_by_positive_dates(positive_df["signal_date"].tolist())
    candidate_df["split"] = v5_mod.assign_split(candidate_df, cutoffs)
    positive_df["split"] = v5_mod.assign_split(positive_df.rename(columns={"signal_date": "signal_date"}), cutoffs)
    write_json(result_dir / "split_cutoffs.json", {k: str(v.date()) for k, v in cutoffs.items()})
    update_progress(result_dir, "inputs_ready", candidate_count=int(len(candidate_df)), positive_count=int(len(positive_df)))

    research_positive = positive_df[positive_df["split"] == "research"].copy()
    research_negative = candidate_df[(candidate_df["split"] == "research") & (candidate_df["negative_30d"])].copy()
    max_negative = max(len(research_positive) * 5, len(research_positive))
    research_negative = research_negative.sort_values("signal_date").head(max_negative).copy()
    if research_positive.empty:
        raise ValueError("research 段没有完美图/B1 正样本")

    summarize_factor_contribution(research_positive, research_negative, result_dir)
    update_progress(result_dir, "factor_contribution_ready")

    candidate_df = add_extended_similarity(candidate_df, research_positive, research_negative)
    candidate_df = add_semantic_scores(candidate_df, research_positive, research_negative)
    candidate_df = add_pool_flags(candidate_df)
    candidate_df, _ = add_cluster_scores(candidate_df, research_positive, result_dir)
    candidate_df, _ = add_feature_discovery_scores(candidate_df, research_positive, research_negative, result_dir)
    candidate_df = add_rank_fusions(candidate_df)
    candidate_df, trained_models = build_ml_scores(candidate_df, research_positive, research_negative)
    update_progress(result_dir, "models_ready", trained_models=trained_models)

    candidate_df.to_csv(result_dir / "candidate_enriched.csv", index=False, encoding="utf-8-sig")

    pool_names = [
        "pool_all",
        "pool_pullback",
        "pool_uptrend",
        "pool_low_cross",
        "pool_confirmed",
        "pool_trend_focus",
        "pool_strict",
        "pool_shrink",
        "pool_near_trend",
        "pool_near_long",
        "pool_no_risk",
        "pool_core_plus",
    ]
    pool_rows: List[Dict[str, Any]] = []
    for pool_name in pool_names:
        for split_name in ["validation", "final_test"]:
            part = candidate_df[(candidate_df["split"] == split_name) & (candidate_df[pool_name])].copy()
            pool_rows.append(
                {
                    "pool": pool_name,
                    "split": split_name,
                    "sample_count": int(len(part)),
                    "date_count": int(part["signal_date"].nunique()) if not part.empty else 0,
                    "ret_20d_mean": float(part["ret_20d"].mean()) if len(part) else np.nan,
                    "up_20d_rate": float(part["up_20d"].mean()) if len(part) else np.nan,
                }
            )
    pd.DataFrame(pool_rows).to_csv(result_dir / "pool_summary.csv", index=False, encoding="utf-8-sig")

    score_specs = [("baseline", "lowest_J", "J", True)]
    score_specs.extend(
        [("similarity", c.replace("sim_", ""), c, False) for c in candidate_df.columns if c.startswith("sim_") and not c.endswith("_contrast")]
    )
    score_specs.extend(
        [
            ("contrast", c.replace("sim_", ""), c, False)
            for c in candidate_df.columns
            if c.startswith("sim_") and c.endswith("_contrast")
        ]
    )
    for family, variant, col in [
        ("habit", "habit_profile", "habit_profile_score"),
        ("habit", "habit_rule", "habit_rule_score"),
        ("semantic", "semantic_profile", "semantic_profile_score"),
        ("semantic", "semantic_rule", "semantic_rule_score"),
        ("cluster", "centroid", "cluster_centroid_score"),
        ("cluster", "knn", "cluster_knn_score"),
        ("cluster", "gmm", "cluster_gmm_score"),
        ("factor", "discovery_factor", "discovery_factor_score"),
        ("factor", "discovery_rank", "discovery_rank_score"),
        ("fusion", "similarity_cluster", "similarity_cluster_fusion_score"),
        ("fusion", "semantic_discovery", "semantic_discovery_fusion_score"),
        ("fusion", "full", "full_fusion_score"),
    ]:
        if col in candidate_df.columns:
            score_specs.append((family, variant, col, False))
    score_specs.extend(
        [("ml", c.replace("_score", ""), c, False) for c in candidate_df.columns if c.endswith("_full_score")]
    )

    leaderboard: List[Dict[str, Any]] = []
    for split_name in ["validation", "final_test"]:
        split_df = candidate_df[candidate_df["split"] == split_name].copy()
        for pool_name in pool_names:
            part = split_df[split_df[pool_name]].copy()
            if part.empty:
                continue
            base_row = {"family": "baseline", "variant": "all_candidates", "pool": pool_name, "topn": 0, "split": split_name}
            base_row.update(evaluate_daily_topn(part, "J", True, topn=max(len(part), 1)))
            leaderboard.append(base_row)
            for family, variant, col, ascending in score_specs:
                if col not in part.columns:
                    continue
                for topn in topn_list:
                    row = {"family": family, "variant": variant, "pool": pool_name, "topn": topn, "split": split_name}
                    row.update(evaluate_daily_topn(part, col, ascending, topn=topn))
                    leaderboard.append(row)
    leader_df = pd.DataFrame(leaderboard).sort_values(
        ["split", "ret_20d_mean", "up_20d_rate", "sample_count"],
        ascending=[True, False, False, False],
    )
    leader_df.to_csv(result_dir / "signal_layer_leaderboard.csv", index=False, encoding="utf-8-sig")

    validation_best = (
        leader_df[leader_df["split"] == "validation"]
        .sort_values(["family", "ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[True, False, False, False])
        .drop_duplicates(subset=["family"], keep="first")
        .reset_index(drop=True)
    )
    validation_best.to_csv(result_dir / "validation_family_best.csv", index=False, encoding="utf-8-sig")

    final_rows: List[Dict[str, Any]] = []
    final_df = candidate_df[candidate_df["split"] == "final_test"].copy()
    selected_rows: List[pd.DataFrame] = []
    for _, best in validation_best.iterrows():
        family = str(best["family"])
        variant = str(best["variant"])
        pool_name = str(best["pool"])
        topn = int(best["topn"])
        if topn <= 0:
            continue
        part = final_df[final_df[pool_name]].copy()
        if part.empty:
            continue
        score_col = "J" if family == "baseline" else next(
            c for fam, var, c, _ in score_specs if fam == family and var == variant
        )
        ascending = family == "baseline"
        selected = v4_mod.select_daily_topn(part, score_col=score_col, topn=topn, ascending=ascending)
        selected = filter_forward_replayable(selected)
        report = {"family": family, "variant": variant, "pool": pool_name, "topn": topn}
        for horizon in [3, 5, 10, 20, 30]:
            report[f"ret_{horizon}d_mean"] = float(selected[f"ret_{horizon}d"].mean()) if not selected.empty else np.nan
            report[f"up_{horizon}d_rate"] = float(selected[f"up_{horizon}d"].mean()) if not selected.empty else np.nan
        report["sample_count"] = int(len(selected))
        final_rows.append(report)
        out = selected.copy()
        out["strategy_tag"] = f"{family}_{variant}_{pool_name}_top{topn}"
        selected_rows.append(out)
    final_report = pd.DataFrame(final_rows).sort_values(["ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[False, False, False])
    final_report.to_csv(result_dir / "final_test_report.csv", index=False, encoding="utf-8-sig")
    if selected_rows:
        pd.concat(selected_rows, ignore_index=True).to_csv(
            result_dir / "final_test_selected_rows.csv", index=False, encoding="utf-8-sig"
        )

    summary = {
        "candidate_count": int(len(candidate_df)),
        "positive_count": int(len(positive_df)),
        "research_positive_count": int(len(research_positive)),
        "research_negative_count": int(len(research_negative)),
        "pool_count": len(pool_names),
        "score_spec_count": len(score_specs),
        "leaderboard_rows": int(len(leader_df)),
        "validation_best_rows": int(len(validation_best)),
        "final_report_rows": int(len(final_report)),
    }
    write_json(result_dir / "summary.json", summary)
    update_progress(result_dir, "finished", **summary)


if __name__ == "__main__":
    main()
