from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
BASE_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
V2_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v2_20260320.py"
V4_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v4_20260320.py"
SEMANTIC_SCRIPT = ROOT / "utils" / "tmp" / "b1_semantic_shared_20260320.py"
DATA_DIR = ROOT / "data" / "20260315" / "normal"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = ROOT / "results" / f"b1_similarity_ml_signal_v5_{RUN_TS}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_TOPN_LIST = [3, 5, 8, 10, 20, 50]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base_mod = load_module(BASE_SCRIPT, "b1_v5_base")
v2_mod = load_module(V2_SCRIPT, "b1_v5_v2")
v4_mod = load_module(V4_SCRIPT, "b1_v5_v4")
sem_mod = load_module(SEMANTIC_SCRIPT, "b1_v5_sem")

HAS_SKLEARN = bool(getattr(v4_mod, "HAS_SKLEARN", False))
HAS_LGB = bool(getattr(v4_mod, "HAS_LGB", False))
HAS_XGB = bool(getattr(v4_mod, "HAS_XGB", False))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(stage: str, **kwargs: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    payload.update(kwargs)
    write_json(RESULT_DIR / "progress.json", payload)


def split_three_way_by_positive_dates(dates: List[pd.Timestamp]) -> Dict[str, pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(pd.Series(dates)).drop_duplicates())
    n = len(unique_dates)
    research_end = min(max(1, int(n * 0.60)), n - 2)
    validation_end = min(max(research_end + 1, int(n * 0.80)), n - 1)
    return {
        "research_end": unique_dates[research_end - 1],
        "validation_start": unique_dates[research_end],
        "validation_end": unique_dates[validation_end - 1],
        "final_start": unique_dates[validation_end],
        "final_end": unique_dates[-1],
    }


def assign_split(df: pd.DataFrame, cutoffs: Dict[str, pd.Timestamp]) -> pd.Series:
    date_s = pd.to_datetime(df["signal_date"])
    out = pd.Series("", index=df.index, dtype="object")
    out[date_s <= cutoffs["research_end"]] = "research"
    out[(date_s >= cutoffs["validation_start"]) & (date_s <= cutoffs["validation_end"])] = "validation"
    out[date_s >= cutoffs["final_start"]] = "final_test"
    return out


def build_candidate_df() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    files = sorted(DATA_DIR.glob("*.txt"))
    total = len(files)
    for i, path in enumerate(files, 1):
        stock_rows = sem_mod.build_semantic_candidates_for_one_stock(str(path))
        if stock_rows:
            rows.extend(stock_rows)
        if i % 100 == 0 or i == total:
            print(f"语义B1候选池构建进度: {i}/{total}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["signal_date", "code"]).reset_index(drop=True)
    df.to_csv(RESULT_DIR / "candidate_rows.csv", index=False, encoding="utf-8-sig")
    df.to_pickle(RESULT_DIR / "candidate_rows.pkl")
    return df


def build_positive_df() -> pd.DataFrame:
    raw_cases = v2_mod.parse_b1_case_files()
    mapping = v2_mod.build_name_code_map()
    enriched = pd.DataFrame([sem_mod.enrich_case_with_semantics(r, mapping) for _, r in raw_cases.iterrows()])
    enriched.to_csv(RESULT_DIR / "perfect_positive_manifest.csv", index=False, encoding="utf-8-sig")
    ok = enriched[enriched["status"] == "ok"].copy().reset_index(drop=True)
    return ok


def add_semantic_habit_scores(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = [
        "near_trend_pullback",
        "near_long_pullback",
        "recent_low_level_context_20",
        "first_pullback_after_cross",
        "half_volume",
        "semi_shrink",
        "double_bull_exist_60",
        "above_double_bull_close",
        "above_double_bull_high",
        "key_k_support",
        "semantic_uptrend_pullback",
        "semantic_low_cross_pullback",
        "semantic_confirmed",
        "semantic_strict",
        "semantic_trend_focus",
        "semantic_long_focus",
    ]
    pos_rate = pos_df[bool_cols].fillna(False).astype(bool).mean()
    neg_rate = neg_df[bool_cols].fillna(False).astype(bool).mean()
    deltas = (pos_rate - neg_rate).sort_values(ascending=False)
    score = pd.Series(0.0, index=candidate_df.index)
    for col, delta in deltas.items():
        if abs(delta) < 0.02:
            continue
        if delta > 0:
            score += candidate_df[col].fillna(False).astype(bool).astype(float) * float(delta)
        else:
            score += (~candidate_df[col].fillna(False).astype(bool)).astype(float) * float(-delta)
    candidate_df["semantic_profile_score"] = score
    candidate_df["semantic_rule_score"] = (
        candidate_df["near_trend_pullback"].astype(int)
        + candidate_df["near_long_pullback"].astype(int)
        + candidate_df["half_volume"].astype(int)
        + candidate_df["double_bull_exist_60"].astype(int)
        + candidate_df["key_k_support"].astype(int)
        + candidate_df["semantic_uptrend_pullback"].astype(int)
        + candidate_df["semantic_low_cross_pullback"].astype(int) * 2
        + candidate_df["semantic_confirmed"].astype(int)
        + candidate_df["semantic_strict"].astype(int) * 2
        - candidate_df["risk_distribution_any_20"].astype(int) * 3
    ).astype(float)
    return candidate_df


def add_pool_flags(candidate_df: pd.DataFrame) -> pd.DataFrame:
    candidate_df["pool_all"] = candidate_df["semantic_candidate"]
    candidate_df["pool_pullback"] = candidate_df["semantic_base"]
    candidate_df["pool_uptrend"] = candidate_df["semantic_uptrend_pullback"]
    candidate_df["pool_low_cross"] = candidate_df["semantic_low_cross_pullback"]
    candidate_df["pool_confirmed"] = candidate_df["semantic_confirmed"]
    candidate_df["pool_trend_focus"] = candidate_df["semantic_trend_focus"]
    candidate_df["pool_strict"] = candidate_df["semantic_strict"]
    return candidate_df


def add_rank_fusions(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rank_inputs = [
        "sim_corr_close_vol_concat",
        "sim_cosine_close_vol_concat",
        "sim_euclidean_close_vol_concat",
        "sim_contrast_corr_close_vol_concat",
        "sim_contrast_cosine_close_vol_concat",
        "sim_weighted_corr_close_vol_concat",
        "sim_lag_corr_close_vol_concat",
        "habit_profile_score",
        "habit_rule_score",
        "semantic_profile_score",
        "semantic_rule_score",
        "buy_semantic_score",
    ]
    for col in rank_inputs:
        if col not in candidate_df.columns:
            continue
        candidate_df[f"rank_{col}"] = candidate_df.groupby("signal_date")[col].rank(pct=True, ascending=True)

    candidate_df["semantic_fusion_score"] = (
        0.35 * candidate_df.get("rank_sim_corr_close_vol_concat", 0)
        + 0.20 * candidate_df.get("rank_sim_cosine_close_vol_concat", 0)
        + 0.15 * candidate_df.get("rank_sim_contrast_corr_close_vol_concat", 0)
        + 0.15 * candidate_df.get("rank_semantic_rule_score", 0)
        + 0.15 * candidate_df.get("rank_buy_semantic_score", 0)
    )
    candidate_df["semantic_habit_fusion_score"] = (
        0.30 * candidate_df.get("rank_semantic_profile_score", 0)
        + 0.25 * candidate_df.get("rank_habit_profile_score", 0)
        + 0.20 * candidate_df.get("rank_habit_rule_score", 0)
        + 0.15 * candidate_df.get("rank_sim_weighted_corr_close_vol_concat", 0)
        + 0.10 * candidate_df.get("rank_buy_semantic_score", 0)
    )
    candidate_df["master_fusion_score"] = (
        0.35 * candidate_df.get("semantic_fusion_score", 0)
        + 0.30 * candidate_df.get("semantic_habit_fusion_score", 0)
        + 0.20 * candidate_df.get("rank_sim_euclidean_close_vol_concat", 0)
        + 0.15 * candidate_df.get("rank_sim_lag_corr_close_vol_concat", 0)
    )
    return candidate_df


def build_feature_matrix(df: pd.DataFrame, include_similarity: bool) -> Tuple[np.ndarray, List[str]]:
    cols = [
        "J",
        "ret1",
        "ret3",
        "ret5",
        "ret10",
        "signal_ret",
        "trend_spread",
        "close_to_trend",
        "close_to_long",
        "signal_vs_ma5",
        "vol_vs_prev",
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
        "recent_low_level_context_20",
        "first_pullback_after_cross",
        "half_volume",
        "semi_shrink",
        "double_bull_exist_60",
        "above_double_bull_close",
        "above_double_bull_high",
        "key_k_support",
        "semantic_uptrend_pullback",
        "semantic_low_cross_pullback",
        "semantic_confirmed",
        "semantic_strict",
        "semantic_trend_focus",
        "semantic_long_focus",
        "buy_semantic_score",
        "risk_distribution_any_20",
        "recent_failed_breakout_20",
        "top_distribution_20",
    ]
    if include_similarity:
        cols.extend(
            [
                c
                for c in [
                    "sim_corr_close_vol_concat",
                    "sim_cosine_close_vol_concat",
                    "sim_euclidean_close_vol_concat",
                    "sim_contrast_corr_close_vol_concat",
                    "sim_contrast_cosine_close_vol_concat",
                    "sim_weighted_corr_close_vol_concat",
                    "sim_lag_corr_close_vol_concat",
                    "sim_corr_close_norm",
                    "habit_profile_score",
                    "habit_rule_score",
                    "semantic_profile_score",
                    "semantic_rule_score",
                    "semantic_fusion_score",
                    "semantic_habit_fusion_score",
                    "master_fusion_score",
                ]
                if c in df.columns
            ]
        )
    arr = df[cols].fillna(0.0).to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, cols


def build_ml_scores(candidate_df: pd.DataFrame, positive_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    research_positive = positive_df[positive_df["split"] == "research"].copy()
    research_negative = candidate_df[(candidate_df["split"] == "research") & (candidate_df["negative_30d"])].copy()
    max_negative = max(len(research_positive) * 5, len(research_positive))
    research_negative = research_negative.sort_values("signal_date").head(max_negative).copy()
    train_df = pd.concat([research_positive.assign(label=1), research_negative.assign(label=0)], ignore_index=True)

    X_plain, _ = build_feature_matrix(train_df, include_similarity=False)
    X_mix, _ = build_feature_matrix(train_df, include_similarity=True)
    y = train_df["label"].to_numpy(dtype=int)
    Xp_all, _ = build_feature_matrix(candidate_df, include_similarity=False)
    Xm_all, _ = build_feature_matrix(candidate_df, include_similarity=True)

    trained: List[str] = []

    gnb = v4_mod.fit_gaussian_nb(X_plain, y)
    candidate_df["gnb_semantic_score"] = v4_mod.predict_gaussian_nb(gnb, Xp_all)
    trained.append("gnb_semantic_score")

    if HAS_SKLEARN:
        lr = v4_mod.LogisticRegression(max_iter=500, class_weight="balanced")
        lr.fit(X_mix, y)
        candidate_df["logistic_semantic_score"] = lr.predict_proba(Xm_all)[:, 1]
        trained.append("logistic_semantic_score")

        rf = v4_mod.RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_mix, y)
        candidate_df["rf_semantic_score"] = rf.predict_proba(Xm_all)[:, 1]
        trained.append("rf_semantic_score")

        et = v4_mod.ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        et.fit(X_mix, y)
        candidate_df["et_semantic_score"] = et.predict_proba(Xm_all)[:, 1]
        trained.append("et_semantic_score")

        hgb = v4_mod.HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=20,
            random_state=42,
        )
        hgb.fit(X_mix, y)
        candidate_df["hgb_semantic_score"] = hgb.predict_proba(Xm_all)[:, 1]
        trained.append("hgb_semantic_score")

    if HAS_LGB:
        model = v4_mod.lgb.LGBMClassifier(
            n_estimators=400,
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
        candidate_df["lgb_semantic_score"] = model.predict_proba(Xm_all)[:, 1]
        trained.append("lgb_semantic_score")

    if HAS_XGB:
        model = v4_mod.xgb.XGBClassifier(
            n_estimators=300,
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
        candidate_df["xgb_semantic_score"] = model.predict_proba(Xm_all)[:, 1]
        trained.append("xgb_semantic_score")

    return candidate_df, trained


def main() -> None:
    update_progress("starting")
    candidate_df = build_candidate_df()
    positive_df = build_positive_df()
    if positive_df.empty:
        raise ValueError("完美图/B1 没有可用正样本")

    cutoffs = split_three_way_by_positive_dates(positive_df["signal_date"].tolist())
    candidate_df["split"] = assign_split(candidate_df, cutoffs)
    positive_df["split"] = assign_split(positive_df.rename(columns={"signal_date": "signal_date"}), cutoffs)
    write_json(RESULT_DIR / "split_cutoffs.json", {k: str(v.date()) for k, v in cutoffs.items()})
    update_progress(
        "inputs_ready",
        candidate_count=int(len(candidate_df)),
        positive_count=int(len(positive_df)),
    )

    research_positive = positive_df[positive_df["split"] == "research"].copy()
    research_negative = candidate_df[(candidate_df["split"] == "research") & (candidate_df["negative_30d"])].copy()
    max_negative = max(len(research_positive) * 5, len(research_positive))
    research_negative = research_negative.sort_values("signal_date").head(max_negative).copy()
    if research_positive.empty:
        raise ValueError("research 段没有完美图/B1 正样本")

    candidate_df = v4_mod.add_similarity_family(candidate_df, research_positive, research_negative)
    update_progress("similarity_ready")
    candidate_df = v4_mod.add_habit_scores(candidate_df, research_positive, research_negative)
    candidate_df = add_semantic_habit_scores(candidate_df, research_positive, research_negative)
    candidate_df = add_pool_flags(candidate_df)
    candidate_df = add_rank_fusions(candidate_df)
    candidate_df, trained_models = build_ml_scores(candidate_df, positive_df)
    update_progress("models_ready", trained_models=trained_models)

    pool_summary: List[Dict[str, Any]] = []
    pool_names = [
        "pool_all",
        "pool_pullback",
        "pool_uptrend",
        "pool_low_cross",
        "pool_confirmed",
        "pool_trend_focus",
        "pool_strict",
    ]
    for pool_name in pool_names:
        for split_name in ["validation", "final_test"]:
            part = candidate_df[(candidate_df["split"] == split_name) & (candidate_df[pool_name])].copy()
            pool_summary.append(
                {
                    "pool": pool_name,
                    "split": split_name,
                    "sample_count": int(len(part)),
                    "date_count": int(part["signal_date"].nunique()) if not part.empty else 0,
                    "ret_20d_mean": float(part["ret_20d"].mean()) if len(part) else np.nan,
                    "up_20d_rate": float(part["up_20d"].mean()) if len(part) else np.nan,
                }
            )
    pd.DataFrame(pool_summary).to_csv(RESULT_DIR / "pool_summary.csv", index=False, encoding="utf-8-sig")

    score_specs = [("baseline", "lowest_J", "J", True)]
    for col in [c for c in candidate_df.columns if c.startswith("sim_") and not c.startswith("sim_contrast_")]:
        score_specs.append(("similarity", col.replace("sim_", ""), col, False))
    for col in [c for c in candidate_df.columns if c.startswith("sim_contrast_")]:
        score_specs.append(("contrast_similarity", col.replace("sim_contrast_", ""), col, False))
    score_specs.extend(
        [
            ("habit", "habit_profile", "habit_profile_score", False),
            ("habit", "habit_rule", "habit_rule_score", False),
            ("semantic", "semantic_profile", "semantic_profile_score", False),
            ("semantic", "semantic_rule", "semantic_rule_score", False),
            ("semantic", "buy_semantic", "buy_semantic_score", False),
            ("fusion", "semantic_fusion", "semantic_fusion_score", False),
            ("fusion", "semantic_habit_fusion", "semantic_habit_fusion_score", False),
            ("fusion", "master_fusion", "master_fusion_score", False),
            ("ml", "gnb_semantic", "gnb_semantic_score", False),
        ]
    )
    for col, variant in [
        ("logistic_semantic_score", "logistic_semantic"),
        ("rf_semantic_score", "random_forest_semantic"),
        ("et_semantic_score", "extra_trees_semantic"),
        ("hgb_semantic_score", "hist_gbdt_semantic"),
        ("lgb_semantic_score", "lightgbm_semantic"),
        ("xgb_semantic_score", "xgboost_semantic"),
    ]:
        if col in candidate_df.columns:
            score_specs.append(("ml_plus_similarity", variant, col, False))

    rows: List[Dict[str, Any]] = []
    for split_name in ["validation", "final_test"]:
        for pool_name in pool_names:
            part = candidate_df[(candidate_df["split"] == split_name) & (candidate_df[pool_name])].copy()
            if part.empty:
                continue
            base_row = {"family": "baseline", "variant": "all_candidates", "pool": pool_name, "topn": 0, "split": split_name}
            base_row.update(v4_mod.summarize_signal_df(part))
            rows.append(base_row)
            for family, variant, score_col, ascending in score_specs:
                for topn in DAILY_TOPN_LIST:
                    selected = v4_mod.select_daily_topn(part, score_col, topn, ascending=ascending)
                    row = {"family": family, "variant": variant, "pool": pool_name, "topn": topn, "split": split_name}
                    row.update(v4_mod.summarize_signal_df(selected))
                    rows.append(row)
    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(RESULT_DIR / "signal_layer_leaderboard.csv", index=False, encoding="utf-8-sig")
    update_progress("leaderboard_ready", row_count=int(len(leaderboard)))

    validation = leaderboard[leaderboard["split"] == "validation"].copy()
    winners = []
    for family in sorted(validation["family"].unique()):
        fam = validation[validation["family"] == family].copy()
        fam = fam.sort_values(["ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[False, False, False])
        winners.append(fam.iloc[0])
    validation_best = pd.DataFrame(winners)
    validation_best.to_csv(RESULT_DIR / "validation_family_best.csv", index=False, encoding="utf-8-sig")

    final_part = candidate_df[candidate_df["split"] == "final_test"].copy()
    final_reports = []
    final_rows = []
    score_map = {variant: score_col for family, variant, score_col, ascending in score_specs}
    ascending_map = {variant: ascending for family, variant, score_col, ascending in score_specs}
    for _, best in validation_best.iterrows():
        pool_name = str(best["pool"])
        family = str(best["family"])
        variant = str(best["variant"])
        topn = int(best["topn"])
        part = final_part[final_part[pool_name]].copy()
        if family == "baseline" and variant == "all_candidates":
            selected = part
        else:
            score_col = score_map.get(variant, "J")
            ascending = ascending_map.get(variant, False)
            selected = v4_mod.select_daily_topn(part, score_col, topn, ascending=ascending)
        report = {"family": family, "variant": variant, "pool": pool_name, "topn": topn}
        report.update(v4_mod.summarize_signal_df(selected))
        final_reports.append(report)
        out = selected.drop(columns=["seq_map"]).copy()
        out["strategy_tag"] = f"{family}_{variant}_{pool_name}_top{topn}"
        final_rows.append(out)

    final_report = pd.DataFrame(final_reports).sort_values(["ret_20d_mean", "up_20d_rate"], ascending=[False, False])
    final_report.to_csv(RESULT_DIR / "final_test_report.csv", index=False, encoding="utf-8-sig")
    if final_rows:
        pd.concat(final_rows, ignore_index=True).to_csv(RESULT_DIR / "final_test_selected_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "result_dir": str(RESULT_DIR),
        "candidate_signal_count": int(len(candidate_df)),
        "candidate_split_counts": {k: int(v) for k, v in candidate_df["split"].value_counts().sort_index().to_dict().items()},
        "positive_total": int(len(positive_df)),
        "positive_split_counts": {k: int(v) for k, v in positive_df["split"].value_counts().sort_index().to_dict().items()},
        "research_positive": int(len(research_positive)),
        "research_negative": int(len(research_negative)),
        "has_sklearn": HAS_SKLEARN,
        "has_lightgbm": HAS_LGB,
        "has_xgboost": HAS_XGB,
        "trained_models": trained_models,
    }
    write_json(RESULT_DIR / "summary.json", summary)
    update_progress("finished")


if __name__ == "__main__":
    main()
