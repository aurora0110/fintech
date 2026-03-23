from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
PREV_V1 = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
PREV_V2 = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v2_20260320.py"
PREV_V1_RESULT = ROOT / "results" / "b1_similarity_ml_signal_20260320_164521"
PREV_V2_RESULT = ROOT / "results" / "b1_similarity_ml_signal_v2_20260320_210752"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = ROOT / "results" / f"b1_similarity_ml_signal_v3_{RUN_TS}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_TOPN_LIST = [3, 5, 8, 10, 20, 50]
SIMILARITY_VARIANTS = [
    ("corr", "close_vol_concat"),
    ("cosine", "close_vol_concat"),
    ("euclidean", "close_vol_concat"),
    ("corr", "close_norm"),
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod1 = load_module(PREV_V1, "b1_prev_v1")
mod2 = load_module(PREV_V2, "b1_prev_v2")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(stage: str, **kwargs: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    payload.update(kwargs)
    write_json(RESULT_DIR / "progress.json", payload)


def load_candidate_df() -> pd.DataFrame:
    candidate_pkl = PREV_V1_RESULT / "candidate_rows.pkl"
    df = pd.read_pickle(candidate_pkl)
    cutoffs = json.loads((PREV_V2_RESULT / "split_cutoffs.json").read_text(encoding="utf-8"))
    cutoffs = {k: pd.Timestamp(v) for k, v in cutoffs.items()}
    df["split"] = mod1.assign_split(df, cutoffs)
    return df


def build_positive_df() -> pd.DataFrame:
    name_code_map = mod1.load_name_code_map()
    raw = mod2.parse_b1_case_files()
    enriched = pd.DataFrame([mod2.enrich_case(r, name_code_map) for _, r in raw.iterrows()])
    enriched = enriched[enriched["status"] == "ok"].copy().reset_index(drop=True)
    cutoffs = json.loads((PREV_V2_RESULT / "split_cutoffs.json").read_text(encoding="utf-8"))
    cutoffs = {k: pd.Timestamp(v) for k, v in cutoffs.items()}
    enriched["split"] = mod1.assign_split(enriched, cutoffs)
    return enriched


def stack_templates(rows: Iterable[Dict[str, Any]], rep_name: str) -> np.ndarray:
    arrs = [np.asarray(r["seq_map"][rep_name], dtype=float) for r in rows]
    if not arrs:
        return np.empty((0, 0), dtype=float)
    return np.vstack(arrs)


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
    ]
    if include_similarity:
        cols.extend(
            [
                "sim_corr_close_vol_concat",
                "sim_cosine_close_vol_concat",
                "sim_euclidean_close_vol_concat",
                "sim_corr_close_norm",
                "sim_contrast_corr_close_vol_concat",
                "sim_contrast_cosine_close_vol_concat",
                "habit_profile_score",
                "habit_rule_score",
                "habit_fusion_score",
            ]
        )
    arr = df[cols].fillna(0.0).to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, cols


def fit_gaussian_nb(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=int)
    mean = X.mean(axis=0)
    std = np.where(X.std(axis=0) < 1e-6, 1.0, X.std(axis=0))
    X = (X - mean) / std
    classes = np.array([0, 1], dtype=int)
    means = []
    vars_ = []
    priors = []
    for c in classes:
        Xc = X[y == c]
        means.append(Xc.mean(axis=0))
        vars_.append(np.where(Xc.var(axis=0) < 1e-6, 1e-6, Xc.var(axis=0)))
        priors.append(float(len(Xc) / len(X)))
    return {
        "classes": classes,
        "mean": mean,
        "std": std,
        "means": np.vstack(means),
        "vars": np.vstack(vars_),
        "priors": np.asarray(priors, dtype=float),
    }


def predict_gaussian_nb(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - model["mean"]) / model["std"]
    means = model["means"]
    vars_ = model["vars"]
    priors = model["priors"]
    log_probs = []
    for i in range(len(priors)):
        log_prior = np.log(priors[i] + 1e-12)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * vars_[i]) + ((X - means[i]) ** 2) / vars_[i], axis=1)
        log_probs.append(log_prior + log_likelihood)
    log_probs = np.vstack(log_probs).T
    log_probs = log_probs - log_probs.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[:, 1]


def safe_scale(values: pd.Series) -> float:
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    scale = max(q75 - q25, values.std(), 1e-4)
    return float(scale)


def compute_habit_profile_scores(df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "J",
        "ret3",
        "ret5",
        "signal_ret",
        "trend_spread",
        "close_to_trend",
        "close_to_long",
        "signal_vs_ma5",
        "vol_vs_prev",
        "ma20_slope_5",
        "trend_slope_5",
        "long_slope_5",
    ]
    pos_stats = {}
    neg_stats = {}
    for col in feature_cols:
        pos_series = pd.to_numeric(pos_df[col], errors="coerce").dropna()
        neg_series = pd.to_numeric(neg_df[col], errors="coerce").dropna()
        pos_stats[col] = (float(pos_series.median()), safe_scale(pos_series))
        neg_stats[col] = (float(neg_series.median()), safe_scale(neg_series))

    pos_score_parts = []
    neg_score_parts = []
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        pos_med, pos_scale = pos_stats[col]
        neg_med, neg_scale = neg_stats[col]
        pos_score_parts.append(np.exp(-np.abs((x - pos_med) / pos_scale)))
        neg_score_parts.append(np.exp(-np.abs((x - neg_med) / neg_scale)))

    pos_score = np.mean(np.vstack(pos_score_parts), axis=0)
    neg_score = np.mean(np.vstack(neg_score_parts), axis=0)
    df["habit_profile_score"] = pos_score - 0.55 * neg_score
    return df


def compute_habit_rule_score(df: pd.DataFrame, pos_df: pd.DataFrame) -> pd.DataFrame:
    q = {}
    cols = [
        "ret3",
        "ret5",
        "close_to_trend",
        "close_to_long",
        "signal_vs_ma5",
        "vol_vs_prev",
        "trend_spread",
        "ma20_slope_5",
        "trend_slope_5",
        "long_slope_5",
    ]
    for col in cols:
        s = pd.to_numeric(pos_df[col], errors="coerce").dropna()
        q[col] = {
            "q10": float(s.quantile(0.10)),
            "q90": float(s.quantile(0.90)),
        }

    score = (
        (df["J"] < 13).astype(int)
        + (df["ret3"].between(q["ret3"]["q10"], q["ret3"]["q90"])).astype(int)
        + (df["ret5"].between(q["ret5"]["q10"], q["ret5"]["q90"])).astype(int)
        + (df["close_to_trend"].between(q["close_to_trend"]["q10"], q["close_to_trend"]["q90"])).astype(int)
        + (df["close_to_long"].between(q["close_to_long"]["q10"], q["close_to_long"]["q90"])).astype(int)
        + (df["signal_vs_ma5"] <= q["signal_vs_ma5"]["q90"]).astype(int)
        + (df["vol_vs_prev"] <= q["vol_vs_prev"]["q90"]).astype(int)
        + (df["trend_spread"] >= q["trend_spread"]["q10"]).astype(int)
        + (df["ma20_slope_5"] >= q["ma20_slope_5"]["q10"]).astype(int)
        + (df["trend_slope_5"] >= q["trend_slope_5"]["q10"]).astype(int)
        + (df["long_slope_5"] >= q["long_slope_5"]["q10"]).astype(int)
    )
    df["habit_rule_score"] = score.astype(float)
    return df


def add_similarity_columns(candidate_df: pd.DataFrame, positive_df: pd.DataFrame, negative_df: pd.DataFrame) -> pd.DataFrame:
    candidate_rows = candidate_df.to_dict("records")
    pos_rows = positive_df.to_dict("records")
    neg_rows = negative_df.to_dict("records")

    for scorer, rep in SIMILARITY_VARIANTS:
        seqs = np.vstack([r["seq_map"][rep] for r in candidate_rows])
        pos_templates = stack_templates(pos_rows, rep)
        neg_templates = stack_templates(neg_rows, rep)
        pos_sim = mod1.similarity_scores(seqs, pos_templates, scorer)
        neg_sim = mod1.similarity_scores(seqs, neg_templates, scorer)
        candidate_df[f"sim_{scorer}_{rep}"] = pos_sim
        candidate_df[f"sim_contrast_{scorer}_{rep}"] = pos_sim - 0.65 * neg_sim
    return candidate_df


def select_daily_topn(df: pd.DataFrame, score_col: str, topn: int) -> pd.DataFrame:
    ranked = df.sort_values(["signal_date", score_col, "code"], ascending=[True, False, True])
    return ranked.groupby("signal_date", group_keys=False).head(topn).copy()


def summarize_signal_df(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "sample_count": 0,
            "date_count": 0,
            "up_5d_rate": np.nan,
            "up_10d_rate": np.nan,
            "up_20d_rate": np.nan,
            "ret_5d_mean": np.nan,
            "ret_10d_mean": np.nan,
            "ret_20d_mean": np.nan,
            "min_close_ret_30_mean": np.nan,
        }
    return {
        "sample_count": int(len(df)),
        "date_count": int(df["signal_date"].nunique()),
        "up_5d_rate": float(df["up_5d"].mean()),
        "up_10d_rate": float(df["up_10d"].mean()),
        "up_20d_rate": float(df["up_20d"].mean()),
        "ret_5d_mean": float(df["ret_5d"].mean()),
        "ret_10d_mean": float(df["ret_10d"].mean()),
        "ret_20d_mean": float(df["ret_20d"].mean()),
        "min_close_ret_30_mean": float(df["min_close_ret_30"].mean()),
    }


def add_pool_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["pool_all"] = True
    df["pool_shrink"] = ((df["vol_vs_prev"] < 1.0) & (df["signal_vs_ma5"] <= 1.05))
    df["pool_near_trend"] = (
        df["close_to_trend"].between(-0.06, 0.02)
        & (df["close_to_long"] > -0.03)
        & (df["trend_spread"] > 0.0)
    )
    df["pool_core"] = (
        (df["vol_vs_prev"] < 1.0)
        & (df["signal_vs_ma5"] <= 1.05)
        & df["close_to_trend"].between(-0.06, 0.02)
        & (df["close_to_long"] > -0.03)
        & (df["ret3"] <= -0.008)
        & (df["ret5"] <= -0.012)
        & (df["trend_spread"] > 0.01)
        & (df["ma20_slope_5"] > 0.0)
        & (df["long_slope_5"] > 0.0)
    )
    df["pool_soft_core"] = (
        (df["vol_vs_prev"] <= 1.1)
        & (df["signal_vs_ma5"] <= 1.10)
        & df["close_to_trend"].between(-0.065, 0.015)
        & (df["close_to_long"] > -0.035)
        & (df["ret3"] < 0)
        & (df["ret5"] < 0)
        & (df["trend_spread"] > 0.005)
    )
    return df


def add_rank_fusion_scores(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "sim_corr_close_vol_concat",
        "sim_cosine_close_vol_concat",
        "sim_contrast_corr_close_vol_concat",
        "sim_contrast_cosine_close_vol_concat",
        "habit_profile_score",
        "habit_rule_score",
    ]:
        df[f"rank_{col}"] = df.groupby("signal_date")[col].rank(pct=True, ascending=True)

    df["sim_fusion_score"] = (
        df["rank_sim_corr_close_vol_concat"]
        + df["rank_sim_cosine_close_vol_concat"]
        + df["rank_sim_contrast_corr_close_vol_concat"]
    ) / 3.0
    df["habit_fusion_score"] = (
        0.55 * df["rank_sim_contrast_corr_close_vol_concat"]
        + 0.25 * df["rank_habit_profile_score"]
        + 0.20 * df["rank_habit_rule_score"]
    )
    return df


def build_ml_scores(candidate_df: pd.DataFrame, positive_df: pd.DataFrame) -> pd.DataFrame:
    research_positive = positive_df[positive_df["split"] == "research"].copy()
    research_negative = candidate_df[(candidate_df["split"] == "research") & (candidate_df["negative_30d"])].copy()
    max_negative = max(len(research_positive) * 5, len(research_positive))
    research_negative = research_negative.sort_values("signal_date").head(max_negative).copy()
    train_df = pd.concat(
        [
            research_positive.assign(label=1, source="positive"),
            research_negative.assign(label=0, source="negative"),
        ],
        ignore_index=True,
    )

    X_plain, _ = build_feature_matrix(train_df, include_similarity=False)
    X_mix, _ = build_feature_matrix(train_df, include_similarity=True)
    y = train_df["label"].to_numpy(dtype=float)
    gnb_plain = fit_gaussian_nb(X_plain, y)
    logistic_mix = mod1.fit_logistic_regression(X_mix, y)

    Xp_all, _ = build_feature_matrix(candidate_df, include_similarity=False)
    Xm_all, _ = build_feature_matrix(candidate_df, include_similarity=True)
    candidate_df["gnb_plain_score"] = predict_gaussian_nb(gnb_plain, Xp_all)
    candidate_df["logistic_mix_score"] = mod1.predict_logistic(logistic_mix, Xm_all)
    return candidate_df


def evaluate_family_rows(candidate_df: pd.DataFrame, family: str, variant: str, split_name: str, pool_name: str, score_col: str, topn: int) -> Dict[str, Any]:
    part = candidate_df[(candidate_df["split"] == split_name) & (candidate_df[pool_name])].copy()
    selected = select_daily_topn(part, score_col, topn)
    row = {"family": family, "variant": variant, "pool": pool_name, "topn": topn, "split": split_name}
    row.update(summarize_signal_df(selected))
    return row


def main() -> None:
    update_progress("starting")
    candidate_df = load_candidate_df()
    positive_df = build_positive_df()
    update_progress("inputs_loaded", candidate_count=int(len(candidate_df)), positive_count=int(len(positive_df)))

    positive_df.to_csv(RESULT_DIR / "positive_manifest_v3.csv", index=False, encoding="utf-8-sig")
    candidate_df.drop(columns=["seq_map"]).head(2000).to_csv(RESULT_DIR / "candidate_preview.csv", index=False, encoding="utf-8-sig")

    research_positive = positive_df[positive_df["split"] == "research"].copy()
    research_negative = candidate_df[(candidate_df["split"] == "research") & (candidate_df["negative_30d"])].copy()
    max_negative = max(len(research_positive) * 5, len(research_positive))
    research_negative = research_negative.sort_values("signal_date").head(max_negative).copy()
    update_progress("train_sets_ready", research_positive=int(len(research_positive)), research_negative=int(len(research_negative)))

    candidate_df = add_similarity_columns(candidate_df, research_positive, research_negative)
    update_progress("similarity_ready")

    candidate_df = compute_habit_profile_scores(candidate_df, research_positive, research_negative)
    candidate_df = compute_habit_rule_score(candidate_df, research_positive)
    candidate_df = add_rank_fusion_scores(candidate_df)
    candidate_df = add_pool_flags(candidate_df)
    candidate_df = build_ml_scores(candidate_df, positive_df)
    update_progress("habit_and_ml_ready")

    pool_summary = []
    for pool_name in ["pool_all", "pool_shrink", "pool_near_trend", "pool_soft_core", "pool_core"]:
        for split_name in ["validation", "final_test"]:
            part = candidate_df[(candidate_df["split"] == split_name) & (candidate_df[pool_name])]
            pool_summary.append(
                {
                    "pool": pool_name,
                    "split": split_name,
                    "sample_count": int(len(part)),
                    "date_count": int(part["signal_date"].nunique()),
                    "ret_20d_mean": float(part["ret_20d"].mean()) if len(part) else np.nan,
                    "up_20d_rate": float(part["up_20d"].mean()) if len(part) else np.nan,
                }
            )
    pd.DataFrame(pool_summary).to_csv(RESULT_DIR / "pool_summary.csv", index=False, encoding="utf-8-sig")

    rows: List[Dict[str, Any]] = []
    score_specs = [
        ("baseline", "lowest_J", "J", True),
        ("similarity", "corr_close_vol_concat", "sim_corr_close_vol_concat", False),
        ("similarity", "cosine_close_vol_concat", "sim_cosine_close_vol_concat", False),
        ("contrast_similarity", "contrast_corr_close_vol_concat", "sim_contrast_corr_close_vol_concat", False),
        ("contrast_similarity", "contrast_cosine_close_vol_concat", "sim_contrast_cosine_close_vol_concat", False),
        ("habit", "habit_profile", "habit_profile_score", False),
        ("habit", "habit_rule", "habit_rule_score", False),
        ("fusion", "habit_plus_contrast", "habit_fusion_score", False),
        ("fusion", "sim_fusion", "sim_fusion_score", False),
        ("ml", "gnb_features_only", "gnb_plain_score", False),
        ("ml_plus_similarity", "logistic_features_plus_similarity", "logistic_mix_score", False),
    ]

    for split_name in ["validation", "final_test"]:
        for pool_name in ["pool_all", "pool_shrink", "pool_near_trend", "pool_soft_core", "pool_core"]:
            part = candidate_df[(candidate_df["split"] == split_name) & (candidate_df[pool_name])].copy()
            if part.empty:
                continue
            all_row = {"family": "baseline", "variant": "all_candidates", "pool": pool_name, "topn": 0, "split": split_name}
            all_row.update(summarize_signal_df(part))
            rows.append(all_row)

            for family, variant, score_col, ascending in score_specs:
                for topn in DAILY_TOPN_LIST:
                    ranked = part.sort_values(["signal_date", score_col, "code"], ascending=[True, ascending, True]) if ascending else None
                    if ascending:
                        selected = ranked.groupby("signal_date", group_keys=False).head(topn).copy()
                    else:
                        selected = select_daily_topn(part, score_col, topn)
                    row = {"family": family, "variant": variant, "pool": pool_name, "topn": topn, "split": split_name}
                    row.update(summarize_signal_df(selected))
                    rows.append(row)

    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(RESULT_DIR / "signal_layer_leaderboard.csv", index=False, encoding="utf-8-sig")
    update_progress("leaderboard_ready", rows=int(len(leaderboard)))

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
    final_selected_frames = []
    for _, best in validation_best.iterrows():
        pool_name = str(best["pool"])
        family = str(best["family"])
        variant = str(best["variant"])
        topn = int(best["topn"])
        part = final_part[final_part[pool_name]].copy()
        if family == "baseline" and variant == "all_candidates":
            selected = part
        elif family == "baseline":
            selected = part.sort_values(["signal_date", "J", "code"], ascending=[True, True, True]).groupby("signal_date", group_keys=False).head(topn).copy()
        else:
            score_map = {
                "corr_close_vol_concat": "sim_corr_close_vol_concat",
                "cosine_close_vol_concat": "sim_cosine_close_vol_concat",
                "contrast_corr_close_vol_concat": "sim_contrast_corr_close_vol_concat",
                "contrast_cosine_close_vol_concat": "sim_contrast_cosine_close_vol_concat",
                "habit_profile": "habit_profile_score",
                "habit_rule": "habit_rule_score",
                "habit_plus_contrast": "habit_fusion_score",
                "sim_fusion": "sim_fusion_score",
                "gnb_features_only": "gnb_plain_score",
                "logistic_features_plus_similarity": "logistic_mix_score",
            }
            score_col = score_map.get(variant, "")
            if family == "baseline":
                selected = part.sort_values(["signal_date", "J", "code"], ascending=[True, True, True]).groupby("signal_date", group_keys=False).head(topn).copy()
            else:
                selected = select_daily_topn(part, score_col, topn)

        report = {"family": family, "variant": variant, "pool": pool_name, "topn": topn}
        report.update(summarize_signal_df(selected))
        final_reports.append(report)
        out = selected.drop(columns=["seq_map"]).copy()
        out["strategy_tag"] = f"{family}_{variant}_{pool_name}_top{topn}"
        final_selected_frames.append(out)

    final_report = pd.DataFrame(final_reports).sort_values(["ret_20d_mean", "up_20d_rate"], ascending=[False, False])
    final_report.to_csv(RESULT_DIR / "final_test_report.csv", index=False, encoding="utf-8-sig")
    pd.concat(final_selected_frames, ignore_index=True).to_csv(RESULT_DIR / "final_test_selected_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "result_dir": str(RESULT_DIR),
        "candidate_signal_count": int(len(candidate_df)),
        "candidate_split_counts": {k: int(v) for k, v in candidate_df["split"].value_counts().sort_index().to_dict().items()},
        "positive_total": int(len(positive_df)),
        "positive_split_counts": {k: int(v) for k, v in positive_df["split"].value_counts().sort_index().to_dict().items()},
        "research_positive": int(len(research_positive)),
        "research_negative": int(len(research_negative)),
    }
    write_json(RESULT_DIR / "summary.json", summary)
    update_progress("finished")


if __name__ == "__main__":
    main()
