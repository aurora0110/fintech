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
RESULT_DIR = ROOT / "results" / f"b1_similarity_ml_signal_v4_{RUN_TS}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_TOPN_LIST = [3, 5, 8, 10, 20, 50]
CHUNK_SIZE = 20000

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False


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


def to_rank_vec(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(arr))
    return order.astype(float) / max(len(order) - 1, 1)


def derive_rep_map(seq_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    close_norm = np.asarray(seq_map["close_norm"], dtype=float)
    returns = np.asarray(seq_map["returns"], dtype=float)
    close_vol = np.asarray(seq_map["close_vol_concat"], dtype=float)
    vol_len = (len(close_vol) - len(close_norm))
    volume_part = close_vol[len(close_norm) :] if vol_len > 0 else np.zeros_like(close_norm)
    out = {
        "close_norm": close_norm,
        "returns": returns,
        "close_vol_concat": close_vol,
        "ret_vol_concat": np.concatenate([returns, volume_part]),
        "close_plus_returns": np.concatenate([close_norm, returns]),
        "deriv_close_norm": np.diff(close_norm),
        "deriv_returns": np.diff(returns),
        "rank_close_norm": to_rank_vec(close_norm),
    }
    return out


def zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd


def weighted_corr_max(seq_chunk: np.ndarray, tpl: np.ndarray, recent_weight: float = 2.0) -> np.ndarray:
    n = seq_chunk.shape[1]
    w = np.linspace(1.0, recent_weight, n)
    w = w / w.sum()
    sx = seq_chunk - (seq_chunk * w).sum(axis=1, keepdims=True)
    tx = tpl - (tpl * w).sum(axis=1, keepdims=True)
    num = (sx[:, None, :] * tx[None, :, :] * w).sum(axis=2)
    den1 = np.sqrt((sx * sx * w).sum(axis=1, keepdims=True))
    den2 = np.sqrt((tx * tx * w).sum(axis=1))
    sims = np.divide(num, den1 * den2, out=np.zeros_like(num), where=(den1 * den2) > 1e-12)
    return np.nanmax(sims, axis=1)


def cosine_max(seq_chunk: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    num = seq_chunk @ tpl.T
    den = np.linalg.norm(seq_chunk, axis=1, keepdims=True) * np.linalg.norm(tpl, axis=1)
    sims = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return np.nanmax(sims, axis=1)


def corr_max(seq_chunk: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    sx = zscore_rows(seq_chunk)
    tx = zscore_rows(tpl)
    return cosine_max(sx, tx)


def euclidean_max(seq_chunk: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    d = np.sqrt(((seq_chunk[:, None, :] - tpl[None, :, :]) ** 2).sum(axis=2))
    return 1.0 / (1.0 + np.nanmin(d, axis=1))


def spearman_max(seq_chunk: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    sx = np.vstack([to_rank_vec(r) for r in seq_chunk])
    tx = np.vstack([to_rank_vec(r) for r in tpl])
    return corr_max(sx, tx)


def lag_corr_max(seq_chunk: np.ndarray, tpl: np.ndarray, max_lag: int = 2) -> np.ndarray:
    best = np.full(len(seq_chunk), -1.0)
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            s = seq_chunk
            t = tpl
        elif lag > 0:
            s = seq_chunk[:, lag:]
            t = tpl[:, :-lag]
        else:
            s = seq_chunk[:, :lag]
            t = tpl[:, -lag:]
        cur = corr_max(s, t)
        best = np.maximum(best, cur)
    return best


def compute_similarity_column(
    seqs: List[np.ndarray], templates: List[np.ndarray], method: str
) -> np.ndarray:
    seq_arr = np.vstack(seqs)
    tpl_arr = np.vstack(templates)
    out = np.zeros(len(seq_arr), dtype=float)
    for start in range(0, len(seq_arr), CHUNK_SIZE):
        part = seq_arr[start : start + CHUNK_SIZE]
        if method == "corr":
            out[start : start + CHUNK_SIZE] = corr_max(part, tpl_arr)
        elif method == "cosine":
            out[start : start + CHUNK_SIZE] = cosine_max(part, tpl_arr)
        elif method == "euclidean":
            out[start : start + CHUNK_SIZE] = euclidean_max(part, tpl_arr)
        elif method == "spearman":
            out[start : start + CHUNK_SIZE] = spearman_max(part, tpl_arr)
        elif method == "weighted_corr":
            out[start : start + CHUNK_SIZE] = weighted_corr_max(part, tpl_arr)
        elif method == "lag_corr":
            out[start : start + CHUNK_SIZE] = lag_corr_max(part, tpl_arr, max_lag=2)
        elif method == "derivative_corr":
            out[start : start + CHUNK_SIZE] = corr_max(part, tpl_arr)
        else:
            raise ValueError(method)
    return out


def add_similarity_family(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    cand_maps = [derive_rep_map(r["seq_map"]) for _, r in candidate_df.iterrows()]
    pos_maps = [derive_rep_map(r["seq_map"]) for _, r in pos_df.iterrows()]
    neg_maps = [derive_rep_map(r["seq_map"]) for _, r in neg_df.iterrows()]

    rep_method_specs = [
        ("close_norm", "corr"),
        ("close_norm", "cosine"),
        ("close_norm", "euclidean"),
        ("close_norm", "spearman"),
        ("close_norm", "weighted_corr"),
        ("close_norm", "lag_corr"),
        ("returns", "corr"),
        ("returns", "cosine"),
        ("returns", "euclidean"),
        ("returns", "spearman"),
        ("close_vol_concat", "corr"),
        ("close_vol_concat", "cosine"),
        ("close_vol_concat", "euclidean"),
        ("close_vol_concat", "weighted_corr"),
        ("close_vol_concat", "lag_corr"),
        ("ret_vol_concat", "corr"),
        ("ret_vol_concat", "cosine"),
        ("ret_vol_concat", "euclidean"),
        ("close_plus_returns", "corr"),
        ("close_plus_returns", "cosine"),
        ("rank_close_norm", "corr"),
        ("deriv_close_norm", "derivative_corr"),
        ("deriv_returns", "derivative_corr"),
    ]

    sim_meta = []
    for rep, method in rep_method_specs:
        pos_templates = [m[rep] for m in pos_maps]
        neg_templates = [m[rep] for m in neg_maps]
        cand_seqs = [m[rep] for m in cand_maps]
        pos_sim = compute_similarity_column(cand_seqs, pos_templates, method)
        neg_sim = compute_similarity_column(cand_seqs, neg_templates, method)
        base_col = f"sim_{method}_{rep}"
        contrast_col = f"sim_contrast_{method}_{rep}"
        candidate_df[base_col] = pos_sim
        candidate_df[contrast_col] = pos_sim - 0.6 * neg_sim
        sim_meta.append({"base_col": base_col, "contrast_col": contrast_col, "rep": rep, "method": method})

    pd.DataFrame(sim_meta).to_csv(RESULT_DIR / "similarity_columns.csv", index=False, encoding="utf-8-sig")
    return candidate_df


def safe_scale(values: pd.Series) -> float:
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    return float(max(q75 - q25, values.std(), 1e-4))


def add_habit_scores(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
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

    pos_stats = {}
    neg_stats = {}
    for col in feature_cols:
        ps = pd.to_numeric(pos_df[col], errors="coerce").dropna()
        ns = pd.to_numeric(neg_df[col], errors="coerce").dropna()
        pos_stats[col] = (float(ps.median()), safe_scale(ps))
        neg_stats[col] = (float(ns.median()), safe_scale(ns))

    pos_parts = []
    neg_parts = []
    for col in feature_cols:
        x = pd.to_numeric(candidate_df[col], errors="coerce").fillna(0.0)
        pm, ps = pos_stats[col]
        nm, ns = neg_stats[col]
        pos_parts.append(np.exp(-np.abs((x - pm) / ps)))
        neg_parts.append(np.exp(-np.abs((x - nm) / ns)))
    candidate_df["habit_profile_score"] = np.mean(np.vstack(pos_parts), axis=0) - 0.55 * np.mean(np.vstack(neg_parts), axis=0)

    q = {}
    rule_cols = ["ret3", "ret5", "close_to_trend", "close_to_long", "signal_vs_ma5", "vol_vs_prev", "trend_spread", "ma20_slope_5", "trend_slope_5", "long_slope_5"]
    for col in rule_cols:
        s = pd.to_numeric(pos_df[col], errors="coerce").dropna()
        q[col] = {"q10": float(s.quantile(0.10)), "q90": float(s.quantile(0.90))}
    rule_score = (
        (candidate_df["J"] < 13).astype(int)
        + (candidate_df["ret3"].between(q["ret3"]["q10"], q["ret3"]["q90"])).astype(int)
        + (candidate_df["ret5"].between(q["ret5"]["q10"], q["ret5"]["q90"])).astype(int)
        + (candidate_df["close_to_trend"].between(q["close_to_trend"]["q10"], q["close_to_trend"]["q90"])).astype(int)
        + (candidate_df["close_to_long"].between(q["close_to_long"]["q10"], q["close_to_long"]["q90"])).astype(int)
        + (candidate_df["signal_vs_ma5"] <= q["signal_vs_ma5"]["q90"]).astype(int)
        + (candidate_df["vol_vs_prev"] <= q["vol_vs_prev"]["q90"]).astype(int)
        + (candidate_df["trend_spread"] >= q["trend_spread"]["q10"]).astype(int)
        + (candidate_df["ma20_slope_5"] >= q["ma20_slope_5"]["q10"]).astype(int)
        + (candidate_df["trend_slope_5"] >= q["trend_slope_5"]["q10"]).astype(int)
        + (candidate_df["long_slope_5"] >= q["long_slope_5"]["q10"]).astype(int)
    )
    candidate_df["habit_rule_score"] = rule_score.astype(float)
    return candidate_df


def add_pool_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["pool_all"] = True
    df["pool_shrink"] = ((df["vol_vs_prev"] < 1.0) & (df["signal_vs_ma5"] <= 1.05))
    df["pool_near_trend"] = (
        df["close_to_trend"].between(-0.06, 0.02)
        & (df["close_to_long"] > -0.03)
        & (df["trend_spread"] > 0.0)
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
    return df


def add_rank_fusions(df: pd.DataFrame) -> pd.DataFrame:
    rank_cols = []
    use_cols = [
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
    ]
    for col in use_cols:
        if col not in df.columns:
            continue
        rank_col = f"rank_{col}"
        df[rank_col] = df.groupby("signal_date")[col].rank(pct=True, ascending=True)
        rank_cols.append(rank_col)

    df["sim_fusion_score"] = (
        df[[c for c in ["rank_sim_corr_close_vol_concat", "rank_sim_cosine_close_vol_concat", "rank_sim_euclidean_close_vol_concat"] if c in df.columns]].mean(axis=1)
    )
    df["contrast_fusion_score"] = (
        df[[c for c in ["rank_sim_contrast_corr_close_vol_concat", "rank_sim_contrast_cosine_close_vol_concat"] if c in df.columns]].mean(axis=1)
    )
    df["habit_fusion_score"] = (
        0.45 * df.get("rank_sim_contrast_corr_close_vol_concat", 0)
        + 0.20 * df.get("rank_habit_profile_score", 0)
        + 0.20 * df.get("rank_habit_rule_score", 0)
        + 0.15 * df.get("rank_sim_weighted_corr_close_vol_concat", 0)
    )
    return df


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
                    "sim_fusion_score",
                    "contrast_fusion_score",
                    "habit_fusion_score",
                ]
                if c in df.columns
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
    return {"classes": classes, "mean": mean, "std": std, "means": np.vstack(means), "vars": np.vstack(vars_), "priors": np.asarray(priors)}


def predict_gaussian_nb(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - model["mean"]) / model["std"]
    log_probs = []
    for i in range(len(model["priors"])):
        log_prior = np.log(model["priors"][i] + 1e-12)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * model["vars"][i]) + ((X - model["means"][i]) ** 2) / model["vars"][i], axis=1)
        log_probs.append(log_prior + log_likelihood)
    log_probs = np.vstack(log_probs).T
    log_probs = log_probs - log_probs.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[:, 1]


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

    trained = []

    gnb = fit_gaussian_nb(X_plain, y)
    candidate_df["gnb_plain_score"] = predict_gaussian_nb(gnb, Xp_all)
    trained.append("gnb_plain_score")

    if HAS_SKLEARN:
        lr = LogisticRegression(max_iter=400, class_weight="balanced")
        lr.fit(X_mix, y)
        candidate_df["logistic_mix_score"] = lr.predict_proba(Xm_all)[:, 1]
        trained.append("logistic_mix_score")

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_mix, y)
        candidate_df["rf_mix_score"] = rf.predict_proba(Xm_all)[:, 1]
        trained.append("rf_mix_score")

        et = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        et.fit(X_mix, y)
        candidate_df["et_mix_score"] = et.predict_proba(Xm_all)[:, 1]
        trained.append("et_mix_score")

        hgb = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=20,
            random_state=42,
        )
        hgb.fit(X_mix, y)
        candidate_df["hgb_mix_score"] = hgb.predict_proba(Xm_all)[:, 1]
        trained.append("hgb_mix_score")

    if HAS_LGB:
        model = lgb.LGBMClassifier(
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
        candidate_df["lgb_mix_score"] = model.predict_proba(Xm_all)[:, 1]
        trained.append("lgb_mix_score")

    if HAS_XGB:
        model = xgb.XGBClassifier(
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
        candidate_df["xgb_mix_score"] = model.predict_proba(Xm_all)[:, 1]
        trained.append("xgb_mix_score")

    return candidate_df, trained


def select_daily_topn(df: pd.DataFrame, score_col: str, topn: int, ascending: bool = False) -> pd.DataFrame:
    ranked = df.sort_values(["signal_date", score_col, "code"], ascending=[True, ascending, True])
    return ranked.groupby("signal_date", group_keys=False).head(topn).copy()


def summarize_signal_df(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"sample_count": 0, "date_count": 0, "up_5d_rate": np.nan, "up_10d_rate": np.nan, "up_20d_rate": np.nan, "ret_5d_mean": np.nan, "ret_10d_mean": np.nan, "ret_20d_mean": np.nan, "min_close_ret_30_mean": np.nan}
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


def main() -> None:
    update_progress("starting")
    candidate_df = load_candidate_df()
    positive_df = build_positive_df()
    positive_df.to_csv(RESULT_DIR / "positive_manifest_v4.csv", index=False, encoding="utf-8-sig")
    update_progress("inputs_loaded", candidate_count=int(len(candidate_df)), positive_count=int(len(positive_df)))

    research_positive = positive_df[positive_df["split"] == "research"].copy()
    research_negative = candidate_df[(candidate_df["split"] == "research") & (candidate_df["negative_30d"])].copy()
    research_negative = research_negative.sort_values("signal_date").head(max(len(research_positive) * 5, len(research_positive))).copy()
    update_progress("train_sets_ready", research_positive=int(len(research_positive)), research_negative=int(len(research_negative)))

    candidate_df = add_similarity_family(candidate_df, research_positive, research_negative)
    update_progress("similarity_ready")

    candidate_df = add_habit_scores(candidate_df, research_positive, research_negative)
    candidate_df = add_pool_flags(candidate_df)
    candidate_df = add_rank_fusions(candidate_df)
    candidate_df, trained_models = build_ml_scores(candidate_df, positive_df)
    update_progress("models_ready", trained_models=trained_models)

    pool_summary = []
    for pool_name in ["pool_all", "pool_shrink", "pool_near_trend", "pool_soft_core", "pool_core"]:
        for split_name in ["validation", "final_test"]:
            part = candidate_df[(candidate_df["split"] == split_name) & (candidate_df[pool_name])]
            pool_summary.append({
                "pool": pool_name,
                "split": split_name,
                "sample_count": int(len(part)),
                "date_count": int(part["signal_date"].nunique()),
                "ret_20d_mean": float(part["ret_20d"].mean()) if len(part) else np.nan,
                "up_20d_rate": float(part["up_20d"].mean()) if len(part) else np.nan,
            })
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
            ("fusion", "sim_fusion", "sim_fusion_score", False),
            ("fusion", "contrast_fusion", "contrast_fusion_score", False),
            ("fusion", "habit_fusion", "habit_fusion_score", False),
            ("ml", "gnb_features_only", "gnb_plain_score", False),
        ]
    )
    for col, variant in [
        ("logistic_mix_score", "logistic_features_plus_similarity"),
        ("rf_mix_score", "random_forest_plus_similarity"),
        ("et_mix_score", "extra_trees_plus_similarity"),
        ("hgb_mix_score", "hist_gbdt_plus_similarity"),
        ("lgb_mix_score", "lightgbm_plus_similarity"),
        ("xgb_mix_score", "xgboost_plus_similarity"),
    ]:
        if col in candidate_df.columns:
            score_specs.append(("ml_plus_similarity", variant, col, False))

    rows = []
    for split_name in ["validation", "final_test"]:
        for pool_name in ["pool_all", "pool_shrink", "pool_near_trend", "pool_soft_core", "pool_core"]:
            part = candidate_df[(candidate_df["split"] == split_name) & (candidate_df[pool_name])].copy()
            if part.empty:
                continue
            base_row = {"family": "baseline", "variant": "all_candidates", "pool": pool_name, "topn": 0, "split": split_name}
            base_row.update(summarize_signal_df(part))
            rows.append(base_row)
            for family, variant, score_col, ascending in score_specs:
                for topn in DAILY_TOPN_LIST:
                    selected = select_daily_topn(part, score_col, topn, ascending=ascending)
                    row = {"family": family, "variant": variant, "pool": pool_name, "topn": topn, "split": split_name}
                    row.update(summarize_signal_df(selected))
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
            selected = select_daily_topn(part, score_col, topn, ascending=ascending)
        report = {"family": family, "variant": variant, "pool": pool_name, "topn": topn}
        report.update(summarize_signal_df(selected))
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
