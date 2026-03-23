from __future__ import annotations

"""
B1 相似度 + 因子 + 机器学习融合策略
====================================

这份策略不是旧版 `b1filter.py` 的“昨日候选 + 今日确认”语义，而是当前研究阶段
跑出来的 B1 冠军买点思路的主流程版实现。它的核心目标是：

1. 先把样本限制在你真实会看的 B1 语义池里：
   - `J` 回到低位；
   - 形态属于“上涨后回踩趋势线”或“上涨后回踩多空线”；
   - 没有明显出货破坏；
   - `关键K / 缩半量 / 倍量柱` 作为加分项，而不是主类型。

2. 再用三类信息给候选打综合分：
   - 成功案例模板相似度；
   - 结构化因子（自动特征发现的前 20 个有效因子）；
   - 监督式买点评分（正例 + 反例 + 自动近似反例训练出来的轻量买点模型）。

3. 主流程里不直接做账户层买卖，而是输出“今天这只票值不值得进候选池、
   综合分是多少、属于哪类回踩、止损粗估在哪里”。

4. `main.py` 侧再对全市场候选按综合分排序，只保留 daily top3，
   对齐冠军策略的“每天只买最高分前 3 名”的交易口径。

注意：
- 这份代码默认复用已经验证通过的全量冠军实验目录：
  `/Users/lidongyang/Desktop/Qstrategy/results/b1_txt_template_signal_v2_full_20260322_201735`
- 实盘扫描时不再人为保留 research/validation/final_test 切分，而是直接使用
  当前所有已知正例/反例做模板和训练样本。
- 如果买入信号所在 K 线仍位于多空线下方，默认把标准止损幅度从 10% 收紧为 5%。
"""

import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
BASE_SIGNAL_DIR = ROOT / "results" / "b1_full_factor_signal_v6_full_20260321_102049"
TEMPLATE_SIGNAL_DIR = ROOT / "results" / "b1_txt_template_signal_v2_full_20260322_201735"
TXT_JOINT_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_txt_joint_opt_v1_20260322.py"
TXT_TEMPLATE_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_txt_template_opt_v2_20260322.py"
BASE_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
V4_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v4_20260320.py"
SEMANTIC_SCRIPT = ROOT / "utils" / "tmp" / "b1_semantic_shared_20260320.py"
FEATURE_REPORT_FILE = BASE_SIGNAL_DIR / "feature_discovery_report.csv"
HISTORICAL_CANDIDATE_FILE = TEMPLATE_SIGNAL_DIR / "candidate_enriched.csv"
TXT_POSITIVE_FILE = ROOT / "data" / "完美图" / "B1" / "正例.txt"
TXT_NEGATIVE_FILE = ROOT / "data" / "完美图" / "B1" / "反例.txt"
TXT_HARD_NEGATIVE_FILE = TEMPLATE_SIGNAL_DIR / "txt_auto_hard_negative_manifest.csv"
MODEL_CACHE_FILE = ROOT / "models" / "b1filter_similar_indicators_ml_bundle_v1.pkl"

SEQ_REP_NAME = "close_vol_concat"
SIM_METHODS = ["corr", "cosine", "weighted_corr", "lag_corr"]
DISPLAY_TOPN = 3
STOP_LOSS_MULTIPLIER = 0.90
STOP_LOSS_MULTIPLIER_BELOW_LONG = 0.95
TOP_FACTOR_COUNT = 20
EPS = 1e-12

try:
    from sklearn.ensemble import ExtraTreesClassifier
    HAS_SKLEARN = True
except Exception:
    ExtraTreesClassifier = None  # type: ignore
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    xgb = None  # type: ignore
    HAS_XGB = False


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base_mod = _load_module(BASE_SCRIPT, "b1_similar_base")
v4_mod = _load_module(V4_SCRIPT, "b1_similar_v4")
sem_mod = _load_module(SEMANTIC_SCRIPT, "b1_similar_sem")
txt_mod = _load_module(TXT_JOINT_SCRIPT, "b1_similar_txt")
template_mod = _load_module(TXT_TEMPLATE_SCRIPT, "b1_similar_template")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    if not np.isfinite(val):
        return default
    return val


def _ecdf_from_sorted(sorted_arr: np.ndarray, value: float) -> float:
    if sorted_arr.size == 0 or not np.isfinite(value):
        return 0.5
    pos = np.searchsorted(sorted_arr, value, side="right")
    return float(pos / sorted_arr.size)


def _sanitize_frame(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    x = df.reindex(columns=list(feature_cols)).copy()
    for col in x.columns:
        if x[col].dtype == bool:
            x[col] = x[col].astype(float)
    return x.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _prepare_numeric_row(row: pd.Series, feature_cols: Iterable[str]) -> pd.DataFrame:
    data: Dict[str, float] = {}
    for col in feature_cols:
        val = row.get(col, 0.0)
        if isinstance(val, (bool, np.bool_)):
            data[col] = float(bool(val))
        else:
            data[col] = _safe_float(val, 0.0)
    return pd.DataFrame([data], columns=list(feature_cols))


def _build_target_dates_by_code(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    hard_neg_df: pd.DataFrame,
    mapping: Dict[str, str],
) -> Dict[str, List[pd.Timestamp]]:
    out: Dict[str, List[pd.Timestamp]] = {}
    base_dates = txt_mod.collect_label_target_dates(pos_df, neg_df, mapping)
    for code, dates in base_dates.items():
        out.setdefault(str(code), []).extend(pd.Timestamp(d) for d in dates)
    if not hard_neg_df.empty:
        for _, row in hard_neg_df.iterrows():
            code = str(row["code"])
            out.setdefault(code, []).append(pd.Timestamp(row["signal_date"]))
    return out


def _map_hard_negatives(hard_neg_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    if hard_neg_df.empty or label_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, row in hard_neg_df.iterrows():
        code = str(row["code"])
        target_date = pd.Timestamp(row["signal_date"])
        sub = label_df[label_df["code"].astype(str) == code].copy()
        if sub.empty:
            continue
        hit = sub[sub["signal_date"] == target_date].copy()
        if hit.empty:
            hit = sub[
                sub["signal_date"].between(target_date - pd.Timedelta(days=3), target_date + pd.Timedelta(days=3))
            ].copy()
        if hit.empty:
            continue
        hit["date_gap"] = (hit["signal_date"] - target_date).abs().dt.days
        hit = hit.sort_values(["date_gap", "buy_semantic_score"], ascending=[True, False])
        best = hit.iloc[0].to_dict()
        best["label"] = 0
        rows.append(best)
    return pd.DataFrame(rows)


def _compute_discovery_score(row: pd.Series, bundle: Dict[str, Any]) -> float:
    score = 0.0
    for item in bundle["top_factor_specs"]:
        feature = item["feature"]
        signed_strength = float(item["signed_strength"])
        mean_val = float(item["mean"])
        std_val = float(item["std"])
        value = row.get(feature, 0.0)
        val = _safe_float(value, 0.0)
        z = (val - mean_val) / max(std_val, EPS)
        score += z * signed_strength
    return float(score)


def _compute_template_scores(row: pd.Series, rep_map: Dict[str, np.ndarray], bundle: Dict[str, Any]) -> Dict[str, float]:
    seq = np.vstack([np.asarray(rep_map[SEQ_REP_NAME], dtype=float)])
    pos_ranks: List[float] = []
    contrast_parts: List[float] = []
    raw_scores: Dict[str, float] = {}

    for method in SIM_METHODS:
        pos_templates = bundle["positive_templates"][SEQ_REP_NAME]
        pos_score = float(v4_mod.compute_similarity_column(seq, pos_templates, method)[0])
        pos_rank = _ecdf_from_sorted(bundle["ecdf_sorted"][f"tpl_{method}_{SEQ_REP_NAME}"], pos_score)
        raw_scores[f"tpl_{method}_{SEQ_REP_NAME}"] = pos_score
        pos_ranks.append(pos_rank)

        neg_user_templates = bundle["user_negative_templates"][SEQ_REP_NAME]
        neg_user_score = float(v4_mod.compute_similarity_column(seq, neg_user_templates, method)[0])
        neg_user_rank = _ecdf_from_sorted(bundle["ecdf_sorted"][f"neg_user_{method}_{SEQ_REP_NAME}"], neg_user_score)
        raw_scores[f"neg_user_{method}_{SEQ_REP_NAME}"] = neg_user_score

        neg_hard_templates = bundle["hard_negative_templates"][SEQ_REP_NAME]
        neg_hard_score = float(v4_mod.compute_similarity_column(seq, neg_hard_templates, method)[0])
        neg_hard_rank = _ecdf_from_sorted(bundle["ecdf_sorted"][f"neg_hard_{method}_{SEQ_REP_NAME}"], neg_hard_score)
        raw_scores[f"neg_hard_{method}_{SEQ_REP_NAME}"] = neg_hard_score

        contrast_parts.append(pos_rank - 0.35 * neg_user_rank - 0.55 * neg_hard_rank)

    template_similarity_score = float(np.mean(pos_ranks)) if pos_ranks else 0.5
    template_hard_contrast_score = float(np.mean(contrast_parts)) if contrast_parts else template_similarity_score
    hard_contrast_rank = _ecdf_from_sorted(
        bundle["ecdf_sorted"]["template_hard_contrast_score"],
        template_hard_contrast_score,
    )

    return {
        "template_similarity_score": template_similarity_score,
        "template_hard_contrast_score": template_hard_contrast_score,
        "template_hard_contrast_rank": hard_contrast_rank,
        **raw_scores,
    }


def _compute_confirm_score(row: pd.Series, bundle: Dict[str, Any]) -> Tuple[float, float]:
    txt_confirm_bonus = (
        float(bool(row.get("key_k_support", False))) * 1.0
        + float(bool(row.get("half_volume", False))) * 0.8
        + float(bool(row.get("double_bull_exist_60", False))) * 0.7
        + float(bool(row.get("semi_shrink", False))) * 0.4
    )
    txt_confirm_rank = _ecdf_from_sorted(bundle["ecdf_sorted"]["txt_confirm_bonus"], txt_confirm_bonus)
    template_confirm_score = (
        0.55 * txt_confirm_rank
        + 0.25 * float(bool(row.get("semantic_uptrend_pullback", False)))
        + 0.20 * float(bool(row.get("semantic_low_cross_pullback", False)))
    )
    return float(txt_confirm_bonus), float(template_confirm_score)


def _compute_buy_ml_score(row: pd.Series, bundle: Dict[str, Any]) -> Dict[str, float]:
    feature_df = _prepare_numeric_row(row, bundle["ml_feature_cols"])
    score_vals: List[float] = []
    score_details: Dict[str, float] = {}
    for model_name, model in bundle["ml_models"].items():
        try:
            prob = float(model.predict_proba(feature_df)[:, 1][0])
        except Exception:
            prob = 0.5
        rank = _ecdf_from_sorted(bundle["model_score_ecdf"][model_name], prob)
        score_vals.append(rank)
        score_details[model_name] = rank
    score_details["template_hard_ml_score"] = float(np.mean(score_vals)) if score_vals else 0.5
    return score_details


def _cache_source_signature() -> Dict[str, float]:
    files = [
        TEMPLATE_SIGNAL_DIR / "candidate_enriched.csv",
        TEMPLATE_SIGNAL_DIR / "txt_positive_manifest.csv",
        TEMPLATE_SIGNAL_DIR / "txt_negative_manifest.csv",
        TEMPLATE_SIGNAL_DIR / "txt_auto_hard_negative_manifest.csv",
        FEATURE_REPORT_FILE,
        TXT_POSITIVE_FILE,
        TXT_NEGATIVE_FILE,
    ]
    return {str(p): p.stat().st_mtime for p in files if p.exists()}


def _build_live_bundle() -> Dict[str, Any]:
    hist = pd.read_csv(HISTORICAL_CANDIDATE_FILE)
    if hist.empty:
        raise RuntimeError("冠军模板候选库为空，无法构建 B1 相似度+因子+ML 策略")

    pos_txt, bad_rows = txt_mod.parse_positive_txt(TXT_POSITIVE_FILE)
    neg_txt = txt_mod.parse_negative_txt(TXT_NEGATIVE_FILE)
    if not bad_rows.empty:
        raise RuntimeError(f"正例.txt 仍存在坏行：{len(bad_rows)}")
    pos_txt = txt_mod.add_reason_flags(pos_txt, "buy_reason")
    neg_txt = txt_mod.add_reason_flags(neg_txt, "no_buy_reason")

    mapping = base_mod.load_name_code_map()
    hard_neg_manifest = pd.read_csv(TXT_HARD_NEGATIVE_FILE)
    hard_neg_manifest["signal_date"] = pd.to_datetime(hard_neg_manifest["signal_date"])

    label_codes = txt_mod.extract_label_codes(pos_txt, neg_txt, mapping)
    label_codes = sorted(set(label_codes) | set(hard_neg_manifest["code"].astype(str)))
    split_windows = txt_mod.infer_split_windows(hist.assign(signal_date=pd.to_datetime(hist["signal_date"])))
    target_dates_by_code = _build_target_dates_by_code(pos_txt, neg_txt, hard_neg_manifest, mapping)
    label_df = template_mod.build_label_seq_feature_df(
        label_codes,
        split_windows,
        target_dates_by_code=target_dates_by_code,
        day_window=5,
    )
    positive_manifest, positive_skipped = txt_mod.map_positive_cases(pos_txt, label_df, mapping)
    negative_manifest, negative_skipped = txt_mod.map_negative_cases(neg_txt, label_df, mapping)
    hard_negative_rows = _map_hard_negatives(hard_neg_manifest, label_df)

    if not positive_skipped.empty or not negative_skipped.empty:
        raise RuntimeError("文本正反例在主流程映射时出现跳过，先修映射再扫描")
    if positive_manifest.empty:
        raise RuntimeError("没有成功映射的正例模板，无法构建 B1 相似度策略")
    if negative_manifest.empty:
        raise RuntimeError("没有成功映射的反例模板，无法构建 B1 相似度策略")
    if hard_negative_rows.empty:
        raise RuntimeError("没有成功映射的自动硬反例，无法构建 B1 相似度策略")

    positive_templates = {
        SEQ_REP_NAME: np.vstack([np.asarray(item["rep_map"][SEQ_REP_NAME], dtype=float) for item in positive_manifest.to_dict("records")])
    }
    user_negative_templates = {
        SEQ_REP_NAME: np.vstack([np.asarray(item["rep_map"][SEQ_REP_NAME], dtype=float) for item in negative_manifest.to_dict("records")])
    }
    hard_negative_templates = {
        SEQ_REP_NAME: np.vstack([np.asarray(item["rep_map"][SEQ_REP_NAME], dtype=float) for item in hard_negative_rows.to_dict("records")])
    }

    feature_report = pd.read_csv(FEATURE_REPORT_FILE).fillna(0.0)
    top_factors = feature_report.head(TOP_FACTOR_COUNT).copy()
    top_factor_specs: List[Dict[str, float]] = []
    for _, factor_row in top_factors.iterrows():
        feature = str(factor_row["feature"])
        if feature not in hist.columns:
            continue
        series = pd.to_numeric(hist[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        top_factor_specs.append(
            {
                "feature": feature,
                "signed_strength": float(factor_row.get("signed_strength", 0.0)),
                "mean": float(series.mean()),
                "std": float(series.std() if series.std() > EPS else 1.0),
            }
        )

    ecdf_cols = {
        "template_hard_contrast_score",
        "txt_confirm_bonus",
        "discovery_factor_score",
    }
    for method in SIM_METHODS:
        ecdf_cols.add(f"tpl_{method}_{SEQ_REP_NAME}")
        ecdf_cols.add(f"neg_user_{method}_{SEQ_REP_NAME}")
        ecdf_cols.add(f"neg_hard_{method}_{SEQ_REP_NAME}")

    ecdf_sorted: Dict[str, np.ndarray] = {}
    for col in sorted(ecdf_cols):
        if col not in hist.columns:
            raise RuntimeError(f"冠军模板结果缺少字段：{col}")
        values = pd.to_numeric(hist[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        ecdf_sorted[col] = np.sort(values)

    ml_feature_cols = [
        "discovery_factor_score",
        "buy_semantic_score",
        "txt_confirm_bonus",
        "template_similarity_score",
        "template_hard_contrast_score",
        "close_to_trend",
        "close_to_long",
        "trend_spread",
        "signal_vs_ma5",
        "vol_vs_prev",
        "ret3",
        "ret5",
        "ret10",
        "rsi14",
        "body_ratio",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "close_location",
        "ma20_slope_5",
        "trend_slope_5",
        "long_slope_5",
        "semantic_uptrend_pullback",
        "semantic_low_cross_pullback",
        "near_trend_pullback",
        "near_long_pullback",
        "key_k_support",
        "half_volume",
        "double_bull_exist_60",
        "risk_distribution_any_20",
    ]
    ml_feature_cols = [col for col in ml_feature_cols if col in hist.columns]

    train_parts = [
        positive_manifest.assign(label=1),
        negative_manifest.assign(label=0),
        hard_negative_rows.assign(label=0),
    ]
    train_df = pd.concat(train_parts, ignore_index=True)
    X_train = _sanitize_frame(train_df, ml_feature_cols)
    y_train = train_df["label"].astype(int).to_numpy()
    X_hist = _sanitize_frame(hist, ml_feature_cols)

    ml_models: Dict[str, Any] = {}
    model_score_ecdf: Dict[str, np.ndarray] = {}

    if HAS_SKLEARN:
        et_model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        et_model.fit(X_train, y_train)
        ml_models["et"] = et_model
        model_score_ecdf["et"] = np.sort(et_model.predict_proba(X_hist)[:, 1].astype(float))

    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        ml_models["xgb"] = xgb_model
        model_score_ecdf["xgb"] = np.sort(xgb_model.predict_proba(X_hist)[:, 1].astype(float))

    if not ml_models:
        raise RuntimeError("当前环境缺少可用的买点机器学习模型依赖")

    return {
        "source_signature": _cache_source_signature(),
        "positive_templates": positive_templates,
        "user_negative_templates": user_negative_templates,
        "hard_negative_templates": hard_negative_templates,
        "top_factor_specs": top_factor_specs,
        "ecdf_sorted": ecdf_sorted,
        "ml_feature_cols": ml_feature_cols,
        "ml_models": ml_models,
        "model_score_ecdf": model_score_ecdf,
        "positive_count": int(len(positive_manifest)),
        "negative_count": int(len(negative_manifest)),
        "hard_negative_count": int(len(hard_negative_rows)),
    }


def _load_live_bundle() -> Dict[str, Any]:
    source_signature = _cache_source_signature()
    if MODEL_CACHE_FILE.exists():
        try:
            with MODEL_CACHE_FILE.open("rb") as f:
                bundle = pickle.load(f)
            if bundle.get("source_signature") == source_signature:
                return bundle
        except Exception:
            pass

    bundle = _build_live_bundle()
    MODEL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with MODEL_CACHE_FILE.open("wb") as f:
            pickle.dump(bundle, f)
    except Exception:
        pass
    return bundle


def _build_live_feature_row(file_path: str, feature_cache=None) -> Optional[pd.Series]:
    raw_df = feature_cache.raw_df().copy() if feature_cache is not None and feature_cache.raw_df() is not None else base_mod.load_stock_data(file_path)
    if raw_df is None or raw_df.empty or len(raw_df) < base_mod.MIN_BARS:
        return None

    feat = sem_mod.add_semantic_buy_features(raw_df)
    if feat is None or feat.empty or len(feat) < sem_mod.SEQ_LEN:
        return None

    latest = feat.iloc[-1].copy()
    latest["txt_core_trend"] = bool(latest["J"] < 13) and bool(latest["near_trend_pullback"]) and bool(latest["semantic_uptrend_pullback"])
    latest["txt_core_long"] = bool(latest["J"] < 13) and bool(latest["near_long_pullback"])
    latest["txt_core_dual"] = bool(latest["txt_core_trend"]) or bool(latest["txt_core_long"])
    latest["txt_confirm_bonus"] = (
        float(bool(latest["key_k_support"])) * 1.0
        + float(bool(latest["half_volume"])) * 0.8
        + float(bool(latest["double_bull_exist_60"])) * 0.7
        + float(bool(latest["semi_shrink"])) * 0.4
    )
    latest["pool_txt_confirmed"] = bool(latest["txt_core_dual"]) and float(latest["txt_confirm_bonus"]) >= 0.8
    latest["pool_txt_dual"] = bool(latest["txt_core_dual"])

    if not bool(latest["pool_txt_confirmed"]):
        return None

    seq_window = feat.iloc[-sem_mod.SEQ_LEN :].copy()
    seq_map = sem_mod.extract_sequence(seq_window)
    latest["seq_map"] = seq_map
    latest["rep_map"] = v4_mod.derive_rep_map(seq_map)
    return latest


def check(file_path: str, hold_list=None, feature_cache=None):
    del hold_list

    latest = _build_live_feature_row(file_path, feature_cache=feature_cache)
    if latest is None:
        return [-1]

    bundle = _load_live_bundle()
    template_scores = _compute_template_scores(latest, latest["rep_map"], bundle)
    discovery_factor_score = _compute_discovery_score(latest, bundle)
    discovery_rank = _ecdf_from_sorted(bundle["ecdf_sorted"]["discovery_factor_score"], discovery_factor_score)
    txt_confirm_bonus, template_confirm_score = _compute_confirm_score(latest, bundle)
    ml_scores = _compute_buy_ml_score(
        pd.Series(
            {
                **latest.to_dict(),
                "discovery_factor_score": discovery_factor_score,
                "template_similarity_score": template_scores["template_similarity_score"],
                "template_hard_contrast_score": template_scores["template_hard_contrast_score"],
                "txt_confirm_bonus": txt_confirm_bonus,
            }
        ),
        bundle,
    )

    fusion_score = (
        0.30 * discovery_rank
        + 0.25 * template_scores["template_similarity_score"]
        + 0.20 * template_scores["template_hard_contrast_rank"]
        + 0.15 * ml_scores["template_hard_ml_score"]
        + 0.10 * template_confirm_score
    )

    close_price = _safe_float(latest.get("close"), np.nan)
    low_price = _safe_float(latest.get("low"), np.nan)
    long_line = _safe_float(latest.get("long_line"), np.nan)
    stop_multiplier = STOP_LOSS_MULTIPLIER_BELOW_LONG if np.isfinite(close_price) and np.isfinite(long_line) and close_price < long_line else STOP_LOSS_MULTIPLIER
    stop_loss_price = round(low_price * stop_multiplier, 1) if np.isfinite(low_price) and low_price > 0 else np.nan

    pullback_label = "回踩趋势线" if bool(latest.get("txt_core_trend", False)) else "回踩多空线"
    if bool(latest.get("txt_core_trend", False)) and bool(latest.get("txt_core_long", False)):
        pullback_label = "趋势线/多空线双回踩"

    extras: List[str] = []
    if bool(latest.get("key_k_support", False)):
        extras.append("关键K")
    if bool(latest.get("half_volume", False)):
        extras.append("缩半量")
    if bool(latest.get("double_bull_exist_60", False)):
        extras.append("倍量柱")
    if bool(latest.get("semi_shrink", False)) and "缩半量" not in extras:
        extras.append("量能收缩")

    note = (
        f"主买点={pullback_label}"
        f" | 融合分={fusion_score:.3f}"
        f" | 模板相似={template_scores['template_similarity_score']:.3f}"
        f" | 反例对比={template_scores['template_hard_contrast_rank']:.3f}"
        f" | 因子分位={discovery_rank:.3f}"
        f" | ML分位={ml_scores['template_hard_ml_score']:.3f}"
        f" | 确认分={template_confirm_score:.3f}"
    )
    if extras:
        note = f"{note} | 加分项={'/'.join(extras)}"
    if stop_multiplier == STOP_LOSS_MULTIPLIER_BELOW_LONG:
        note = f"{note} | 多空线下买入，止损收紧为5%"

    return [1, stop_loss_price, close_price, fusion_score, note]
