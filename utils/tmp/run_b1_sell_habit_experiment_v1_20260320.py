from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
V2_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v2_20260320.py"
RISK_SCRIPT = ROOT / "utils" / "market_risk_tags.py"
B1_PERFECT_DIR = ROOT / "data" / "完美图" / "B1"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = ROOT / "results" / f"b1_sell_habit_experiment_v1_{RUN_TS}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEG_PER_CASE = 5
MIN_HOLD_DAYS = 8


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = load_module(V2_SCRIPT, "b1_v2")
risk_mod = load_module(RISK_SCRIPT, "risk_mod")


try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import roc_auc_score
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


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(stage: str, **kwargs: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    payload.update(kwargs)
    write_json(RESULT_DIR / "progress.json", payload)


def summarize_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "因子": col,
                "样本数": int(s.size),
                "均值": float(s.mean()),
                "中位数": float(s.median()),
                "25分位": float(s.quantile(0.25)),
                "75分位": float(s.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def summarize_delta(pos_df: pd.DataFrame, neg_df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        pos = pd.to_numeric(pos_df[col], errors="coerce").dropna()
        neg = pd.to_numeric(neg_df[col], errors="coerce").dropna()
        if pos.empty or neg.empty:
            continue
        rows.append(
            {
                "因子": col,
                "卖点均值": float(pos.mean()),
                "非卖点均值": float(neg.mean()),
                "均值差": float(pos.mean() - neg.mean()),
                "卖点中位数": float(pos.median()),
                "非卖点中位数": float(neg.median()),
                "中位数差": float(pos.median() - neg.median()),
            }
        )
    return pd.DataFrame(rows).sort_values("均值差", ascending=False)


def assign_split(df: pd.DataFrame, cutoffs: Dict[str, pd.Timestamp]) -> pd.Series:
    date_s = pd.to_datetime(df["sample_date"])
    out = pd.Series("", index=df.index, dtype="object")
    out[date_s <= cutoffs["research_end"]] = "research"
    out[(date_s >= cutoffs["validation_start"]) & (date_s <= cutoffs["validation_end"])] = "validation"
    out[date_s >= cutoffs["final_start"]] = "final_test"
    return out


def split_three_way_by_dates(dates: List[pd.Timestamp]) -> Dict[str, pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(pd.Series(dates)).drop_duplicates())
    n = len(unique_dates)
    research_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    research_end = min(max(1, research_end), n - 2)
    validation_end = min(max(research_end + 1, validation_end), n - 1)
    return {
        "research_end": unique_dates[research_end - 1],
        "validation_start": unique_dates[research_end],
        "validation_end": unique_dates[validation_end - 1],
        "final_start": unique_dates[validation_end],
        "final_end": unique_dates[-1],
    }


def build_sell_samples() -> pd.DataFrame:
    case_df = mod.parse_b1_case_files()
    mapping = mod.build_name_code_map()
    enriched = [mod.enrich_case(row, mapping) for _, row in case_df.iterrows()]
    case_enriched = pd.DataFrame(enriched)
    case_enriched.to_csv(RESULT_DIR / "case_manifest.csv", index=False, encoding="utf-8-sig")

    ok = case_enriched[
        (case_enriched["status"] == "ok")
        & (case_enriched["sample_type"] == "buy_sell")
        & (case_enriched.get("sell_found", False) == True)
        & (pd.to_numeric(case_enriched["hold_trading_days"], errors="coerce") >= MIN_HOLD_DAYS)
    ].copy()

    rows: List[Dict[str, Any]] = []
    total = len(ok)
    for i, (_, row) in enumerate(ok.iterrows(), 1):
        code = str(row["code"])
        path = mod.mod.DATA_DIR / f"{code}.txt"
        df = mod.mod.load_stock_data(str(path))
        if df is None or df.empty:
            continue
        feat = mod.mod.compute_b1_features(df).copy()
        base_df = feat[["open", "high", "low", "close", "volume", "trend_line", "long_line"]].copy()
        risk_df = risk_mod.add_risk_features(base_df, precomputed_base=base_df)

        ds = pd.to_datetime(feat["date"]).dt.strftime("%Y%m%d")
        buy_matches = np.flatnonzero(ds.to_numpy() == str(row["buy_date"]))
        sell_matches = np.flatnonzero(ds.to_numpy() == str(row["sell_date"]))
        if len(buy_matches) == 0 or len(sell_matches) == 0:
            continue
        buy_idx = int(buy_matches[-1])
        sell_idx = int(sell_matches[-1])
        if sell_idx <= buy_idx:
            continue

        hold_days = sell_idx - buy_idx
        post_buy_high = feat["high"].iloc[buy_idx : sell_idx + 1].cummax()

        def make_row(sample_idx: int, label: int, sample_role: str) -> Dict[str, Any]:
            feat_row = feat.iloc[sample_idx]
            risk_row = risk_df.iloc[sample_idx]
            buy_close = float(feat.iloc[buy_idx]["close"])
            cur_close = float(feat_row["close"])
            profit_since_buy = cur_close / buy_close - 1.0 if buy_close > 0 else np.nan
            peak_close = float(post_buy_high.iloc[sample_idx - buy_idx])
            drawdown_from_peak = cur_close / peak_close - 1.0 if peak_close > 0 else np.nan
            return {
                "stock_name": row["stock_name"],
                "code": code,
                "buy_date": pd.Timestamp(row["signal_date"]),
                "sell_date": pd.Timestamp(row["sell_date"]),
                "sample_date": pd.Timestamp(feat_row["date"]),
                "label": label,
                "sample_role": sample_role,
                "hold_day_idx": int(sample_idx - buy_idx),
                "profit_since_buy": profit_since_buy,
                "drawdown_from_peak": drawdown_from_peak,
                "J": float(feat_row["J"]) if pd.notna(feat_row["J"]) else np.nan,
                "ret1": float(feat_row["ret1"]) if pd.notna(feat_row["ret1"]) else np.nan,
                "ret3": float(feat_row["ret3"]) if pd.notna(feat_row["ret3"]) else np.nan,
                "ret5": float(feat_row["ret5"]) if pd.notna(feat_row["ret5"]) else np.nan,
                "ret10": float(feat_row["ret10"]) if pd.notna(feat_row["ret10"]) else np.nan,
                "signal_ret": float(feat_row["signal_ret"]) if pd.notna(feat_row["signal_ret"]) else np.nan,
                "trend_spread": float(feat_row["trend_spread"]) if pd.notna(feat_row["trend_spread"]) else np.nan,
                "close_to_trend": float(feat_row["close_to_trend"]) if pd.notna(feat_row["close_to_trend"]) else np.nan,
                "close_to_long": float(feat_row["close_to_long"]) if pd.notna(feat_row["close_to_long"]) else np.nan,
                "signal_vs_ma5": float(feat_row["signal_vs_ma5"]) if pd.notna(feat_row["signal_vs_ma5"]) else np.nan,
                "vol_vs_prev": float(feat_row["vol_vs_prev"]) if pd.notna(feat_row["vol_vs_prev"]) else np.nan,
                "body_ratio": float(feat_row["body_ratio"]) if pd.notna(feat_row["body_ratio"]) else np.nan,
                "upper_shadow_pct": float(feat_row["upper_shadow_pct"]) if pd.notna(feat_row["upper_shadow_pct"]) else np.nan,
                "lower_shadow_pct": float(feat_row["lower_shadow_pct"]) if pd.notna(feat_row["lower_shadow_pct"]) else np.nan,
                "close_location": float(feat_row["close_location"]) if pd.notna(feat_row["close_location"]) else np.nan,
                "ma5_slope_5": float(feat_row["ma5_slope_5"]) if pd.notna(feat_row["ma5_slope_5"]) else np.nan,
                "ma10_slope_5": float(feat_row["ma10_slope_5"]) if pd.notna(feat_row["ma10_slope_5"]) else np.nan,
                "ma20_slope_5": float(feat_row["ma20_slope_5"]) if pd.notna(feat_row["ma20_slope_5"]) else np.nan,
                "trend_slope_5": float(feat_row["trend_slope_5"]) if pd.notna(feat_row["trend_slope_5"]) else np.nan,
                "long_slope_5": float(feat_row["long_slope_5"]) if pd.notna(feat_row["long_slope_5"]) else np.nan,
                "risk_recent_heavy_bear_top_20": int(bool(risk_row["recent_heavy_bear_top_20"])),
                "risk_recent_failed_breakout_20": int(bool(risk_row["recent_failed_breakout_20"])),
                "risk_top_distribution_20": int(bool(risk_row["top_distribution_20"])),
                "risk_recent_stair_bear_20": int(bool(risk_row["recent_stair_bear_20"])),
                "risk_fast_rise_10d_40": int(bool(risk_row["risk_fast_rise_10d_40"])),
                "risk_segment_rise_slope_10_006": int(bool(risk_row["risk_segment_rise_slope_10_006"])),
                "risk_distribution_any_20": int(bool(risk_row["risk_distribution_any_20"])),
            }

        rows.append(make_row(sell_idx, 1, "sell"))

        candidates = list(range(buy_idx + 3, max(buy_idx + 3, sell_idx - 2)))
        if sell_idx - buy_idx > 8:
            filtered = [idx for idx in candidates if idx <= sell_idx - 3]
        else:
            filtered = candidates
        if not filtered:
            continue
        if len(filtered) > MAX_NEG_PER_CASE:
            choose_pos = np.linspace(0, len(filtered) - 1, MAX_NEG_PER_CASE, dtype=int)
            neg_indices = sorted({filtered[pos] for pos in choose_pos})
        else:
            neg_indices = filtered
        for neg_idx in neg_indices:
            rows.append(make_row(neg_idx, 0, "hold_not_sell"))

        if i % 20 == 0 or i == total:
            print(f"卖点样本构建进度: {i}/{total}")

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "hold_day_idx",
    "profit_since_buy",
    "drawdown_from_peak",
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
    "risk_recent_heavy_bear_top_20",
    "risk_recent_failed_breakout_20",
    "risk_top_distribution_20",
    "risk_recent_stair_bear_20",
    "risk_fast_rise_10d_40",
    "risk_segment_rise_slope_10_006",
    "risk_distribution_any_20",
]


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)


def score_rule(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    score += (df["profit_since_buy"] >= 0.40).astype(float) * 1.0
    score += (df["drawdown_from_peak"] <= -0.08).astype(float) * 1.2
    score += (df["signal_ret"] <= -0.03).astype(float) * 0.8
    score += (df["close_to_trend"] >= 0.08).astype(float) * 0.6
    score += (df["risk_distribution_any_20"] > 0).astype(float) * 0.8
    score += (df["risk_recent_failed_breakout_20"] > 0).astype(float) * 0.6
    score += (df["upper_shadow_pct"] >= 0.35).astype(float) * 0.6
    score += (df["hold_day_idx"] >= 25).astype(float) * 0.4
    return score


def fit_and_score(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, pd.Series]:
    scores: Dict[str, pd.Series] = {
        "rule_score": score_rule(test_df),
    }
    if not HAS_SKLEARN:
        return scores

    X_train = build_feature_matrix(train_df)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_test = build_feature_matrix(test_df)

    try:
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(X_train, y_train)
        scores["logistic_score"] = pd.Series(lr.predict_proba(X_test)[:, 1], index=test_df.index)
    except Exception:
        pass

    try:
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        scores["gnb_score"] = pd.Series(gnb.predict_proba(X_test)[:, 1], index=test_df.index)
    except Exception:
        pass

    try:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        scores["rf_score"] = pd.Series(rf.predict_proba(X_test)[:, 1], index=test_df.index)
    except Exception:
        pass

    try:
        et = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        et.fit(X_train, y_train)
        scores["et_score"] = pd.Series(et.predict_proba(X_test)[:, 1], index=test_df.index)
    except Exception:
        pass

    if HAS_LGB:
        try:
            clf = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                min_child_samples=5,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            scores["lgb_score"] = pd.Series(clf.predict_proba(X_test)[:, 1], index=test_df.index)
        except Exception:
            pass

    if HAS_XGB:
        try:
            clf = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
            clf.fit(X_train, y_train)
            scores["xgb_score"] = pd.Series(clf.predict_proba(X_test)[:, 1], index=test_df.index)
        except Exception:
            pass
    return scores


def eval_scores(df: pd.DataFrame, score_cols: List[str], split_name: str) -> pd.DataFrame:
    rows = []
    for col in score_cols:
        if col not in df.columns:
            continue
        sub = df[[col, "label"]].dropna()
        if sub.empty or sub["label"].nunique() < 2:
            continue
        auc = np.nan
        if HAS_SKLEARN:
            try:
                auc = float(roc_auc_score(sub["label"], sub[col]))
            except Exception:
                auc = np.nan
        top30 = sub.sort_values(col, ascending=False).head(min(30, len(sub)))
        rows.append(
            {
                "split": split_name,
                "score_col": col,
                "sample_count": int(len(sub)),
                "positive_rate": float(sub["label"].mean()),
                "auc": auc,
                "top30_sell_rate": float(top30["label"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["auc", "top30_sell_rate"], ascending=[False, False])


def main() -> None:
    update_progress("building_sell_samples")
    sample_df = build_sell_samples()
    cutoffs = split_three_way_by_dates(list(pd.to_datetime(sample_df["sample_date"])))
    write_json(RESULT_DIR / "split_cutoffs.json", {k: str(v.date()) for k, v in cutoffs.items()})
    sample_df["split"] = assign_split(sample_df, cutoffs)
    sample_df.to_csv(RESULT_DIR / "sell_samples.csv", index=False, encoding="utf-8-sig")

    pos_df = sample_df[sample_df["label"] == 1].copy()
    neg_df = sample_df[sample_df["label"] == 0].copy()
    summarize_numeric(pos_df, FEATURE_COLS).to_csv(RESULT_DIR / "sell_positive_summary.csv", index=False, encoding="utf-8-sig")
    summarize_numeric(neg_df, FEATURE_COLS).to_csv(RESULT_DIR / "sell_negative_summary.csv", index=False, encoding="utf-8-sig")
    summarize_delta(pos_df, neg_df, FEATURE_COLS).to_csv(RESULT_DIR / "sell_feature_delta.csv", index=False, encoding="utf-8-sig")
    update_progress("feature_summaries_ready", total_samples=int(len(sample_df)))

    research_df = sample_df[sample_df["split"] == "research"].copy()
    validation_df = sample_df[sample_df["split"] == "validation"].copy()
    final_df = sample_df[sample_df["split"] == "final_test"].copy()

    validation_scores = fit_and_score(research_df, validation_df)
    for k, s in validation_scores.items():
        validation_df[k] = s
    validation_report = eval_scores(validation_df, list(validation_scores.keys()), "validation")
    validation_report.to_csv(RESULT_DIR / "validation_model_report.csv", index=False, encoding="utf-8-sig")

    final_scores = fit_and_score(pd.concat([research_df, validation_df], ignore_index=True), final_df)
    for k, s in final_scores.items():
        final_df[k] = s
    final_report = eval_scores(final_df, list(final_scores.keys()), "final_test")
    final_report.to_csv(RESULT_DIR / "final_test_model_report.csv", index=False, encoding="utf-8-sig")

    if "rule_score" in final_df.columns:
        top_examples = final_df.sort_values("rule_score", ascending=False).head(50)
        top_examples.to_csv(RESULT_DIR / "final_test_top_rule_examples.csv", index=False, encoding="utf-8-sig")

    summary = {
        "result_dir": str(RESULT_DIR),
        "sample_count": int(len(sample_df)),
        "positive_count": int((sample_df["label"] == 1).sum()),
        "negative_count": int((sample_df["label"] == 0).sum()),
        "split_counts": {k: int(v) for k, v in sample_df["split"].value_counts().sort_index().to_dict().items()},
        "has_sklearn": HAS_SKLEARN,
        "has_lightgbm": HAS_LGB,
        "has_xgboost": HAS_XGB,
    }
    write_json(RESULT_DIR / "summary.json", summary)
    update_progress("finished", sample_count=int(len(sample_df)))


if __name__ == "__main__":
    main()
