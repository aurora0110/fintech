from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
V2_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v2_20260320.py"
SEMANTIC_SCRIPT = ROOT / "utils" / "tmp" / "b1_semantic_shared_20260320.py"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = ROOT / "results" / f"b1_sell_habit_experiment_v2_{RUN_TS}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEG_PER_CASE = 5
MIN_HOLD_DAYS = 8


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v2_mod = load_module(V2_SCRIPT, "b1_sell_v2_case")
sem_mod = load_module(SEMANTIC_SCRIPT, "b1_sell_v2_sem")

HAS_SKLEARN = bool(getattr(v2_mod.mod, "HAS_SKLEARN", False)) if hasattr(v2_mod, "mod") else False
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
    research_end = min(max(1, int(n * 0.60)), n - 2)
    validation_end = min(max(research_end + 1, int(n * 0.80)), n - 1)
    return {
        "research_end": unique_dates[research_end - 1],
        "validation_start": unique_dates[research_end],
        "validation_end": unique_dates[validation_end - 1],
        "final_start": unique_dates[validation_end],
        "final_end": unique_dates[-1],
    }


def build_sell_samples() -> pd.DataFrame:
    case_df = v2_mod.parse_b1_case_files()
    mapping = v2_mod.build_name_code_map()
    enriched = [v2_mod.enrich_case(r, mapping) for _, r in case_df.iterrows()]
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
        path = sem_mod.base_mod.DATA_DIR / f"{code}.txt"
        df = sem_mod.base_mod.load_stock_data(str(path))
        if df is None or df.empty:
            continue
        feat = sem_mod.add_semantic_buy_features(df).copy()

        ds = pd.to_datetime(feat["date"]).dt.strftime("%Y%m%d")
        buy_matches = np.flatnonzero(ds.to_numpy() == str(row["buy_date"]))
        sell_matches = np.flatnonzero(ds.to_numpy() == str(row["sell_date"]))
        if len(buy_matches) == 0 or len(sell_matches) == 0:
            continue
        buy_idx = int(buy_matches[-1])
        sell_idx = int(sell_matches[-1])
        if sell_idx <= buy_idx:
            continue

        def make_row(sample_idx: int, label: int, sample_role: str) -> Dict[str, Any]:
            out = sem_mod.build_daily_sell_semantic_features(feat, buy_idx, sample_idx)
            out.update(
                {
                    "stock_name": row["stock_name"],
                    "code": code,
                    "buy_date": pd.Timestamp(row["signal_date"]),
                    "sell_date": pd.Timestamp(row["sell_date"]),
                    "sample_date": pd.Timestamp(feat.iloc[sample_idx]["date"]),
                    "label": label,
                    "sample_role": sample_role,
                }
            )
            return out

        rows.append(make_row(sell_idx, 1, "sell"))

        candidates = list(range(buy_idx + 3, max(buy_idx + 3, sell_idx - 2)))
        filtered = [idx for idx in candidates if idx <= sell_idx - 3] if sell_idx - buy_idx > 8 else candidates
        if filtered:
            if len(filtered) > MAX_NEG_PER_CASE:
                choose_pos = np.linspace(0, len(filtered) - 1, MAX_NEG_PER_CASE, dtype=int)
                neg_indices = sorted({filtered[pos] for pos in choose_pos})
            else:
                neg_indices = filtered
            for neg_idx in neg_indices:
                rows.append(make_row(neg_idx, 0, "hold_not_sell"))

        if i % 20 == 0 or i == total:
            print(f"语义卖点样本构建进度: {i}/{total}")

    return pd.DataFrame(rows)


FEATURE_COLS = sem_mod.SELL_FEATURE_COLS_V2


def available_feature_cols(*dfs: pd.DataFrame) -> List[str]:
    cols = FEATURE_COLS.copy()
    for df in dfs:
        cols = [c for c in cols if c in df.columns]
    return cols


def build_feature_matrix(df: pd.DataFrame, force_cols: List[str] | None = None) -> np.ndarray:
    cols = force_cols if force_cols is not None else available_feature_cols(df)
    return df[cols].fillna(0.0).to_numpy(dtype=float)


def fit_and_score(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, pd.Series]:
    scores: Dict[str, pd.Series] = {
        "rule_score_v2": sem_mod.score_sell_rule_v2(test_df),
    }
    if not HAS_SKLEARN:
        return scores

    cols = available_feature_cols(train_df, test_df)
    X_train = build_feature_matrix(train_df, force_cols=cols)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_test = build_feature_matrix(test_df, force_cols=cols)

    try:
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(X_train, y_train)
        scores["logistic_score_v2"] = pd.Series(lr.predict_proba(X_test)[:, 1], index=test_df.index)
    except Exception:
        pass

    try:
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        scores["gnb_score_v2"] = pd.Series(gnb.predict_proba(X_test)[:, 1], index=test_df.index)
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
        scores["rf_score_v2"] = pd.Series(rf.predict_proba(X_test)[:, 1], index=test_df.index)
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
        scores["et_score_v2"] = pd.Series(et.predict_proba(X_test)[:, 1], index=test_df.index)
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
            scores["lgb_score_v2"] = pd.Series(clf.predict_proba(X_test)[:, 1], index=test_df.index)
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
            scores["xgb_score_v2"] = pd.Series(clf.predict_proba(X_test)[:, 1], index=test_df.index)
        except Exception:
            pass
    return scores


def evaluate_scores(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in score_cols:
        if col not in df.columns:
            continue
        sub = df[[col, "label"]].dropna().copy()
        if sub.empty or sub["label"].nunique() < 2:
            continue
        auc = roc_auc_score(sub["label"], sub[col])
        rows.append(
            {
                "score_col": col,
                "sample_count": int(len(sub)),
                "positive_count": int(sub["label"].sum()),
                "auc": float(auc),
            }
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def main() -> None:
    update_progress("starting")
    sample_df = build_sell_samples()
    if sample_df.empty:
        raise ValueError("语义卖点样本为空")
    cutoffs = split_three_way_by_dates(sample_df["sample_date"].tolist())
    sample_df["split"] = assign_split(sample_df, cutoffs)
    sample_df.to_csv(RESULT_DIR / "sell_samples.csv", index=False, encoding="utf-8-sig")
    write_json(RESULT_DIR / "split_cutoffs.json", {k: str(v.date()) for k, v in cutoffs.items()})
    update_progress("samples_ready", sample_count=int(len(sample_df)))

    pos_df = sample_df[sample_df["label"] == 1].copy()
    neg_df = sample_df[sample_df["label"] == 0].copy()
    summarize_numeric(pos_df, FEATURE_COLS).to_csv(RESULT_DIR / "sell_feature_summary.csv", index=False, encoding="utf-8-sig")
    summarize_delta(pos_df, neg_df, FEATURE_COLS).to_csv(RESULT_DIR / "sell_feature_delta.csv", index=False, encoding="utf-8-sig")

    research_df = sample_df[sample_df["split"] == "research"].copy()
    validation_df = sample_df[sample_df["split"] == "validation"].copy()
    final_df = sample_df[sample_df["split"] == "final_test"].copy()

    validation_scores = fit_and_score(research_df, validation_df)
    for k, s in validation_scores.items():
        validation_df[k] = s
    validation_cols = list(validation_scores.keys())
    val_report = evaluate_scores(validation_df, validation_cols)
    val_report.to_csv(RESULT_DIR / "validation_model_report.csv", index=False, encoding="utf-8-sig")

    train_df = sample_df[sample_df["split"].isin(["research", "validation"])].copy()
    final_scores = fit_and_score(train_df, final_df)
    for k, s in final_scores.items():
        final_df[k] = s
    final_cols = list(final_scores.keys())
    final_report = evaluate_scores(final_df, final_cols)
    final_report.to_csv(RESULT_DIR / "final_test_model_report.csv", index=False, encoding="utf-8-sig")

    summary = {
        "sample_count": int(len(sample_df)),
        "sell_positive_count": int(sample_df["label"].sum()),
        "hold_negative_count": int((sample_df["label"] == 0).sum()),
        "feature_count": int(len(FEATURE_COLS)),
        "best_validation_row": val_report.iloc[0].to_dict() if not val_report.empty else {},
        "best_final_test_row": final_report.iloc[0].to_dict() if not final_report.empty else {},
    }
    write_json(RESULT_DIR / "summary.json", summary)
    update_progress("finished")


if __name__ == "__main__":
    main()
