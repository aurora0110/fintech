from __future__ import annotations

import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
PREV_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
PREV_RESULT_DIR = ROOT / "results" / "b1_similarity_ml_signal_20260320_164521"
B1_PERFECT_DIR = ROOT / "data" / "完美图" / "B1"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = ROOT / "results" / f"b1_similarity_ml_signal_v2_{RUN_TS}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEGATIVE_MULTIPLIER = 5
DAILY_TOPN_LIST = [3, 5, 8, 10, 20, 50]
SIMILARITY_VARIANTS = [
    ("corr", "close_norm"),
    ("corr", "close_vol_concat"),
    ("cosine", "close_norm"),
    ("cosine", "close_vol_concat"),
    ("euclidean", "close_norm"),
    ("euclidean", "close_vol_concat"),
]


def load_prev_module():
    spec = importlib.util.spec_from_file_location("b1_prev", PREV_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = load_prev_module()


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(stage: str, **kwargs: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    payload.update(kwargs)
    write_json(RESULT_DIR / "progress.json", payload)


def parse_b1_case_files() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in sorted(B1_PERFECT_DIR.glob("*.png")):
        name = p.name
        m2 = re.match(r"(.+?)(\d{8})(\d{8})\.png$", name)
        if m2:
            rows.append(
                {
                    "stock_name": m2.group(1).strip(),
                    "buy_date": m2.group(2),
                    "sell_date": m2.group(3),
                    "sample_type": "buy_sell",
                    "image_name": name,
                }
            )
            continue
        m1 = re.match(r"(.+?)(\d{8})\.png$", name)
        if m1:
            rows.append(
                {
                    "stock_name": m1.group(1).strip(),
                    "buy_date": m1.group(2),
                    "sell_date": "",
                    "sample_type": "buy_only",
                    "image_name": name,
                }
            )
            continue
        rows.append(
            {
                "stock_name": re.sub(r"\.png$", "", name),
                "buy_date": "",
                "sell_date": "",
                "sample_type": "no_date",
                "image_name": name,
            }
        )
    return pd.DataFrame(rows)


def build_name_code_map() -> Dict[str, str]:
    return mod.load_name_code_map()


def resolve_code(stock_name: str, mapping: Dict[str, str]) -> Optional[str]:
    alias = mod.CASE_NAME_ALIASES.get(stock_name, stock_name)
    if alias in mapping:
        return mapping[alias]
    for mapped_name, code in mapping.items():
        if alias in mapped_name or mapped_name in alias:
            return code
    return None


def enrich_case(row: pd.Series, mapping: Dict[str, str]) -> Dict[str, Any]:
    out = row.to_dict()
    if row["sample_type"] == "no_date":
        out["status"] = "no_date"
        out["code"] = ""
        return out

    code = resolve_code(str(row["stock_name"]), mapping)
    out["code"] = code or ""
    if not code:
        out["status"] = "name_unmapped"
        return out

    path = mod.DATA_DIR / f"{code}.txt"
    if not path.exists():
        out["status"] = "normal_missing"
        return out

    df = mod.load_stock_data(str(path))
    if df is None or df.empty:
        out["status"] = "load_failed"
        return out

    feat = mod.compute_b1_features(df)
    ds = pd.to_datetime(feat["date"]).dt.strftime("%Y%m%d")
    idxs = np.flatnonzero(ds.to_numpy() == str(row["buy_date"]))
    if len(idxs) == 0:
        out["status"] = "buy_date_missing"
        return out

    idx = int(idxs[-1])
    if idx < mod.SEQ_LEN - 1:
        out["status"] = "bars_insufficient"
        return out

    feat_row = feat.iloc[idx]
    metrics = mod.future_metrics(feat, idx)
    seq_map = mod.extract_sequence(feat.iloc[idx - mod.SEQ_LEN + 1 : idx + 1])

    out.update(
        {
            "status": "ok",
            "signal_date": pd.Timestamp(feat_row["date"]),
            "entry_date_model": metrics.get("entry_date"),
            "entry_price_model": metrics.get("entry_price"),
            "stop_loss_price_model": metrics.get("stop_loss_price"),
            "ret_20d_model": metrics.get("ret_20d"),
            "ret_30d_model": metrics.get("ret_30d"),
            "J": float(feat_row["J"]) if pd.notna(feat_row["J"]) else np.nan,
            "ret1": float(feat_row["ret1"]) if pd.notna(feat_row["ret1"]) else 0.0,
            "ret3": float(feat_row["ret3"]) if pd.notna(feat_row["ret3"]) else 0.0,
            "ret5": float(feat_row["ret5"]) if pd.notna(feat_row["ret5"]) else 0.0,
            "ret10": float(feat_row["ret10"]) if pd.notna(feat_row["ret10"]) else 0.0,
            "signal_ret": float(feat_row["signal_ret"]) if pd.notna(feat_row["signal_ret"]) else 0.0,
            "trend_spread": float(feat_row["trend_spread"]) if pd.notna(feat_row["trend_spread"]) else 0.0,
            "close_to_trend": float(feat_row["close_to_trend"]) if pd.notna(feat_row["close_to_trend"]) else 0.0,
            "close_to_long": float(feat_row["close_to_long"]) if pd.notna(feat_row["close_to_long"]) else 0.0,
            "signal_vs_ma5": float(feat_row["signal_vs_ma5"]) if pd.notna(feat_row["signal_vs_ma5"]) else 0.0,
            "vol_vs_prev": float(feat_row["vol_vs_prev"]) if pd.notna(feat_row["vol_vs_prev"]) else 0.0,
            "body_ratio": float(feat_row["body_ratio"]) if pd.notna(feat_row["body_ratio"]) else 0.0,
            "upper_shadow_pct": float(feat_row["upper_shadow_pct"]) if pd.notna(feat_row["upper_shadow_pct"]) else 0.0,
            "lower_shadow_pct": float(feat_row["lower_shadow_pct"]) if pd.notna(feat_row["lower_shadow_pct"]) else 0.0,
            "close_location": float(feat_row["close_location"]) if pd.notna(feat_row["close_location"]) else 0.0,
            "ma5_slope_5": float(feat_row["ma5_slope_5"]) if pd.notna(feat_row["ma5_slope_5"]) else 0.0,
            "ma10_slope_5": float(feat_row["ma10_slope_5"]) if pd.notna(feat_row["ma10_slope_5"]) else 0.0,
            "ma20_slope_5": float(feat_row["ma20_slope_5"]) if pd.notna(feat_row["ma20_slope_5"]) else 0.0,
            "trend_slope_5": float(feat_row["trend_slope_5"]) if pd.notna(feat_row["trend_slope_5"]) else 0.0,
            "long_slope_5": float(feat_row["long_slope_5"]) if pd.notna(feat_row["long_slope_5"]) else 0.0,
            "seq_map": seq_map,
        }
    )

    if row["sample_type"] == "buy_sell" and row["sell_date"]:
        sell_idxs = np.flatnonzero(ds.to_numpy() == str(row["sell_date"]))
        if len(sell_idxs) > 0:
            sell_idx = int(sell_idxs[-1])
            out["sell_found"] = True
            out["sell_index"] = sell_idx
            out["hold_calendar_days"] = int((pd.Timestamp(row["sell_date"]) - pd.Timestamp(row["buy_date"])).days)
            out["hold_trading_days"] = int(sell_idx - idx)
            buy_close = float(feat.iloc[idx]["close"])
            sell_close = float(feat.iloc[sell_idx]["close"])
            out["sell_close_return"] = sell_close / buy_close - 1.0
            out["sell_signal_ret"] = float(feat.iloc[sell_idx]["signal_ret"]) if pd.notna(feat.iloc[sell_idx]["signal_ret"]) else np.nan
            out["sell_close_to_trend"] = float(feat.iloc[sell_idx]["close_to_trend"]) if pd.notna(feat.iloc[sell_idx]["close_to_trend"]) else np.nan
        else:
            out["sell_found"] = False
    return out


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


def build_feature_matrix(df: pd.DataFrame, include_similarity: bool) -> tuple[np.ndarray, List[str]]:
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
        cols.extend([f"sim_{scorer}_{rep}" for scorer, rep in SIMILARITY_VARIANTS])
    return df[cols].fillna(0.0).to_numpy(dtype=float), cols


def assign_split(df: pd.DataFrame, cutoffs: Dict[str, pd.Timestamp]) -> pd.Series:
    date_s = pd.to_datetime(df["signal_date"])
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


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, max_iter: int = 400, lr: float = 0.05, l2: float = 1e-3) -> Dict[str, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Xs = (X - mean) / std

    w = np.zeros(Xs.shape[1], dtype=float)
    b = 0.0
    for _ in range(max_iter):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = (Xs.T @ (p - y)) / len(y) + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return {"mean": mean, "std": std, "w": w, "b": np.array([b], dtype=float)}


def predict_logistic(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    mean = model["mean"]
    std = model["std"]
    w = model["w"]
    b = float(model["b"][0])
    Xs = (np.asarray(X, dtype=float) - mean) / std
    z = Xs @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def main() -> None:
    update_progress("starting")
    candidate_df = pd.read_pickle(PREV_RESULT_DIR / "candidate_rows.pkl")
    candidate_df["signal_date"] = pd.to_datetime(candidate_df["signal_date"])
    update_progress("candidate_loaded", candidate_signal_count=int(len(candidate_df)))

    name_code_map = build_name_code_map()
    raw_cases = parse_b1_case_files()
    enriched_cases = pd.DataFrame([enrich_case(r, name_code_map) for _, r in raw_cases.iterrows()])
    enriched_cases.to_csv(RESULT_DIR / "b1_case_manifest.csv", index=False, encoding="utf-8-sig")
    update_progress(
        "cases_enriched",
        total_cases=int(len(enriched_cases)),
        ok_cases=int((enriched_cases["status"] == "ok").sum()),
        buy_sell_cases=int((enriched_cases["sample_type"] == "buy_sell").sum()),
    )

    positive_df = enriched_cases[enriched_cases["status"] == "ok"].copy().reset_index(drop=True)
    positive_df["signal_date"] = pd.to_datetime(positive_df["signal_date"])
    cutoffs = split_three_way_by_dates(list(positive_df["signal_date"]))
    candidate_df["split"] = assign_split(candidate_df, cutoffs)
    positive_df["split"] = assign_split(positive_df, cutoffs)
    write_json(RESULT_DIR / "split_cutoffs.json", cutoffs)
    update_progress("split_ready", cutoffs=cutoffs)

    buy_feature_cols = [
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
        "ret_20d_model",
        "ret_30d_model",
    ]
    summarize_numeric(positive_df, buy_feature_cols).to_csv(
        RESULT_DIR / "buy_point_feature_summary.csv", index=False, encoding="utf-8-sig"
    )

    buy_sell_df = positive_df[positive_df["sample_type"] == "buy_sell"].copy()
    if not buy_sell_df.empty:
        habit_cols = ["hold_calendar_days", "hold_trading_days", "sell_close_return", "sell_signal_ret", "sell_close_to_trend"]
        summarize_numeric(buy_sell_df, habit_cols).to_csv(
            RESULT_DIR / "trading_habit_summary.csv", index=False, encoding="utf-8-sig"
        )
        buy_sell_df[
            [
                "stock_name",
                "code",
                "signal_date",
                "sell_date",
                "hold_calendar_days",
                "hold_trading_days",
                "sell_close_return",
                "sell_signal_ret",
                "sell_close_to_trend",
            ]
        ].to_csv(RESULT_DIR / "trading_habit_rows.csv", index=False, encoding="utf-8-sig")
    update_progress("habit_analysis_done")

    template_rows = positive_df[positive_df["split"] == "research"].to_dict("records")
    if not template_rows:
        raise ValueError("research 段没有正样本，无法构建模板")

    for scorer, rep in SIMILARITY_VARIANTS:
        templates = mod.stack_templates(template_rows, rep)
        seqs = np.vstack([m[rep] for m in candidate_df["seq_map"]])
        candidate_df[f"sim_{scorer}_{rep}"] = mod.similarity_scores(seqs, templates, scorer)
        pos_seqs = np.vstack([m[rep] for m in positive_df["seq_map"]])
        positive_df[f"sim_{scorer}_{rep}"] = mod.similarity_scores(pos_seqs, templates, scorer)
    update_progress("similarity_scores_ready")

    research_positive = positive_df[positive_df["split"] == "research"].copy()
    research_negative = candidate_df[(candidate_df["split"] == "research") & (candidate_df["negative_30d"])].copy()
    max_negative = max(len(research_positive) * MAX_NEGATIVE_MULTIPLIER, len(research_positive))
    research_negative = research_negative.sort_values("signal_date").head(max_negative).copy()

    train_plain = pd.concat(
        [
            research_positive.assign(label=1, source="perfect_positive"),
            research_negative.assign(label=0, source="auto_negative"),
        ],
        ignore_index=True,
    )

    X_plain, feature_cols_plain = build_feature_matrix(train_plain, include_similarity=False)
    y_plain = train_plain["label"].to_numpy(dtype=float)
    plain_model = fit_logistic_regression(X_plain, y_plain)

    X_mix, feature_cols_mix = build_feature_matrix(train_plain, include_similarity=True)
    mix_model = fit_logistic_regression(X_mix, y_plain)
    update_progress("ml_models_trained", positive_train_count=int(len(research_positive)), negative_train_count=int(len(research_negative)))

    for split_name in ["validation", "final_test"]:
        part = candidate_df[candidate_df["split"] == split_name].copy()
        Xp, _ = build_feature_matrix(part, include_similarity=False)
        Xm, _ = build_feature_matrix(part, include_similarity=True)
        candidate_df.loc[part.index, "ml_plain_score"] = predict_logistic(plain_model, Xp)
        candidate_df.loc[part.index, "ml_mix_score"] = predict_logistic(mix_model, Xm)

    baseline_summary = []
    for split_name in ["validation", "final_test"]:
        part = candidate_df[candidate_df["split"] == split_name].copy()
        row = {"family": "baseline", "variant": "all_candidates", "topn": 0, "split": split_name}
        row.update(summarize_signal_df(part))
        baseline_summary.append(row)
        for topn in DAILY_TOPN_LIST:
            top_df = part.sort_values(["signal_date", "J", "code"], ascending=[True, True, True]).groupby("signal_date", group_keys=False).head(topn)
            row = {"family": "baseline", "variant": "lowest_J", "topn": topn, "split": split_name}
            row.update(summarize_signal_df(top_df))
            baseline_summary.append(row)

    sim_rows: List[Dict[str, Any]] = []
    for split_name in ["validation", "final_test"]:
        part = candidate_df[candidate_df["split"] == split_name].copy()
        for scorer, rep in SIMILARITY_VARIANTS:
            score_col = f"sim_{scorer}_{rep}"
            for topn in DAILY_TOPN_LIST:
                top_df = select_daily_topn(part, score_col, topn)
                row = {"family": "similarity", "variant": f"{scorer}_{rep}", "topn": topn, "split": split_name}
                row.update(summarize_signal_df(top_df))
                sim_rows.append(row)

    ml_rows: List[Dict[str, Any]] = []
    for split_name in ["validation", "final_test"]:
        part = candidate_df[candidate_df["split"] == split_name].copy()
        for score_col, family, variant in [
            ("ml_plain_score", "ml", "logistic_features_only"),
            ("ml_mix_score", "ml_plus_similarity", "logistic_features_plus_similarity"),
        ]:
            for topn in DAILY_TOPN_LIST:
                top_df = select_daily_topn(part, score_col, topn)
                row = {"family": family, "variant": variant, "topn": topn, "split": split_name}
                row.update(summarize_signal_df(top_df))
                ml_rows.append(row)

    leaderboard_df = pd.concat(
        [pd.DataFrame(baseline_summary), pd.DataFrame(sim_rows), pd.DataFrame(ml_rows)],
        ignore_index=True,
    )
    leaderboard_df.to_csv(RESULT_DIR / "signal_layer_leaderboard.csv", index=False, encoding="utf-8-sig")

    validation_df = leaderboard_df[leaderboard_df["split"] == "validation"].copy()
    validation_best_rows: List[pd.Series] = []
    for family in ["baseline", "similarity", "ml", "ml_plus_similarity"]:
        fam = validation_df[validation_df["family"] == family].copy()
        if fam.empty:
            continue
        fam = fam.sort_values(["ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[False, False, False])
        validation_best_rows.append(fam.iloc[0])
    validation_best_df = pd.DataFrame(validation_best_rows)
    validation_best_df.to_csv(RESULT_DIR / "validation_family_best.csv", index=False, encoding="utf-8-sig")

    final_part = candidate_df[candidate_df["split"] == "final_test"].copy()
    final_reports: List[Dict[str, Any]] = []
    selected_frames: List[pd.DataFrame] = []
    final_positive = positive_df[positive_df["split"] == "final_test"][["stock_name", "code", "signal_date"]].copy()
    recall_rows: List[Dict[str, Any]] = []

    for _, best_row in validation_best_df.iterrows():
        family = str(best_row["family"])
        variant = str(best_row["variant"])
        topn = int(best_row["topn"])
        if family == "baseline" and variant == "all_candidates":
            selected = final_part.copy()
        elif family == "baseline":
            selected = final_part.sort_values(["signal_date", "J", "code"], ascending=[True, True, True]).groupby("signal_date", group_keys=False).head(topn)
        elif family == "similarity":
            selected = select_daily_topn(final_part, f"sim_{variant}", topn)
        elif family == "ml":
            selected = select_daily_topn(final_part, "ml_plain_score", topn)
        else:
            selected = select_daily_topn(final_part, "ml_mix_score", topn)

        report = {"family": family, "variant": variant, "topn": topn}
        report.update(summarize_signal_df(selected))
        final_reports.append(report)

        hits = final_positive.merge(selected[["code", "signal_date"]].drop_duplicates(), on=["code", "signal_date"], how="inner")
        recall_rows.append(
            {
                "family": family,
                "variant": variant,
                "topn": topn,
                "hit_final_positive": int(len(hits)),
                "final_positive_total": int(len(final_positive)),
            }
        )

        tmp = selected.drop(columns=["seq_map"]).copy()
        tmp["strategy_tag"] = f"{family}_{variant}_top{topn}"
        selected_frames.append(tmp)

    pd.DataFrame(final_reports).to_csv(RESULT_DIR / "final_test_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(recall_rows).to_csv(RESULT_DIR / "final_test_recall.csv", index=False, encoding="utf-8-sig")
    if selected_frames:
        pd.concat(selected_frames, ignore_index=True).to_csv(RESULT_DIR / "final_test_selected_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "result_dir": str(RESULT_DIR),
        "candidate_signal_count": int(len(candidate_df)),
        "candidate_split_counts": candidate_df["split"].value_counts().sort_index().to_dict(),
        "positive_total": int(len(positive_df)),
        "positive_split_counts": positive_df["split"].value_counts().sort_index().to_dict(),
        "sample_type_counts": positive_df["sample_type"].value_counts().sort_index().to_dict(),
        "negative_train_count": int(len(research_negative)),
        "positive_train_count": int(len(research_positive)),
    }
    write_json(RESULT_DIR / "summary.json", summary)
    update_progress("finished", summary=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
