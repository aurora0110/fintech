from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
BASE_SIGNAL_DIR = ROOT / "results" / "b1_full_factor_signal_v6_full_20260321_102049"
BASE_CANDIDATE_PKL = ROOT / "results" / "b1_similarity_ml_signal_20260320_162022" / "candidate_rows.pkl"
BASE_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
V4_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v4_20260320.py"
SEMANTIC_SCRIPT = ROOT / "utils" / "tmp" / "b1_semantic_shared_20260320.py"
TXT_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_txt_joint_opt_v1_20260322.py"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_RESULT_DIR = ROOT / "results" / f"b1_txt_template_split_signal_v1_{RUN_TS}"

TARGET_REP_METHODS = [
    ("corr", "close_vol_concat"),
    ("cosine", "close_vol_concat"),
    ("weighted_corr", "close_vol_concat"),
    ("lag_corr", "close_vol_concat"),
]
DEFAULT_TOPN_LIST = [3, 5, 8]
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base_mod = load_module(BASE_SCRIPT, "b1_txt_template_base")
v4_mod = load_module(V4_SCRIPT, "b1_txt_template_v4")
sem_mod = load_module(SEMANTIC_SCRIPT, "b1_txt_template_sem")
txt_mod = load_module(TXT_SCRIPT, "b1_txt_template_txt")

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
    parser = argparse.ArgumentParser(description="B1 文本正例分模板库相似度 + 近似反例增强优化")
    parser.add_argument("--base-signal-dir", type=Path, default=BASE_SIGNAL_DIR)
    parser.add_argument("--base-candidate-pkl", type=Path, default=BASE_CANDIDATE_PKL)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--file-limit", type=int, default=0)
    parser.add_argument("--topn-list", type=str, default="")
    parser.add_argument("--template-kind", choices=["trend", "long"], required=True)
    return parser.parse_args()


def filter_positive_templates(pos_txt: pd.DataFrame, template_kind: str) -> pd.DataFrame:
    if template_kind == "trend":
        return pos_txt[pos_txt["reason_trend_pullback"].fillna(False)].copy()
    return pos_txt[pos_txt["reason_long_pullback"].fillna(False)].copy()


def kind_pool_names(template_kind: str) -> Tuple[str, str]:
    if template_kind == "trend":
        return "pool_txt_trend", "pool_txt_trend_confirmed"
    return "pool_txt_long", "pool_txt_long_confirmed"


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
        if item:
            vals.append(int(item))
    return sorted(set(v for v in vals if v > 0))


def ecdf_score(train_vals: pd.Series, values: pd.Series) -> pd.Series:
    arr = np.sort(pd.to_numeric(train_vals, errors="coerce").dropna().astype(float).to_numpy())
    if len(arr) == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    pos = np.searchsorted(arr, x, side="right")
    score = pos / float(len(arr))
    score[~np.isfinite(x)] = 0.5
    return pd.Series(score, index=values.index, dtype=float)


def load_base_candidates(base_signal_dir: Path, file_limit: int, keep_codes: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(base_signal_dir / "candidate_enriched.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    mask = (df["signal_date"] < EXCLUDE_START) | (df["signal_date"] > EXCLUDE_END)
    df = df.loc[mask].copy()
    if file_limit > 0:
        first_codes = sorted(df["code"].astype(str).drop_duplicates().tolist())[:file_limit]
        keep = sorted(set(first_codes) | {str(c) for c in keep_codes if str(c)})
        df = df[df["code"].astype(str).isin(keep)].copy()
    return df.reset_index(drop=True)


def recover_seq_map(candidate_df: pd.DataFrame, pkl_path: Path) -> pd.DataFrame:
    raw_df = pd.read_pickle(pkl_path)[["code", "signal_date", "seq_map"]].copy()
    raw_df["signal_date"] = pd.to_datetime(raw_df["signal_date"])
    merged = candidate_df.merge(raw_df, on=["code", "signal_date"], how="left")
    if "seq_map_y" in merged.columns:
        merged["seq_map"] = merged["seq_map_y"]
        merged = merged.drop(columns=[c for c in ["seq_map_x", "seq_map_y"] if c in merged.columns])
    elif "seq_map" not in merged.columns:
        merged["seq_map"] = None
    merged["has_seq_map"] = merged["seq_map"].notna()
    return merged


def infer_split_windows(candidate_df: pd.DataFrame) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split in ["research", "validation", "final_test"]:
        sub = candidate_df[candidate_df["split"] == split]
        if sub.empty:
            continue
        windows[split] = (pd.Timestamp(sub["signal_date"].min()), pd.Timestamp(sub["signal_date"].max()))
    return windows


def assign_split_by_date(date_val: pd.Timestamp, windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    for split, (start, end) in windows.items():
        if start <= date_val <= end:
            return split
    return ""


def build_label_seq_feature_df(
    codes: Iterable[str],
    split_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    target_dates_by_code: Optional[Dict[str, List[pd.Timestamp]]] = None,
    day_window: int = 5,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    keep_codes = sorted({str(c) for c in codes if str(c)})
    total = len(keep_codes)
    for i, code in enumerate(keep_codes, 1):
        path = base_mod.DATA_DIR / f"{code}.txt"
        if not path.exists():
            continue
        df = base_mod.load_stock_data(str(path))
        if df is None or df.empty:
            continue
        feat = sem_mod.add_semantic_buy_features(df)
        target_dates = [pd.Timestamp(x) for x in (target_dates_by_code or {}).get(code, [])]
        start_idx = max(base_mod.MIN_BARS, base_mod.SEQ_LEN - 1)
        end_idx = len(feat) - (sem_mod.BUY_DELAY_DAYS + 1)
        if target_dates:
            ds = pd.to_datetime(feat["date"])
            mask = pd.Series(False, index=feat.index)
            for dt in target_dates:
                mask = mask | ds.between(dt - pd.Timedelta(days=day_window + 2), dt + pd.Timedelta(days=day_window))
            idx_list = [int(j) for j in np.flatnonzero(mask.to_numpy()) if start_idx <= int(j) < end_idx]
        else:
            idx_list = list(range(start_idx, end_idx))
        for idx in idx_list:
            metrics = sem_mod.future_metrics(feat, idx)
            if not metrics:
                continue
            row = feat.iloc[idx]
            seq_window = feat.iloc[idx - base_mod.SEQ_LEN + 1 : idx + 1]
            if len(seq_window) != base_mod.SEQ_LEN:
                continue
            seq_map = base_mod.extract_sequence(seq_window)
            rep_map = v4_mod.derive_rep_map(seq_map)
            signal_date = pd.Timestamp(row["date"])
            entry_date = pd.Timestamp(metrics["entry_date"])
            rows.append(
                {
                    "code": code,
                    "signal_date": signal_date,
                    "signal_idx": int(idx),
                    "entry_date": entry_date,
                    "entry_price": metrics["entry_price"],
                    "stop_loss_price": metrics["stop_loss_price"],
                    "ret_3d": metrics["ret_3d"],
                    "ret_5d": metrics["ret_5d"],
                    "ret_10d": metrics["ret_10d"],
                    "ret_20d": metrics["ret_20d"],
                    "ret_30d": metrics["ret_30d"],
                    "up_3d": metrics["up_3d"],
                    "up_5d": metrics["up_5d"],
                    "up_10d": metrics["up_10d"],
                    "up_20d": metrics["up_20d"],
                    "up_30d": metrics["up_30d"],
                    "min_close_ret_30": metrics["min_close_ret_30"],
                    "max_high_ret_30": metrics["max_high_ret_30"],
                    "negative_30d": metrics["negative_30d"],
                    "split": assign_split_by_date(signal_date, split_windows),
                    "J": float(row["J"]) if pd.notna(row["J"]) else np.nan,
                    "ret1": float(row["ret1"]) if pd.notna(row["ret1"]) else 0.0,
                    "ret3": float(row["ret3"]) if pd.notna(row["ret3"]) else 0.0,
                    "ret5": float(row["ret5"]) if pd.notna(row["ret5"]) else 0.0,
                    "ret10": float(row["ret10"]) if pd.notna(row["ret10"]) else 0.0,
                    "ret20": float(row["ret20"]) if pd.notna(row["ret20"]) else 0.0,
                    "ret30": float(row["ret30"]) if pd.notna(row["ret30"]) else 0.0,
                    "signal_ret": float(row["signal_ret"]) if pd.notna(row["signal_ret"]) else 0.0,
                    "trend_spread": float(row["trend_spread"]) if pd.notna(row["trend_spread"]) else 0.0,
                    "close_to_trend": float(row["close_to_trend"]) if pd.notna(row["close_to_trend"]) else 0.0,
                    "close_to_long": float(row["close_to_long"]) if pd.notna(row["close_to_long"]) else 0.0,
                    "signal_vs_ma5": float(row["signal_vs_ma5"]) if pd.notna(row["signal_vs_ma5"]) else 0.0,
                    "vol_vs_prev": float(row["vol_vs_prev"]) if pd.notna(row["vol_vs_prev"]) else 0.0,
                    "body_ratio": float(row["body_ratio"]) if pd.notna(row["body_ratio"]) else 0.0,
                    "upper_shadow_pct": float(row["upper_shadow_pct"]) if pd.notna(row["upper_shadow_pct"]) else 0.0,
                    "lower_shadow_pct": float(row["lower_shadow_pct"]) if pd.notna(row["lower_shadow_pct"]) else 0.0,
                    "close_location": float(row["close_location"]) if pd.notna(row["close_location"]) else 0.0,
                    "ma5_slope_5": float(row["ma5_slope_5"]) if pd.notna(row["ma5_slope_5"]) else 0.0,
                    "ma10_slope_5": float(row["ma10_slope_5"]) if pd.notna(row["ma10_slope_5"]) else 0.0,
                    "ma20_slope_5": float(row["ma20_slope_5"]) if pd.notna(row["ma20_slope_5"]) else 0.0,
                    "trend_slope_5": float(row["trend_slope_5"]) if pd.notna(row["trend_slope_5"]) else 0.0,
                    "long_slope_5": float(row["long_slope_5"]) if pd.notna(row["long_slope_5"]) else 0.0,
                    "near_trend_pullback": bool(row["near_trend_pullback"]),
                    "near_long_pullback": bool(row["near_long_pullback"]),
                    "pullback_any": bool(row["pullback_any"]),
                    "cross_up_event": bool(row["cross_up_event"]),
                    "first_pullback_after_cross": bool(row["first_pullback_after_cross"]),
                    "half_volume": bool(row["half_volume"]),
                    "semi_shrink": bool(row["semi_shrink"]),
                    "double_bull_exist_60": bool(row["double_bull_exist_60"]),
                    "key_k_support": bool(row["key_k_support"]),
                    "risk_distribution_any_20": bool(row["risk_distribution_any_20"]),
                    "recent_failed_breakout_20": bool(row["recent_failed_breakout_20"]),
                    "top_distribution_20": bool(row["top_distribution_20"]),
                    "semantic_base": bool(row["semantic_base"]),
                    "semantic_candidate": bool(row["semantic_candidate"]),
                    "semantic_uptrend_pullback": bool(row["semantic_uptrend_pullback"]),
                    "semantic_low_cross_pullback": bool(row["semantic_low_cross_pullback"]),
                    "semantic_confirmed": bool(row["semantic_confirmed"]),
                    "buy_semantic_score": float(row["buy_semantic_score"]) if pd.notna(row["buy_semantic_score"]) else 0.0,
                    "seq_map": seq_map,
                    "rep_map": rep_map,
                }
            )
        if i % 10 == 0 or i == total:
            print(f"模板标签特征构建进度: {i}/{total}")
    return pd.DataFrame(rows)


def choose_best_label_hit(hit: pd.DataFrame, target_date: pd.Timestamp, prefer_trend: bool, prefer_long: bool) -> Tuple[Optional[pd.Series], str]:
    if hit.empty:
        return None, ""
    hit = hit.copy()
    hit["entry_gap"] = (hit["entry_date"] - target_date).abs().dt.days
    hit["signal_gap"] = (hit["signal_date"] - target_date).abs().dt.days
    hit["date_gap"] = hit[["entry_gap", "signal_gap"]].min(axis=1)
    hit["reason_match"] = 0.0
    if prefer_trend and "near_trend_pullback" in hit.columns:
        hit["reason_match"] += hit["near_trend_pullback"].fillna(False).astype(float)
    if prefer_long and "near_long_pullback" in hit.columns:
        hit["reason_match"] += hit["near_long_pullback"].fillna(False).astype(float)
    hit = hit.sort_values(["date_gap", "reason_match", "buy_semantic_score"], ascending=[True, False, False])
    best = hit.iloc[0]
    mode = "entry_date" if pd.Timestamp(best["entry_date"]) == target_date else "signal_date" if pd.Timestamp(best["signal_date"]) == target_date else "nearest_date"
    return best, mode


def map_positive_cases(pos_txt: pd.DataFrame, label_df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for _, r in pos_txt.iterrows():
        code = txt_mod.resolve_code(str(r["stock_name"]), mapping)
        if not code:
            skipped.append({**r.to_dict(), "skip_reason": "股票名未映射"})
            continue
        sub = label_df[label_df["code"].astype(str) == str(code)].copy()
        if sub.empty:
            skipped.append({**r.to_dict(), "skip_reason": "候选池无该股票"})
            continue
        target_date = pd.Timestamp(r["buy_date"])
        hit = sub[(sub["entry_date"] == target_date) | (sub["signal_date"] == target_date)].copy()
        if hit.empty:
            hit = sub[
                sub["entry_date"].between(target_date - pd.Timedelta(days=3), target_date + pd.Timedelta(days=3))
                | sub["signal_date"].between(target_date - pd.Timedelta(days=3), target_date + pd.Timedelta(days=3))
            ].copy()
        best_row, mode = choose_best_label_hit(
            hit,
            target_date,
            bool(r.get("reason_trend_pullback", False)),
            bool(r.get("reason_long_pullback", False)),
        )
        if best_row is None:
            skipped.append({**r.to_dict(), "resolved_code": code, "skip_reason": "买入日期无法映射"})
            continue
        best = best_row.to_dict()
        best.update(r.to_dict())
        best["resolved_code"] = code
        best["mapping_mode"] = mode
        rows.append(best)
    out = pd.DataFrame(rows)
    return out, pd.DataFrame(skipped)


def map_negative_cases(neg_txt: pd.DataFrame, label_df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for _, r in neg_txt.iterrows():
        code = txt_mod.resolve_code(str(r["stock_name"]), mapping)
        if not code:
            skipped.append({**r.to_dict(), "skip_reason": "股票名未映射"})
            continue
        sub = label_df[label_df["code"].astype(str) == str(code)].copy()
        if sub.empty:
            skipped.append({**r.to_dict(), "skip_reason": "候选池无该股票"})
            continue
        target_date = pd.Timestamp(r["b1_date"])
        hit = sub[(sub["entry_date"] == target_date) | (sub["signal_date"] == target_date)].copy()
        if hit.empty:
            hit = sub[
                sub["entry_date"].between(target_date - pd.Timedelta(days=3), target_date + pd.Timedelta(days=3))
                | sub["signal_date"].between(target_date - pd.Timedelta(days=3), target_date + pd.Timedelta(days=3))
            ].copy()
        best_row, mode = choose_best_label_hit(hit, target_date, False, False)
        if best_row is None:
            skipped.append({**r.to_dict(), "resolved_code": code, "skip_reason": "反例日期无法映射"})
            continue
        best = best_row.to_dict()
        best.update(r.to_dict())
        best["resolved_code"] = code
        best["mapping_mode"] = mode
        rows.append(best)
    out = pd.DataFrame(rows)
    return out, pd.DataFrame(skipped)


def assign_target_pools(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["txt_core_trend"] = (x["J"] < 13) & x["near_trend_pullback"].fillna(False) & x["semantic_uptrend_pullback"].fillna(False)
    x["txt_core_long"] = (x["J"] < 13) & x["near_long_pullback"].fillna(False)
    x["txt_core_dual"] = x["txt_core_trend"] | x["txt_core_long"]
    x["txt_confirm_bonus"] = (
        x["key_k_support"].fillna(False).astype(int) * 1.0
        + x["half_volume"].fillna(False).astype(int) * 0.8
        + x["double_bull_exist_60"].fillna(False).astype(int) * 0.7
        + x["semi_shrink"].fillna(False).astype(int) * 0.4
    ).astype(float)
    x["pool_txt_trend"] = x["txt_core_trend"]
    x["pool_txt_long"] = x["txt_core_long"]
    x["pool_txt_dual"] = x["txt_core_dual"]
    x["pool_txt_trend_confirmed"] = x["txt_core_trend"] & (x["txt_confirm_bonus"] >= 0.8)
    x["pool_txt_long_confirmed"] = x["txt_core_long"] & (x["txt_confirm_bonus"] >= 0.8)
    x["pool_txt_confirmed"] = x["txt_core_dual"] & (x["txt_confirm_bonus"] >= 0.8)
    x["pool_txt_strict"] = x["txt_core_dual"] & (x["txt_confirm_bonus"] >= 1.5)
    return x


def hard_negative_feature_cols() -> List[str]:
    return [
        "J",
        "ret3",
        "ret5",
        "ret10",
        "close_to_trend",
        "close_to_long",
        "trend_spread",
        "signal_vs_ma5",
        "vol_vs_prev",
        "body_ratio",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "close_location",
        "ma20_slope_5",
        "trend_slope_5",
        "long_slope_5",
        "near_trend_pullback",
        "near_long_pullback",
        "half_volume",
        "double_bull_exist_60",
        "key_k_support",
        "buy_semantic_score",
    ]


def _prepare_feature_matrix(df: pd.DataFrame, feature_cols: List[str], mean_s: pd.Series, std_s: pd.Series) -> np.ndarray:
    mat = df.reindex(columns=feature_cols).copy()
    for col in feature_cols:
        if mat[col].dtype == bool:
            mat[col] = mat[col].astype(float)
    mat = mat.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (((mat - mean_s) / std_s).to_numpy(dtype=float))


def _nearest_positive_distance(cand_mat: np.ndarray, pos_mat: np.ndarray, batch_size: int = 4000) -> np.ndarray:
    if len(cand_mat) == 0:
        return np.empty(0, dtype=float)
    if len(pos_mat) == 0:
        return np.full(len(cand_mat), np.inf, dtype=float)
    out = np.full(len(cand_mat), np.inf, dtype=float)
    for start in range(0, len(cand_mat), batch_size):
        end = min(len(cand_mat), start + batch_size)
        diff = cand_mat[start:end, None, :] - pos_mat[None, :, :]
        dist = np.sqrt(np.square(diff).sum(axis=2))
        out[start:end] = dist.min(axis=1)
    return out


def build_auto_hard_negatives(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame, template_kind: str) -> pd.DataFrame:
    research_pos = pos_df[pos_df["split"] == "research"].copy()
    if research_pos.empty:
        return pd.DataFrame()
    focus_pool, _ = kind_pool_names(template_kind)

    pool = candidate_df[
        (candidate_df["split"] == "research")
        & candidate_df[focus_pool].fillna(False)
        & candidate_df["has_seq_map"].fillna(False)
    ].copy()
    if pool.empty:
        return pd.DataFrame()

    used_keys = {
        (str(r["code"]), pd.Timestamp(r["signal_date"]))
        for _, r in research_pos.iterrows()
    }
    used_keys |= {
        (str(r["code"]), pd.Timestamp(r["signal_date"]))
        for _, r in neg_df[neg_df["split"] == "research"].iterrows()
    }
    key_mask = pool.apply(lambda r: (str(r["code"]), pd.Timestamp(r["signal_date"])) in used_keys, axis=1)
    pool = pool.loc[~key_mask].copy()
    if pool.empty:
        return pd.DataFrame()

    poor_mask = (
        pool["negative_30d"].fillna(False)
        | (pd.to_numeric(pool["ret_20d"], errors="coerce") <= -0.05)
        | (pd.to_numeric(pool["min_close_ret_30"], errors="coerce") <= -0.10)
    )
    pool = pool.loc[poor_mask].copy()
    if pool.empty:
        return pd.DataFrame()

    feature_cols = [c for c in hard_negative_feature_cols() if c in research_pos.columns and c in pool.columns]
    pos_feat = research_pos.reindex(columns=feature_cols).copy()
    for col in feature_cols:
        if pos_feat[col].dtype == bool:
            pos_feat[col] = pos_feat[col].astype(float)
    pos_feat = pos_feat.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean_s = pos_feat.mean()
    std_s = pos_feat.std().replace(0, 1.0).fillna(1.0)
    pos_mat = ((pos_feat - mean_s) / std_s).to_numpy(dtype=float)
    pool_mat = _prepare_feature_matrix(pool, feature_cols, mean_s, std_s)
    pool["nearest_pos_dist"] = _nearest_positive_distance(pool_mat, pos_mat)

    local_rows: List[pd.DataFrame] = []
    for _, pos_row in research_pos.iterrows():
        code = str(pos_row["code"])
        target = pd.Timestamp(pos_row["signal_date"])
        local = pool[
            (pool["code"].astype(str) == code)
            & pool["signal_date"].between(target - pd.Timedelta(days=7), target + pd.Timedelta(days=7))
        ].copy()
        if local.empty:
            continue
        local["date_gap"] = (local["signal_date"] - target).abs().dt.days
        local = local.sort_values(["date_gap", "nearest_pos_dist", "ret_20d"], ascending=[True, True, True]).head(2)
        local["hard_neg_source"] = "local_same_code"
        local_rows.append(local)

    global_limit = max(180, len(research_pos) * 6)
    global_neg = pool.sort_values(["nearest_pos_dist", "ret_20d", "min_close_ret_30"], ascending=[True, True, True]).head(global_limit).copy()
    global_neg["hard_neg_source"] = "global_nearest_bad"

    hard_neg = pd.concat(local_rows + [global_neg], ignore_index=True) if local_rows else global_neg
    if hard_neg.empty:
        return pd.DataFrame()
    hard_neg = hard_neg.sort_values(["nearest_pos_dist", "ret_20d", "min_close_ret_30"], ascending=[True, True, True])
    hard_neg = hard_neg.drop_duplicates(subset=["code", "signal_date"], keep="first").head(max(160, len(research_pos) * 8)).copy()
    hard_neg["label"] = 0
    return hard_neg.reset_index(drop=True)


def attach_template_rep_maps(candidate_df: pd.DataFrame) -> pd.DataFrame:
    x = candidate_df[candidate_df["has_seq_map"].fillna(False)].copy()
    x["rep_map"] = x["seq_map"].apply(v4_mod.derive_rep_map)
    return x


def stack_rep(rows: Iterable[Dict[str, Any]], rep_name: str) -> np.ndarray:
    arrs = [np.asarray(r["rep_map"][rep_name], dtype=float) for r in rows if "rep_map" in r and rep_name in r["rep_map"]]
    if not arrs:
        return np.empty((0, 0), dtype=float)
    return np.vstack(arrs)


def compute_template_scores(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame, hard_neg_df: pd.DataFrame) -> pd.DataFrame:
    x = candidate_df.copy()
    template_rows = pos_df[pos_df["split"] == "research"].to_dict("records")
    user_neg_rows = neg_df[neg_df["split"] == "research"].to_dict("records")
    auto_neg_rows = hard_neg_df.to_dict("records") if not hard_neg_df.empty else []

    for method, rep in TARGET_REP_METHODS:
        tpl = stack_rep(template_rows, rep)
        seqs = np.vstack([r["rep_map"][rep] for r in x.to_dict("records")])
        score_col = f"tpl_{method}_{rep}"
        x[score_col] = v4_mod.compute_similarity_column(seqs, tpl, method)
        x[f"rank_{score_col}"] = ecdf_score(x.loc[x["split"] == "research", score_col], x[score_col])

        if user_neg_rows:
            neg_tpl = stack_rep(user_neg_rows, rep)
            neg_col = f"neg_user_{method}_{rep}"
            x[neg_col] = v4_mod.compute_similarity_column(seqs, neg_tpl, method)
            x[f"rank_{neg_col}"] = ecdf_score(x.loc[x["split"] == "research", neg_col], x[neg_col])
        if auto_neg_rows:
            neg_tpl = stack_rep(auto_neg_rows, rep)
            neg_col = f"neg_hard_{method}_{rep}"
            x[neg_col] = v4_mod.compute_similarity_column(seqs, neg_tpl, method)
            x[f"rank_{neg_col}"] = ecdf_score(x.loc[x["split"] == "research", neg_col], x[neg_col])

    sim_rank_cols = [c for c in ["rank_tpl_corr_close_vol_concat", "rank_tpl_cosine_close_vol_concat", "rank_tpl_weighted_corr_close_vol_concat", "rank_tpl_lag_corr_close_vol_concat"] if c in x.columns]
    if sim_rank_cols:
        x["template_similarity_score"] = np.mean(np.column_stack([x[c] for c in sim_rank_cols]), axis=1)
    else:
        x["template_similarity_score"] = 0.5

    hard_contrast_cols = []
    for method, rep in TARGET_REP_METHODS:
        pos_rank = f"rank_tpl_{method}_{rep}"
        user_rank = f"rank_neg_user_{method}_{rep}"
        hard_rank = f"rank_neg_hard_{method}_{rep}"
        if pos_rank in x.columns:
            score = x[pos_rank].copy()
            if user_rank in x.columns:
                score = score - 0.35 * x[user_rank]
            if hard_rank in x.columns:
                score = score - 0.55 * x[hard_rank]
            col = f"hard_contrast_{method}_{rep}"
            x[col] = score
            hard_contrast_cols.append(col)
    if hard_contrast_cols:
        x["template_hard_contrast_score"] = np.mean(np.column_stack([x[c] for c in hard_contrast_cols]), axis=1)
    else:
        x["template_hard_contrast_score"] = x["template_similarity_score"]

    x["template_confirm_score"] = (
        0.55 * ecdf_score(x.loc[x["split"] == "research", "txt_confirm_bonus"], x["txt_confirm_bonus"])
        + 0.25 * x["semantic_uptrend_pullback"].fillna(False).astype(float)
        + 0.20 * x["semantic_low_cross_pullback"].fillna(False).astype(float)
    )
    return x


def build_ml_scores(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame, hard_neg_df: pd.DataFrame) -> pd.DataFrame:
    x = candidate_df.copy()
    research_pos = pos_df[pos_df["split"] == "research"].copy()
    research_neg = neg_df[neg_df["split"] == "research"].copy()
    train_rows = []
    if not research_pos.empty:
        tmp = research_pos.copy()
        tmp["label"] = 1
        train_rows.append(tmp)
    if not research_neg.empty:
        tmp = research_neg.copy()
        tmp["label"] = 0
        train_rows.append(tmp)
    if not hard_neg_df.empty:
        train_rows.append(hard_neg_df.copy())
    if not train_rows:
        x["template_hard_ml_score"] = 0.5
        return x
    train_df = pd.concat(train_rows, ignore_index=True)
    if train_df["label"].nunique() < 2:
        x["template_hard_ml_score"] = 0.5
        return x

    feature_cols = [
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
    feature_cols = [c for c in feature_cols if c in x.columns]

    X = train_df.reindex(columns=feature_cols).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = train_df["label"].to_numpy(dtype=int)
    X_all = x.reindex(columns=feature_cols).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    score_cols: List[str] = []

    if HAS_SKLEARN:
        try:
            lr = LogisticRegression(max_iter=500, class_weight="balanced")
            lr.fit(X, y)
            x["template_hard_lr_score"] = lr.predict_proba(X_all)[:, 1]
            score_cols.append("template_hard_lr_score")
        except Exception:
            pass
        try:
            et = ExtraTreesClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            et.fit(X, y)
            x["template_hard_et_score"] = et.predict_proba(X_all)[:, 1]
            score_cols.append("template_hard_et_score")
        except Exception:
            pass
    if HAS_LGB:
        try:
            clf = lgb.LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                min_child_samples=8,
                random_state=42,
            )
            clf.fit(X, y)
            x["template_hard_lgb_score"] = clf.predict_proba(X_all)[:, 1]
            score_cols.append("template_hard_lgb_score")
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
            x["template_hard_xgb_score"] = clf.predict_proba(X_all)[:, 1]
            score_cols.append("template_hard_xgb_score")
        except Exception:
            pass

    if score_cols:
        for col in score_cols:
            x[f"rank_{col}"] = ecdf_score(x.loc[x["split"] == "research", col], x[col])
        x["template_hard_ml_score"] = np.mean(np.column_stack([x[f"rank_{col}"] for col in score_cols]), axis=1)
    else:
        x["template_hard_ml_score"] = 0.5
    return x


def add_fusion_scores(candidate_df: pd.DataFrame) -> pd.DataFrame:
    x = candidate_df.copy()
    x["template_hard_factor_fusion_score"] = (
        0.45 * ecdf_score(x.loc[x["split"] == "research", "discovery_factor_score"], x["discovery_factor_score"])
        + 0.35 * x["template_similarity_score"]
        + 0.20 * x["template_confirm_score"]
    )
    hard_contrast_rank = ecdf_score(x.loc[x["split"] == "research", "template_hard_contrast_score"], x["template_hard_contrast_score"])
    x["template_hard_full_fusion_score"] = (
        0.30 * ecdf_score(x.loc[x["split"] == "research", "discovery_factor_score"], x["discovery_factor_score"])
        + 0.25 * x["template_similarity_score"]
        + 0.20 * hard_contrast_rank
        + 0.15 * x["template_hard_ml_score"]
        + 0.10 * x["template_confirm_score"]
    )
    return x


def signal_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"sample_count": int(len(df))}
    for horizon in [3, 5, 10, 20, 30]:
        out[f"ret_{horizon}d_mean"] = float(pd.to_numeric(df[f"ret_{horizon}d"], errors="coerce").mean()) if not df.empty else np.nan
        out[f"up_{horizon}d_rate"] = float(pd.to_numeric(df[f"up_{horizon}d"], errors="coerce").mean()) if not df.empty else np.nan
    return out


def evaluate_strategies(df: pd.DataFrame, topn_list: Iterable[int], template_kind: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    focus_pool, focus_confirmed_pool = kind_pool_names(template_kind)
    strategy_specs = [
        ("baseline", "champion_discovery", "pool_low_cross", "discovery_factor_score"),
        ("factor", f"champion_on_txt_pool_{template_kind}", focus_pool, "discovery_factor_score"),
        ("similarity", f"template_similarity_{template_kind}", focus_pool, "template_similarity_score"),
        ("contrast", f"template_hard_contrast_{template_kind}", focus_pool, "template_hard_contrast_score"),
        ("ml", f"template_hard_ml_{template_kind}", focus_confirmed_pool, "template_hard_ml_score"),
        ("fusion", f"template_hard_factor_fusion_{template_kind}", focus_confirmed_pool, "template_hard_factor_fusion_score"),
        ("fusion", f"template_hard_full_fusion_{template_kind}", focus_confirmed_pool, "template_hard_full_fusion_score"),
    ]
    for family, variant, pool_name, score_col in strategy_specs:
        if pool_name not in df.columns or score_col not in df.columns:
            continue
        for topn in topn_list:
            for split in ["validation", "final_test"]:
                part = df[(df["split"] == split) & (df[pool_name].fillna(False))].copy()
                if part.empty:
                    continue
                selected = part.sort_values(["signal_date", score_col], ascending=[True, False]).groupby("signal_date").head(topn).copy()
                rows.append(
                    {
                        "split": split,
                        "family": family,
                        "variant": variant,
                        "pool": pool_name,
                        "topn": int(topn),
                        "score_col": score_col,
                        **signal_metrics(selected),
                    }
                )
    return pd.DataFrame(rows)


def choose_validation_family_best(leader_df: pd.DataFrame) -> pd.DataFrame:
    val = leader_df[leader_df["split"] == "validation"].copy()
    if val.empty:
        return pd.DataFrame()
    val = val.sort_values(["ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[False, False, False])
    return val.groupby("family", as_index=False).head(1).reset_index(drop=True)


def build_final_test_report(leader_df: pd.DataFrame, family_best: pd.DataFrame) -> pd.DataFrame:
    final_df = leader_df[leader_df["split"] == "final_test"].copy()
    rows = []
    for _, r in family_best.iterrows():
        hit = final_df[
            (final_df["family"] == r["family"])
            & (final_df["variant"] == r["variant"])
            & (final_df["pool"] == r["pool"])
            & (final_df["topn"] == r["topn"])
        ]
        if not hit.empty:
            rows.append(hit.iloc[0].to_dict())
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def build_selected_rows(df: pd.DataFrame, family_best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    final_df = df[df["split"] == "final_test"].copy()
    for _, r in family_best.iterrows():
        part = final_df[final_df[r["pool"]].fillna(False)].copy()
        if part.empty:
            continue
        selected = part.sort_values(["signal_date", r["score_col"]], ascending=[True, False]).groupby("signal_date").head(int(r["topn"])).copy()
        selected["strategy_tag"] = f"{r['family']}_{r['variant']}_{r['pool']}_top{int(r['topn'])}"
        rows.append(selected[["strategy_tag", "code", "signal_date", "entry_date", "entry_price", "split"]])
    if not rows:
        return pd.DataFrame(columns=["strategy_tag", "code", "signal_date", "entry_date", "entry_price", "split"])
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    topn_list = parse_topn_list(args.topn_list)

    name_code_map = base_mod.load_name_code_map()
    pos_txt, pos_bad = txt_mod.parse_positive_txt(txt_mod.TXT_POS_PATH)
    neg_txt = txt_mod.parse_negative_txt(txt_mod.TXT_NEG_PATH)
    pos_txt = txt_mod.add_reason_flags(pos_txt, "buy_reason")
    neg_txt = txt_mod.add_reason_flags(neg_txt, "no_buy_reason")
    pos_txt = filter_positive_templates(pos_txt, args.template_kind)
    must_keep_codes = txt_mod.extract_label_codes(pos_txt, neg_txt, name_code_map)
    target_dates_by_code = txt_mod.collect_label_target_dates(pos_txt, neg_txt, name_code_map)

    update_progress(
        result_dir,
        "loading_candidates",
        template_kind=args.template_kind,
        file_limit=args.file_limit,
        must_keep_code_count=len(must_keep_codes),
    )
    candidate_df = load_base_candidates(args.base_signal_dir, args.file_limit, must_keep_codes)
    candidate_df = assign_target_pools(candidate_df)
    candidate_df = recover_seq_map(candidate_df, args.base_candidate_pkl)
    candidate_df.drop(columns=["seq_map"], errors="ignore").to_csv(result_dir / "candidate_rows.csv", index=False, encoding="utf-8-sig")
    update_progress(
        result_dir,
        "candidate_ready",
        candidate_count=int(len(candidate_df)),
        seq_covered_count=int(candidate_df["has_seq_map"].fillna(False).sum()),
    )

    split_windows = infer_split_windows(candidate_df)
    label_df = build_label_seq_feature_df(must_keep_codes, split_windows, target_dates_by_code=target_dates_by_code, day_window=5)
    label_df.drop(columns=["seq_map", "rep_map"], errors="ignore").to_csv(result_dir / "label_feature_rows.csv", index=False, encoding="utf-8-sig")

    pos_df, pos_skipped = map_positive_cases(pos_txt, label_df, name_code_map)
    neg_df, neg_skipped = map_negative_cases(neg_txt, label_df, name_code_map)
    pos_df.to_csv(result_dir / "txt_positive_manifest.csv", index=False, encoding="utf-8-sig")
    neg_df.to_csv(result_dir / "txt_negative_manifest.csv", index=False, encoding="utf-8-sig")
    pos_bad.to_csv(result_dir / "txt_positive_bad_rows.csv", index=False, encoding="utf-8-sig")
    pos_skipped.to_csv(result_dir / "txt_positive_skipped.csv", index=False, encoding="utf-8-sig")
    neg_skipped.to_csv(result_dir / "txt_negative_skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(
        result_dir,
        "txt_templates_ready",
        positive_count=int(len(pos_df)),
        negative_count=int(len(neg_df)),
        positive_bad_count=int(len(pos_bad)),
        positive_skipped_count=int(len(pos_skipped)),
        negative_skipped_count=int(len(neg_skipped)),
    )

    candidate_df = attach_template_rep_maps(candidate_df[candidate_df["has_seq_map"].fillna(False)].copy())
    hard_neg_df = build_auto_hard_negatives(candidate_df, pos_df, neg_df, args.template_kind)
    hard_neg_df.to_csv(result_dir / "txt_auto_hard_negative_manifest.csv", index=False, encoding="utf-8-sig")
    candidate_df = compute_template_scores(candidate_df, pos_df, neg_df, hard_neg_df)
    candidate_df = build_ml_scores(candidate_df, pos_df, neg_df, hard_neg_df)
    candidate_df = add_fusion_scores(candidate_df)
    candidate_df.drop(columns=["seq_map", "rep_map"], errors="ignore").to_csv(result_dir / "candidate_enriched.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "scores_ready", candidate_count=int(len(candidate_df)))

    leader_df = evaluate_strategies(candidate_df, topn_list, args.template_kind)
    leader_df.to_csv(result_dir / "signal_layer_leaderboard.csv", index=False, encoding="utf-8-sig")
    family_best = choose_validation_family_best(leader_df)
    family_best.to_csv(result_dir / "validation_family_best.csv", index=False, encoding="utf-8-sig")
    final_report = build_final_test_report(leader_df, family_best)
    final_report.to_csv(result_dir / "final_test_report.csv", index=False, encoding="utf-8-sig")
    selected_rows = build_selected_rows(candidate_df, family_best)
    selected_rows.to_csv(result_dir / "final_test_selected_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "base_signal_dir": str(args.base_signal_dir),
        "base_candidate_pkl": str(args.base_candidate_pkl),
        "result_dir": str(result_dir),
        "template_kind": args.template_kind,
        "file_limit": int(args.file_limit),
        "candidate_count": int(len(candidate_df)),
        "positive_count": int(len(pos_df)),
        "negative_count": int(len(neg_df)),
        "auto_hard_negative_count": int(len(hard_neg_df)),
        "positive_bad_count": int(len(pos_bad)),
        "positive_skipped_count": int(len(pos_skipped)),
        "negative_skipped_count": int(len(neg_skipped)),
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
