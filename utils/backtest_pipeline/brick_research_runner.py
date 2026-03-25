from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 受控环境下 loky 读取物理核数会在 macOS service 上报错，提前固定线程/核数。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from utils.backtest.run_momentum_tail_experiment import (
    INITIAL_CAPITAL as BRICK_INITIAL_CAPITAL,
    build_portfolio_curve,
    compute_equity_metrics,
    max_consecutive_failures,
)
from utils.tmp import similarity_filter_research as sim

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except Exception:
    CATBOOST_OK = False


RESULT_ROOT = Path("/Users/lidongyang/Desktop/Qstrategy/results")
FORMAL_BEST_BASELINE_SUMMARY = Path(
    "/Users/lidongyang/Desktop/Qstrategy/results/brick_pipeline_half_tp_green_compare_v2_full_20260323/summary.json"
)
LEDGER_JSON = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/experiment_ledger.json")
LEDGER_MD = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/experiment_ledger.md")

FACTOR_FEATURES = [
    "trend_spread",
    "close_to_trend",
    "close_to_long",
    "signal_vs_ma5_proxy",
    "ret1",
    "ret5",
    "brick_green_len_prev",
    "brick_red_len",
    "rebound_ratio",
    "RSI14",
    "MACD_hist",
    "body_ratio",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "green4_flag_num",
    "green4_low_flag_num",
    "red4_flag_num",
    "turn_strength_layer_num",
    "trend_layer_num",
]


@dataclass(frozen=True)
class EvalConfig:
    family: str
    candidate_pool: str
    builder: str
    seq_len: int
    rep: str
    scorer: str
    sim_gate: float
    daily_topn: int
    model_name: str = ""
    sim_weight: float = 0.0
    factor_weight: float = 0.0
    ml_weight: float = 0.0
    ranker_name: str = ""

    def name(self) -> str:
        parts = [
            self.family,
            self.candidate_pool,
            self.builder,
            f"len{self.seq_len}",
            self.rep,
            self.scorer,
            f"gate{self.sim_gate:.2f}",
            f"top{self.daily_topn}",
        ]
        if self.model_name:
            parts.append(self.model_name)
        if self.family == "similarity_plus_factor":
            parts.append(f"w{self.sim_weight:.1f}_{self.factor_weight:.1f}")
        if self.family == "full_fusion":
            parts.append(f"w{self.sim_weight:.1f}_{self.factor_weight:.1f}_{self.ml_weight:.1f}")
        return "|".join(parts)

    def base_config(self) -> sim.BaseConfig:
        return sim.BaseConfig(self.builder, self.seq_len, self.rep, self.scorer)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def normalize_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    return series.rank(method="average", pct=True).fillna(0.0)


def records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).copy()
    df = df.rename(columns={"date": "signal_date"})
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["signal_vs_ma5_proxy"] = (1.0 + pd.to_numeric(df["signal_ret"], errors="coerce").fillna(0.0)).clip(lower=0.0)
    df["green4_flag_num"] = df["green4_flag"].astype(int)
    df["green4_low_flag"] = df["green4_flag"] & (~df["trend_ok"])
    df["green4_low_flag_num"] = df["green4_low_flag"].astype(int)
    df["red4_flag_num"] = df["red4_flag"].astype(int)
    df["trend_layer"] = np.where(df["trend_ok"], "high", "low")
    df["trend_layer_num"] = np.where(df["trend_ok"], 1.0, 0.0)
    q1 = pd.to_numeric(df["brick_green_len_prev"], errors="coerce").fillna(0.0).quantile(0.33)
    q2 = pd.to_numeric(df["brick_green_len_prev"], errors="coerce").fillna(0.0).quantile(0.67)
    df["turn_strength_layer"] = np.where(
        df["brick_green_len_prev"] <= q1,
        "low",
        np.where(df["brick_green_len_prev"] >= q2, "high", "mid"),
    )
    df["turn_strength_layer_num"] = np.where(
        df["turn_strength_layer"] == "low",
        0.0,
        np.where(df["turn_strength_layer"] == "mid", 0.5, 1.0),
    )
    return df.sort_values(["signal_date", "code"]).reset_index(drop=True)


def apply_candidate_pool_variant(base_df: pd.DataFrame, pool_name: str) -> pd.DataFrame:
    if base_df.empty:
        return base_df.copy()
    df = base_df.copy()
    df["candidate_pool"] = pool_name
    df["pool_bonus"] = 0.0
    if pool_name == "brick.green4_enhance":
        df["pool_bonus"] = np.where(df["green4_flag"], 0.10, 0.0)
    elif pool_name == "brick.green4_low_enhance":
        df["pool_bonus"] = np.where(df["green4_low_flag"], 0.15, 0.0)
    elif pool_name == "brick.red4_filter":
        df["pool_bonus"] = np.where(df["red4_flag"], -0.10, 0.0)
    elif pool_name == "brick.green4_low_hardfilter":
        df = df[df["green4_low_flag"]].copy()
        df["pool_bonus"] = 0.20
    return df.reset_index(drop=True)


def summarize_single_trade(selected: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
    if selected.empty:
        return {
            "strategy": strategy_name,
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_trade_ret": 0.0,
            "median_trade_ret": 0.0,
        }
    ret_series = pd.to_numeric(selected["ret"], errors="coerce").fillna(0.0)
    return {
        "strategy": strategy_name,
        "trade_count": int(len(selected)),
        "win_rate": float((ret_series > 0).mean()),
        "avg_trade_ret": float(ret_series.mean()),
        "median_trade_ret": float(ret_series.median()),
    }


def summarize_account(selected: pd.DataFrame, strategy_name: str, score_col: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    equity_df, trades_df, summary = sim.run_portfolio_backtest(selected, strategy_name, score_col, return_details=True)
    return equity_df, trades_df, summary


def merge_pool_with_scores(pool_df: pd.DataFrame, scored_df: pd.DataFrame, score_col_name: str = "sim_score") -> pd.DataFrame:
    if pool_df.empty or scored_df.empty:
        return pd.DataFrame()
    merge_cols = ["code", "signal_date"]
    scored = scored_df.rename(columns={"date": "signal_date", "score": score_col_name}).copy()
    keep_cols = merge_cols + [score_col_name]
    merged = pool_df.merge(scored[keep_cols], on=merge_cols, how="inner")
    return merged


def build_pool_scored_frame(raw_df: pd.DataFrame, scored_df: pd.DataFrame, pool_name: str, score_col_name: str = "sim_score") -> pd.DataFrame:
    pool_df = apply_candidate_pool_variant(raw_df, pool_name)
    return merge_pool_with_scores(pool_df, scored_df, score_col_name)


def build_factor_model(train_df: pd.DataFrame) -> dict[str, Any]:
    if train_df.empty:
        return {"features": {}, "feature_order": []}
    work = train_df.copy()
    pos = work[work["label"] == 1]
    neg = work[work["label"] == 0]
    feature_state: dict[str, Any] = {}
    for feature in FACTOR_FEATURES:
        series = pd.to_numeric(work[feature], errors="coerce").fillna(0.0)
        mean = float(series.mean())
        std = float(series.std())
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        pos_mean = float(pd.to_numeric(pos[feature], errors="coerce").fillna(0.0).mean()) if not pos.empty else mean
        neg_mean = float(pd.to_numeric(neg[feature], errors="coerce").fillna(0.0).mean()) if not neg.empty else mean
        direction = 1.0 if pos_mean >= neg_mean else -1.0
        feature_state[feature] = {"mean": mean, "std": std, "direction": direction}
    return {"features": feature_state, "feature_order": list(FACTOR_FEATURES)}


def apply_factor_model(df: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["factor_score_raw"] = pd.Series(dtype=float)
        out["factor_score"] = pd.Series(dtype=float)
        return out
    out = df.copy()
    score = pd.Series(0.0, index=out.index, dtype=float)
    for feature in model["feature_order"]:
        state = model["features"][feature]
        series = pd.to_numeric(out[feature], errors="coerce").fillna(state["mean"])
        z = (series - state["mean"]) / state["std"]
        score = score + z * state["direction"]
    out["factor_score_raw"] = score
    out["factor_score"] = normalize_rank(score)
    return out


def ml_feature_columns() -> List[str]:
    return ["sim_score", "factor_score"] + FACTOR_FEATURES


def fit_ml_model(model_name: str, train_df: pd.DataFrame):
    if train_df.empty or train_df["label"].nunique() < 2:
        return None
    X = train_df[ml_feature_columns()].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train_df["label"].astype(int)
    if model_name == "lightgbm":
        if not LGB_OK:
            return None
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=42,
            verbose=-1,
        )
    elif model_name == "xgboost":
        if not XGB_OK:
            return None
        model = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    elif model_name == "random_forest":
        if not SKLEARN_OK:
            return None
        model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=1)
    elif model_name == "extra_trees":
        if not SKLEARN_OK:
            return None
        model = ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=1)
    elif model_name == "logistic_regression":
        if not SKLEARN_OK:
            return None
        model = LogisticRegression(max_iter=500, random_state=42)
    elif model_name == "gaussian_nb":
        if not SKLEARN_OK:
            return None
        model = GaussianNB()
    elif model_name == "catboost":
        if not CATBOOST_OK:
            return "skipped_blocked"
        model = CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
        )
    else:
        raise ValueError(f"未知模型: {model_name}")
    model.fit(X, y)
    return model


def apply_ml_model(stage_df: pd.DataFrame, model_name: str, model: Any) -> pd.DataFrame:
    out = stage_df.copy()
    if isinstance(model, str) and model == "skipped_blocked":
        out["ml_score"] = np.nan
        out["ml_status"] = "skipped_blocked"
        return out
    if model is None:
        out["ml_score"] = np.nan
        out["ml_status"] = "skipped_unavailable"
        return out
    X = out[ml_feature_columns()].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if hasattr(model, "predict_proba"):
        out["ml_score"] = model.predict_proba(X)[:, 1]
    else:
        pred = model.predict(X)
        out["ml_score"] = pd.Series(pred, index=out.index, dtype=float)
    out["ml_score"] = normalize_rank(out["ml_score"])
    out["ml_status"] = "ok"
    return out


def prepare_ranked_stage(train_scored: pd.DataFrame, stage_scored: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    if stage_scored.empty:
        return stage_scored.copy()
    if cfg.family == "similarity_only":
        out = stage_scored.copy()
        out["rank_score"] = normalize_rank(out["sim_score"]) + pd.to_numeric(out["pool_bonus"], errors="coerce").fillna(0.0)
        return out
    factor_model = build_factor_model(train_scored)
    train_with_factor = apply_factor_model(train_scored, factor_model)
    stage_with_factor = apply_factor_model(stage_scored, factor_model)
    if cfg.family == "similarity_plus_factor":
        out = stage_with_factor.copy()
        out["rank_score"] = (
            normalize_rank(out["sim_score"]) * cfg.sim_weight
            + pd.to_numeric(out["factor_score"], errors="coerce").fillna(0.0) * cfg.factor_weight
            + pd.to_numeric(out["pool_bonus"], errors="coerce").fillna(0.0)
        )
        return out
    model = fit_ml_model(cfg.model_name, train_with_factor)
    stage_with_ml = apply_ml_model(stage_with_factor, cfg.model_name, model)
    if cfg.family == "similarity_plus_ml":
        out = stage_with_ml.copy()
        out["rank_score"] = normalize_rank(out["ml_score"]) + pd.to_numeric(out["pool_bonus"], errors="coerce").fillna(0.0)
        return out
    out = stage_with_ml.copy()
    out["rank_score"] = (
        normalize_rank(out["sim_score"]) * cfg.sim_weight
        + pd.to_numeric(out["factor_score"], errors="coerce").fillna(0.0) * cfg.factor_weight
        + pd.to_numeric(out["ml_score"], errors="coerce").fillna(0.0) * cfg.ml_weight
        + pd.to_numeric(out["pool_bonus"], errors="coerce").fillna(0.0)
    )
    return out


def cfg_from_row(row: dict[str, Any]) -> EvalConfig:
    def _f(key: str, default: Any = "") -> Any:
        value = row.get(key, default)
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return value

    return EvalConfig(
        family=str(_f("family")),
        candidate_pool=str(_f("candidate_pool")),
        builder=str(_f("builder")),
        seq_len=int(_f("seq_len", 21)),
        rep=str(_f("rep")),
        scorer=str(_f("scorer")),
        sim_gate=float(_f("sim_gate", 0.8)),
        daily_topn=int(_f("daily_topn", 10)),
        model_name=str(_f("model_name")),
        sim_weight=float(_f("sim_weight", 0.0) or 0.0),
        factor_weight=float(_f("factor_weight", 0.0) or 0.0),
        ml_weight=float(_f("ml_weight", 0.0) or 0.0),
        ranker_name=str(_f("ranker_name")),
    )


def select_with_gate(df: pd.DataFrame, rank_col: str, sim_gate: float, daily_topn: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    x = df[df["sim_score"] >= sim_gate].copy()
    if x.empty:
        return x
    x = x.sort_values(["signal_date", rank_col, "code"], ascending=[True, False, True], kind="mergesort")
    x = x.groupby("signal_date", group_keys=False).head(daily_topn).reset_index(drop=True)
    return x


def build_similarity_only_configs(smoke: bool) -> List[EvalConfig]:
    builders = ["sample_300", "recent_100"] if smoke else ["sample_300", "recent_100", "cluster_100"]
    seq_lens = [21] if smoke else [21, 30]
    reps = ["close_norm"] if smoke else ["close_norm", "close_vol_concat"]
    scorers = ["pipeline_corr_dtw", "cosine"]
    gates = [0.80] if smoke else [0.75, 0.80, 0.85]
    topns = [10] if smoke else [8, 10, 12]
    pools = [
        "brick.relaxed_base",
        "brick.green4_enhance",
        "brick.green4_low_enhance",
        "brick.red4_filter",
        "brick.green4_low_hardfilter",
    ]
    rows: List[EvalConfig] = []
    for pool in pools:
        for builder in builders:
            for seq_len in seq_lens:
                for rep in reps:
                    for scorer in scorers:
                        for gate in gates:
                            for topn in topns:
                                rows.append(
                                    EvalConfig(
                                        family="similarity_only",
                                        candidate_pool=pool,
                                        builder=builder,
                                        seq_len=seq_len,
                                        rep=rep,
                                        scorer=scorer,
                                        sim_gate=gate,
                                        daily_topn=topn,
                                        ranker_name="ranker.brick_similarity_champion",
                                    )
                                )
    return rows


def build_similarity_plus_factor_configs(smoke: bool) -> List[EvalConfig]:
    weights = [(0.8, 0.2)] if smoke else [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]
    gates = [0.80] if smoke else [0.75, 0.80, 0.85]
    topns = [10] if smoke else [8, 10, 12]
    pools = [
        "brick.relaxed_base",
        "brick.green4_enhance",
        "brick.green4_low_enhance",
        "brick.red4_filter",
        "brick.green4_low_hardfilter",
    ]
    rows: List[EvalConfig] = []
    for pool in pools:
        for sim_weight, factor_weight in weights:
            for gate in gates:
                for topn in topns:
                    rows.append(
                        EvalConfig(
                            family="similarity_plus_factor",
                            candidate_pool=pool,
                            builder="sample_300",
                            seq_len=21,
                            rep="close_norm",
                            scorer="pipeline_corr_dtw",
                            sim_gate=gate,
                            daily_topn=topn,
                            sim_weight=sim_weight,
                            factor_weight=factor_weight,
                            ranker_name="ranker.brick_similarity_plus_factor",
                        )
                    )
    return rows


def build_similarity_plus_ml_configs(smoke: bool) -> List[EvalConfig]:
    models = ["lightgbm", "xgboost"] if smoke else ["lightgbm", "xgboost", "random_forest", "extra_trees", "logistic_regression", "gaussian_nb", "catboost"]
    gates = [0.80] if smoke else [0.75, 0.80, 0.85]
    topns = [10] if smoke else [8, 10, 12]
    pools = [
        "brick.relaxed_base",
        "brick.green4_enhance",
        "brick.green4_low_enhance",
        "brick.red4_filter",
        "brick.green4_low_hardfilter",
    ]
    rows: List[EvalConfig] = []
    for pool in pools:
        for model in models:
            for gate in gates:
                for topn in topns:
                    rows.append(
                        EvalConfig(
                            family="similarity_plus_ml",
                            candidate_pool=pool,
                            builder="sample_300",
                            seq_len=21,
                            rep="close_norm",
                            scorer="pipeline_corr_dtw",
                            sim_gate=gate,
                            daily_topn=topn,
                            model_name=model,
                            ranker_name="ranker.brick_similarity_plus_ml",
                        )
                    )
    return rows


def build_full_fusion_configs(smoke: bool) -> List[EvalConfig]:
    models = ["lightgbm", "xgboost"] if smoke else ["lightgbm", "xgboost", "random_forest", "extra_trees", "logistic_regression", "gaussian_nb", "catboost"]
    weights = [(0.5, 0.2, 0.3)] if smoke else [(0.5, 0.2, 0.3), (0.4, 0.3, 0.3), (0.4, 0.2, 0.4)]
    gates = [0.80] if smoke else [0.75, 0.80, 0.85]
    topns = [10] if smoke else [8, 10, 12]
    pools = [
        "brick.relaxed_base",
        "brick.green4_enhance",
        "brick.green4_low_enhance",
        "brick.red4_filter",
        "brick.green4_low_hardfilter",
    ]
    rows: List[EvalConfig] = []
    for pool in pools:
        for model in models:
            for sim_weight, factor_weight, ml_weight in weights:
                for gate in gates:
                    for topn in topns:
                        rows.append(
                            EvalConfig(
                                family="full_fusion",
                                candidate_pool=pool,
                                builder="sample_300",
                                seq_len=21,
                                rep="close_norm",
                                scorer="pipeline_corr_dtw",
                                sim_gate=gate,
                                daily_topn=topn,
                                model_name=model,
                                sim_weight=sim_weight,
                                factor_weight=factor_weight,
                                ml_weight=ml_weight,
                                ranker_name="ranker.brick_full_fusion",
                            )
                        )
    return rows


def fixed_tp_trade(df: pd.DataFrame, signal_idx: int, tp_pct: float, max_holding_days: int = 3) -> Optional[dict[str, Any]]:
    n = len(df)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None
    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    signal_low = float(df.at[signal_idx, "low"])
    stop_price = signal_low * 0.99
    target_price = entry_price * (1.0 + tp_pct)
    exit_idx = min(entry_idx + max_holding_days + 1, n - 1)
    exit_price = float(df.at[exit_idx, "open"]) if exit_idx > entry_idx else float(df.at[n - 1, "close"])
    exit_reason = "time_exit_next_open"
    for j in range(entry_idx + 1, min(entry_idx + max_holding_days, n - 1) + 1):
        high_j = float(df.at[j, "high"])
        low_j = float(df.at[j, "low"])
        if np.isfinite(high_j) and high_j >= target_price:
            next_idx = min(j + 1, n - 1)
            px = float(df.at[next_idx, "open"])
            if np.isfinite(px) and px > 0:
                exit_idx = next_idx
                exit_price = px
                exit_reason = "take_profit_next_open"
                break
        if np.isfinite(low_j) and low_j <= stop_price:
            next_idx = min(j + 1, n - 1)
            px = float(df.at[next_idx, "open"])
            if np.isfinite(px) and px > 0:
                exit_idx = next_idx
                exit_price = px
                exit_reason = "stop_loss_next_open"
                break
    ret = exit_price / entry_price - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "ret": ret,
        "holding_days": int(exit_idx - entry_idx),
        "success": ret > 0,
        "exit_reason": exit_reason,
        "signal_low": signal_low,
    }


def partial_tp_trade(
    df: pd.DataFrame,
    signal_idx: int,
    first_tp_pct: float,
    remainder_mode: str,
    second_tp_pct: float | None = None,
    max_holding_days: int = 3,
) -> Optional[dict[str, Any]]:
    n = len(df)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None
    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    signal_low = float(df.at[signal_idx, "low"])
    stop_price = signal_low * 0.99
    first_tp = entry_price * (1.0 + first_tp_pct)
    scheduled_idx = min(entry_idx + max_holding_days + 1, n - 1)
    half_exit_idx = None
    half_exit_price = None
    final_exit_idx = scheduled_idx
    final_exit_price = float(df.at[scheduled_idx, "open"]) if scheduled_idx > entry_idx else float(df.at[n - 1, "close"])
    exit_reason = "time_exit_next_open"

    for j in range(entry_idx + 1, min(entry_idx + max_holding_days, n - 1) + 1):
        high_j = float(df.at[j, "high"])
        low_j = float(df.at[j, "low"])
        if np.isfinite(high_j) and high_j >= first_tp:
            half_exit_idx = min(j + 1, n - 1)
            half_exit_price = float(df.at[half_exit_idx, "open"])
            if not (np.isfinite(half_exit_price) and half_exit_price > 0):
                return None
            if remainder_mode == "day3_exit":
                final_exit_idx = scheduled_idx
                final_exit_price = float(df.at[final_exit_idx, "open"]) if final_exit_idx > entry_idx else float(df.at[n - 1, "close"])
                exit_reason = "partial_then_day3_exit"
                break
            for k in range(j, min(entry_idx + max_holding_days, n - 1) + 1):
                high_k = float(df.at[k, "high"])
                low_k = float(df.at[k, "low"])
                green_k = bool(df.at[k, "brick_green"])
                if np.isfinite(low_k) and low_k <= stop_price:
                    out_idx = min(k + 1, n - 1)
                    out_open = float(df.at[out_idx, "open"])
                    if np.isfinite(out_open) and out_open > 0:
                        final_exit_idx = out_idx
                        final_exit_price = out_open
                        exit_reason = "partial_then_stop_next_open"
                        break
                if remainder_mode == "green_next_open" and green_k:
                    out_idx = min(k + 1, n - 1)
                    out_open = float(df.at[out_idx, "open"])
                    if np.isfinite(out_open) and out_open > 0:
                        final_exit_idx = out_idx
                        final_exit_price = out_open
                        exit_reason = "partial_then_green_next_open"
                        break
                if remainder_mode == "green_next_open_profit_only" and green_k:
                    out_idx = min(k + 1, n - 1)
                    out_open = float(df.at[out_idx, "open"])
                    if np.isfinite(out_open) and out_open > 0 and out_open > entry_price:
                        final_exit_idx = out_idx
                        final_exit_price = out_open
                        exit_reason = "partial_then_green_next_open_profit_only"
                        break
                if remainder_mode == "second_tp" and second_tp_pct is not None:
                    second_tp = entry_price * (1.0 + second_tp_pct)
                    if np.isfinite(high_k) and high_k >= second_tp:
                        out_idx = min(k + 1, n - 1)
                        out_open = float(df.at[out_idx, "open"])
                        if np.isfinite(out_open) and out_open > 0:
                            final_exit_idx = out_idx
                            final_exit_price = out_open
                            exit_reason = "partial_then_second_tp"
                            break
            break
        if np.isfinite(low_j) and low_j <= stop_price:
            out_idx = min(j + 1, n - 1)
            out_open = float(df.at[out_idx, "open"])
            if np.isfinite(out_open) and out_open > 0:
                final_exit_idx = out_idx
                final_exit_price = out_open
                exit_reason = "stop_loss_next_open"
                break

    if half_exit_idx is None or half_exit_price is None:
        ret = final_exit_price / entry_price - 1.0
        half_exit_date = None
    else:
        ret = 0.5 * (half_exit_price / entry_price - 1.0) + 0.5 * (final_exit_price / entry_price - 1.0)
        half_exit_date = df.at[half_exit_idx, "date"]
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "half_exit_date": half_exit_date,
        "exit_date": df.at[final_exit_idx, "date"],
        "entry_price": entry_price,
        "half_exit_price": half_exit_price,
        "exit_price": final_exit_price,
        "ret": ret,
        "holding_days": int(final_exit_idx - entry_idx),
        "success": ret > 0,
        "exit_reason": exit_reason,
        "signal_low": signal_low,
    }


def summarize_signal_basket(trade_df: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
    if trade_df.empty:
        return {
            "strategy": strategy_name,
            "trade_count": 0,
            "avg_trade_return": np.nan,
            "success_rate": np.nan,
            "avg_holding_days": np.nan,
            "profit_factor": np.nan,
            "max_consecutive_failures": np.nan,
            "annual_return_signal_basket": np.nan,
            "max_drawdown_signal_basket": np.nan,
            "final_equity_signal_basket": np.nan,
            "equity_days_signal_basket": 0,
        }
    portfolio_df = build_portfolio_curve(trade_df)
    metrics = compute_equity_metrics(portfolio_df)
    ret_series = pd.to_numeric(trade_df["ret"], errors="coerce").fillna(0.0)
    neg = float(ret_series[ret_series < 0].sum())
    pf = np.nan if abs(neg) < 1e-12 else float(ret_series[ret_series > 0].sum()) / abs(neg)
    return {
        "strategy": strategy_name,
        "trade_count": int(len(trade_df)),
        "avg_trade_return": float(ret_series.mean()),
        "success_rate": float((ret_series > 0).mean()),
        "avg_holding_days": float(pd.to_numeric(trade_df["holding_days"], errors="coerce").fillna(0.0).mean()),
        "profit_factor": pf,
        "max_consecutive_failures": int(max_consecutive_failures((ret_series > 0).tolist())),
        "annual_return_signal_basket": float(metrics["annual_return"]),
        "max_drawdown_signal_basket": float(metrics["max_drawdown"]),
        "final_equity_signal_basket": float(metrics["final_equity"]),
        "equity_days_signal_basket": int(metrics["equity_days"]),
    }


def run_exit_grid(
    feature_map: Dict[str, pd.DataFrame],
    selected_df: pd.DataFrame,
    strategy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    details = []
    fixed_grid = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    partial_specs = []
    for first_tp in [0.03, 0.035, 0.04]:
        partial_specs.append((first_tp, "day3_exit", None))
        partial_specs.append((first_tp, "green_next_open", None))
        partial_specs.append((first_tp, "green_next_open_profit_only", None))
        for second_tp in [0.05, 0.06]:
            partial_specs.append((first_tp, "second_tp", second_tp))

    def _run_trades(exit_name: str, tp_value: float | None = None, first_tp: float | None = None, remainder_mode: str = "", second_tp: float | None = None) -> pd.DataFrame:
        out_rows = []
        for row in selected_df.itertuples(index=False):
            stock_df = feature_map.get(str(row.code))
            if stock_df is None:
                continue
            if exit_name == "fixed":
                trade = fixed_tp_trade(stock_df, int(row.signal_idx), float(tp_value))
            else:
                trade = partial_tp_trade(stock_df, int(row.signal_idx), float(first_tp), remainder_mode, second_tp)
            if trade is None:
                continue
            trade["code"] = row.code
            trade["sort_score"] = float(getattr(row, "rank_score", 0.0))
            out_rows.append(trade)
        if not out_rows:
            return pd.DataFrame()
        return pd.DataFrame(out_rows).sort_values(["signal_date", "code"]).reset_index(drop=True)

    for tp in fixed_grid:
        trade_df = _run_trades("fixed", tp_value=tp)
        strategy = f"{strategy_name}|fixed_tp_{tp:.3f}"
        summary = summarize_signal_basket(trade_df, strategy)
        summary["exit_family"] = "fixed_tp"
        summary["take_profit_pct"] = tp
        rows.append(summary)
        details.append({"strategy": strategy, "trade_df": trade_df})

    for first_tp, remainder_mode, second_tp in partial_specs:
        trade_df = _run_trades("partial", first_tp=first_tp, remainder_mode=remainder_mode, second_tp=second_tp)
        strategy = f"{strategy_name}|partial_{first_tp:.3f}|{remainder_mode}"
        if second_tp is not None:
            strategy += f"|{second_tp:.3f}"
        summary = summarize_signal_basket(trade_df, strategy)
        summary["exit_family"] = "partial_tp"
        summary["first_take_profit_pct"] = first_tp
        summary["remainder_mode"] = remainder_mode
        summary["second_take_profit_pct"] = second_tp
        rows.append(summary)
        details.append({"strategy": strategy, "trade_df": trade_df})

    return pd.DataFrame(rows).sort_values(
        ["final_equity_signal_basket", "annual_return_signal_basket", "max_drawdown_signal_basket"],
        ascending=[False, False, False],
    ).reset_index(drop=True), pd.DataFrame(details)


def rolling_windows(all_dates: List[pd.Timestamp], train_months: int = 24, val_months: int = 6, test_months: int = 6, step_months: int = 3) -> List[dict[str, pd.Timestamp]]:
    if not all_dates:
        return []
    start = pd.Timestamp(min(all_dates)).normalize()
    end = pd.Timestamp(max(all_dates)).normalize()
    windows = []
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        val_end = train_end + pd.DateOffset(months=val_months)
        test_end = val_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        windows.append(
            {
                "train_start": cursor,
                "train_end": train_end,
                "val_start": train_end + pd.Timedelta(days=1),
                "val_end": val_end,
                "test_start": val_end + pd.Timedelta(days=1),
                "test_end": test_end,
            }
        )
        cursor = cursor + pd.DateOffset(months=step_months)
    return windows


def evaluate_fixed_config_on_stage(stage_df: pd.DataFrame, cfg: EvalConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = select_with_gate(stage_df, "rank_score", cfg.sim_gate, cfg.daily_topn)
    _, _, account_summary = summarize_account(selected, cfg.name(), "rank_score")
    single_trade_summary = summarize_single_trade(selected, cfg.name())
    summary = {**single_trade_summary, **account_summary, "family": cfg.family, "candidate_pool": cfg.candidate_pool}
    return selected, summary


def run_fixed_config_rolling_validation(records_df: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame()
    dates = sorted(pd.to_datetime(records_df["signal_date"]).unique())
    windows = rolling_windows(dates)
    rows = []
    for idx, window in enumerate(windows, start=1):
        test_df = records_df[
            (records_df["signal_date"] >= window["test_start"])
            & (records_df["signal_date"] <= window["test_end"])
        ].copy()
        if test_df.empty:
            continue
        selected = select_with_gate(test_df, "rank_score", cfg.sim_gate, cfg.daily_topn)
        _, _, account_summary = summarize_account(selected, f"{cfg.name()}|rolling_{idx}", "rank_score")
        rows.append(
            {
                "window_id": idx,
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "trade_count": int(account_summary["trades"]),
                "win_rate": float(account_summary["win_rate"]),
                "avg_trade_ret": float(account_summary["avg_trade_ret"]),
                "total_return": float(account_summary["total_return"]),
                "max_drawdown": float(account_summary["max_drawdown"]),
                "ending_equity": float(account_summary["ending_equity"]),
            }
        )
    return pd.DataFrame(rows)


def load_formal_best_baseline() -> dict[str, Any]:
    payload = json.loads(FORMAL_BEST_BASELINE_SUMMARY.read_text(encoding="utf-8"))
    baseline = payload["baseline"]
    return {
        "strategy": baseline["strategy"],
        "trade_count": baseline["trade_count"],
        "avg_trade_return": baseline["avg_trade_return"],
        "success_rate": baseline["success_rate"],
        "annual_return_signal_basket": baseline["annual_return_signal_basket"],
        "max_drawdown_signal_basket": baseline["max_drawdown_signal_basket"],
        "final_equity_signal_basket": baseline["final_equity_signal_basket"],
    }


def update_ledger(result_dir: Path, overall_summary: dict[str, Any]) -> None:
    payload = json.loads(LEDGER_JSON.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", [])
    experiments.append(
        {
            "id": f"brick-comprehensive-lab-{result_dir.name}",
            "family": "brick",
            "title": "BRICK 相似度/因子/机器学习/全融合综合实验",
            "status": "completed",
            "conclusion": overall_summary["conclusion"],
            "result_dir": str(result_dir),
        }
    )
    payload["experiments"] = experiments
    write_json(LEDGER_JSON, payload)
    lines = ["# Experiment Ledger", ""]
    for item in payload["experiments"]:
        lines.append(f"- `{item['family']}` | `{item['id']}` | {item['title']} | {item['status']}")
        if item.get("conclusion"):
            lines.append(f"  结论：{item['conclusion']}")
        if item.get("result_dir"):
            lines.append(f"  目录：{item['result_dir']}")
    LEDGER_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK 综合实验 runner")
    parser.add_argument("config", help="json 配置文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    smoke = bool(config.get("smoke", False))
    file_limit = int(config.get("file_limit", 0) or 0)
    data_dir = Path(config.get("data_dir", "/Users/lidongyang/Desktop/Qstrategy/data"))
    output_dir = Path(config.get("output_dir") or RESULT_ROOT / f"brick_comprehensive_lab_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(output_dir, "starting", smoke=smoke, file_limit=file_limit)

    all_records = sim.load_full_signal_dataset(file_limit=file_limit, max_workers=int(config.get("max_workers", 1)), data_dir=data_dir)
    if not all_records:
        raise RuntimeError("BRICK relaxed 数据集为空")
    feature_map = sim.load_relaxed_feature_map(data_dir=data_dir, file_limit=file_limit)
    all_df = records_to_df(all_records)
    research_records, validation_records, final_test_records, split_meta = sim.split_three_way(all_records)
    research_df = records_to_df(research_records)
    validation_df = records_to_df(validation_records)
    final_test_df = records_to_df(final_test_records)
    update_progress(output_dir, "dataset_ready", total_records=len(all_df), research=len(research_df), validation=len(validation_df), final_test=len(final_test_df))

    formal_best_baseline = load_formal_best_baseline()
    relaxed_baseline_df = final_test_df.copy()
    relaxed_baseline_df["baseline_score"] = 0.0
    _, _, relaxed_baseline_summary = summarize_account(relaxed_baseline_df, "baseline_relaxed_prefilter", "baseline_score")

    similarity_configs = build_similarity_only_configs(smoke)
    base_configs = list({cfg.builder + f"|{cfg.seq_len}|{cfg.rep}|{cfg.scorer}": sim.BaseConfig(cfg.builder, cfg.seq_len, cfg.rep, cfg.scorer) for cfg in similarity_configs}.values())
    validation_base_cache = sim.build_base_scored_cache(research_records, validation_records, base_configs, "brick comprehensive validation", score_workers=int(config.get("score_workers", 1)))
    validation_pool_map = {name: apply_candidate_pool_variant(validation_df, name) for name in sorted({cfg.candidate_pool for cfg in similarity_configs})}

    family_validation_rows: List[dict[str, Any]] = []
    family_final_rows: List[dict[str, Any]] = []
    family_best_final: dict[str, dict[str, Any]] = {}
    final_candidate_scored: Optional[pd.DataFrame] = None
    comparison_rows: List[dict[str, Any]] = []

    def _append_comparison(summary_row: dict[str, Any], baseline_name: str, baseline_summary: dict[str, Any]) -> None:
        comparison_rows.append(
            {
                "strategy": summary_row["strategy"],
                "baseline": baseline_name,
                "total_return_diff": float(summary_row["total_return"] - baseline_summary.get("total_return", 0.0)),
                "max_drawdown_diff": float(summary_row["max_drawdown"] - baseline_summary.get("max_drawdown", 0.0)),
                "win_rate_diff": float(summary_row["win_rate"] - baseline_summary.get("win_rate", 0.0)),
                "ending_equity_diff": float(summary_row["ending_equity"] - baseline_summary.get("ending_equity", BRICK_INITIAL_CAPITAL)),
            }
        )

    similarity_validation_results: List[tuple[EvalConfig, dict[str, Any]]] = []
    for cfg in similarity_configs:
        scored_df = validation_base_cache[cfg.base_config().key()]
        pool_df = validation_pool_map[cfg.candidate_pool]
        stage_df = merge_pool_with_scores(pool_df, scored_df, "sim_score")
        if stage_df.empty:
            continue
        stage_df["rank_score"] = normalize_rank(stage_df["sim_score"]) + pd.to_numeric(stage_df["pool_bonus"], errors="coerce").fillna(0.0)
        selected, summary = evaluate_fixed_config_on_stage(stage_df, cfg)
        family_validation_rows.append(summary | asdict(cfg))
        similarity_validation_results.append((cfg, summary))

    similarity_validation_df = pd.DataFrame(family_validation_rows)
    if similarity_validation_df.empty:
        raise RuntimeError("similarity_only validation 为空")
    similarity_validation_df = similarity_validation_df.sort_values(["total_return", "max_drawdown", "win_rate"], ascending=[False, False, False]).reset_index(drop=True)

    top_similarity_cfgs = [cfg for cfg, _ in sorted(similarity_validation_results, key=lambda x: (x[1]["total_return"], x[1]["win_rate"]), reverse=True)[: max(5, 1)]]
    final_similarity_cache = sim.build_base_scored_cache(research_records + validation_records, final_test_records, [cfg.base_config() for cfg in top_similarity_cfgs], "brick comprehensive final similarity", score_workers=int(config.get("score_workers", 1)))
    final_pool_map = {name: apply_candidate_pool_variant(final_test_df, name) for name in validation_pool_map}
    similarity_final_best = None
    for cfg in top_similarity_cfgs:
        scored_df = final_similarity_cache[cfg.base_config().key()]
        pool_df = final_pool_map[cfg.candidate_pool]
        stage_df = merge_pool_with_scores(pool_df, scored_df, "sim_score")
        if stage_df.empty:
            continue
        stage_df["rank_score"] = normalize_rank(stage_df["sim_score"]) + pd.to_numeric(stage_df["pool_bonus"], errors="coerce").fillna(0.0)
        selected, summary = evaluate_fixed_config_on_stage(stage_df, cfg)
        row = summary | asdict(cfg)
        family_final_rows.append(row)
        if similarity_final_best is None or row["total_return"] > similarity_final_best["summary"]["total_return"]:
            similarity_final_best = {"cfg": cfg, "stage_df": stage_df, "selected": selected, "summary": row}
            final_candidate_scored = stage_df.copy()

    if similarity_final_best is None:
        raise RuntimeError("similarity_only final_test 为空")
    family_best_final["similarity_only"] = similarity_final_best["summary"]
    relaxed_similarity_champion_summary = similarity_final_best["summary"].copy()

    champion_base = sim.BaseConfig("sample_300", 21, "close_norm", "pipeline_corr_dtw")
    champion_key = champion_base.key()
    champion_research_self = sim.build_base_scored_cache(
        research_records,
        research_records,
        [champion_base],
        "brick comprehensive champion research self",
        score_workers=1,
    )[champion_key]
    champion_validation = sim.build_base_scored_cache(
        research_records,
        validation_records,
        [champion_base],
        "brick comprehensive champion validation",
        score_workers=int(config.get("score_workers", 1)),
    )[champion_key]
    champion_train_final_self = sim.build_base_scored_cache(
        research_records + validation_records,
        research_records + validation_records,
        [champion_base],
        "brick comprehensive champion train final self",
        score_workers=1,
    )[champion_key]
    champion_final = sim.build_base_scored_cache(
        research_records + validation_records,
        final_test_records,
        [champion_base],
        "brick comprehensive final champion base",
        score_workers=int(config.get("score_workers", 1)),
    )[champion_key]
    champion_all_self = sim.build_base_scored_cache(
        all_records,
        all_records,
        [champion_base],
        "brick comprehensive champion all self",
        score_workers=1,
    )[champion_key]

    def _evaluate_family_configs(configs: List[EvalConfig], family_name: str) -> tuple[pd.DataFrame, List[tuple[EvalConfig, pd.DataFrame, dict[str, Any]]]]:
        validation_rows: List[dict[str, Any]] = []
        finalists: List[tuple[EvalConfig, pd.DataFrame, dict[str, Any]]] = []
        top_cfgs: List[EvalConfig] = []
        if not configs:
            return pd.DataFrame(), finalists
        for cfg in configs:
            train_scored = build_pool_scored_frame(research_df, champion_research_self, cfg.candidate_pool, "sim_score")
            validation_scored = build_pool_scored_frame(validation_df, champion_validation, cfg.candidate_pool, "sim_score")
            if train_scored.empty or validation_scored.empty:
                continue
            validation_stage = prepare_ranked_stage(train_scored, validation_scored, cfg)
            if "ml_status" in validation_stage.columns and validation_stage["ml_status"].eq("skipped_blocked").all():
                continue
            _, validation_summary = evaluate_fixed_config_on_stage(validation_stage, cfg)
            validation_rows.append(validation_summary | asdict(cfg) | {"ml_status": validation_stage.get("ml_status", pd.Series(["ok"])).iloc[0] if not validation_stage.empty and "ml_status" in validation_stage.columns else "ok"})
        validation_df_out = pd.DataFrame(validation_rows)
        if validation_df_out.empty:
            return validation_df_out, finalists
        validation_df_out = validation_df_out.sort_values(["total_return", "max_drawdown", "win_rate"], ascending=[False, False, False]).reset_index(drop=True)
        top_cfgs = [cfg_from_row(row) for row in validation_df_out.head(5).to_dict(orient="records")]
        for cfg in top_cfgs:
            train_scored = build_pool_scored_frame(records_to_df(research_records + validation_records), champion_train_final_self, cfg.candidate_pool, "sim_score")
            final_scored = build_pool_scored_frame(final_test_df, champion_final, cfg.candidate_pool, "sim_score")
            if train_scored.empty or final_scored.empty:
                continue
            final_stage = prepare_ranked_stage(train_scored, final_scored, cfg)
            if "ml_status" in final_stage.columns and final_stage["ml_status"].eq("skipped_blocked").all():
                continue
            selected, final_summary = evaluate_fixed_config_on_stage(final_stage, cfg)
            row = final_summary | asdict(cfg) | {"ml_status": final_stage.get("ml_status", pd.Series(["ok"])).iloc[0] if not final_stage.empty and "ml_status" in final_stage.columns else "ok"}
            finalists.append((cfg, final_stage, row | {"strategy": cfg.name(), "family": family_name}))
        return validation_df_out, finalists

    factor_validation_df, factor_final_candidates = _evaluate_family_configs(build_similarity_plus_factor_configs(smoke), "similarity_plus_factor")
    factor_best = None
    for cfg, stage_df, row in factor_final_candidates:
        family_final_rows.append(row)
        if factor_best is None or row["total_return"] > factor_best["summary"]["total_return"]:
            selected = select_with_gate(stage_df, "rank_score", cfg.sim_gate, cfg.daily_topn)
            factor_best = {"cfg": cfg, "stage_df": stage_df, "selected": selected, "summary": row}
    if factor_best:
        family_best_final["similarity_plus_factor"] = factor_best["summary"]

    ml_validation_df, ml_final_candidates = _evaluate_family_configs(build_similarity_plus_ml_configs(smoke), "similarity_plus_ml")
    ml_best = None
    for cfg, stage_df, row in ml_final_candidates:
        family_final_rows.append(row)
        if ml_best is None or row["total_return"] > ml_best["summary"]["total_return"]:
            selected = select_with_gate(stage_df, "rank_score", cfg.sim_gate, cfg.daily_topn)
            ml_best = {"cfg": cfg, "stage_df": stage_df, "selected": selected, "summary": row}
    if ml_best:
        family_best_final["similarity_plus_ml"] = ml_best["summary"]

    fusion_validation_df, fusion_final_candidates = _evaluate_family_configs(build_full_fusion_configs(smoke), "full_fusion")
    fusion_best = None
    for cfg, stage_df, row in fusion_final_candidates:
        family_final_rows.append(row)
        if fusion_best is None or row["total_return"] > fusion_best["summary"]["total_return"]:
            selected = select_with_gate(stage_df, "rank_score", cfg.sim_gate, cfg.daily_topn)
            fusion_best = {"cfg": cfg, "stage_df": stage_df, "selected": selected, "summary": row}
    if fusion_best:
        family_best_final["full_fusion"] = fusion_best["summary"]

    family_final_df = pd.DataFrame(family_final_rows).sort_values(["total_return", "max_drawdown", "win_rate"], ascending=[False, False, False]).reset_index(drop=True)
    validation_summary = pd.concat(
        [
            similarity_validation_df,
            factor_validation_df,
            ml_validation_df,
            fusion_validation_df,
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["total_return", "max_drawdown", "win_rate"], ascending=[False, False, False]).reset_index(drop=True)

    validation_summary.to_csv(output_dir / "validation_summary.csv", index=False, encoding="utf-8-sig")
    validation_summary.head(20).to_csv(output_dir / "validation_top20.csv", index=False, encoding="utf-8-sig")
    family_final_df.to_csv(output_dir / "final_test_summary.csv", index=False, encoding="utf-8-sig")
    update_progress(output_dir, "families_ready", validation_count=len(validation_summary), final_count=len(family_final_df))

    overall_best = family_final_df.iloc[0].to_dict()
    best_strategy_name = overall_best["strategy"]
    if similarity_final_best and similarity_final_best["summary"]["strategy"] == best_strategy_name:
        final_candidate_scored = similarity_final_best["stage_df"].copy()
    elif factor_best and factor_best["summary"]["strategy"] == best_strategy_name:
        final_candidate_scored = factor_best["stage_df"].copy()
    elif ml_best and ml_best["summary"]["strategy"] == best_strategy_name:
        final_candidate_scored = ml_best["stage_df"].copy()
    elif fusion_best and fusion_best["summary"]["strategy"] == best_strategy_name:
        final_candidate_scored = fusion_best["stage_df"].copy()
    if final_candidate_scored is not None:
        final_candidate_scored.to_csv(output_dir / "candidate_scored.csv", index=False, encoding="utf-8-sig")

    selected_map = {
        similarity_final_best["summary"]["strategy"]: similarity_final_best["selected"],
    }
    if factor_best:
        selected_map[factor_best["summary"]["strategy"]] = factor_best["selected"]
    if ml_best:
        selected_map[ml_best["summary"]["strategy"]] = ml_best["selected"]
    if fusion_best:
        selected_map[fusion_best["summary"]["strategy"]] = fusion_best["selected"]

    exit_rows = []
    best_exit_equity = pd.DataFrame()
    best_exit_trades = pd.DataFrame()
    for family_name, family_summary in family_best_final.items():
        family_strategy = family_summary["strategy"]
        selected_df = selected_map.get(family_strategy)
        if selected_df is None or selected_df.empty:
            continue
        exit_df, detail_df = run_exit_grid(feature_map, selected_df, family_strategy)
        if exit_df.empty:
            continue
        exit_df["family"] = family_name
        exit_rows.append(exit_df.head(5))
        best_exit = exit_df.iloc[0].to_dict()
        if family_name == overall_best["family"]:
            best_detail = detail_df[detail_df["strategy"] == best_exit["strategy"]]
            if not best_detail.empty:
                trade_df = best_detail.iloc[0]["trade_df"]
                if isinstance(trade_df, pd.DataFrame) and not trade_df.empty:
                    best_exit_trades = trade_df
                    best_exit_equity = build_portfolio_curve(trade_df)
    exit_summary_df = pd.concat(exit_rows, ignore_index=True, sort=False) if exit_rows else pd.DataFrame()
    exit_summary_df.to_csv(output_dir / "exit_grid_summary.csv", index=False, encoding="utf-8-sig")
    best_exit_trades.to_csv(output_dir / "best_trades.csv", index=False, encoding="utf-8-sig")
    best_exit_equity.to_csv(output_dir / "best_equity.csv", index=False, encoding="utf-8-sig")
    update_progress(output_dir, "exit_grid_ready", exit_rows=len(exit_summary_df))

    all_rows_df = records_to_df(all_records)
    rolling_targets = []
    seen = set()
    for row in family_final_df.head(10).to_dict(orient="records"):
        key = row["strategy"]
        if key in seen:
            continue
        seen.add(key)
        cfg_obj = cfg_from_row(row)
        all_scored = build_pool_scored_frame(all_rows_df, champion_all_self, cfg_obj.candidate_pool, "sim_score")
        if all_scored.empty:
            continue
        all_ranked = prepare_ranked_stage(all_scored, all_scored, cfg_obj)
        rolling_targets.append((key, all_ranked, cfg_obj))
    rolling_rows = []
    for strategy_name, stage_df, cfg_obj in rolling_targets:
        rolling_df = run_fixed_config_rolling_validation(stage_df, cfg_obj)
        if rolling_df.empty:
            continue
        rolling_df["strategy"] = strategy_name
        rolling_rows.append(rolling_df)
    rolling_results_df = pd.concat(rolling_rows, ignore_index=True, sort=False) if rolling_rows else pd.DataFrame()
    rolling_results_df.to_csv(output_dir / "rolling_window_results.csv", index=False, encoding="utf-8-sig")
    if not rolling_results_df.empty:
        rolling_summary_df = rolling_results_df.groupby("strategy", as_index=False).agg(
            window_count=("window_id", "count"),
            mean_total_return=("total_return", "mean"),
            median_total_return=("total_return", "median"),
            worst_total_return=("total_return", "min"),
            mean_max_drawdown=("max_drawdown", "mean"),
            positive_window_rate=("total_return", lambda s: float((pd.Series(s) > 0).mean())),
        )
    else:
        rolling_summary_df = pd.DataFrame()
    rolling_summary_df.to_csv(output_dir / "rolling_window_summary.csv", index=False, encoding="utf-8-sig")
    update_progress(output_dir, "rolling_ready", rolling_rows=len(rolling_results_df))

    formal_best_as_account = {
        "total_return": float(formal_best_baseline["final_equity_signal_basket"] / BRICK_INITIAL_CAPITAL - 1.0),
        "max_drawdown": float(formal_best_baseline["max_drawdown_signal_basket"]),
        "win_rate": float(formal_best_baseline["success_rate"]),
        "ending_equity": float(formal_best_baseline["final_equity_signal_basket"]),
    }
    for row in family_final_df.to_dict(orient="records"):
        _append_comparison(row, "relaxed_baseline", relaxed_baseline_summary)
        _append_comparison(row, "formal_best_baseline", formal_best_as_account)
        _append_comparison(row, "relaxed_similarity_champion", relaxed_similarity_champion_summary)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "comparison_to_baselines.csv", index=False, encoding="utf-8-sig")

    rolling_gate_ok = False
    if not rolling_summary_df.empty and best_strategy_name in set(rolling_summary_df["strategy"]):
        best_roll = rolling_summary_df[rolling_summary_df["strategy"] == best_strategy_name].iloc[0]
        rolling_gate_ok = bool(best_roll.get("positive_window_rate", 0.0) >= 0.60)
    summary = {
        "name": config.get("name", "brick_comprehensive_lab"),
        "smoke": smoke,
        "file_limit": file_limit,
        "split_meta": split_meta,
        "formal_best_baseline": formal_best_baseline,
        "relaxed_baseline": relaxed_baseline_summary,
        "relaxed_similarity_champion": relaxed_similarity_champion_summary,
        "family_best_final": family_best_final,
        "overall_best": overall_best,
        "stability_gate": {
            "final_test_beats_relaxed_baseline": bool(overall_best["total_return"] > relaxed_baseline_summary["total_return"]),
            "rolling_results_available": bool(not rolling_summary_df.empty),
            "rolling_positive_window_rate_ge_60pct": rolling_gate_ok,
        },
        "conclusion": f"BRICK 综合实验当前总榜第一为 {best_strategy_name}，final_test total_return={overall_best['total_return']:.4f}。",
    }
    write_json(output_dir / "best_config.json", overall_best)
    write_json(output_dir / "summary.json", summary)
    update_ledger(output_dir, summary)
    update_progress(output_dir, "finished", output_path=str(output_dir))


if __name__ == "__main__":
    main()
