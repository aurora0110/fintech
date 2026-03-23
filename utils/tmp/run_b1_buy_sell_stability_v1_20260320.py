from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = ROOT / "results" / f"b1_buy_sell_stability_v1_{RUN_TS}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BUY_V1_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
BUY_V2_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v2_20260320.py"
BUY_V4_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v4_20260320.py"
SELL_HABIT_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_sell_habit_experiment_v1_20260320.py"
ACCOUNT_V1_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_buy_sell_model_account_v1_20260320.py"

PREV_SIGNAL_RESULT = ROOT / "results" / "b1_similarity_ml_signal_20260320_164521"
PREV_V2_RESULT = ROOT / "results" / "b1_similarity_ml_signal_v2_20260320_210752"
SELL_RESULT_DIR = ROOT / "results" / "b1_sell_habit_experiment_v1_20260320_223347"

BUY_STRATEGY_TAG = "similarity_corr_close_vol_concat_pool_all_top3"
FIXED_TP_LEVELS = [0.20, 0.30]
MODEL_SCORE_COLS = ["xgb_score", "lgb_score", "rule_score"]
THRESHOLD_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
WINDOW_SIGNAL_DATES = 20
WINDOW_STEP = 10


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


buy_v1 = load_module(BUY_V1_SCRIPT, "b1_stab_v1")
buy_v2 = load_module(BUY_V2_SCRIPT, "b1_stab_v2")
buy_v4 = load_module(BUY_V4_SCRIPT, "b1_stab_v4")
sell_mod = load_module(SELL_HABIT_SCRIPT, "b1_stab_sell")
account_mod = load_module(ACCOUNT_V1_SCRIPT, "b1_stab_account")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(stage: str, extra: Dict[str, Any] | None = None) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    if extra:
        payload.update(extra)
    write_json(RESULT_DIR / "progress.json", payload)


def load_candidate_df() -> pd.DataFrame:
    candidate_pkl = PREV_SIGNAL_RESULT / "candidate_rows.pkl"
    df = pd.read_pickle(candidate_pkl)
    cutoffs = json.loads((PREV_V2_RESULT / "split_cutoffs.json").read_text(encoding="utf-8"))
    cutoffs = {k: pd.Timestamp(v) for k, v in cutoffs.items()}
    df["split"] = buy_v1.assign_split(df, cutoffs)
    return df


def build_positive_df() -> pd.DataFrame:
    name_code_map = buy_v1.load_name_code_map()
    raw = buy_v2.parse_b1_case_files()
    enriched = pd.DataFrame([buy_v2.enrich_case(r, name_code_map) for _, r in raw.iterrows()])
    enriched = enriched[enriched["status"] == "ok"].copy().reset_index(drop=True)
    cutoffs = json.loads((PREV_V2_RESULT / "split_cutoffs.json").read_text(encoding="utf-8"))
    cutoffs = {k: pd.Timestamp(v) for k, v in cutoffs.items()}
    enriched["split"] = buy_v1.assign_split(enriched, cutoffs)
    return enriched


def build_selected_signals() -> pd.DataFrame:
    candidate_df = load_candidate_df()
    positive_df = build_positive_df()
    research_positive = positive_df[positive_df["split"] == "research"].copy()
    research_negative = candidate_df[(candidate_df["split"] == "research") & (candidate_df["negative_30d"])].copy()
    research_negative = research_negative.sort_values("signal_date").head(max(len(research_positive) * 5, len(research_positive))).copy()
    cand_maps = [buy_v4.derive_rep_map(r["seq_map"]) for _, r in candidate_df.iterrows()]
    pos_maps = [buy_v4.derive_rep_map(r["seq_map"]) for _, r in research_positive.iterrows()]
    cand_seqs = [m["close_vol_concat"] for m in cand_maps]
    pos_templates = [m["close_vol_concat"] for m in pos_maps]
    candidate_df["sim_corr_close_vol_concat"] = buy_v4.compute_similarity_column(
        cand_seqs,
        pos_templates,
        "corr",
    )
    candidate_df = buy_v4.add_pool_flags(candidate_df)

    part = candidate_df[
        candidate_df["split"].isin(["validation", "final_test"]) & candidate_df["pool_all"]
    ].copy()
    selected = buy_v4.select_daily_topn(part, "sim_corr_close_vol_concat", 3, ascending=False)
    selected["strategy_tag"] = BUY_STRATEGY_TAG
    selected.to_csv(RESULT_DIR / "selected_signals_validation_final.csv", index=False, encoding="utf-8-sig")
    return selected


def choose_thresholds(validation_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for score_col in MODEL_SCORE_COLS:
        if score_col == "rule_score":
            sub = validation_df.copy()
            sub["rule_score"] = sell_mod.score_rule(sub)
        elif score_col not in validation_df.columns:
            continue
        else:
            sub = validation_df[[score_col, "label"]].dropna().copy()
        if score_col == "rule_score":
            sub = sub[[score_col, "label"]].dropna().copy()
        if sub.empty or sub["label"].nunique() < 2:
            continue
        best = None
        for q in THRESHOLD_GRID:
            threshold = float(sub[score_col].quantile(q))
            pred = (sub[score_col] >= threshold).astype(int)
            tp = int(((pred == 1) & (sub["label"] == 1)).sum())
            fp = int(((pred == 1) & (sub["label"] == 0)).sum())
            fn = int(((pred == 0) & (sub["label"] == 1)).sum())
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            hit_rate = float(pred.mean())
            candidate = {
                "score_col": score_col,
                "quantile": q,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hit_rate": hit_rate,
            }
            if best is None or (candidate["f1"], candidate["precision"], -candidate["hit_rate"]) > (
                best["f1"],
                best["precision"],
                -best["hit_rate"],
            ):
                best = candidate
        if best:
            rows.append(best)
    return pd.DataFrame(rows).sort_values(["f1", "precision"], ascending=[False, False]).reset_index(drop=True)


def build_combo_plan(threshold_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for tp in FIXED_TP_LEVELS:
        rows.append(
            {
                "strategy_tag": BUY_STRATEGY_TAG,
                "exit_mode": "fixed_tp",
                "take_profit_pct": float(tp),
                "score_col": "",
                "threshold": np.nan,
                "threshold_quantile": np.nan,
            }
        )
    for _, row in threshold_df.iterrows():
        rows.append(
            {
                "strategy_tag": BUY_STRATEGY_TAG,
                "exit_mode": "model_only",
                "take_profit_pct": np.nan,
                "score_col": str(row["score_col"]),
                "threshold": float(row["threshold"]),
                "threshold_quantile": float(row["quantile"]),
            }
        )
        rows.append(
            {
                "strategy_tag": BUY_STRATEGY_TAG,
                "exit_mode": "model_plus_tp",
                "take_profit_pct": 0.20,
                "score_col": str(row["score_col"]),
                "threshold": float(row["threshold"]),
                "threshold_quantile": float(row["quantile"]),
            }
        )
    return pd.DataFrame(rows)


def run_period_backtests(
    period_name: str,
    signal_df: pd.DataFrame,
    stock_cache: Dict[str, pd.DataFrame],
    all_dates: List[pd.Timestamp],
    combo_df: pd.DataFrame,
    fitted_models: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[dict] = []
    for _, combo in combo_df.iterrows():
        equity_df, trade_df, summary = account_mod.run_account_backtest(
            all_dates=all_dates,
            stock_cache=stock_cache,
            signal_df=signal_df,
            combo=combo.to_dict(),
            fitted_models=fitted_models,
        )
        stem = f"{period_name}__{combo['exit_mode']}"
        if pd.notna(combo.get("score_col")) and combo.get("score_col"):
            stem += f"__{combo['score_col']}"
        if pd.notna(combo.get("take_profit_pct", np.nan)):
            stem += f"__tp{int(float(combo['take_profit_pct']) * 100)}"
        equity_df.to_csv(RESULT_DIR / f"equity_{stem}.csv", index=False, encoding="utf-8-sig")
        trade_df.to_csv(RESULT_DIR / f"trades_{stem}.csv", index=False, encoding="utf-8-sig")
        rows.append({"period": period_name, **combo.to_dict(), **summary})
    return pd.DataFrame(rows)


def build_signal_date_windows(signal_df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    signal_dates = sorted(pd.to_datetime(signal_df["signal_date"].drop_duplicates()))
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    window_idx = 0
    for start in range(0, len(signal_dates), WINDOW_STEP):
        end = start + WINDOW_SIGNAL_DATES
        if end > len(signal_dates):
            break
        window_idx += 1
        windows.append((signal_dates[start], signal_dates[end - 1], window_idx))
    return windows


def rolling_stability(
    signal_df: pd.DataFrame,
    stock_cache: Dict[str, pd.DataFrame],
    all_dates: List[pd.Timestamp],
    combo_df: pd.DataFrame,
    fitted_models: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: List[dict] = []
    windows = build_signal_date_windows(signal_df)
    for start_date, end_date, window_idx in windows:
        sub = signal_df[
            (pd.to_datetime(signal_df["signal_date"]) >= start_date)
            & (pd.to_datetime(signal_df["signal_date"]) <= end_date)
        ].copy()
        if sub.empty:
            continue
        all_dates_window = [
            d for d in all_dates if d >= start_date and d <= (end_date + pd.Timedelta(days=120))
        ]
        if len(all_dates_window) < 2:
            continue
        for _, combo in combo_df.iterrows():
            equity_df, trade_df, summary = account_mod.run_account_backtest(
                all_dates=all_dates_window,
                stock_cache=stock_cache,
                signal_df=sub,
                combo=combo.to_dict(),
                fitted_models=fitted_models,
            )
            detail_rows.append(
                {
                    "window_idx": window_idx,
                    "window_start": start_date,
                    "window_end": end_date,
                    **combo.to_dict(),
                    **summary,
                }
            )
    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame()
    summary_df = (
        detail_df.groupby(["exit_mode", "score_col", "take_profit_pct"], dropna=False)
        .agg(
            window_count=("final_multiple", "size"),
            mean_final_multiple=("final_multiple", "mean"),
            median_final_multiple=("final_multiple", "median"),
            positive_window_ratio=("final_multiple", lambda s: float((s > 1.0).mean())),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_trade_count=("trade_count", "mean"),
            mean_win_rate=("win_rate", "mean"),
        )
        .reset_index()
        .sort_values(["mean_final_multiple", "positive_window_ratio"], ascending=[False, False])
    )
    return detail_df, summary_df


def main() -> None:
    update_progress("starting")
    selected_df = build_selected_signals()
    update_progress("signals_ready", {"signal_count": int(len(selected_df))})

    sell_sample_df = pd.read_csv(SELL_RESULT_DIR / "sell_samples.csv")
    sell_sample_df["sample_date"] = pd.to_datetime(sell_sample_df["sample_date"])
    research_sell = sell_sample_df[sell_sample_df["split"] == "research"].copy()
    validation_sell = sell_sample_df[sell_sample_df["split"] == "validation"].copy()
    validation_scores = account_mod.fit_sell_models(research_sell, validation_sell)
    for k, s in validation_scores.items():
        validation_sell[k] = s
    validation_sell["rule_score"] = sell_mod.score_rule(validation_sell)
    threshold_df = choose_thresholds(validation_sell)
    threshold_df.to_csv(RESULT_DIR / "sell_thresholds.csv", index=False, encoding="utf-8-sig")
    combo_df = build_combo_plan(threshold_df)
    combo_df.to_csv(RESULT_DIR / "combo_plan.csv", index=False, encoding="utf-8-sig")

    train_sell = sell_sample_df[sell_sample_df["split"].isin(["research", "validation"])].copy()
    fitted_models = account_mod.train_final_models(train_sell)
    update_progress(
        "sell_models_ready",
        {
            "threshold_count": int(len(threshold_df)),
            "fitted_models": list(fitted_models.keys()),
        },
    )

    needed_codes = sorted(selected_df["code"].astype(str).unique().tolist())
    stock_cache = account_mod.load_stock_feature_cache(needed_codes)
    all_dates = sorted(set(pd.concat([pd.to_datetime(df["date"]) for df in stock_cache.values()]).tolist()))
    update_progress(
        "stock_cache_ready",
        {
            "stock_count": int(len(stock_cache)),
            "date_count": int(len(all_dates)),
            "combo_count": int(len(combo_df)),
        },
    )

    period_rows: List[pd.DataFrame] = []
    for period_name in ["validation", "final_test", "combined"]:
        if period_name == "combined":
            period_signal = selected_df[selected_df["split"].isin(["validation", "final_test"])].copy()
        else:
            period_signal = selected_df[selected_df["split"] == period_name].copy()
        res = run_period_backtests(period_name, period_signal, stock_cache, all_dates, combo_df, fitted_models)
        res.to_csv(RESULT_DIR / f"account_results_{period_name}.csv", index=False, encoding="utf-8-sig")
        period_rows.append(res)
        update_progress("period_done", {"period": period_name, "rows": int(len(res))})

    all_period_df = pd.concat(period_rows, ignore_index=True)
    all_period_df.to_csv(RESULT_DIR / "account_results_all_periods.csv", index=False, encoding="utf-8-sig")

    combined_signal = selected_df[selected_df["split"].isin(["validation", "final_test"])].copy()
    rolling_detail_df, rolling_summary_df = rolling_stability(
        combined_signal,
        stock_cache,
        all_dates,
        combo_df,
        fitted_models,
    )
    rolling_detail_df.to_csv(RESULT_DIR / "rolling_window_results.csv", index=False, encoding="utf-8-sig")
    rolling_summary_df.to_csv(RESULT_DIR / "rolling_window_summary.csv", index=False, encoding="utf-8-sig")

    invalid_mask = (
        (all_period_df["max_drawdown"] < -1.0)
        | (all_period_df["final_multiple"] <= 0)
        | (((all_period_df["cagr"] > 0) & (all_period_df["final_multiple"] < 1)) | ((all_period_df["cagr"] < 0) & (all_period_df["final_multiple"] > 1)))
    )
    invalid_df = all_period_df.loc[invalid_mask].copy()
    invalid_df.to_csv(RESULT_DIR / "sanity_invalid_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "result_dir": str(RESULT_DIR),
        "buy_strategy_tag": BUY_STRATEGY_TAG,
        "signal_count_total": int(len(selected_df)),
        "signal_count_validation": int((selected_df["split"] == "validation").sum()),
        "signal_count_final_test": int((selected_df["split"] == "final_test").sum()),
        "combo_count": int(len(combo_df)),
        "best_validation_row": all_period_df[all_period_df["period"] == "validation"].sort_values(["final_multiple", "sharpe"], ascending=[False, False]).iloc[0].to_dict(),
        "best_final_test_row": all_period_df[all_period_df["period"] == "final_test"].sort_values(["final_multiple", "sharpe"], ascending=[False, False]).iloc[0].to_dict(),
        "best_combined_row": all_period_df[all_period_df["period"] == "combined"].sort_values(["final_multiple", "sharpe"], ascending=[False, False]).iloc[0].to_dict(),
        "best_rolling_row": rolling_summary_df.iloc[0].to_dict() if not rolling_summary_df.empty else {},
        "sanity_invalid_count": int(len(invalid_df)),
    }
    write_json(RESULT_DIR / "summary.json", summary)
    update_progress("finished", {"combo_count": int(len(combo_df))})


if __name__ == "__main__":
    main()
