from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
BUY_SIGNAL_DIR = ROOT / "results" / "b1_similarity_ml_signal_v4_20260320_220031"
SELL_MODEL_DIR = ROOT / "results" / "b1_sell_habit_experiment_v1_20260320_223347"
FORWARD_DATA_DIR = ROOT / "data" / "forward_data"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_RESULT_DIR = ROOT / "results" / f"b1_buy_sell_model_account_v1_{RUN_TS}"

V1_BUY_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
SELL_HABIT_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_sell_habit_experiment_v1_20260320.py"
BACKTEST_UTIL = ROOT / "utils" / "backtest" / "backtest_b1_strategy.py"
RISK_SCRIPT = ROOT / "utils" / "market_risk_tags.py"

BUY_STRATEGY_TAGS = [
    "similarity_corr_close_vol_concat_pool_all_top3",
    "fusion_sim_fusion_pool_near_trend_top3",
    "baseline_lowest_J_pool_core_top5",
]
FIXED_TP_LEVELS = [0.20, 0.30]
SELL_MODEL_COLS = [
    "rule_score",
    "rf_score",
    "et_score",
    "lgb_score",
    "xgb_score",
]
THRESHOLD_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

INITIAL_CAPITAL = 1_000_000.0
FEE_RATE = 0.001
SLIPPAGE = 0.001
STOP_MULTIPLIER = 0.90
MAX_HOLD_DAYS = 60
SINGLE_POS_CAP = 0.10
DAY_CASH_CAP = 1.00
MAX_POSITIONS = 999
PAUSE_AFTER_STOPS = 3
PAUSE_DAYS = 5
PROGRESS_STEP = 100


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


buy_mod = load_module(V1_BUY_SCRIPT, "b1_buy_mod")
sell_mod = load_module(SELL_HABIT_SCRIPT, "b1_sell_mod")
bt_mod = load_module(BACKTEST_UTIL, "b1_bt_mod")
risk_mod = load_module(RISK_SCRIPT, "risk_mod")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 买点主线 + 卖点模型账户层回测")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--limit-combos", type=int, default=0, help="仅跑前N个组合，0表示全量")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, extra: dict | None = None) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    if extra:
        payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def get_stock_loc(df: pd.DataFrame, date: pd.Timestamp) -> int:
    matches = np.flatnonzero(pd.to_datetime(df["date"]).to_numpy() == np.datetime64(date))
    return int(matches[-1]) if len(matches) else -1


def load_selected_signals() -> pd.DataFrame:
    path = BUY_SIGNAL_DIR / "final_test_selected_rows.csv"
    df = pd.read_csv(path)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df[df["strategy_tag"].isin(BUY_STRATEGY_TAGS)].copy()
    return df


def load_sell_samples() -> pd.DataFrame:
    path = SELL_MODEL_DIR / "sell_samples.csv"
    df = pd.read_csv(path)
    df["sample_date"] = pd.to_datetime(df["sample_date"])
    return df


def choose_thresholds(validation_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for score_col in SELL_MODEL_COLS:
        if score_col not in validation_df.columns:
            continue
        sub = validation_df[[score_col, "label"]].dropna().copy()
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
    return pd.DataFrame(rows).sort_values(["f1", "precision"], ascending=[False, False])


def fit_sell_models(research_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, pd.Series]:
    return sell_mod.fit_and_score(research_df, test_df)


def load_stock_feature_cache(codes: List[str]) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    total = len(codes)
    for i, code in enumerate(sorted(set(codes)), 1):
        path = FORWARD_DATA_DIR / f"{code}.txt"
        df = buy_mod.load_stock_data(str(path))
        if df is None or df.empty:
            continue
        feat = buy_mod.compute_b1_features(df).copy()
        base_df = feat[["open", "high", "low", "close", "volume", "trend_line", "long_line"]].copy()
        risk_df = risk_mod.add_risk_features(base_df, precomputed_base=base_df)
        for col in [
            "recent_heavy_bear_top_20",
            "recent_failed_breakout_20",
            "top_distribution_20",
            "recent_stair_bear_20",
            "risk_fast_rise_10d_40",
            "risk_segment_rise_slope_10_006",
            "risk_distribution_any_20",
        ]:
            feat[col] = risk_df[col].astype(int)
        cache[code] = feat
        if i % 50 == 0 or i == total:
            print(f"特征缓存加载进度: {i}/{total}")
    return cache


def build_daily_sell_features(
    feat: pd.DataFrame,
    buy_idx: int,
    cur_idx: int,
) -> dict:
    feat_row = feat.iloc[cur_idx]
    buy_close = float(feat.iloc[buy_idx]["close"])
    cur_close = float(feat_row["close"])
    peak_high = float(feat.iloc[buy_idx : cur_idx + 1]["high"].max())
    profit_since_buy = cur_close / buy_close - 1.0 if buy_close > 0 else np.nan
    drawdown_from_peak = cur_close / peak_high - 1.0 if peak_high > 0 else np.nan
    return {
        "hold_day_idx": int(cur_idx - buy_idx),
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
        "risk_recent_heavy_bear_top_20": int(feat_row["recent_heavy_bear_top_20"]),
        "risk_recent_failed_breakout_20": int(feat_row["recent_failed_breakout_20"]),
        "risk_top_distribution_20": int(feat_row["top_distribution_20"]),
        "risk_recent_stair_bear_20": int(feat_row["recent_stair_bear_20"]),
        "risk_fast_rise_10d_40": int(feat_row["risk_fast_rise_10d_40"]),
        "risk_segment_rise_slope_10_006": int(feat_row["risk_segment_rise_slope_10_006"]),
        "risk_distribution_any_20": int(feat_row["risk_distribution_any_20"]),
    }


def compute_sell_score_row(model_name: str, row: dict, fitted_models: Dict[str, Any]) -> float:
    if model_name == "rule_score":
        return float(sell_mod.score_rule(pd.DataFrame([row])).iloc[0])
    X = sell_mod.build_feature_matrix(pd.DataFrame([row]))
    model = fitted_models.get(model_name)
    if model is None:
        return np.nan
    return float(model.predict_proba(X)[:, 1][0])


def train_final_models(train_df: pd.DataFrame) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    X_train = sell_mod.build_feature_matrix(train_df)
    y_train = train_df["label"].to_numpy(dtype=int)

    if "logistic_score" in SELL_MODEL_COLS:
        pass
    if getattr(sell_mod, "HAS_SKLEARN", False):
        try:
            rf = sell_mod.RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            models["rf_score"] = rf
        except Exception:
            pass
        try:
            et = sell_mod.ExtraTreesClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            et.fit(X_train, y_train)
            models["et_score"] = et
        except Exception:
            pass
    if getattr(sell_mod, "HAS_LGB", False):
        try:
            clf = sell_mod.lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                min_child_samples=5,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            models["lgb_score"] = clf
        except Exception:
            pass
    if getattr(sell_mod, "HAS_XGB", False):
        try:
            clf = sell_mod.xgb.XGBClassifier(
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
            models["xgb_score"] = clf
        except Exception:
            pass
    return models


def summarize_account(
    equity_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    stop_loss_exit_count: int,
    pause_event_count: int,
    max_consecutive_stop_losses: int,
) -> dict:
    equity_arr = equity_df["equity"].to_numpy(dtype=float)
    daily_ret = equity_df["daily_return"].fillna(0.0).to_numpy(dtype=float)
    final_multiple = float(equity_arr[-1] / INITIAL_CAPITAL) if len(equity_arr) else np.nan
    total_days = int(len(equity_df))
    closed_trade_count = int(len(trade_df))
    win_trade_count = int((trade_df["return_pct"] > 0).sum()) if closed_trade_count else 0
    win_rate = float(win_trade_count / closed_trade_count) if closed_trade_count else np.nan
    avg_trade_return = float(trade_df["return_pct"].mean() / 100.0) if closed_trade_count else np.nan
    profit_factor = float(bt_mod.calc_profit_factor(trade_df["return_pct"] / 100.0)) if closed_trade_count else np.nan
    return {
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": float(equity_arr[-1]) if len(equity_arr) else np.nan,
        "final_multiple": final_multiple,
        "cagr": float(bt_mod.calc_cagr(final_multiple, total_days)) if len(equity_arr) else np.nan,
        "max_drawdown": float(bt_mod.calc_max_drawdown(equity_arr)) if len(equity_arr) else np.nan,
        "sharpe": float(bt_mod.calc_sharpe(daily_ret)) if len(equity_arr) > 1 else np.nan,
        "trade_count": closed_trade_count,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "profit_factor": profit_factor,
        "stop_loss_exit_count": int(stop_loss_exit_count),
        "pause_event_count": int(pause_event_count),
        "max_consecutive_stop_losses": int(max_consecutive_stop_losses),
    }


def run_account_backtest(
    all_dates: List[pd.Timestamp],
    stock_cache: Dict[str, pd.DataFrame],
    signal_df: pd.DataFrame,
    combo: dict,
    fitted_models: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    strategy_tag = combo["strategy_tag"]
    exit_mode = combo["exit_mode"]
    score_col = combo.get("score_col", "")
    threshold = combo.get("threshold", np.nan)
    tp_pct = combo.get("take_profit_pct", np.nan)

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    pending_buy: Dict[pd.Timestamp, List[dict]] = {}
    for _, row in signal_df.iterrows():
        pending_buy.setdefault(pd.Timestamp(row["entry_date"]), []).append(row.to_dict())

    cash = INITIAL_CAPITAL
    positions: List[dict] = []
    equity_rows: List[dict] = []
    trade_rows: List[dict] = []
    consecutive_stop_losses = 0
    max_consecutive_stop_losses = 0
    pause_days_left = 0
    pause_event_count = 0
    stop_loss_exit_count = 0
    total_days = len(all_dates)

    for day_i, current_date in enumerate(all_dates, 1):
        current_idx_global = day_i - 1
        if pause_days_left > 0:
            pause_days_left -= 1

        remaining_positions: List[dict] = []
        for pos in positions:
            if pos.get("scheduled_exit_date") != current_date:
                remaining_positions.append(pos)
                continue
            feat = stock_cache[pos["stock"]]
            stock_idx = get_stock_loc(feat, current_date)
            if stock_idx < 0:
                remaining_positions.append(pos)
                continue
            exit_price = float(feat.iloc[stock_idx]["open"])
            if not np.isfinite(exit_price) or exit_price <= 0:
                remaining_positions.append(pos)
                continue
            proceeds = pos["shares"] * exit_price * (1 - FEE_RATE - SLIPPAGE)
            cash += proceeds
            pnl = exit_price / pos["entry_price"] - 1.0
            hold_days = current_idx_global - pos["entry_global_idx"]
            reason = pos.get("scheduled_exit_reason", "unknown")

            if reason == "stop_loss":
                stop_loss_exit_count += 1
                consecutive_stop_losses += 1
                max_consecutive_stop_losses = max(max_consecutive_stop_losses, consecutive_stop_losses)
                if consecutive_stop_losses >= PAUSE_AFTER_STOPS:
                    pause_days_left = PAUSE_DAYS
                    pause_event_count += 1
                    consecutive_stop_losses = 0
            else:
                consecutive_stop_losses = 0

            trade_rows.append(
                {
                    "strategy_tag": strategy_tag,
                    "exit_mode": exit_mode,
                    "score_col": score_col,
                    "threshold": threshold,
                    "take_profit_pct": tp_pct,
                    "stock": pos["stock"],
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "exit_reason": reason,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "hold_days": hold_days,
                    "return_pct": pnl * 100.0,
                }
            )
        positions = remaining_positions

        for pos in positions:
            if pos.get("scheduled_exit_date") is not None:
                continue
            if current_idx_global <= pos["entry_global_idx"]:
                continue

            feat = stock_cache[pos["stock"]]
            stock_idx = get_stock_loc(feat, current_date)
            if stock_idx < 0:
                continue
            row = feat.iloc[stock_idx]
            high_p = float(row["high"]) if pd.notna(row["high"]) else np.nan
            low_p = float(row["low"]) if pd.notna(row["low"]) else np.nan
            hold_days = current_idx_global - pos["entry_global_idx"]

            stop_hit = np.isfinite(low_p) and low_p <= pos["stop_price"]
            tp_hit = False
            if exit_mode in {"fixed_tp", "model_plus_tp"} and np.isfinite(tp_pct):
                tp_price = pos["entry_price"] * (1.0 + tp_pct)
                tp_hit = np.isfinite(high_p) and high_p >= tp_price

            model_hit = False
            model_score = np.nan
            if exit_mode in {"model_only", "model_plus_tp"}:
                feat_row = build_daily_sell_features(feat, pos["entry_stock_idx"], stock_idx)
                model_score = compute_sell_score_row(score_col, feat_row, fitted_models)
                if np.isfinite(model_score) and np.isfinite(threshold):
                    model_hit = model_score >= threshold

            if current_idx_global + 1 >= total_days:
                continue
            next_date = all_dates[current_idx_global + 1]
            if stop_hit:
                pos["scheduled_exit_date"] = next_date
                pos["scheduled_exit_reason"] = "stop_loss"
            elif model_hit:
                pos["scheduled_exit_date"] = next_date
                pos["scheduled_exit_reason"] = f"model_{score_col}"
                pos["scheduled_exit_score"] = model_score
            elif tp_hit:
                pos["scheduled_exit_date"] = next_date
                pos["scheduled_exit_reason"] = f"tp_{int(tp_pct * 100)}"
            elif hold_days >= MAX_HOLD_DAYS:
                pos["scheduled_exit_date"] = next_date
                pos["scheduled_exit_reason"] = "time_exit_60"

        if pause_days_left == 0 and current_date in pending_buy:
            already_holding = {p["stock"] for p in positions}
            candidates = sorted(pending_buy[current_date], key=lambda x: (x["signal_date"], x["code"]))
            position_value_now = 0.0
            for pos in positions:
                feat = stock_cache[pos["stock"]]
                stock_idx = get_stock_loc(feat, current_date)
                if stock_idx >= 0:
                    mark_price = float(feat.iloc[stock_idx]["close"])
                    if np.isfinite(mark_price) and mark_price > 0:
                        position_value_now += pos["shares"] * mark_price
            total_equity_now = cash + position_value_now
            day_cash_limit = total_equity_now * DAY_CASH_CAP
            day_used_cash = 0.0

            for item in candidates:
                if len(positions) >= MAX_POSITIONS or cash <= 0 or day_used_cash >= day_cash_limit:
                    break
                stock = str(item["code"])
                if stock in already_holding or stock not in stock_cache:
                    continue
                feat = stock_cache[stock]
                stock_idx = get_stock_loc(feat, current_date)
                if stock_idx < 0:
                    continue
                entry_price = float(feat.iloc[stock_idx]["open"])
                entry_low = float(feat.iloc[stock_idx]["low"])
                if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(entry_low) or entry_low <= 0:
                    continue
                position_cap_cash = total_equity_now * SINGLE_POS_CAP
                budget = min(cash, position_cap_cash, day_cash_limit - day_used_cash)
                shares = int(budget // (entry_price * (1 + FEE_RATE + SLIPPAGE)))
                if shares <= 0:
                    continue
                cost = shares * entry_price * (1 + FEE_RATE + SLIPPAGE)
                if cost > cash + 1e-9:
                    continue
                cash -= cost
                day_used_cash += cost
                positions.append(
                    {
                        "stock": stock,
                        "signal_date": pd.Timestamp(item["signal_date"]),
                        "entry_date": current_date,
                        "entry_price": entry_price,
                        "stop_price": entry_low * STOP_MULTIPLIER,
                        "shares": shares,
                        "entry_global_idx": current_idx_global,
                        "entry_stock_idx": stock_idx,
                        "scheduled_exit_date": None,
                        "scheduled_exit_reason": None,
                    }
                )
                already_holding.add(stock)

        mtm = cash
        for pos in positions:
            feat = stock_cache[pos["stock"]]
            stock_idx = get_stock_loc(feat, current_date)
            if stock_idx >= 0:
                mark_px = float(feat.iloc[stock_idx]["close"])
            else:
                hist = feat.loc[pd.to_datetime(feat["date"]) <= current_date, "close"]
                mark_px = float(hist.iloc[-1]) if len(hist) else np.nan
            if np.isfinite(mark_px) and mark_px > 0:
                mtm += pos["shares"] * mark_px
        equity_rows.append(
            {
                "date": current_date,
                "equity": mtm,
                "cash": cash,
                "position_count": len(positions),
            }
        )
        if day_i % PROGRESS_STEP == 0 or day_i == total_days:
            print(f"账户层回放进度[{strategy_tag}|{exit_mode}|{score_col or 'na'}]: {day_i}/{total_days}")

    equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0.0)
    trade_df = pd.DataFrame(trade_rows)
    summary = summarize_account(
        equity_df,
        trade_df,
        stop_loss_exit_count=stop_loss_exit_count,
        pause_event_count=pause_event_count,
        max_consecutive_stop_losses=max_consecutive_stop_losses,
    )
    return equity_df, trade_df, summary


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    update_progress(result_dir, "loading_inputs")
    selected_df = load_selected_signals()
    sell_sample_df = load_sell_samples()

    validation_sell = sell_sample_df[sell_sample_df["split"] == "validation"].copy()
    research_sell = sell_sample_df[sell_sample_df["split"] == "research"].copy()
    train_sell = sell_sample_df[sell_sample_df["split"].isin(["research", "validation"])].copy()

    validation_scores = fit_sell_models(research_sell, validation_sell)
    for k, s in validation_scores.items():
        validation_sell[k] = s
    threshold_df = choose_thresholds(validation_sell)
    threshold_df.to_csv(result_dir / "sell_thresholds.csv", index=False, encoding="utf-8-sig")

    fitted_models = train_final_models(train_sell)
    update_progress(
        result_dir,
        "sell_models_ready",
        {
            "thresholds": int(len(threshold_df)),
            "fitted_models": list(fitted_models.keys()),
        },
    )

    buy_meta_rows: List[dict] = []
    combos: List[dict] = []
    for strategy_tag in BUY_STRATEGY_TAGS:
        sig = selected_df[selected_df["strategy_tag"] == strategy_tag].copy()
        buy_meta_rows.append({"strategy_tag": strategy_tag, "signal_count": int(len(sig)), "date_count": int(sig["signal_date"].nunique())})
        for tp in FIXED_TP_LEVELS:
            combos.append({"strategy_tag": strategy_tag, "exit_mode": "fixed_tp", "take_profit_pct": float(tp), "score_col": "", "threshold": np.nan})
        for _, row in threshold_df.iterrows():
            combos.append(
                {
                    "strategy_tag": strategy_tag,
                    "exit_mode": "model_only",
                    "take_profit_pct": np.nan,
                    "score_col": str(row["score_col"]),
                    "threshold": float(row["threshold"]),
                    "threshold_quantile": float(row["quantile"]),
                }
            )
            combos.append(
                {
                    "strategy_tag": strategy_tag,
                    "exit_mode": "model_plus_tp",
                    "take_profit_pct": 0.20,
                    "score_col": str(row["score_col"]),
                    "threshold": float(row["threshold"]),
                    "threshold_quantile": float(row["quantile"]),
                }
            )

    if args.limit_combos > 0:
        combos = combos[: args.limit_combos]
    pd.DataFrame(buy_meta_rows).to_csv(result_dir / "buy_strategy_meta.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(combos).to_csv(result_dir / "combo_plan.csv", index=False, encoding="utf-8-sig")

    needed_codes = sorted(selected_df["code"].astype(str).unique().tolist())
    update_progress(
        result_dir,
        "loading_stock_features",
        {
            "stock_count": int(len(needed_codes)),
            "combo_count": int(len(combos)),
        },
    )
    stock_cache = load_stock_feature_cache(needed_codes)
    all_dates = sorted(set(pd.concat([pd.to_datetime(df["date"]) for df in stock_cache.values()]).tolist()))

    results: List[dict] = []
    total = len(combos)
    for i, combo in enumerate(combos, 1):
        strategy_tag = combo["strategy_tag"]
        sig = selected_df[selected_df["strategy_tag"] == strategy_tag].copy()
        equity_df, trade_df, summary = run_account_backtest(all_dates, stock_cache, sig, combo, fitted_models)

        stem = f"{strategy_tag}__{combo['exit_mode']}"
        if combo.get("score_col"):
            stem += f"__{combo['score_col']}"
        if np.isfinite(combo.get("take_profit_pct", np.nan)):
            stem += f"__tp{int(combo['take_profit_pct'] * 100)}"

        equity_df.to_csv(result_dir / f"equity_{stem}.csv", index=False, encoding="utf-8-sig")
        trade_df.to_csv(result_dir / f"trades_{stem}.csv", index=False, encoding="utf-8-sig")

        row = {**combo, **summary}
        results.append(row)
        pd.DataFrame(results).to_csv(result_dir / "account_results_partial.csv", index=False, encoding="utf-8-sig")
        update_progress(
            result_dir,
            "account_backtest_running",
            {
                "combo_index": int(i),
                "combo_total": int(total),
                "current_combo": combo,
            },
        )

    result_df = pd.DataFrame(results).sort_values(["final_multiple", "sharpe"], ascending=[False, False])
    result_df.to_csv(result_dir / "account_results.csv", index=False, encoding="utf-8-sig")

    if not result_df.empty:
        invalid_mask = (
            (result_df["max_drawdown"] < -1.0)
            | (result_df["final_multiple"] <= 0)
            | (((result_df["cagr"] > 0) & (result_df["final_multiple"] < 1)) | ((result_df["cagr"] < 0) & (result_df["final_multiple"] > 1)))
        )
        invalid_rows = result_df.loc[invalid_mask].copy()
    else:
        invalid_rows = pd.DataFrame()
    invalid_rows.to_csv(result_dir / "sanity_invalid_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "result_dir": str(result_dir),
        "signal_source": str(BUY_SIGNAL_DIR / "final_test_selected_rows.csv"),
        "sell_sample_source": str(SELL_MODEL_DIR / "sell_samples.csv"),
        "buy_strategy_count": len(BUY_STRATEGY_TAGS),
        "combo_count": len(combos),
        "stop_multiplier": STOP_MULTIPLIER,
        "max_hold_days": MAX_HOLD_DAYS,
        "best_account_row": result_df.iloc[0].to_dict() if not result_df.empty else {},
        "sanity_invalid_count": int(len(invalid_rows)),
    }
    write_json(result_dir / "summary.json", summary)
    update_progress(result_dir, "finished", {"combo_count": int(len(combos))})


if __name__ == "__main__":
    main()
