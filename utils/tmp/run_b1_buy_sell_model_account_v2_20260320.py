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
FORWARD_DATA_DIR = ROOT / "data" / "forward_data"
BASE_BUY_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
SEMANTIC_SCRIPT = ROOT / "utils" / "tmp" / "b1_semantic_shared_20260320.py"
SELL_V2_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_sell_habit_experiment_v2_20260320.py"
BACKTEST_UTIL = ROOT / "utils" / "backtest" / "backtest_b1_strategy.py"

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_RESULT_DIR = ROOT / "results" / f"b1_buy_sell_model_account_v2_{RUN_TS}"

FIXED_TP_LEVELS = [0.20, 0.30, 0.50]
SELL_MODEL_COLS = [
    "rule_score_v2",
    "rf_score_v2",
    "et_score_v2",
    "lgb_score_v2",
    "xgb_score_v2",
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


def get_effective_stop_multiplier(feat_row: pd.Series) -> float:
    base = STOP_MULTIPLIER
    long_line = pd.to_numeric(pd.Series([feat_row.get("long_line")]), errors="coerce").iloc[0]
    close_px = pd.to_numeric(pd.Series([feat_row.get("close")]), errors="coerce").iloc[0]
    if np.isfinite(long_line) and long_line > 0 and np.isfinite(close_px) and close_px < long_line:
        stop_distance = 1.0 - base
        return 1.0 - stop_distance / 2.0
    return base


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base_buy = load_module(BASE_BUY_SCRIPT, "b1_acc_v2_base")
sem_mod = load_module(SEMANTIC_SCRIPT, "b1_acc_v2_sem")
sell_mod = load_module(SELL_V2_SCRIPT, "b1_acc_v2_sell")
bt_mod = load_module(BACKTEST_UTIL, "b1_acc_v2_bt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 语义化买点 + 语义化卖点模型账户层回测")
    parser.add_argument("--buy-signal-dir", type=Path, required=True)
    parser.add_argument("--sell-model-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--limit-combos", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, extra: dict | None = None) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    if extra:
        payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def norm_key_value(v: Any) -> Any:
    if pd.isna(v):
        return "__nan__"
    if isinstance(v, float):
        return round(float(v), 10)
    return str(v)


def combo_key(row: dict | pd.Series) -> tuple:
    return (
        norm_key_value(row.get("strategy_tag")),
        norm_key_value(row.get("exit_mode")),
        norm_key_value(row.get("take_profit_pct")),
        norm_key_value(row.get("score_col")),
        norm_key_value(row.get("threshold")),
    )


def dedupe_partial_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    tmp = df.copy()
    key_cols = ["strategy_tag", "exit_mode", "take_profit_pct", "score_col", "threshold"]
    for col in key_cols:
        if col not in tmp.columns:
            tmp[col] = np.nan
    tmp["_combo_key"] = [combo_key(r) for _, r in tmp.iterrows()]
    tmp = tmp.drop_duplicates(subset=["_combo_key"], keep="last").drop(columns=["_combo_key"])
    return tmp.reset_index(drop=True)


def get_stock_loc(df: pd.DataFrame, date: pd.Timestamp) -> int:
    matches = np.flatnonzero(pd.to_datetime(df["date"]).to_numpy() == np.datetime64(date))
    return int(matches[-1]) if len(matches) else -1


def load_selected_signals(buy_signal_dir: Path) -> pd.DataFrame:
    selected = pd.read_csv(buy_signal_dir / "final_test_selected_rows.csv")
    selected["signal_date"] = pd.to_datetime(selected["signal_date"])
    selected["entry_date"] = pd.to_datetime(selected["entry_date"])
    return selected


def load_sell_samples(sell_model_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(sell_model_dir / "sell_samples.csv")
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


def fit_sell_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, pd.Series]:
    return sell_mod.fit_and_score(train_df, test_df)


def load_stock_feature_cache(codes: List[str]) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    total = len(codes)
    for i, code in enumerate(sorted(set(codes)), 1):
        path = FORWARD_DATA_DIR / f"{code}.txt"
        df = base_buy.load_stock_data(str(path))
        if df is None or df.empty:
            continue
        feat = sem_mod.add_semantic_buy_features(df).copy()
        cache[code] = feat
        if i % 50 == 0 or i == total:
            print(f"语义特征缓存加载进度: {i}/{total}")
    return cache


def filter_replayable_signals(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        return selected_df.copy()
    available = selected_df["code"].astype(str).map(lambda c: (FORWARD_DATA_DIR / f"{c}.txt").exists())
    out = selected_df.loc[available].copy()
    return out


def build_daily_sell_features(feat: pd.DataFrame, buy_idx: int, cur_idx: int) -> dict:
    return sem_mod.build_daily_sell_semantic_features(feat, buy_idx, cur_idx)


def compute_sell_score_row(
    model_name: str,
    row: dict,
    fitted_models: Dict[str, Any],
    model_feature_cols: Dict[str, List[str]],
) -> float:
    if model_name == "rule_score_v2":
        return float(sem_mod.score_sell_rule_v2(pd.DataFrame([row])).iloc[0])
    model = fitted_models.get(model_name)
    if model is None:
        return np.nan
    force_cols = model_feature_cols.get(model_name)
    if not force_cols:
        return np.nan
    X = sell_mod.build_feature_matrix(pd.DataFrame([row]), force_cols=force_cols)
    return float(model.predict_proba(X)[:, 1][0])


def train_final_models(train_df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    models: Dict[str, Any] = {}
    feature_cols = sell_mod.available_feature_cols(train_df)
    X_train = sell_mod.build_feature_matrix(train_df, force_cols=feature_cols)
    y_train = train_df["label"].to_numpy(dtype=int)
    model_feature_cols: Dict[str, List[str]] = {}

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
            models["rf_score_v2"] = rf
            model_feature_cols["rf_score_v2"] = feature_cols
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
            models["et_score_v2"] = et
            model_feature_cols["et_score_v2"] = feature_cols
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
            models["lgb_score_v2"] = clf
            model_feature_cols["lgb_score_v2"] = feature_cols
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
            models["xgb_score_v2"] = clf
            model_feature_cols["xgb_score_v2"] = feature_cols
        except Exception:
            pass
    return models, model_feature_cols


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
    model_feature_cols: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    strategy_tag = combo["strategy_tag"]
    exit_mode = combo["exit_mode"]
    score_col = combo.get("score_col", "")
    threshold = combo.get("threshold", np.nan)
    tp_pct = combo.get("take_profit_pct", np.nan)

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
                model_score = compute_sell_score_row(score_col, feat_row, fitted_models, model_feature_cols)
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
                stop_multiplier = get_effective_stop_multiplier(feat.iloc[stock_idx])
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
                        "stop_price": entry_low * stop_multiplier,
                        "stop_multiplier": stop_multiplier,
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
            mark_px = float(feat.iloc[stock_idx]["close"]) if stock_idx >= 0 else np.nan
            if np.isfinite(mark_px) and mark_px > 0:
                mtm += pos["shares"] * mark_px
        equity_rows.append({"date": current_date, "equity": mtm, "cash": cash, "position_count": len(positions)})
        if day_i % PROGRESS_STEP == 0 or day_i == total_days:
            print(f"语义账户层回放进度[{strategy_tag}|{exit_mode}|{score_col or 'na'}]: {day_i}/{total_days}")

    # 样本末尾仍未退出的持仓，统一按最后一个交易日收盘价强制平仓，避免账户层统计遗漏未实现盈亏。
    if positions and all_dates:
        terminal_date = all_dates[-1]
        terminal_global_idx = len(all_dates) - 1
        for pos in positions:
            feat = stock_cache[pos["stock"]]
            stock_idx = get_stock_loc(feat, terminal_date)
            if stock_idx < 0:
                continue
            exit_price = float(feat.iloc[stock_idx]["close"])
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue
            proceeds = pos["shares"] * exit_price * (1 - FEE_RATE - SLIPPAGE)
            cash += proceeds
            pnl = exit_price / pos["entry_price"] - 1.0
            hold_days = terminal_global_idx - pos["entry_global_idx"]
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
                    "exit_date": terminal_date,
                    "exit_reason": "terminal_close",
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "hold_days": hold_days,
                    "return_pct": pnl * 100.0,
                }
            )
        positions = []

        if equity_rows:
            equity_rows[-1]["equity"] = cash
            equity_rows[-1]["cash"] = cash
            equity_rows[-1]["position_count"] = 0

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
    selected_df = load_selected_signals(args.buy_signal_dir)
    selected_df = filter_replayable_signals(selected_df)
    sell_sample_df = load_sell_samples(args.sell_model_dir)

    strategy_tags = sorted(selected_df["strategy_tag"].astype(str).unique().tolist())
    validation_sell = sell_sample_df[sell_sample_df["split"] == "validation"].copy()
    research_sell = sell_sample_df[sell_sample_df["split"] == "research"].copy()
    train_sell = sell_sample_df[sell_sample_df["split"].isin(["research", "validation"])].copy()

    validation_scores = fit_sell_models(research_sell, validation_sell)
    for k, s in validation_scores.items():
        validation_sell[k] = s
    threshold_df = choose_thresholds(validation_sell)
    threshold_df.to_csv(result_dir / "sell_thresholds.csv", index=False, encoding="utf-8-sig")

    fitted_models, model_feature_cols = train_final_models(train_sell)
    update_progress(
        result_dir,
        "sell_models_ready",
        {
            "thresholds": int(len(threshold_df)),
            "fitted_models": list(fitted_models.keys()),
            "model_feature_cols": {k: len(v) for k, v in model_feature_cols.items()},
        },
    )

    buy_meta_rows: List[dict] = []
    combos: List[dict] = []
    for strategy_tag in strategy_tags:
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

    results: List[dict] = []
    completed_keys = set()
    partial_path = result_dir / "account_results_partial.csv"
    if partial_path.exists():
        try:
            prev = pd.read_csv(partial_path)
            if not prev.empty:
                prev = dedupe_partial_results(prev)
                prev.to_csv(partial_path, index=False, encoding="utf-8-sig")
                results = prev.to_dict("records")
                completed_keys = {combo_key(r) for _, r in prev.iterrows()}
        except Exception:
            results = []
            completed_keys = set()

    needed_codes = sorted(selected_df["code"].astype(str).unique().tolist())
    update_progress(result_dir, "loading_stock_features", {"stock_count": int(len(needed_codes)), "combo_count": int(len(combos))})
    stock_cache = load_stock_feature_cache(needed_codes)
    if not stock_cache:
        pd.DataFrame(results).to_csv(result_dir / "account_results.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(results).to_csv(result_dir / "account_results_partial.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(result_dir / "sanity_invalid_rows.csv", index=False, encoding="utf-8-sig")
        summary = {
            "result_dir": str(result_dir),
            "buy_signal_dir": str(args.buy_signal_dir),
            "sell_model_dir": str(args.sell_model_dir),
            "buy_strategy_count": len(strategy_tags),
            "combo_count": len(combos),
            "stop_multiplier": STOP_MULTIPLIER,
            "max_hold_days": MAX_HOLD_DAYS,
            "best_account_row": {},
            "sanity_invalid_count": 0,
            "note": "筛选后无可在 forward_data 回放的股票",
        }
        write_json(result_dir / "summary.json", summary)
        update_progress(result_dir, "finished", {"combo_count": int(len(combos)), "note": "no_replayable_signals"})
        return
    all_dates = sorted(set(pd.concat([pd.to_datetime(df["date"]) for df in stock_cache.values()]).tolist()))

    total = len(combos)
    completed_count = len(completed_keys)
    failed_rows: List[dict] = []
    for combo in combos:
        if combo_key(combo) in completed_keys:
            continue
        strategy_tag = combo["strategy_tag"]
        sig = selected_df[selected_df["strategy_tag"] == strategy_tag].copy()
        try:
            equity_df, trade_df, summary = run_account_backtest(
                all_dates, stock_cache, sig, combo, fitted_models, model_feature_cols
            )
            stem = f"{strategy_tag}__{combo['exit_mode']}"
            if combo.get("score_col"):
                stem += f"__{combo['score_col']}"
            if np.isfinite(combo.get("take_profit_pct", np.nan)):
                stem += f"__tp{int(combo['take_profit_pct'] * 100)}"
            equity_df.to_csv(result_dir / f"equity_{stem}.csv", index=False, encoding="utf-8-sig")
            trade_df.to_csv(result_dir / f"trades_{stem}.csv", index=False, encoding="utf-8-sig")
            row = {**combo, **summary}
            results.append(row)
            completed_keys.add(combo_key(combo))
            completed_count += 1
            cleaned = dedupe_partial_results(pd.DataFrame(results))
            results = cleaned.to_dict("records")
            cleaned.to_csv(result_dir / "account_results_partial.csv", index=False, encoding="utf-8-sig")
            update_progress(
                result_dir,
                "account_backtest_running",
                {"combo_index": int(completed_count), "combo_total": int(total), "current_combo": combo},
            )
        except Exception as exc:
            failed_rows.append({**combo, "error": repr(exc)})
            pd.DataFrame(failed_rows).to_csv(result_dir / "failed_combos.csv", index=False, encoding="utf-8-sig")
            update_progress(
                result_dir,
                "account_backtest_running",
                {"combo_index": int(completed_count), "combo_total": int(total), "current_combo": combo, "last_error": repr(exc)},
            )
            continue

    result_df = pd.DataFrame(results).sort_values(["final_multiple", "sharpe"], ascending=[False, False])
    result_df.to_csv(result_dir / "account_results.csv", index=False, encoding="utf-8-sig")
    invalid_mask = (
        (result_df["max_drawdown"] < -1.0)
        | (result_df["final_multiple"] <= 0)
        | (((result_df["cagr"] > 0) & (result_df["final_multiple"] < 1)) | ((result_df["cagr"] < 0) & (result_df["final_multiple"] > 1)))
    )
    invalid_rows = result_df.loc[invalid_mask].copy() if not result_df.empty else pd.DataFrame()
    invalid_rows.to_csv(result_dir / "sanity_invalid_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "result_dir": str(result_dir),
        "buy_signal_dir": str(args.buy_signal_dir),
        "sell_model_dir": str(args.sell_model_dir),
        "buy_strategy_count": len(strategy_tags),
        "combo_count": len(combos),
        "stop_multiplier": STOP_MULTIPLIER,
        "max_hold_days": MAX_HOLD_DAYS,
        "best_account_row": result_df.iloc[0].to_dict() if not result_df.empty else {},
        "sanity_invalid_count": int(len(invalid_rows)),
        "failed_combo_count": int(len(failed_rows)),
    }
    write_json(result_dir / "summary.json", summary)
    update_progress(result_dir, "finished", {"combo_count": int(len(combos))})


if __name__ == "__main__":
    main()
