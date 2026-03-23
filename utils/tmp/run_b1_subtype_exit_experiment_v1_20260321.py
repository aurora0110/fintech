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
ACCOUNT_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_buy_sell_model_account_v2_20260320.py"
DEFAULT_BUY_SIGNAL_DIR = ROOT / "results" / "b1_full_factor_signal_v6_full_20260321_102049"
DEFAULT_SUBTYPE_SIGNAL_DIR = ROOT / "results" / "b1_cluster_subtype_signal_v1_full_20260321_171945"
DEFAULT_SELL_MODEL_DIR = ROOT / "results" / "b1_sell_habit_experiment_v2_20260320_233121"
DEFAULT_STRATEGY_TAG = "factor_discovery_factor_pool_low_cross_top5"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_RESULT_DIR = ROOT / "results" / f"b1_subtype_exit_experiment_v1_{RUN_TS}"

POOL_NAMES = [
    "pool_all",
    "pool_pullback",
    "pool_uptrend",
    "pool_low_cross",
    "pool_confirmed",
    "pool_trend_focus",
    "pool_strict",
    "pool_shrink",
    "pool_near_trend",
    "pool_near_long",
    "pool_no_risk",
    "pool_core_plus",
]

FIXED_TP_LEVELS = [0.20, 0.30, 0.50]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


acc_mod = load_module(ACCOUNT_SCRIPT, "b1_subtype_exit_acc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 主链最优买点 + 子类型分流卖法实验")
    parser.add_argument("--buy-signal-dir", type=Path, default=DEFAULT_BUY_SIGNAL_DIR)
    parser.add_argument("--subtype-signal-dir", type=Path, default=DEFAULT_SUBTYPE_SIGNAL_DIR)
    parser.add_argument("--sell-model-dir", type=Path, default=DEFAULT_SELL_MODEL_DIR)
    parser.add_argument("--target-strategy-tag", type=str, default=DEFAULT_STRATEGY_TAG)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, extra: dict | None = None) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    if extra:
        payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def parse_strategy_tag(strategy_tag: str, available_pools: List[str]) -> tuple[str, str, str, int]:
    stem, topn_text = strategy_tag.rsplit("_top", 1)
    topn = int(topn_text)
    pool_match = None
    for pool_name in sorted(available_pools, key=len, reverse=True):
        token = "_" + pool_name
        if stem.endswith(token):
            pool_match = pool_name
            prefix = stem[: -len(token)]
            break
    if pool_match is None:
        raise ValueError(f"无法从 strategy_tag 解析 pool: {strategy_tag}")
    family, variant = prefix.split("_", 1)
    return family, variant, pool_match, topn


def resolve_score_spec(family: str, variant: str) -> tuple[str, bool]:
    if family == "baseline" and variant == "lowest_J":
        return "J", True
    if family in {"similarity", "contrast"}:
        return f"sim_{variant}", False
    mapping = {
        ("habit", "habit_profile"): "habit_profile_score",
        ("habit", "habit_rule"): "habit_rule_score",
        ("semantic", "semantic_profile"): "semantic_profile_score",
        ("semantic", "semantic_rule"): "semantic_rule_score",
        ("cluster", "centroid"): "cluster_centroid_score",
        ("cluster", "knn"): "cluster_knn_score",
        ("cluster", "gmm"): "cluster_gmm_score",
        ("factor", "discovery_factor"): "discovery_factor_score",
        ("factor", "discovery_rank"): "discovery_rank_score",
        ("fusion", "similarity_cluster"): "similarity_cluster_fusion_score",
        ("fusion", "semantic_discovery"): "semantic_discovery_fusion_score",
        ("fusion", "full"): "full_fusion_score",
    }
    if family == "ml":
        return f"{variant}_score", False
    score_col = mapping.get((family, variant))
    if score_col is None:
        raise ValueError(f"无法解析 strategy_tag 的打分列: {family=} {variant=}")
    return score_col, False


def build_selected_rows(signal_dir: Path, strategy_tag: str) -> pd.DataFrame:
    cand = pd.read_csv(signal_dir / "candidate_enriched.csv")
    cand["signal_date"] = pd.to_datetime(cand["signal_date"])
    cand["entry_date"] = pd.to_datetime(cand["entry_date"])
    available_pools = [c for c in cand.columns if c.startswith("pool_")]
    family, variant, pool_name, topn = parse_strategy_tag(strategy_tag, available_pools)
    score_col, ascending = resolve_score_spec(family, variant)
    if score_col not in cand.columns:
        raise ValueError(f"{strategy_tag} 对应打分列不存在: {score_col}")

    rows: List[pd.DataFrame] = []
    for split_name in ["validation", "final_test"]:
        part = cand[(cand["split"] == split_name) & (cand[pool_name].fillna(False))].copy()
        if part.empty:
            continue
        selected = (
            part.sort_values(["signal_date", score_col], ascending=[True, ascending])
            .groupby("signal_date")
            .head(topn)
            .copy()
        )
        selected["strategy_tag"] = strategy_tag
        rows.append(selected)
    if not rows:
        raise ValueError(f"{strategy_tag} 在 validation/final_test 没有选中任何信号")
    return pd.concat(rows, ignore_index=True)


def attach_subtype(selected_df: pd.DataFrame, subtype_signal_dir: Path) -> pd.DataFrame:
    subtype_df = pd.read_csv(subtype_signal_dir / "subtype_candidate_enriched.csv", usecols=["code", "signal_date", "subtype_cluster"])
    subtype_df["signal_date"] = pd.to_datetime(subtype_df["signal_date"])
    out = selected_df.merge(subtype_df, on=["code", "signal_date"], how="left")
    out["subtype_cluster"] = out["subtype_cluster"].fillna(-1).astype(int)
    return out


def ensure_validation_final_splits(selected_df: pd.DataFrame) -> pd.DataFrame:
    out = selected_df.copy()
    validation_sig = out[out["split"] == "validation"]
    final_sig = out[out["split"] == "final_test"]
    if not validation_sig.empty and not final_sig.empty:
        return out

    unique_dates = sorted(pd.to_datetime(out["signal_date"]).drop_duplicates().tolist())
    if len(unique_dates) >= 2:
        cut = max(1, len(unique_dates) // 2)
        validation_dates = set(unique_dates[:cut])
        final_dates = set(unique_dates[cut:])
        out["split"] = np.where(
            pd.to_datetime(out["signal_date"]).isin(validation_dates),
            "validation",
            np.where(pd.to_datetime(out["signal_date"]).isin(final_dates), "final_test", out["split"]),
        )
        return out

    # 小样本极端情况下只有 1 个交易日，允许复制一份作为 smoke 链路兜底，只用于打通链路，不用于正式结论。
    if len(out) == 1:
        row = out.iloc[[0]].copy()
        out.loc[out.index[0], "split"] = "validation"
        dup = row.copy()
        dup["split"] = "final_test"
        return pd.concat([out, dup], ignore_index=True)

    out["split"] = np.where(np.arange(len(out)) % 2 == 0, "validation", "final_test")
    return out


def build_combo_plan(threshold_df: pd.DataFrame) -> List[dict]:
    combos: List[dict] = []
    for tp in FIXED_TP_LEVELS:
        combos.append(
            {
                "strategy_tag": DEFAULT_STRATEGY_TAG,
                "exit_mode": "fixed_tp",
                "take_profit_pct": float(tp),
                "score_col": "",
                "threshold": np.nan,
            }
        )
    for _, row in threshold_df.iterrows():
        combos.append(
            {
                "strategy_tag": DEFAULT_STRATEGY_TAG,
                "exit_mode": "model_only",
                "take_profit_pct": np.nan,
                "score_col": str(row["score_col"]),
                "threshold": float(row["threshold"]),
                "threshold_quantile": float(row["quantile"]),
            }
        )
        combos.append(
            {
                "strategy_tag": DEFAULT_STRATEGY_TAG,
                "exit_mode": "model_plus_tp",
                "take_profit_pct": 0.20,
                "score_col": str(row["score_col"]),
                "threshold": float(row["threshold"]),
                "threshold_quantile": float(row["quantile"]),
            }
        )
    return combos


def rank_account_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["final_multiple", "max_drawdown", "trade_count", "win_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def choose_best_combo_for_subset(
    sig: pd.DataFrame,
    combos: List[dict],
    all_dates: List[pd.Timestamp],
    stock_cache: Dict[str, pd.DataFrame],
    fitted_models: Dict[str, Any],
    model_feature_cols: Dict[str, List[str]],
) -> tuple[dict, pd.DataFrame]:
    rows: List[dict] = []
    for combo in combos:
        equity_df, trade_df, summary = acc_mod.run_account_backtest(
            all_dates, stock_cache, sig, combo, fitted_models, model_feature_cols
        )
        rows.append({**combo, **summary})
    result_df = rank_account_rows(pd.DataFrame(rows))
    return result_df.iloc[0].to_dict(), result_df


def run_subtype_policy_backtest(
    all_dates: List[pd.Timestamp],
    stock_cache: Dict[str, pd.DataFrame],
    signal_df: pd.DataFrame,
    subtype_policy: Dict[int, dict],
    fallback_combo: dict,
    fitted_models: Dict[str, Any],
    model_feature_cols: Dict[str, List[str]],
    strategy_tag: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    pending_buy: Dict[pd.Timestamp, List[dict]] = {}
    for _, row in signal_df.iterrows():
        pending_buy.setdefault(pd.Timestamp(row["entry_date"]), []).append(row.to_dict())

    cash = acc_mod.INITIAL_CAPITAL
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
            stock_idx = acc_mod.get_stock_loc(feat, current_date)
            if stock_idx < 0:
                remaining_positions.append(pos)
                continue
            exit_price = float(feat.iloc[stock_idx]["open"])
            if not np.isfinite(exit_price) or exit_price <= 0:
                remaining_positions.append(pos)
                continue
            proceeds = pos["shares"] * exit_price * (1 - acc_mod.FEE_RATE - acc_mod.SLIPPAGE)
            cash += proceeds
            pnl = exit_price / pos["entry_price"] - 1.0
            hold_days = current_idx_global - pos["entry_global_idx"]
            reason = pos.get("scheduled_exit_reason", "unknown")
            if reason == "stop_loss":
                stop_loss_exit_count += 1
                consecutive_stop_losses += 1
                max_consecutive_stop_losses = max(max_consecutive_stop_losses, consecutive_stop_losses)
                if consecutive_stop_losses >= acc_mod.PAUSE_AFTER_STOPS:
                    pause_days_left = acc_mod.PAUSE_DAYS
                    pause_event_count += 1
                    consecutive_stop_losses = 0
            else:
                consecutive_stop_losses = 0
            trade_rows.append(
                {
                    "strategy_tag": strategy_tag,
                    "subtype_cluster": pos["subtype_cluster"],
                    "exit_mode": pos["exit_mode"],
                    "score_col": pos["score_col"],
                    "threshold": pos["threshold"],
                    "take_profit_pct": pos["take_profit_pct"],
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
            stock_idx = acc_mod.get_stock_loc(feat, current_date)
            if stock_idx < 0:
                continue
            row = feat.iloc[stock_idx]
            high_p = float(row["high"]) if pd.notna(row["high"]) else np.nan
            low_p = float(row["low"]) if pd.notna(row["low"]) else np.nan
            hold_days = current_idx_global - pos["entry_global_idx"]

            stop_hit = np.isfinite(low_p) and low_p <= pos["stop_price"]
            tp_hit = False
            if pos["exit_mode"] in {"fixed_tp", "model_plus_tp"} and np.isfinite(pos["take_profit_pct"]):
                tp_price = pos["entry_price"] * (1.0 + pos["take_profit_pct"])
                tp_hit = np.isfinite(high_p) and high_p >= tp_price

            model_hit = False
            model_score = np.nan
            if pos["exit_mode"] in {"model_only", "model_plus_tp"}:
                feat_row = acc_mod.build_daily_sell_features(feat, pos["entry_stock_idx"], stock_idx)
                model_score = acc_mod.compute_sell_score_row(
                    pos["score_col"], feat_row, fitted_models, model_feature_cols
                )
                if np.isfinite(model_score) and np.isfinite(pos["threshold"]):
                    model_hit = model_score >= pos["threshold"]

            if current_idx_global + 1 >= total_days:
                continue
            next_date = all_dates[current_idx_global + 1]
            if stop_hit:
                pos["scheduled_exit_date"] = next_date
                pos["scheduled_exit_reason"] = "stop_loss"
            elif model_hit:
                pos["scheduled_exit_date"] = next_date
                pos["scheduled_exit_reason"] = f"model_{pos['score_col']}"
                pos["scheduled_exit_score"] = model_score
            elif tp_hit:
                pos["scheduled_exit_date"] = next_date
                pos["scheduled_exit_reason"] = f"tp_{int(pos['take_profit_pct'] * 100)}"
            elif hold_days >= acc_mod.MAX_HOLD_DAYS:
                pos["scheduled_exit_date"] = next_date
                pos["scheduled_exit_reason"] = "time_exit_60"

        if pause_days_left == 0 and current_date in pending_buy:
            already_holding = {p["stock"] for p in positions}
            candidates = sorted(pending_buy[current_date], key=lambda x: (x["signal_date"], x["code"]))
            position_value_now = 0.0
            for pos in positions:
                feat = stock_cache[pos["stock"]]
                stock_idx = acc_mod.get_stock_loc(feat, current_date)
                if stock_idx >= 0:
                    mark_price = float(feat.iloc[stock_idx]["close"])
                    if np.isfinite(mark_price) and mark_price > 0:
                        position_value_now += pos["shares"] * mark_price
            total_equity_now = cash + position_value_now
            day_cash_limit = total_equity_now * acc_mod.DAY_CASH_CAP
            day_used_cash = 0.0

            for item in candidates:
                if len(positions) >= acc_mod.MAX_POSITIONS or cash <= 0 or day_used_cash >= day_cash_limit:
                    break
                stock = str(item["code"])
                if stock in already_holding or stock not in stock_cache:
                    continue
                feat = stock_cache[stock]
                stock_idx = acc_mod.get_stock_loc(feat, current_date)
                if stock_idx < 0:
                    continue
                entry_price = float(feat.iloc[stock_idx]["open"])
                entry_low = float(feat.iloc[stock_idx]["low"])
                if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(entry_low) or entry_low <= 0:
                    continue
                position_cap_cash = total_equity_now * acc_mod.SINGLE_POS_CAP
                budget = min(cash, position_cap_cash, day_cash_limit - day_used_cash)
                shares = int(budget // (entry_price * (1 + acc_mod.FEE_RATE + acc_mod.SLIPPAGE)))
                if shares <= 0:
                    continue
                cost = shares * entry_price * (1 + acc_mod.FEE_RATE + acc_mod.SLIPPAGE)
                if cost > cash + 1e-9:
                    continue
                cash -= cost
                day_used_cash += cost
                subtype_cluster = int(item.get("subtype_cluster", -1))
                policy = subtype_policy.get(subtype_cluster, fallback_combo)
                positions.append(
                    {
                        "stock": stock,
                        "signal_date": pd.Timestamp(item["signal_date"]),
                        "entry_date": current_date,
                        "entry_price": entry_price,
                        "stop_price": entry_low * acc_mod.STOP_MULTIPLIER,
                        "shares": shares,
                        "entry_global_idx": current_idx_global,
                        "entry_stock_idx": stock_idx,
                        "scheduled_exit_date": None,
                        "scheduled_exit_reason": None,
                        "subtype_cluster": subtype_cluster,
                        "exit_mode": policy["exit_mode"],
                        "score_col": policy.get("score_col", ""),
                        "threshold": policy.get("threshold", np.nan),
                        "take_profit_pct": policy.get("take_profit_pct", np.nan),
                    }
                )
                already_holding.add(stock)

        mtm = cash
        for pos in positions:
            feat = stock_cache[pos["stock"]]
            stock_idx = acc_mod.get_stock_loc(feat, current_date)
            mark_px = float(feat.iloc[stock_idx]["close"]) if stock_idx >= 0 else np.nan
            if np.isfinite(mark_px) and mark_px > 0:
                mtm += pos["shares"] * mark_px
        equity_rows.append({"date": current_date, "equity": mtm, "cash": cash, "position_count": len(positions)})

    if positions and all_dates:
        terminal_date = all_dates[-1]
        terminal_global_idx = len(all_dates) - 1
        for pos in positions:
            feat = stock_cache[pos["stock"]]
            stock_idx = acc_mod.get_stock_loc(feat, terminal_date)
            if stock_idx < 0:
                continue
            exit_price = float(feat.iloc[stock_idx]["close"])
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue
            proceeds = pos["shares"] * exit_price * (1 - acc_mod.FEE_RATE - acc_mod.SLIPPAGE)
            cash += proceeds
            pnl = exit_price / pos["entry_price"] - 1.0
            hold_days = terminal_global_idx - pos["entry_global_idx"]
            trade_rows.append(
                {
                    "strategy_tag": strategy_tag,
                    "subtype_cluster": pos["subtype_cluster"],
                    "exit_mode": pos["exit_mode"],
                    "score_col": pos["score_col"],
                    "threshold": pos["threshold"],
                    "take_profit_pct": pos["take_profit_pct"],
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
    summary = acc_mod.summarize_account(
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

    selected_df = build_selected_rows(args.buy_signal_dir, args.target_strategy_tag)
    selected_df = attach_subtype(selected_df, args.subtype_signal_dir)
    selected_df = ensure_validation_final_splits(selected_df)
    selected_df = acc_mod.filter_replayable_signals(selected_df)
    selected_df = ensure_validation_final_splits(selected_df)
    selected_df.to_csv(result_dir / "selected_signals.csv", index=False, encoding="utf-8-sig")

    sell_sample_df = acc_mod.load_sell_samples(args.sell_model_dir)
    validation_sell = sell_sample_df[sell_sample_df["split"] == "validation"].copy()
    research_sell = sell_sample_df[sell_sample_df["split"] == "research"].copy()
    train_sell = sell_sample_df[sell_sample_df["split"].isin(["research", "validation"])].copy()

    validation_scores = acc_mod.fit_sell_models(research_sell, validation_sell)
    for k, s in validation_scores.items():
        validation_sell[k] = s
    threshold_df = acc_mod.choose_thresholds(validation_sell)
    threshold_df.to_csv(result_dir / "sell_thresholds.csv", index=False, encoding="utf-8-sig")
    fitted_models, model_feature_cols = acc_mod.train_final_models(train_sell)
    update_progress(
        result_dir,
        "sell_models_ready",
        {
            "threshold_count": int(len(threshold_df)),
            "fitted_models": list(fitted_models.keys()),
            "signal_count": int(len(selected_df)),
        },
    )

    combos = build_combo_plan(threshold_df)
    needed_codes = sorted(selected_df["code"].astype(str).unique().tolist())
    stock_cache = acc_mod.load_stock_feature_cache(needed_codes)
    all_dates = sorted(set(pd.concat([pd.to_datetime(df["date"]) for df in stock_cache.values()]).tolist()))

    validation_sig = selected_df[selected_df["split"] == "validation"].copy()
    final_sig = selected_df[selected_df["split"] == "final_test"].copy()
    if validation_sig.empty or final_sig.empty:
        raise ValueError("validation 或 final_test 选中信号为空，无法做子类型卖法实验")

    validation_rows: List[dict] = []
    validation_combo_df_rows: List[pd.DataFrame] = []
    for subtype_cluster in sorted(validation_sig["subtype_cluster"].dropna().astype(int).unique().tolist()):
        sig_sub = validation_sig[validation_sig["subtype_cluster"] == subtype_cluster].copy()
        if sig_sub.empty:
            continue
        best_combo, detail_df = choose_best_combo_for_subset(
            sig_sub, combos, all_dates, stock_cache, fitted_models, model_feature_cols
        )
        best_combo["subtype_cluster"] = subtype_cluster
        validation_rows.append(best_combo)
        detail_df["subtype_cluster"] = subtype_cluster
        validation_combo_df_rows.append(detail_df)

    validation_policy_df = rank_account_rows(pd.DataFrame(validation_rows))
    validation_policy_df.to_csv(result_dir / "validation_subtype_best.csv", index=False, encoding="utf-8-sig")
    if validation_combo_df_rows:
        pd.concat(validation_combo_df_rows, ignore_index=True).to_csv(
            result_dir / "validation_subtype_combo_detail.csv", index=False, encoding="utf-8-sig"
        )

    global_best_combo, global_detail_df = choose_best_combo_for_subset(
        validation_sig, combos, all_dates, stock_cache, fitted_models, model_feature_cols
    )
    global_detail_df.to_csv(result_dir / "validation_global_combo_detail.csv", index=False, encoding="utf-8-sig")

    subtype_fixed_policy: Dict[int, dict] = {}
    subtype_full_policy: Dict[int, dict] = {}
    for subtype_cluster, g in validation_policy_df.groupby("subtype_cluster"):
        subtype_full_policy[int(subtype_cluster)] = g.iloc[0].to_dict()
        fixed_g = g[g["exit_mode"] == "fixed_tp"].copy()
        if fixed_g.empty:
            subtype_fixed_policy[int(subtype_cluster)] = global_best_combo
        else:
            subtype_fixed_policy[int(subtype_cluster)] = rank_account_rows(fixed_g).iloc[0].to_dict()

    write_json(
        result_dir / "subtype_policy.json",
        {
            "global_best_combo": global_best_combo,
            "subtype_fixed_policy": subtype_fixed_policy,
            "subtype_full_policy": subtype_full_policy,
        },
    )
    update_progress(result_dir, "validation_policy_ready", {"subtype_count": int(len(subtype_full_policy))})

    result_rows: List[dict] = []

    for combo_name, combo in [
        ("global_best_combo", global_best_combo),
        ("global_fixed_tp30", {"strategy_tag": args.target_strategy_tag, "exit_mode": "fixed_tp", "take_profit_pct": 0.30, "score_col": "", "threshold": np.nan}),
    ]:
        equity_df, trade_df, summary = acc_mod.run_account_backtest(
            all_dates, stock_cache, final_sig, combo, fitted_models, model_feature_cols
        )
        stem = combo_name
        equity_df.to_csv(result_dir / f"equity_{stem}.csv", index=False, encoding="utf-8-sig")
        trade_df.to_csv(result_dir / f"trades_{stem}.csv", index=False, encoding="utf-8-sig")
        result_rows.append({"policy_name": combo_name, **combo, **summary})

    for policy_name, policy_map in [
        ("subtype_fixed_policy", subtype_fixed_policy),
        ("subtype_full_policy", subtype_full_policy),
    ]:
        equity_df, trade_df, summary = run_subtype_policy_backtest(
            all_dates,
            stock_cache,
            final_sig,
            policy_map,
            global_best_combo,
            fitted_models,
            model_feature_cols,
            strategy_tag=args.target_strategy_tag,
        )
        equity_df.to_csv(result_dir / f"equity_{policy_name}.csv", index=False, encoding="utf-8-sig")
        trade_df.to_csv(result_dir / f"trades_{policy_name}.csv", index=False, encoding="utf-8-sig")
        result_rows.append({"policy_name": policy_name, **summary})

    result_df = rank_account_rows(pd.DataFrame(result_rows))
    result_df.to_csv(result_dir / "account_results.csv", index=False, encoding="utf-8-sig")
    best_row = result_df.iloc[0].to_dict() if not result_df.empty else {}
    write_json(
        result_dir / "summary.json",
        {
            "buy_signal_dir": str(args.buy_signal_dir),
            "subtype_signal_dir": str(args.subtype_signal_dir),
            "sell_model_dir": str(args.sell_model_dir),
            "target_strategy_tag": args.target_strategy_tag,
            "selected_signal_count": int(len(selected_df)),
            "validation_signal_count": int(len(validation_sig)),
            "final_signal_count": int(len(final_sig)),
            "subtype_count": int(selected_df["subtype_cluster"].nunique()),
            "policy_result_count": int(len(result_df)),
            "best_account_row": best_row,
        },
    )
    update_progress(result_dir, "finished", {"policy_result_count": int(len(result_df))})


if __name__ == "__main__":
    main()
