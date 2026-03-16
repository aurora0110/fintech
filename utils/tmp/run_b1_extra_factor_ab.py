from __future__ import annotations

import json
import os
from collections import deque

import numpy as np
import pandas as pd

from utils.backtest import backtest_b1_strategy as b1bt


DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
RESULT_DIR = "/Users/lidongyang/Desktop/Qstrategy/results/b1_extra_factor_ab_20260313"
os.makedirs(RESULT_DIR, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")


def calc_key_k_close(df: pd.DataFrame, window: int = 60) -> np.ndarray:
    bullish_volume = np.where((df["CLOSE"] > df["OPEN"]).to_numpy(dtype=bool), df["VOLUME"].to_numpy(dtype=float), -np.inf)
    closes = df["CLOSE"].to_numpy(dtype=float)
    key_close = np.full(len(df), np.nan, dtype=float)

    dq: deque[int] = deque()
    for idx in range(len(df)):
        left = idx - window + 1
        while dq and dq[0] < left:
            dq.popleft()
        current_volume = bullish_volume[idx]
        if np.isfinite(current_volume):
            while dq and bullish_volume[dq[-1]] <= current_volume:
                dq.pop()
            dq.append(idx)
        if dq:
            key_idx = dq[0]
            key_close[idx] = closes[key_idx]
    return key_close


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["J_Q10_20"] = x["J"].rolling(20).quantile(0.10)

    prev_volume = x["VOLUME"].shift(1).replace(0, np.nan)
    double_bull = (x["CLOSE"] > x["OPEN"]) & ((x["VOLUME"] / prev_volume) >= 2.0)
    x["double_bull"] = double_bull.fillna(False)
    x["double_bull_exist_60"] = x["double_bull"].rolling(60, min_periods=1).max().fillna(0.0).astype(bool)

    last_close = np.full(len(x), np.nan, dtype=float)
    last_high = np.full(len(x), np.nan, dtype=float)
    flag = x["double_bull"].to_numpy(dtype=bool)
    closes = x["CLOSE"].to_numpy(dtype=float)
    highs = x["HIGH"].to_numpy(dtype=float)
    last_idx = -1
    for idx in range(len(x)):
        left = idx - 60
        if last_idx < left:
            last_idx = -1
        if idx > 0 and flag[idx - 1]:
            last_idx = idx - 1
        if last_idx >= 0:
            last_close[idx] = closes[last_idx]
            last_high[idx] = highs[last_idx]

    x["last_double_bull_close"] = last_close
    x["last_double_bull_high"] = last_high
    x["above_double_bull_close"] = x["CLOSE"] >= x["last_double_bull_close"]
    x["above_double_bull_high"] = x["CLOSE"] >= x["last_double_bull_high"]

    x["key_k_close"] = calc_key_k_close(x, window=60)
    x["key_k_support"] = x["key_k_close"].notna() & (x["CLOSE"] >= x["key_k_close"])
    return x


def build_variant_signals_for_stock(df: pd.DataFrame, variant: dict) -> pd.DataFrame:
    x = add_extra_features(df)
    cond = (
        x["J"].notna()
        & x["J_Q10_20"].notna()
        & (x["J"] <= x["J_Q10_20"])
        & x["trend_ok"]
        & x["bullish_filter"]
    )

    if variant.get("require_double_bull_exist", False):
        cond &= x["double_bull_exist_60"]
    if variant.get("require_above_double_bull_close", False):
        cond &= x["above_double_bull_close"].fillna(False)
    if variant.get("require_above_double_bull_high", False):
        cond &= x["above_double_bull_high"].fillna(False)
    if variant.get("require_key_k_support", False):
        cond &= x["key_k_support"].fillna(False)

    out = x.loc[cond, ["OPEN", "HIGH", "LOW", "CLOSE", "J", "J_Q10_20", "ATR14"]].copy()
    out["score"] = (-out["J"]).fillna(0.0)
    return out


def build_daily_signals(stock_data: dict[str, pd.DataFrame], variant: dict) -> dict:
    daily_scores: dict[pd.Timestamp, list[dict]] = {}
    items = list(stock_data.items())
    total = len(items)

    print(f"构建信号: {variant['variant_name']}")
    for i, (stock_code, df) in enumerate(items, 1):
        sig_df = build_variant_signals_for_stock(df, variant)
        if not sig_df.empty:
            for dt, row in sig_df.iterrows():
                if EXCLUDE_START <= dt <= EXCLUDE_END:
                    continue
                daily_scores.setdefault(dt, []).append(
                    {
                        "stock": stock_code,
                        "score": float(row["score"]),
                        "J": float(row["J"]) if pd.notna(row["J"]) else np.nan,
                        "J_Q10_20": float(row["J_Q10_20"]) if pd.notna(row["J_Q10_20"]) else np.nan,
                        "ATR14": float(row["ATR14"]) if pd.notna(row["ATR14"]) else np.nan,
                    }
                )
        if i % 500 == 0 or i == total:
            print(f"信号构建进度: {i}/{total}")
    return daily_scores


def generate_pending_buy_signals_filtered(daily_scores: dict, all_dates_full: list[pd.Timestamp]) -> dict:
    date_to_idx = {d: i for i, d in enumerate(all_dates_full)}
    pending_buy: dict[pd.Timestamp, list[dict]] = {}
    for signal_date, items in daily_scores.items():
        i = date_to_idx.get(signal_date)
        if i is None or i + 1 >= len(all_dates_full):
            continue
        next_date = all_dates_full[i + 1]
        if EXCLUDE_START <= next_date <= EXCLUDE_END:
            continue
        if next_date - signal_date > pd.Timedelta(days=15):
            continue
        pending_buy.setdefault(next_date, []).extend(items)
    return pending_buy


def make_variant_name(variant: dict) -> str:
    parts = [variant["variant_name"]]
    if variant.get("require_double_bull_exist"):
        parts.append("倍量阳柱存在")
    if variant.get("require_above_double_bull_close"):
        parts.append("站上倍量柱收盘")
    if variant.get("require_above_double_bull_high"):
        parts.append("站上倍量柱高点")
    if variant.get("require_key_k_support"):
        parts.append("关键K支撑")
    return "+".join(parts)


def main():
    stock_data, all_dates_full = b1bt.load_all_data(DATA_DIR)
    all_dates = [d for d in all_dates_full if not (EXCLUDE_START <= d <= EXCLUDE_END)]
    regime_df = b1bt.build_market_regime(stock_data, all_dates)

    params = {
        "max_positions": 10,
        "max_new_buys_per_day": 1,
        "max_hold_days": 2,
        "day_cash_cap": 0.30,
        "single_pos_cap": 0.10,
        "take_profit_mode": "fixed_3",
        "stop_mode": "entry_low_095",
        "pause_rule": "loss_streak_3_pause_5",
        "regime_mode": "none",
        "score_bucket": "all",
    }

    variants = [
        {"variant_name": "B1_J20分位10%", "require_double_bull_exist": False, "require_above_double_bull_close": False, "require_above_double_bull_high": False, "require_key_k_support": False},
        {"variant_name": "B1_J20分位10%", "require_double_bull_exist": True, "require_above_double_bull_close": False, "require_above_double_bull_high": False, "require_key_k_support": False},
        {"variant_name": "B1_J20分位10%", "require_double_bull_exist": True, "require_above_double_bull_close": True, "require_above_double_bull_high": False, "require_key_k_support": False},
        {"variant_name": "B1_J20分位10%", "require_double_bull_exist": False, "require_above_double_bull_close": False, "require_above_double_bull_high": False, "require_key_k_support": True},
        {"variant_name": "B1_J20分位10%", "require_double_bull_exist": True, "require_above_double_bull_close": True, "require_above_double_bull_high": False, "require_key_k_support": True},
        {"variant_name": "B1_J20分位10%", "require_double_bull_exist": True, "require_above_double_bull_close": False, "require_above_double_bull_high": True, "require_key_k_support": True},
    ]

    results = []
    for variant in variants:
        variant_label = make_variant_name(variant)
        daily_scores = build_daily_signals(stock_data, variant)
        pending_buy = generate_pending_buy_signals_filtered(daily_scores, all_dates_full)
        res = b1bt.run_backtest(
            stock_data=stock_data,
            all_dates=all_dates,
            pending_buy_signals=pending_buy,
            regime_df=regime_df,
            params=params,
            exp_name=variant_label,
        )
        res["信号天数"] = len(daily_scores)
        res["变体"] = variant_label
        results.append(res)

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(RESULT_DIR, "variant_results.csv"), index=False, encoding="utf-8-sig")

    best_by_return = result_df.sort_values(
        by=["年化收益率", "成功率", "平均持有期间收益率"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    best_by_drawdown = result_df.sort_values(
        by=["最大回撤", "成功率", "平均持有期间收益率"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()

    summary = {
        "data_dir": DATA_DIR,
        "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "params": params,
        "variant_count": len(variants),
        "best_by_return": best_by_return,
        "best_by_drawdown": best_by_drawdown,
    }
    with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n完成。结果目录:", RESULT_DIR)
    print(result_df[["变体", "总交易次数", "成功率", "平均持有期间收益率", "最大回撤", "年化收益率", "信号天数"]].to_string(index=False))


if __name__ == "__main__":
    main()
