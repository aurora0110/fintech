from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.backtest.run_b2_startup_experiment import (
    DATA_DIR,
    EXCLUDE_END,
    EXCLUDE_START,
    INITIAL_CAPITAL,
    MAX_FILES,
    MAX_POSITIONS,
    MAX_SINGLE_WEIGHT,
    TRADING_DAYS_PER_YEAR,
    build_feature_df,
    compute_equity_metrics,
    load_one_csv,
    max_consecutive_failures,
)


OUTPUT_DIR = os.environ.get("B2_EXIT_OUTPUT_DIR", "/Users/lidongyang/Desktop/Qstrategy/results/b2_exit_comparison")


def load_feature_map(data_dir: str) -> Dict[str, pd.DataFrame]:
    feature_map: Dict[str, pd.DataFrame] = {}
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".txt"))]
    if MAX_FILES > 0:
        files = files[:MAX_FILES]
    total = len(files)
    for idx, file_name in enumerate(files, 1):
        df = load_one_csv(os.path.join(data_dir, file_name))
        if df is None:
            continue
        code = str(df["code"].iloc[0])
        feature_map[code] = build_feature_df(df)
        if idx % 500 == 0 or idx == total:
            print(f"特征进度: {idx}/{total}")
    return feature_map


def base_mask(df: pd.DataFrame) -> pd.Series:
    return df["基础信号"] & df["双线附近启动"]


def simulate_trade(df: pd.DataFrame, signal_idx: int, exit_rule: str) -> Optional[dict]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    stop_price = float(df.at[signal_idx, "信号最低点"])
    if not np.isfinite(stop_price) or stop_price <= 0:
        return None

    max_check_idx = min(len(df) - 2, signal_idx + 12)
    prev_day_was_yang = df["close"] > df["open"]

    exit_idx = None
    exit_reason = None
    first_eligible = entry_idx + 1
    for i in range(first_eligible, max_check_idx + 1):
        # 公共止损：跌破信号K最低点，次日开盘卖
        if float(df.at[i, "low"]) < stop_price:
            exit_idx = i + 1
            exit_reason = "跌破信号低点止损"
            break

        if exit_rule == "放量阴线卖出":
            if i - 1 >= 0 and bool(prev_day_was_yang.iloc[i - 1]):
                today_is_yin = float(df.at[i, "close"]) < float(df.at[i, "open"])
                volume_ok = float(df.at[i, "volume"]) > float(df.at[i - 1, "volume"]) * 1.3
                if today_is_yin and volume_ok:
                    exit_idx = i + 1
                    exit_reason = "放量阴线卖出"
                    break

        elif exit_rule == "跌破趋势线全卖":
            if float(df.at[i, "close"]) < float(df.at[i, "趋势线"]):
                exit_idx = i + 1
                exit_reason = "跌破趋势线退出"
                break

        elif exit_rule == "两天收盘跌破前低":
            if i - 1 >= entry_idx:
                cond1 = float(df.at[i - 1, "close"]) < float(df.at[i - 2, "low"]) if i - 2 >= 0 else False
                cond2 = float(df.at[i, "close"]) < float(df.at[i - 1, "low"])
                if cond1 and cond2:
                    exit_idx = i + 1
                    exit_reason = "连续两天跌破前低"
                    break

    if exit_idx is None:
        exit_idx = min(signal_idx + 4, len(df) - 1)
        exit_reason = "固定持有3天到期"

    if exit_idx >= len(df):
        return None
    exit_price = float(df.at[exit_idx, "open"])
    if not np.isfinite(exit_price) or exit_price <= 0:
        return None

    ret = exit_price / entry_price - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "first_eligible_exit_date": df.at[first_eligible, "date"] if first_eligible < len(df) else pd.NaT,
        "exit_date": df.at[exit_idx, "date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "ret": ret,
        "holding_days": int(exit_idx - entry_idx),
        "success": ret > 0,
        "stop_price": stop_price,
        "sort_score": float(df.at[signal_idx, "sort_score"]),
        "exit_reason": exit_reason,
    }


def build_portfolio_curve(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["signal_date", "portfolio_ret", "equity"])
    rows = []
    equity = INITIAL_CAPITAL
    trade_df = trade_df.copy()
    trade_df["signal_date"] = pd.to_datetime(trade_df["signal_date"])
    for signal_date, group in trade_df.groupby("signal_date", sort=True):
        g = group.copy().sort_values(["sort_score", "code"], ascending=[False, True]).head(MAX_POSITIONS)
        score = g["sort_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        if score.sum() <= 0:
            weights = np.repeat(1 / len(g), len(g))
        else:
            weights = (score / score.sum()).clip(upper=MAX_SINGLE_WEIGHT).to_numpy()
            weights = weights / weights.sum()
        basket_ret = float(np.sum(g["ret"].to_numpy() * weights))
        equity *= 1.0 + basket_ret
        rows.append({"signal_date": signal_date, "portfolio_ret": basket_ret, "equity": equity})
    return pd.DataFrame(rows)


def summarize(rule: str, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    row = {
        "exit_rule": rule,
        "sample_count": int(len(trade_df)) if not trade_df.empty else 0,
        "avg_trade_return": float(trade_df["ret"].mean()) if not trade_df.empty else np.nan,
        "success_rate": float(trade_df["success"].mean()) if not trade_df.empty else np.nan,
        "max_consecutive_failures": int(max_consecutive_failures(trade_df["success"].tolist())) if not trade_df.empty else np.nan,
    }
    row.update(compute_equity_metrics(portfolio_df))
    return row


def validate(result_df: pd.DataFrame) -> None:
    finite_dd = result_df["max_drawdown"].dropna()
    if not finite_dd.empty and ((finite_dd < -1.0) | (finite_dd > 0.0)).any():
        raise ValueError("存在非法最大回撤")
    valid = result_df.dropna(subset=["annual_return", "final_equity"])
    inconsistent = valid[
        ((valid["final_equity"] > INITIAL_CAPITAL) & (valid["annual_return"] <= 0))
        | ((valid["final_equity"] < INITIAL_CAPITAL) & (valid["annual_return"] >= 0))
    ]
    if not inconsistent.empty:
        raise ValueError("年化与净值方向不一致")


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    feature_map = load_feature_map(DATA_DIR)
    rules = ["放量阴线卖出", "跌破趋势线全卖", "两天收盘跌破前低"]
    rows = []

    for idx, rule in enumerate(rules, 1):
        trades: List[dict] = []
        for code, df in feature_map.items():
            mask = base_mask(df)
            idxs = np.flatnonzero(mask.fillna(False).to_numpy())
            for signal_idx in idxs:
                trade = simulate_trade(df, int(signal_idx), rule)
                if trade is None:
                    continue
                trade["code"] = code
                trades.append(trade)
        trade_df = pd.DataFrame(trades)
        portfolio_df = build_portfolio_curve(trade_df)
        rows.append(summarize(rule, trade_df, portfolio_df))
        print(f"规则进度: {idx}/{len(rules)}")

    result_df = pd.DataFrame(rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    validate(result_df)
    best = result_df.assign(drawdown_abs=result_df["max_drawdown"].abs()).sort_values(
        ["annual_return", "drawdown_abs", "avg_trade_return", "success_rate"],
        ascending=[False, True, False, False],
    ).iloc[0]
    result_df.to_csv(os.path.join(OUTPUT_DIR, "comparison.csv"), index=False, encoding="utf-8-sig")
    summary = {
        "data_dir": DATA_DIR,
        "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "entry": "T+1_open",
        "base_signal": [
            "双线启动",
            "当日涨幅>=4%",
            "上影线长度<=实体长度*0.3",
            "J<80 且当日J>昨日J，昨日J<前日J<前前日J",
            "趋势线>多空线",
            "当日量>昨日量且>5日均量",
        ],
        "stop_loss": "跌破信号K最低点后次日开盘卖",
        "exit_rules": rules,
        "best_rule": best.to_dict(),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
