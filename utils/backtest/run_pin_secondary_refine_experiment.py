from __future__ import annotations

import json
import os
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import technical_indicators


DATA_DIR = os.environ.get("PIN_DATA_DIR", "/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
OUTPUT_DIR = os.environ.get("PIN_OUTPUT_DIR", "/Users/lidongyang/Desktop/Qstrategy/results/pin_secondary_refine_experiment")
MAX_FILES = int(os.environ.get("PIN_MAX_FILES", "0"))
MIN_BARS = 160
EPS = 1e-12
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
MAX_SINGLE_WEIGHT = 0.2
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

DATE_COL_CANDIDATES = ["date", "Date", "trade_date", "日期", "DATE"]
OPEN_COL_CANDIDATES = ["open", "Open", "开盘", "OPEN"]
HIGH_COL_CANDIDATES = ["high", "High", "最高", "HIGH"]
LOW_COL_CANDIDATES = ["low", "Low", "最低", "LOW"]
CLOSE_COL_CANDIDATES = ["close", "Close", "收盘", "CLOSE"]
VOL_COL_CANDIDATES = ["volume", "vol", "Volume", "成交量", "VOL"]
CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码", "CODE"]


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"缺少字段，候选字段={candidates}")
    return None


def read_csv_auto(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+|\t+", engine="python")


def load_one_csv(path: str) -> Optional[pd.DataFrame]:
    raw = read_csv_auto(path)
    date_col = pick_col(raw, DATE_COL_CANDIDATES)
    open_col = pick_col(raw, OPEN_COL_CANDIDATES)
    high_col = pick_col(raw, HIGH_COL_CANDIDATES)
    low_col = pick_col(raw, LOW_COL_CANDIDATES)
    close_col = pick_col(raw, CLOSE_COL_CANDIDATES)
    vol_col = pick_col(raw, VOL_COL_CANDIDATES)
    code_col = pick_col(raw, CODE_COL_CANDIDATES, required=False)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "open": pd.to_numeric(raw[open_col], errors="coerce"),
            "high": pd.to_numeric(raw[high_col], errors="coerce"),
            "low": pd.to_numeric(raw[low_col], errors="coerce"),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
            "volume": pd.to_numeric(raw[vol_col], errors="coerce"),
        }
    )
    df["code"] = raw[code_col].astype(str).iloc[0] if code_col else os.path.splitext(os.path.basename(path))[0]
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df[(df["date"] < EXCLUDE_START) | (df["date"] > EXCLUDE_END)].copy()
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)].copy()
    if len(df) < MIN_BARS:
        return None
    return df


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.shape(a), np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[mask] = a[mask] / b[mask]
    return out


def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    cn = pd.DataFrame({"日期": x["date"], "开盘": x["open"], "最高": x["high"], "最低": x["low"], "收盘": x["close"], "成交量": x["volume"]})
    cn = technical_indicators.calculate_trend(cn)
    x["trend_line"] = cn["知行短期趋势线"].astype(float)
    x["long_line"] = cn["知行多空线"].astype(float)
    x["trend_ok"] = x["trend_line"] > x["long_line"]

    short_llv = x["low"].rolling(3).min()
    short_hhv = x["close"].rolling(3).max()
    short_den = (short_hhv - short_llv).replace(0, np.nan)
    short_value = (x["close"] - short_llv) / short_den * 100
    long_llv = x["low"].rolling(21).min()
    long_hhv = x["close"].rolling(21).max()
    long_den = (long_hhv - long_llv).replace(0, np.nan)
    long_value = (x["close"] - long_llv) / long_den * 100
    x["pin_signal"] = (short_value <= 30) & (long_value >= 85)

    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    body_low = np.minimum(x["open"], x["close"])
    x["lower_shadow_ratio"] = safe_div(body_low - x["low"], full_range)
    x["trend_slope_3"] = safe_div(x["trend_line"], x["trend_line"].shift(3)) - 1.0
    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5_prev"])
    return x


def load_feature_map() -> Dict[str, pd.DataFrame]:
    feature_map: Dict[str, pd.DataFrame] = {}
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".csv", ".txt"))]
    if MAX_FILES > 0:
        files = files[:MAX_FILES]
    for idx, file_name in enumerate(files, 1):
        df = load_one_csv(os.path.join(DATA_DIR, file_name))
        if df is None:
            continue
        feature_map[str(df["code"].iloc[0])] = build_feature_df(df)
        if idx % 500 == 0:
            print(f"特征进度: {idx}/{len(files)}")
    return feature_map


def simulate_trade(df: pd.DataFrame, signal_idx: int) -> Optional[dict]:
    entry_idx = signal_idx + 1
    exit_idx = signal_idx + 2
    if exit_idx >= len(df):
        return None
    entry = float(df.at[entry_idx, "open"])
    exit_ = float(df.at[exit_idx, "open"])
    if not np.isfinite(entry) or not np.isfinite(exit_) or entry <= 0 or exit_ <= 0:
        return None
    ret = exit_ / entry - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "code": df.at[signal_idx, "code"],
        "ret": ret,
        "success": ret > 0,
        "lower_shadow_ratio": float(df.at[signal_idx, "lower_shadow_ratio"]),
        "signal_vs_ma5": float(df.at[signal_idx, "signal_vs_ma5"]),
        "trend_slope_3": float(df.at[signal_idx, "trend_slope_3"]),
    }


def build_trade_df(feature_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    trades = []
    for code, df in feature_map.items():
        idxs = np.flatnonzero((df["trend_ok"] & df["pin_signal"]).to_numpy())
        for signal_idx in idxs:
            trade = simulate_trade(df, int(signal_idx))
            if trade:
                trades.append(trade)
    return pd.DataFrame(trades)


def build_portfolio_curve(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["signal_date", "portfolio_ret", "equity"])
    rows = []
    equity = INITIAL_CAPITAL
    x = trade_df.copy()
    x["signal_date"] = pd.to_datetime(x["signal_date"])
    for signal_date, group in x.groupby("signal_date", sort=True):
        g = group.sort_values(["sort_score", "code"], ascending=[False, True]).head(MAX_POSITIONS).copy()
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


def compute_equity_metrics(portfolio_df: pd.DataFrame) -> dict:
    if portfolio_df.empty:
        return {"annual_return": np.nan, "max_drawdown": np.nan, "equity_days": 0, "final_equity": np.nan}
    eq = portfolio_df["equity"].astype(float)
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    final_equity = float(eq.iloc[-1])
    days = len(portfolio_df)
    annual_return = (final_equity / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / days) - 1 if final_equity > 0 and days > 0 else np.nan
    return {"annual_return": float(annual_return), "max_drawdown": float(drawdown.min()), "equity_days": int(days), "final_equity": final_equity}


def max_consecutive_failures(flags: List[bool]) -> int:
    cur = worst = 0
    for flag in flags:
        if flag:
            cur = 0
        else:
            cur += 1
            worst = max(worst, cur)
    return worst


def build_mask(df: pd.DataFrame, lower_shadow_rule: str, vol_rule: str, slope_rule: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if lower_shadow_rule == "ls_le_0.05":
        mask &= df["lower_shadow_ratio"] <= 0.05
    elif lower_shadow_rule == "ls_0.05_0.10":
        mask &= (df["lower_shadow_ratio"] > 0.05) & (df["lower_shadow_ratio"] <= 0.10)
    elif lower_shadow_rule == "ls_le_0.10":
        mask &= df["lower_shadow_ratio"] <= 0.10

    if vol_rule == "vol_1.2_1.5":
        mask &= (df["signal_vs_ma5"] > 1.2) & (df["signal_vs_ma5"] <= 1.5)
    elif vol_rule == "vol_1.5_1.8":
        mask &= (df["signal_vs_ma5"] > 1.5) & (df["signal_vs_ma5"] <= 1.8)
    elif vol_rule == "vol_1.2_1.8":
        mask &= (df["signal_vs_ma5"] > 1.2) & (df["signal_vs_ma5"] <= 1.8)

    if slope_rule == "slope_gt_0":
        mask &= df["trend_slope_3"] > 0
    elif slope_rule == "slope_gt_0.3":
        mask &= df["trend_slope_3"] > 0.003
    elif slope_rule == "slope_gt_0.8":
        mask &= df["trend_slope_3"] > 0.008
    return mask.fillna(False)


def summarize(name: str, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    row = {
        "combo_name": name,
        "sample_count": int(len(trade_df)),
        "avg_trade_return": float(trade_df["ret"].mean()) if not trade_df.empty else np.nan,
        "success_rate": float(trade_df["success"].mean()) if not trade_df.empty else np.nan,
        "max_consecutive_failures": int(max_consecutive_failures(trade_df["success"].tolist())) if not trade_df.empty else np.nan,
    }
    row.update(compute_equity_metrics(portfolio_df))
    return row


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    feature_map = load_feature_map()
    trade_df = build_trade_df(feature_map)
    if trade_df.empty:
        raise ValueError("基础单针策略无样本")

    lower_shadow_rules = ["none", "ls_le_0.05", "ls_0.05_0.10", "ls_le_0.10"]
    vol_rules = ["none", "vol_1.2_1.5", "vol_1.5_1.8", "vol_1.2_1.8"]
    slope_rules = ["none", "slope_gt_0", "slope_gt_0.3", "slope_gt_0.8"]

    rows = []
    combos = list(product(lower_shadow_rules, vol_rules, slope_rules))
    print(f"组合数: {len(combos)}")
    for idx, (ls_rule, vol_rule, slope_rule) in enumerate(combos, 1):
        mask = build_mask(trade_df, ls_rule, vol_rule, slope_rule)
        selected = trade_df.loc[mask].copy()
        selected["sort_score"] = 0.0
        if not selected.empty:
            selected["sort_score"] = (
                (1.0 - selected["lower_shadow_ratio"].clip(lower=0, upper=1)).fillna(0.0) * 0.45
                + selected["signal_vs_ma5"].clip(lower=0, upper=3).fillna(0.0) * 0.35
                + selected["trend_slope_3"].clip(lower=-0.05, upper=0.05).fillna(0.0) * 10 * 0.20
            )
        portfolio_df = build_portfolio_curve(selected)
        rows.append(summarize(f"{ls_rule}__{vol_rule}__{slope_rule}", selected, portfolio_df))
        print(f"组合进度: {idx}/{len(combos)}")

    result_df = pd.DataFrame(rows).sort_values(["annual_return", "max_drawdown", "avg_trade_return", "success_rate"], ascending=[False, False, False, False]).reset_index(drop=True)
    result_df.to_csv(os.path.join(OUTPUT_DIR, "combo_results.csv"), index=False, encoding="utf-8-sig")
    eligible = result_df[result_df["sample_count"] >= 200].copy()
    if eligible.empty:
        eligible = result_df[result_df["sample_count"] > 0].copy()
    eligible["drawdown_abs"] = eligible["max_drawdown"].abs()
    best = eligible.sort_values(["annual_return", "drawdown_abs", "avg_trade_return", "success_rate"], ascending=[False, True, False, False]).iloc[0]
    summary = {
        "data_dir": DATA_DIR,
        "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "base_signal": ["趋势线 > 多空线", "符合单针条件"],
        "entry_exit": {"entry": "signal_date_next_open", "holding_days": 1, "exit": "entry_date_plus_1_open"},
        "refined_factors": ["lower_shadow_ratio", "signal_vs_ma5", "trend_slope_3"],
        "best_combo": best.to_dict(),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
