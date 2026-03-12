from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.backtest.run_b2_startup_experiment import load_one_csv, build_feature_df as build_b2_features


DATA_DIR = os.environ.get("BRICK_B2_DATA_DIR", "/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
OUTPUT_DIR = os.environ.get("BRICK_B2_OUTPUT_DIR", "/Users/lidongyang/Desktop/Qstrategy/results/brick_b2_signal_expansion")
MAX_FILES = int(os.environ.get("BRICK_B2_MAX_FILES", "0"))
INITIAL_CAPITAL = 1_000_000.0
TRADING_DAYS_PER_YEAR = 252
MAX_POSITIONS = 10
MAX_SINGLE_WEIGHT = 0.2
BASELINE_SUCCESS_RATE = 0.59375


def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > 1e-12)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    return series.ewm(alpha=m / n, adjust=False).mean()


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


@dataclass(frozen=True)
class Combo:
    砖型反包阈值: float
    量能下限: float
    量能上限: float
    B2涨幅下限: float
    上影线系数: float
    启动线: str

    @property
    def combo_name(self) -> str:
        return (
            f"反包_{self.砖型反包阈值:.1f}"
            f"__量能_{self.量能下限:.1f}_{self.量能上限:.1f}"
            f"__B2涨幅_{self.B2涨幅下限:.2f}"
            f"__上影_{self.上影线系数:.1f}"
            f"__启动线_{self.启动线}"
        )


def build_brick_base(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x["ret1"] = x["close"].pct_change()
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0
    x["trend_ok"] = x["trend_line"] > x["long_line"]

    code = str(x["code"].iloc[0])
    if code.startswith("SH#688"):
        gain_limit = 0.08
    elif code.startswith("SZ#300"):
        gain_limit = 0.08
    else:
        gain_limit = 0.07
    x["gain_limit"] = gain_limit

    x["close_pullback_white"] = x["close"] < x["trend_line"] * 1.01
    x["close_above_white"] = x["close"] > x["trend_line"]
    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5_prev"])

    hhv4 = x["high"].rolling(4).max()
    llv4 = x["low"].rolling(4).min()
    den4 = (hhv4 - llv4).replace(0, np.nan)
    var1a = safe_div((hhv4 - x["close"]), den4) * 100 - 90
    var2a = tdx_sma(pd.Series(var1a, index=x.index), 4, 1) + 100
    var3a = safe_div((x["close"] - llv4), den4) * 100
    var4a = tdx_sma(pd.Series(var3a, index=x.index), 6, 1)
    var5a = tdx_sma(var4a, 6, 1) + 100
    var6a = var5a - var2a
    x["brick"] = np.where(var6a > 4, var6a - 4, 0.0)
    x["brick_prev"] = x["brick"].shift(1)
    x["brick_red_len"] = np.where(x["brick"] > x["brick_prev"], x["brick"] - x["brick_prev"], 0.0)
    x["brick_green_len"] = np.where(x["brick"] < x["brick_prev"], x["brick_prev"] - x["brick"], 0.0)
    x["brick_red"] = x["brick_red_len"] > 0
    x["brick_green"] = x["brick_green_len"] > 0
    x["prev_green_streak"] = pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(1)
    x["close_slope_10"] = x["close"].rolling(10).apply(
        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if np.isfinite(s).all() else np.nan,
        raw=False,
    )
    x["not_sideways"] = np.abs(safe_div(x["close_slope_10"], x["close"].rolling(10).mean())) > 0.002
    x["up_leg_avg_vol"] = x["volume"].shift(4).rolling(3).mean()
    x["pullback_avg_vol"] = x["volume"].shift(1).rolling(3).mean()
    x["pullback_shrinking"] = x["pullback_avg_vol"] < x["up_leg_avg_vol"]
    x["pullback_shrink_ratio"] = safe_div(x["pullback_avg_vol"], x["up_leg_avg_vol"])

    x["pattern_a"] = (
        (x["prev_green_streak"] >= 3)
        & x["brick_red"]
        & x["close_pullback_white"].shift(1).fillna(False)
        & x["close_above_white"]
    )
    x["pattern_b"] = (
        (pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(3) >= 3)
        & x["brick_red"]
        & x["brick_green"].shift(1).fillna(False)
        & x["brick_red"].shift(2).fillna(False)
        & x["close_pullback_white"].shift(1).fillna(False)
        & x["close_above_white"]
    )
    x["rebound_ratio"] = safe_div(x["brick_red_len"], x["brick_green_len"].shift(1))

    trend_spread = np.maximum((x["trend_line"] - x["long_line"]) / x["close"], 0.0)
    rebound_rank_raw = pd.Series(x["rebound_ratio"], index=x.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shrink_quality = 1.0 - np.minimum(np.abs(pd.Series(x["pullback_shrink_ratio"], index=x.index) - 0.8) / 0.3, 1.0)
    x["brick_sort_score"] = 0.50 * shrink_quality.fillna(0.0) + 0.30 * rebound_rank_raw + 0.20 * pd.Series(trend_spread, index=x.index).fillna(0.0)
    return x


def build_feature_map(data_dir: str) -> Dict[str, pd.DataFrame]:
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
        brick_df = build_brick_base(df)
        b2_df = build_b2_features(df)
        merged = brick_df.copy()
        keep_cols = ["基础信号", "双线附近启动", "趋势线启动", "多空线启动", "sort_score", "信号最低点", "小上影", "涨幅"]
        for col in keep_cols:
            merged[f"b2_{col}"] = b2_df[col]
        feature_map[code] = merged
        if idx % 500 == 0 or idx == total:
            print(f"特征进度: {idx}/{total}")
    return feature_map


def build_combos() -> List[Combo]:
    return [
        Combo(rebound, vol_lo, vol_hi, b2_gain, shadow, line_mode)
        for rebound, vol_lo, vol_hi, b2_gain, shadow, line_mode in product(
            [1.0, 1.1, 1.2],
            [1.2, 1.3],
            [2.2, 2.5],
            [0.03, 0.04],
            [0.3, 0.4],
            ["双线启动", "任一线启动"],
        )
        if vol_lo < vol_hi
    ]


def build_signal_cache(feature_map: Dict[str, pd.DataFrame], combos: List[Combo]) -> Dict[str, Dict[str, np.ndarray]]:
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    for combo in combos:
        per_code: Dict[str, np.ndarray] = {}
        for code, df in feature_map.items():
            brick_a = df["pattern_a"] & (df["rebound_ratio"] >= combo.砖型反包阈值)
            brick_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
            brick_mask = (
                (brick_a | brick_b)
                & df["pullback_shrinking"].fillna(False)
                & df["signal_vs_ma5"].between(combo.量能下限, combo.量能上限, inclusive="both")
                & df["not_sideways"].fillna(False)
                & (df["ret1"] <= df["gain_limit"])
                & df["trend_ok"]
            )

            b2_mask = (
                df["b2_基础信号"].fillna(False)
                & (df["b2_涨幅"] >= combo.B2涨幅下限)
            )
            # 放宽小上影：0.3 是原值，0.4 在这里重新算
            if combo.上影线系数 > 0.3:
                real_body = (df["close"] - df["open"]).abs()
                upper_shadow = df["high"] - np.maximum(df["open"], df["close"])
                shadow_mask = (real_body <= 1e-12) | (upper_shadow <= real_body * combo.上影线系数 + 1e-12)
                b2_mask &= shadow_mask
            else:
                b2_mask &= df["b2_小上影"].fillna(False)

            if combo.启动线 == "双线启动":
                b2_mask &= df["b2_双线附近启动"].fillna(False)
            else:
                anyline = df["b2_趋势线启动"].fillna(False) | df["b2_多空线启动"].fillna(False)
                b2_mask &= anyline

            mask = brick_mask & b2_mask
            idxs = np.flatnonzero(mask.fillna(False).to_numpy())
            if len(idxs) > 0:
                per_code[code] = idxs
        cache[combo.combo_name] = per_code
    return cache


def simulate_trade(df: pd.DataFrame, signal_idx: int) -> Optional[dict]:
    entry_idx = signal_idx + 1
    exit_idx = signal_idx + 4
    if exit_idx >= len(df):
        return None
    entry_price = float(df.at[entry_idx, "open"])
    exit_price = float(df.at[exit_idx, "open"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0 or exit_price <= 0:
        return None
    ret = exit_price / entry_price - 1.0
    sort_score = float(df.at[signal_idx, "brick_sort_score"] + df.at[signal_idx, "b2_sort_score"])
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "ret": ret,
        "success": ret > 0,
        "sort_score": sort_score,
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


def compute_equity_metrics(portfolio_df: pd.DataFrame) -> dict:
    if portfolio_df.empty:
        return {"annual_return": np.nan, "max_drawdown": np.nan, "equity_days": 0, "final_equity": np.nan}
    eq = portfolio_df["equity"].astype(float)
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    final_equity = float(eq.iloc[-1])
    days = len(portfolio_df)
    annual_return = (final_equity / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / days) - 1 if final_equity > 0 and days > 0 else np.nan
    return {
        "annual_return": float(annual_return),
        "max_drawdown": float(drawdown.min()),
        "equity_days": int(days),
        "final_equity": final_equity,
    }


def max_consecutive_failures(success_flags: List[bool]) -> int:
    current = 0
    worst = 0
    for flag in success_flags:
        if flag:
            current = 0
        else:
            current += 1
            worst = max(worst, current)
    return worst


def summarize(combo: Combo, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    row = {
        "sample_count": int(len(trade_df)) if not trade_df.empty else 0,
        "avg_trade_return": float(trade_df["ret"].mean()) if not trade_df.empty else np.nan,
        "success_rate": float(trade_df["success"].mean()) if not trade_df.empty else np.nan,
        "max_consecutive_failures": int(max_consecutive_failures(trade_df["success"].tolist())) if not trade_df.empty else np.nan,
    }
    row.update(compute_equity_metrics(portfolio_df))
    row.update(asdict(combo))
    row["combo_name"] = combo.combo_name
    row["meets_baseline_success"] = bool(row["success_rate"] >= BASELINE_SUCCESS_RATE) if np.isfinite(row["success_rate"]) else False
    return row


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    feature_map = build_feature_map(DATA_DIR)
    combos = build_combos()
    print(f"组合数: {len(combos)}")
    signal_cache = build_signal_cache(feature_map, combos)

    rows = []
    for idx, combo in enumerate(combos, 1):
        trades = []
        for code, idxs in signal_cache[combo.combo_name].items():
            df = feature_map[code]
            for signal_idx in idxs:
                trade = simulate_trade(df, int(signal_idx))
                if trade is None:
                    continue
                trade["code"] = code
                trades.append(trade)
        trade_df = pd.DataFrame(trades)
        portfolio_df = build_portfolio_curve(trade_df)
        rows.append(summarize(combo, trade_df, portfolio_df))
        print(f"组合进度: {idx}/{len(combos)}")

    result_df = pd.DataFrame(rows).sort_values(
        ["sample_count", "success_rate", "annual_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    result_df.to_csv(os.path.join(OUTPUT_DIR, "combo_results.csv"), index=False, encoding="utf-8-sig")

    eligible = result_df[(result_df["sample_count"] > 0) & (result_df["meets_baseline_success"])].copy()
    if eligible.empty:
        best = result_df[result_df["sample_count"] > 0].sort_values(
            ["success_rate", "sample_count", "annual_return"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        best = eligible.sort_values(
            ["sample_count", "annual_return", "avg_trade_return"],
            ascending=[False, False, False],
        ).iloc[0]

    summary = {
        "data_dir": DATA_DIR,
        "goal": f"在成功率不低于基线 {BASELINE_SUCCESS_RATE:.6f} 的前提下增加信号数",
        "entry_exit": {"entry": "T+1_open", "holding_days": 3, "exit": "T+4_open"},
        "baseline_success_rate": BASELINE_SUCCESS_RATE,
        "best_combo": best.to_dict(),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
