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

from utils import technical_indicators


DATA_DIR = os.environ.get("B2_T1_DATA_DIR", "/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
OUTPUT_DIR = os.environ.get(
    "B2_T1_REFINED_OUTPUT_DIR", "/Users/lidongyang/Desktop/Qstrategy/results/b2_t1_green_optimization_refined"
)
MAX_FILES = int(os.environ.get("B2_T1_MAX_FILES", "0"))
MIN_BARS = 180
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


@dataclass(frozen=True)
class Combo:
    启动线: str
    放量强度: str
    趋势线5日斜率: bool
    收盘位置强: bool
    上影线占比优: bool
    近20日高点: bool
    前10日不过热: bool

    @property
    def combo_name(self) -> str:
        return (
            f"启动线_{self.启动线}__放量_{self.放量强度}"
            f"__趋势线5日斜率_{self.趋势线5日斜率}__收盘位置强_{self.收盘位置强}"
            f"__上影线占比优_{self.上影线占比优}__近20日高点_{self.近20日高点}"
            f"__前10日不过热_{self.前10日不过热}"
        )


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
    if code_col:
        df["code"] = raw[code_col].astype(str).iloc[0]
    else:
        df["code"] = os.path.splitext(os.path.basename(path))[0]
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df[(df["date"] < EXCLUDE_START) | (df["date"] > EXCLUDE_END)].copy()
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] > 0)
    ].copy()
    if len(df) < MIN_BARS:
        return None
    return df


def safe_div(a: pd.Series, b: pd.Series | float, default: float = np.nan) -> pd.Series:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full_like(a_arr, default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > EPS)
    out[mask] = a_arr[mask] / b_arr[mask]
    return pd.Series(out, index=a.index if isinstance(a, pd.Series) else None)


def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    cn = pd.DataFrame(
        {
            "日期": x["date"],
            "开盘": x["open"],
            "最高": x["high"],
            "最低": x["low"],
            "收盘": x["close"],
            "成交量": x["volume"],
        }
    )
    cn = technical_indicators.calculate_trend(cn)
    cn = technical_indicators.calculate_kdj(cn)

    x["趋势线"] = cn["知行短期趋势线"].astype(float)
    x["多空线"] = cn["知行多空线"].astype(float)
    x["J"] = cn["J"].astype(float)
    x["涨幅"] = x["close"].pct_change()
    x["trend_ok"] = x["趋势线"] > x["多空线"]

    real_body = (x["close"] - x["open"]).abs()
    upper_shadow = x["high"] - np.maximum(x["open"], x["close"])
    full_range = x["high"] - x["low"]
    x["小上影"] = (real_body <= EPS) | (upper_shadow <= real_body * 0.3 + EPS)
    x["上影线占比"] = np.where(full_range > 0, upper_shadow / full_range, np.nan)
    x["收盘位置"] = np.where(full_range > 0, (x["close"] - x["low"]) / full_range, np.nan)

    x["放量基础"] = (x["volume"] > x["volume"].shift(1)) & (x["volume"] > x["volume"].rolling(5).mean())
    x["量比_五日"] = safe_div(x["volume"], x["volume"].rolling(5).mean())

    x["J启动"] = (
        (x["J"] < 80)
        & (x["J"] > x["J"].shift(1))
        & (x["J"].shift(1) < x["J"].shift(2))
        & (x["J"].shift(2) < x["J"].shift(3))
    )
    x["趋势线启动"] = (x["close"].shift(1) <= x["趋势线"].shift(1) * 1.01) & (x["close"] > x["趋势线"])
    x["多空线启动"] = (x["close"].shift(1) <= x["多空线"].shift(1) * 1.01) & (x["close"] > x["多空线"])
    x["双线附近启动"] = x["趋势线启动"] & x["多空线启动"]

    x["趋势线5日斜率"] = np.where(x["趋势线"].shift(5) > 0, x["趋势线"] / x["趋势线"].shift(5) - 1.0, np.nan)
    x["距20日高点"] = safe_div(x["close"], x["close"].rolling(20).max(), default=np.nan) - 1.0
    x["前10日涨幅"] = x["close"].shift(1) / x["close"].shift(11) - 1.0

    x["基础信号"] = x["trend_ok"] & (x["涨幅"] >= 0.04) & x["小上影"] & x["放量基础"] & x["J启动"]
    gain_edge = (x["涨幅"] - 0.04).clip(lower=0)
    volume_moderate = -(x["量比_五日"] - 1.8).abs()
    line_strength = safe_div(x["close"] - np.maximum(x["趋势线"], x["多空线"]), x["close"], default=0.0).fillna(0.0)
    j_strength = safe_div(x["J"] - x["J"].shift(1), x["J"].shift(1).abs() + 1.0, default=0.0).fillna(0.0)
    x["sort_score"] = (
        -gain_edge.fillna(0.0) * 4.0
        + volume_moderate.fillna(-10.0) * 0.8
        + line_strength * 6.0
        + j_strength * 2.0
    )
    return x


def load_feature_map(data_dir: str) -> Dict[str, pd.DataFrame]:
    feature_map: Dict[str, pd.DataFrame] = {}
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".txt"))])
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


def build_combos() -> List[Combo]:
    return [
        Combo(
            启动线=启动线,
            放量强度=放量强度,
            趋势线5日斜率=趋势线5日斜率,
            收盘位置强=收盘位置强,
            上影线占比优=上影线占比优,
            近20日高点=近20日高点,
            前10日不过热=前10日不过热,
        )
        for 启动线, 放量强度, 趋势线5日斜率, 收盘位置强, 上影线占比优, 近20日高点, 前10日不过热 in product(
            ["趋势线启动", "双线启动"],
            ["仅基础放量", "温和放量", "中等放量"],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
            [False, True],
        )
    ]


def build_signal_cache(feature_map: Dict[str, pd.DataFrame], combos: List[Combo]) -> Dict[str, Dict[str, np.ndarray]]:
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    for combo in combos:
        per_code: Dict[str, np.ndarray] = {}
        for code, df in feature_map.items():
            mask = df["基础信号"].copy()
            if combo.启动线 == "趋势线启动":
                mask &= df["趋势线启动"]
            elif combo.启动线 == "双线启动":
                mask &= df["双线附近启动"]

            if combo.放量强度 == "温和放量":
                mask &= df["量比_五日"].between(1.2, 1.8, inclusive="both")
            elif combo.放量强度 == "中等放量":
                mask &= df["量比_五日"].between(1.5, 2.5, inclusive="both")

            if combo.趋势线5日斜率:
                mask &= df["趋势线5日斜率"] > 0
            if combo.收盘位置强:
                mask &= df["收盘位置"] >= 0.7
            if combo.上影线占比优:
                mask &= df["上影线占比"] <= 0.15
            if combo.近20日高点:
                mask &= df["距20日高点"] >= -0.05
            if combo.前10日不过热:
                mask &= df["前10日涨幅"] <= 0.03

            idxs = np.flatnonzero(mask.fillna(False).to_numpy())
            if len(idxs) > 0:
                per_code[code] = idxs
        cache[combo.combo_name] = per_code
    return cache


def simulate_trade(df: pd.DataFrame, signal_idx: int) -> Optional[dict]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    entry_price = float(df.at[entry_idx, "open"])
    exit_price = float(df.at[entry_idx, "close"])
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(exit_price) or exit_price <= 0:
        return None
    ret = exit_price / entry_price - 1.0
    success = exit_price > entry_price
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "ret": ret,
        "success": success,
        "sort_score": float(df.at[signal_idx, "sort_score"]),
        "量比_五日": float(df.at[signal_idx, "量比_五日"]),
        "收盘位置": float(df.at[signal_idx, "收盘位置"]),
        "上影线占比": float(df.at[signal_idx, "上影线占比"]),
        "趋势线5日斜率": float(df.at[signal_idx, "趋势线5日斜率"]),
        "距20日高点": float(df.at[signal_idx, "距20日高点"]),
        "前10日涨幅": float(df.at[signal_idx, "前10日涨幅"]),
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


def summarize_combo(combo: Combo, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    if trade_df.empty:
        row = {
            "sample_count": 0,
            "avg_trade_return": np.nan,
            "success_rate": np.nan,
            "max_consecutive_failures": np.nan,
        }
    else:
        row = {
            "sample_count": int(len(trade_df)),
            "avg_trade_return": float(trade_df["ret"].mean()),
            "success_rate": float(trade_df["success"].mean()),
            "max_consecutive_failures": int(max_consecutive_failures(trade_df["success"].tolist())),
        }
    row.update(compute_equity_metrics(portfolio_df))
    row.update(asdict(combo))
    row["combo_name"] = combo.combo_name
    return row


def validate_result_df(result_df: pd.DataFrame) -> None:
    if not result_df["sample_count"].fillna(0).gt(0).any():
        raise ValueError("没有任何非零样本组合")
    finite_dd = result_df["max_drawdown"].dropna()
    if not finite_dd.empty and ((finite_dd < -1.0) | (finite_dd > 0.0)).any():
        raise ValueError("存在非法最大回撤")


def pick_best(result_df: pd.DataFrame) -> pd.Series:
    eligible = result_df[result_df["sample_count"] >= 300].copy()
    if eligible.empty:
        eligible = result_df[result_df["sample_count"] > 0].copy()
    eligible["drawdown_abs"] = eligible["max_drawdown"].abs()
    ranked = eligible.sort_values(
        ["success_rate", "avg_trade_return", "annual_return", "drawdown_abs"],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]


def run_combo(feature_map: Dict[str, pd.DataFrame], combo: Combo, signal_cache: Dict[str, Dict[str, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = []
    per_code = signal_cache[combo.combo_name]
    for code, idxs in per_code.items():
        df = feature_map[code]
        for signal_idx in idxs:
            trade = simulate_trade(df, int(signal_idx))
            if trade is None:
                continue
            trade["code"] = code
            trades.append(trade)
    trade_df = pd.DataFrame(trades)
    portfolio_df = build_portfolio_curve(trade_df)
    return trade_df, portfolio_df


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    feature_map = load_feature_map(DATA_DIR)
    combos = build_combos()
    print(f"组合数: {len(combos)}")
    signal_cache = build_signal_cache(feature_map, combos)

    rows = []
    for idx, combo in enumerate(combos, 1):
        trade_df, portfolio_df = run_combo(feature_map, combo, signal_cache)
        rows.append(summarize_combo(combo, trade_df, portfolio_df))
        print(f"组合进度: {idx}/{len(combos)}")

    result_df = pd.DataFrame(rows).sort_values(
        ["success_rate", "avg_trade_return", "annual_return", "max_drawdown"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    validate_result_df(result_df)
    best = pick_best(result_df)

    result_df.to_csv(os.path.join(OUTPUT_DIR, "combo_results.csv"), index=False, encoding="utf-8-sig")
    summary = {
        "data_dir": DATA_DIR,
        "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "entry_exit": {
            "signal_date": "T",
            "entry": "T+1_open",
            "exit": "T+1_close",
            "optimization_target": "次日开盘买入后，次日收涨阳线(success = T+1 close > T+1 open)",
        },
        "best_combo": best.to_dict(),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
