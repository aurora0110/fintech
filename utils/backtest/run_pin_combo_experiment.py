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


DATA_DIR = os.environ.get("PIN_DATA_DIR", "/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
OUTPUT_DIR = os.environ.get("PIN_OUTPUT_DIR", "/Users/lidongyang/Desktop/Qstrategy/results/pin_combo_experiment")
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


@dataclass(frozen=True)
class Combo:
    shrink_today: bool
    top2_bullish: bool
    rsi_bullish: bool
    macd_positive: bool
    top2_window: int

    @property
    def combo_name(self) -> str:
        return (
            f"shrink_{int(self.shrink_today)}"
            f"__top2_{int(self.top2_bullish)}"
            f"__rsi_{int(self.rsi_bullish)}"
            f"__macd_{int(self.macd_positive)}"
            f"__top2_window_{self.top2_window}"
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
        & (df["volume"] >= 0)
    ].copy()
    if len(df) < MIN_BARS:
        return None
    return df


def top2_bullish_flag(close: pd.Series, open_: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    is_bull = close > open_
    out = np.zeros(len(close), dtype=bool)
    for i in range(window - 1, len(close)):
        vol_window = volume.iloc[i - window + 1 : i + 1]
        idx = vol_window.nlargest(2).index
        out[i] = bool(is_bull.loc[idx].all())
    return pd.Series(out, index=close.index)


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
    cn = technical_indicators.calculate_rsi(cn)
    cn = technical_indicators.calculate_macd(cn)

    x["trend_line"] = cn["知行短期趋势线"].astype(float)
    x["long_line"] = cn["知行多空线"].astype(float)
    x["trend_ok"] = x["trend_line"] > x["long_line"]
    short_n = 3
    long_n = 21
    short_llv = x["low"].rolling(short_n).min()
    short_hhv = x["close"].rolling(short_n).max()
    short_den = (short_hhv - short_llv).replace(0, np.nan)
    short_value = (x["close"] - short_llv) / short_den * 100
    long_llv = x["low"].rolling(long_n).min()
    long_hhv = x["close"].rolling(long_n).max()
    long_den = (long_hhv - long_llv).replace(0, np.nan)
    long_value = (x["close"] - long_llv) / long_den * 100
    x["pin_signal"] = (short_value <= 30) & (long_value >= 85)

    x["today_shrink"] = x["volume"] < x["volume"].shift(1)
    x["top2_bullish_30"] = top2_bullish_flag(x["close"], x["open"], x["volume"], 30)
    x["top2_bullish_60"] = top2_bullish_flag(x["close"], x["open"], x["volume"], 60)
    x["rsi_bullish"] = (
        (cn["RSI_14"] > cn["RSI_28"])
        & (cn["RSI_28"] > cn["RSI_57"])
    )
    x["macd_positive"] = cn["MACD_DIFF"] > 0
    x["ret1"] = x["close"].pct_change()

    hit_count = (
        x["today_shrink"].fillna(False).astype(int)
        + x["top2_bullish_30"].fillna(False).astype(int)
        + x["top2_bullish_60"].fillna(False).astype(int)
        + x["rsi_bullish"].fillna(False).astype(int)
        + x["macd_positive"].fillna(False).astype(int)
    )
    x["sort_score"] = hit_count - x["ret1"].clip(lower=-0.1, upper=0.2).fillna(0.0)
    return x


def load_feature_map(data_dir: str) -> Dict[str, pd.DataFrame]:
    feature_map: Dict[str, pd.DataFrame] = {}
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".txt"))]
    if MAX_FILES > 0:
        files = files[:MAX_FILES]
    for idx, file_name in enumerate(files, 1):
        df = load_one_csv(os.path.join(data_dir, file_name))
        if df is None:
            continue
        code = str(df["code"].iloc[0])
        feature_map[code] = build_feature_df(df)
        if idx % 500 == 0:
            print(f"特征进度: {idx}/{len(files)}")
    return feature_map


def build_combos() -> List[Combo]:
    return [
        Combo(shrink_today=shrink, top2_bullish=top2, rsi_bullish=rsi, macd_positive=macd, top2_window=window)
        for window in [30, 60]
        for shrink, top2, rsi, macd in product([False, True], repeat=4)
    ]


def simulate_trade(df: pd.DataFrame, signal_idx: int) -> Optional[dict]:
    entry_idx = signal_idx + 1
    exit_idx = signal_idx + 2
    if exit_idx >= len(df):
        return None
    entry_price = float(df.at[entry_idx, "open"])
    exit_price = float(df.at[exit_idx, "open"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0 or exit_price <= 0:
        return None
    ret = exit_price / entry_price - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "ret": ret,
        "holding_days": 1,
        "success": ret > 0,
        "sort_score": float(df.at[signal_idx, "sort_score"]),
        "today_shrink": bool(df.at[signal_idx, "today_shrink"]),
        "top2_bullish_30": bool(df.at[signal_idx, "top2_bullish_30"]),
        "top2_bullish_60": bool(df.at[signal_idx, "top2_bullish_60"]),
        "rsi_bullish": bool(df.at[signal_idx, "rsi_bullish"]),
        "macd_positive": bool(df.at[signal_idx, "macd_positive"]),
    }


def build_signal_cache(feature_map: Dict[str, pd.DataFrame], combos: List[Combo]) -> Dict[str, Dict[str, np.ndarray]]:
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    for combo in combos:
        per_code: Dict[str, np.ndarray] = {}
        top2_col = f"top2_bullish_{combo.top2_window}"
        for code, df in feature_map.items():
            mask = df["trend_ok"] & df["pin_signal"]
            if combo.shrink_today:
                mask &= df["today_shrink"].fillna(False)
            if combo.top2_bullish:
                mask &= df[top2_col].fillna(False)
            if combo.rsi_bullish:
                mask &= df["rsi_bullish"].fillna(False)
            if combo.macd_positive:
                mask &= df["macd_positive"].fillna(False)
            idxs = np.flatnonzero(mask.to_numpy())
            if len(idxs) > 0:
                per_code[code] = idxs
        cache[combo.combo_name] = per_code
    return cache


def build_portfolio_curve(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["signal_date", "portfolio_ret", "equity"])
    rows = []
    equity = INITIAL_CAPITAL
    trade_df = trade_df.copy()
    trade_df["signal_date"] = pd.to_datetime(trade_df["signal_date"])
    for signal_date, group in trade_df.groupby("signal_date", sort=True):
        g = group.copy().sort_values(["sort_score", "code"], ascending=[False, True]).head(MAX_POSITIONS)
        score = g["sort_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        score = score.clip(lower=0.0)
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
    non_empty = result_df["sample_count"].fillna(0).gt(0).any()
    if not non_empty:
        raise ValueError("没有任何非零样本组合")
    finite_dd = result_df["max_drawdown"].dropna()
    if not finite_dd.empty and ((finite_dd < -1.0) | (finite_dd > 0.0)).any():
        raise ValueError("存在非法最大回撤")
    valid = result_df.dropna(subset=["annual_return", "final_equity"])
    inconsistent = valid[
        ((valid["final_equity"] > INITIAL_CAPITAL) & (valid["annual_return"] <= -EPS))
        | ((valid["final_equity"] < INITIAL_CAPITAL) & (valid["annual_return"] >= EPS))
    ]
    if not inconsistent.empty:
        raise ValueError("年化收益与最终净值方向不一致")


def pick_best(result_df: pd.DataFrame) -> pd.Series:
    eligible = result_df[result_df["sample_count"] >= 1000].copy()
    if eligible.empty:
        eligible = result_df[result_df["sample_count"] > 0].copy()
    eligible["drawdown_abs"] = eligible["max_drawdown"].abs()
    ranked = eligible.sort_values(
        ["annual_return", "drawdown_abs", "avg_trade_return", "success_rate"],
        ascending=[False, True, False, False],
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
        ["annual_return", "max_drawdown", "avg_trade_return", "success_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    validate_result_df(result_df)
    best = pick_best(result_df)

    result_df.to_csv(os.path.join(OUTPUT_DIR, "combo_results.csv"), index=False, encoding="utf-8-sig")
    summary = {
        "data_dir": DATA_DIR,
        "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "entry_exit": {
            "entry": "signal_date_next_open",
            "holding_days": 1,
            "exit": "entry_date_plus_1_open",
            "stop_loss": None,
            "take_profit": None,
        },
        "capital": "100万，最多10只，按可选条件命中数排序加权，单票上限20%",
        "base_signal": ["趋势线 > 多空线", "符合单针条件"],
        "optional_conditions": [
            "当日缩量(今量<昨量)",
            "30日或60日内最大两个成交量对应阳线",
            "RSI_14 > RSI_28 > RSI_57",
            "MACD_DIFF > 0",
        ],
        "best_combo": best.to_dict(),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
