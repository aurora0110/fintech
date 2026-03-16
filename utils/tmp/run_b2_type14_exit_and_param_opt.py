from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
RESULT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/b2_type14_exit_param_opt_20260313")

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
TRADING_DAYS_PER_YEAR = 252

RET1_MIN_VALUES = [0.03, 0.04, 0.05]
J_MAX_VALUES = [80.0, 90.0, 100.0]
UPPER_SHADOW_BODY_RATIO_VALUES = [0.3, 0.5, 0.8]
TYPE1_NEAR_RATIO_VALUES = [1.01, 1.02, 1.03]
TYPE1_J_RANK20_MAX_VALUES = [0.05, 0.10, 0.15]
TYPE4_TOUCH_RATIO_VALUES = [1.00, 1.01, 1.02]
TYPE4_LOOKBACK = 20


@dataclass(frozen=True)
class SignalParams:
    ret1_min: float
    j_max: float
    upper_shadow_body_ratio: float
    type1_near_ratio: float
    type1_j_rank20_max: float
    type4_touch_ratio: float

    @property
    def name(self) -> str:
        return (
            f"r{self.ret1_min:.2f}_j{int(self.j_max)}_u{self.upper_shadow_body_ratio:.1f}_"
            f"n{self.type1_near_ratio:.2f}_jr{self.type1_j_rank20_max:.2f}_t{self.type4_touch_ratio:.2f}"
        )


@dataclass(frozen=True)
class ExitRule:
    name: str
    max_hold_days: int
    take_profit: Optional[float] = None
    take_profit_exec: str = "next_open"
    stop_loss_mode: Optional[str] = None  # entry_low_099 / entry_day_low / none
    stop_exec: str = "same_day_threshold"
    trend_exit_mode: Optional[str] = None  # close_below_trend / two_closes_below_trend / unrecovered_2d / bearish_vol


EXIT_RULES = [
    ExitRule("hold20_close", 20),
    ExitRule("hold30_close", 30),
    ExitRule("tp8_hold30", 30, take_profit=0.08),
    ExitRule("tp12_hold30", 30, take_profit=0.12),
    ExitRule("tp15_hold30", 30, take_profit=0.15),
    ExitRule("trend_break_hold30", 30, trend_exit_mode="close_below_trend"),
    ExitRule("trend_break_2d_hold30", 30, trend_exit_mode="two_closes_below_trend"),
    ExitRule("trend_unrecovered_2d_hold30", 30, trend_exit_mode="unrecovered_2d"),
    ExitRule("bearish_vol_hold30", 30, trend_exit_mode="bearish_vol"),
    ExitRule("entry_day_low099_hold30", 30, stop_loss_mode="entry_day_low_099"),
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a.astype(float) / b.astype(float).replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def load_one_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        raw = pd.read_csv(path)
        if raw.shape[1] <= 1:
            raw = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
    except Exception:
        return None

    col_map = {}
    for want, candidates in {
        "date": ["date", "Date", "trade_date", "日期", "DATE"],
        "open": ["open", "Open", "开盘", "OPEN"],
        "high": ["high", "High", "最高", "HIGH"],
        "low": ["low", "Low", "最低", "LOW"],
        "close": ["close", "Close", "收盘", "CLOSE"],
        "volume": ["volume", "vol", "Volume", "成交量", "VOL"],
    }.items():
        found = next((c for c in candidates if c in raw.columns), None)
        if found is None:
            return None
        col_map[want] = found

    x = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[col_map["date"]], errors="coerce"),
            "open": pd.to_numeric(raw[col_map["open"]], errors="coerce"),
            "high": pd.to_numeric(raw[col_map["high"]], errors="coerce"),
            "low": pd.to_numeric(raw[col_map["low"]], errors="coerce"),
            "close": pd.to_numeric(raw[col_map["close"]], errors="coerce"),
            "volume": pd.to_numeric(raw[col_map["volume"]], errors="coerce"),
        }
    ).dropna()
    x = x[(x["open"] > 0) & (x["high"] > 0) & (x["low"] > 0) & (x["close"] > 0) & (x["volume"] > 0)].copy()
    if len(x) < 150:
        return None
    x = x.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    x["code"] = path.stem
    return x


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ret1"] = x["close"].pct_change()
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0
    x["trend_ok"] = x["trend_line"] > x["long_line"]

    low_9 = x["low"].rolling(9).min()
    high_9 = x["high"].rolling(9).max()
    rsv = _safe_div(x["close"] - low_9, high_9 - low_9) * 100.0
    x["K"] = rsv.ewm(com=2, adjust=False).mean()
    x["D"] = x["K"].ewm(com=2, adjust=False).mean()
    x["J"] = 3 * x["K"] - 2 * x["D"]
    x["j_rank20"] = x["J"].rolling(20, min_periods=20).apply(
        lambda win: pd.Series(win).rank(pct=True).iloc[-1], raw=False
    )
    x["j_rank20_prev"] = x["j_rank20"].shift(1)

    x["vol_ma5"] = x["volume"].rolling(5).mean()
    x["vol_ma10"] = x["volume"].rolling(10).mean()
    x["signal_vs_ma5"] = _safe_div(x["volume"], x["vol_ma5"])
    x["vol_vs_prev"] = _safe_div(x["volume"], x["volume"].shift(1))
    x["vol_vs_ma10"] = _safe_div(x["volume"], x["vol_ma10"])

    real_body = (x["close"] - x["open"]).abs()
    upper_shadow = x["high"] - np.maximum(x["open"], x["close"])
    lower_shadow = np.minimum(x["open"], x["close"]) - x["low"]
    full_range = (x["high"] - x["low"]).replace(0.0, np.nan)
    x["body"] = real_body
    x["upper_shadow"] = upper_shadow
    x["lower_shadow"] = lower_shadow
    x["upper_shadow_body_ratio"] = _safe_div(upper_shadow, real_body)
    x["close_position"] = _safe_div(x["close"] - x["low"], full_range).clip(0.0, 1.0)
    x["body_ratio"] = _safe_div(real_body, full_range).clip(0.0, 1.0)
    x["lower_shadow_ratio"] = _safe_div(lower_shadow, full_range).clip(0.0, 1.0)

    x["near_20d_high_ratio"] = _safe_div(x["close"], x["high"].rolling(20).max())
    x["near_20d_low_ratio"] = _safe_div(x["close"], x["low"].rolling(20).min())
    x["trend_slope_3"] = _safe_div(x["trend_line"], x["trend_line"].shift(3)) - 1.0
    x["trend_slope_5"] = _safe_div(x["trend_line"], x["trend_line"].shift(5)) - 1.0
    x["long_slope_3"] = _safe_div(x["long_line"], x["long_line"].shift(3)) - 1.0
    x["long_slope_5"] = _safe_div(x["long_line"], x["long_line"].shift(5)) - 1.0
    x["trend_line_lead"] = _safe_div(x["trend_line"] - x["long_line"], x["close"])
    x["prev_ret1"] = x["ret1"].shift(1)

    bull_cross = (x["trend_line"] > x["long_line"]) & (x["trend_line"].shift(1) <= x["long_line"].shift(1))
    last_cross_idx: List[float] = [np.nan] * len(x)
    last_seen = np.nan
    for i, is_cross in enumerate(bull_cross.fillna(False).tolist()):
        if is_cross:
            last_seen = float(i)
        last_cross_idx[i] = last_seen
    x["last_bull_cross_idx"] = last_cross_idx

    x["sort_score"] = (
        0.35 * x["close_position"].fillna(0.0)
        + 0.20 * (1 - np.minimum(np.abs(x["signal_vs_ma5"].fillna(0.0) - 1.9) / 0.6, 1.0)).clip(lower=0.0)
        + 0.20 * x["trend_line_lead"].fillna(0.0).clip(lower=0.0)
        + 0.10 * (1 - np.minimum(x["J"].fillna(999.0) / 90.0, 1.0)).clip(lower=0.0)
        + 0.10 * np.maximum(x["trend_slope_5"].fillna(0.0), 0.0)
        + 0.05 * x["body_ratio"].fillna(0.0)
    )
    return x


def signal_mask(x: pd.DataFrame, params: SignalParams) -> Tuple[pd.Series, pd.Series, pd.Series]:
    base = (
        x["trend_ok"]
        & (x["close"] > x["open"])
        & (x["ret1"] >= params.ret1_min)
        & ((x["body"] <= 1e-12) | (x["upper_shadow"] <= x["body"] * params.upper_shadow_body_ratio + 1e-12))
        & (x["volume"] > x["volume"].shift(1))
        & (x["volume"] > x["vol_ma5"])
        & (x["J"] < params.j_max)
        & (x["J"] > x["J"].shift(1))
        & (x["J"].shift(1) < x["J"].shift(2))
        & (x["J"].shift(2) < x["J"].shift(3))
    )
    type1 = (
        (x["close"].shift(1) <= x["long_line"].shift(1) * params.type1_near_ratio)
        & (x["j_rank20_prev"] <= params.type1_j_rank20_max)
    )
    type4 = pd.Series(False, index=x.index)
    for i in range(len(x)):
        if not bool(base.iat[i]):
            continue
        if i < 2:
            continue
        last_cross = x["last_bull_cross_idx"].iat[i]
        if pd.isna(last_cross):
            continue
        cross_idx = int(last_cross)
        if i - cross_idx > TYPE4_LOOKBACK:
            continue
        prev_touch = (
            (x["low"].iat[i - 1] <= x["trend_line"].iat[i - 1] * params.type4_touch_ratio)
            or (x["close"].iat[i - 1] <= x["trend_line"].iat[i - 1] * params.type4_touch_ratio)
        )
        if not prev_touch:
            continue
        if i - 2 > cross_idx:
            between = x.iloc[cross_idx + 1 : i - 1]
            had_touch = (
                (between["low"] <= between["trend_line"] * params.type4_touch_ratio)
                | (between["close"] <= between["trend_line"] * params.type4_touch_ratio)
            ).any()
            if had_touch:
                continue
        type4.iat[i] = True
    active = base & (type1 | type4)
    return active, type1, type4


def valid_signal_date(d: pd.Timestamp) -> bool:
    return (d < EXCLUDE_START) or (d > EXCLUDE_END)


def precompute_signal_rows(dfs: Dict[str, pd.DataFrame], params: SignalParams) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total = len(dfs)
    for idx, (code, x) in enumerate(dfs.items(), start=1):
        active, type1, type4 = signal_mask(x, params)
        active_idx = np.where(active.fillna(False).to_numpy())[0]
        for i in active_idx:
            sig_date = x.at[i, "date"]
            if not valid_signal_date(sig_date):
                continue
            if i + 1 >= len(x):
                continue
            rows.append(
                {
                    "code": code,
                    "signal_idx": i,
                    "signal_date": sig_date,
                    "entry_idx": i + 1,
                    "entry_date": x.at[i + 1, "date"],
                    "entry_open": float(x.at[i + 1, "open"]),
                    "signal_low": float(x.at[i, "low"]),
                    "sort_score": float(x.at[i, "sort_score"]),
                    "type1": bool(type1.iat[i]),
                    "type4": bool(type4.iat[i]),
                }
            )
        if idx % 500 == 0:
            print(f"信号预计算进度: {idx}/{total}")
    return pd.DataFrame(rows)


def _exit_trade(x: pd.DataFrame, signal_idx: int, rule: ExitRule) -> Tuple[int, float, str]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(x):
        return len(x) - 1, float(x.iloc[-1]["close"]), "insufficient_data"
    entry_open = float(x.at[entry_idx, "open"])
    entry_day_low = float(x.at[entry_idx, "low"])
    signal_low = float(x.at[signal_idx, "low"])
    stop_price = None
    if rule.stop_loss_mode == "entry_day_low_099":
        stop_price = entry_day_low * 0.99

    max_exit_idx = min(entry_idx + rule.max_hold_days, len(x) - 1)
    unrecovered_count = 0
    below_trend_count = 0
    for i in range(entry_idx + 1, max_exit_idx + 1):
        row = x.iloc[i]

        if stop_price is not None and float(row["low"]) <= stop_price:
            return i, stop_price, "stop_loss_same_day"

        if rule.take_profit is not None:
            tp_price = entry_open * (1.0 + rule.take_profit)
            if float(row["high"]) >= tp_price:
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, "open"]), f"take_profit_{rule.take_profit:.2f}"

        if rule.trend_exit_mode == "close_below_trend":
            if float(row["close"]) < float(row["trend_line"]):
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, "open"]), "trend_break"
        elif rule.trend_exit_mode == "two_closes_below_trend":
            if float(row["close"]) < float(row["trend_line"]):
                below_trend_count += 1
            else:
                below_trend_count = 0
            if below_trend_count >= 2:
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, "open"]), "trend_break_2d"
        elif rule.trend_exit_mode == "unrecovered_2d":
            if float(row["close"]) < float(row["trend_line"]):
                unrecovered_count += 1
            else:
                unrecovered_count = 0
            if unrecovered_count >= 2:
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, "open"]), "trend_unrecovered_2d"
        elif rule.trend_exit_mode == "bearish_vol":
            if (
                float(row["close"]) < float(row["open"])
                and float(row["volume"]) >= float(x.iloc[i - 1]["volume"]) * 1.3
            ):
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, "open"]), "bearish_vol"

    return max_exit_idx, float(x.iloc[max_exit_idx]["close"]), f"hold_{rule.max_hold_days}_close"


def build_trade_table(signals: pd.DataFrame, dfs: Dict[str, pd.DataFrame], rule: ExitRule) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rec in signals.itertuples(index=False):
        x = dfs[rec.code]
        exit_idx, exit_price, reason = _exit_trade(x, int(rec.signal_idx), rule)
        entry_open = float(rec.entry_open)
        ret = exit_price / entry_open - 1.0
        path = x.iloc[int(rec.entry_idx) : exit_idx + 1].copy()
        max_favorable = float(path["high"].max() / entry_open - 1.0)
        max_adverse = float(path["low"].min() / entry_open - 1.0)
        rows.append(
            {
                "code": rec.code,
                "signal_date": rec.signal_date,
                "entry_date": rec.entry_date,
                "exit_date": x.at[exit_idx, "date"],
                "entry_open": entry_open,
                "exit_price": exit_price,
                "return": ret,
                "reason": reason,
                "type1": bool(rec.type1),
                "type4": bool(rec.type4),
                "sort_score": float(rec.sort_score),
                "max_favorable": max_favorable,
                "max_adverse": max_adverse,
            }
        )
    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "sample_count": 0,
            "success_rate": float("nan"),
            "avg_return": float("nan"),
            "avg_max_favorable": float("nan"),
            "avg_max_adverse": float("nan"),
        }
    wins = (trades["return"] > 0).mean()
    return {
        "sample_count": int(len(trades)),
        "success_rate": float(wins),
        "avg_return": float(trades["return"].mean()),
        "avg_max_favorable": float(trades["max_favorable"].mean()),
        "avg_max_adverse": float(trades["max_adverse"].mean()),
    }


def simulate_portfolio(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "annual_return": float("nan"),
            "max_drawdown": float("nan"),
            "final_equity": float("nan"),
            "trade_count": 0,
            "success_rate": float("nan"),
            "avg_return": float("nan"),
            "max_losing_streak": 0,
        }

    cash = INITIAL_CAPITAL
    positions: Dict[str, Dict[str, object]] = {}
    trade_records: List[Dict[str, object]] = []
    dates = sorted(set(trades["entry_date"]).union(set(trades["exit_date"])))
    trades_by_entry = {d: g.sort_values("sort_score", ascending=False).to_dict("records") for d, g in trades.groupby("entry_date")}
    trades_by_exit = {d: g.to_dict("records") for d, g in trades.groupby("exit_date")}
    equity_curve = []

    for current_date in dates:
        for tr in trades_by_exit.get(current_date, []):
            code = tr["code"]
            if code in positions:
                pos = positions.pop(code)
                proceeds = pos["shares"] * tr["exit_price"]
                cash += proceeds
                trade_records.append({**tr, "capital_return": tr["exit_price"] / pos["entry_price"] - 1.0})

        if current_date in trades_by_entry:
            available_slots = MAX_POSITIONS - len(positions)
            if available_slots > 0:
                new_entries = [tr for tr in trades_by_entry[current_date] if tr["code"] not in positions][:available_slots]
                if new_entries:
                    alloc_per_pos = cash / len(new_entries)
                    for tr in new_entries:
                        price = tr["entry_open"]
                        if price <= 0:
                            continue
                        shares = alloc_per_pos / price
                        cash -= shares * price
                        positions[tr["code"]] = {"shares": shares, "entry_price": price}

        equity = cash
        for code, pos in positions.items():
            row = trades[(trades["code"] == code) & (trades["entry_date"] <= current_date) & (trades["exit_date"] >= current_date)]
            if row.empty:
                continue
            current_price = float(row.iloc[0]["entry_open"])
            equity += pos["shares"] * current_price
        equity_curve.append({"date": current_date, "equity": equity})

    eq = pd.DataFrame(equity_curve).sort_values("date")
    if eq.empty:
        return {
            "annual_return": float("nan"),
            "max_drawdown": float("nan"),
            "final_equity": float("nan"),
            "trade_count": len(trade_records),
            "success_rate": float("nan"),
            "avg_return": float("nan"),
            "max_losing_streak": 0,
        }
    running_max = eq["equity"].cummax()
    drawdown = eq["equity"] / running_max - 1.0
    total_days = max(len(eq), 1)
    annual_return = float((eq.iloc[-1]["equity"] / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / total_days) - 1.0)
    cap_returns = [r["capital_return"] for r in trade_records]
    losing_streak = 0
    max_losing_streak = 0
    for r in cap_returns:
        if r <= 0:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0
    return {
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()),
        "final_equity": float(eq.iloc[-1]["equity"]),
        "trade_count": len(trade_records),
        "success_rate": float(np.mean(np.array(cap_returns) > 0)) if cap_returns else float("nan"),
        "avg_return": float(np.mean(cap_returns)) if cap_returns else float("nan"),
        "max_losing_streak": int(max_losing_streak),
    }


def load_all_data() -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    paths = sorted(DATA_DIR.glob("*.txt"))
    for idx, path in enumerate(paths, start=1):
        df = load_one_csv(path)
        if df is None:
            continue
        dfs[path.stem] = add_base_features(df)
        if idx % 500 == 0:
            print(f"数据加载进度: {idx}/{len(paths)}")
    return dfs


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = load_all_data()
    params_baseline = SignalParams(
        ret1_min=0.04,
        j_max=90.0,
        upper_shadow_body_ratio=0.3,
        type1_near_ratio=1.02,
        type1_j_rank20_max=0.10,
        type4_touch_ratio=1.01,
    )

    baseline_signals = precompute_signal_rows(dfs, params_baseline)
    baseline_signals.to_csv(RESULT_DIR / "baseline_signals.csv", index=False)

    exit_rows = []
    for rule in EXIT_RULES:
        trades = build_trade_table(baseline_signals, dfs, rule)
        for tag_name, mask in {
            "combined": pd.Series(True, index=trades.index),
            "type1_only": trades["type1"] & (~trades["type4"]),
            "type4_only": trades["type4"] & (~trades["type1"]),
            "overlap": trades["type1"] & trades["type4"],
        }.items():
            subset = trades[mask].copy()
            s = summarize_trades(subset)
            p = simulate_portfolio(subset)
            exit_rows.append(
                {
                    "exit_rule": rule.name,
                    "tag": tag_name,
                    **s,
                    **p,
                }
            )
    exit_df = pd.DataFrame(exit_rows).sort_values(["tag", "annual_return"], ascending=[True, False])
    exit_df.to_csv(RESULT_DIR / "exit_comparison.csv", index=False)

    best_combined_row = exit_df[exit_df["tag"] == "combined"].sort_values(
        ["annual_return", "success_rate", "avg_return"], ascending=[False, False, False]
    ).iloc[0]
    best_exit_name = str(best_combined_row["exit_rule"])
    best_rule = next(rule for rule in EXIT_RULES if rule.name == best_exit_name)

    param_grid = [
        SignalParams(*vals)
        for vals in [
            (ret1_min, j_max, upper, near, jr, touch)
            for ret1_min in RET1_MIN_VALUES
            for j_max in J_MAX_VALUES
            for upper in UPPER_SHADOW_BODY_RATIO_VALUES
            for near in TYPE1_NEAR_RATIO_VALUES
            for jr in TYPE1_J_RANK20_MAX_VALUES
            for touch in TYPE4_TOUCH_RATIO_VALUES
        ]
    ]

    param_rows = []
    total = len(param_grid)
    for idx, params in enumerate(param_grid, start=1):
        signals = precompute_signal_rows(dfs, params)
        trades = build_trade_table(signals, dfs, best_rule)
        s = summarize_trades(trades)
        p = simulate_portfolio(trades)
        score = (
            0.45 * np.nan_to_num(p["annual_return"], nan=-9.0)
            + 0.25 * np.nan_to_num(s["success_rate"], nan=0.0)
            + 0.20 * np.nan_to_num(s["avg_return"], nan=-9.0)
            - 0.10 * abs(np.nan_to_num(p["max_drawdown"], nan=-1.0))
        )
        param_rows.append(
            {
                "params": params.name,
                "ret1_min": params.ret1_min,
                "j_max": params.j_max,
                "upper_shadow_body_ratio": params.upper_shadow_body_ratio,
                "type1_near_ratio": params.type1_near_ratio,
                "type1_j_rank20_max": params.type1_j_rank20_max,
                "type4_touch_ratio": params.type4_touch_ratio,
                "score": score,
                **s,
                **p,
            }
        )
        if idx % 20 == 0 or idx == total:
            print(f"参数搜索进度: {idx}/{total}")
    param_df = pd.DataFrame(param_rows).sort_values(
        ["score", "annual_return", "success_rate"], ascending=[False, False, False]
    )
    param_df.to_csv(RESULT_DIR / "parameter_search.csv", index=False)

    summary = {
        "baseline_signal_count": int(len(baseline_signals)),
        "best_exit": best_combined_row.to_dict(),
        "best_type1_exit": exit_df[exit_df["tag"] == "type1_only"].iloc[0].to_dict()
        if not exit_df[exit_df["tag"] == "type1_only"].empty
        else None,
        "best_type4_exit": exit_df[exit_df["tag"] == "type4_only"].iloc[0].to_dict()
        if not exit_df[exit_df["tag"] == "type4_only"].empty
        else None,
        "best_params": param_df.iloc[0].to_dict(),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
