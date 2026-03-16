from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b3filter, market_structure_tags, pinfilter  # type: ignore


DATA_DIR = ROOT / "data/forward_data"
RESULT_DIR = ROOT / "results/b3_exit_combo_search_20260314"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1_000_000.0
EPS = 1e-12


@dataclass(frozen=True)
class SingleCombo:
    max_hold_days: int
    profit_rule: str
    profit_param: float
    protect_rule: str
    protect_param: float
    stop_rule: str
    stop_param: float

    @property
    def name(self) -> str:
        return (
            f"h{self.max_hold_days}_pr{self.profit_rule}_{self.profit_param}_"
            f"gr{self.protect_rule}_{self.protect_param}_sr{self.stop_rule}_{self.stop_param}"
        )


@dataclass(frozen=True)
class PartialCombo:
    max_hold_days: int
    first_rule: str
    first_param: float
    second_rule: str
    second_param: float
    stop_rule: str
    stop_param: float

    @property
    def name(self) -> str:
        return (
            f"h{self.max_hold_days}_fr{self.first_rule}_{self.first_param}_"
            f"sr{self.second_rule}_{self.second_param}_st{self.stop_rule}_{self.stop_param}"
        )


@dataclass(frozen=True)
class AccountConfig:
    max_positions: int
    daily_new_limit: int
    daily_budget_frac: float
    position_cap_frac: float
    allocation_mode: str

    @property
    def name(self) -> str:
        return (
            f"pos{self.max_positions}_new{self.daily_new_limit}_"
            f"b{int(self.daily_budget_frac * 100)}_cap{int(self.position_cap_frac * 100)}_"
            f"{self.allocation_mode}"
        )


def _in_sample(d: pd.Timestamp) -> bool:
    return (d < EXCLUDE_START) or (d > EXCLUDE_END)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a.astype(float) / b.astype(float).replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_n_prev_low(x: pd.DataFrame) -> pd.Series:
    values20 = pinfilter.rolling_last_percentile(x["J"], 20)
    values30 = pinfilter.rolling_last_percentile(x["J"], 30)
    out = np.full(len(x), np.nan, dtype=float)

    for rank_series in (values20, values30):
        rank_values = rank_series.astype(float)
        lows = x["low"].astype(float).to_numpy()
        highs = x["high"].astype(float).to_numpy()
        closes = x["close"].astype(float).to_numpy()
        for idx in range(len(x)):
            left = max(0, idx - 79)
            sub_rank = rank_values.iloc[left : idx + 1].reset_index(drop=True)
            zones = pinfilter.identify_low_zones(sub_rank <= 0.10)
            if len(zones) < 2:
                continue
            z1, z2 = zones[-2], zones[-1]
            z1_start, z1_end = left + z1[0], left + z1[1]
            z2_start, z2_end = left + z2[0], left + z2[1]
            first_low = float(np.min(lows[z1_start : z1_end + 1]))
            second_low = float(np.min(lows[z2_start : z2_end + 1]))
            if not (second_low > first_low):
                continue
            mid_left = z1_end + 1
            mid_right = z2_start - 1
            if mid_right < mid_left:
                continue
            rebound_high = float(np.max(highs[mid_left : mid_right + 1]))
            if closes[idx] > rebound_high:
                if not np.isfinite(out[idx]) or second_low > out[idx]:
                    out[idx] = second_low
    return pd.Series(out, index=x.index)


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    x = b3filter.add_features(df).copy()
    x["ma5"] = x["close"].rolling(5).mean()
    x["ma10"] = x["close"].rolling(10).mean()
    x["ma20"] = x["close"].rolling(20).mean()

    prev_close = x["close"].shift(1)
    tr = pd.concat(
        [
            (x["high"] - x["low"]).abs(),
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr14"] = tr.rolling(14).mean()
    x["atr14_pct"] = _safe_div(x["atr14"], x["close"])

    base_df = market_structure_tags.add_base_features(df)
    dist_df = market_structure_tags.add_distribution_labels(base_df)
    x["distribution_point"] = dist_df["point_any"].astype(bool).to_numpy()
    x["distribution_zone_end"] = dist_df["zone_end"].astype(bool).to_numpy()

    struct_input = pd.DataFrame(
        {
            "open": x["open"].astype(float),
            "high": x["high"].astype(float),
            "low": x["low"].astype(float),
            "close": x["close"].astype(float),
            "volume": x["volume"].astype(float),
            "trend_line": x["trend_line"].astype(float),
            "long_line": x["long_line"].astype(float),
            "J": x["J"].astype(float),
        }
    )
    struct_df = pinfilter.add_structure_tags(struct_input)
    x["n_up_any"] = struct_df["n_up_any"].astype(bool).to_numpy()
    x["n_prev_low"] = compute_n_prev_low(struct_df)

    x["platform_low_10"] = x["low"].shift(1).rolling(10).min()
    x["platform_low_20"] = x["low"].shift(1).rolling(20).min()
    x["platform_low_30"] = x["low"].shift(1).rolling(30).min()

    x["ret10_prev"] = _safe_div(x["close"].shift(1), x["close"].shift(11)) - 1.0
    x["dyn_tp_pct"] = (
        0.04
        + 1.80 * x["atr14_pct"].fillna(0.0)
        + 0.25 * np.maximum(x["ret10_prev"].fillna(0.0), 0.0)
        + 0.02 * x["prev_b2_type4"].astype(float)
        + 0.01 * x["prev_b2_type1"].astype(float)
    ).clip(lower=0.05, upper=0.18)
    x["dyn_sl_pct"] = (0.02 + 1.20 * x["atr14_pct"].fillna(0.0)).clip(lower=0.03, upper=0.10)
    return x


def load_all_data() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    data_map: Dict[str, pd.DataFrame] = {}
    signals: List[Dict[str, object]] = []
    paths = sorted(DATA_DIR.glob("*.txt"))
    for idx, path in enumerate(paths, start=1):
        df = b3filter.load_one_csv(str(path))
        if df is None or df.empty:
            continue
        x = add_extra_features(df)
        code = path.stem
        data_map[code] = x
        signal_mask = x["b3_signal"].fillna(False).to_numpy(dtype=bool)
        signal_idxs = np.flatnonzero(signal_mask)
        for signal_idx in signal_idxs:
            entry_idx = signal_idx + 1
            if entry_idx >= len(x):
                continue
            signal_date = pd.Timestamp(x.at[signal_idx, "date"])
            entry_date = pd.Timestamp(x.at[entry_idx, "date"])
            if not _in_sample(signal_date):
                continue
            signal_low = float(x.at[signal_idx, "low"])
            entry_low = float(x.at[entry_idx, "low"])
            entry_open = float(x.at[entry_idx, "open"])
            if not np.isfinite(entry_open) or entry_open <= 0:
                continue
            n_prev_low = float(x.at[signal_idx, "n_prev_low"]) if np.isfinite(x.at[signal_idx, "n_prev_low"]) else np.nan
            signals.append(
                {
                    "code": code,
                    "signal_idx": int(signal_idx),
                    "signal_date": signal_date,
                    "entry_idx": int(entry_idx),
                    "entry_date": entry_date,
                    "entry_open": entry_open,
                    "signal_low": signal_low,
                    "entry_low": entry_low,
                    "sort_score": float(x.at[signal_idx, "b3_score"]),
                    "prev_b2_type1": bool(x.at[signal_idx, "prev_b2_type1"]),
                    "prev_b2_type4": bool(x.at[signal_idx, "prev_b2_type4"]),
                    "dyn_tp_pct": float(x.at[signal_idx, "dyn_tp_pct"]),
                    "dyn_sl_pct": float(x.at[signal_idx, "dyn_sl_pct"]),
                    "n_prev_low": n_prev_low,
                    "platform_low_10": float(x.at[signal_idx, "platform_low_10"]) if np.isfinite(x.at[signal_idx, "platform_low_10"]) else np.nan,
                    "platform_low_20": float(x.at[signal_idx, "platform_low_20"]) if np.isfinite(x.at[signal_idx, "platform_low_20"]) else np.nan,
                    "platform_low_30": float(x.at[signal_idx, "platform_low_30"]) if np.isfinite(x.at[signal_idx, "platform_low_30"]) else np.nan,
                }
            )
        if idx % 500 == 0:
            print({"load_progress": idx, "signals": len(signals)}, flush=True)

    signal_df = pd.DataFrame(signals).sort_values(["entry_date", "code"]).reset_index(drop=True) if signals else pd.DataFrame()
    return data_map, signal_df


def _next_open_exit(x: pd.DataFrame, idx: int, reason: str) -> Tuple[int, float, str]:
    next_idx = idx + 1
    if next_idx < len(x):
        return next_idx, float(x.at[next_idx, "open"]), reason
    return idx, float(x.at[idx, "close"]), reason + "_fallback_close"


def _profit_triggered(combo: SingleCombo, row: pd.Series, highest_close: float, entry_open: float, signal: pd.Series) -> bool:
    if combo.profit_rule == "none":
        return False
    if combo.profit_rule == "fixed_tp":
        return float(row["high"]) >= entry_open * (1.0 + combo.profit_param)
    if combo.profit_rule == "close_drawdown":
        return highest_close > entry_open and float(row["close"]) <= highest_close * (1.0 - combo.profit_param)
    if combo.profit_rule == "dynamic_tp":
        return float(row["high"]) >= entry_open * (1.0 + float(signal["dyn_tp_pct"]))
    return False


def _protect_triggered(rule: str, param: float, row: pd.Series, signal: pd.Series) -> bool:
    close = float(row["close"])
    if rule == "none":
        return False
    if rule == "ma_break":
        ma_value = float(row[f"ma{int(param)}"])
        return np.isfinite(ma_value) and close < ma_value
    if rule == "trend_break":
        return close < float(row["trend_line"])
    if rule == "n_low":
        n_low = float(signal["n_prev_low"])
        return np.isfinite(n_low) and close < n_low
    if rule == "platform_break":
        level = float(signal[f"platform_low_{int(param)}"])
        return np.isfinite(level) and close < level
    if rule == "abnormal_vol":
        return bool(row["distribution_point"]) or bool(row["distribution_zone_end"])
    return False


def _stop_triggered(rule: str, param: float, row: pd.Series, signal: pd.Series, entry_open: float) -> bool:
    close = float(row["close"])
    low = float(row["low"])
    if rule == "signal_low":
        return low <= float(signal["signal_low"])
    if rule == "entry_low":
        return low <= float(signal["entry_low"])
    if rule == "fixed_sl":
        return low <= entry_open * (1.0 - param)
    if rule == "dynamic_sl":
        dyn_stop = max(
            float(signal["signal_low"]),
            float(signal["entry_low"]),
            entry_open * (1.0 - float(signal["dyn_sl_pct"])),
        )
        return low <= dyn_stop
    if rule == "ma_break":
        ma_value = float(row[f"ma{int(param)}"])
        return np.isfinite(ma_value) and close < ma_value
    if rule == "trend_break":
        return close < float(row["trend_line"])
    if rule == "n_low":
        n_low = float(signal["n_prev_low"])
        return np.isfinite(n_low) and close < n_low
    if rule == "platform_break":
        level = float(signal[f"platform_low_{int(param)}"])
        return np.isfinite(level) and close < level
    return False


def evaluate_single(signal: pd.Series, x: pd.DataFrame, combo: SingleCombo) -> Dict[str, object]:
    entry_idx = int(signal["entry_idx"])
    entry_open = float(signal["entry_open"])
    max_exit_idx = min(entry_idx + combo.max_hold_days, len(x) - 1)
    highest_close = max(entry_open, float(x.at[entry_idx, "close"]))

    for i in range(entry_idx + 1, max_exit_idx + 1):
        row = x.iloc[i]
        highest_close = max(highest_close, float(row["close"]))

        if _stop_triggered(combo.stop_rule, combo.stop_param, row, signal, entry_open):
            exit_idx, exit_price, reason = _next_open_exit(x, i, f"stop_{combo.stop_rule}_{combo.stop_param}")
            break
        elif _profit_triggered(combo, row, highest_close, entry_open, signal):
            exit_idx, exit_price, reason = _next_open_exit(x, i, f"profit_{combo.profit_rule}_{combo.profit_param}")
            break
        elif _protect_triggered(combo.protect_rule, combo.protect_param, row, signal):
            exit_idx, exit_price, reason = _next_open_exit(x, i, f"protect_{combo.protect_rule}_{combo.protect_param}")
            break
    else:
        exit_idx, exit_price, reason = max_exit_idx, float(x.at[max_exit_idx, "close"]), f"hold_{combo.max_hold_days}_close"

    path = x.iloc[entry_idx : exit_idx + 1]
    trade_return = exit_price / entry_open - 1.0
    return {
        "code": signal["code"],
        "signal_date": signal["signal_date"],
        "entry_date": signal["entry_date"],
        "exit_date": pd.Timestamp(x.at[exit_idx, "date"]),
        "entry_price": entry_open,
        "exit_price": exit_price,
        "return": trade_return,
        "reason": reason,
        "sort_score": float(signal["sort_score"]),
        "max_favorable": float(path["high"].max() / entry_open - 1.0),
        "max_adverse": float(path["low"].min() / entry_open - 1.0),
        "holding_days": int(exit_idx - entry_idx),
    }


def evaluate_partial(signal: pd.Series, x: pd.DataFrame, combo: PartialCombo) -> Dict[str, object]:
    entry_idx = int(signal["entry_idx"])
    entry_open = float(signal["entry_open"])
    max_exit_idx = min(entry_idx + combo.max_hold_days, len(x) - 1)
    highest_close = max(entry_open, float(x.at[entry_idx, "close"]))

    first_exit_idx = None
    first_exit_price = None
    first_reason = None

    for i in range(entry_idx + 1, max_exit_idx + 1):
        row = x.iloc[i]
        highest_close = max(highest_close, float(row["close"]))

        if _stop_triggered(combo.stop_rule, combo.stop_param, row, signal, entry_open):
            exit_idx, exit_price, reason = _next_open_exit(x, i, f"stop_{combo.stop_rule}_{combo.stop_param}")
            path = x.iloc[entry_idx : exit_idx + 1]
            trade_return = exit_price / entry_open - 1.0
            return {
                "code": signal["code"],
                "signal_date": signal["signal_date"],
                "entry_date": signal["entry_date"],
                "exit_date": pd.Timestamp(x.at[exit_idx, "date"]),
                "entry_price": entry_open,
                "exit_price": exit_price,
                "return": trade_return,
                "reason": reason,
                "sort_score": float(signal["sort_score"]),
                "max_favorable": float(path["high"].max() / entry_open - 1.0),
                "max_adverse": float(path["low"].min() / entry_open - 1.0),
                "holding_days": int(exit_idx - entry_idx),
                "leg1_exit_date": pd.NaT,
                "leg2_exit_date": pd.Timestamp(x.at[exit_idx, "date"]),
                "leg1_price": np.nan,
                "leg2_price": exit_price,
            }

        trigger_first = False
        if combo.first_rule == "fixed_tp":
            trigger_first = float(row["high"]) >= entry_open * (1.0 + combo.first_param)
        elif combo.first_rule == "close_drawdown":
            trigger_first = highest_close > entry_open and float(row["close"]) <= highest_close * (1.0 - combo.first_param)
        elif combo.first_rule == "dynamic_tp":
            trigger_first = float(row["high"]) >= entry_open * (1.0 + float(signal["dyn_tp_pct"]))
        if trigger_first:
            first_exit_idx, first_exit_price, first_reason = _next_open_exit(
                x, i, f"partial1_{combo.first_rule}_{combo.first_param}"
            )
            break

    if first_exit_idx is None:
        exit_idx = max_exit_idx
        exit_price = float(x.at[max_exit_idx, "close"])
        path = x.iloc[entry_idx : exit_idx + 1]
        trade_return = exit_price / entry_open - 1.0
        return {
            "code": signal["code"],
            "signal_date": signal["signal_date"],
            "entry_date": signal["entry_date"],
            "exit_date": pd.Timestamp(x.at[exit_idx, "date"]),
            "entry_price": entry_open,
            "exit_price": exit_price,
            "return": trade_return,
            "reason": f"hold_{combo.max_hold_days}_close",
            "sort_score": float(signal["sort_score"]),
            "max_favorable": float(path["high"].max() / entry_open - 1.0),
            "max_adverse": float(path["low"].min() / entry_open - 1.0),
            "holding_days": int(exit_idx - entry_idx),
            "leg1_exit_date": pd.NaT,
            "leg2_exit_date": pd.Timestamp(x.at[exit_idx, "date"]),
            "leg1_price": np.nan,
            "leg2_price": exit_price,
        }

    highest_close_after = max(highest_close, float(x.at[first_exit_idx, "close"]) if first_exit_idx < len(x) else highest_close)
    for i in range(first_exit_idx, max_exit_idx + 1):
        row = x.iloc[i]
        highest_close_after = max(highest_close_after, float(row["close"]))
        if _stop_triggered(combo.stop_rule, combo.stop_param, row, signal, entry_open):
            second_exit_idx, second_exit_price, second_reason = _next_open_exit(
                x, i, f"stop_{combo.stop_rule}_{combo.stop_param}"
            )
            break

        second_trigger = False
        if combo.second_rule == "time_only":
            second_trigger = False
        elif combo.second_rule == "close_drawdown":
            second_trigger = highest_close_after > entry_open and float(row["close"]) <= highest_close_after * (1.0 - combo.second_param)
        elif combo.second_rule == "dynamic_tp":
            second_trigger = float(row["high"]) >= entry_open * (1.0 + float(signal["dyn_tp_pct"]))
        else:
            second_trigger = _protect_triggered(combo.second_rule, combo.second_param, row, signal)
        if second_trigger:
            second_exit_idx, second_exit_price, second_reason = _next_open_exit(
                x, i, f"partial2_{combo.second_rule}_{combo.second_param}"
            )
            break
    else:
        second_exit_idx = max_exit_idx
        second_exit_price = float(x.at[max_exit_idx, "close"])
        second_reason = f"hold_{combo.max_hold_days}_close"

    leg1_ret = float(first_exit_price) / entry_open - 1.0
    leg2_ret = float(second_exit_price) / entry_open - 1.0
    final_return = 0.5 * leg1_ret + 0.5 * leg2_ret
    path = x.iloc[entry_idx : second_exit_idx + 1]
    return {
        "code": signal["code"],
        "signal_date": signal["signal_date"],
        "entry_date": signal["entry_date"],
        "exit_date": pd.Timestamp(x.at[second_exit_idx, "date"]),
        "entry_price": entry_open,
        "exit_price": 0.5 * float(first_exit_price) + 0.5 * float(second_exit_price),
        "return": final_return,
        "reason": f"{first_reason}+{second_reason}",
        "sort_score": float(signal["sort_score"]),
        "max_favorable": float(path["high"].max() / entry_open - 1.0),
        "max_adverse": float(path["low"].min() / entry_open - 1.0),
        "holding_days": int(second_exit_idx - entry_idx),
        "leg1_exit_date": pd.Timestamp(x.at[first_exit_idx, "date"]),
        "leg2_exit_date": pd.Timestamp(x.at[second_exit_idx, "date"]),
        "leg1_price": float(first_exit_price),
        "leg2_price": float(second_exit_price),
    }


def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "sample_count": 0,
            "success_rate": np.nan,
            "avg_return": np.nan,
            "avg_max_favorable": np.nan,
            "avg_max_adverse": np.nan,
            "avg_holding_days": np.nan,
        }
    return {
        "sample_count": int(len(trades)),
        "success_rate": float((trades["return"] > 0).mean()),
        "avg_return": float(trades["return"].mean()),
        "avg_max_favorable": float(trades["max_favorable"].mean()),
        "avg_max_adverse": float(trades["max_adverse"].mean()),
        "avg_holding_days": float(trades["holding_days"].mean()),
    }


def run_account_backtest(trades: pd.DataFrame, data_map: Dict[str, pd.DataFrame], config: AccountConfig) -> Dict[str, float]:
    if trades.empty:
        return {
            "trade_count": 0,
            "success_rate": np.nan,
            "avg_return": np.nan,
            "annual_return": np.nan,
            "max_drawdown": np.nan,
            "final_equity": np.nan,
            "equity_days": 0,
        }

    trade_rows = trades.sort_values(["entry_date", "sort_score"], ascending=[True, False]).to_dict("records")
    trades_by_entry: Dict[pd.Timestamp, List[dict]] = {}
    for row in trade_rows:
        trades_by_entry.setdefault(pd.Timestamp(row["entry_date"]), []).append(row)

    close_cache = {code: df.set_index("date")["close"] for code, df in data_map.items()}
    start_date = pd.Timestamp(trades["entry_date"].min())
    end_date = pd.Timestamp(trades["exit_date"].max())
    all_dates = sorted(set(pd.concat([df["date"] for df in data_map.values() if not df.empty]).tolist()))
    all_dates = [d for d in all_dates if start_date <= pd.Timestamp(d) <= end_date]

    cash = INITIAL_CAPITAL
    positions: Dict[str, dict] = {}
    equity_curve: List[Dict[str, object]] = []
    trade_outcomes: List[float] = []

    for current_date in all_dates:
        current_date = pd.Timestamp(current_date)

        # exits first
        to_delete = []
        for pid, pos in positions.items():
            if pos["leg1_pending"] and pd.Timestamp(pos["leg1_exit_date"]) == current_date:
                shares = pos["shares"] * 0.5
                cash += shares * float(pos["leg1_price"])
                pos["shares"] -= shares
                pos["leg1_pending"] = False
            if pd.Timestamp(pos["leg2_exit_date"]) == current_date:
                cash += pos["shares"] * float(pos["leg2_price"])
                trade_outcomes.append(float(pos["total_return"]))
                to_delete.append(pid)
        for pid in to_delete:
            del positions[pid]

        # mark equity before entries
        equity = cash
        for pid, pos in positions.items():
            code = pos["code"]
            px = float(close_cache[code].get(current_date, pos["entry_price"]))
            equity += pos["shares"] * px

        # entries
        entries = trades_by_entry.get(current_date, [])
        if entries:
            available_slots = max(config.max_positions - len(positions), 0)
            if available_slots > 0:
                to_take = entries[: min(available_slots, config.daily_new_limit)]
                if to_take:
                    investable = min(cash, equity * config.daily_budget_frac)
                    if config.allocation_mode == "score_weighted":
                        total_score = sum(max(float(r["sort_score"]), EPS) for r in to_take)
                    else:
                        total_score = float(len(to_take))
                    for idx, row in enumerate(to_take):
                        if cash <= 0:
                            break
                        weight = (
                            max(float(row["sort_score"]), EPS) / total_score
                            if config.allocation_mode == "score_weighted"
                            else 1.0 / len(to_take)
                        )
                        target_cash = min(investable * weight, equity * config.position_cap_frac, cash)
                        shares = target_cash / float(row["entry_price"])
                        if shares <= 0:
                            continue
                        pid = f"{row['code']}|{row['entry_date']}|{idx}"
                        positions[pid] = {
                            "code": row["code"],
                            "entry_price": float(row["entry_price"]),
                            "shares": shares,
                            "leg1_exit_date": row.get("leg1_exit_date"),
                            "leg2_exit_date": row["leg2_exit_date"] if "leg2_exit_date" in row else row["exit_date"],
                            "leg1_price": row.get("leg1_price", np.nan),
                            "leg2_price": row.get("leg2_price", row["exit_price"]),
                            "leg1_pending": pd.notna(row.get("leg1_exit_date")),
                            "total_return": float(row["return"]),
                        }
                        cash -= shares * float(row["entry_price"])

        # mark equity at close
        equity = cash
        for pid, pos in positions.items():
            code = pos["code"]
            px = float(close_cache[code].get(current_date, pos["entry_price"]))
            equity += pos["shares"] * px
        equity_curve.append({"date": current_date, "equity": equity})

    eq = pd.DataFrame(equity_curve).sort_values("date")
    if eq.empty:
        return {
            "trade_count": 0,
            "success_rate": np.nan,
            "avg_return": np.nan,
            "annual_return": np.nan,
            "max_drawdown": np.nan,
            "final_equity": np.nan,
            "equity_days": 0,
        }
    running_max = eq["equity"].cummax()
    drawdown = eq["equity"] / running_max - 1.0
    total_days = len(eq)
    annual_return = float((eq.iloc[-1]["equity"] / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / max(total_days, 1)) - 1.0)
    return {
        "trade_count": int(len(trades)),
        "success_rate": float((trades["return"] > 0).mean()),
        "avg_return": float(trades["return"].mean()),
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()),
        "final_equity": float(eq.iloc[-1]["equity"]),
        "equity_days": int(total_days),
    }


def build_single_combos() -> List[SingleCombo]:
    profit_rules = [
        ("none", 0.0),
        ("fixed_tp", 0.05),
        ("fixed_tp", 0.08),
        ("fixed_tp", 0.10),
        ("fixed_tp", 0.15),
        ("close_drawdown", 0.08),
        ("close_drawdown", 0.12),
        ("dynamic_tp", 0.0),
    ]
    protect_rules = [
        ("none", 0.0),
        ("ma_break", 5.0),
        ("ma_break", 10.0),
        ("ma_break", 20.0),
        ("trend_break", 0.0),
        ("n_low", 0.0),
        ("platform_break", 10.0),
        ("platform_break", 20.0),
        ("platform_break", 30.0),
        ("abnormal_vol", 0.0),
    ]
    stop_rules = [
        ("signal_low", 0.0),
        ("entry_low", 0.0),
        ("fixed_sl", 0.05),
        ("fixed_sl", 0.08),
        ("dynamic_sl", 0.0),
        ("ma_break", 5.0),
        ("ma_break", 10.0),
        ("ma_break", 20.0),
        ("trend_break", 0.0),
        ("n_low", 0.0),
        ("platform_break", 20.0),
        ("platform_break", 30.0),
    ]
    combos: List[SingleCombo] = []
    for hold in [5, 10, 20, 30]:
        for pr, pv in profit_rules:
            for gr, gv in protect_rules:
                for sr, sv in stop_rules:
                    combos.append(SingleCombo(hold, pr, pv, gr, gv, sr, sv))
    return combos


def build_partial_combos() -> List[PartialCombo]:
    first_rules = [
        ("fixed_tp", 0.05),
        ("fixed_tp", 0.08),
        ("fixed_tp", 0.10),
        ("close_drawdown", 0.08),
        ("dynamic_tp", 0.0),
    ]
    second_rules = [
        ("time_only", 0.0),
        ("ma_break", 10.0),
        ("ma_break", 20.0),
        ("trend_break", 0.0),
        ("n_low", 0.0),
        ("platform_break", 20.0),
        ("abnormal_vol", 0.0),
        ("close_drawdown", 0.08),
        ("dynamic_tp", 0.0),
    ]
    stop_rules = [
        ("signal_low", 0.0),
        ("entry_low", 0.0),
        ("fixed_sl", 0.05),
        ("dynamic_sl", 0.0),
        ("ma_break", 20.0),
        ("trend_break", 0.0),
        ("n_low", 0.0),
        ("platform_break", 20.0),
    ]
    combos: List[PartialCombo] = []
    for hold in [10, 20, 30]:
        for fr, fp in first_rules:
            for sr, sp in second_rules:
                for st, sv in stop_rules:
                    combos.append(PartialCombo(hold, fr, fp, sr, sp, st, sv))
    return combos


def build_account_configs() -> List[AccountConfig]:
    configs = []
    for max_positions in [3, 5, 8, 10]:
        for daily_new_limit in [1, 2, 3]:
            for daily_budget_frac in [0.30, 0.50, 1.00]:
                for position_cap_frac in [0.10, 0.15, 0.20]:
                    for allocation_mode in ["equal", "score_weighted"]:
                        configs.append(
                            AccountConfig(
                                max_positions=max_positions,
                                daily_new_limit=daily_new_limit,
                                daily_budget_frac=daily_budget_frac,
                                position_cap_frac=position_cap_frac,
                                allocation_mode=allocation_mode,
                            )
                        )
    return configs


def main() -> None:
    data_map, signal_df = load_all_data()
    if signal_df.empty:
        raise SystemExit("没有B3信号，无法回测")

    signal_df.to_csv(RESULT_DIR / "signals.csv", index=False, encoding="utf-8")
    (RESULT_DIR / "signal_summary.json").write_text(
        json.dumps(
            {
                "signal_count": int(len(signal_df)),
                "code_count": int(signal_df["code"].nunique()),
                "date_count": int(signal_df["signal_date"].nunique()),
                "date_min": str(signal_df["signal_date"].min().date()),
                "date_max": str(signal_df["signal_date"].max().date()),
                "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    single_combos = build_single_combos()
    single_rows: List[Dict[str, object]] = []
    for idx, combo in enumerate(single_combos, start=1):
        trade_rows = [
            evaluate_single(signal, data_map[str(signal["code"])], combo)
            for _, signal in signal_df.iterrows()
        ]
        trades = pd.DataFrame(trade_rows)
        metrics = summarize_trades(trades)
        row = {
            "combo_type": "single",
            "combo_name": combo.name,
            **asdict(combo),
            **metrics,
        }
        single_rows.append(row)
        if idx % 50 == 0 or idx == len(single_combos):
            pd.DataFrame(single_rows).to_csv(
                RESULT_DIR / "single_exit_results_checkpoint.csv", index=False, encoding="utf-8"
            )
            print({"single_progress": idx, "total": len(single_combos)}, flush=True)

    single_df = pd.DataFrame(single_rows)
    single_df["score"] = (
        0.45 * single_df["avg_return"].fillna(-9.0)
        + 0.35 * single_df["success_rate"].fillna(0.0)
        + 0.10 * single_df["avg_max_favorable"].fillna(0.0)
        - 0.10 * single_df["avg_max_adverse"].abs().fillna(1.0)
    )
    single_df = single_df.sort_values("score", ascending=False).reset_index(drop=True)
    single_df.to_csv(RESULT_DIR / "single_exit_results.csv", index=False, encoding="utf-8")

    partial_combos = build_partial_combos()
    partial_rows: List[Dict[str, object]] = []
    for idx, combo in enumerate(partial_combos, start=1):
        trade_rows = [
            evaluate_partial(signal, data_map[str(signal["code"])], combo)
            for _, signal in signal_df.iterrows()
        ]
        trades = pd.DataFrame(trade_rows)
        metrics = summarize_trades(trades)
        row = {
            "combo_type": "partial",
            "combo_name": combo.name,
            **asdict(combo),
            **metrics,
        }
        partial_rows.append(row)
        if idx % 50 == 0 or idx == len(partial_combos):
            pd.DataFrame(partial_rows).to_csv(
                RESULT_DIR / "partial_exit_results_checkpoint.csv", index=False, encoding="utf-8"
            )
            print({"partial_progress": idx, "total": len(partial_combos)}, flush=True)

    partial_df = pd.DataFrame(partial_rows)
    partial_df["score"] = (
        0.45 * partial_df["avg_return"].fillna(-9.0)
        + 0.35 * partial_df["success_rate"].fillna(0.0)
        + 0.10 * partial_df["avg_max_favorable"].fillna(0.0)
        - 0.10 * partial_df["avg_max_adverse"].abs().fillna(1.0)
    )
    partial_df = partial_df.sort_values("score", ascending=False).reset_index(drop=True)
    partial_df.to_csv(RESULT_DIR / "partial_exit_results.csv", index=False, encoding="utf-8")

    top_single = single_df.head(12).copy()
    top_partial = partial_df.head(8).copy()
    account_configs = build_account_configs()
    account_rows: List[Dict[str, object]] = []

    for _, row in top_single.iterrows():
        combo = SingleCombo(
            max_hold_days=int(row["max_hold_days"]),
            profit_rule=str(row["profit_rule"]),
            profit_param=float(row["profit_param"]),
            protect_rule=str(row["protect_rule"]),
            protect_param=float(row["protect_param"]),
            stop_rule=str(row["stop_rule"]),
            stop_param=float(row["stop_param"]),
        )
        trade_rows = [evaluate_single(signal, data_map[str(signal["code"])], combo) for _, signal in signal_df.iterrows()]
        trades = pd.DataFrame(trade_rows)
        trades.to_csv(RESULT_DIR / f"trades_single_{combo.name}.csv", index=False, encoding="utf-8")
        for config in account_configs:
            metrics = run_account_backtest(trades, data_map, config)
            account_rows.append(
                {
                    "combo_type": "single",
                    "combo_name": combo.name,
                    "account_name": config.name,
                    **asdict(combo),
                    **asdict(config),
                    **metrics,
                }
            )

    for _, row in top_partial.iterrows():
        combo = PartialCombo(
            max_hold_days=int(row["max_hold_days"]),
            first_rule=str(row["first_rule"]),
            first_param=float(row["first_param"]),
            second_rule=str(row["second_rule"]),
            second_param=float(row["second_param"]),
            stop_rule=str(row["stop_rule"]),
            stop_param=float(row["stop_param"]),
        )
        trade_rows = [evaluate_partial(signal, data_map[str(signal["code"])], combo) for _, signal in signal_df.iterrows()]
        trades = pd.DataFrame(trade_rows)
        trades.to_csv(RESULT_DIR / f"trades_partial_{combo.name}.csv", index=False, encoding="utf-8")
        for config in account_configs:
            metrics = run_account_backtest(trades, data_map, config)
            account_rows.append(
                {
                    "combo_type": "partial",
                    "combo_name": combo.name,
                    "account_name": config.name,
                    **asdict(combo),
                    **asdict(config),
                    **metrics,
                }
            )

    account_df = pd.DataFrame(account_rows)
    account_df["score"] = (
        0.50 * account_df["annual_return"].fillna(-9.0)
        + 0.25 * account_df["success_rate"].fillna(0.0)
        + 0.15 * account_df["avg_return"].fillna(-9.0)
        - 0.10 * account_df["max_drawdown"].abs().fillna(1.0)
    )
    account_df = account_df.sort_values("score", ascending=False).reset_index(drop=True)
    account_df.to_csv(RESULT_DIR / "account_results.csv", index=False, encoding="utf-8")

    summary = {
        "signal_count": int(len(signal_df)),
        "best_single_trade": single_df.iloc[0].to_dict() if not single_df.empty else {},
        "best_partial_trade": partial_df.iloc[0].to_dict() if not partial_df.empty else {},
        "best_account": account_df.iloc[0].to_dict() if not account_df.empty else {},
        "best_success_rate_account": account_df.sort_values("success_rate", ascending=False).iloc[0].to_dict() if not account_df.empty else {},
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
