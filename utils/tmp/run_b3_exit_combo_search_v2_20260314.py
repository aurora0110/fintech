from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b1filter, b3filter, market_structure_tags, pinfilter  # type: ignore


DATA_DIR = ROOT / "data/forward_data"
RESULT_DIR = ROOT / "results/b3_exit_combo_search_v3_20260315"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1_000_000.0
EPS = 1e-12
SCAN_WORKERS = 8


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
        + 0.02 * x["prev_b2_any"].astype(float)
    ).clip(lower=0.05, upper=0.18)
    x["dyn_sl_pct"] = (0.02 + 1.20 * x["atr14_pct"].fillna(0.0)).clip(lower=0.03, upper=0.10)
    return x


def scan_one_file(path_str: str) -> List[Dict[str, object]]:
    path = Path(path_str)
    weekly_ok, _ = b1filter.weekly_screen(str(path))
    if not weekly_ok:
        return []
    df = b3filter.load_one_csv(str(path))
    if df is None or df.empty:
        return []
    x = b3filter.add_features(df)
    signal_idxs = np.flatnonzero(x["b3_signal"].fillna(False).to_numpy(dtype=bool))
    if len(signal_idxs) == 0:
        return []
    code = path.stem
    rows: List[Dict[str, object]] = []
    for signal_idx in signal_idxs:
        entry_idx = signal_idx + 1
        if entry_idx >= len(x):
            continue
        signal_date = pd.Timestamp(x.at[signal_idx, "date"])
        if not _in_sample(signal_date):
            continue
        entry_open = float(x.at[entry_idx, "open"])
        if not np.isfinite(entry_open) or entry_open <= 0:
            continue
        rows.append(
            {
                "code": code,
                "signal_idx": int(signal_idx),
                "signal_date": signal_date,
                "entry_idx": int(entry_idx),
                "entry_date": pd.Timestamp(x.at[entry_idx, "date"]),
            }
        )
    return rows


def collect_signal_rows() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    paths = sorted(DATA_DIR.glob("*.txt"))
    path_strs = [str(p) for p in paths]
    completed = 0
    with ProcessPoolExecutor(max_workers=SCAN_WORKERS) as executor:
        futures = {executor.submit(scan_one_file, path_str): path_str for path_str in path_strs}
        for future in as_completed(futures):
            completed += 1
            rows.extend(future.result())
            if completed % 200 == 0:
                print({"signal_scan_progress": completed, "total": len(path_strs), "signals": len(rows)}, flush=True)
    signal_stub = pd.DataFrame(rows).sort_values(["entry_date", "code"]).reset_index(drop=True) if rows else pd.DataFrame()
    signal_stub.to_csv(RESULT_DIR / "signal_stub.csv", index=False, encoding="utf-8")
    return signal_stub


def load_signal_data(signal_stub: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    if signal_stub.empty:
        return {}, signal_stub

    code_to_events = signal_stub.groupby("code")
    data_map: Dict[str, pd.DataFrame] = {}
    signals: List[Dict[str, object]] = []
    codes = sorted(code_to_events.groups.keys())
    for idx, code in enumerate(codes, start=1):
        path = DATA_DIR / f"{code}.txt"
        df = b3filter.load_one_csv(str(path))
        if df is None or df.empty:
            continue
        x = add_extra_features(df)
        data_map[code] = x
        event_df = code_to_events.get_group(code)
        for _, event in event_df.iterrows():
            signal_idx = int(event["signal_idx"])
            entry_idx = int(event["entry_idx"])
            if signal_idx >= len(x) or entry_idx >= len(x):
                continue
            signals.append(
                {
                    "code": code,
                    "signal_idx": signal_idx,
                    "signal_date": pd.Timestamp(x.at[signal_idx, "date"]),
                    "entry_idx": entry_idx,
                    "entry_date": pd.Timestamp(x.at[entry_idx, "date"]),
                    "entry_open": float(x.at[entry_idx, "open"]),
                    "signal_low": float(x.at[signal_idx, "low"]),
                    "entry_low": float(x.at[entry_idx, "low"]),
                    "sort_score": float(x.at[signal_idx, "b3_score"]),
                    "prev_b2_any": bool(x.at[signal_idx, "prev_b2_any"]),
                    "dyn_tp_pct": float(x.at[signal_idx, "dyn_tp_pct"]),
                    "dyn_sl_pct": float(x.at[signal_idx, "dyn_sl_pct"]),
                    "n_prev_low": float(x.at[signal_idx, "n_prev_low"]) if np.isfinite(x.at[signal_idx, "n_prev_low"]) else np.nan,
                    "platform_low_10": float(x.at[signal_idx, "platform_low_10"]) if np.isfinite(x.at[signal_idx, "platform_low_10"]) else np.nan,
                    "platform_low_20": float(x.at[signal_idx, "platform_low_20"]) if np.isfinite(x.at[signal_idx, "platform_low_20"]) else np.nan,
                    "platform_low_30": float(x.at[signal_idx, "platform_low_30"]) if np.isfinite(x.at[signal_idx, "platform_low_30"]) else np.nan,
                }
            )
        if idx % 200 == 0:
            print({"signal_enrich_progress": idx, "codes": len(codes), "signals": len(signals)}, flush=True)
    signal_df = pd.DataFrame(signals).sort_values(["entry_date", "code"]).reset_index(drop=True) if signals else pd.DataFrame()
    signal_df.to_csv(RESULT_DIR / "signals.csv", index=False, encoding="utf-8")
    return data_map, signal_df


def _next_open_exit(x: pd.DataFrame, idx: int, reason: str) -> Tuple[int, float, str]:
    next_idx = idx + 1
    if next_idx < len(x):
        return next_idx, float(x.at[next_idx, "open"]), reason
    return idx, float(x.at[idx, "close"]), reason + "_fallback_close"


def _profit_triggered(combo: SingleCombo, row: pd.Series, highest_high: float, entry_open: float, signal: pd.Series) -> bool:
    if combo.profit_rule == "none":
        return False
    if combo.profit_rule == "fixed_tp":
        return float(row["high"]) >= entry_open * (1.0 + combo.profit_param)
    if combo.profit_rule == "close_drawdown":
        return highest_high > entry_open and float(row["close"]) <= highest_high * (1.0 - combo.profit_param)
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
    highest_high = max(entry_open, float(x.at[entry_idx, "high"]))

    for i in range(entry_idx + 1, max_exit_idx + 1):
        row = x.iloc[i]
        highest_high = max(highest_high, float(row["high"]))

        if _stop_triggered(combo.stop_rule, combo.stop_param, row, signal, entry_open):
            exit_idx, exit_price, reason = _next_open_exit(x, i, f"stop_{combo.stop_rule}_{combo.stop_param}")
            break
        elif _profit_triggered(combo, row, highest_high, entry_open, signal):
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
    highest_high = max(entry_open, float(x.at[entry_idx, "high"]))

    first_exit_idx = None
    first_exit_price = None
    first_reason = None

    for i in range(entry_idx + 1, max_exit_idx + 1):
        row = x.iloc[i]
        highest_high = max(highest_high, float(row["high"]))

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
            trigger_first = highest_high > entry_open and float(row["close"]) <= highest_high * (1.0 - combo.first_param)
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

    highest_high_after = max(highest_high, float(x.at[first_exit_idx, "high"]) if first_exit_idx < len(x) else highest_high)
    for i in range(first_exit_idx, max_exit_idx + 1):
        row = x.iloc[i]
        highest_high_after = max(highest_high_after, float(row["high"]))
        if _stop_triggered(combo.stop_rule, combo.stop_param, row, signal, entry_open):
            second_exit_idx, second_exit_price, second_reason = _next_open_exit(
                x, i, f"stop_{combo.stop_rule}_{combo.stop_param}"
            )
            break

        second_trigger = False
        if combo.second_rule == "time_only":
            second_trigger = False
        elif combo.second_rule == "close_drawdown":
            second_trigger = highest_high_after > entry_open and float(row["close"]) <= highest_high_after * (1.0 - combo.second_param)
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

    for current_date in all_dates:
        current_date = pd.Timestamp(current_date)

        to_delete = []
        for pid, pos in positions.items():
            if pos["leg1_pending"] and pd.Timestamp(pos["leg1_exit_date"]) == current_date:
                shares = pos["shares"] * 0.5
                cash += shares * float(pos["leg1_price"])
                pos["shares"] -= shares
                pos["leg1_pending"] = False
            if pd.Timestamp(pos["leg2_exit_date"]) == current_date:
                cash += pos["shares"] * float(pos["leg2_price"])
                to_delete.append(pid)
        for pid in to_delete:
            del positions[pid]

        equity = cash
        for pid, pos in positions.items():
            code = pos["code"]
            px = float(close_cache[code].get(current_date, pos["entry_price"]))
            equity += pos["shares"] * px

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
                        }
                        cash -= shares * float(row["entry_price"])

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
    for hold in (5, 10, 20, 30):
        for pr, pp in profit_rules:
            for gr, gp in protect_rules:
                for sr, sp in stop_rules:
                    combos.append(SingleCombo(hold, pr, pp, gr, gp, sr, sp))
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
    for hold in (10, 20, 30):
        for fr, fp in first_rules:
            for sr, sp in second_rules:
                for tr, tp in stop_rules:
                    combos.append(PartialCombo(hold, fr, fp, sr, sp, tr, tp))
    return combos


def build_account_configs() -> List[AccountConfig]:
    configs: List[AccountConfig] = []
    for max_positions in (3, 5, 8, 10):
        for daily_new_limit in (1, 2, 3):
            for daily_budget_frac in (0.30, 0.50, 1.00):
                for position_cap_frac in (0.10, 0.15, 0.20):
                    for allocation_mode in ("equal", "score_weighted"):
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


def run() -> None:
    signal_stub = collect_signal_rows()
    data_map, signal_df = load_signal_data(signal_stub)
    signal_summary = {
        "signal_count": int(len(signal_df)),
        "signal_codes": int(signal_df["code"].nunique()) if not signal_df.empty else 0,
        "signal_days": int(signal_df["signal_date"].nunique()) if not signal_df.empty else 0,
    }
    (RESULT_DIR / "signal_summary.json").write_text(
        json.dumps(signal_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if signal_df.empty:
        print(json.dumps({"status": "no_signals"}, ensure_ascii=False))
        return

    single_combos = build_single_combos()
    single_rows: List[Dict[str, object]] = []
    for idx, combo in enumerate(single_combos, start=1):
        trades = [
            evaluate_single(signal, data_map[str(signal["code"])], combo)
            for _, signal in signal_df.iterrows()
        ]
        trades_df = pd.DataFrame(trades)
        summary = summarize_trades(trades_df)
        summary.update(asdict(combo))
        summary["combo_name"] = combo.name
        single_rows.append(summary)
        if idx % 100 == 0:
            pd.DataFrame(single_rows).to_csv(
                RESULT_DIR / "single_exit_results_checkpoint.csv", index=False, encoding="utf-8"
            )
            print({"single_progress": idx, "total": len(single_combos)}, flush=True)
    single_df = pd.DataFrame(single_rows).sort_values(
        ["avg_return", "success_rate", "avg_max_favorable"],
        ascending=[False, False, False],
    )
    single_df.to_csv(RESULT_DIR / "single_exit_results.csv", index=False, encoding="utf-8")

    partial_combos = build_partial_combos()
    partial_rows: List[Dict[str, object]] = []
    for idx, combo in enumerate(partial_combos, start=1):
        trades = [
            evaluate_partial(signal, data_map[str(signal["code"])], combo)
            for _, signal in signal_df.iterrows()
        ]
        trades_df = pd.DataFrame(trades)
        summary = summarize_trades(trades_df)
        summary.update(asdict(combo))
        summary["combo_name"] = combo.name
        partial_rows.append(summary)
        if idx % 100 == 0:
            pd.DataFrame(partial_rows).to_csv(
                RESULT_DIR / "partial_exit_results_checkpoint.csv", index=False, encoding="utf-8"
            )
            print({"partial_progress": idx, "total": len(partial_combos)}, flush=True)
    partial_df = pd.DataFrame(partial_rows).sort_values(
        ["avg_return", "success_rate", "avg_max_favorable"],
        ascending=[False, False, False],
    )
    partial_df.to_csv(RESULT_DIR / "partial_exit_results.csv", index=False, encoding="utf-8")

    top_single = single_df.head(12).copy()
    top_partial = partial_df.head(12).copy()
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
        trades_df = pd.DataFrame(
            [evaluate_single(signal, data_map[str(signal["code"])], combo) for _, signal in signal_df.iterrows()]
        )
        trades_df.to_csv(RESULT_DIR / f"trades_single_{combo.name}.csv", index=False, encoding="utf-8")
        for config in account_configs:
            metrics = run_account_backtest(trades_df, data_map, config)
            metrics.update({"combo_type": "single", "combo_name": combo.name})
            metrics.update(asdict(combo))
            metrics.update(asdict(config))
            account_rows.append(metrics)
        pd.DataFrame(account_rows).to_csv(RESULT_DIR / "account_results_checkpoint.csv", index=False, encoding="utf-8")
        print({"account_single_done": combo.name}, flush=True)

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
        trades_df = pd.DataFrame(
            [evaluate_partial(signal, data_map[str(signal["code"])], combo) for _, signal in signal_df.iterrows()]
        )
        trades_df.to_csv(RESULT_DIR / f"trades_partial_{combo.name}.csv", index=False, encoding="utf-8")
        for config in account_configs:
            metrics = run_account_backtest(trades_df, data_map, config)
            metrics.update({"combo_type": "partial", "combo_name": combo.name})
            metrics.update(asdict(combo))
            metrics.update(asdict(config))
            account_rows.append(metrics)
        pd.DataFrame(account_rows).to_csv(RESULT_DIR / "account_results_checkpoint.csv", index=False, encoding="utf-8")
        print({"account_partial_done": combo.name}, flush=True)

    account_df = pd.DataFrame(account_rows).sort_values(
        ["annual_return", "avg_return", "success_rate", "max_drawdown"],
        ascending=[False, False, False, False],
    )
    account_df.to_csv(RESULT_DIR / "account_results.csv", index=False, encoding="utf-8")

    summary = {
        "signal_summary": signal_summary,
        "single_top": single_df.head(10).to_dict("records"),
        "partial_top": partial_df.head(10).to_dict("records"),
        "account_top": account_df.head(20).to_dict("records"),
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    run()
