from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import b2filter, brick_filter, pinfilter, stoploss, technical_indicators


DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/n_up_ab_all_strategies_20260313")
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
N_LOOKBACK = 80


@dataclass(frozen=True)
class NVariant:
    use_n: bool
    rank_window: int
    rank_threshold: float

    @property
    def label(self) -> str:
        if not self.use_n:
            return "不加N"
        return f"J{self.rank_window}日分位<={self.rank_threshold:.0%}"


def rolling_last_percentile(series: pd.Series, window: int) -> pd.Series:
    values = series.astype(float)

    def _pct_last(arr: np.ndarray) -> float:
        if len(arr) == 0 or not np.isfinite(arr[-1]):
            return np.nan
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid <= arr[-1]) / len(valid))

    return values.rolling(window, min_periods=window).apply(_pct_last, raw=True)


def identify_low_zones(mask_series: pd.Series) -> List[Tuple[int, int]]:
    mask = mask_series.fillna(False).to_numpy(dtype=bool)
    zones: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            zones.append((start, i - 1))
            start = None
    if start is not None:
        zones.append((start, len(mask) - 1))
    return zones


def build_n_up_feature(df: pd.DataFrame, rank_col: str, rank_threshold: float) -> pd.Series:
    out = np.zeros(len(df), dtype=bool)
    lows = df["low"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    rank_values = df[rank_col].astype(float)

    for idx in range(len(df)):
        left = max(0, idx - N_LOOKBACK + 1)
        sub_rank = rank_values.iloc[left : idx + 1].reset_index(drop=True)
        zones = identify_low_zones(sub_rank <= rank_threshold)
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
            out[idx] = True
    return pd.Series(out, index=df.index)


def add_n_up_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "J" not in x.columns:
        low_9 = x["low"].rolling(9).min()
        high_9 = x["high"].rolling(9).max()
        rsv = (x["close"] - low_9) / (high_9 - low_9 + 1e-9) * 100
        x["K"] = pd.Series(rsv, index=x.index).ewm(com=2, adjust=False).mean()
        x["D"] = x["K"].ewm(com=2, adjust=False).mean()
        x["J"] = 3 * x["K"] - 2 * x["D"]
    x["j_rank_20"] = rolling_last_percentile(x["J"], 20)
    x["j_rank_30"] = rolling_last_percentile(x["J"], 30)
    x["n_up_rank20_p10"] = build_n_up_feature(x, "j_rank_20", 0.10)
    x["n_up_rank20_p05"] = build_n_up_feature(x, "j_rank_20", 0.05)
    x["n_up_rank30_p10"] = build_n_up_feature(x, "j_rank_30", 0.10)
    x["n_up_rank30_p05"] = build_n_up_feature(x, "j_rank_30", 0.05)
    return x


def n_mask(df: pd.DataFrame, variant: NVariant) -> pd.Series:
    if not variant.use_n:
        return pd.Series(True, index=df.index)
    return df[f"n_up_rank{variant.rank_window}_p{int(variant.rank_threshold * 100):02d}"].fillna(False)


def load_pin_feature_df(file_path: str) -> Optional[pd.DataFrame]:
    df, err = stoploss.load_data(file_path)
    if err or df is None or len(df) < 160:
        return None
    df = df[(df["日期"] < EXCLUDE_START) | (df["日期"] > EXCLUDE_END)].copy()
    if len(df) < 160:
        return None
    df = technical_indicators.calculate_trend(df)
    x = pd.DataFrame(
        {
            "date": df["日期"],
            "open": df["开盘"].astype(float),
            "high": df["最高"].astype(float),
            "low": df["最低"].astype(float),
            "close": df["收盘"].astype(float),
            "code": Path(file_path).stem,
            "trend_line": df["知行短期趋势线"].astype(float),
            "long_line": df["知行多空线"].astype(float),
        }
    ).reset_index(drop=True)
    trend_ok = x["trend_line"] > x["long_line"]
    short_llv = x["low"].rolling(3).min()
    short_hhv = x["close"].rolling(3).max()
    short_den = (short_hhv - short_llv).replace(0, np.nan)
    short_value = (x["close"] - short_llv) / short_den * 100
    long_llv = x["low"].rolling(21).min()
    long_hhv = x["close"].rolling(21).max()
    long_den = (long_hhv - long_llv).replace(0, np.nan)
    long_value = (x["close"] - long_llv) / long_den * 100
    pin_signal = (short_value <= 30) & (long_value >= 85)
    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    body_low = np.minimum(x["open"], x["close"])
    x["lower_shadow_ratio"] = (body_low - x["low"]) / full_range
    x["trend_slope_3"] = x["trend_line"] / x["trend_line"].shift(3) - 1.0
    x["signal_mask"] = (
        trend_ok.fillna(False)
        & pin_signal.fillna(False)
        & (x["lower_shadow_ratio"] <= 0.05)
        & (x["trend_slope_3"] > 0.008)
    )
    x["sort_score"] = (
        (1.0 - x["lower_shadow_ratio"].clip(lower=0, upper=1)).fillna(0.0) * 0.55
        + x["trend_slope_3"].clip(lower=-0.05, upper=0.05).fillna(0.0) * 10 * 0.45
    )
    x = add_n_up_features(x)
    return x


def load_generic_feature_map() -> Dict[str, pd.DataFrame]:
    fmap: Dict[str, pd.DataFrame] = {}
    files = sorted(DATA_DIR.glob("*.txt"))
    total = len(files)
    for idx, fp in enumerate(files, 1):
        df = b2filter.load_one_csv(str(fp))
        if df is None or df.empty:
            continue
        df = df[(df["date"] < EXCLUDE_START) | (df["date"] > EXCLUDE_END)].reset_index(drop=True)
        if len(df) < 180:
            continue
        fmap[str(df["code"].iloc[0])] = df
        if idx % 500 == 0 or idx == total:
            print(f"基础载入进度: {idx}/{total}")
    return fmap


def load_strategy_features(base_map: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    out = {"brick": {}, "b2": {}, "pin": {}}
    total = len(base_map)
    for idx, (code, df) in enumerate(base_map.items(), 1):
        out["brick"][code] = add_n_up_features(brick_filter.add_features(df))
        out["b2"][code] = add_n_up_features(b2filter.add_features(df))
        pin_df = load_pin_feature_df(str(DATA_DIR / f"{code}.txt"))
        if pin_df is not None:
            out["pin"][code] = pin_df
        if idx % 500 == 0 or idx == total:
            print(f"策略特征进度: {idx}/{total}")
    return out


def simulate_open_to_open(df: pd.DataFrame, signal_idx: int, hold_days: int) -> Optional[dict]:
    entry_idx = signal_idx + 1
    exit_idx = signal_idx + 1 + hold_days
    if exit_idx >= len(df):
        return None
    entry_price = float(df.at[entry_idx, "open"])
    exit_price = float(df.at[exit_idx, "open"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0 or exit_price <= 0:
        return None
    ret = exit_price / entry_price - 1.0
    return {"ret": ret, "success": ret > 0}


def simulate_open_to_close(df: pd.DataFrame, signal_idx: int, close_offset_days: int) -> Optional[dict]:
    entry_idx = signal_idx + 1
    exit_idx = entry_idx + close_offset_days - 1
    if exit_idx >= len(df):
        return None
    entry_price = float(df.at[entry_idx, "open"])
    exit_price = float(df.at[exit_idx, "close"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0 or exit_price <= 0:
        return None
    ret = exit_price / entry_price - 1.0
    return {"ret": ret, "success": ret > 0}


def simulate_brick_trade(df: pd.DataFrame, signal_idx: int) -> Optional[dict]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    stop_price = float(df.at[signal_idx, "low"]) * 0.99
    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    first_eligible_idx = entry_idx + 1
    last_hold_idx = signal_idx + 4
    if last_hold_idx >= len(df):
        return None
    tp_price = entry_price * 1.03
    tp_pending = False
    sl_pending = False
    for i in range(first_eligible_idx, last_hold_idx + 1):
        low_i = float(df.at[i, "low"])
        high_i = float(df.at[i, "high"])
        if low_i <= stop_price and i + 1 < len(df):
            sl_pending = True
        if high_i >= tp_price and i + 1 < len(df):
            tp_pending = True
        if sl_pending and i + 1 < len(df):
            exit_price = float(df.at[i + 1, "open"])
            ret = exit_price / entry_price - 1.0
            return {"ret": ret, "success": ret > 0}
        if tp_pending and i + 1 < len(df):
            exit_price = float(df.at[i + 1, "open"])
            ret = exit_price / entry_price - 1.0
            return {"ret": ret, "success": ret > 0}
    exit_price = float(df.at[last_hold_idx, "open"])
    ret = exit_price / entry_price - 1.0
    return {"ret": ret, "success": ret > 0}


def brick_selected_indices(df: pd.DataFrame, variant: NVariant) -> List[int]:
    mask_a = df["pattern_a"] & (df["rebound_ratio"] >= 1.2)
    mask_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
    mask = (
        df["signal_base"]
        & (df["ret1"] <= 0.08)
        & (mask_a | mask_b)
        & (df["trend_line"] > df["long_line"])
        & n_mask(df, variant)
    ).fillna(False)
    idxs = np.flatnonzero(mask.to_numpy())
    return [int(i) for i in idxs]


def b2_signal_indices(df: pd.DataFrame, variant: NVariant) -> List[int]:
    mask = (
        df["trend_ok"]
        & df["dual_start"]
        & df["small_upper_shadow"]
        & df["b2_volume_ok"]
        & df["b2_medium_volume"]
        & df["b2_j_ok"]
        & df["near_20d_high"]
        & (df["close_position"] >= b2filter.CLOSE_POSITION_MIN)
        & (df["ret1"] >= b2filter.RET1_MIN)
        & n_mask(df, variant)
    ).fillna(False)
    return [int(i) for i in np.flatnonzero(mask.to_numpy())]


def pin_signal_indices(df: pd.DataFrame, variant: NVariant) -> List[int]:
    mask = (df["signal_mask"] & n_mask(df, variant)).fillna(False)
    return [int(i) for i in np.flatnonzero(mask.to_numpy())]


def summarize_trades(strategy: str, variant: NVariant, trades: List[dict]) -> dict:
    if not trades:
        return {
            "strategy": strategy,
            "n_variant": variant.label,
            "sample_count": 0,
            "avg_trade_return": np.nan,
            "success_rate": np.nan,
        }
    tdf = pd.DataFrame(trades)
    return {
        "strategy": strategy,
        "n_variant": variant.label,
        "sample_count": int(len(tdf)),
        "avg_trade_return": float(tdf["ret"].mean()),
        "success_rate": float(tdf["success"].mean()),
    }


def run_strategy_ab(feature_map: Dict[str, pd.DataFrame], strategy: str, variant: NVariant) -> dict:
    trades: List[dict] = []
    for code, df in feature_map.items():
        if strategy == "brick":
            idxs = brick_selected_indices(df, variant)
            for sidx in idxs:
                trade = simulate_brick_trade(df, sidx)
                if trade:
                    trades.append(trade)
        elif strategy == "pin":
            idxs = pin_signal_indices(df, variant)
            for sidx in idxs:
                trade = simulate_open_to_open(df, sidx, 3)
                if trade:
                    trades.append(trade)
        elif strategy == "b2":
            idxs = b2_signal_indices(df, variant)
            for sidx in idxs:
                trade = simulate_open_to_close(df, sidx, 30)
                if trade:
                    trades.append(trade)
        else:
            raise ValueError(strategy)
    return summarize_trades(strategy, variant, trades)


def best_n_vs_base(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in result_df.groupby("strategy"):
        base = g[g["n_variant"] == "不加N"].iloc[0]
        n_only = g[g["n_variant"] != "不加N"].sort_values(
            ["success_rate", "avg_trade_return", "sample_count"],
            ascending=[False, False, False],
        )
        best_n = n_only.iloc[0]
        rows.append(
            {
                "strategy": strategy,
                "base_sample_count": int(base["sample_count"]),
                "base_avg_trade_return": float(base["avg_trade_return"]),
                "base_success_rate": float(base["success_rate"]),
                "best_n_variant": best_n["n_variant"],
                "best_n_sample_count": int(best_n["sample_count"]),
                "best_n_avg_trade_return": float(best_n["avg_trade_return"]),
                "best_n_success_rate": float(best_n["success_rate"]),
                "success_rate_delta": float(best_n["success_rate"] - base["success_rate"]),
                "avg_trade_return_delta": float(best_n["avg_trade_return"] - base["avg_trade_return"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_map = load_generic_feature_map()
    fmap = load_strategy_features(base_map)
    variants = [
        NVariant(False, 20, 0.10),
        NVariant(True, 20, 0.10),
        NVariant(True, 20, 0.05),
        NVariant(True, 30, 0.10),
        NVariant(True, 30, 0.05),
    ]
    strategies = ["brick", "pin", "b2"]

    rows = []
    total = len(strategies) * len(variants)
    done = 0
    for strategy in strategies:
        for variant in variants:
            rows.append(run_strategy_ab(fmap[strategy], strategy, variant))
            done += 1
            print(f"组合进度: {done}/{total}")

    result_df = pd.DataFrame(rows)
    best_df = best_n_vs_base(result_df)
    result_df.to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(OUTPUT_DIR / "best_n_vs_base.csv", index=False, encoding="utf-8-sig")
    summary = {
        "data_dir": str(DATA_DIR),
        "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "strategies": strategies,
        "n_variants": [v.label for v in variants],
        "best_n_vs_base": best_df.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
