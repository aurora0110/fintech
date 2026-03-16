from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b2filter

DATA_DIR = ROOT / "data/forward_data"
RESULT_DIR = ROOT / "results/b2_type14_param_search_fast_20260313"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")


def exclude_mask(dates: pd.Series) -> pd.Series:
    return ~((dates >= EXCLUDE_START) & (dates <= EXCLUDE_END))


def safe_div(a, b):
    return a / (b + 1e-12)


def load_all() -> list[tuple[str, pd.DataFrame]]:
    files = sorted(DATA_DIR.glob("*.txt"))
    out = []
    for idx, path in enumerate(files, start=1):
        df = b2filter.load_one_csv(str(path))
        if df is None or df.empty:
            continue
        df = df[exclude_mask(df["date"])].reset_index(drop=True)
        if len(df) < 120:
            continue
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
        rsv = (x["close"] - low_9) / (high_9 - low_9 + 1e-12) * 100
        x["K"] = pd.Series(rsv, index=x.index).ewm(com=2, adjust=False).mean()
        x["D"] = x["K"].ewm(com=2, adjust=False).mean()
        x["J"] = 3 * x["K"] - 2 * x["D"]
        x["j_rank20"] = x["J"].rolling(20, min_periods=20).apply(
            lambda win: pd.Series(win).rank(pct=True).iloc[-1],
            raw=False,
        )
        x["j_rank20_prev"] = x["j_rank20"].shift(1)

        x["vol_ma5"] = x["volume"].rolling(5).mean()
        x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5"])

        real_body = (x["close"] - x["open"]).abs()
        x["body"] = real_body
        x["upper_shadow"] = x["high"] - np.maximum(x["open"], x["close"])
        x["lower_shadow"] = np.minimum(x["open"], x["close"]) - x["low"]
        x["full_range"] = (x["high"] - x["low"]).replace(0, np.nan)
        x["close_position"] = safe_div(x["close"] - x["low"], x["full_range"])

        x["trend_spread"] = safe_div(x["trend_line"] - x["long_line"], x["close"])
        x["trend_slope5"] = safe_div(x["trend_line"], x["trend_line"].shift(5)) - 1.0

        bull_cross = (x["trend_line"] > x["long_line"]) & (x["trend_line"].shift(1) <= x["long_line"].shift(1))
        x["bull_cross"] = bull_cross.fillna(False)
        out.append((path.stem, x))
        if idx % 500 == 0:
            print(f"数据加载进度: {idx}/{len(files)}")
    return out


def type4_mask_for_df(x: pd.DataFrame, touch_ratio: float, lookback: int = 20) -> pd.Series:
    result = pd.Series(False, index=x.index)
    bull_cross = x["bull_cross"].to_numpy(dtype=bool)
    lows = x["low"].to_numpy(dtype=float)
    closes = x["close"].to_numpy(dtype=float)
    trends = x["trend_line"].to_numpy(dtype=float)
    n = len(x)
    for i in range(10, n):
        left = max(1, i - lookback)
        cross_idx = None
        for j in range(i - 1, left - 1, -1):
            if bull_cross[j]:
                cross_idx = j
                break
        if cross_idx is None:
            continue
        prev_touch = (lows[i - 1] <= trends[i - 1] * touch_ratio) or (closes[i - 1] <= trends[i - 1] * touch_ratio)
        if not prev_touch:
            continue
        had_touch = False
        for k in range(cross_idx + 1, i - 1):
            if (lows[k] <= trends[k] * touch_ratio) or (closes[k] <= trends[k] * touch_ratio):
                had_touch = True
                break
        if not had_touch:
            result.iat[i] = True
    return result


def build_signals(all_data, params: dict, which: str) -> pd.DataFrame:
    rows = []
    for idx, (code, x) in enumerate(all_data, start=1):
        small_upper = (x["body"] <= 1e-12) | (x["upper_shadow"] <= x["body"] * params["upper_shadow_body_ratio"] + 1e-12)
        b2_j_ok = (
            (x["J"] < params["j_max"])
            & (x["J"] > x["J"].shift(1))
            & (x["J"].shift(1) < x["J"].shift(2))
            & (x["J"].shift(2) < x["J"].shift(3))
        )
        base_b2 = (
            x["trend_ok"]
            & (x["close"] > x["open"])
            & (x["ret1"] >= params["ret1_min"])
            & small_upper
            & (x["volume"] > x["volume"].shift(1))
            & (x["volume"] > x["vol_ma5"])
            & b2_j_ok
        )
        if which == "type1":
            type_mask = (
                (x["close"].shift(1) <= x["long_line"].shift(1) * params["type1_near_ratio"])
                & (x["j_rank20_prev"] <= params["type1_j_rank20_max"])
            )
            score = (
                0.35 * x["close_position"].fillna(0.0).clip(0, 1)
                + 0.20 * (1.0 - np.minimum(np.abs(x["signal_vs_ma5"].fillna(0.0) - 1.9) / 0.6, 1.0)).clip(0, 1)
                + 0.20 * x["trend_spread"].fillna(0.0).clip(lower=0.0)
                + 0.10 * (1.0 - np.minimum(x["J"].fillna(999) / params["j_max"], 1.0)).clip(0, 1)
                + 0.10 * np.maximum(x["trend_slope5"].fillna(0.0), 0.0)
                + 0.05 * type_mask.astype(float)
            )
        else:
            type_mask = type4_mask_for_df(x, params["type4_touch_ratio"])
            score = (
                0.35 * x["close_position"].fillna(0.0).clip(0, 1)
                + 0.20 * (1.0 - np.minimum(np.abs(x["signal_vs_ma5"].fillna(0.0) - 1.9) / 0.6, 1.0)).clip(0, 1)
                + 0.20 * x["trend_spread"].fillna(0.0).clip(lower=0.0)
                + 0.10 * (1.0 - np.minimum(x["J"].fillna(999) / params["j_max"], 1.0)).clip(0, 1)
                + 0.10 * np.maximum(x["trend_slope5"].fillna(0.0), 0.0)
                + 0.05 * type_mask.astype(float)
            )

        signal_mask = base_b2 & type_mask
        idxs = np.where(signal_mask.to_numpy(dtype=bool))[0]
        for i in idxs:
            entry_idx = i + 1
            if entry_idx >= len(x):
                continue
            rows.append(
                {
                    "code": code,
                    "signal_idx": i,
                    "signal_date": x.at[i, "date"],
                    "entry_idx": entry_idx,
                    "entry_date": x.at[entry_idx, "date"],
                    "entry_open": float(x.at[entry_idx, "open"]),
                    "signal_low": float(x.at[i, "low"]),
                    "sort_score": float(score.iat[i]),
                }
            )
        if idx % 500 == 0:
            print(f"{which} 信号构建进度: {idx}/{len(all_data)}")
    return pd.DataFrame(rows)


def apply_exit(x: pd.DataFrame, sig: pd.Series, exit_rule: str) -> dict:
    entry_idx = int(sig["entry_idx"])
    if entry_idx >= len(x) - 1:
        return {}
    entry_price = float(sig["entry_open"])
    signal_low = float(sig["signal_low"])
    horizon = 30 if "30" in exit_rule else 20
    last_idx = min(len(x) - 1, entry_idx + horizon - 1)
    exit_idx = last_idx
    exit_price = float(x.at[exit_idx, "close"])
    exit_reason = "hold_to_close"
    stop_level = signal_low * 0.99

    for i in range(entry_idx + 1, last_idx + 1):
        row = x.iloc[i]
        if exit_rule == "hold20_close" or exit_rule == "hold30_close":
            continue
        if exit_rule.startswith("tp"):
            tp = float(exit_rule.replace("tp", "").split("_")[0]) / 100.0
            if float(row["high"]) >= entry_price * (1.0 + tp):
                exit_idx = i
                exit_price = float(row["close"])
                exit_reason = f"take_profit_{tp:.2%}"
                break
        elif exit_rule == "entry_day_low099_hold30":
            if float(row["low"]) <= stop_level:
                exit_idx = i
                exit_price = stop_level
                exit_reason = "stop_entryday_low099"
                break
        elif exit_rule == "trend_break_hold30":
            if float(row["close"]) < float(row["trend_line"]):
                exit_idx = i
                exit_price = float(row["close"])
                exit_reason = "trend_break"
                break
        elif exit_rule == "trend_break_2d_hold30":
            if i >= entry_idx + 2:
                if (float(x.iloc[i]["close"]) < float(x.iloc[i]["trend_line"])) and (
                    float(x.iloc[i - 1]["close"]) < float(x.iloc[i - 1]["trend_line"])
                ):
                    exit_idx = i
                    exit_price = float(row["close"])
                    exit_reason = "trend_break_2d"
                    break
        elif exit_rule == "trend_unrecovered_2d_hold30":
            if i >= entry_idx + 2:
                broke_yesterday = float(x.iloc[i - 1]["close"]) < float(x.iloc[i - 1]["trend_line"])
                unrecovered_today = float(x.iloc[i]["close"]) < float(x.iloc[i]["trend_line"])
                if broke_yesterday and unrecovered_today:
                    exit_idx = i
                    exit_price = float(row["close"])
                    exit_reason = "trend_unrecovered_2d"
                    break
        elif exit_rule == "bearish_vol_hold30":
            prev_idx = i - 1
            if prev_idx < 0:
                continue
            prev_row = x.iloc[prev_idx]
            is_bear = float(row["close"]) < float(row["open"])
            if is_bear and (float(row["volume"]) >= float(prev_row["volume"]) * 1.3):
                exit_idx = i
                exit_price = float(row["close"])
                exit_reason = "bearish_volume"
                break
        else:
            raise ValueError(exit_rule)

    lows = x.loc[entry_idx : exit_idx, "low"]
    highs = x.loc[entry_idx : exit_idx, "high"]
    return {
        "return": exit_price / entry_price - 1.0,
        "success": float(exit_price > entry_price),
        "max_favorable": highs.max() / entry_price - 1.0,
        "max_adverse": lows.min() / entry_price - 1.0,
        "exit_reason": exit_reason,
        "exit_date": x.at[exit_idx, "date"],
    }


def run_trade_level(all_data, signals: pd.DataFrame, exit_rule: str) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    code_map = {code: x for code, x in all_data}
    out = []
    for row in signals.itertuples(index=False):
        x = code_map[row.code]
        metrics = apply_exit(x, pd.Series(row._asdict()), exit_rule)
        if metrics:
            out.append({**row._asdict(), **metrics})
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"sample_count": 0}
    return {
        "sample_count": int(len(df)),
        "success_rate": float(df["success"].mean()),
        "avg_return": float(df["return"].mean()),
        "avg_max_favorable": float(df["max_favorable"].mean()),
        "avg_max_adverse": float(df["max_adverse"].mean()),
    }


def pick_best(df: pd.DataFrame, min_count: int) -> pd.Series:
    cand = df[df["sample_count"] >= min_count].copy()
    if cand.empty:
        cand = df.copy()
    cand = cand.sort_values(
        by=["avg_return", "success_rate", "sample_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return cand.iloc[0]


def main():
    all_data = load_all()

    type1_exit = "hold30_close"
    type4_exit = "hold20_close"

    type1_grid = []
    for ret1_min in [0.03, 0.04, 0.05]:
        for upper_ratio in [0.3, 0.5]:
            for j_max in [80.0, 90.0, 100.0]:
                for near_ratio in [1.01, 1.02, 1.03]:
                    for jrank in [0.05, 0.10, 0.15]:
                        params = {
                            "ret1_min": ret1_min,
                            "upper_shadow_body_ratio": upper_ratio,
                            "j_max": j_max,
                            "type1_near_ratio": near_ratio,
                            "type1_j_rank20_max": jrank,
                            "type4_touch_ratio": 1.01,
                        }
                        sig = build_signals(all_data, params, "type1")
                        trades = run_trade_level(all_data, sig, type1_exit)
                        s = summarize(trades)
                        s.update(params)
                        type1_grid.append(s)

    type4_grid = []
    for ret1_min in [0.03, 0.04, 0.05]:
        for upper_ratio in [0.3, 0.5]:
            for j_max in [80.0, 90.0, 100.0]:
                for touch_ratio in [1.00, 1.01, 1.02]:
                    params = {
                        "ret1_min": ret1_min,
                        "upper_shadow_body_ratio": upper_ratio,
                        "j_max": j_max,
                        "type1_near_ratio": 1.02,
                        "type1_j_rank20_max": 0.10,
                        "type4_touch_ratio": touch_ratio,
                    }
                    sig = build_signals(all_data, params, "type4")
                    trades = run_trade_level(all_data, sig, type4_exit)
                    s = summarize(trades)
                    s.update(params)
                    type4_grid.append(s)

    type1_df = pd.DataFrame(type1_grid)
    type4_df = pd.DataFrame(type4_grid)
    type1_df.to_csv(RESULT_DIR / "type1_param_search.csv", index=False)
    type4_df.to_csv(RESULT_DIR / "type4_param_search.csv", index=False)

    best1 = pick_best(type1_df, 500)
    best4 = pick_best(type4_df, 150)
    summary = {
        "type1_exit": type1_exit,
        "type4_exit": type4_exit,
        "best_type1": best1.to_dict(),
        "best_type4": best4.to_dict(),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
