from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
DAILY_DIR = ROOT / "data" / "20260408"
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
BUY_GAP_LIMIT = 0.04
ATR_PERIODS = (10, 14, 20)
INITIAL_TP_K = (1.0, 1.5, 2.0, 2.5, 3.0)
INITIAL_HOLD_DAYS = (3, 4, 5)
INITIAL_ENTRY_STOP_K = (0.5, 0.8, 1.0, 1.2, 1.5)
INITIAL_SIGNAL_STOP_K = (0.3, 0.5, 0.8, 1.0, 1.2)
TP_K_HARD_MIN = 0.5
TP_K_HARD_MAX = 4.0
STOP_K_HARD_MIN = 0.2
STOP_K_HARD_MAX = 2.0
HOLD_DAYS_HARD_MIN = 2
HOLD_DAYS_HARD_MAX = 8

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brick_filter  # noqa: E402
from utils.backtest.run_momentum_tail_experiment import (  # noqa: E402
    build_portfolio_curve,
    compute_equity_metrics,
    max_consecutive_failures,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def write_error(result_dir: Path, exc: BaseException) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(result_dir / "error.json", payload)
    update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))


def read_text_auto(path: Path) -> list[str]:
    for enc in ("gbk", "utf-8", "latin1"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except Exception:
            pass
    raise RuntimeError(f"无法读取文件: {path}")


def load_daily_df(path: Path) -> pd.DataFrame | None:
    rows: list[dict[str, Any]] = []
    for line in read_text_auto(path)[2:]:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        rows.append(
            {
                "date": pd.to_datetime(parts[0], errors="coerce"),
                "open": pd.to_numeric(parts[1], errors="coerce"),
                "high": pd.to_numeric(parts[2], errors="coerce"),
                "low": pd.to_numeric(parts[3], errors="coerce"),
                "close": pd.to_numeric(parts[4], errors="coerce"),
                "volume": pd.to_numeric(parts[5], errors="coerce"),
                "code": path.stem,
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = (
        df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)].copy().reset_index(drop=True)
    if len(df) < brick_filter.MIN_BARS:
        return None
    return df


def add_atr_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    for period in ATR_PERIODS:
        out[f"atr_{period}"] = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return out


def iter_daily_files(data_dir: Path, file_limit: int = 0) -> list[Path]:
    files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {".txt", ".csv"}])
    return files[:file_limit] if file_limit > 0 else files


def build_selected_signals(data_dir: Path, file_limit: int = 0) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    daily_map: dict[str, pd.DataFrame] = {}
    files = iter_daily_files(data_dir, file_limit=file_limit)
    total = len(files)
    for idx, path in enumerate(files, 1):
        df = load_daily_df(path)
        if df is None or df.empty:
            continue
        df = add_atr_columns(df)
        code = str(df["code"].iloc[0])
        daily_map[code] = df.copy()
        x = brick_filter.add_features(df)
        mask_a = x["pattern_a"] & (x["rebound_ratio"] >= 0.8)
        mask_b = x["pattern_b"] & (x["rebound_ratio"] >= 1.0)
        signal_mask = (
            x["signal_base"]
            & (x["ret1"] <= 0.08)
            & (mask_a | mask_b)
            & (x["trend_line"] > x["long_line"])
            & x["ret1"].between(-0.03, 0.11, inclusive="both")
        )
        for signal_idx in np.flatnonzero(signal_mask.to_numpy()):
            signal_idx = int(signal_idx)
            signal_row = x.iloc[signal_idx]
            daily_row = df.iloc[signal_idx]
            rows.append(
                {
                    "signal_date": pd.Timestamp(signal_row["date"]),
                    "code": code,
                    "signal_idx": signal_idx,
                    "sort_score": float(signal_row["score"]) if "score" in x.columns and pd.notna(signal_row["score"]) else 0.0,
                    "signal_open": float(signal_row["open"]),
                    "signal_close": float(signal_row["close"]),
                    "signal_low": float(signal_row["low"]),
                    "atr_10": float(daily_row["atr_10"]) if pd.notna(daily_row["atr_10"]) else np.nan,
                    "atr_14": float(daily_row["atr_14"]) if pd.notna(daily_row["atr_14"]) else np.nan,
                    "atr_20": float(daily_row["atr_20"]) if pd.notna(daily_row["atr_20"]) else np.nan,
                }
            )
        if idx % 500 == 0 or idx == total:
            print(f"ATR实验信号构建进度: {idx}/{total}")
    if not rows:
        return pd.DataFrame(), daily_map
    signal_df = pd.DataFrame(rows).sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    signal_df = signal_df[(signal_df["signal_date"] < EXCLUDE_START) | (signal_df["signal_date"] > EXCLUDE_END)].reset_index(drop=True)
    return signal_df, daily_map


def find_next_daily_entry(daily_df: pd.DataFrame, signal_date: pd.Timestamp) -> tuple[int | None, pd.Timestamp | None, float | None]:
    later_pos = np.flatnonzero((daily_df["date"] > signal_date).to_numpy())
    if len(later_pos) == 0:
        return None, None, None
    entry_idx = int(later_pos[0])
    entry_row = daily_df.iloc[entry_idx]
    entry_date = pd.Timestamp(entry_row["date"])
    entry_open = float(entry_row["open"])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return None, None, None
    return entry_idx, entry_date, entry_open


def prepare_signal_payloads(selected_df: pd.DataFrame, daily_map: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    total = len(selected_df)
    for idx, row in enumerate(selected_df.itertuples(index=False), 1):
        code = str(row.code)
        daily_df = daily_map.get(code)
        if daily_df is None or daily_df.empty:
            continue
        entry_idx, entry_date, entry_open = find_next_daily_entry(daily_df, pd.Timestamp(row.signal_date))
        if entry_idx is None or entry_date is None or entry_open is None:
            continue
        signal_close = float(row.signal_close)
        if not np.isfinite(signal_close) or signal_close <= 0:
            continue
        if entry_open / signal_close - 1.0 >= BUY_GAP_LIMIT:
            continue
        entry_low = float(daily_df.iloc[entry_idx]["low"])
        if not np.isfinite(entry_low) or entry_low <= 0:
            continue
        atr_map = {
            10: float(row.atr_10) if pd.notna(row.atr_10) else np.nan,
            14: float(row.atr_14) if pd.notna(row.atr_14) else np.nan,
            20: float(row.atr_20) if pd.notna(row.atr_20) else np.nan,
        }
        if not any(np.isfinite(v) and v > 0 for v in atr_map.values()):
            continue
        payloads.append(
            {
                "code": code,
                "signal_date": pd.Timestamp(row.signal_date),
                "sort_score": float(row.sort_score),
                "signal_close": signal_close,
                "signal_low": float(row.signal_low),
                "entry_idx": int(entry_idx),
                "entry_date": entry_date,
                "entry_open": float(entry_open),
                "entry_low": float(entry_low),
                "daily_df": daily_df,
                "atr_map": atr_map,
            }
        )
        if idx % 1000 == 0 or idx == total:
            print(f"ATR实验预处理有效信号进度: {idx}/{total}")
    return payloads


def summarize_trades(trade_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, g in trade_df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        portfolio_input = g[["signal_date", "code", "sort_score", "return_pct"]].rename(columns={"return_pct": "ret"})
        portfolio_df = build_portfolio_curve(portfolio_input)
        eq = compute_equity_metrics(portfolio_df)
        row.update(
            {
                "trade_count": int(len(g)),
                "coverage_days": int(g["signal_date"].nunique()),
                "avg_trade_return": float(g["return_pct"].mean()),
                "success_rate": float(g["success"].astype(float).mean()),
                "avg_holding_days": float(g["holding_days"].mean()),
                "max_consecutive_failures": int(max_consecutive_failures(g["success"].astype(bool).tolist())),
                "annual_return_signal_basket": float(eq["annual_return"]) if pd.notna(eq["annual_return"]) else np.nan,
                "max_drawdown_signal_basket": float(eq["max_drawdown"]) if pd.notna(eq["max_drawdown"]) else np.nan,
                "final_equity_signal_basket": float(eq["final_equity"]) if pd.notna(eq["final_equity"]) else np.nan,
                "equity_days_signal_basket": int(eq["equity_days"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_profiles(tp_k_values: tuple[float, ...], hold_days_values: tuple[int, ...], entry_stop_k_values: tuple[float, ...], signal_stop_k_values: tuple[float, ...]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for atr_period in ATR_PERIODS:
        for tp_k in tp_k_values:
            for hold_days in hold_days_values:
                profiles.append(
                    {
                        "stop_family": "fixed_signal_low",
                        "atr_period": atr_period,
                        "tp_k": float(tp_k),
                        "hold_days": int(hold_days),
                        "stop_k": np.nan,
                        "profile_name": f"fixed_signal_low__p{atr_period}__tp{tp_k:.2f}__h{hold_days}",
                    }
                )
                profiles.append(
                    {
                        "stop_family": "fixed_entry_low",
                        "atr_period": atr_period,
                        "tp_k": float(tp_k),
                        "hold_days": int(hold_days),
                        "stop_k": np.nan,
                        "profile_name": f"fixed_entry_low__p{atr_period}__tp{tp_k:.2f}__h{hold_days}",
                    }
                )
                profiles.append(
                    {
                        "stop_family": "fixed_entry_low_095",
                        "atr_period": atr_period,
                        "tp_k": float(tp_k),
                        "hold_days": int(hold_days),
                        "stop_k": np.nan,
                        "profile_name": f"fixed_entry_low_095__p{atr_period}__tp{tp_k:.2f}__h{hold_days}",
                    }
                )
                for stop_k in entry_stop_k_values:
                    profiles.append(
                        {
                            "stop_family": "atr_stop_entry_price",
                            "atr_period": atr_period,
                            "tp_k": float(tp_k),
                            "hold_days": int(hold_days),
                            "stop_k": float(stop_k),
                            "profile_name": f"atr_stop_entry_price__p{atr_period}__tp{tp_k:.2f}__sk{stop_k:.2f}__h{hold_days}",
                        }
                    )
                for stop_k in signal_stop_k_values:
                    profiles.append(
                        {
                            "stop_family": "atr_stop_signal_low",
                            "atr_period": atr_period,
                            "tp_k": float(tp_k),
                            "hold_days": int(hold_days),
                            "stop_k": float(stop_k),
                            "profile_name": f"atr_stop_signal_low__p{atr_period}__tp{tp_k:.2f}__sk{stop_k:.2f}__h{hold_days}",
                        }
                    )
    return profiles


def compute_stop_price(profile: dict[str, Any], signal_low: float, entry_low: float, entry_price: float, atr_value: float) -> float | None:
    family = str(profile["stop_family"])
    if family == "fixed_signal_low":
        return signal_low
    if family == "fixed_entry_low":
        return entry_low
    if family == "fixed_entry_low_095":
        return entry_low * 0.95
    if not np.isfinite(atr_value) or atr_value <= 0:
        return None
    stop_k = float(profile["stop_k"])
    if family == "atr_stop_entry_price":
        return entry_price - atr_value * stop_k
    if family == "atr_stop_signal_low":
        return signal_low - atr_value * stop_k
    raise ValueError(family)


def simulate_profile_for_payload(signal_payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any] | None:
    daily_df = signal_payload["daily_df"]
    signal_date = signal_payload["signal_date"]
    entry_idx = int(signal_payload["entry_idx"])
    entry_date = pd.Timestamp(signal_payload["entry_date"])
    entry_open = float(signal_payload["entry_open"])
    signal_close = float(signal_payload["signal_close"])
    atr_period = int(profile["atr_period"])
    atr_value = float(signal_payload["atr_map"].get(atr_period, np.nan))
    if not np.isfinite(atr_value) or atr_value <= 0:
        return None
    entry_low = float(signal_payload["entry_low"])
    stop_price = compute_stop_price(profile, float(signal_payload["signal_low"]), entry_low, float(entry_open), atr_value)
    if stop_price is None or not np.isfinite(stop_price) or stop_price <= 0:
        return None
    tp_pct = float(profile["tp_k"]) * atr_value / signal_close
    tp_price = float(entry_open) * (1.0 + tp_pct)
    last_idx = min(len(daily_df) - 1, entry_idx + int(profile["hold_days"]) - 1)
    exit_idx = last_idx
    exit_price = float(daily_df.iloc[last_idx]["close"])
    exit_reason = "max_hold_close"
    for idx in range(entry_idx, last_idx + 1):
        day_row = daily_df.iloc[idx]
        day_low = float(day_row["low"])
        day_high = float(day_row["high"])
        if np.isfinite(day_low) and day_low <= stop_price:
            exit_idx = idx
            exit_price = float(day_row["close"])
            exit_reason = "stop_same_day_close"
            break
        if np.isfinite(day_high) and day_high >= tp_price:
            next_idx = idx + 1
            if next_idx < len(daily_df):
                exit_idx = next_idx
                exit_price = float(daily_df.iloc[next_idx]["open"])
                exit_reason = "atr_tp_next_day_open"
            else:
                exit_idx = idx
                exit_price = float(day_row["close"])
                exit_reason = "atr_tp_fallback_same_day_close"
            break
    return_pct = exit_price / float(entry_open) - 1.0
    return {
        "code": str(signal_payload["code"]),
        "signal_date": signal_date,
        "entry_date": entry_date,
        "exit_date": pd.Timestamp(daily_df.iloc[exit_idx]["date"]),
        "entry_price": float(entry_open),
        "exit_price": float(exit_price),
        "return_pct": float(return_pct),
        "success": bool(return_pct > 0),
        "holding_days": int(exit_idx - entry_idx + 1),
        "sort_score": float(signal_payload["sort_score"]),
        "profile_name": str(profile["profile_name"]),
        "stop_family": str(profile["stop_family"]),
        "atr_period": atr_period,
        "tp_k": float(profile["tp_k"]),
        "stop_k": float(profile["stop_k"]) if pd.notna(profile["stop_k"]) else np.nan,
        "hold_days": int(profile["hold_days"]),
        "tp_pct": float(tp_pct),
        "trigger_take_profit": float(tp_price),
        "trigger_stop_loss": float(stop_price),
        "exit_reason": exit_reason,
    }


def run_round(signal_payloads: list[dict[str, Any]], tp_k_values: tuple[float, ...], hold_days_values: tuple[int, ...], entry_stop_k_values: tuple[float, ...], signal_stop_k_values: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = build_profiles(tp_k_values, hold_days_values, entry_stop_k_values, signal_stop_k_values)
    trades: list[dict[str, Any]] = []
    total = len(signal_payloads)
    for idx, signal_payload in enumerate(signal_payloads, 1):
        for profile in profiles:
            trade = simulate_profile_for_payload(signal_payload, profile)
            if trade is not None:
                trades.append(trade)
        if idx % 500 == 0 or idx == total:
            print(f"ATR参数搜索进度: {idx}/{total}")
    trade_df = pd.DataFrame(trades)
    summary_df = summarize_trades(trade_df, ["profile_name", "stop_family", "atr_period", "tp_k", "stop_k", "hold_days"])
    return trade_df, summary_df


def expand_values(best: pd.Series, tp_k_values: tuple[float, ...], hold_days_values: tuple[int, ...], entry_stop_k_values: tuple[float, ...], signal_stop_k_values: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[int, ...], tuple[float, ...], tuple[float, ...], bool]:
    new_tp = set(tp_k_values)
    new_hold = set(hold_days_values)
    new_entry = set(entry_stop_k_values)
    new_signal = set(signal_stop_k_values)
    expanded = False

    best_tp = round(float(best["tp_k"]), 3)
    if best_tp == round(min(tp_k_values), 3) and min(tp_k_values) > TP_K_HARD_MIN:
        new_tp.add(max(TP_K_HARD_MIN, round(min(tp_k_values) - 0.5, 3)))
        expanded = True
    if best_tp == round(max(tp_k_values), 3) and max(tp_k_values) < TP_K_HARD_MAX:
        new_tp.add(min(TP_K_HARD_MAX, round(max(tp_k_values) + 0.5, 3)))
        expanded = True

    best_hold = int(best["hold_days"])
    if best_hold == min(hold_days_values) and min(hold_days_values) > HOLD_DAYS_HARD_MIN:
        new_hold.add(max(HOLD_DAYS_HARD_MIN, min(hold_days_values) - 1))
        expanded = True
    if best_hold == max(hold_days_values) and max(hold_days_values) < HOLD_DAYS_HARD_MAX:
        new_hold.add(min(HOLD_DAYS_HARD_MAX, max(hold_days_values) + 1))
        expanded = True

    family = str(best["stop_family"])
    if family == "atr_stop_entry_price" and pd.notna(best["stop_k"]):
        best_stop = round(float(best["stop_k"]), 3)
        if best_stop == round(min(entry_stop_k_values), 3) and min(entry_stop_k_values) > STOP_K_HARD_MIN:
            new_entry.add(max(STOP_K_HARD_MIN, round(min(entry_stop_k_values) - 0.2, 3)))
            expanded = True
        if best_stop == round(max(entry_stop_k_values), 3) and max(entry_stop_k_values) < STOP_K_HARD_MAX:
            new_entry.add(min(STOP_K_HARD_MAX, round(max(entry_stop_k_values) + 0.2, 3)))
            expanded = True
    if family == "atr_stop_signal_low" and pd.notna(best["stop_k"]):
        best_stop = round(float(best["stop_k"]), 3)
        if best_stop == round(min(signal_stop_k_values), 3) and min(signal_stop_k_values) > STOP_K_HARD_MIN:
            new_signal.add(max(STOP_K_HARD_MIN, round(min(signal_stop_k_values) - 0.2, 3)))
            expanded = True
        if best_stop == round(max(signal_stop_k_values), 3) and max(signal_stop_k_values) < STOP_K_HARD_MAX:
            new_signal.add(min(STOP_K_HARD_MAX, round(max(signal_stop_k_values) + 0.2, 3)))
            expanded = True
    return (
        tuple(sorted(new_tp)),
        tuple(sorted(new_hold)),
        tuple(sorted(new_entry)),
        tuple(sorted(new_signal)),
        expanded,
    )


def best_row(summary_df: pd.DataFrame) -> pd.Series:
    return summary_df.sort_values(["avg_trade_return", "annual_return_signal_basket", "success_rate"], ascending=[False, False, False]).iloc[0]


def run_experiment(args: argparse.Namespace, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "building_signals")
    selected_df, daily_map = build_selected_signals(Path(args.daily_dir), file_limit=args.file_limit)
    if selected_df.empty:
        raise RuntimeError("当前参数下没有任何 brick 信号")
    selected_df.to_csv(result_dir / "selected_daily_signals.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "preparing_signal_payloads", signal_count=int(len(selected_df)))
    signal_payloads = prepare_signal_payloads(selected_df, daily_map)
    if not signal_payloads:
        raise RuntimeError("当前参数下没有可交易的有效 brick 信号")

    tp_k_values = INITIAL_TP_K
    hold_days_values = INITIAL_HOLD_DAYS
    entry_stop_k_values = INITIAL_ENTRY_STOP_K
    signal_stop_k_values = INITIAL_SIGNAL_STOP_K
    boundary_trace: list[dict[str, Any]] = []
    round_no = 1
    final_summary = pd.DataFrame()
    while True:
        update_progress(result_dir, "running_round", round_no=round_no, payload_signal_count=int(len(signal_payloads)))
        trade_df, summary_df = run_round(signal_payloads, tp_k_values, hold_days_values, entry_stop_k_values, signal_stop_k_values)
        round_path = result_dir / f"atr_exit_grid_round_{round_no}.csv"
        summary_df.to_csv(round_path, index=False, encoding="utf-8-sig")
        if trade_df.empty or summary_df.empty:
            raise RuntimeError("ATR止盈止损实验没有生成有效交易")
        best = best_row(summary_df)
        boundary_trace.append(
            {
                "round_no": round_no,
                "tp_k_values": [float(x) for x in tp_k_values],
                "hold_days_values": [int(x) for x in hold_days_values],
                "entry_stop_k_values": [float(x) for x in entry_stop_k_values],
                "signal_stop_k_values": [float(x) for x in signal_stop_k_values],
                "best_profile": best.to_dict(),
            }
        )
        final_summary = summary_df.copy()
        tp_k_values, hold_days_values, entry_stop_k_values, signal_stop_k_values, expanded = expand_values(
            best, tp_k_values, hold_days_values, entry_stop_k_values, signal_stop_k_values
        )
        if not expanded:
            break
        round_no += 1
        if round_no > 8:
            break

    fixed_df = final_summary[final_summary["stop_family"].isin(["fixed_signal_low", "fixed_entry_low", "fixed_entry_low_095"])].copy()
    fixed_df.to_csv(result_dir / "fixed_stop_baseline_search.csv", index=False, encoding="utf-8-sig")
    best_fixed = best_row(fixed_df) if not fixed_df.empty else None
    best_overall = best_row(final_summary)
    best_entry_atr = best_row(final_summary[final_summary["stop_family"] == "atr_stop_entry_price"]) if not final_summary[final_summary["stop_family"] == "atr_stop_entry_price"].empty else None
    best_signal_atr = best_row(final_summary[final_summary["stop_family"] == "atr_stop_signal_low"]) if not final_summary[final_summary["stop_family"] == "atr_stop_signal_low"].empty else None

    best_payload = {
        "best_overall": None if best_overall is None else best_overall.to_dict(),
        "best_fixed_stop_baseline": None if best_fixed is None else best_fixed.to_dict(),
        "best_atr_stop_entry_price": None if best_entry_atr is None else best_entry_atr.to_dict(),
        "best_atr_stop_signal_low": None if best_signal_atr is None else best_signal_atr.to_dict(),
    }
    write_json(result_dir / "best_atr_exit.json", best_payload)
    write_json(result_dir / "boundary_trace.json", {"rounds": boundary_trace})
    summary = {
        "signal_count": int(len(selected_df)),
        "best_overall": best_payload["best_overall"],
        "best_fixed_stop_baseline": best_payload["best_fixed_stop_baseline"],
        "best_atr_stop_entry_price": best_payload["best_atr_stop_entry_price"],
        "best_atr_stop_signal_low": best_payload["best_atr_stop_signal_low"],
        "timeline_definition": {
            "signal_date": "日线 brick 信号日",
            "entry_date": "signal_date 次日开盘",
            "stop_execution": "同日收盘止损",
            "tp_execution": "ATR触发后次日开盘止盈",
            "buy_gap_limit": BUY_GAP_LIMIT,
        },
    }
    write_json(result_dir / "summary.json", summary)
    update_progress(result_dir, "finished", summary_path=str(result_dir / "summary.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK ATR止盈止损对比实验")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--daily-dir", default=str(DAILY_DIR))
    parser.add_argument("--file-limit", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.mode == "smoke" and args.file_limit <= 0:
        args.file_limit = 150
    result_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_atr_tp_sl_compare_v1_{args.mode}_{timestamp}"
    try:
        run_experiment(args, result_dir)
    except BaseException as exc:  # pragma: no cover
        write_error(result_dir, exc)
        raise


if __name__ == "__main__":
    main()
