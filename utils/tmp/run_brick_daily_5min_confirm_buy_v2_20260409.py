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
DAILY_DIR = ROOT / "data" / "20260324"
MIN5_DIR = ROOT / "data" / "202603245min"
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
ATR_PERIOD = 20
ATR_K = 2.0
ATR_HOLD_DAYS = 4
INITIAL_WINDOWS = (15, 30, 45)
WINDOW_HARD_MIN = 5
WINDOW_HARD_MAX = 90
NOON_CUTOFF = "1130"

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


def window_to_bars(window_min: int) -> int:
    if window_min <= 0 or window_min % 5 != 0:
        raise ValueError(f"非法确认窗口: {window_min}")
    return max(1, window_min // 5)


def read_text_auto(path: Path) -> list[str]:
    for enc in ("gbk", "utf-8", "latin1"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except Exception:
            pass
    raise RuntimeError(f"无法读取文件: {path}")


def load_minute_df(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in read_text_auto(path)[2:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        date_text = parts[0]
        time_text = str(parts[1]).zfill(4)
        rows.append(
            {
                "datetime": pd.to_datetime(f"{date_text} {time_text}", format="%Y/%m/%d %H%M", errors="coerce"),
                "date": pd.to_datetime(date_text, errors="coerce"),
                "time": time_text,
                "open": pd.to_numeric(parts[2], errors="coerce"),
                "high": pd.to_numeric(parts[3], errors="coerce"),
                "low": pd.to_numeric(parts[4], errors="coerce"),
                "close": pd.to_numeric(parts[5], errors="coerce"),
                "volume": pd.to_numeric(parts[6], errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["datetime", "date", "time", "open", "high", "low", "close", "volume"])
    return (
        pd.DataFrame(rows)
        .dropna(subset=["datetime", "date", "open", "high", "low", "close"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )


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


def add_atr_column(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
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
        df = add_atr_column(df)
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
            close = float(signal_row["close"])
            trend_quality = brick_filter.triangle_quality(float(signal_row["trend_spread"]), 0.08, 0.06)
            support_quality = 1.0 if bool(signal_row["support_above_double_high"]) else 0.75 if bool(signal_row["support_above_double_close"]) else 0.45
            momentum_quality = (
                0.5 * brick_filter.triangle_quality(float(signal_row["RSI14"]), 62.0, 12.0)
                + 0.5 * brick_filter.triangle_quality(float(signal_row["ret20"]), 0.18, 0.15)
            )
            candle_quality = (
                0.6 * brick_filter.triangle_quality(float(signal_row["close_location"]), 0.86, 0.18)
                + 0.4 * brick_filter.triangle_quality(float(signal_row["upper_shadow_pct"]), 0.08, 0.20)
            )
            volume_quality = brick_filter.triangle_quality(float(signal_row["signal_vs_ma5"]), 1.25, 1.8)
            rebound_ratio = float(signal_row["rebound_ratio"]) if pd.notna(signal_row["rebound_ratio"]) else 0.0
            brick_quality = brick_filter.triangle_quality(min(rebound_ratio, 8.0), 4.5, 4.0)
            sort_score = 0.28 * trend_quality + 0.22 * support_quality + 0.20 * momentum_quality + 0.15 * candle_quality + 0.10 * volume_quality + 0.05 * brick_quality
            rows.append(
                {
                    "signal_date": pd.Timestamp(signal_row["date"]),
                    "code": code,
                    "signal_idx": signal_idx,
                    "sort_score": sort_score,
                    "signal_close": close,
                    "signal_low": float(signal_row["low"]),
                    "signal_atr20": float(daily_row[f"atr_{ATR_PERIOD}"]) if pd.notna(daily_row[f"atr_{ATR_PERIOD}"]) else np.nan,
                }
            )
        if idx % 500 == 0 or idx == total:
            print(f"日线信号构建进度: {idx}/{total}")
    if not rows:
        return pd.DataFrame(), daily_map
    signal_df = pd.DataFrame(rows).sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    signal_df["score_pct_rank"] = signal_df.groupby("signal_date")["sort_score"].rank(pct=True, method="first")
    selected = signal_df[signal_df["score_pct_rank"] >= 0.20].copy()
    selected = selected.sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True])
    selected["daily_rank"] = selected.groupby("signal_date").cumcount() + 1
    selected = selected.groupby("signal_date", group_keys=False).head(brick_filter.TOP_N).reset_index(drop=True)
    selected = selected[(selected["signal_date"] < EXCLUDE_START) | (selected["signal_date"] > EXCLUDE_END)].reset_index(drop=True)
    return selected, daily_map


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


def analyze_confirm_window(min5_df: pd.DataFrame | None, entry_date: pd.Timestamp, signal_low: float, window_min: int) -> dict[str, Any] | None:
    if min5_df is None or min5_df.empty:
        return None
    bars = window_to_bars(int(window_min))
    day_df = min5_df[min5_df["date"] == entry_date].sort_values("datetime").reset_index(drop=True)
    if len(day_df) <= bars:
        return None
    session_open = float(day_df.iloc[0]["open"])
    if not np.isfinite(session_open) or session_open <= 0:
        return None
    work = day_df.copy()
    work["cum_low"] = work["low"].cummin()
    confirm_idx = None
    confirm_low = np.nan
    for i in range(bars):
        bar_row = work.iloc[i]
        low_ok = float(bar_row["cum_low"]) >= signal_low
        reclaim_ok = float(bar_row["close"]) >= session_open
        if low_ok and reclaim_ok and i + 1 < len(work):
            confirm_idx = i
            confirm_low = float(bar_row["cum_low"])
            break
    if confirm_idx is None:
        return None
    after_confirm_idx = confirm_idx + 1
    after_confirm_row = work.iloc[after_confirm_idx]
    next_bar_open = float(after_confirm_row["open"])
    if not np.isfinite(next_bar_open) or next_bar_open <= 0:
        return None
    noon_df = work[work["time"] <= NOON_CUTOFF].reset_index(drop=True)
    retest_entry_idx = None
    for j in range(after_confirm_idx + 1, len(noon_df) - 1):
        cur_row = noon_df.iloc[j]
        prev_row = noon_df.iloc[j - 1]
        cur_low = float(cur_row["low"])
        prev_low = float(prev_row["low"])
        cur_close = float(cur_row["close"])
        if cur_low < confirm_low:
            continue
        if cur_low <= prev_low and cur_close >= confirm_low:
            retest_entry_idx = j + 1
            break
    retest_entry_price = None
    retest_entry_time = None
    if retest_entry_idx is not None and retest_entry_idx < len(noon_df):
        retest_entry_row = noon_df.iloc[retest_entry_idx]
        retest_entry_price = float(retest_entry_row["open"])
        retest_entry_time = str(retest_entry_row["time"])
        if not np.isfinite(retest_entry_price) or retest_entry_price <= 0:
            retest_entry_price = None
            retest_entry_time = None
    return {
        "window_min": int(window_min),
        "confirm_idx": int(confirm_idx),
        "confirm_low": float(confirm_low),
        "session_open": float(session_open),
        "confirm_entry_price": float(next_bar_open),
        "confirm_entry_time": str(after_confirm_row["time"]),
        "confirm_entry_dt": pd.Timestamp(after_confirm_row["datetime"]),
        "retest_entry_price": retest_entry_price,
        "retest_entry_time": retest_entry_time,
        "window_low": float(work.head(bars)["low"].min()),
    }


def simulate_atr_exit_trade(record: dict[str, Any], daily_df: pd.DataFrame, entry_idx: int, signal_close: float, signal_low: float, atr20: float) -> dict[str, Any] | None:
    if not np.isfinite(atr20) or atr20 <= 0 or not np.isfinite(signal_close) or signal_close <= 0:
        return None
    entry_price = float(record["entry_price"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    tp_pct = ATR_K * atr20 / signal_close
    tp_price = entry_price * (1.0 + tp_pct)
    last_idx = min(len(daily_df) - 1, entry_idx + ATR_HOLD_DAYS - 1)
    exit_idx = last_idx
    exit_price = float(daily_df.iloc[last_idx]["close"])
    exit_reason = "max_hold_close"
    for idx in range(entry_idx, last_idx + 1):
        day_row = daily_df.iloc[idx]
        day_low = float(day_row["low"])
        day_high = float(day_row["high"])
        if np.isfinite(day_low) and day_low <= signal_low:
            exit_idx = idx
            exit_price = float(day_row["close"])
            exit_reason = "signal_low_same_day_close"
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
    out = dict(record)
    exit_date = pd.Timestamp(daily_df.iloc[exit_idx]["date"])
    return_pct = exit_price / entry_price - 1.0
    out.update(
        {
            "exit_date": exit_date,
            "exit_price": float(exit_price),
            "return_pct": float(return_pct),
            "success": bool(return_pct > 0),
            "holding_days": int(exit_idx - entry_idx + 1),
            "exit_reason": exit_reason,
            "trigger_take_profit": float(tp_price),
            "trigger_stop_loss": float(signal_low),
            "atr_period": ATR_PERIOD,
            "atr_k": ATR_K,
            "max_hold_days": ATR_HOLD_DAYS,
        }
    )
    return out


def summarize_trades(group_cols: list[str], trade_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trade_df.empty:
        return pd.DataFrame()
    for keys, g in trade_df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        portfolio_input = g[["signal_date", "code", "sort_score", "return_pct"]].rename(columns={"return_pct": "ret"})
        portfolio_df = build_portfolio_curve(portfolio_input)
        eq = compute_equity_metrics(portfolio_df)
        row.update(
            {
                "sample_count": int(len(g)),
                "coverage_days": int(g["signal_date"].nunique()),
                "avg_trade_return": float(g["return_pct"].mean()),
                "win_rate": float(g["success"].astype(float).mean()),
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


def simulate_trade_sets(selected_df: pd.DataFrame, daily_map: dict[str, pd.DataFrame], min5_dir: Path, confirm_windows: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    next_bar_rows: list[dict[str, Any]] = []
    retest_rows: list[dict[str, Any]] = []
    minute_cache: dict[str, pd.DataFrame | None] = {}
    total = len(selected_df)
    for idx, row in enumerate(selected_df.itertuples(index=False), 1):
        code = str(row.code)
        daily_df = daily_map.get(code)
        if daily_df is None or daily_df.empty:
            continue
        signal_date = pd.Timestamp(row.signal_date)
        entry_idx, entry_date, baseline_open = find_next_daily_entry(daily_df, signal_date)
        if entry_idx is None or entry_date is None or baseline_open is None:
            continue
        base = {
            "mode": "baseline_open",
            "confirm_window_min": 0,
            "code": code,
            "signal_date": signal_date,
            "entry_date": entry_date,
            "entry_price": float(baseline_open),
            "signal_low": float(row.signal_low),
            "signal_close": float(row.signal_close),
            "signal_atr20": float(row.signal_atr20) if pd.notna(row.signal_atr20) else np.nan,
            "sort_score": float(row.sort_score),
            "daily_rank": int(row.daily_rank),
        }
        baseline_trade = simulate_atr_exit_trade(base, daily_df, entry_idx, float(row.signal_close), float(row.signal_low), float(row.signal_atr20) if pd.notna(row.signal_atr20) else np.nan)
        if baseline_trade is not None:
            baseline_rows.append(baseline_trade)

        if code not in minute_cache:
            min5_path = min5_dir / f"{code}.txt"
            minute_cache[code] = load_minute_df(min5_path) if min5_path.exists() else None
        min5_df = minute_cache[code]

        for window_min in confirm_windows:
            confirm = analyze_confirm_window(min5_df, entry_date, float(row.signal_low), window_min)
            if confirm is None:
                continue
            subset_base = dict(base)
            subset_base.update(
                {
                    "mode": "confirmed_subset_open",
                    "confirm_window_min": int(window_min),
                    "confirm_entry_time": confirm["confirm_entry_time"],
                    "confirm_window_low": confirm["window_low"],
                }
            )
            subset_trade = simulate_atr_exit_trade(subset_base, daily_df, entry_idx, float(row.signal_close), float(row.signal_low), float(row.signal_atr20) if pd.notna(row.signal_atr20) else np.nan)
            if subset_trade is not None:
                subset_rows.append(subset_trade)

            next_bar_base = dict(base)
            next_bar_base.update(
                {
                    "mode": "confirmed_next_bar_open",
                    "confirm_window_min": int(window_min),
                    "entry_price": float(confirm["confirm_entry_price"]),
                    "entry_time": confirm["confirm_entry_time"],
                    "entry_dt": confirm["confirm_entry_dt"],
                    "confirm_window_low": confirm["window_low"],
                }
            )
            next_bar_trade = simulate_atr_exit_trade(next_bar_base, daily_df, entry_idx, float(row.signal_close), float(row.signal_low), float(row.signal_atr20) if pd.notna(row.signal_atr20) else np.nan)
            if next_bar_trade is not None:
                next_bar_rows.append(next_bar_trade)

            if confirm["retest_entry_price"] is not None:
                retest_base = dict(base)
                retest_base.update(
                    {
                        "mode": "confirmed_retest_entry",
                        "confirm_window_min": int(window_min),
                        "entry_price": float(confirm["retest_entry_price"]),
                        "entry_time": str(confirm["retest_entry_time"]),
                        "confirm_window_low": confirm["window_low"],
                    }
                )
                retest_trade = simulate_atr_exit_trade(retest_base, daily_df, entry_idx, float(row.signal_close), float(row.signal_low), float(row.signal_atr20) if pd.notna(row.signal_atr20) else np.nan)
                if retest_trade is not None:
                    retest_rows.append(retest_trade)
        if idx % 500 == 0 or idx == total:
            print(f"5min确认进度: {idx}/{total}")
    return (
        pd.DataFrame(baseline_rows),
        pd.DataFrame(subset_rows),
        pd.DataFrame(next_bar_rows),
        pd.DataFrame(retest_rows),
    )


def expand_confirm_windows(trade_summary: pd.DataFrame, confirm_windows: tuple[int, ...]) -> tuple[tuple[int, ...], bool]:
    confirmed = trade_summary[trade_summary["mode"] != "baseline_open"].copy()
    if confirmed.empty:
        return confirm_windows, False
    new_windows = set(int(x) for x in confirm_windows)
    expanded = False
    for mode, g in confirmed.groupby("mode", sort=True):
        best = g.sort_values(["avg_trade_return", "annual_return_signal_basket", "win_rate"], ascending=[False, False, False]).iloc[0]
        best_window = int(best["confirm_window_min"])
        if best_window == min(confirm_windows) and min(confirm_windows) > WINDOW_HARD_MIN:
            candidate = max(WINDOW_HARD_MIN, min(confirm_windows) - 5)
            if candidate not in new_windows:
                new_windows.add(candidate)
                expanded = True
        if best_window == max(confirm_windows) and max(confirm_windows) < WINDOW_HARD_MAX:
            candidate = min(WINDOW_HARD_MAX, max(confirm_windows) + 15)
            if candidate not in new_windows:
                new_windows.add(candidate)
                expanded = True
    return tuple(sorted(new_windows)), expanded


def build_overall_summary(selected_df: pd.DataFrame, trade_summary: pd.DataFrame, confirm_windows: tuple[int, ...], boundary_trace: list[dict[str, Any]]) -> dict[str, Any]:
    def best_row(mode: str) -> dict[str, Any] | None:
        g = trade_summary[trade_summary["mode"] == mode].copy()
        if g.empty:
            return None
        return g.sort_values(["avg_trade_return", "annual_return_signal_basket", "win_rate"], ascending=[False, False, False]).iloc[0].to_dict()

    best_subset = best_row("confirmed_subset_open")
    best_next = best_row("confirmed_next_bar_open")
    best_retest = best_row("confirmed_retest_entry")
    return {
        "daily_signal_count": int(len(selected_df)),
        "best_baseline_open": best_row("baseline_open"),
        "best_confirmed_subset_open": best_subset,
        "best_confirmed_next_bar_open": best_next,
        "best_confirmed_retest_entry": best_retest,
        "window_edge_flags": {
            "confirmed_subset_open": None if best_subset is None else int(best_subset["confirm_window_min"]) in {min(confirm_windows), max(confirm_windows)},
            "confirmed_next_bar_open": None if best_next is None else int(best_next["confirm_window_min"]) in {min(confirm_windows), max(confirm_windows)},
            "confirmed_retest_entry": None if best_retest is None else int(best_retest["confirm_window_min"]) in {min(confirm_windows), max(confirm_windows)},
        },
        "final_confirm_windows": [int(x) for x in confirm_windows],
        "boundary_trace": boundary_trace,
        "timeline_definition": {
            "signal_date": "日线 brick 信号日",
            "baseline_open": "signal_date 次日开盘直接买入",
            "confirmed_subset_open": "通过5min确认后，仍按次日开盘买入",
            "confirmed_next_bar_open": "确认成立后下一根5min开盘买入",
            "confirmed_retest_entry": "确认后午前首次不破确认区低点的小回踩，下一根5min开盘买入；若午前无回踩则不成交",
            "confirm_windows": [int(x) for x in confirm_windows],
            "exit_definition": "ATR20×2.0 止盈，触发后次日开盘卖；signal_low 同日收盘止损；最大持有4天",
        },
    }


def run_experiment(args: argparse.Namespace, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "building_daily_signals")
    selected_df, daily_map = build_selected_signals(Path(args.daily_dir), file_limit=args.file_limit)
    if selected_df.empty:
        raise RuntimeError("当前参数下没有生成任何 brick 日线信号")
    selected_df.to_csv(result_dir / "selected_daily_signals.csv", index=False, encoding="utf-8-sig")

    confirm_windows = INITIAL_WINDOWS
    boundary_trace: list[dict[str, Any]] = []
    round_no = 1
    final_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    trade_summary = pd.DataFrame()
    while True:
        update_progress(result_dir, "simulating_5min_confirm", selected_signal_count=int(len(selected_df)), round_no=round_no, confirm_windows=list(confirm_windows))
        baseline_df, subset_df, next_df, retest_df = simulate_trade_sets(selected_df, daily_map, Path(args.min5_dir), confirm_windows)
        if baseline_df.empty:
            raise RuntimeError("baseline_open 没有任何可交易样本")
        round_summary = pd.concat(
            [
                summarize_trades(["mode", "confirm_window_min"], baseline_df),
                summarize_trades(["mode", "confirm_window_min"], subset_df),
                summarize_trades(["mode", "confirm_window_min"], next_df),
                summarize_trades(["mode", "confirm_window_min"], retest_df),
            ],
            ignore_index=True,
        )
        round_summary.to_csv(result_dir / f"trade_summary_round_{round_no}.csv", index=False, encoding="utf-8-sig")
        boundary_trace.append(
            {
                "round_no": round_no,
                "confirm_windows": [int(x) for x in confirm_windows],
                "best_rows": build_overall_summary(selected_df, round_summary, confirm_windows, []).copy(),
            }
        )
        next_windows, expanded = expand_confirm_windows(round_summary, confirm_windows)
        final_frames = (baseline_df, subset_df, next_df, retest_df)
        trade_summary = round_summary
        confirm_windows = next_windows
        if not expanded:
            break
        round_no += 1
        if round_no > 8:
            break

    assert final_frames is not None
    baseline_df, subset_df, next_df, retest_df = final_frames
    baseline_df.to_csv(result_dir / "baseline_open_trades.csv", index=False, encoding="utf-8-sig")
    subset_df.to_csv(result_dir / "confirmed_subset_open_trades.csv", index=False, encoding="utf-8-sig")
    next_df.to_csv(result_dir / "confirmed_next_bar_open_trades.csv", index=False, encoding="utf-8-sig")
    retest_df.to_csv(result_dir / "confirmed_retest_entry_trades.csv", index=False, encoding="utf-8-sig")

    update_progress(result_dir, "summarizing", final_confirm_windows=list(confirm_windows))
    trade_summary.to_csv(result_dir / "trade_summary.csv", index=False, encoding="utf-8-sig")
    summary = build_overall_summary(selected_df, trade_summary, confirm_windows, boundary_trace)
    write_json(result_dir / "summary.json", summary)
    update_progress(result_dir, "finished", summary_path=str(result_dir / "summary.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK 日线信号 + 5min 辅助买入实验")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--daily-dir", default=str(DAILY_DIR))
    parser.add_argument("--min5-dir", default=str(MIN5_DIR))
    parser.add_argument("--file-limit", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.mode == "smoke" and args.file_limit <= 0:
        args.file_limit = 120
    result_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_daily_5min_confirm_buy_v2_{args.mode}_{timestamp}"
    try:
        run_experiment(args, result_dir)
    except BaseException as exc:  # pragma: no cover
        write_error(result_dir, exc)
        raise


if __name__ == "__main__":
    main()
