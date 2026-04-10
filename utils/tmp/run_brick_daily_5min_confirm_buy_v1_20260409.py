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
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1_000_000.0
HORIZONS = [1, 3, 5, 10, 20]
CONFIRM_BARS = 6  # 9:35~10:00 共 30 分钟
ATR_PERIOD = 20
ATR_K = 2.0
ATR_HOLD_DAYS = 4

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brick_filter
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
    df = pd.DataFrame(rows)
    return df.dropna(subset=["datetime", "date", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)


def load_daily_df(path: Path) -> pd.DataFrame | None:
    rows: list[dict[str, Any]] = []
    lines = read_text_auto(path)
    for line in lines[2:]:
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
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)].copy()
    if len(df) < brick_filter.MIN_BARS:
        return None
    return df


def add_atr_column(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
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
        code = str(df["code"].iloc[0])
        df = add_atr_column(df)
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
        signal_idxs = np.flatnonzero(signal_mask.to_numpy())
        for signal_idx in signal_idxs:
            signal_idx = int(signal_idx)
            close = float(x.at[signal_idx, "close"])
            trend_quality = brick_filter.triangle_quality(float(x.at[signal_idx, "trend_spread"]), 0.08, 0.06)
            support_quality = (
                1.0 if bool(x.at[signal_idx, "support_above_double_high"]) else
                0.75 if bool(x.at[signal_idx, "support_above_double_close"]) else
                0.45
            )
            momentum_quality = (
                0.5 * brick_filter.triangle_quality(float(x.at[signal_idx, "RSI14"]), 62.0, 12.0)
                + 0.5 * brick_filter.triangle_quality(float(x.at[signal_idx, "ret20"]), 0.18, 0.15)
            )
            candle_quality = (
                0.6 * brick_filter.triangle_quality(float(x.at[signal_idx, "close_location"]), 0.86, 0.18)
                + 0.4 * brick_filter.triangle_quality(float(x.at[signal_idx, "upper_shadow_pct"]), 0.08, 0.20)
            )
            volume_quality = brick_filter.triangle_quality(float(x.at[signal_idx, "signal_vs_ma5"]), 1.25, 1.8)
            rebound_ratio = float(x.at[signal_idx, "rebound_ratio"]) if pd.notna(x.at[signal_idx, "rebound_ratio"]) else 0.0
            brick_quality = brick_filter.triangle_quality(min(rebound_ratio, 8.0), 4.5, 4.0)
            sort_score = (
                0.28 * trend_quality
                + 0.22 * support_quality
                + 0.20 * momentum_quality
                + 0.15 * candle_quality
                + 0.10 * volume_quality
                + 0.05 * brick_quality
            )
            rows.append(
                {
                    "signal_date": pd.Timestamp(x.at[signal_idx, "date"]),
                    "code": code,
                    "signal_idx": signal_idx,
                    "sort_score": sort_score,
                    "signal_close": close,
                    "signal_low": float(x.at[signal_idx, "low"]),
                    "signal_vs_ma5": float(x.at[signal_idx, "signal_vs_ma5"]),
                    "rebound_ratio": rebound_ratio,
                    "signal_atr20": float(df.at[signal_idx, f"atr_{ATR_PERIOD}"]) if pd.notna(df.at[signal_idx, f"atr_{ATR_PERIOD}"]) else np.nan,
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
    later = daily_df[daily_df["date"] > signal_date]
    if later.empty:
        return None, None, None
    entry_idx = int(later.index[0])
    entry_date = pd.Timestamp(daily_df.at[entry_idx, "date"])
    entry_open = float(daily_df.at[entry_idx, "open"])
    if not np.isfinite(entry_open) or entry_open <= 0:
        return None, None, None
    return entry_idx, entry_date, entry_open


def confirm_5min_entry(min5_df: pd.DataFrame | None, entry_date: pd.Timestamp, signal_low: float) -> dict[str, Any] | None:
    if min5_df is None or min5_df.empty:
        return None
    day_df = min5_df[min5_df["date"] == entry_date].sort_values("datetime").reset_index(drop=True)
    if len(day_df) <= CONFIRM_BARS:
        return None
    window_df = day_df.head(CONFIRM_BARS)
    session_open = float(window_df.iloc[0]["open"])
    if not np.isfinite(session_open) or session_open <= 0:
        return None
    low_ok = bool(window_df["low"].min() >= signal_low)
    reclaim_ok = bool((window_df["close"] >= session_open).any())
    if not (low_ok and reclaim_ok):
        return None
    exec_row = day_df.iloc[CONFIRM_BARS]
    entry_price = float(exec_row["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    return {
        "entry_price": entry_price,
        "entry_dt": pd.Timestamp(exec_row["datetime"]),
        "entry_time": str(exec_row["time"]),
        "session_open": session_open,
        "low_ok": low_ok,
        "reclaim_ok": reclaim_ok,
        "window_low": float(window_df["low"].min()),
    }


def add_horizon_returns(record: dict[str, Any], daily_df: pd.DataFrame, entry_idx: int, entry_price: float) -> dict[str, Any] | None:
    out = dict(record)
    for h in HORIZONS:
        exit_idx = entry_idx + h - 1
        if exit_idx >= len(daily_df):
            out[f"ret_{h}d"] = np.nan
            out[f"success_{h}d"] = np.nan
            out[f"exit_date_{h}d"] = None
            continue
        exit_price = float(daily_df.at[exit_idx, "close"])
        if not np.isfinite(exit_price) or exit_price <= 0:
            out[f"ret_{h}d"] = np.nan
            out[f"success_{h}d"] = np.nan
            out[f"exit_date_{h}d"] = None
            continue
        ret = exit_price / entry_price - 1.0
        out[f"ret_{h}d"] = ret
        out[f"success_{h}d"] = ret > 0
        out[f"exit_date_{h}d"] = pd.Timestamp(daily_df.at[exit_idx, "date"])
    return out


def simulate_atr_exit_trade(
    record: dict[str, Any],
    daily_df: pd.DataFrame,
    entry_idx: int,
    signal_close: float,
    signal_low: float,
    atr20: float,
) -> dict[str, Any] | None:
    if not np.isfinite(atr20) or atr20 <= 0 or not np.isfinite(signal_close) or signal_close <= 0:
        return None
    entry_price = float(record["entry_price"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    tp_pct = ATR_K * atr20 / signal_close
    tp_price = entry_price * (1.0 + tp_pct)
    last_idx = min(len(daily_df) - 1, entry_idx + ATR_HOLD_DAYS - 1)
    exit_idx = last_idx
    exit_price = float(daily_df.at[last_idx, "close"])
    exit_reason = "max_hold_close"

    for idx in range(entry_idx, last_idx + 1):
        day_low = float(daily_df.at[idx, "low"])
        day_high = float(daily_df.at[idx, "high"])
        if np.isfinite(day_low) and day_low <= signal_low:
            exit_idx = idx
            exit_price = float(daily_df.at[idx, "close"])
            exit_reason = "signal_low_same_day_close"
            break
        if np.isfinite(day_high) and day_high >= tp_price:
            next_idx = idx + 1
            if next_idx < len(daily_df):
                exit_idx = next_idx
                exit_price = float(daily_df.at[next_idx, "open"])
                exit_reason = "atr_tp_next_day_open"
            else:
                exit_idx = idx
                exit_price = float(daily_df.at[idx, "close"])
                exit_reason = "atr_tp_fallback_same_day_close"
            break

    out = dict(record)
    exit_date = pd.Timestamp(daily_df.at[exit_idx, "date"])
    return_pct = exit_price / entry_price - 1.0
    out.update(
        {
            "atr_period": ATR_PERIOD,
            "atr_k": ATR_K,
            "max_hold_days": ATR_HOLD_DAYS,
            "trigger_take_profit": float(tp_price),
            "trigger_stop_loss": float(signal_low),
            "exit_date": exit_date,
            "exit_price": float(exit_price),
            "return_pct": float(return_pct),
            "success": bool(return_pct > 0),
            "holding_days": int(exit_idx - entry_idx + 1),
            "exit_reason": exit_reason,
        }
    )
    return out


def simulate_trade_sets(
    selected_df: pd.DataFrame,
    daily_map: dict[str, pd.DataFrame],
    min5_dir: Path,
    exit_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_all: list[dict[str, Any]] = []
    baseline_confirmed_subset: list[dict[str, Any]] = []
    confirm_entry: list[dict[str, Any]] = []
    minute_cache: dict[str, pd.DataFrame | None] = {}
    total = len(selected_df)
    for idx, row in enumerate(selected_df.itertuples(index=False), 1):
        code = str(row.code)
        daily_df = daily_map.get(code)
        if daily_df is None or daily_df.empty:
            continue
        signal_date = pd.Timestamp(row.signal_date)
        entry_idx, entry_date, base_entry_open = find_next_daily_entry(daily_df, signal_date)
        if entry_idx is None or entry_date is None or base_entry_open is None:
            continue
        base_record = {
            "mode": "baseline_all",
            "code": code,
            "signal_date": signal_date,
            "entry_date": entry_date,
            "entry_price": base_entry_open,
            "signal_low": float(row.signal_low),
            "sort_score": float(row.sort_score),
            "daily_rank": int(row.daily_rank),
            "signal_vs_ma5": float(row.signal_vs_ma5),
            "rebound_ratio": float(row.rebound_ratio),
            "signal_close": float(row.signal_close),
            "signal_atr20": float(row.signal_atr20) if pd.notna(row.signal_atr20) else np.nan,
        }
        if exit_mode == "atr":
            base_record = simulate_atr_exit_trade(
                base_record,
                daily_df,
                entry_idx,
                float(row.signal_close),
                float(row.signal_low),
                float(row.signal_atr20) if pd.notna(row.signal_atr20) else np.nan,
            )
        else:
            base_record = add_horizon_returns(base_record, daily_df, entry_idx, base_entry_open)
        if base_record is not None:
            baseline_all.append(base_record)

        if code not in minute_cache:
            min5_path = min5_dir / f"{code}.txt"
            minute_cache[code] = load_minute_df(min5_path) if min5_path.exists() else None
        min5_df = minute_cache[code]
        confirm = confirm_5min_entry(min5_df, entry_date, float(row.signal_low))
        if confirm is None:
            if idx % 500 == 0 or idx == total:
                print(f"5min确认进度: {idx}/{total}")
            continue

        subset_record = dict(base_record)
        subset_record["mode"] = "baseline_confirmed_subset"
        subset_record["confirm_entry_time"] = confirm["entry_time"]
        subset_record["confirm_window_low"] = confirm["window_low"]
        baseline_confirmed_subset.append(subset_record)

        confirm_record = {
            "mode": "confirm_5min_entry",
            "code": code,
            "signal_date": signal_date,
            "entry_date": entry_date,
            "entry_price": float(confirm["entry_price"]),
            "entry_time": confirm["entry_time"],
            "entry_dt": confirm["entry_dt"],
            "signal_low": float(row.signal_low),
            "sort_score": float(row.sort_score),
            "daily_rank": int(row.daily_rank),
            "signal_vs_ma5": float(row.signal_vs_ma5),
            "rebound_ratio": float(row.rebound_ratio),
            "confirm_window_low": confirm["window_low"],
            "session_open": confirm["session_open"],
            "signal_close": float(row.signal_close),
            "signal_atr20": float(row.signal_atr20) if pd.notna(row.signal_atr20) else np.nan,
        }
        if exit_mode == "atr":
            confirm_record = simulate_atr_exit_trade(
                confirm_record,
                daily_df,
                entry_idx,
                float(row.signal_close),
                float(row.signal_low),
                float(row.signal_atr20) if pd.notna(row.signal_atr20) else np.nan,
            )
        else:
            confirm_record = add_horizon_returns(confirm_record, daily_df, entry_idx, float(confirm["entry_price"]))
        if confirm_record is not None:
            confirm_entry.append(confirm_record)
        if idx % 500 == 0 or idx == total:
            print(f"5min确认进度: {idx}/{total}")
    return pd.DataFrame(baseline_all), pd.DataFrame(baseline_confirmed_subset), pd.DataFrame(confirm_entry)


def summarize_horizons(name: str, trade_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trade_df.empty:
        return pd.DataFrame()
    for h in HORIZONS:
        ret_col = f"ret_{h}d"
        success_col = f"success_{h}d"
        g = trade_df.dropna(subset=[ret_col]).copy()
        if g.empty:
            continue
        portfolio_input = g[["signal_date", "code", "sort_score", ret_col]].rename(columns={ret_col: "ret"})
        portfolio_df = build_portfolio_curve(portfolio_input)
        eq = compute_equity_metrics(portfolio_df)
        rows.append(
            {
                "mode": name,
                "horizon_days": h,
                "sample_count": int(len(g)),
                "coverage_days": int(g["signal_date"].nunique()),
                "avg_return": float(g[ret_col].mean()),
                "win_rate": float(g[success_col].astype(float).mean()),
                "max_consecutive_failures": int(max_consecutive_failures(g[success_col].astype(bool).tolist())),
                "annual_return_signal_basket": float(eq["annual_return"]) if pd.notna(eq["annual_return"]) else np.nan,
                "max_drawdown_signal_basket": float(eq["max_drawdown"]) if pd.notna(eq["max_drawdown"]) else np.nan,
                "final_equity_signal_basket": float(eq["final_equity"]) if pd.notna(eq["final_equity"]) else np.nan,
                "equity_days_signal_basket": int(eq["equity_days"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_atr_trades(name: str, trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()
    g = trade_df.dropna(subset=["return_pct"]).copy()
    if g.empty:
        return pd.DataFrame()
    portfolio_input = g[["signal_date", "code", "sort_score", "return_pct"]].rename(columns={"return_pct": "ret"})
    portfolio_df = build_portfolio_curve(portfolio_input)
    eq = compute_equity_metrics(portfolio_df)
    return pd.DataFrame(
        [
            {
                "mode": name,
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
                "atr_period": ATR_PERIOD,
                "atr_k": ATR_K,
                "max_hold_days": ATR_HOLD_DAYS,
            }
        ]
    )


def build_overall_summary(
    selected_df: pd.DataFrame,
    baseline_all_df: pd.DataFrame,
    baseline_confirmed_df: pd.DataFrame,
    confirm_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    exit_mode: str,
) -> dict[str, Any]:
    def best_row(mode: str) -> dict[str, Any] | None:
        g = summary_df[summary_df["mode"] == mode].copy()
        if g.empty:
            return None
        sort_cols = ["win_rate", "sample_count"]
        if exit_mode == "atr":
            sort_cols = ["avg_trade_return"] + sort_cols
        else:
            sort_cols = ["avg_return"] + sort_cols
        return g.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0].to_dict()

    confirm_rate = len(baseline_confirmed_df) / len(baseline_all_df) if len(baseline_all_df) > 0 else np.nan
    timeline = {
        "signal_date": "日线 brick 信号日",
        "baseline_entry": "signal_date 次日开盘直接买入",
        "confirm_window": "entry_date 前 30 分钟（前 6 根 5min K 线）",
        "confirm_condition": "前30分钟最低价不跌破 signal_low，且至少一根5min收盘价重新站上当日开盘价",
        "confirm_entry": "若确认成立，则在第7根5min（通常10:05）开盘买入",
    }
    if exit_mode == "atr":
        timeline["exit_definition"] = "ATR20×2.0 止盈，触发后次日开盘卖；signal_low 同日收盘止损；最大持有4天"
    else:
        timeline["exit_definition"] = "本实验只比较买入优化，不引入分钟级卖出；收益按固定 horizon 的日线收盘计算"

    return {
        "daily_signal_count": int(len(selected_df)),
        "baseline_all_count": int(len(baseline_all_df)),
        "confirmed_subset_count": int(len(baseline_confirmed_df)),
        "confirm_entry_count": int(len(confirm_df)),
        "confirm_rate_vs_baseline": float(confirm_rate) if pd.notna(confirm_rate) else np.nan,
        "best_baseline_all": best_row("baseline_all"),
        "best_baseline_confirmed_subset": best_row("baseline_confirmed_subset"),
        "best_confirm_5min_entry": best_row("confirm_5min_entry"),
        "exit_mode": exit_mode,
        "timeline_definition": timeline,
    }


def run_experiment(args: argparse.Namespace, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "building_daily_signals")
    selected_df, daily_map = build_selected_signals(Path(args.daily_dir), file_limit=args.file_limit)
    if selected_df.empty:
        raise RuntimeError("当前参数下没有生成任何 brick 日线信号")
    selected_df.to_csv(result_dir / "selected_daily_signals.csv", index=False, encoding="utf-8-sig")

    update_progress(result_dir, "simulating_5min_confirm", selected_signal_count=int(len(selected_df)))
    baseline_all_df, baseline_confirmed_df, confirm_df = simulate_trade_sets(selected_df, daily_map, Path(args.min5_dir), args.exit_mode)
    if baseline_all_df.empty:
        raise RuntimeError("baseline_all 没有任何可交易样本")
    baseline_all_df.to_csv(result_dir / "baseline_all_trades.csv", index=False, encoding="utf-8-sig")
    baseline_confirmed_df.to_csv(result_dir / "baseline_confirmed_subset_trades.csv", index=False, encoding="utf-8-sig")
    confirm_df.to_csv(result_dir / "confirm_5min_entry_trades.csv", index=False, encoding="utf-8-sig")

    update_progress(
        result_dir,
        "summarizing",
        baseline_all_count=int(len(baseline_all_df)),
        confirmed_subset_count=int(len(baseline_confirmed_df)),
        confirm_entry_count=int(len(confirm_df)),
    )
    if args.exit_mode == "atr":
        summary_df = pd.concat(
            [
                summarize_atr_trades("baseline_all", baseline_all_df),
                summarize_atr_trades("baseline_confirmed_subset", baseline_confirmed_df),
                summarize_atr_trades("confirm_5min_entry", confirm_df),
            ],
            ignore_index=True,
        )
        summary_df.to_csv(result_dir / "atr_trade_summary.csv", index=False, encoding="utf-8-sig")
    else:
        summary_df = pd.concat(
            [
                summarize_horizons("baseline_all", baseline_all_df),
                summarize_horizons("baseline_confirmed_subset", baseline_confirmed_df),
                summarize_horizons("confirm_5min_entry", confirm_df),
            ],
            ignore_index=True,
        )
        summary_df.to_csv(result_dir / "horizon_summary.csv", index=False, encoding="utf-8-sig")

    overall = build_overall_summary(selected_df, baseline_all_df, baseline_confirmed_df, confirm_df, summary_df, args.exit_mode)
    write_json(result_dir / "summary.json", overall)
    update_progress(result_dir, "finished", summary_path=str(result_dir / "summary.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK 日线信号 + 次日 5min 确认买点研究")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--exit-mode", choices=["horizon", "atr"], default="atr")
    parser.add_argument("--daily-dir", default=str(DAILY_DIR))
    parser.add_argument("--min5-dir", default=str(MIN5_DIR))
    parser.add_argument("--file-limit", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.mode == "smoke" and args.file_limit <= 0:
        args.file_limit = 300
    result_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_daily_5min_confirm_buy_v1_{args.mode}_{timestamp}"
    try:
        run_experiment(args, result_dir)
    except BaseException as exc:  # pragma: no cover
        write_error(result_dir, exc)
        raise


if __name__ == "__main__":
    main()
