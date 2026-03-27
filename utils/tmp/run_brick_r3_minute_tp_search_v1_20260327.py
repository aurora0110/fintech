from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
SOURCE_RESULT_DIR = RESULT_ROOT / "brick_relaxed_seq_model_search_v1_full_20260326_r3"
SOURCE_CANDIDATES = SOURCE_RESULT_DIR / "relaxed_selected_candidates.csv"
DAILY_DIR = ROOT / "data" / "20260324"
MIN5_DIR = ROOT / "data" / "202603245min"
HYBRID_PATH = ROOT / "utils" / "tmp" / "run_brick_hybrid_local_search_v1_20260325.py"
REAL_ACCOUNT_PATH = ROOT / "utils" / "tmp" / "run_brick_real_account_compare_v1_20260326.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TP_PCT = 0.055
STOP_BASE = "min_oc"
STOP_MULTIPLIER = 1.0
BUY_GAP_LIMIT = 0.04
MAX_HOLD_DAYS = 3
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
_DAILY_STEM_MAP: dict[str, str] | None = None
_MIN5_STEM_MAP: dict[str, str] | None = None


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


hybrid = load_module(HYBRID_PATH, "brick_r3_tp_hybrid")
real_account = load_module(REAL_ACCOUNT_PATH, "brick_r3_tp_real_account")


TP_PROFILES: list[dict[str, str]] = [
    {"name": "tp_immediate", "family": "baseline"},
    {"name": "tp_close", "family": "baseline"},
    {"name": "tp_next_open", "family": "baseline"},
    {"name": "tp_break_prev_low", "family": "price"},
    {"name": "tp_close_below_ema5", "family": "price"},
    {"name": "tp_close_below_vwap", "family": "price"},
    {"name": "tp_two_bar_weak", "family": "price"},
    {"name": "tp_macd_hist_neg", "family": "indicator"},
    {"name": "tp_rsi_below_70", "family": "indicator"},
    {"name": "tp_kdj_j_below_80", "family": "indicator"},
    {"name": "tp_ema5_below_ema10", "family": "indicator"},
]


@dataclass
class SimResult:
    trade: dict[str, Any] | None
    skipped: bool = False
    skip_reason: str = ""


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_error(result_dir: Path, exc: BaseException) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (result_dir / "error.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))


def _code_key(value: Any) -> str:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return digits
    return digits[-6:] if len(digits) >= 6 else digits.zfill(6)


def _build_stem_map(directory: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(directory.glob("*.txt")):
        key = _code_key(path.stem)
        if key:
            mapping[key] = path.stem
    return mapping


def _resolve_daily_stem(code: Any) -> str:
    global _DAILY_STEM_MAP
    if _DAILY_STEM_MAP is None:
        _DAILY_STEM_MAP = _build_stem_map(DAILY_DIR)
    text = str(code)
    key = _code_key(text)
    return _DAILY_STEM_MAP.get(key, text)


def load_source_candidates(source_csv: Path, score_col: str, strategy_key: str, file_limit_codes: int, date_limit: int) -> pd.DataFrame:
    sample_cols = pd.read_csv(source_csv, nrows=0).columns.tolist()
    parse_dates = [col for col in ["signal_date", "entry_date", "exit_date"] if col in sample_cols]
    df = pd.read_csv(source_csv, parse_dates=parse_dates)
    sort_cols = ["signal_date", score_col, "code"] if score_col in df.columns else ["signal_date", "code"]
    ascending = [True, False, True] if len(sort_cols) == 3 else [True, True]
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    if file_limit_codes > 0:
        keep_codes = sorted(df["code"].astype(str).unique())[:file_limit_codes]
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    if date_limit > 0:
        keep_dates = sorted(pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d").unique())[:date_limit]
        df = df[df["signal_date"].dt.strftime("%Y-%m-%d").isin(keep_dates)].copy()
    if score_col not in df.columns:
        raise RuntimeError(f"源候选缺少排序列: {score_col}")
    df["code"] = df["code"].map(_resolve_daily_stem)
    df["sort_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    df["strategy_key"] = strategy_key
    return df.reset_index(drop=True)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1 / period, adjust=False).mean()
    loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def _compute_kdj(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    low_n = df["low"].rolling(period, min_periods=3).min()
    high_n = df["high"].rolling(period, min_periods=3).max()
    denom = (high_n - low_n).replace(0.0, np.nan)
    rsv = (df["close"] - low_n) / denom * 100.0
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"K": k.fillna(50.0), "D": d.fillna(50.0), "J": j.fillna(50.0)}, index=df.index)


def _prepare_min5_indicators(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    out = df.copy().sort_values("datetime").reset_index(drop=True)
    out["ema5"] = out["close"].ewm(span=5, adjust=False).mean()
    out["ema10"] = out["close"].ewm(span=10, adjust=False).mean()
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = 2.0 * (dif - dea)
    out["rsi14"] = _compute_rsi(out["close"], 14)
    kdj = _compute_kdj(out)
    out["kdj_j"] = kdj["J"].values
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    out["tpv"] = out["close"] * volume
    out["cum_tpv"] = out.groupby("date")["tpv"].cumsum()
    out["cum_vol"] = volume.groupby(out["date"]).cumsum()
    out["vwap"] = out["cum_tpv"] / out["cum_vol"].replace(0.0, np.nan)
    out["vwap"] = out["vwap"].ffill().fillna(out["close"])
    out["prev_low"] = out["low"].shift(1)
    out["prev_close"] = out["close"].shift(1)
    return out


def _next_trade_date(daily_df: pd.DataFrame, current_date: pd.Timestamp) -> pd.Timestamp | None:
    later = daily_df[daily_df["date"] > current_date]["date"]
    if later.empty:
        return None
    return pd.Timestamp(later.iloc[0])


def _indicator_triggered(profile_name: str, bar: pd.Series, prev_bar: pd.Series | None, weak_count: int) -> tuple[bool, int]:
    triggered = False
    next_weak_count = weak_count
    if profile_name == "tp_break_prev_low":
        prev_low = float(prev_bar["low"]) if prev_bar is not None and pd.notna(prev_bar["low"]) else np.nan
        triggered = bool(np.isfinite(prev_low) and float(bar["low"]) < prev_low)
    elif profile_name == "tp_close_below_ema5":
        triggered = bool(float(bar["close"]) < float(bar["ema5"]))
    elif profile_name == "tp_close_below_vwap":
        triggered = bool(float(bar["close"]) < float(bar["vwap"]))
    elif profile_name == "tp_two_bar_weak":
        is_weak = (float(bar["close"]) < float(bar["open"])) or (float(bar["close"]) < float(bar["prev_close"]) if pd.notna(bar["prev_close"]) else False)
        next_weak_count = weak_count + 1 if is_weak else 0
        triggered = next_weak_count >= 2
    elif profile_name == "tp_macd_hist_neg":
        triggered = bool(float(bar["macd_hist"]) < 0.0)
    elif profile_name == "tp_rsi_below_70":
        triggered = bool(float(bar["rsi14"]) < 70.0)
    elif profile_name == "tp_kdj_j_below_80":
        triggered = bool(float(bar["kdj_j"]) < 80.0)
    elif profile_name == "tp_ema5_below_ema10":
        triggered = bool(float(bar["ema5"]) < float(bar["ema10"]))
    return triggered, next_weak_count


def simulate_one_trade_profile(
    code: str,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    signal_idx: int,
    daily_df: pd.DataFrame,
    min5_df: pd.DataFrame | None,
    profile_name: str,
    stop_base: str,
    stop_multiplier: float,
    tp_pct: float,
    buy_gap_limit: float,
    max_hold_days: int,
    strategy_key_prefix: str,
) -> SimResult:
    entry_rows = daily_df[daily_df["date"] == entry_date]
    signal_rows = daily_df[daily_df["date"] == signal_date]
    if entry_rows.empty or signal_rows.empty:
        return SimResult(trade=None, skipped=True, skip_reason="missing_daily_row")

    entry_row = entry_rows.iloc[0]
    signal_row = signal_rows.iloc[0]
    entry_open = float(entry_row["open"])
    entry_close = float(entry_row["close"])
    entry_low = float(entry_row["low"])
    signal_close = float(signal_row["close"])
    if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(signal_close) or signal_close <= 0:
        return SimResult(trade=None, skipped=True, skip_reason="invalid_entry_or_signal")
    if entry_open / signal_close - 1.0 > buy_gap_limit:
        return SimResult(trade=None, skipped=True, skip_reason="gap_gt_4pct")

    stop_base_price = hybrid.resolve_stop_base(entry_open, entry_close, entry_low, stop_base)
    stop_price = float(stop_base_price) * stop_multiplier
    tp_price = entry_open * (1.0 + tp_pct)
    eligible_dates = daily_df[daily_df["date"] > entry_date]["date"].head(max_hold_days).tolist()
    if not eligible_dates:
        return SimResult(trade=None, skipped=True, skip_reason="no_exit_window")

    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    trigger_source: str = "max_hold"
    armed_tp_date: pd.Timestamp | None = None

    for d in eligible_dates:
        day_daily = daily_df[daily_df["date"] == d]
        if day_daily.empty:
            continue
        day_daily_row = day_daily.iloc[0]
        day_min = pd.DataFrame()
        if min5_df is not None:
            day_min = min5_df[min5_df["date"] == d].copy()

        if day_min.empty:
            day_low = float(day_daily_row["low"])
            day_high = float(day_daily_row["high"])
            if day_low <= stop_price:
                exit_date = pd.Timestamp(d)
                exit_price = stop_price
                exit_reason = "sl_same_day"
                trigger_source = "daily_fallback"
                break
            if day_high >= tp_price:
                armed_tp_date = pd.Timestamp(d)
                trigger_source = "daily_fallback"
                if profile_name == "tp_immediate":
                    exit_date = pd.Timestamp(d)
                    exit_price = tp_price
                    exit_reason = "tp_same_day_immediate"
                elif profile_name == "tp_close":
                    exit_date = pd.Timestamp(d)
                    exit_price = float(day_daily_row["close"])
                    exit_reason = "tp_same_day_close"
                else:
                    next_date = _next_trade_date(daily_df, pd.Timestamp(d))
                    if next_date is None:
                        exit_date = pd.Timestamp(d)
                        exit_price = float(day_daily_row["close"])
                        exit_reason = "tp_same_day_fallback_close"
                    else:
                        next_row = daily_df[daily_df["date"] == next_date]
                        if next_row.empty:
                            return SimResult(trade=None, skipped=True, skip_reason="missing_next_exec_day")
                        exit_date = next_date
                        exit_price = float(next_row.iloc[0]["open"])
                        exit_reason = "tp_next_open"
                break
            continue

        armed = False
        weak_count = 0
        prev_bar: pd.Series | None = None
        for idx, bar in day_min.reset_index(drop=True).iterrows():
            day_low = float(bar["low"])
            day_high = float(bar["high"])
            if day_low <= stop_price:
                exit_date = pd.Timestamp(d)
                exit_price = stop_price
                exit_reason = "sl_same_day"
                trigger_source = "5min"
                break

            if not armed and day_high >= tp_price:
                armed = True
                armed_tp_date = pd.Timestamp(d)
                trigger_source = "5min"
                if profile_name == "tp_immediate":
                    exit_date = pd.Timestamp(d)
                    exit_price = tp_price
                    exit_reason = "tp_same_day_immediate"
                    break
                if profile_name == "tp_close":
                    exit_date = pd.Timestamp(d)
                    exit_price = float(day_daily_row["close"])
                    exit_reason = "tp_same_day_close"
                    break
                if profile_name == "tp_next_open":
                    next_date = _next_trade_date(daily_df, pd.Timestamp(d))
                    if next_date is None:
                        exit_date = pd.Timestamp(d)
                        exit_price = float(day_daily_row["close"])
                        exit_reason = "tp_same_day_fallback_close"
                    else:
                        next_row = daily_df[daily_df["date"] == next_date]
                        if next_row.empty:
                            return SimResult(trade=None, skipped=True, skip_reason="missing_next_exec_day")
                        exit_date = next_date
                        exit_price = float(next_row.iloc[0]["open"])
                        exit_reason = "tp_next_open"
                    break
                prev_bar = bar
                continue

            if armed:
                triggered, weak_count = _indicator_triggered(profile_name, bar, prev_bar, weak_count)
                if triggered:
                    exit_date = pd.Timestamp(d)
                    exit_price = float(bar["close"])
                    exit_reason = f"{profile_name}_same_day_close"
                    break
            prev_bar = bar

        if exit_reason is not None:
            break

        if armed and exit_reason is None:
            next_date = _next_trade_date(daily_df, pd.Timestamp(d))
            if next_date is None:
                exit_date = pd.Timestamp(d)
                exit_price = float(day_daily_row["close"])
                exit_reason = f"{profile_name}_fallback_close"
            else:
                next_row = daily_df[daily_df["date"] == next_date]
                if next_row.empty:
                    return SimResult(trade=None, skipped=True, skip_reason="missing_next_exec_day")
                exit_date = next_date
                exit_price = float(next_row.iloc[0]["open"])
                exit_reason = f"{profile_name}_fallback_next_open"
            break

    if exit_reason is None or exit_date is None or exit_price is None:
        final_date = pd.Timestamp(eligible_dates[-1])
        final_row = daily_df[daily_df["date"] == final_date]
        if final_row.empty:
            return SimResult(trade=None, skipped=True, skip_reason="missing_forced_exit_day")
        exit_date = final_date
        exit_price = float(final_row.iloc[0]["close"])
        exit_reason = "max_hold_close"

    return_pct = float(exit_price) / entry_open - 1.0
    return SimResult(
        trade={
            "code": code,
            "signal_idx": int(signal_idx),
            "signal_date": pd.Timestamp(signal_date),
            "entry_date": pd.Timestamp(entry_date),
            "exit_date": pd.Timestamp(exit_date),
            "entry_price": float(entry_open),
            "exit_price": float(exit_price),
            "return_pct": float(return_pct),
            "hold_days": int((pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days),
            "exit_reason": str(exit_reason),
            "trigger_take_profit": float(tp_price),
            "trigger_stop_loss": float(stop_price),
            "stop_base_price": float(stop_base_price),
            "minute_source": "5min" if min5_df is not None else "missing_5min",
            "trigger_source": trigger_source,
            "tp_profile": profile_name,
            "tp_arm_date": pd.Timestamp(armed_tp_date) if armed_tp_date is not None else pd.NaT,
            "strategy_key": f"{strategy_key_prefix}|{profile_name}",
        }
    )


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    stop_base = str(payload["stop_base"])
    stop_multiplier = float(payload["stop_multiplier"])
    tp_pct = float(payload["tp_pct"])
    buy_gap_limit = float(payload["buy_gap_limit"])
    max_hold_days = int(payload["max_hold_days"])
    strategy_key_prefix = str(payload["strategy_key_prefix"])
    daily_path = DAILY_DIR / f"{code}.txt"
    if not daily_path.exists():
        skipped = []
        for item in rows:
            for profile in TP_PROFILES:
                skipped.append(
                    {
                        "code": code,
                        "signal_date": item["signal_date"],
                        "entry_date": item["entry_date"],
                        "tp_profile": profile["name"],
                        "reason": "missing_daily_series",
                    }
                )
        return [], skipped

    daily_df = hybrid.base.load_daily_df(daily_path)
    min5_df = None
    min5_path = MIN5_DIR / f"{code}.txt"
    if min5_path.exists():
        loaded = hybrid.base.load_minute_df(min5_path)
        min5_df = _prepare_min5_indicators(loaded) if not loaded.empty else None

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in rows:
        for profile in TP_PROFILES:
            sim = simulate_one_trade_profile(
                code=code,
                signal_date=pd.Timestamp(item["signal_date"]),
                entry_date=pd.Timestamp(item["entry_date"]),
                signal_idx=int(item.get("signal_idx", -1)),
                daily_df=daily_df,
                min5_df=min5_df,
                profile_name=profile["name"],
                stop_base=stop_base,
                stop_multiplier=stop_multiplier,
                tp_pct=tp_pct,
                buy_gap_limit=buy_gap_limit,
                max_hold_days=max_hold_days,
                strategy_key_prefix=strategy_key_prefix,
            )
            if sim.skipped or sim.trade is None:
                skipped.append(
                    {
                        "code": code,
                        "signal_date": item["signal_date"],
                        "entry_date": item["entry_date"],
                        "tp_profile": profile["name"],
                        "reason": sim.skip_reason,
                    }
                )
                continue
            tr = sim.trade
            tr["sort_score"] = float(item["sort_score"])
            trades.append(tr)
    return trades, skipped


def _summarize_signal_basket(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile_name, g in trades.groupby("tp_profile", sort=True):
        row = hybrid.base.summarize_trades(g, f"r3|{profile_name}")
        row["tp_profile"] = profile_name
        row["trigger_source_5min_ratio"] = float((g["trigger_source"] == "5min").mean())
        row["trigger_source_daily_ratio"] = float((g["trigger_source"] == "daily_fallback").mean())
        row["tp_armed_ratio"] = float(g["tp_arm_date"].notna().mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["annual_return_signal_basket", "avg_trade_return", "success_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _summarize_account(trades: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    all_codes = sorted(trades["code"].astype(str).unique())
    market_dates, close_map = real_account.build_close_map(
        all_codes,
        progress_cb=lambda done, total: update_progress(result_dir, "building_close_map", done_codes=int(done), total_codes=int(total)),
    )
    if len(market_dates) == 0:
        raise RuntimeError("无法构建账户层 close_map")

    rows: list[dict[str, Any]] = []
    config = real_account.AccountConfig()
    for profile_name, g in trades.groupby("tp_profile", sort=True):
        use = g.copy()
        use["strategy_key"] = f"r3|{profile_name}"
        equity_df, executed_df, summary = real_account.simulate_real_account(use, close_map, market_dates, config)
        equity_df.to_csv(result_dir / f"equity_{profile_name}.csv", index=False, encoding="utf-8-sig")
        executed_df.to_csv(result_dir / f"executed_{profile_name}.csv", index=False, encoding="utf-8-sig")
        row = {"tp_profile": profile_name, **summary}
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["annual_return", "avg_trade_return", "success_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def run_search(
    result_dir: Path,
    source_csv: Path,
    score_col: str,
    strategy_key: str,
    file_limit_codes: int,
    date_limit: int,
    max_workers: int,
    stop_base: str,
    stop_multiplier: float,
    tp_pct: float,
    buy_gap_limit: float,
    max_hold_days: int,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(
        result_dir,
        "loading_source",
        source_csv=str(source_csv),
        score_col=score_col,
        strategy_key=strategy_key,
        file_limit_codes=file_limit_codes,
        date_limit=date_limit,
        max_workers=max_workers,
        stop_base=stop_base,
        stop_multiplier=stop_multiplier,
        tp_pct=tp_pct,
    )
    candidates = load_source_candidates(
        source_csv=source_csv,
        score_col=score_col,
        strategy_key=strategy_key,
        file_limit_codes=file_limit_codes,
        date_limit=date_limit,
    )
    candidates.to_csv(result_dir / "source_candidates.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        raise RuntimeError("r3 源候选为空")

    grouped_payloads = []
    if "signal_idx" not in candidates.columns:
        candidates["signal_idx"] = -1
    for code, g in candidates.groupby("code", sort=True):
        grouped_payloads.append(
            {
                "code": str(code),
                "rows": g[["signal_date", "entry_date", "signal_idx", "sort_score"]].to_dict("records"),
                "stop_base": stop_base,
                "stop_multiplier": float(stop_multiplier),
                "tp_pct": float(tp_pct),
                "buy_gap_limit": float(buy_gap_limit),
                "max_hold_days": int(max_hold_days),
                "strategy_key_prefix": str(strategy_key),
            }
        )

    total_codes = len(grouped_payloads)
    total_jobs = len(candidates) * len(TP_PROFILES)
    trade_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(simulate_code_bundle, payload): payload["code"] for payload in grouped_payloads}
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            code = future_map[future]
            trades_part, skipped_part = future.result()
            trade_rows.extend(trades_part)
            skipped_rows.extend(skipped_part)
            if completed == 1 or completed % 25 == 0 or completed == total_codes:
                update_progress(
                    result_dir,
                    "simulating_trades",
                    done_codes=completed,
                    total_codes=total_codes,
                    done_jobs=len(trade_rows) + len(skipped_rows),
                    total_jobs=total_jobs,
                    last_code=code,
                )

    trades = pd.DataFrame(trade_rows).sort_values(["tp_profile", "signal_date", "code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows).sort_values(["tp_profile", "signal_date", "code"], na_position="last").reset_index(drop=True) if skipped_rows else pd.DataFrame()
    trades.to_csv(result_dir / "tp_profile_trades.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(result_dir / "tp_profile_skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "trades_ready", trade_count=int(len(trades)), skipped_count=int(len(skipped)))

    signal_summary = _summarize_signal_basket(trades)
    signal_summary.to_csv(result_dir / "signal_basket_summary.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "signal_basket_ready", profile_count=int(len(signal_summary)))

    account_summary = _summarize_account(trades, result_dir)
    account_summary.to_csv(result_dir / "account_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "source_candidates": str(source_csv),
            "fixed_buy_model": strategy_key,
            "stop_base": stop_base,
            "stop_multiplier": stop_multiplier,
            "tp_pct": tp_pct,
            "buy_gap_limit": buy_gap_limit,
            "max_hold_days": max_hold_days,
            "minute_priority": "5min_then_daily_fallback",
            "same_bar_priority": "stop_first",
        },
        "best_signal_basket_profile": signal_summary.iloc[0].to_dict() if not signal_summary.empty else {},
        "best_account_profile": account_summary.iloc[0].to_dict() if not account_summary.empty else {},
        "signal_profile_count": int(len(signal_summary)),
        "account_profile_count": int(len(account_summary)),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定 r3 买点，搜索 5 分钟止盈方式")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--source-csv", type=str, default=str(SOURCE_CANDIDATES))
    parser.add_argument("--score-col", type=str, default="rank_score")
    parser.add_argument("--strategy-key", type=str, default="r3_xgb_len5")
    parser.add_argument("--stop-base", type=str, default=STOP_BASE)
    parser.add_argument("--stop-multiplier", type=float, default=STOP_MULTIPLIER)
    parser.add_argument("--tp-pct", type=float, default=TP_PCT)
    parser.add_argument("--buy-gap-limit", type=float, default=BUY_GAP_LIMIT)
    parser.add_argument("--max-hold-days", type=int, default=MAX_HOLD_DAYS)
    parser.add_argument("--file-limit-codes", type=int, default=200)
    parser.add_argument("--date-limit", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_r3_minute_tp_search_v1_{args.mode}_{timestamp}"
    file_limit_codes = int(args.file_limit_codes)
    date_limit = int(args.date_limit)
    if args.mode == "full":
        file_limit_codes = 0
        date_limit = 0
    try:
        run_search(
            result_dir=output_dir,
            source_csv=Path(args.source_csv),
            score_col=str(args.score_col),
            strategy_key=str(args.strategy_key),
            file_limit_codes=file_limit_codes,
            date_limit=date_limit,
            max_workers=int(args.max_workers),
            stop_base=str(args.stop_base),
            stop_multiplier=float(args.stop_multiplier),
            tp_pct=float(args.tp_pct),
            buy_gap_limit=float(args.buy_gap_limit),
            max_hold_days=int(args.max_hold_days),
        )
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
