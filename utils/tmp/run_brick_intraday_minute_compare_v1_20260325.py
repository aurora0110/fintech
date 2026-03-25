from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
DAILY_DIR = ROOT / "data" / "20260324"
MIN5_DIR = ROOT / "data" / "202603245min"
MIN1_DIR = ROOT / "data" / "202603241min"
RESULT_ROOT = ROOT / "results"

BRICK_BASE_PATH = ROOT / "utils" / "backtest" / "run_momentum_tail_experiment.py"
BRICK_RANKING_PATH = ROOT / "utils" / "backtest" / "compare_momentum_tail_ranking_models.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
INITIAL_CAPITAL = 1_000_000.0
TOP_N = 10
BUY_GAP_LIMIT = 0.04
TAKE_PROFIT = 0.03
MAX_HOLD_DAYS = 3


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


brick_base = load_module(BRICK_BASE_PATH, "brick_minute_compare_base")
brick_ranking = load_module(BRICK_RANKING_PATH, "brick_minute_compare_ranking")


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_text_auto(path: Path) -> list[str]:
    for enc in ("gbk", "utf-8", "latin1"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except Exception:
            pass
    raise RuntimeError(f"无法读取文件: {path}")


def load_daily_df(path: Path) -> pd.DataFrame:
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
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    df = df[(df["date"] < EXCLUDE_START) | (df["date"] > EXCLUDE_END)].copy()
    return df


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
    df = df.dropna(subset=["datetime", "date", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
    return df


def build_daily_map(data_dir: Path, file_limit: int = 0, codes: set[str] | None = None) -> dict[str, pd.DataFrame]:
    files = sorted(data_dir.glob("*.txt"))
    if codes is not None:
        files = [p for p in files if p.stem in codes]
    elif file_limit > 0:
        files = files[:file_limit]
    out: dict[str, pd.DataFrame] = {}
    for path in files:
        df = load_daily_df(path)
        if df.empty:
            continue
        out[path.stem] = df
    return out


def build_minute_maps(codes: set[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    min1_map: dict[str, pd.DataFrame] = {}
    min5_map: dict[str, pd.DataFrame] = {}
    for code in codes:
        p1 = MIN1_DIR / f"{code}.txt"
        if p1.exists():
            df1 = load_minute_df(p1)
            if not df1.empty:
                min1_map[code] = df1
        p5 = MIN5_DIR / f"{code}.txt"
        if p5.exists():
            df5 = load_minute_df(p5)
            if not df5.empty:
                min5_map[code] = df5
    return min1_map, min5_map


def build_formal_best_candidates(file_limit: int = 0) -> pd.DataFrame:
    combo = brick_base.Combo(
        rebound_threshold=1.2,
        gain_limit=0.08,
        take_profit=0.03,
        stop_mode="entry_low_x_0.99",
    )
    rows: list[dict[str, Any]] = []
    files = sorted(DAILY_DIR.glob("*.txt"))
    if file_limit > 0:
        files = files[:file_limit]
    for path in files:
        raw_df = load_daily_df(path)
        if raw_df.empty or len(raw_df) < 120:
            continue
        raw_df["code"] = path.stem
        df = brick_base.build_feature_df(raw_df)
        df = brick_ranking.add_long_line(df)
        mask_a = df["pattern_a"] & (df["rebound_ratio"] >= combo.rebound_threshold)
        mask_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
        mask = (
            df["signal_base"]
            & (df["ret1"] <= combo.gain_limit)
            & (mask_a | mask_b)
            & (df["trend_line"] > df["long_line"])
        )
        signal_idxs = np.flatnonzero(mask.to_numpy())
        for signal_idx in signal_idxs:
            signal_idx = int(signal_idx)
            entry_idx = signal_idx + 1
            if entry_idx >= len(df):
                continue
            feat = brick_ranking.compute_signal_features(df, signal_idx)
            rows.append(
                {
                    "code": path.stem,
                    "signal_idx": signal_idx,
                    "signal_date": pd.Timestamp(df.at[signal_idx, "date"]),
                    "entry_date": pd.Timestamp(df.at[entry_idx, "date"]),
                    "candidate_pool": "brick.formal_best",
                    "strategy_family": "brick",
                    "signal_type": "brick_formal_best",
                    "pattern_a": bool(df.at[signal_idx, "pattern_a"]),
                    "pattern_b": bool(df.at[signal_idx, "pattern_b"]),
                    "ret1": float(df.at[signal_idx, "ret1"]),
                    "rebound_ratio": float(df.at[signal_idx, "rebound_ratio"]),
                    "brick_red_len": float(df.at[signal_idx, "brick_red_len"]),
                    "signal_vs_ma5": float(df.at[signal_idx, "signal_vs_ma5"]),
                    "trend_line": float(df.at[signal_idx, "trend_line"]),
                    "long_line": float(df.at[signal_idx, "long_line"]),
                    "signal_low": float(df.at[signal_idx, "low"]),
                    "base_score": 0.0,
                    **feat,
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)
    out = brick_ranking.assign_rank_scores(out)
    out["base_score"] = brick_ranking.build_sort_score(out, "shrink_focus")
    out["daily_pct_rank"] = out.groupby("signal_date")["base_score"].rank(pct=True, method="first")
    out = out[out["daily_pct_rank"] >= 0.50].copy()
    out = out.sort_values(["signal_date", "base_score", "code"], ascending=[True, False, True])
    out = out.groupby("signal_date", group_keys=False).head(TOP_N).reset_index(drop=True)
    return out


def _portfolio_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "daily_ret", "equity"])
    x = trades.copy()
    x["signal_date"] = pd.to_datetime(x["signal_date"])
    daily = x.groupby("signal_date", as_index=False)["return_pct"].mean()
    daily = daily.sort_values("signal_date").reset_index(drop=True)
    daily["equity"] = INITIAL_CAPITAL * (1 + daily["return_pct"]).cumprod()
    daily["date"] = daily["signal_date"]
    daily["daily_ret"] = daily["return_pct"]
    return daily[["date", "daily_ret", "equity"]]


def summarize_trades(trades: pd.DataFrame, strategy: str) -> dict[str, Any]:
    if trades.empty:
        return {
            "strategy": strategy,
            "trade_count": 0,
            "avg_trade_return": 0.0,
            "success_rate": 0.0,
            "avg_holding_days": 0.0,
            "profit_factor": 0.0,
            "annual_return_signal_basket": 0.0,
            "max_drawdown_signal_basket": 0.0,
            "final_equity_signal_basket": INITIAL_CAPITAL,
        }
    equity = _portfolio_curve(trades)
    rets = trades["return_pct"].astype(float)
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if not losses.empty and abs(losses.sum()) > 1e-12 else float("inf")
    dd = equity["equity"] / equity["equity"].cummax() - 1.0
    years = max((equity["date"].iloc[-1] - equity["date"].iloc[0]).days / 365.25, 1 / 365.25)
    annual = float((equity["equity"].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) if len(equity) > 1 else float(rets.mean() * 252)
    return {
        "strategy": strategy,
        "trade_count": int(len(trades)),
        "avg_trade_return": float(rets.mean()),
        "success_rate": float((rets > 0).mean()),
        "avg_holding_days": float(trades["hold_days"].mean()),
        "profit_factor": profit_factor,
        "annual_return_signal_basket": annual,
        "max_drawdown_signal_basket": float(dd.min()) if not dd.empty else 0.0,
        "final_equity_signal_basket": float(equity["equity"].iloc[-1]),
    }


@dataclass
class SimResult:
    trade: dict[str, Any] | None
    skipped: bool = False
    skip_reason: str = ""


def classify_gap(signal_close: float, entry_open: float) -> str:
    gap = entry_open / signal_close - 1.0
    if gap > 1e-12:
        return "gap_up"
    if gap < -1e-12:
        return "gap_down"
    return "gap_flat"


def minute_trigger_trade(
    code: str,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    signal_idx: int,
    daily_df: pd.DataFrame,
    min1_df: pd.DataFrame | None,
    min5_df: pd.DataFrame | None,
    execution_mode: str,
) -> SimResult:
    entry_row = daily_df[daily_df["date"] == entry_date]
    signal_row = daily_df[daily_df["date"] == signal_date]
    if entry_row.empty or signal_row.empty:
        return SimResult(trade=None, skipped=True, skip_reason="missing_daily_row")
    entry_open = float(entry_row.iloc[0]["open"])
    signal_close = float(signal_row.iloc[0]["close"])
    if entry_open / signal_close - 1.0 > BUY_GAP_LIMIT:
        return SimResult(trade=None, skipped=True, skip_reason="gap_gt_4pct")

    stop_price = float(signal_row.iloc[0]["low"]) * 0.99
    tp_price = entry_open * (1.0 + TAKE_PROFIT)
    gap_group = classify_gap(signal_close, entry_open)

    minute_df = None
    minute_source = ""
    if min1_df is not None:
        minute_df = min1_df
        minute_source = "1min"
    elif min5_df is not None:
        minute_df = min5_df
        minute_source = "5min"
    else:
        return SimResult(trade=None, skipped=True, skip_reason="missing_minute_data")

    eligible_dates = daily_df[daily_df["date"] > entry_date]["date"].head(MAX_HOLD_DAYS).tolist()
    if not eligible_dates:
        return SimResult(trade=None, skipped=True, skip_reason="no_exit_window")

    trigger_date = None
    trigger_reason = None
    trigger_price = None
    next_exec_date = None

    for d in eligible_dates:
        day_min = minute_df[minute_df["date"] == d]
        if day_min.empty:
            day_daily = daily_df[daily_df["date"] == d]
            if day_daily.empty:
                continue
            day_high = float(day_daily.iloc[0]["high"])
            day_low = float(day_daily.iloc[0]["low"])
            if day_high >= tp_price:
                trigger_date = d
                trigger_reason = "tp"
                trigger_price = tp_price
                break
            if day_low <= stop_price:
                trigger_date = d
                trigger_reason = "sl"
                trigger_price = stop_price
                break
            continue
        for row in day_min.itertuples(index=False):
            day_high = float(row.high)
            day_low = float(row.low)
            if day_high >= tp_price:
                trigger_date = d
                trigger_reason = "tp"
                trigger_price = tp_price
                break
            if day_low <= stop_price:
                trigger_date = d
                trigger_reason = "sl"
                trigger_price = stop_price
                break
        if trigger_reason is not None:
            break

    if trigger_reason is None:
        exit_date = eligible_dates[-1]
        exit_row = daily_df[daily_df["date"] == exit_date]
        if exit_row.empty:
            return SimResult(trade=None, skipped=True, skip_reason="missing_forced_exit_day")
        exit_price = float(exit_row.iloc[0]["close"])
        exit_reason = "max_hold_close"
    else:
        if execution_mode == "intraday_trigger":
            exit_date = trigger_date
            exit_price = float(trigger_price)
            exit_reason = f"{trigger_reason}_intraday"
        else:
            later = daily_df[daily_df["date"] > trigger_date]["date"]
            if later.empty:
                exit_date = trigger_date
                exit_row = daily_df[daily_df["date"] == exit_date]
                exit_price = float(exit_row.iloc[0]["close"])
                exit_reason = f"{trigger_reason}_same_day_fallback_close"
            else:
                next_exec_date = pd.Timestamp(later.iloc[0])
                exit_date = next_exec_date
                exit_row = daily_df[daily_df["date"] == exit_date]
                if exit_row.empty:
                    return SimResult(trade=None, skipped=True, skip_reason="missing_next_exec_day")
                exit_price = float(exit_row.iloc[0]["open"])
                exit_reason = f"{trigger_reason}_next_open"

    hold_days = int((pd.Timestamp(exit_date) - entry_date).days)
    return_pct = exit_price / entry_open - 1.0
    return SimResult(
        trade={
            "code": code,
            "signal_idx": signal_idx,
            "signal_date": signal_date,
            "entry_date": entry_date,
            "exit_date": pd.Timestamp(exit_date),
            "entry_price": entry_open,
            "exit_price": exit_price,
            "return_pct": return_pct,
            "hold_days": hold_days,
            "exit_reason": exit_reason,
            "trigger_take_profit": tp_price,
            "trigger_stop_loss": stop_price,
            "gap_group": gap_group,
            "entry_gap_pct": entry_open / signal_close - 1.0,
            "minute_source": minute_source,
            "execution_mode": execution_mode,
        }
    )


def run_compare(file_limit: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(output_dir, "loading_candidates", file_limit=file_limit)

    candidates = build_formal_best_candidates(file_limit=file_limit)
    candidates.to_csv(output_dir / "selected_signals.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        empty = pd.DataFrame()
        for name in [
            "execution_compare_trades.csv",
            "execution_compare_skipped.csv",
            "execution_compare_summary.csv",
            "gap_group_summary.csv",
        ]:
            empty.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
        summary_json = {
            "candidate_count": 0,
            "trade_count": 0,
            "skipped_count": 0,
            "mode_best": None,
            "summary_rows": [],
            "gap_group_rows": [],
            "comparison": {},
            "assumptions": {
                "signal_pool": "brick.formal_best",
                "buy_gap_limit_pct": BUY_GAP_LIMIT,
                "take_profit_pct": TAKE_PROFIT,
                "stop_loss_rule": "signal_low_x_0.99",
                "max_hold_days": MAX_HOLD_DAYS,
                "buy_day_cannot_sell": True,
                "same_bar_priority": "take_profit_first",
                "minute_priority": "1min_then_5min_fallback",
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        update_progress(output_dir, "finished", output_dir=str(output_dir), candidate_count=0)
        return
    candidate_codes = set(candidates["code"].astype(str))
    daily_map = build_daily_map(DAILY_DIR, file_limit=file_limit, codes=candidate_codes)
    min1_map, min5_map = build_minute_maps(candidate_codes)
    update_progress(
        output_dir,
        "data_ready",
        candidate_count=len(candidates),
        daily_count=len(daily_map),
        min1_count=len(min1_map),
        min5_count=len(min5_map),
    )

    trade_rows = []
    skipped_rows = []
    for mode in ["intraday_trigger", "nextday_open"]:
        for row in candidates.itertuples(index=False):
            code = str(row.code)
            daily_df = daily_map.get(code)
            if daily_df is None:
                skipped_rows.append({"code": code, "mode": mode, "reason": "missing_daily_series"})
                continue
            result = minute_trigger_trade(
                code=code,
                signal_date=pd.Timestamp(row.signal_date),
                entry_date=pd.Timestamp(row.entry_date),
                signal_idx=int(row.signal_idx),
                daily_df=daily_df,
                min1_df=min1_map.get(code),
                min5_df=min5_map.get(code),
                execution_mode=mode,
            )
            if result.skipped or result.trade is None:
                skipped_rows.append(
                    {
                        "code": code,
                        "signal_date": row.signal_date,
                        "entry_date": row.entry_date,
                        "mode": mode,
                        "reason": result.skip_reason,
                    }
                )
                continue
            trade_rows.append(result.trade)

    trades = pd.DataFrame(trade_rows).sort_values(["execution_mode", "signal_date", "code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows).sort_values(["mode", "signal_date", "code"], na_position="last").reset_index(drop=True) if skipped_rows else pd.DataFrame()
    trades.to_csv(output_dir / "execution_compare_trades.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(output_dir / "execution_compare_skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(output_dir, "trades_ready", trade_count=len(trades), skipped_count=len(skipped))

    summary_rows = []
    gap_rows = []
    for mode, mode_df in trades.groupby("execution_mode"):
        summary = summarize_trades(mode_df, mode)
        summary["mode"] = mode
        summary["buy_gap_limit"] = BUY_GAP_LIMIT
        summary["take_profit_pct"] = TAKE_PROFIT
        summary["stop_loss_rule"] = "signal_low_x_0.99"
        summary["max_hold_days"] = MAX_HOLD_DAYS
        summary_rows.append(summary)
        for gap_group, gap_df in mode_df.groupby("gap_group"):
            row = summarize_trades(gap_df, f"{mode}|{gap_group}")
            row["mode"] = mode
            row["gap_group"] = gap_group
            row["mean_entry_gap_pct"] = float(gap_df["entry_gap_pct"].mean())
            gap_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("final_equity_signal_basket", ascending=False)
    gap_df = pd.DataFrame(gap_rows).sort_values(["mode", "gap_group"])
    summary_df.to_csv(output_dir / "execution_compare_summary.csv", index=False, encoding="utf-8-sig")
    gap_df.to_csv(output_dir / "gap_group_summary.csv", index=False, encoding="utf-8-sig")

    comparison = {}
    if not summary_df.empty and {"intraday_trigger", "nextday_open"} <= set(summary_df["mode"]):
        intraday = summary_df[summary_df["mode"] == "intraday_trigger"].iloc[0].to_dict()
        nextday = summary_df[summary_df["mode"] == "nextday_open"].iloc[0].to_dict()
        comparison = {
            "nextday_minus_intraday_total_return": float(nextday["final_equity_signal_basket"] / INITIAL_CAPITAL - intraday["final_equity_signal_basket"] / INITIAL_CAPITAL),
            "nextday_minus_intraday_avg_trade_return": float(nextday["avg_trade_return"] - intraday["avg_trade_return"]),
            "nextday_minus_intraday_success_rate": float(nextday["success_rate"] - intraday["success_rate"]),
            "nextday_minus_intraday_max_drawdown": float(nextday["max_drawdown_signal_basket"] - intraday["max_drawdown_signal_basket"]),
        }

    summary_json = {
        "candidate_count": int(len(candidates)),
        "trade_count": int(len(trades)),
        "skipped_count": int(len(skipped)),
        "mode_best": summary_df.iloc[0]["mode"] if not summary_df.empty else None,
        "summary_rows": summary_rows,
        "gap_group_rows": gap_rows,
        "comparison": comparison,
        "assumptions": {
            "signal_pool": "brick.formal_best",
            "buy_gap_limit_pct": BUY_GAP_LIMIT,
            "take_profit_pct": TAKE_PROFIT,
            "stop_loss_rule": "signal_low_x_0.99",
            "max_hold_days": MAX_HOLD_DAYS,
            "buy_day_cannot_sell": True,
            "same_bar_priority": "take_profit_first",
            "minute_priority": "1min_then_5min_fallback",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(output_dir, "finished", output_dir=str(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK 分钟线盘中/次日执行对比实验")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--file-limit", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_minute_execution_compare_v1_{args.mode}_{timestamp}"
    file_limit = int(args.file_limit)
    if args.mode == "full":
        file_limit = 0
    run_compare(file_limit=file_limit, output_dir=output_dir)


if __name__ == "__main__":
    main()
