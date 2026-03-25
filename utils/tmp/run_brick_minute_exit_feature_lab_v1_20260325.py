from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
BASE_SCRIPT = ROOT / "utils" / "tmp" / "run_brick_intraday_minute_compare_v1_20260325.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_HOLD_DAYS = 3
BUY_GAP_LIMIT = 0.04
STOP_MULTIPLIERS = [0.985, 0.99, 0.995]


@dataclass(frozen=True)
class ExitMethod:
    name: str
    bar_source: str
    family: str
    params: dict[str, Any]


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "brick_minute_exit_feature_lab_base")


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {
        "stage": stage,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_gap(signal_close: float, entry_open: float) -> str:
    return base.classify_gap(signal_close, entry_open)


def calc_kdj(df: pd.DataFrame, n: int = 9, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    out = df.copy()
    low_min = out["low"].rolling(n, min_periods=1).min()
    high_max = out["high"].rolling(n, min_periods=1).max()
    denom = (high_max - low_min).replace(0.0, np.nan)
    rsv = ((out["close"] - low_min) / denom * 100.0).fillna(50.0)
    out["K"] = rsv.ewm(alpha=1.0 / k_period, adjust=False).mean()
    out["D"] = out["K"].ewm(alpha=1.0 / d_period, adjust=False).mean()
    out["J"] = 3.0 * out["K"] - 2.0 * out["D"]
    return out


def calc_ema(df: pd.DataFrame, fast: int = 12, slow: int = 36) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=slow, adjust=False).mean()
    return out


def calc_vwap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["turnover"] = out["close"] * out["volume"]
    out["cum_turnover"] = out.groupby("date")["turnover"].cumsum()
    out["cum_volume"] = out.groupby("date")["volume"].cumsum().replace(0.0, np.nan)
    out["vwap"] = (out["cum_turnover"] / out["cum_volume"]).fillna(out["close"])
    return out


def build_methods(smoke: bool) -> list[ExitMethod]:
    methods: list[ExitMethod] = [
        ExitMethod("baseline_nextday_open", "1min", "baseline", {"mode": "nextday_open"}),
        ExitMethod("baseline_intraday_trigger", "1min", "baseline", {"mode": "intraday_trigger"}),
    ]

    j_thresholds = [90] if smoke else [85, 90, 95]
    min_profit_levels = [0.02] if smoke else [0.01, 0.02, 0.03]
    for th in j_thresholds:
        for mp in min_profit_levels:
            methods.append(ExitMethod(f"m1_kdj_j{th}_p{mp:.2f}", "1min", "kdj_reversal", {"j_threshold": th, "min_profit": mp}))
            methods.append(ExitMethod(f"m5_kdj_j{th}_p{mp:.2f}", "5min", "kdj_reversal", {"j_threshold": th, "min_profit": mp}))

    ema_min_profit_levels = [0.02] if smoke else [0.01, 0.02, 0.03]
    for mp in ema_min_profit_levels:
        methods.append(ExitMethod(f"m5_ema_cross_p{mp:.2f}", "5min", "ema_cross", {"min_profit": mp}))

    down_ns = [3] if smoke else [3, 5]
    down_profit_levels = [0.02] if smoke else [0.01, 0.02, 0.03]
    loss_floors = [-0.01] if smoke else [-0.005, -0.01]
    for n in down_ns:
        for mp in down_profit_levels:
            for lf in loss_floors:
                methods.append(
                    ExitMethod(
                        f"m1_down{n}_p{mp:.2f}_l{abs(lf):.3f}",
                        "1min",
                        "consecutive_down",
                        {"down_n": n, "min_profit": mp, "loss_floor": lf},
                    )
                )

    activate_levels = [0.03] if smoke else [0.02, 0.03, 0.04]
    retrace_levels = [0.012] if smoke else [0.008, 0.012, 0.016]
    for act in activate_levels:
        for retr in retrace_levels:
            methods.append(
                ExitMethod(
                    f"m1_trailing_a{act:.2f}_r{retr:.3f}",
                    "1min",
                    "trailing_stop",
                    {"activate_profit": act, "retrace": retr},
                )
            )

    vwap_profit_levels = [0.02] if smoke else [0.01, 0.02, 0.03]
    for mp in vwap_profit_levels:
        methods.append(ExitMethod(f"m5_vwap_break_p{mp:.2f}", "5min", "vwap_break", {"min_profit": mp}))
    return methods


def get_minute_frame(code: str, bar_source: str, min1_map: dict[str, pd.DataFrame], min5_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame | None, str]:
    if bar_source == "1min":
        if code in min1_map:
            return min1_map[code], "1min"
        if code in min5_map:
            return min5_map[code], "5min_fallback"
        return None, ""
    if code in min5_map:
        return min5_map[code], "5min"
    if code in min1_map:
        return min1_map[code], "1min_fallback"
    return None, ""


def build_eligible_dates(daily_df: pd.DataFrame, entry_date: pd.Timestamp) -> list[pd.Timestamp]:
    return daily_df[daily_df["date"] > entry_date]["date"].head(MAX_HOLD_DAYS).tolist()


def trigger_baseline(
    method: ExitMethod,
    code: str,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    signal_idx: int,
    daily_df: pd.DataFrame,
    min1_map: dict[str, pd.DataFrame],
    min5_map: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    res = base.minute_trigger_trade(
        code=code,
        signal_date=signal_date,
        entry_date=entry_date,
        signal_idx=signal_idx,
        daily_df=daily_df,
        min1_df=min1_map.get(code),
        min5_df=min5_map.get(code),
        execution_mode=method.params["mode"],
    )
    if res.skipped:
        return None, {"skip_reason": res.skip_reason}
    return res.trade, None


def build_bar_features(raw_minute: pd.DataFrame, family: str) -> pd.DataFrame:
    df = raw_minute.copy()
    if family == "kdj_reversal":
        return calc_kdj(df)
    if family == "ema_cross":
        return calc_ema(df)
    if family == "vwap_break":
        df = calc_ema(df)
        return calc_vwap(df)
    if family in {"consecutive_down", "trailing_stop"}:
        return df
    return df


def simulate_feature_method(
    method: ExitMethod,
    code: str,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    signal_idx: int,
    daily_df: pd.DataFrame,
    min1_map: dict[str, pd.DataFrame],
    min5_map: dict[str, pd.DataFrame],
    sl_mult: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    entry_row = daily_df[daily_df["date"] == entry_date]
    signal_row = daily_df[daily_df["date"] == signal_date]
    if entry_row.empty or signal_row.empty:
        return None, {"skip_reason": "missing_daily_row"}
    entry_open = float(entry_row.iloc[0]["open"])
    signal_close = float(signal_row.iloc[0]["close"])
    if not (np.isfinite(entry_open) and entry_open > 0 and np.isfinite(signal_close) and signal_close > 0):
        return None, {"skip_reason": "invalid_entry_or_signal_close"}
    entry_gap_pct = entry_open / signal_close - 1.0
    if entry_gap_pct > BUY_GAP_LIMIT:
        return None, {"skip_reason": "gap_gt_4pct"}

    raw_minute, minute_source = get_minute_frame(code, method.bar_source, min1_map, min5_map)
    if raw_minute is None or raw_minute.empty:
        return None, {"skip_reason": "missing_minute_data"}

    eligible_dates = build_eligible_dates(daily_df, entry_date)
    if not eligible_dates:
        return None, {"skip_reason": "no_exit_window"}

    minute_df = raw_minute[raw_minute["date"].isin(eligible_dates)].copy()
    if minute_df.empty:
        return None, {"skip_reason": "empty_eligible_minute"}
    minute_df = build_bar_features(minute_df, method.family)

    stop_price = float(signal_row.iloc[0]["low"]) * sl_mult
    gap_group = classify_gap(signal_close, entry_open)
    peak_high = entry_open
    tp_triggered = False
    exit_reason = None
    exit_date = None
    exit_price = None

    closes = minute_df["close"].astype(float).to_numpy()
    highs = minute_df["high"].astype(float).to_numpy()
    lows = minute_df["low"].astype(float).to_numpy()

    for i, row in enumerate(minute_df.itertuples(index=False)):
        curr_close = float(row.close)
        curr_high = float(row.high)
        curr_low = float(row.low)
        peak_high = max(peak_high, curr_high)
        current_ret = curr_close / entry_open - 1.0

        if np.isfinite(curr_low) and curr_low <= stop_price:
            exit_reason = "hard_stop_intraday"
            exit_date = pd.Timestamp(row.date)
            exit_price = stop_price
            break

        if method.family == "kdj_reversal":
            if i == 0:
                continue
            prev_j = float(minute_df.iloc[i - 1]["J"])
            curr_j = float(minute_df.iloc[i]["J"])
            if max(prev_j, curr_j) >= float(method.params["j_threshold"]) and curr_j < prev_j and current_ret >= float(method.params["min_profit"]):
                exit_reason = "kdj_reversal_close"
                exit_date = pd.Timestamp(row.date)
                exit_price = curr_close
                break

        elif method.family == "ema_cross":
            if i == 0:
                continue
            prev_fast = float(minute_df.iloc[i - 1]["ema_fast"])
            prev_slow = float(minute_df.iloc[i - 1]["ema_slow"])
            curr_fast = float(minute_df.iloc[i]["ema_fast"])
            curr_slow = float(minute_df.iloc[i]["ema_slow"])
            if prev_fast >= prev_slow and curr_fast < curr_slow and current_ret >= float(method.params["min_profit"]):
                exit_reason = "ema_cross_close"
                exit_date = pd.Timestamp(row.date)
                exit_price = curr_close
                break

        elif method.family == "consecutive_down":
            n = int(method.params["down_n"])
            if i + 1 < n:
                continue
            window = closes[i - n + 1 : i + 1]
            down_streak = bool(np.all(np.diff(window) < 0))
            if not down_streak:
                continue
            if current_ret >= float(method.params["min_profit"]):
                exit_reason = f"down{n}_take_profit_close"
                exit_date = pd.Timestamp(row.date)
                exit_price = curr_close
                break
            if current_ret <= float(method.params["loss_floor"]):
                exit_reason = f"down{n}_loss_cut_close"
                exit_date = pd.Timestamp(row.date)
                exit_price = curr_close
                break

        elif method.family == "trailing_stop":
            peak_ret = peak_high / entry_open - 1.0
            retrace = 1.0 - curr_close / peak_high if peak_high > 0 else 0.0
            if peak_ret >= float(method.params["activate_profit"]) and retrace >= float(method.params["retrace"]):
                exit_reason = "trailing_take_profit_close"
                exit_date = pd.Timestamp(row.date)
                exit_price = curr_close
                break

        elif method.family == "vwap_break":
            if i == 0:
                continue
            curr_vwap = float(minute_df.iloc[i]["vwap"])
            curr_fast = float(minute_df.iloc[i]["ema_fast"])
            if current_ret >= float(method.params["min_profit"]) and curr_close < curr_vwap and curr_close < curr_fast:
                exit_reason = "vwap_break_close"
                exit_date = pd.Timestamp(row.date)
                exit_price = curr_close
                break

    if exit_reason is None:
        last_date = eligible_dates[-1]
        exit_row = daily_df[daily_df["date"] == last_date]
        if exit_row.empty:
            return None, {"skip_reason": "missing_forced_exit_day"}
        exit_date = pd.Timestamp(last_date)
        exit_price = float(exit_row.iloc[0]["close"])
        exit_reason = "time_exit_close"

    hold_days = int((pd.Timestamp(exit_date) - entry_date).days)
    ret = float(exit_price / entry_open - 1.0)
    return {
        "strategy": method.name,
        "family": method.family,
        "code": code,
        "signal_idx": signal_idx,
        "signal_date": signal_date,
        "entry_date": entry_date,
        "exit_date": pd.Timestamp(exit_date),
        "entry_price": entry_open,
        "exit_price": exit_price,
        "return_pct": ret,
        "hold_days": hold_days,
        "exit_reason": exit_reason,
        "entry_gap_pct": entry_gap_pct,
        "gap_group": gap_group,
        "stop_loss_multiplier": sl_mult,
        "minute_source": minute_source,
        "bar_source": method.bar_source,
    }, None


def run_methods(
    candidates: pd.DataFrame,
    daily_map: dict[str, pd.DataFrame],
    min1_map: dict[str, pd.DataFrame],
    min5_map: dict[str, pd.DataFrame],
    methods: list[ExitMethod],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        code = str(row.code)
        daily_df = daily_map.get(code)
        if daily_df is None or daily_df.empty:
            for method in methods:
                skipped.append({"strategy": method.name, "code": code, "signal_date": row.signal_date, "skip_reason": "missing_daily_df"})
            continue
        for method in methods:
            if method.family == "baseline":
                trade, skip = trigger_baseline(method, code, pd.Timestamp(row.signal_date), pd.Timestamp(row.entry_date), int(row.signal_idx), daily_df, min1_map, min5_map)
                if trade is not None:
                    trades.append(trade)
                else:
                    skipped.append({"strategy": method.name, "code": code, "signal_date": row.signal_date, **(skip or {})})
                continue
            for sl_mult in STOP_MULTIPLIERS:
                trade, skip = simulate_feature_method(
                    method,
                    code,
                    pd.Timestamp(row.signal_date),
                    pd.Timestamp(row.entry_date),
                    int(row.signal_idx),
                    daily_df,
                    min1_map,
                    min5_map,
                    sl_mult,
                )
                strategy_name = f"{method.name}|sl{sl_mult:.3f}"
                if trade is not None:
                    trade["strategy"] = strategy_name
                    trades.append(trade)
                else:
                    skipped.append({"strategy": strategy_name, "code": code, "signal_date": row.signal_date, **(skip or {})})
    return pd.DataFrame(trades), pd.DataFrame(skipped)


def summarize_top(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    gap_rows = []
    for strategy, g in trades.groupby("strategy"):
        base_summary = base.summarize_trades(g, strategy)
        family = g["family"].iloc[0] if "family" in g.columns else ""
        stop_mult = g["stop_loss_multiplier"].iloc[0] if "stop_loss_multiplier" in g.columns else np.nan
        bar_source = g["bar_source"].iloc[0] if "bar_source" in g.columns else ""
        minute_source = g["minute_source"].mode().iloc[0] if "minute_source" in g.columns and not g["minute_source"].mode().empty else ""
        base_summary.update(
            {
                "family": family,
                "stop_loss_multiplier": stop_mult,
                "bar_source": bar_source,
                "minute_source": minute_source,
                "buy_gap_limit": BUY_GAP_LIMIT,
                "max_hold_days": MAX_HOLD_DAYS,
            }
        )
        rows.append(base_summary)
        for gap_group, gg in g.groupby("gap_group"):
            row = base.summarize_trades(gg, f"{strategy}|{gap_group}")
            row["family"] = family
            row["gap_group"] = gap_group
            row["mean_entry_gap_pct"] = float(gg["entry_gap_pct"].mean())
            gap_rows.append(row)
    summary_df = pd.DataFrame(rows).sort_values(
        ["final_equity_signal_basket", "success_rate", "avg_trade_return"],
        ascending=[False, False, False],
    )
    gap_df = pd.DataFrame(gap_rows).sort_values(["strategy", "gap_group"]) if gap_rows else pd.DataFrame()
    return summary_df, gap_df


def build_summary_json(candidates: pd.DataFrame, trades: pd.DataFrame, skipped: pd.DataFrame, summary_df: pd.DataFrame, gap_df: pd.DataFrame) -> dict[str, Any]:
    best_row = summary_df.iloc[0].to_dict() if not summary_df.empty else None
    family_best = []
    if not summary_df.empty:
        for family, g in summary_df.groupby("family"):
            family_best.append(g.iloc[0].to_dict())
    return {
        "candidate_count": int(len(candidates)),
        "trade_count": int(len(trades)),
        "skipped_count": int(len(skipped)),
        "best_strategy": best_row,
        "family_best": family_best,
        "top10": summary_df.head(10).to_dict(orient="records") if not summary_df.empty else [],
        "assumptions": {
            "signal_pool": "brick.formal_best",
            "buy_gap_limit_pct": BUY_GAP_LIMIT,
            "buy_day_cannot_sell": True,
            "same_bar_priority_baseline": "take_profit_first",
            "feature_methods_same_bar_priority": "hard_stop_before_close_feature_exit",
            "max_hold_days": MAX_HOLD_DAYS,
            "minute_priority": "method_bar_source_then_fallback",
            "stop_loss_multipliers": STOP_MULTIPLIERS,
        },
    }


def run(file_limit: int, output_dir: Path, smoke: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(output_dir, "loading_candidates", file_limit=file_limit, smoke=smoke)
    candidates = base.build_formal_best_candidates(file_limit=file_limit)
    candidates.to_csv(output_dir / "selected_signals.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        empty = pd.DataFrame()
        for name in ["feature_trades.csv", "feature_skipped.csv", "feature_summary.csv", "gap_group_summary.csv"]:
            empty.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
        (output_dir / "summary.json").write_text(json.dumps({"candidate_count": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
        update_progress(output_dir, "finished", candidate_count=0)
        return

    candidate_codes = set(candidates["code"].astype(str))
    daily_map = base.build_daily_map(base.DAILY_DIR, file_limit=file_limit, codes=candidate_codes)
    min1_map, min5_map = base.build_minute_maps(candidate_codes)
    methods = build_methods(smoke)
    method_df = pd.DataFrame([{"strategy": m.name, "family": m.family, "bar_source": m.bar_source, **m.params} for m in methods])
    method_df.to_csv(output_dir / "method_inventory.csv", index=False, encoding="utf-8-sig")
    update_progress(
        output_dir,
        "data_ready",
        candidate_count=len(candidates),
        method_count=len(methods),
        daily_count=len(daily_map),
        min1_count=len(min1_map),
        min5_count=len(min5_map),
    )

    trades, skipped = run_methods(candidates, daily_map, min1_map, min5_map, methods)
    trades.to_csv(output_dir / "feature_trades.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(output_dir / "feature_skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(output_dir, "trades_ready", trade_count=len(trades), skipped_count=len(skipped))

    summary_df, gap_df = summarize_top(trades)
    summary_df.to_csv(output_dir / "feature_summary.csv", index=False, encoding="utf-8-sig")
    gap_df.to_csv(output_dir / "gap_group_summary.csv", index=False, encoding="utf-8-sig")
    best_only = summary_df.groupby("family", as_index=False).head(1) if not summary_df.empty else pd.DataFrame()
    best_only.to_csv(output_dir / "family_best_summary.csv", index=False, encoding="utf-8-sig")

    summary_json = build_summary_json(candidates, trades, skipped, summary_df, gap_df)
    (output_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(output_dir, "finished", output_dir=str(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRICK 分钟线退出特征搜索实验")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--file-limit", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_minute_exit_feature_lab_v1_{args.mode}_{ts}"
    file_limit = int(args.file_limit)
    if args.mode == "full":
        file_limit = 0
    run(file_limit=file_limit, output_dir=output_dir, smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
