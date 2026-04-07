from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "20260402"
STRUCTURE_DETAIL = ROOT / "results" / "b3_second_red_brick_analysis_v1_20260407_140920" / "b3_signal_detail.csv"
VOLUME_DETAIL = ROOT / "results" / "b3_volume_definition_compare_v2_20260407_143715" / "b3_volume_definition_signal_detail.csv"
OUTPUT_ROOT = ROOT / "results"
MAX_WORKERS = max(1, min((mp.cpu_count() or 4), 10))
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
LOT_SIZE = 100
MAX_HOLD_DAYS = 30


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


technical = load_module(ROOT / "utils" / "technical_indicators.py", "b3_fixed_tp_boundary_technical")


@dataclass(frozen=True)
class ExitProfile:
    family: str
    group_name: str
    tp_value: float

    @property
    def profile_name(self) -> str:
        return f"{self.family}__{self.group_name}__fixed_tp_{self.tp_value * 100:.2f}%"


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_div(a: float, b: float, default: float = float("nan")) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return default
    return float(a / b)


def tp_grid(low: float, high: float, step: float) -> list[float]:
    values = []
    x = low
    while x <= high + 1e-12:
        values.append(round(x, 6))
        x += step
    return values


def build_source_signals() -> pd.DataFrame:
    struct = pd.read_csv(STRUCTURE_DETAIL, parse_dates=["signal_date"]).copy()
    struct["family"] = "structure"
    struct_rows: list[pd.DataFrame] = []
    for col in ["second_red_after_green_ge2", "second_red_after_green_ge3", "second_red_after_green_ge4"]:
        base = struct.copy()
        prefix = col.replace("second_red_after_green_", "")
        base["group_name"] = np.where(base[col].fillna(False), prefix + "_true", prefix + "_false")
        struct_rows.append(base[["code", "signal_date", "family", "group_name", "ret1", "amplitude"]])
    struct_df = pd.concat(struct_rows, ignore_index=True)

    vol = pd.read_csv(VOLUME_DETAIL, parse_dates=["signal_date"]).copy()
    vol["family"] = "volume"
    vol["group_name"] = vol["volume_group"].astype(str)
    vol_df = vol[["code", "signal_date", "family", "group_name", "ret1", "amplitude"]].copy()

    signals = pd.concat([struct_df, vol_df], ignore_index=True)
    signals["code"] = signals["code"].astype(str)
    signals["sort_ratio"] = signals.apply(
        lambda r: safe_div(float(r["amplitude"]), float(r["ret1"]), default=float("inf")) if float(r["ret1"]) > 0 else float("inf"),
        axis=1,
    )
    signals = signals.sort_values(["signal_date", "family", "group_name", "sort_ratio", "code"]).reset_index(drop=True)
    return signals


def build_profiles(signals: pd.DataFrame, tp_values: list[float]) -> list[ExitProfile]:
    groups = signals[["family", "group_name"]].drop_duplicates().sort_values(["family", "group_name"])
    profiles: list[ExitProfile] = []
    for _, row in groups.iterrows():
        for tp in tp_values:
            profiles.append(ExitProfile(family=str(row["family"]), group_name=str(row["group_name"]), tp_value=float(tp)))
    return profiles


def load_one_daily(path: Path) -> pd.DataFrame:
    df = technical._load_price_data(str(path))
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.rename(
        columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    )[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def _load_one_price_pair(code: str) -> tuple[str, pd.DataFrame]:
    return code, load_one_daily(DATA_DIR / f"{code}.txt")


def load_price_map(codes: list[str]) -> dict[str, pd.DataFrame]:
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=MAX_WORKERS) as pool:
        pairs = pool.map(_load_one_price_pair, codes, chunksize=1)
    return {code: df for code, df in pairs if df is not None and not df.empty}


def build_unique_signals(signals: pd.DataFrame, mode: str) -> pd.DataFrame:
    base = signals[["code", "signal_date", "ret1", "amplitude", "sort_ratio"]].drop_duplicates().copy()
    base = base.sort_values(["signal_date", "code"]).reset_index(drop=True)
    if mode == "smoke":
        keep_dates = sorted(base["signal_date"].drop_duplicates())[-20:]
        base = base[base["signal_date"].isin(keep_dates)].reset_index(drop=True)
    return base


def eval_signal_all_tps(signal_row: dict[str, Any], price_rows: list[dict[str, Any]], tp_values: list[float]) -> list[dict[str, Any]]:
    price_df = pd.DataFrame(price_rows)
    if price_df.empty:
        return []
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values("date").reset_index(drop=True)
    signal_date = pd.Timestamp(signal_row["signal_date"])
    match = np.flatnonzero(price_df["date"].to_numpy(dtype="datetime64[ns]") == np.datetime64(signal_date))
    if len(match) == 0:
        return []
    signal_idx = int(match[0])
    entry_idx = signal_idx + 1
    if entry_idx >= len(price_df):
        return []

    entry_row = price_df.iloc[entry_idx]
    entry_open = float(entry_row["open"])
    stop_price = float(entry_row["low"])
    if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(stop_price) or stop_price <= 0:
        return []

    last_holding_idx = min(entry_idx + MAX_HOLD_DAYS - 1, len(price_df) - 1)
    rolling = []
    running_high = float(entry_row["high"])
    for idx in range(entry_idx + 1, last_holding_idx + 1):
        row = price_df.iloc[idx]
        running_high = max(running_high, float(row["high"]))
        rolling.append(
            {
                "idx": idx,
                "date": pd.Timestamp(row["date"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "rolling_high": running_high,
            }
        )

    results: list[dict[str, Any]] = []
    for tp_value in tp_values:
        exit_date = None
        exit_price = None
        exit_reason = None
        holding_days = None
        tp_price = entry_open * (1.0 + float(tp_value))

        for item in rolling:
            idx = item["idx"]
            stop_hit = np.isfinite(item["low"]) and item["low"] <= stop_price
            tp_hit = np.isfinite(item["high"]) and item["high"] >= tp_price
            if tp_hit:
                next_idx = idx + 1
                if next_idx < len(price_df):
                    next_row = price_df.iloc[next_idx]
                    exit_date = pd.Timestamp(next_row["date"])
                    exit_price = float(next_row["open"])
                    exit_reason = "fixed_tp_next_open"
                    holding_days = idx - entry_idx + 1
                    break
                exit_date = item["date"]
                exit_price = item["close"] if np.isfinite(item["close"]) and item["close"] > 0 else entry_open
                exit_reason = "fixed_tp_fallback_close"
                holding_days = idx - entry_idx + 1
                break
            if stop_hit:
                exit_date = item["date"]
                exit_price = stop_price
                exit_reason = "stop_same_day"
                holding_days = idx - entry_idx + 1
                break

        if exit_date is None:
            forced_idx = last_holding_idx + 1
            if forced_idx < len(price_df):
                forced_row = price_df.iloc[forced_idx]
                exit_date = pd.Timestamp(forced_row["date"])
                exit_price = float(forced_row["open"])
                exit_reason = "hold_30_next_open"
                holding_days = MAX_HOLD_DAYS
            else:
                forced_row = price_df.iloc[last_holding_idx]
                exit_date = pd.Timestamp(forced_row["date"])
                close_price = float(forced_row["close"])
                exit_price = close_price if np.isfinite(close_price) and close_price > 0 else entry_open
                exit_reason = "hold_30_fallback_close"
                holding_days = MAX_HOLD_DAYS

        if not np.isfinite(exit_price) or exit_price <= 0:
            continue
        results.append(
            {
                "code": str(signal_row["code"]),
                "signal_date": signal_date,
                "entry_date": pd.Timestamp(entry_row["date"]),
                "entry_open": entry_open,
                "stop_price": stop_price,
                "sort_ratio": float(signal_row["sort_ratio"]),
                "tp_value": float(tp_value),
                "exit_date": exit_date,
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "holding_days": int(holding_days),
                "trade_return": float(exit_price / entry_open - 1.0),
            }
        )
    return results


def _precompute_one(args: tuple[dict[str, Any], list[dict[str, Any]], list[float]]) -> list[dict[str, Any]]:
    signal_row, price_rows, tp_values = args
    return eval_signal_all_tps(signal_row, price_rows, tp_values)


def precompute_signal_tp_results(unique_signals: pd.DataFrame, price_map: dict[str, pd.DataFrame], tp_values: list[float]) -> pd.DataFrame:
    tasks = []
    for row in unique_signals.to_dict("records"):
        price_df = price_map.get(str(row["code"]))
        if price_df is None or price_df.empty:
            continue
        tasks.append((row, price_df.to_dict("records"), tp_values))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=MAX_WORKERS) as pool:
        chunks = pool.imap_unordered(_precompute_one, tasks, chunksize=1)
        rows: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            rows.extend(chunk)
            if idx % 200 == 0:
                print({"precompute_progress": idx, "total_tasks": len(tasks), "row_count": len(rows)}, flush=True)
    return pd.DataFrame(rows)


def build_trade_candidates(signals: pd.DataFrame, profiles: list[ExitProfile], precomputed: pd.DataFrame) -> pd.DataFrame:
    if precomputed.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for profile in profiles:
        group_signals = signals[(signals["family"] == profile.family) & (signals["group_name"] == profile.group_name)][
            ["code", "signal_date", "sort_ratio"]
        ].drop_duplicates()
        merged = group_signals.merge(
            precomputed[precomputed["tp_value"] == float(profile.tp_value)],
            on=["code", "signal_date", "sort_ratio"],
            how="inner",
        )
        if merged.empty:
            continue
        merged["profile_name"] = profile.profile_name
        merged["family"] = profile.family
        merged["group_name"] = profile.group_name
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["profile_name", "entry_date", "sort_ratio", "code"]).reset_index(drop=True)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def annual_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    years = len(equity) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def sharpe_ratio(equity: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    rets = equity.pct_change().dropna()
    if rets.empty:
        return float("nan")
    std = float(rets.std())
    if std <= 1e-12:
        return float("nan")
    return float(rets.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def simulate_account(trades_df: pd.DataFrame, price_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_dates = sorted({d for df in price_map.values() for d in pd.to_datetime(df["date"]).tolist()})
    close_maps = {code: df.set_index("date")["close"].astype(float).to_dict() for code, df in price_map.items()}

    account_rows: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []

    for profile_name, g in trades_df.groupby("profile_name"):
        g = g.sort_values(["entry_date", "sort_ratio", "code"], ascending=[True, True, True]).reset_index(drop=True)
        by_entry: dict[pd.Timestamp, list[dict[str, Any]]] = {}
        for row in g.to_dict("records"):
            by_entry.setdefault(pd.Timestamp(row["entry_date"]), []).append(row)

        cash = float(INITIAL_CAPITAL)
        positions: dict[str, dict[str, Any]] = {}
        completed: list[dict[str, Any]] = []
        curve: list[dict[str, Any]] = []

        for current_date in all_dates:
            todays = by_entry.get(pd.Timestamp(current_date), [])
            if todays:
                available_slots = max(MAX_POSITIONS - len(positions), 0)
                if available_slots > 0 and cash > 0:
                    selected: list[dict[str, Any]] = []
                    seen_codes = set()
                    for row in todays:
                        code = str(row["code"])
                        if code in positions or code in seen_codes:
                            continue
                        seen_codes.add(code)
                        selected.append(row)
                        if len(selected) >= available_slots:
                            break
                    if selected:
                        budget = cash / len(selected)
                        for row in selected:
                            entry_open = float(row["entry_open"])
                            shares = int(budget // (entry_open * LOT_SIZE)) * LOT_SIZE
                            if shares <= 0:
                                continue
                            cost = shares * entry_open
                            if cost > cash + 1e-9:
                                continue
                            cash -= cost
                            positions[str(row["code"])] = {**row, "shares": shares, "cost": cost}

            equity = cash
            for code, pos in positions.items():
                close_price = close_maps.get(code, {}).get(pd.Timestamp(current_date))
                if close_price is None or not np.isfinite(close_price):
                    close_price = float(pos["entry_open"])
                equity += float(pos["shares"]) * float(close_price)
            curve.append({"profile_name": profile_name, "date": pd.Timestamp(current_date), "equity": float(equity)})

            to_close = [code for code, pos in positions.items() if pd.Timestamp(pos["exit_date"]) == pd.Timestamp(current_date)]
            for code in to_close:
                pos = positions.pop(code)
                proceeds = float(pos["shares"]) * float(pos["exit_price"])
                cash += proceeds
                completed.append(
                    {
                        "profile_name": profile_name,
                        "family": pos["family"],
                        "group_name": pos["group_name"],
                        "code": code,
                        "signal_date": pos["signal_date"],
                        "entry_date": pos["entry_date"],
                        "exit_date": pos["exit_date"],
                        "entry_open": pos["entry_open"],
                        "exit_price": pos["exit_price"],
                        "shares": int(pos["shares"]),
                        "exit_reason": pos["exit_reason"],
                        "trade_return": pos["trade_return"],
                        "holding_days": int(pos["holding_days"]),
                        "pnl": proceeds - float(pos["cost"]),
                    }
                )

        equity_df = pd.DataFrame(curve)
        completed_df = pd.DataFrame(completed)
        if completed_df.empty or equity_df.empty:
            continue
        final_equity = float(equity_df.iloc[-1]["equity"])
        account_rows.append(
            {
                "profile_name": profile_name,
                "family": str(g.iloc[0]["family"]),
                "group_name": str(g.iloc[0]["group_name"]),
                "tp_value": float(g.iloc[0]["tp_value"]),
                "signal_count": int(len(g)),
                "trade_count": int(len(completed_df)),
                "annual_return": annual_return(equity_df["equity"]),
                "holding_return": float(final_equity / INITIAL_CAPITAL - 1.0),
                "max_drawdown": max_drawdown(equity_df["equity"]),
                "sharpe": sharpe_ratio(equity_df["equity"]),
                "success_rate": float((completed_df["trade_return"] > 0).mean()),
                "avg_trade_return": float(completed_df["trade_return"].mean()),
                "avg_holding_days": float(completed_df["holding_days"].mean()),
                "final_equity": final_equity,
            }
        )
        equity_frames.append(equity_df)

    account_df = pd.DataFrame(account_rows)
    if not account_df.empty:
        account_df = account_df.sort_values(
            ["annual_return", "holding_return", "success_rate", "max_drawdown"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    equity_all = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    return account_df, equity_all


def run_one_round(result_dir: Path, round_idx: int, signals: pd.DataFrame, unique_signals: pd.DataFrame, price_map: dict[str, pd.DataFrame], low: float, high: float, step: float) -> dict[str, Any]:
    round_dir = result_dir / f"round_{round_idx:02d}_{low:.4f}_{high:.4f}_{step:.4f}"
    round_dir.mkdir(parents=True, exist_ok=True)
    tp_values = tp_grid(low, high, step)
    update_progress(result_dir, "precomputing_round", round=round_idx, tp_low=low, tp_high=high, tp_step=step, tp_count=len(tp_values))

    precomputed = precompute_signal_tp_results(unique_signals, price_map, tp_values)
    if precomputed.empty:
        raise RuntimeError("预计算交易为空")
    precomputed.to_csv(round_dir / "precomputed_trades.csv", index=False, encoding="utf-8-sig")

    profiles = build_profiles(signals, tp_values)
    trades_df = build_trade_candidates(signals, profiles, precomputed)
    if trades_df.empty:
        raise RuntimeError("本轮交易候选为空")
    trades_df.to_csv(round_dir / "trade_candidates.csv", index=False, encoding="utf-8-sig")

    update_progress(result_dir, "simulating_round", round=round_idx, profile_count=int(trades_df["profile_name"].nunique()))
    account_df, equity_df = simulate_account(trades_df, price_map)
    if account_df.empty:
        raise RuntimeError("本轮账户结果为空")
    account_df.to_csv(round_dir / "account_summary.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(round_dir / "equity_curves.csv", index=False, encoding="utf-8-sig")

    best_row = account_df.iloc[0].to_dict()
    summary = {
        "round": round_idx,
        "tp_low": low,
        "tp_high": high,
        "tp_step": step,
        "tp_values": tp_values,
        "best_profile": best_row,
        "best_is_lower_edge": math.isclose(float(best_row["tp_value"]), float(min(tp_values)), rel_tol=0, abs_tol=1e-9),
        "best_is_upper_edge": math.isclose(float(best_row["tp_value"]), float(max(tp_values)), rel_tol=0, abs_tol=1e-9),
    }
    (round_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def choose_next_range(summary: dict[str, Any]) -> tuple[float, float, float] | None:
    low = float(summary["tp_low"])
    high = float(summary["tp_high"])
    step = float(summary["tp_step"])
    best_tp = float(summary["best_profile"]["tp_value"])

    if summary["best_is_lower_edge"]:
        new_step = max(0.0025, round(step / 2, 6))
        new_low = max(0.0025, round(best_tp - step * 4, 6))
        return new_low, high, new_step
    if summary["best_is_upper_edge"]:
        new_high = round(high + max(step * 4, 0.04), 6)
        return low, new_high, step
    return None


def run(mode: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = OUTPUT_ROOT / f"b3_fixed_tp_boundary_search_v1_{mode}_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "loading_source_signals")

    signals = build_source_signals()
    unique_signals = build_unique_signals(signals, mode)
    if mode == "smoke":
        signal_keys = set(zip(unique_signals["code"], unique_signals["signal_date"]))
        signals = signals[signals.apply(lambda r: (str(r["code"]), pd.Timestamp(r["signal_date"])) in signal_keys, axis=1)].reset_index(drop=True)
    codes = sorted(unique_signals["code"].astype(str).unique().tolist())
    update_progress(result_dir, "loading_price_data", code_count=len(codes), unique_signal_count=int(len(unique_signals)))
    price_map = load_price_map(codes)

    low, high, step = (0.03, 0.08, 0.01) if mode == "smoke" else (0.01, 0.15, 0.01)
    round_summaries: list[dict[str, Any]] = []
    for round_idx in range(1, 6):
        summary = run_one_round(result_dir, round_idx, signals, unique_signals, price_map, low, high, step)
        round_summaries.append(summary)
        next_range = choose_next_range(summary)
        if next_range is None:
            break
        low, high, step = next_range

    best_round = max(round_summaries, key=lambda x: float(x["best_profile"]["annual_return"]))
    final_summary = {
        "assumptions": {
            "data_dir": str(DATA_DIR),
            "mode": mode,
            "max_workers": MAX_WORKERS,
            "entry": "signal_date_next_open",
            "sort_rule": "amplitude_div_ret1_ascending",
            "allocation": "equal_weight_remaining_cash",
            "lot_size": LOT_SIZE,
            "stop": "entry_day_low_same_day",
            "fixed_tp_exec": "next_day_open",
            "max_hold_days": MAX_HOLD_DAYS,
            "forced_exit": "hold_30_next_day_open",
        },
        "round_count": len(round_summaries),
        "rounds": round_summaries,
        "best_round": best_round,
        "best_profile": best_round["best_profile"],
    }
    (result_dir / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished", round_count=len(round_summaries), best_tp=float(best_round["best_profile"]["tp_value"]))
    return result_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    out = run(args.mode)
    print(json.dumps({"result_dir": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
