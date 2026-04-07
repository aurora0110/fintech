from __future__ import annotations

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
FIXED_TPS = [0.05, 0.10, 0.15, 0.20, 0.30]
DRAWDOWN_TPS = [0.03, 0.05, 0.08, 0.11]


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


technical = load_module(ROOT / "utils" / "technical_indicators.py", "b3_acc_strict_technical")


@dataclass(frozen=True)
class ExitProfile:
    family: str
    group_name: str
    tp_kind: str
    tp_value: float

    @property
    def profile_name(self) -> str:
        if self.tp_kind == "fixed_tp":
            tag = f"{self.tp_value * 100:.2f}%"
        else:
            tag = f"{self.tp_value * 100:.2f}%"
        return f"{self.family}__{self.group_name}__{self.tp_kind}_{tag}"


@dataclass
class TradeCandidate:
    profile_name: str
    family: str
    group_name: str
    code: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_open: float
    stop_price: float
    sort_ratio: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str
    holding_days: int
    trade_return: float


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_div(a: float, b: float, default: float = float("nan")) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return default
    return float(a / b)


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


def build_source_signals() -> pd.DataFrame:
    struct = pd.read_csv(STRUCTURE_DETAIL, parse_dates=["signal_date"]).copy()
    struct["family"] = "structure"
    struct_rows: list[pd.DataFrame] = []
    for col in ["second_red_after_green_ge2", "second_red_after_green_ge3", "second_red_after_green_ge4"]:
        base = struct.copy()
        base["group_name"] = np.where(base[col].fillna(False), col.replace("second_red_after_green_", "") + "_true", col.replace("second_red_after_green_", "") + "_false")
        struct_rows.append(base[["code", "signal_date", "family", "group_name", "ret1", "amplitude"]])
    struct_df = pd.concat(struct_rows, ignore_index=True)

    vol = pd.read_csv(VOLUME_DETAIL, parse_dates=["signal_date"]).copy()
    vol["family"] = "volume"
    vol["group_name"] = vol["volume_group"].astype(str)
    vol_df = vol[["code", "signal_date", "family", "group_name", "ret1", "amplitude"]].copy()

    signals = pd.concat([struct_df, vol_df], ignore_index=True)
    signals["sort_ratio"] = signals.apply(
        lambda r: safe_div(float(r["amplitude"]), float(r["ret1"]), default=float("inf")) if float(r["ret1"]) > 0 else float("inf"),
        axis=1,
    )
    signals = signals.sort_values(["signal_date", "family", "group_name", "sort_ratio", "code"]).reset_index(drop=True)
    return signals


def build_exit_profiles(signals: pd.DataFrame) -> list[ExitProfile]:
    groups = signals[["family", "group_name"]].drop_duplicates().sort_values(["family", "group_name"])
    profiles: list[ExitProfile] = []
    for _, row in groups.iterrows():
        family = str(row["family"])
        group_name = str(row["group_name"])
        for tp in FIXED_TPS:
            profiles.append(ExitProfile(family=family, group_name=group_name, tp_kind="fixed_tp", tp_value=float(tp)))
        for dd in DRAWDOWN_TPS:
            profiles.append(ExitProfile(family=family, group_name=group_name, tp_kind="drawdown_tp", tp_value=float(dd)))
    return profiles


def evaluate_trade(signal_row: pd.Series, price_df: pd.DataFrame, profile: ExitProfile) -> TradeCandidate | None:
    signal_date = pd.Timestamp(signal_row["signal_date"])
    match = np.flatnonzero(price_df["date"].to_numpy(dtype="datetime64[ns]") == np.datetime64(signal_date))
    if len(match) == 0:
        return None
    signal_idx = int(match[0])
    entry_idx = signal_idx + 1
    if entry_idx >= len(price_df):
        return None

    entry_row = price_df.iloc[entry_idx]
    entry_date = pd.Timestamp(entry_row["date"])
    entry_open = float(entry_row["open"])
    stop_price = float(entry_row["low"])
    if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(stop_price) or stop_price <= 0:
        return None

    rolling_high = float(entry_row["high"])
    last_holding_idx = min(entry_idx + MAX_HOLD_DAYS - 1, len(price_df) - 1)

    for idx in range(entry_idx + 1, last_holding_idx + 1):
        row = price_df.iloc[idx]
        row_high = float(row["high"])
        row_low = float(row["low"])
        row_close = float(row["close"])

        if np.isfinite(row_high):
            rolling_high = max(rolling_high, row_high)

        tp_hit = False
        if profile.tp_kind == "fixed_tp":
            tp_price = entry_open * (1.0 + float(profile.tp_value))
            tp_hit = np.isfinite(row_high) and row_high >= tp_price
        else:
            tp_trigger_price = rolling_high * (1.0 - float(profile.tp_value))
            tp_hit = np.isfinite(row_low) and row_low <= tp_trigger_price and rolling_high > entry_open

        stop_hit = np.isfinite(row_low) and row_low <= stop_price

        if tp_hit:
            next_idx = idx + 1
            if next_idx < len(price_df):
                exit_row = price_df.iloc[next_idx]
                exit_date = pd.Timestamp(exit_row["date"])
                exit_price = float(exit_row["open"])
                if np.isfinite(exit_price) and exit_price > 0:
                    return TradeCandidate(
                        profile_name=profile.profile_name,
                        family=profile.family,
                        group_name=profile.group_name,
                        code=str(signal_row["code"]),
                        signal_date=signal_date,
                        entry_date=entry_date,
                        entry_open=entry_open,
                        stop_price=stop_price,
                        sort_ratio=float(signal_row["sort_ratio"]),
                        exit_date=exit_date,
                        exit_price=exit_price,
                        exit_reason=profile.tp_kind,
                        holding_days=idx - entry_idx + 1,
                        trade_return=float(exit_price / entry_open - 1.0),
                    )
            fallback_price = row_close if np.isfinite(row_close) and row_close > 0 else entry_open
            return TradeCandidate(
                profile_name=profile.profile_name,
                family=profile.family,
                group_name=profile.group_name,
                code=str(signal_row["code"]),
                signal_date=signal_date,
                entry_date=entry_date,
                entry_open=entry_open,
                stop_price=stop_price,
                sort_ratio=float(signal_row["sort_ratio"]),
                exit_date=pd.Timestamp(row["date"]),
                exit_price=float(fallback_price),
                exit_reason=profile.tp_kind + "_fallback_close",
                holding_days=idx - entry_idx + 1,
                trade_return=float(fallback_price / entry_open - 1.0),
            )

        if stop_hit:
            return TradeCandidate(
                profile_name=profile.profile_name,
                family=profile.family,
                group_name=profile.group_name,
                code=str(signal_row["code"]),
                signal_date=signal_date,
                entry_date=entry_date,
                entry_open=entry_open,
                stop_price=stop_price,
                sort_ratio=float(signal_row["sort_ratio"]),
                exit_date=pd.Timestamp(row["date"]),
                exit_price=stop_price,
                exit_reason="stop_same_day",
                holding_days=idx - entry_idx + 1,
                trade_return=float(stop_price / entry_open - 1.0),
            )

    forced_idx = last_holding_idx + 1
    if forced_idx < len(price_df):
        exit_row = price_df.iloc[forced_idx]
        exit_date = pd.Timestamp(exit_row["date"])
        exit_price = float(exit_row["open"])
        if np.isfinite(exit_price) and exit_price > 0:
            return TradeCandidate(
                profile_name=profile.profile_name,
                family=profile.family,
                group_name=profile.group_name,
                code=str(signal_row["code"]),
                signal_date=signal_date,
                entry_date=entry_date,
                entry_open=entry_open,
                stop_price=stop_price,
                sort_ratio=float(signal_row["sort_ratio"]),
                exit_date=exit_date,
                exit_price=exit_price,
                exit_reason="hold_30_next_open",
                holding_days=MAX_HOLD_DAYS,
                trade_return=float(exit_price / entry_open - 1.0),
            )
    exit_row = price_df.iloc[last_holding_idx]
    fallback_price = float(exit_row["close"])
    if not np.isfinite(fallback_price) or fallback_price <= 0:
        fallback_price = entry_open
    return TradeCandidate(
        profile_name=profile.profile_name,
        family=profile.family,
        group_name=profile.group_name,
        code=str(signal_row["code"]),
        signal_date=signal_date,
        entry_date=entry_date,
        entry_open=entry_open,
        stop_price=stop_price,
        sort_ratio=float(signal_row["sort_ratio"]),
        exit_date=pd.Timestamp(exit_row["date"]),
        exit_price=fallback_price,
        exit_reason="hold_30_fallback_close",
        holding_days=MAX_HOLD_DAYS,
        trade_return=float(fallback_price / entry_open - 1.0),
    )


def build_trade_candidates(signals: pd.DataFrame, profiles: list[ExitProfile], price_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile_idx, profile in enumerate(profiles, start=1):
        group_signals = signals[(signals["family"] == profile.family) & (signals["group_name"] == profile.group_name)].copy()
        for _, signal_row in group_signals.iterrows():
            code = str(signal_row["code"])
            price_df = price_map.get(code)
            if price_df is None or price_df.empty:
                continue
            trade = evaluate_trade(signal_row, price_df, profile)
            if trade is None:
                continue
            rows.append(trade.__dict__)
        if profile_idx % 20 == 0:
            print({"trade_profile_progress": profile_idx, "total_profiles": len(profiles), "trade_rows": len(rows)}, flush=True)
    return pd.DataFrame(rows)


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
                "tp_kind": str(g.iloc[0]["exit_reason"]).split("_")[0] if "exit_reason" in g.columns else "",
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
            ["family", "annual_return", "holding_return", "success_rate", "max_drawdown"],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
    equity_all = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    return account_df, equity_all


def run() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = OUTPUT_ROOT / f"b3_account_validation_strict_exit_v1_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "loading_source_signals")

    signals = build_source_signals()
    profiles = build_exit_profiles(signals)
    codes = sorted(signals["code"].astype(str).unique().tolist())

    update_progress(result_dir, "loading_price_data", code_count=len(codes))
    price_map = load_price_map(codes)

    update_progress(result_dir, "building_trade_candidates", profile_count=len(profiles))
    trades_df = build_trade_candidates(signals, profiles, price_map)
    if trades_df.empty:
        raise RuntimeError("未生成任何交易候选")
    trades_df.to_csv(result_dir / "trade_candidates.csv", index=False, encoding="utf-8-sig")

    update_progress(result_dir, "simulating_account", profile_count=int(trades_df["profile_name"].nunique()))
    account_df, equity_df = simulate_account(trades_df, price_map)
    if account_df.empty or equity_df.empty:
        raise RuntimeError("账户层结果为空")
    account_df.to_csv(result_dir / "account_summary.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(result_dir / "equity_curves.csv", index=False, encoding="utf-8-sig")

    structure_df = account_df[account_df["family"] == "structure"].copy()
    volume_df = account_df[account_df["family"] == "volume"].copy()
    structure_df.to_csv(result_dir / "structure_account_summary.csv", index=False, encoding="utf-8-sig")
    volume_df.to_csv(result_dir / "volume_account_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "data_dir": str(DATA_DIR),
            "account_rule": {
                "initial_capital": INITIAL_CAPITAL,
                "max_positions": MAX_POSITIONS,
                "allocation": "equal_weight_remaining_cash",
                "entry": "next_day_open",
                "sort_rule": "amplitude_div_ret1_ascending",
                "lot_size": LOT_SIZE,
                "stop": "entry_day_low_same_day",
                "tp": {
                    "drawdown": DRAWDOWN_TPS,
                    "fixed": FIXED_TPS,
                    "tp_exec": "next_day_open",
                    "tp_priority_over_stop_same_day": True,
                },
                "max_hold_days": MAX_HOLD_DAYS,
                "forced_exit": "hold_30_next_day_open",
                "same_code_reentry": "disallow_while_holding",
            },
            "source_experiments": {
                "structure": str(STRUCTURE_DETAIL),
                "volume": str(VOLUME_DETAIL),
            },
        },
        "best_structure_profile": structure_df.iloc[0].to_dict() if not structure_df.empty else {},
        "best_volume_profile": volume_df.iloc[0].to_dict() if not volume_df.empty else {},
        "profile_count": int(account_df["profile_name"].nunique()),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished", profile_count=int(account_df["profile_name"].nunique()))
    return result_dir


def main() -> None:
    run()


if __name__ == "__main__":
    main()
