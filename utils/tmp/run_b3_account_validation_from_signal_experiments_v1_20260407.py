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
HORIZONS = [1, 3, 5, 10]


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


technical = load_module(ROOT / "utils" / "technical_indicators.py", "b3_acc_validate_technical")
b3filter = load_module(ROOT / "utils" / "B3filter.py", "b3_acc_validate_b3")


@dataclass
class TradeSignal:
    profile_name: str
    experiment_family: str
    group_name: str
    horizon: int
    code: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_open: float
    exit_close: float
    holding_return: float
    b3_score: float


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return float("nan")
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


def load_code_feature_pair(code: str) -> tuple[str, pd.DataFrame]:
    raw = load_one_daily(DATA_DIR / f"{code}.txt")
    if raw.empty:
        return code, pd.DataFrame()
    feat = b3filter.add_features(raw)
    return code, feat[["date", "b3_score"]].copy() if not feat.empty else pd.DataFrame()


def load_feature_map(codes: list[str]) -> dict[str, pd.DataFrame]:
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=MAX_WORKERS) as pool:
        parts = pool.map(load_code_feature_pair, codes, chunksize=1)
    return {code: df for code, df in parts if df is not None and not df.empty}


def load_price_map(codes: list[str]) -> dict[str, pd.DataFrame]:
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=MAX_WORKERS) as pool:
        parts = pool.map(load_one_daily, [DATA_DIR / f"{code}.txt" for code in codes], chunksize=1)
    out: dict[str, pd.DataFrame] = {}
    for code, df in zip(codes, parts):
        if df is not None and not df.empty:
            out[code] = df
    return out


def _entry_exit_from_daily(df: pd.DataFrame, signal_date: pd.Timestamp, horizon: int) -> tuple[pd.Timestamp, pd.Timestamp, float, float] | None:
    locs = np.flatnonzero(df["date"].to_numpy(dtype="datetime64[ns]") == np.datetime64(signal_date))
    if len(locs) == 0:
        return None
    idx = int(locs[0])
    entry_idx = idx + 1
    exit_idx = idx + horizon
    if entry_idx >= len(df) or exit_idx >= len(df):
        return None
    entry_open = float(df.iloc[entry_idx]["open"])
    exit_close = float(df.iloc[exit_idx]["close"])
    if not np.isfinite(entry_open) or not np.isfinite(exit_close) or entry_open <= 0:
        return None
    return pd.Timestamp(df.iloc[entry_idx]["date"]), pd.Timestamp(df.iloc[exit_idx]["date"]), entry_open, exit_close


def build_structure_signals(price_map: dict[str, pd.DataFrame], feature_map: dict[str, pd.DataFrame]) -> list[TradeSignal]:
    detail = pd.read_csv(STRUCTURE_DETAIL, parse_dates=["signal_date"])
    out: list[TradeSignal] = []
    flag_defs = [
        ("ge2_true", "second_red_after_green_ge2", True),
        ("ge2_false", "second_red_after_green_ge2", False),
        ("ge3_true", "second_red_after_green_ge3", True),
        ("ge3_false", "second_red_after_green_ge3", False),
        ("ge4_true", "second_red_after_green_ge4", True),
        ("ge4_false", "second_red_after_green_ge4", False),
    ]
    for _, row in detail.iterrows():
        code = str(row["code"])
        if code not in price_map or code not in feature_map:
            continue
        score_df = feature_map[code]
        score_match = score_df[score_df["date"] == pd.Timestamp(row["signal_date"])]
        if score_match.empty:
            continue
        b3_score = float(score_match.iloc[0]["b3_score"])
        for horizon in HORIZONS:
            entry_exit = _entry_exit_from_daily(price_map[code], pd.Timestamp(row["signal_date"]), horizon)
            if entry_exit is None:
                continue
            entry_date, exit_date, entry_open, exit_close = entry_exit
            holding_return = float(exit_close / entry_open - 1.0)
            for group_name, col, expected in flag_defs:
                if bool(row[col]) != expected:
                    continue
                out.append(
                    TradeSignal(
                        profile_name=f"struct__{group_name}__h{horizon}",
                        experiment_family="structure",
                        group_name=group_name,
                        horizon=horizon,
                        code=code,
                        signal_date=pd.Timestamp(row["signal_date"]),
                        entry_date=entry_date,
                        exit_date=exit_date,
                        entry_open=entry_open,
                        exit_close=exit_close,
                        holding_return=holding_return,
                        b3_score=b3_score,
                    )
                )
    return out


def build_volume_signals(price_map: dict[str, pd.DataFrame]) -> list[TradeSignal]:
    detail = pd.read_csv(VOLUME_DETAIL, parse_dates=["signal_date"])
    out: list[TradeSignal] = []
    for _, row in detail.iterrows():
        code = str(row["code"])
        if code not in price_map:
            continue
        for horizon in HORIZONS:
            entry_exit = _entry_exit_from_daily(price_map[code], pd.Timestamp(row["signal_date"]), horizon)
            if entry_exit is None:
                continue
            entry_date, exit_date, entry_open, exit_close = entry_exit
            out.append(
                TradeSignal(
                    profile_name=f"volume__{row['volume_group']}__h{horizon}",
                    experiment_family="volume",
                    group_name=str(row["volume_group"]),
                    horizon=horizon,
                    code=code,
                    signal_date=pd.Timestamp(row["signal_date"]),
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_open=entry_open,
                    exit_close=exit_close,
                    holding_return=float(exit_close / entry_open - 1.0),
                    b3_score=float(row.get("b3_score", np.nan)),
                )
            )
    return out


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
    if trades_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_dates = sorted({d for df in price_map.values() for d in pd.to_datetime(df["date"]).tolist()})
    price_close_map = {
        code: df.set_index("date")["close"].astype(float).to_dict()
        for code, df in price_map.items()
    }

    results: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for profile_name, g in trades_df.groupby("profile_name"):
        g = g.sort_values(["entry_date", "b3_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
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
                    candidates: list[dict[str, Any]] = []
                    seen_codes = set()
                    for row in todays:
                        code = str(row["code"])
                        if code in positions or code in seen_codes:
                            continue
                        seen_codes.add(code)
                        candidates.append(row)
                    selected = candidates[:available_slots]
                    if selected:
                        budget = cash / len(selected)
                        for row in selected:
                            entry_open = float(row["entry_open"])
                            if not np.isfinite(entry_open) or entry_open <= 0:
                                continue
                            shares = int(budget // (entry_open * LOT_SIZE)) * LOT_SIZE
                            if shares <= 0:
                                continue
                            cost = shares * entry_open
                            if cost > cash + 1e-9:
                                continue
                            cash -= cost
                            positions[str(row["code"])] = {
                                **row,
                                "shares": shares,
                                "cost": cost,
                            }

            equity = cash
            for code, pos in positions.items():
                close_price = price_close_map.get(code, {}).get(pd.Timestamp(current_date))
                if close_price is None or not np.isfinite(close_price):
                    close_price = pos["entry_open"]
                equity += float(pos["shares"]) * float(close_price)
            curve.append({"profile_name": profile_name, "date": pd.Timestamp(current_date), "equity": float(equity)})

            to_exit = [code for code, pos in positions.items() if pd.Timestamp(pos["exit_date"]) == pd.Timestamp(current_date)]
            for code in to_exit:
                pos = positions.pop(code)
                proceeds = float(pos["shares"]) * float(pos["exit_close"])
                cash += proceeds
                completed.append(
                    {
                        "profile_name": profile_name,
                        "code": code,
                        "signal_date": pos["signal_date"],
                        "entry_date": pos["entry_date"],
                        "exit_date": pos["exit_date"],
                        "entry_open": pos["entry_open"],
                        "exit_close": pos["exit_close"],
                        "shares": int(pos["shares"]),
                        "holding_return": float(pos["holding_return"]),
                        "holding_days": int(pos["horizon"]),
                        "pnl": proceeds - float(pos["cost"]),
                    }
                )

        equity_df = pd.DataFrame(curve)
        completed_df = pd.DataFrame(completed)
        if completed_df.empty or equity_df.empty:
            continue

        final_equity = float(equity_df.iloc[-1]["equity"])
        hold_return = float(final_equity / INITIAL_CAPITAL - 1.0)
        results.append(
            {
                "profile_name": profile_name,
                "experiment_family": str(g.iloc[0]["experiment_family"]),
                "group_name": str(g.iloc[0]["group_name"]),
                "horizon": int(g.iloc[0]["horizon"]),
                "signal_count": int(len(g)),
                "trade_count": int(len(completed_df)),
                "annual_return": annual_return(equity_df["equity"]),
                "holding_return": hold_return,
                "max_drawdown": max_drawdown(equity_df["equity"]),
                "sharpe": sharpe_ratio(equity_df["equity"]),
                "success_rate": float((completed_df["holding_return"] > 0).mean()),
                "avg_trade_return": float(completed_df["holding_return"].mean()),
                "avg_holding_return": float(completed_df["holding_return"].mean()),
                "avg_holding_days": float(completed_df["holding_days"].mean()),
                "final_equity": final_equity,
            }
        )
        equity_rows.append(equity_df)

    account_df = pd.DataFrame(results)
    if not account_df.empty:
        account_df = account_df.sort_values(
            ["experiment_family", "annual_return", "holding_return", "success_rate", "max_drawdown"],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
    equity_all = pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame()
    return account_df, equity_all


def build_summary_tables(account_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if account_df.empty:
        return tables
    structure = account_df[account_df["experiment_family"] == "structure"].copy()
    volume = account_df[account_df["experiment_family"] == "volume"].copy()
    if not structure.empty:
        tables["structure"] = structure
    if not volume.empty:
        tables["volume"] = volume
    return tables


def run() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = OUTPUT_ROOT / f"b3_account_validation_from_signal_experiments_v1_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "loading_detail_files")

    struct_codes = sorted(pd.read_csv(STRUCTURE_DETAIL, usecols=["code"])["code"].astype(str).unique().tolist())
    vol_codes = sorted(pd.read_csv(VOLUME_DETAIL, usecols=["code"])["code"].astype(str).unique().tolist())
    all_codes = sorted(set(struct_codes) | set(vol_codes))

    update_progress(result_dir, "loading_price_data", code_count=len(all_codes))
    price_map = load_price_map(all_codes)
    feature_map = load_feature_map(sorted(set(struct_codes)))

    update_progress(result_dir, "building_trade_signals", structure_codes=len(struct_codes), volume_codes=len(vol_codes))
    structure_signals = build_structure_signals(price_map, feature_map)
    volume_signals = build_volume_signals(price_map)
    all_signals = structure_signals + volume_signals
    signals_df = pd.DataFrame([s.__dict__ for s in all_signals])
    if signals_df.empty:
        raise RuntimeError("未生成任何账户层交易信号")
    signals_df.to_csv(result_dir / "trade_signals.csv", index=False, encoding="utf-8-sig")

    update_progress(result_dir, "simulating_account", profile_count=int(signals_df["profile_name"].nunique()))
    account_df, equity_df = simulate_account(signals_df, price_map)
    if account_df.empty or equity_df.empty:
        raise RuntimeError("账户层结果为空")
    account_df.to_csv(result_dir / "account_summary.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(result_dir / "equity_curves.csv", index=False, encoding="utf-8-sig")

    tables = build_summary_tables(account_df)
    for name, df in tables.items():
        df.to_csv(result_dir / f"{name}_account_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "data_dir": str(DATA_DIR),
            "account_rule": {
                "initial_capital": INITIAL_CAPITAL,
                "max_positions": MAX_POSITIONS,
                "allocation": "equal_weight",
                "entry": "next_day_open",
                "exit": "horizon_day_close",
                "same_code_reentry": "disallow_while_holding",
                "same_day_sort": "b3_score_desc",
            },
            "horizons": HORIZONS,
            "source_experiments": {
                "structure": str(STRUCTURE_DETAIL),
                "volume": str(VOLUME_DETAIL),
            },
        },
        "best_structure_profile": tables["structure"].iloc[0].to_dict() if "structure" in tables and not tables["structure"].empty else {},
        "best_volume_profile": tables["volume"].iloc[0].to_dict() if "volume" in tables and not tables["volume"].empty else {},
        "profile_count": int(account_df["profile_name"].nunique()),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished", profile_count=int(account_df["profile_name"].nunique()))
    return result_dir


def main() -> None:
    run()


if __name__ == "__main__":
    main()
