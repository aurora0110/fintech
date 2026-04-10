from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import technical_indicators


SIGNAL_RESULT_DIR = ROOT / "results" / "b1_long_bear_short_volume_compare_v1_20260407_171225"
SIGNAL_DETAIL_PATH = SIGNAL_RESULT_DIR / "signal_detail.csv"
DATA_DIR = ROOT / "data" / "20260402"
RESULTS_DIR = ROOT / "results"
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
LOT_SIZE = 100
MAX_HOLD_DAYS = 120
TRADING_DAYS_PER_YEAR = 252
DRAWDOWN_PCT = 0.09
STOP_MULTIPLIER = 0.95
GROUPS = ("all_context", "long_bear_short_vol", "long_bear_long_vol", "short_bear_short_vol", "short_bear_long_vol")


@dataclass
class Position:
    code: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    signal_low: float
    group_name: str
    rolling_high: float
    holding_days: int
    scheduled_exit_date: pd.Timestamp | None = None
    scheduled_exit_reason: str | None = None
    stop_price: float = np.nan


def _load_signal_df() -> pd.DataFrame:
    df = pd.read_csv(SIGNAL_DETAIL_PATH, parse_dates=["signal_date"])
    return df[["code", "signal_date", "group_name", "body_atr", "vol_ratio", "j_rank20"]].copy()


def _load_price_df(code: str) -> pd.DataFrame:
    paths = list(DATA_DIR.glob(f"*#{code}.txt"))
    if not paths:
        return pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘"])
    df = technical_indicators._load_price_data(str(paths[0]))
    if df.empty:
        return pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘"])
    return df[["日期", "开盘", "最高", "最低", "收盘"]].copy()


def _build_candidates(signal_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DatetimeIndex]:
    rows: list[dict] = []
    price_map: dict[str, pd.DataFrame] = {}
    all_dates: list[pd.Timestamp] = []

    for code, sub in signal_df.groupby("code"):
        price_df = _load_price_df(code)
        if price_df.empty:
            continue
        price_df = price_df.sort_values("日期").reset_index(drop=True)
        price_df["日期"] = pd.to_datetime(price_df["日期"])
        price_map[code] = price_df
        all_dates.extend(price_df["日期"].tolist())
        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(price_df["日期"])}

        for rec in sub.itertuples(index=False):
            signal_date = pd.Timestamp(rec.signal_date)
            idx = date_to_idx.get(signal_date)
            if idx is None or idx + 1 >= len(price_df):
                continue
            entry_idx = idx + 1
            entry_date = pd.Timestamp(price_df["日期"].iloc[entry_idx])
            entry_open = float(price_df["开盘"].iloc[entry_idx])
            signal_low = float(price_df["最低"].iloc[idx])
            if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(signal_low) or signal_low <= 0:
                continue

            row = {
                "code": code,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "entry_open": entry_open,
                "signal_low": signal_low,
                "group_name": rec.group_name,
                "body_atr": float(rec.body_atr),
                "vol_ratio": float(rec.vol_ratio),
                "j_rank20": float(rec.j_rank20),
            }
            rows.append(row)
            if rec.group_name != "all_context":
                alias = row.copy()
                alias["group_name"] = "all_context"
                rows.append(alias)

    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        raise RuntimeError("candidate_df 为空，无法继续")
    candidate_df = candidate_df.sort_values(["entry_date", "j_rank20", "body_atr", "vol_ratio", "code"], ascending=[True, True, False, True, True]).reset_index(drop=True)
    calendar = pd.DatetimeIndex(sorted(pd.unique(pd.Series(all_dates).dropna())))
    return candidate_df, price_map, calendar


def _mark_to_market(positions: dict[str, Position], current_date: pd.Timestamp, price_map: dict[str, pd.DataFrame]) -> float:
    total = 0.0
    for code, pos in positions.items():
        df = price_map[code]
        hit = df[df["日期"] == current_date]
        if hit.empty:
            continue
        px = float(hit["收盘"].iloc[0])
        if np.isfinite(px):
            total += pos.shares * px
    return total


def _summarize(nav_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    nav = nav_df["nav"].astype(float)
    if nav.empty or (nav <= 0).any():
        raise RuntimeError("净值序列无效")
    peak = nav.cummax()
    dd = nav / peak - 1.0
    max_drawdown = float(dd.min())
    if max_drawdown < -1.0:
        raise RuntimeError("最大回撤小于-100%，结果无效")
    final_nav = float(nav.iloc[-1])
    holding_return = final_nav / INITIAL_CAPITAL - 1.0
    years = len(nav_df) / TRADING_DAYS_PER_YEAR if len(nav_df) > 0 else np.nan
    annual_return = float((final_nav / INITIAL_CAPITAL) ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    daily_ret = nav.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = float(daily_ret.std(ddof=1)) if len(daily_ret) > 1 else np.nan
    sharpe = float(daily_ret.mean() / (std + 1e-12) * np.sqrt(TRADING_DAYS_PER_YEAR)) if np.isfinite(std) else np.nan
    return {
        "initial_capital": INITIAL_CAPITAL,
        "final_nav": final_nav,
        "holding_return": holding_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trade_count": int(len(trades_df)),
        "win_rate": float((trades_df["holding_return"] > 0).mean()) if not trades_df.empty else np.nan,
        "avg_holding_return": float(trades_df["holding_return"].mean()) if not trades_df.empty else np.nan,
        "avg_holding_days": float(trades_df["holding_days"].mean()) if not trades_df.empty else np.nan,
    }


def _simulate_group(group_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cash = INITIAL_CAPITAL
    positions: dict[str, Position] = {}
    nav_rows: list[dict] = []
    trade_rows: list[dict] = []
    entries = {d: g.copy() for d, g in group_df.groupby("entry_date")}

    for current_date in calendar:
        for code, pos in list(positions.items()):
            if pos.scheduled_exit_date is not None and pos.scheduled_exit_date == current_date:
                df = price_map[code]
                hit = df[df["日期"] == current_date]
                if hit.empty:
                    continue
                exit_open = float(hit["开盘"].iloc[0])
                if not np.isfinite(exit_open) or exit_open <= 0:
                    continue
                cash += pos.shares * exit_open
                trade_rows.append(
                    {
                        "code": code,
                        "signal_date": pos.signal_date,
                        "entry_date": pos.entry_date,
                        "exit_date": current_date,
                        "shares": pos.shares,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_open,
                        "holding_return": exit_open / pos.entry_price - 1.0,
                        "holding_days": pos.holding_days,
                        "exit_reason": pos.scheduled_exit_reason,
                    }
                )
                positions.pop(code, None)

        todays = entries.get(current_date)
        if todays is not None and not todays.empty:
            todays = todays[~todays["code"].isin(positions.keys())].copy()
            slots = MAX_POSITIONS - len(positions)
            if slots > 0 and not todays.empty:
                todays = todays.sort_values(["j_rank20", "body_atr", "vol_ratio", "code"], ascending=[True, False, True, True]).head(slots)
                allocation = cash / len(todays) if len(todays) else 0.0
                for rec in todays.itertuples(index=False):
                    shares = int(math.floor(allocation / rec.entry_open / LOT_SIZE) * LOT_SIZE)
                    if shares < LOT_SIZE:
                        continue
                    cost = shares * rec.entry_open
                    if cost > cash + 1e-9:
                        continue
                    cash -= cost
                    positions[rec.code] = Position(
                        code=rec.code,
                        signal_date=pd.Timestamp(rec.signal_date),
                        entry_date=pd.Timestamp(rec.entry_date),
                        entry_price=float(rec.entry_open),
                        shares=shares,
                        signal_low=float(rec.signal_low),
                        group_name=str(rec.group_name),
                        rolling_high=float(rec.entry_open),
                        holding_days=1,
                        stop_price=float(rec.signal_low) * STOP_MULTIPLIER,
                    )

        for code, pos in list(positions.items()):
            if current_date <= pos.entry_date:
                continue
            df = price_map[code]
            hit = df[df["日期"] == current_date]
            if hit.empty:
                continue
            high_price = float(hit["最高"].iloc[0])
            low_price = float(hit["最低"].iloc[0])
            if np.isfinite(high_price):
                pos.rolling_high = max(pos.rolling_high, high_price)
            stop_hit = np.isfinite(low_price) and np.isfinite(pos.stop_price) and low_price <= pos.stop_price
            drawdown_hit = np.isfinite(low_price) and np.isfinite(pos.rolling_high) and low_price <= pos.rolling_high * (1.0 - DRAWDOWN_PCT)

            if stop_hit and pos.scheduled_exit_date is None:
                cash += pos.shares * pos.stop_price
                trade_rows.append(
                    {
                        "code": code,
                        "signal_date": pos.signal_date,
                        "entry_date": pos.entry_date,
                        "exit_date": current_date,
                        "shares": pos.shares,
                        "entry_price": pos.entry_price,
                        "exit_price": pos.stop_price,
                        "holding_return": pos.stop_price / pos.entry_price - 1.0,
                        "holding_days": pos.holding_days,
                        "exit_reason": "stop_loss_same_day",
                    }
                )
                positions.pop(code, None)
                continue

            if drawdown_hit and pos.scheduled_exit_date is None:
                next_hits = df[df["日期"] > current_date]
                if not next_hits.empty:
                    pos.scheduled_exit_date = pd.Timestamp(next_hits["日期"].iloc[0])
                    pos.scheduled_exit_reason = "drawdown_9pct_next_open"

            if pos.holding_days >= MAX_HOLD_DAYS and pos.scheduled_exit_date is None:
                next_hits = df[df["日期"] > current_date]
                if not next_hits.empty:
                    pos.scheduled_exit_date = pd.Timestamp(next_hits["日期"].iloc[0])
                    pos.scheduled_exit_reason = "max_hold_next_open"

            pos.holding_days += 1

        market_value = _mark_to_market(positions, current_date, price_map)
        nav_rows.append(
            {
                "date": current_date,
                "cash": cash,
                "market_value": market_value,
                "nav": cash + market_value,
                "open_positions": len(positions),
            }
        )

    nav_df = pd.DataFrame(nav_rows)
    trades_df = pd.DataFrame(trade_rows)
    summary = _summarize(nav_df, trades_df)
    return nav_df, trades_df, summary


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / f"b1_long_bear_short_volume_b1exit_v1_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)

    signal_df = _load_signal_df()
    candidate_df, price_map, calendar = _build_candidates(signal_df)
    candidate_df.to_csv(result_dir / "trade_candidates.csv", index=False)

    summary_rows: list[dict] = []
    nav_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []

    for group_name in GROUPS:
        group_df = candidate_df[candidate_df["group_name"] == group_name].copy()
        if group_df.empty:
            continue
        nav_df, trades_df, summary = _simulate_group(group_df, price_map, calendar)
        profile_name = f"{group_name}__b1_best_exit"
        summary["profile_name"] = profile_name
        summary["group_name"] = group_name
        summary["signal_count"] = int(len(group_df))
        summary_rows.append(summary)
        nav_df["profile_name"] = profile_name
        trades_df["profile_name"] = profile_name
        nav_rows.append(nav_df)
        trade_rows.append(trades_df)

    account_df = pd.DataFrame(summary_rows).sort_values(["annual_return", "holding_return"], ascending=[False, False]).reset_index(drop=True)
    nav_all = pd.concat(nav_rows, ignore_index=True) if nav_rows else pd.DataFrame()
    trades_all = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()

    account_df.to_csv(result_dir / "account_summary.csv", index=False)
    nav_all.to_csv(result_dir / "daily_nav.csv", index=False)
    trades_all.to_csv(result_dir / "trades.csv", index=False)

    payload = {
        "result_dir": str(result_dir),
        "signal_result_dir": str(SIGNAL_RESULT_DIR),
        "buy_rule": "signal_date次日开盘买",
        "stop_rule": "signal_low*0.95, 触发后当日按止损价卖",
        "take_profit_rule": "最高点回撤9%, 触发后次日开盘卖",
        "max_hold_days": MAX_HOLD_DAYS,
        "lot_size": LOT_SIZE,
        "max_positions": MAX_POSITIONS,
        "profiles": account_df.to_dict(orient="records"),
    }
    (result_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"结果目录: {result_dir}")
    print(account_df.to_string(index=False))


if __name__ == "__main__":
    main()
