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
HOLD_DAYS = (1, 3, 5, 10)
GROUPS = ("all_context", "long_bear_short_vol", "long_bear_long_vol", "short_bear_short_vol", "short_bear_long_vol")


@dataclass
class Position:
    code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    shares: int
    entry_price: float
    signal_date: pd.Timestamp
    group_name: str
    rank_key: tuple


def _load_signal_df() -> pd.DataFrame:
    df = pd.read_csv(SIGNAL_DETAIL_PATH, parse_dates=["signal_date"])
    cols = ["code", "signal_date", "group_name", "body_atr", "vol_ratio", "j_rank20"]
    return df[cols].copy()


def _load_price_map(code: str) -> pd.DataFrame:
    candidates = list(DATA_DIR.glob(f"*#{code}.txt"))
    if not candidates:
        return pd.DataFrame(columns=["日期", "开盘", "收盘"])
    df = technical_indicators._load_price_data(str(candidates[0]))
    if df.empty:
        return pd.DataFrame(columns=["日期", "开盘", "收盘"])
    return df[["日期", "开盘", "收盘"]].copy()


def _build_trade_candidates(signal_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DatetimeIndex]:
    rows: list[dict] = []
    close_map: dict[str, pd.Series] = {}
    all_dates: list[pd.Timestamp] = []

    grouped = signal_df.groupby("code")
    for code, sub in grouped:
        price_df = _load_price_map(code)
        if price_df.empty:
            continue
        price_df = price_df.sort_values("日期").reset_index(drop=True)
        price_df["日期"] = pd.to_datetime(price_df["日期"])
        close_series = price_df.set_index("日期")["收盘"].astype(float)
        close_map[code] = close_series
        all_dates.extend(price_df["日期"].tolist())

        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(price_df["日期"])}
        for rec in sub.itertuples(index=False):
            signal_date = pd.Timestamp(rec.signal_date)
            idx = date_to_idx.get(signal_date)
            if idx is None:
                continue
            entry_idx = idx + 1
            if entry_idx >= len(price_df):
                continue
            entry_date = pd.Timestamp(price_df["日期"].iloc[entry_idx])
            entry_open = float(price_df["开盘"].iloc[entry_idx])
            if not np.isfinite(entry_open) or entry_open <= 0:
                continue
            for hold_days in HOLD_DAYS:
                exit_idx = entry_idx + hold_days - 1
                if exit_idx >= len(price_df):
                    continue
                exit_date = pd.Timestamp(price_df["日期"].iloc[exit_idx])
                exit_close = float(price_df["收盘"].iloc[exit_idx])
                if not np.isfinite(exit_close) or exit_close <= 0:
                    continue
                rows.append(
                    {
                        "code": code,
                        "signal_date": signal_date,
                        "group_name": rec.group_name,
                        "hold_days": hold_days,
                        "entry_date": entry_date,
                        "entry_open": entry_open,
                        "exit_date": exit_date,
                        "exit_close": exit_close,
                        "body_atr": float(rec.body_atr),
                        "vol_ratio": float(rec.vol_ratio),
                        "j_rank20": float(rec.j_rank20),
                    }
                )
                if rec.group_name != "all_context":
                    rows.append(
                        {
                            "code": code,
                            "signal_date": signal_date,
                            "group_name": "all_context",
                            "hold_days": hold_days,
                            "entry_date": entry_date,
                            "entry_open": entry_open,
                            "exit_date": exit_date,
                            "exit_close": exit_close,
                            "body_atr": float(rec.body_atr),
                            "vol_ratio": float(rec.vol_ratio),
                            "j_rank20": float(rec.j_rank20),
                        }
                    )

    trade_df = pd.DataFrame(rows)
    if trade_df.empty:
        raise RuntimeError("trade candidates 为空，无法继续账户层验证")
    trade_df = trade_df.sort_values(["entry_date", "j_rank20", "body_atr", "vol_ratio", "code"]).reset_index(drop=True)
    calendar = pd.DatetimeIndex(sorted(pd.unique(pd.Series(all_dates).dropna())))
    return trade_df, close_map, calendar


def _simulate_profile(profile_df: pd.DataFrame, close_map: dict[str, pd.Series], calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cash = INITIAL_CAPITAL
    positions: dict[str, Position] = {}
    trade_rows: list[dict] = []
    nav_rows: list[dict] = []

    entry_groups = {d: g.copy() for d, g in profile_df.groupby("entry_date")}

    for current_date in calendar:
        todays = entry_groups.get(current_date)
        if todays is not None and not todays.empty:
            available_slots = MAX_POSITIONS - len(positions)
            if available_slots > 0:
                todays = todays[~todays["code"].isin(positions.keys())].copy()
                if not todays.empty:
                    todays = todays.sort_values(["j_rank20", "body_atr", "vol_ratio", "code"], ascending=[True, False, True, True])
                    todays = todays.head(available_slots)
                    allocation = cash / len(todays) if len(todays) > 0 else 0.0
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
                            entry_date=pd.Timestamp(rec.entry_date),
                            exit_date=pd.Timestamp(rec.exit_date),
                            shares=shares,
                            entry_price=float(rec.entry_open),
                            signal_date=pd.Timestamp(rec.signal_date),
                            group_name=str(rec.group_name),
                            rank_key=(float(rec.j_rank20), -float(rec.body_atr), float(rec.vol_ratio), str(rec.code)),
                        )

        exit_codes = [code for code, pos in positions.items() if pos.exit_date == current_date]
        for code in exit_codes:
            pos = positions.pop(code)
            proceeds = pos.shares * float(close_map[code].get(current_date, np.nan))
            exit_price = float(close_map[code].get(current_date, np.nan))
            if not np.isfinite(proceeds) or not np.isfinite(exit_price):
                continue
            cash += proceeds
            trade_rows.append(
                {
                    "code": code,
                    "signal_date": pos.signal_date,
                    "entry_date": pos.entry_date,
                    "exit_date": current_date,
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "holding_days": int((current_date - pos.entry_date).days) + 1,
                    "holding_return": exit_price / pos.entry_price - 1.0,
                }
            )

        market_value = 0.0
        for code, pos in positions.items():
            px = float(close_map[code].get(current_date, np.nan))
            if np.isfinite(px):
                market_value += pos.shares * px

        nav = cash + market_value
        nav_rows.append(
            {
                "date": current_date,
                "cash": cash,
                "market_value": market_value,
                "nav": nav,
                "open_positions": len(positions),
            }
        )

    nav_df = pd.DataFrame(nav_rows)
    trades_df = pd.DataFrame(trade_rows)
    summary = _summarize(nav_df, trades_df)
    return nav_df, trades_df, summary


def _summarize(nav_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    if nav_df.empty:
        raise RuntimeError("净值序列为空")
    nav = nav_df["nav"].astype(float)
    if (nav <= 0).any():
        raise RuntimeError("净值出现非正数，结果无效")
    final_nav = float(nav.iloc[-1])
    holding_return = final_nav / INITIAL_CAPITAL - 1.0
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    max_drawdown = float(drawdown.min())
    if max_drawdown < -1.0:
        raise RuntimeError("最大回撤小于-100%，结果无效")
    rets = nav.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    annualized = float((final_nav / INITIAL_CAPITAL) ** (252 / max(len(nav_df), 1)) - 1.0)
    sharpe = float(rets.mean() / (rets.std(ddof=0) + 1e-12) * np.sqrt(252))
    win_rate = float((trades_df["holding_return"] > 0).mean()) if not trades_df.empty else np.nan
    avg_holding_return = float(trades_df["holding_return"].mean()) if not trades_df.empty else np.nan
    avg_holding_days = float(trades_df["holding_days"].mean()) if not trades_df.empty else np.nan
    return {
        "initial_capital": INITIAL_CAPITAL,
        "final_nav": final_nav,
        "holding_return": holding_return,
        "annualized_return": annualized,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trade_count": int(len(trades_df)),
        "win_rate": win_rate,
        "avg_holding_return": avg_holding_return,
        "avg_holding_days": avg_holding_days,
    }


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / f"b1_long_bear_short_volume_account_v1_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)

    signal_df = _load_signal_df()
    trade_candidates_df, close_map, calendar = _build_trade_candidates(signal_df)
    trade_candidates_path = result_dir / "trade_candidates.csv"
    trade_candidates_df.to_csv(trade_candidates_path, index=False)

    summary_rows: list[dict] = []
    all_nav_rows: list[pd.DataFrame] = []
    all_trade_rows: list[pd.DataFrame] = []

    for group_name in GROUPS:
        for hold_days in HOLD_DAYS:
            profile_df = trade_candidates_df[
                (trade_candidates_df["group_name"] == group_name)
                & (trade_candidates_df["hold_days"] == hold_days)
            ].copy()
            if profile_df.empty:
                continue
            nav_df, trades_df, summary = _simulate_profile(profile_df, close_map, calendar)
            profile_name = f"{group_name}__h{hold_days}"
            summary["profile_name"] = profile_name
            summary["group_name"] = group_name
            summary["hold_days"] = hold_days
            summary["signal_count"] = int(len(profile_df))
            summary_rows.append(summary)

            nav_df = nav_df.copy()
            nav_df["profile_name"] = profile_name
            all_nav_rows.append(nav_df)

            trades_df = trades_df.copy()
            trades_df["profile_name"] = profile_name
            all_trade_rows.append(trades_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(["annualized_return", "holding_return"], ascending=[False, False]).reset_index(drop=True)
    nav_all_df = pd.concat(all_nav_rows, ignore_index=True) if all_nav_rows else pd.DataFrame()
    trades_all_df = pd.concat(all_trade_rows, ignore_index=True) if all_trade_rows else pd.DataFrame()

    summary_path = result_dir / "account_summary.csv"
    nav_path = result_dir / "daily_nav.csv"
    trades_path = result_dir / "trades.csv"
    json_path = result_dir / "summary.json"

    summary_df.to_csv(summary_path, index=False)
    nav_all_df.to_csv(nav_path, index=False)
    trades_all_df.to_csv(trades_path, index=False)

    payload = {
        "result_dir": str(result_dir),
        "signal_result_dir": str(SIGNAL_RESULT_DIR),
        "initial_capital": INITIAL_CAPITAL,
        "max_positions": MAX_POSITIONS,
        "lot_size": LOT_SIZE,
        "buy_rule": "signal_date次日开盘买",
        "sell_rule": "固定持有h天到收盘卖",
        "ranking_rule": "同日按j_rank20升序、body_atr降序、vol_ratio升序排序",
        "profiles": summary_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"结果目录: {result_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
