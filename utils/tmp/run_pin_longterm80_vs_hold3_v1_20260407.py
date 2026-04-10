from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b1filter, pinfilter, stoploss, technical_indicators


DATA_DIR = ROOT / "data" / "20260402"
RESULTS_DIR = ROOT / "results"
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
LOT_SIZE = 100
TRADING_DAYS_PER_YEAR = 252
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")


@dataclass(frozen=True)
class ExitProfile:
    name: str
    description: str


EXIT_PROFILES = (
    ExitProfile("hold3", "固定持有3天，到第3个持有交易日收盘卖"),
    ExitProfile("long_term_below_80", "长期线跌破80，次日开盘卖"),
)


def _stock_code(file_path: Path) -> str:
    return file_path.stem.split("#")[-1]


def _prepare_df(file_path: str) -> pd.DataFrame | None:
    df, load_error = stoploss.load_data(file_path)
    if load_error or df is None or len(df) < 40:
        return None
    df = df[(df["日期"] < EXCLUDE_START) | (df["日期"] > EXCLUDE_END)].copy()
    if len(df) < 40:
        return None
    df = technical_indicators.calculate_trend(df.copy())
    df = technical_indicators.calculate_kdj(df)
    return df.reset_index(drop=True)


def _calc_short_long(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n1, n2 = 3, 21
    llv_l_n1 = out["最低"].rolling(window=n1).min()
    hhv_c_n1 = out["收盘"].rolling(window=n1).max()
    out["短期"] = (out["收盘"] - llv_l_n1) / (hhv_c_n1 - llv_l_n1) * 100
    llv_l_n2 = out["最低"].rolling(window=n2).min()
    hhv_l_n2 = out["收盘"].rolling(window=n2).max()
    out["长期"] = (out["收盘"] - llv_l_n2) / (hhv_l_n2 - llv_l_n2) * 100
    return out


def _build_signal_rows(file_path_str: str) -> list[dict]:
    file_path = Path(file_path_str)
    df = _prepare_df(str(file_path))
    if df is None or df.empty:
        return []

    weekly_map = b1filter.map_weekly_screen_to_daily_df(df)
    df = df.merge(weekly_map, on="日期", how="left")
    df["weekly_ok"] = df["weekly_ok"].fillna(False).astype(bool)
    df = _calc_short_long(df)
    vol_ma5 = pd.to_numeric(df["成交量"], errors="coerce").rolling(5, min_periods=5).mean()
    code = _stock_code(file_path)

    rows: list[dict] = []
    for i in range(len(df) - 1):
        today = df.iloc[i]
        if not bool(today["weekly_ok"]):
            continue
        trend_line = float(today["知行短期趋势线"]) if pd.notna(today["知行短期趋势线"]) else np.nan
        long_line = float(today["知行多空线"]) if pd.notna(today["知行多空线"]) else np.nan
        short_v = float(today["短期"]) if pd.notna(today["短期"]) else np.nan
        long_v = float(today["长期"]) if pd.notna(today["长期"]) else np.nan
        today_vol = float(today["成交量"]) if pd.notna(today["成交量"]) else np.nan
        vol_ma5_i = float(vol_ma5.iloc[i]) if pd.notna(vol_ma5.iloc[i]) else np.nan

        if not np.isfinite(trend_line) or not np.isfinite(long_line) or trend_line <= long_line:
            continue
        if not np.isfinite(short_v) or not np.isfinite(long_v):
            continue
        if not (short_v <= 30 and long_v >= 85):
            continue
        if not np.isfinite(today_vol) or not np.isfinite(vol_ma5_i) or not (today_vol < vol_ma5_i):
            continue

        entry_idx = i + 1
        entry_date = pd.Timestamp(df["日期"].iloc[entry_idx])
        entry_open = float(df["开盘"].iloc[entry_idx])
        if not np.isfinite(entry_open) or entry_open <= 0:
            continue

        rows.append(
            {
                "code": code,
                "signal_date": pd.Timestamp(today["日期"]),
                "entry_date": entry_date,
                "entry_idx": entry_idx,
                "entry_open": entry_open,
                "short_value": short_v,
                "long_value": long_v,
            }
        )
    return rows


def _load_price_bundle(code: str) -> pd.DataFrame:
    paths = list(DATA_DIR.glob(f"*#{code}.txt"))
    if not paths:
        return pd.DataFrame()
    df = _prepare_df(str(paths[0]))
    if df is None or df.empty:
        return pd.DataFrame()
    return _calc_short_long(df)


def _exit_for_profile(price_df: pd.DataFrame, entry_idx: int, profile_name: str) -> tuple[int, float, str] | None:
    if profile_name == "hold3":
        exit_idx = entry_idx + 3 - 1
        if exit_idx >= len(price_df):
            return None
        exit_price = float(price_df["收盘"].iloc[exit_idx])
        return exit_idx, exit_price, "hold3_close"

    if profile_name == "long_term_below_80":
        for i in range(entry_idx + 1, len(price_df)):
            long_v = float(price_df["长期"].iloc[i]) if pd.notna(price_df["长期"].iloc[i]) else np.nan
            if np.isfinite(long_v) and long_v < 80:
                next_idx = i + 1
                if next_idx >= len(price_df):
                    return None
                exit_price = float(price_df["开盘"].iloc[next_idx])
                if not np.isfinite(exit_price) or exit_price <= 0:
                    return None
                return next_idx, exit_price, "long_term_below_80_next_open"
        return None

    raise ValueError(profile_name)


def _build_trade_candidates(signal_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DatetimeIndex]:
    rows: list[dict] = []
    bundle_map: dict[str, pd.DataFrame] = {}
    all_dates: list[pd.Timestamp] = []

    for code, sub in signal_df.groupby("code"):
        price_df = _load_price_bundle(code)
        if price_df.empty:
            continue
        price_df = price_df.sort_values("日期").reset_index(drop=True)
        bundle_map[code] = price_df
        all_dates.extend(pd.to_datetime(price_df["日期"]).tolist())

        date_to_idx = {pd.Timestamp(d): idx for idx, d in enumerate(price_df["日期"])}
        for rec in sub.itertuples(index=False):
            signal_idx = date_to_idx.get(pd.Timestamp(rec.signal_date))
            if signal_idx is None:
                continue
            entry_idx = date_to_idx.get(pd.Timestamp(rec.entry_date))
            if entry_idx is None:
                continue
            for profile in EXIT_PROFILES:
                exit_info = _exit_for_profile(price_df, entry_idx, profile.name)
                if exit_info is None:
                    continue
                exit_idx, exit_price, exit_reason = exit_info
                rows.append(
                    {
                        "profile_name": profile.name,
                        "code": code,
                        "signal_date": pd.Timestamp(rec.signal_date),
                        "entry_date": pd.Timestamp(rec.entry_date),
                        "entry_open": float(rec.entry_open),
                        "exit_date": pd.Timestamp(price_df["日期"].iloc[exit_idx]),
                        "exit_price": float(exit_price),
                        "exit_reason": exit_reason,
                        "short_value": float(rec.short_value),
                        "long_value": float(rec.long_value),
                    }
                )

    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        raise RuntimeError("trade candidates 为空")
    candidate_df = candidate_df.sort_values(
        ["profile_name", "entry_date", "long_value", "short_value", "code"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    calendar = pd.DatetimeIndex(sorted(pd.unique(pd.Series(all_dates).dropna())))
    return candidate_df, bundle_map, calendar


def _mark_to_market(positions: dict[str, dict], current_date: pd.Timestamp, bundle_map: dict[str, pd.DataFrame]) -> float:
    total = 0.0
    for code, pos in positions.items():
        df = bundle_map[code]
        hit = df[df["日期"] == current_date]
        if hit.empty:
            continue
        px = float(hit["收盘"].iloc[0])
        if np.isfinite(px):
            total += pos["shares"] * px
    return total


def _summarize(nav_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    nav = nav_df["nav"].astype(float)
    if nav.empty or (nav <= 0).any():
        raise RuntimeError("净值序列无效")
    peak = nav.cummax()
    dd = nav / peak - 1.0
    max_drawdown = float(dd.min())
    if max_drawdown < -1.0:
        raise RuntimeError("最大回撤小于-100%")
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


def _simulate_profile(profile_df: pd.DataFrame, bundle_map: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cash = INITIAL_CAPITAL
    positions: dict[str, dict] = {}
    nav_rows: list[dict] = []
    trade_rows: list[dict] = []
    entries = {d: g.copy() for d, g in profile_df.groupby("entry_date")}

    for current_date in calendar:
        exit_codes = [code for code, pos in positions.items() if pos["exit_date"] == current_date]
        for code in exit_codes:
            pos = positions.pop(code)
            cash += pos["shares"] * pos["exit_price"]
            trade_rows.append(
                {
                    "code": code,
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "entry_price": pos["entry_price"],
                    "exit_price": pos["exit_price"],
                    "shares": pos["shares"],
                    "holding_return": pos["exit_price"] / pos["entry_price"] - 1.0,
                    "holding_days": int((current_date - pos["entry_date"]).days) + 1,
                    "exit_reason": pos["exit_reason"],
                }
            )

        todays = entries.get(current_date)
        if todays is not None and not todays.empty:
            todays = todays[~todays["code"].isin(positions.keys())].copy()
            slots = MAX_POSITIONS - len(positions)
            if slots > 0 and not todays.empty:
                todays = todays.sort_values(["long_value", "short_value", "code"], ascending=[False, True, True]).head(slots)
                allocation = cash / len(todays) if len(todays) else 0.0
                for rec in todays.itertuples(index=False):
                    shares = int(math.floor(allocation / rec.entry_open / LOT_SIZE) * LOT_SIZE)
                    if shares < LOT_SIZE:
                        continue
                    cost = shares * rec.entry_open
                    if cost > cash + 1e-9:
                        continue
                    cash -= cost
                    positions[rec.code] = {
                        "signal_date": pd.Timestamp(rec.signal_date),
                        "entry_date": pd.Timestamp(rec.entry_date),
                        "entry_price": float(rec.entry_open),
                        "shares": shares,
                        "exit_date": pd.Timestamp(rec.exit_date),
                        "exit_price": float(rec.exit_price),
                        "exit_reason": rec.exit_reason,
                    }

        market_value = _mark_to_market(positions, current_date, bundle_map)
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
    result_dir = RESULTS_DIR / f"pin_longterm80_vs_hold3_v1_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)

    file_paths = sorted(str(p) for p in DATA_DIR.glob("*.txt"))
    max_workers = min(10, max(1, cpu_count() - 1))
    all_rows: list[dict] = []
    with Pool(processes=max_workers) as pool:
        for rows in pool.imap_unordered(_build_signal_rows, file_paths, chunksize=8):
            if rows:
                all_rows.extend(rows)

    signal_df = pd.DataFrame(all_rows)
    if signal_df.empty:
        raise RuntimeError("当前 pinfilter 没有产生信号")
    signal_df = signal_df.sort_values(["signal_date", "long_value", "short_value", "code"], ascending=[True, False, True, True]).reset_index(drop=True)
    signal_df.to_csv(result_dir / "signal_detail.csv", index=False)

    candidate_df, bundle_map, calendar = _build_trade_candidates(signal_df)
    candidate_df.to_csv(result_dir / "trade_candidates.csv", index=False)

    summary_rows: list[dict] = []
    nav_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    for profile in EXIT_PROFILES:
        profile_df = candidate_df[candidate_df["profile_name"] == profile.name].copy()
        nav_df, trades_df, summary = _simulate_profile(profile_df, bundle_map, calendar)
        summary["profile_name"] = profile.name
        summary["description"] = profile.description
        summary["signal_count"] = int(signal_df["signal_date"].count())
        summary_rows.append(summary)
        nav_df["profile_name"] = profile.name
        trades_df["profile_name"] = profile.name
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
        "data_dir": str(DATA_DIR),
        "signal_definition": "当前 pinfilter.py",
        "sorting_rule": "同日按长期降序，再按短期升序",
        "profiles": account_df.to_dict(orient="records"),
    }
    (result_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"结果目录: {result_dir}")
    print(account_df.to_string(index=False))


if __name__ == "__main__":
    main()
