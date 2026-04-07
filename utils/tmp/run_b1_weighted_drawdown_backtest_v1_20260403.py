from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import b1filter
from utils import stoploss, technical_indicators


DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
RESULTS_ROOT = Path("/Users/lidongyang/Desktop/Qstrategy/results")
INITIAL_CAPITAL = 1_000_000.0
TRADING_DAYS_PER_YEAR = 252
FEE_RATE = 0.0003
SLIPPAGE = 0.001
MAX_POSITIONS = 10
MAX_HOLD_DAYS = 120
LOT_SIZE = 100
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

FACTOR_COLUMNS = [
    "weekly_ok",
    "ma_pullback",
    "trend_pullback",
    "first_pullback",
    "low_volume_low_price",
    "sb1",
    "declining_volume",
    "consistent_close_range",
    "recent_max_volume_is_bullish",
    "bullish_volume_dominance",
    "gap_up_followed_by_big_bullish",
    "long_negative_short_volume",
]

WEIGHT_SCHEMES = {
    "equal": {col: 1.0 for col in FACTOR_COLUMNS},
    "structure_first": {
        "weekly_ok": 2.0,
        "ma_pullback": 2.0,
        "trend_pullback": 2.0,
        "first_pullback": 2.0,
        "low_volume_low_price": 1.0,
        "sb1": 1.5,
        "declining_volume": 1.0,
        "consistent_close_range": 1.0,
        "recent_max_volume_is_bullish": 1.0,
        "bullish_volume_dominance": 1.0,
        "gap_up_followed_by_big_bullish": 1.0,
        "long_negative_short_volume": 1.0,
    },
    "volume_first": {
        "weekly_ok": 1.0,
        "ma_pullback": 1.0,
        "trend_pullback": 1.0,
        "first_pullback": 1.0,
        "low_volume_low_price": 2.0,
        "sb1": 1.0,
        "declining_volume": 1.5,
        "consistent_close_range": 1.0,
        "recent_max_volume_is_bullish": 2.0,
        "bullish_volume_dominance": 2.0,
        "gap_up_followed_by_big_bullish": 1.0,
        "long_negative_short_volume": 1.0,
    },
}

ALLOCATION_MODES = ["equal", "score_prop"]
DRAWDOWN_PCTS = [0.07, 0.08, 0.09]


@dataclass(frozen=True)
class Combo:
    weight_scheme: str
    allocation_mode: str
    drawdown_pct: float

    @property
    def combo_name(self) -> str:
        return (
            f"{self.weight_scheme}"
            f"__{self.allocation_mode}"
            f"__dd{int(self.drawdown_pct * 1000):03d}"
        )


def build_output_dir(mode: str, output_dir: Optional[str]) -> Path:
    if output_dir:
        return Path(output_dir)
    return RESULTS_ROOT / f"b1_weighted_drawdown_backtest_v1_{mode}_20260403_r1"


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.shape(a), np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def calc_max_drawdown(equity_arr: np.ndarray) -> float:
    if len(equity_arr) == 0:
        return np.nan
    running_max = np.maximum.accumulate(equity_arr)
    dd = equity_arr / running_max - 1.0
    return float(dd.min())


def calc_sharpe(daily_ret: np.ndarray) -> float:
    if len(daily_ret) <= 1:
        return np.nan
    std = np.std(daily_ret, ddof=1)
    if std <= 1e-12:
        return np.nan
    return float(np.mean(daily_ret) / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def calc_cagr(final_equity: float, n_days: int) -> float:
    if n_days <= 0 or final_equity <= 0:
        return np.nan
    years = n_days / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return np.nan
    return float((final_equity / INITIAL_CAPITAL) ** (1.0 / years) - 1.0)


def get_date_loc(bundle: dict, current_date: pd.Timestamp) -> int:
    return bundle["date_to_loc"].get(current_date, -1)


def get_mark_price(bundle: dict, current_date: pd.Timestamp) -> float:
    dates = bundle["dates"]
    loc = dates.searchsorted(pd.Timestamp(current_date), side="right") - 1
    if loc < 0:
        return np.nan
    return float(bundle["close_arr"][loc])


def load_one_stock(file_path: Path) -> Optional[dict]:
    df, load_error = stoploss.load_data(str(file_path))
    if load_error or df is None or len(df) < 160:
        return None

    df = df[
        (pd.to_numeric(df["开盘"], errors="coerce") > 0)
        & (pd.to_numeric(df["最高"], errors="coerce") > 0)
        & (pd.to_numeric(df["最低"], errors="coerce") > 0)
        & (pd.to_numeric(df["收盘"], errors="coerce") > 0)
        & (pd.to_numeric(df["成交量"], errors="coerce") >= 0)
    ].copy()
    df = df[(df["日期"] < EXCLUDE_START) | (df["日期"] > EXCLUDE_END)].copy()
    if len(df) < 160:
        return None

    df = technical_indicators.calculate_trend(df.copy())
    df_kdj = technical_indicators.calculate_kdj(df.copy())
    df_ma = technical_indicators.calculate_daily_ma(df.copy())
    weekly_map = b1filter.map_weekly_screen_to_daily_df(df)
    df = df.merge(weekly_map, on="日期", how="left")
    df["weekly_ok"] = df["weekly_ok"].fillna(False).astype(bool)
    df = df.reset_index(drop=True)

    return {
        "code": file_path.stem,
        "df": df,
        "df_kdj": df_kdj.reset_index(drop=True),
        "df_ma": df_ma.reset_index(drop=True),
        "dates": pd.DatetimeIndex(df["日期"]),
        "date_to_loc": {pd.Timestamp(d): int(i) for i, d in enumerate(df["日期"])},
        "close_arr": df["收盘"].astype(float).to_numpy(),
    }


def compute_signal_rows(bundle: dict, window_start: Optional[pd.Timestamp]) -> List[dict]:
    df = bundle["df"]
    df_kdj = bundle["df_kdj"]
    df_ma = bundle["df_ma"]
    signal_rows: List[dict] = []

    trend_line = df["知行短期趋势线"].astype(float)
    long_line = df["知行多空线"].astype(float)
    j_values = df_kdj["J"].astype(float)

    for idx in range(1, len(df) - 1):
        if not (pd.notna(trend_line.iloc[idx]) and pd.notna(long_line.iloc[idx]) and trend_line.iloc[idx] > long_line.iloc[idx]):
            continue
        if not (pd.notna(j_values.iloc[idx]) and float(j_values.iloc[idx]) < -5.0):
            continue

        today_row = df.iloc[idx]
        yesterday_row = df.iloc[idx - 1]
        df_slice = df.iloc[: idx + 1]
        trend_slice = df.iloc[: idx + 1]
        ma_slice = df_ma.iloc[: idx + 1]
        kdj_slice = df_kdj.iloc[: idx + 1]

        ma_pullback, trend_pullback = b1filter._pullback_to_key_lines(df_slice, trend_slice, ma_slice)
        signal_rows.append(
            {
                "code": bundle["code"],
                "signal_idx": idx,
                "signal_date": pd.Timestamp(today_row["日期"]),
                "entry_date": pd.Timestamp(df.iloc[idx + 1]["日期"]),
                "signal_low": float(today_row["最低"]),
                "j_value": float(j_values.iloc[idx]),
                "trend_diff": float(trend_line.iloc[idx] - long_line.iloc[idx]),
                "weekly_ok": bool(today_row["weekly_ok"]),
                "ma_pullback": bool(ma_pullback),
                "trend_pullback": bool(trend_pullback),
                "first_pullback": bool(b1filter._first_pullback_after_cross(df_slice, trend_slice)),
                "low_volume_low_price": bool(b1filter._is_low_volume_low_price(df_slice)),
                "sb1": bool(b1filter._is_sb1(today_row, yesterday_row, kdj_slice)),
                "declining_volume": bool(float(today_row["成交量"]) < float(yesterday_row["成交量"])),
                "consistent_close_range": bool(
                    b1filter._within_pct_range(
                        float(today_row["收盘"]),
                        float(yesterday_row["收盘"]),
                        b1filter.PRICE_RANGE_DOWN,
                        b1filter.PRICE_RANGE_UP,
                    )
                ),
                "recent_max_volume_is_bullish": bool(b1filter._recent_max_volume_is_bullish(df_slice)),
                "bullish_volume_dominance": bool(b1filter._bullish_volume_dominance(df_slice)),
                "gap_up_followed_by_big_bullish": bool(b1filter._has_gap_up_followed_by_big_bullish(df_slice, trend_slice)),
                "long_negative_short_volume": bool(b1filter._has_long_negative_short_volume(df_slice)),
            }
        )

    if window_start is None:
        return signal_rows
    return [row for row in signal_rows if row["signal_date"] >= window_start and row["entry_date"] >= window_start]


def score_candidates(signal_df: pd.DataFrame) -> pd.DataFrame:
    x = signal_df.copy()
    for name, weights in WEIGHT_SCHEMES.items():
        score = np.zeros(len(x), dtype=float)
        weight_sum = float(sum(weights.values()))
        for col, weight in weights.items():
            score += x[col].astype(int).to_numpy() * weight
        x[f"score_{name}"] = score / weight_sum if weight_sum > 0 else score
    return x


def load_signal_df(
    data_dir: Path,
    max_files: int,
    lookback_years: Optional[int],
) -> tuple[pd.DataFrame, Dict[str, dict], List[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    files = sorted(data_dir.glob("*.txt"))
    if max_files > 0:
        files = files[:max_files]

    stock_map: Dict[str, dict] = {}
    signal_rows: List[dict] = []
    all_dates: set[pd.Timestamp] = set()

    total = len(files)
    for idx, file_path in enumerate(files, 1):
        bundle = load_one_stock(file_path)
        if bundle is not None:
            stock_map[bundle["code"]] = bundle
            all_dates.update(pd.Timestamp(d) for d in bundle["df"]["日期"])
        if idx % 200 == 0 or idx == total:
            print(f"特征/信号进度: {idx}/{total}")

    if not all_dates:
        return pd.DataFrame(), stock_map, [], None, None

    all_dates_sorted = sorted(all_dates)
    window_end = all_dates_sorted[-1]
    window_start = None
    if lookback_years and lookback_years > 0:
        window_start = window_end - pd.DateOffset(years=lookback_years)

    for bundle in stock_map.values():
        signal_rows.extend(compute_signal_rows(bundle, window_start))

    signal_df = pd.DataFrame(signal_rows)
    if signal_df.empty:
        if window_start is None:
            backtest_dates = all_dates_sorted
        else:
            backtest_dates = [d for d in all_dates_sorted if d >= window_start]
        return signal_df, stock_map, backtest_dates, window_start, window_end
    signal_df = score_candidates(signal_df)
    signal_df = signal_df.sort_values(["entry_date", "signal_date", "code"]).reset_index(drop=True)
    if window_start is None:
        backtest_dates = all_dates_sorted
    else:
        backtest_dates = [d for d in all_dates_sorted if d >= window_start]
    return signal_df, stock_map, backtest_dates, window_start, window_end


def get_next_stock_date(bundle: dict, loc: int) -> Optional[pd.Timestamp]:
    if loc + 1 >= len(bundle["df"]):
        return None
    return pd.Timestamp(bundle["df"].iloc[loc + 1]["日期"])


def execute_close_exit(cash: float, pos: dict, close_price: float) -> tuple[float, dict]:
    proceeds = pos["shares"] * close_price * (1 - FEE_RATE - SLIPPAGE)
    cash += proceeds
    trade = {
        "code": pos["code"],
        "signal_date": pos["signal_date"],
        "entry_date": pos["entry_date"],
        "exit_date": pos["current_date"],
        "entry_price": pos["entry_price"],
        "exit_price": close_price,
        "shares": pos["shares"],
        "pnl_pct": close_price / pos["entry_price"] - 1.0,
        "reason": "max_hold_close",
        "hold_days": pos["hold_days"],
        "weight_scheme": pos["weight_scheme"],
        "allocation_mode": pos["allocation_mode"],
        "drawdown_pct": pos["drawdown_pct"],
    }
    return cash, trade


def simulate_account(
    combo: Combo,
    signal_df: pd.DataFrame,
    stock_map: Dict[str, dict],
    all_dates: List[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    score_col = f"score_{combo.weight_scheme}"
    entry_buckets = {
        pd.Timestamp(dt): grp.sort_values([score_col, "code"], ascending=[False, True]).copy()
        for dt, grp in signal_df.groupby("entry_date", sort=True)
    }

    cash = INITIAL_CAPITAL
    positions: Dict[str, dict] = {}
    trades: List[dict] = []
    equity_rows: List[dict] = []

    for current_date in all_dates:
        current_date = pd.Timestamp(current_date)

        # 1) 次日开盘执行的卖出
        to_remove = []
        for code, pos in list(positions.items()):
            if pos.get("scheduled_exit_date") != current_date:
                continue
            bundle = stock_map[code]
            loc = get_date_loc(bundle, current_date)
            if loc < 0:
                continue
            open_price = float(bundle["df"].iloc[loc]["开盘"])
            proceeds = pos["shares"] * open_price * (1 - FEE_RATE - SLIPPAGE)
            cash += proceeds
            trades.append(
                {
                    "code": code,
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "entry_price": pos["entry_price"],
                    "exit_price": open_price,
                    "shares": pos["shares"],
                    "pnl_pct": open_price / pos["entry_price"] - 1.0,
                    "reason": pos["scheduled_exit_reason"],
                    "hold_days": pos["hold_days"],
                    "weight_scheme": pos["weight_scheme"],
                    "allocation_mode": pos["allocation_mode"],
                    "drawdown_pct": pos["drawdown_pct"],
                }
            )
            to_remove.append(code)
        for code in to_remove:
            positions.pop(code, None)

        # 2) 次日开盘执行的买入
        if current_date in entry_buckets and len(positions) < MAX_POSITIONS:
            bucket = entry_buckets[current_date]
            bucket = bucket[~bucket["code"].isin(positions.keys())].copy()
            if not bucket.empty:
                available_slots = MAX_POSITIONS - len(positions)
                selected = bucket.head(available_slots).copy()
                raw_score = selected[score_col].astype(float).clip(lower=0.0)
                if combo.allocation_mode == "score_prop" and raw_score.sum() > 1e-12:
                    weights = raw_score / raw_score.sum()
                else:
                    weights = pd.Series(np.repeat(1.0 / len(selected), len(selected)), index=selected.index)

                for row_idx, row in selected.iterrows():
                    bundle = stock_map[row["code"]]
                    loc = get_date_loc(bundle, current_date)
                    if loc < 0:
                        continue
                    open_price = float(bundle["df"].iloc[loc]["开盘"])
                    high_price = float(bundle["df"].iloc[loc]["最高"])
                    if not np.isfinite(open_price) or open_price <= 0:
                        continue
                    alloc_cash = cash * float(weights.loc[row_idx])
                    shares = int(alloc_cash / (open_price * (1 + FEE_RATE + SLIPPAGE)) / LOT_SIZE) * LOT_SIZE
                    if shares <= 0:
                        continue
                    cost = shares * open_price * (1 + FEE_RATE + SLIPPAGE)
                    if cost > cash:
                        shares = int(cash / (open_price * (1 + FEE_RATE + SLIPPAGE)) / LOT_SIZE) * LOT_SIZE
                        if shares <= 0:
                            continue
                        cost = shares * open_price * (1 + FEE_RATE + SLIPPAGE)
                    cash -= cost
                    positions[row["code"]] = {
                        "code": row["code"],
                        "signal_date": pd.Timestamp(row["signal_date"]),
                        "entry_date": current_date,
                        "entry_price": open_price,
                        "shares": shares,
                        "signal_low": float(row["signal_low"]),
                        "stop_price": float(row["signal_low"]) * 0.95,
                        "entry_loc": loc,
                        "rolling_high": high_price,
                        "scheduled_exit_date": None,
                        "scheduled_exit_reason": "",
                        "weight_scheme": combo.weight_scheme,
                        "allocation_mode": combo.allocation_mode,
                        "drawdown_pct": combo.drawdown_pct,
                        "hold_days": 1,
                    }

        # 3) 当日盘后更新止盈/止损/到期
        close_exit_codes: List[str] = []
        for code, pos in list(positions.items()):
            bundle = stock_map[code]
            loc = get_date_loc(bundle, current_date)
            if loc < 0:
                continue
            row = bundle["df"].iloc[loc]
            high_price = float(row["最高"])
            low_price = float(row["最低"])
            close_price = float(row["收盘"])
            pos["rolling_high"] = max(pos["rolling_high"], high_price)
            pos["hold_days"] = loc - pos["entry_loc"] + 1

            # 买入当天不能卖出，但仍更新滚动最高价
            if current_date == pos["entry_date"]:
                continue

            if pos["hold_days"] >= MAX_HOLD_DAYS:
                pos["current_date"] = current_date
                cash, trade = execute_close_exit(cash, pos, close_price)
                trades.append(trade)
                close_exit_codes.append(code)
                continue

            if pos["scheduled_exit_date"] is not None:
                continue

            next_date = get_next_stock_date(bundle, loc)
            if next_date is None:
                continue

            stop_hit = pd.notna(low_price) and low_price <= pos["stop_price"]
            dd_hit = (
                pd.notna(low_price)
                and pd.notna(pos["rolling_high"])
                and low_price <= pos["rolling_high"] * (1.0 - combo.drawdown_pct)
            )

            # 同日同时触发时，硬止损优先于回撤止盈
            if stop_hit:
                pos["scheduled_exit_date"] = next_date
                pos["scheduled_exit_reason"] = "stop_loss_next_open"
            elif dd_hit:
                pos["scheduled_exit_date"] = next_date
                pos["scheduled_exit_reason"] = f"drawdown_{int(combo.drawdown_pct * 100)}pct_next_open"

        for code in close_exit_codes:
            positions.pop(code, None)

        # 4) 记录净值
        position_value = 0.0
        for code, pos in positions.items():
            mark_price = get_mark_price(stock_map[code], current_date)
            if np.isfinite(mark_price):
                position_value += pos["shares"] * mark_price
        equity_rows.append(
            {
                "date": current_date,
                "cash": cash,
                "position_value": position_value,
                "equity": cash + position_value,
                "open_positions": len(positions),
                "combo_name": combo.combo_name,
            }
        )

    trade_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    if equity_df.empty:
        summary = {
            "combo_name": combo.combo_name,
            "weight_scheme": combo.weight_scheme,
            "allocation_mode": combo.allocation_mode,
            "drawdown_pct": combo.drawdown_pct,
            "trade_count": 0,
            "win_rate": np.nan,
            "avg_trade_return": np.nan,
            "final_equity": np.nan,
            "holding_return": np.nan,
            "annual_return": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
        }
        return trade_df, equity_df, summary

    daily_ret = equity_df["equity"].pct_change().fillna(0.0).to_numpy(dtype=float)
    final_equity = float(equity_df["equity"].iloc[-1])
    summary = {
        "combo_name": combo.combo_name,
        "weight_scheme": combo.weight_scheme,
        "allocation_mode": combo.allocation_mode,
        "drawdown_pct": combo.drawdown_pct,
        "trade_count": int(len(trade_df)),
        "win_rate": float((trade_df["pnl_pct"] > 0).mean()) if not trade_df.empty else np.nan,
        "avg_trade_return": float(trade_df["pnl_pct"].mean()) if not trade_df.empty else np.nan,
        "final_equity": final_equity,
        "holding_return": float(final_equity / INITIAL_CAPITAL - 1.0),
        "annual_return": calc_cagr(final_equity, len(equity_df)),
        "max_drawdown": calc_max_drawdown(equity_df["equity"].to_numpy(dtype=float)),
        "sharpe": calc_sharpe(daily_ret),
    }
    return trade_df, equity_df, summary


def run(mode: str, data_dir: Path, output_dir: Path, max_files: int, lookback_years: Optional[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_df, stock_map, all_dates, window_start, window_end = load_signal_df(
        data_dir,
        max_files=max_files,
        lookback_years=lookback_years,
    )
    signal_path = output_dir / "signal_candidates.csv"
    signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")

    combos = [Combo(weight_scheme=w, allocation_mode=a, drawdown_pct=d) for w in WEIGHT_SCHEMES for a in ALLOCATION_MODES for d in DRAWDOWN_PCTS]
    summary_rows = []
    all_trades = []

    print(f"信号数: {len(signal_df)}")
    print(f"组合数: {len(combos)}")

    for idx, combo in enumerate(combos, 1):
        trade_df, equity_df, summary = simulate_account(combo, signal_df, stock_map, all_dates)
        summary_rows.append(summary)

        combo_dir = output_dir / combo.combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)
        trade_df.to_csv(combo_dir / "trades.csv", index=False, encoding="utf-8-sig")
        equity_df.to_csv(combo_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        with open(combo_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if not trade_df.empty:
            all_trades.append(trade_df.assign(combo_name=combo.combo_name))

        print(f"组合进度: {idx}/{len(combos)} -> {combo.combo_name}")

    account_df = pd.DataFrame(summary_rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    account_df.to_csv(output_dir / "account_summary.csv", index=False, encoding="utf-8-sig")

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(output_dir / "all_trades.csv", index=False, encoding="utf-8-sig")

    sanity_invalid = account_df[
        ((account_df["final_equity"] > INITIAL_CAPITAL) & (account_df["annual_return"] <= 0))
        | ((account_df["final_equity"] < INITIAL_CAPITAL) & (account_df["annual_return"] >= 0))
        | (account_df["final_equity"] <= 0)
        | (account_df["max_drawdown"] < -1.0)
    ]
    sanity_invalid.to_csv(output_dir / "sanity_invalid_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "mode": mode,
        "data_dir": str(data_dir),
        "max_files": max_files,
        "lookback_years": lookback_years,
        "window_start": None if window_start is None else str(pd.Timestamp(window_start).date()),
        "window_end": None if window_end is None else str(pd.Timestamp(window_end).date()),
        "signal_count": int(len(signal_df)),
        "combo_count": int(len(combos)),
        "best_account_row": account_df.iloc[0].to_dict() if not account_df.empty else None,
        "sanity_invalid_count": int(len(sanity_invalid)),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 因子加权 + 最高点回撤止盈回测")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--lookback-years", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args.mode, args.output_dir)
    max_files = args.max_files if args.max_files is not None else (200 if args.mode == "smoke" else 0)
    run(
        mode=args.mode,
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        max_files=max_files,
        lookback_years=args.lookback_years,
    )


if __name__ == "__main__":
    main()
