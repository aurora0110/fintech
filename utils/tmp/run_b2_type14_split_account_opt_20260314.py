from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tmp.run_b2_type14_exit_and_param_opt import (  # type: ignore
    DATA_DIR,
    EXCLUDE_END,
    EXCLUDE_START,
    TRADING_DAYS_PER_YEAR,
    add_base_features,
    load_one_csv,
)
from utils.tmp.run_b2_type14_param_search_cached import (  # type: ignore
    select_type1,
    select_type4,
)


RESULT_DIR = ROOT / "results/b2_type14_split_account_opt_20260314"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 1_000_000.0

TYPE1_CANDIDATES_PATH = ROOT / "results/b2_type14_param_search_cached_20260313/type1_candidates.csv"
TYPE4_CANDIDATES_PATH = ROOT / "results/b2_type14_param_search_cached_20260313/type4_candidates.csv"


@dataclass(frozen=True)
class ExitRule:
    name: str
    max_hold_days: int
    take_profit: Optional[float] = None
    take_profit_exec: str = "next_open"
    stop_loss_mode: Optional[str] = None  # entry_day_low_099 / signal_low_099 / signal_low
    trend_exit_mode: Optional[str] = None  # close_below_trend / two_closes_below_trend / bearish_vol


@dataclass(frozen=True)
class AccountConfig:
    name: str
    max_positions: int
    daily_new_limit: int
    daily_budget_frac: float
    position_cap_frac: float
    allocation_mode: str  # equal / score_weighted


TYPE1_PARAM_GRID = [
    {
        "ret1_min": ret1_min,
        "upper_shadow_body_ratio": upper_ratio,
        "j_max": j_max,
        "type1_near_ratio": near_ratio,
        "type1_j_rank20_max": jrank,
    }
    for ret1_min in [0.04, 0.05]
    for upper_ratio in [0.3, 0.4]
    for j_max in [90.0, 100.0]
    for near_ratio in [1.00, 1.01, 1.02]
    for jrank in [0.05, 0.10]
]

TYPE4_PARAM_GRID = [
    {
        "ret1_min": ret1_min,
        "upper_shadow_body_ratio": upper_ratio,
        "j_max": j_max,
        "type4_touch_ratio": touch_ratio,
    }
    for ret1_min in [0.03, 0.04]
    for upper_ratio in [0.3, 0.5, 0.8]
    for j_max in [90.0, 100.0]
    for touch_ratio in [1.00, 1.01]
]

TYPE1_EXIT_RULES = [
    ExitRule("hold20_close", 20),
    ExitRule("hold25_close", 25),
    ExitRule("hold30_close", 30),
    ExitRule("tp8_hold30", 30, take_profit=0.08),
    ExitRule("tp10_hold30", 30, take_profit=0.10),
    ExitRule("tp12_hold30", 30, take_profit=0.12),
    ExitRule("tp15_hold30", 30, take_profit=0.15),
    ExitRule("bearish_vol_hold30", 30, trend_exit_mode="bearish_vol"),
    ExitRule("entry_day_low099_hold30", 30, stop_loss_mode="entry_day_low_099"),
    ExitRule("signal_low099_hold30", 30, stop_loss_mode="signal_low_099"),
]

TYPE4_EXIT_RULES = [
    ExitRule("hold15_close", 15),
    ExitRule("hold20_close", 20),
    ExitRule("hold25_close", 25),
    ExitRule("hold30_close", 30),
    ExitRule("tp8_hold30", 30, take_profit=0.08),
    ExitRule("tp10_hold30", 30, take_profit=0.10),
    ExitRule("tp12_hold30", 30, take_profit=0.12),
    ExitRule("bearish_vol_hold30", 30, trend_exit_mode="bearish_vol"),
    ExitRule("entry_day_low099_hold20", 20, stop_loss_mode="entry_day_low_099"),
    ExitRule("signal_low099_hold20", 20, stop_loss_mode="signal_low_099"),
]

ACCOUNT_CONFIGS = [
    AccountConfig(
        name=f"pos{max_positions}_new{daily_new_limit}_b{int(daily_budget_frac*100)}_cap{int(position_cap_frac*100)}_{allocation_mode}",
        max_positions=max_positions,
        daily_new_limit=daily_new_limit,
        daily_budget_frac=daily_budget_frac,
        position_cap_frac=position_cap_frac,
        allocation_mode=allocation_mode,
    )
    for max_positions in [3, 5, 8, 10]
    for daily_new_limit in [1, 2, 3]
    for daily_budget_frac in [0.20, 0.30, 0.50, 1.00]
    for position_cap_frac in [0.10, 0.15, 0.20]
    for allocation_mode in ["equal", "score_weighted"]
]


def _valid_signal_date(d: pd.Timestamp) -> bool:
    return (d < EXCLUDE_START) or (d > EXCLUDE_END)


def load_all_data() -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    paths = sorted(DATA_DIR.glob("*.txt"))
    for idx, path in enumerate(paths, start=1):
        df = load_one_csv(path)
        if df is None:
            continue
        dfs[path.stem] = add_base_features(df)
        if idx % 500 == 0:
            print(f"数据加载进度: {idx}/{len(paths)}")
    return dfs


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signal_date", "entry_date"])
    return df


def _exit_trade(x: pd.DataFrame, signal_idx: int, rule: ExitRule) -> Tuple[int, float, str]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(x):
        return len(x) - 1, float(x.iloc[-1]["close"]), "insufficient_data"

    entry_open = float(x.at[entry_idx, "open"])
    entry_day_low = float(x.at[entry_idx, "low"])
    signal_low = float(x.at[signal_idx, "low"])
    stop_price = None
    if rule.stop_loss_mode == "entry_day_low_099":
        stop_price = entry_day_low * 0.99
    elif rule.stop_loss_mode == "signal_low_099":
        stop_price = signal_low * 0.99
    elif rule.stop_loss_mode == "signal_low":
        stop_price = signal_low

    max_exit_idx = min(entry_idx + rule.max_hold_days, len(x) - 1)
    below_trend_count = 0
    for i in range(entry_idx + 1, max_exit_idx + 1):
        row = x.iloc[i]

        if stop_price is not None and float(row["low"]) <= stop_price:
            return i, stop_price, f"stop_{rule.stop_loss_mode}"

        if rule.take_profit is not None:
            tp_price = entry_open * (1.0 + rule.take_profit)
            if float(row["high"]) >= tp_price:
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, "open"]), f"tp_{rule.take_profit:.2f}"

        if rule.trend_exit_mode == "close_below_trend":
            if float(row["close"]) < float(row["trend_line"]):
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, "open"]), "trend_break"
        elif rule.trend_exit_mode == "two_closes_below_trend":
            if float(row["close"]) < float(row["trend_line"]):
                below_trend_count += 1
            else:
                below_trend_count = 0
            if below_trend_count >= 2:
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, "open"]), "trend_break_2d"
        elif rule.trend_exit_mode == "bearish_vol":
            prev_vol = float(x.iloc[i - 1]["volume"]) if i - 1 >= 0 else np.nan
            if (
                float(row["close"]) < float(row["open"])
                and np.isfinite(prev_vol)
                and float(row["volume"]) >= prev_vol * 1.3
            ):
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, "open"]), "bearish_vol"

    return max_exit_idx, float(x.iloc[max_exit_idx]["close"]), f"hold_{rule.max_hold_days}_close"


def build_trade_table(signals: pd.DataFrame, dfs: Dict[str, pd.DataFrame], rule: ExitRule, tag: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rec in signals.itertuples(index=False):
        x = dfs[rec.code]
        exit_idx, exit_price, reason = _exit_trade(x, int(rec.signal_idx), rule)
        entry_open = float(rec.entry_open)
        ret = exit_price / entry_open - 1.0
        path = x.iloc[int(rec.entry_idx) : exit_idx + 1].copy()
        rows.append(
            {
                "tag": tag,
                "code": rec.code,
                "signal_idx": int(rec.signal_idx),
                "signal_date": rec.signal_date,
                "entry_idx": int(rec.entry_idx),
                "entry_date": rec.entry_date,
                "exit_idx": int(exit_idx),
                "exit_date": x.at[exit_idx, "date"],
                "entry_open": entry_open,
                "exit_price": exit_price,
                "return": ret,
                "reason": reason,
                "sort_score": float(rec.sort_score),
                "max_favorable": float(path["high"].max() / entry_open - 1.0),
                "max_adverse": float(path["low"].min() / entry_open - 1.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "sample_count": 0,
            "success_rate": float("nan"),
            "avg_return": float("nan"),
            "avg_max_favorable": float("nan"),
            "avg_max_adverse": float("nan"),
        }
    wins = (trades["return"] > 0).mean()
    return {
        "sample_count": int(len(trades)),
        "success_rate": float(wins),
        "avg_return": float(trades["return"].mean()),
        "avg_max_favorable": float(trades["max_favorable"].mean()),
        "avg_max_adverse": float(trades["max_adverse"].mean()),
    }


def prepare_close_matrix(dfs: Dict[str, pd.DataFrame], codes: Iterable[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[pd.DatetimeIndex, Dict[str, pd.Series]]:
    relevant = {}
    all_dates = set()
    for code in sorted(set(codes)):
        x = dfs[code][["date", "close"]].copy()
        x = x[(x["date"] >= start_date) & (x["date"] <= end_date)].copy()
        if x.empty:
            continue
        relevant[code] = x.set_index("date")["close"].astype(float)
        all_dates.update(relevant[code].index.tolist())
    market_dates = pd.DatetimeIndex(sorted(all_dates))
    close_map: Dict[str, pd.Series] = {}
    for code, s in relevant.items():
        close_map[code] = s.reindex(market_dates).ffill()
    return market_dates, close_map


def simulate_portfolio(trades: pd.DataFrame, dfs: Dict[str, pd.DataFrame], config: AccountConfig) -> Dict[str, float]:
    if trades.empty:
        return {
            "annual_return": float("nan"),
            "max_drawdown": float("nan"),
            "final_equity": float("nan"),
            "trade_count": 0,
            "success_rate": float("nan"),
            "avg_return": float("nan"),
            "max_losing_streak": 0,
            "equity_days": 0,
        }

    trades = trades.sort_values(["entry_date", "sort_score"], ascending=[True, False]).reset_index(drop=True)
    start_date = pd.Timestamp(trades["entry_date"].min())
    end_date = pd.Timestamp(trades["exit_date"].max())
    market_dates, close_map = prepare_close_matrix(dfs, trades["code"].unique(), start_date, end_date)

    cash = INITIAL_CAPITAL
    positions: Dict[str, Dict[str, float]] = {}
    closed_returns: List[float] = []
    equity_curve: List[Dict[str, object]] = []

    entries_by_date = {
        d: g.sort_values("sort_score", ascending=False).to_dict("records")
        for d, g in trades.groupby("entry_date")
    }
    exits_by_date = {
        d: g.to_dict("records")
        for d, g in trades.groupby("exit_date")
    }

    for current_date in market_dates:
        # exit first
        for tr in exits_by_date.get(current_date, []):
            code = tr["code"]
            if code in positions:
                pos = positions.pop(code)
                proceeds = pos["shares"] * float(tr["exit_price"])
                cash += proceeds
                closed_returns.append(float(tr["exit_price"]) / pos["entry_price"] - 1.0)

        # mark equity before new entries
        equity = cash
        for code, pos in positions.items():
            px = float(close_map[code].get(current_date, pos["entry_price"]))
            equity += pos["shares"] * px

        # enter new positions
        entry_candidates = entries_by_date.get(current_date, [])
        if entry_candidates:
            available_slots = max(config.max_positions - len(positions), 0)
            if available_slots > 0:
                to_add = []
                for tr in entry_candidates:
                    code = tr["code"]
                    if code in positions:
                        continue
                    to_add.append(tr)
                    if len(to_add) >= min(available_slots, config.daily_new_limit):
                        break
                if to_add:
                    investable = min(cash, equity * config.daily_budget_frac)
                    if investable > 0:
                        if config.allocation_mode == "score_weighted":
                            raw_scores = np.array([max(float(tr["sort_score"]), 0.01) for tr in to_add], dtype=float)
                            weights = raw_scores / raw_scores.sum()
                        else:
                            weights = np.full(len(to_add), 1.0 / len(to_add), dtype=float)
                        per_pos_cap = equity * config.position_cap_frac
                        for tr, weight in zip(to_add, weights):
                            alloc = min(investable * float(weight), per_pos_cap, cash)
                            price = float(tr["entry_open"])
                            if alloc <= 0 or price <= 0:
                                continue
                            shares = alloc / price
                            cash -= shares * price
                            positions[tr["code"]] = {"shares": shares, "entry_price": price}

        # mark equity at close
        equity = cash
        for code, pos in positions.items():
            px = float(close_map[code].get(current_date, pos["entry_price"]))
            equity += pos["shares"] * px
        equity_curve.append({"date": current_date, "equity": equity})

    eq = pd.DataFrame(equity_curve).sort_values("date")
    running_max = eq["equity"].cummax()
    drawdown = eq["equity"] / running_max - 1.0
    total_days = max(len(eq), 1)
    annual_return = float((eq.iloc[-1]["equity"] / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / total_days) - 1.0)

    losing_streak = 0
    max_losing_streak = 0
    for r in closed_returns:
        if r <= 0:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0

    return {
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()),
        "final_equity": float(eq.iloc[-1]["equity"]),
        "trade_count": int(len(closed_returns)),
        "success_rate": float(np.mean(np.array(closed_returns) > 0)) if closed_returns else float("nan"),
        "avg_return": float(np.mean(closed_returns)) if closed_returns else float("nan"),
        "max_losing_streak": int(max_losing_streak),
        "equity_days": int(total_days),
    }


def pick_best(df: pd.DataFrame, min_sample_count: int, min_trade_count: int) -> pd.Series:
    cand = df[(df["sample_count"] >= min_sample_count) & (df["trade_count"] >= min_trade_count)].copy()
    if cand.empty:
        cand = df.copy()
    cand["score"] = (
        0.45 * cand["annual_return"].fillna(-9.0)
        + 0.20 * cand["avg_return"].fillna(-9.0)
        + 0.20 * cand["success_rate"].fillna(0.0)
        + 0.10 * cand["avg_trade_return"].fillna(-9.0)
        - 0.05 * cand["max_drawdown"].abs().fillna(1.0)
    )
    cand = cand.sort_values(
        ["score", "annual_return", "success_rate", "avg_return", "sample_count"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return cand.iloc[0]


def top_rows_for_account(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = df.copy()
    df["pre_score"] = (
        0.45 * df["avg_return"].fillna(-9.0)
        + 0.20 * df["success_rate"].fillna(0.0)
        + 0.15 * df["avg_max_favorable"].fillna(-9.0)
        + 0.10 * df["sample_count"].clip(lower=0.0).pow(0.5)
        + 0.10 * df["avg_max_adverse"].fillna(-9.0)
    )
    return df.sort_values(
        ["pre_score", "avg_return", "success_rate", "sample_count"],
        ascending=[False, False, False, False],
    ).head(top_n).reset_index(drop=True)


def params_name(d: Dict[str, float]) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, float):
            if abs(v - round(v)) < 1e-9:
                txt = str(int(round(v)))
            else:
                txt = f"{v:.2f}"
        else:
            txt = str(v)
        parts.append(f"{k}={txt}")
    return "|".join(parts)


def search_type(
    tag: str,
    candidates: pd.DataFrame,
    dfs: Dict[str, pd.DataFrame],
    param_grid: List[Dict[str, float]],
    exit_rules: List[ExitRule],
    min_sample_count: int,
    min_trade_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    trade_rows: List[Dict[str, object]] = []
    trades_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    total = len(param_grid) * len(exit_rules)
    done = 0
    for params in param_grid:
        if tag == "type1":
            signals = select_type1(candidates, params)
        else:
            signals = select_type4(candidates, params)

        for rule in exit_rules:
            done += 1
            trades = build_trade_table(signals, dfs, rule, tag)
            trades_cache[(params_name(params), rule.name)] = trades
            s = summarize_trades(trades)
            row = {
                "tag": tag,
                "params_name": params_name(params),
                "exit_rule": rule.name,
                **params,
                **s,
            }
            trade_rows.append(row)
            if done % 20 == 0 or done == total:
                print(f"{tag} 交易层搜索进度: {done}/{total}")

    trade_df = pd.DataFrame(trade_rows)
    top_trade_df = top_rows_for_account(trade_df, top_n=12)

    account_rows: List[Dict[str, object]] = []
    total_account = len(top_trade_df) * len(ACCOUNT_CONFIGS)
    done = 0
    for rec in top_trade_df.itertuples(index=False):
        trades = trades_cache[(rec.params_name, rec.exit_rule)]
        for config in ACCOUNT_CONFIGS:
            done += 1
            p = simulate_portfolio(trades, dfs, config)
            account_rows.append(
                {
                    "tag": tag,
                    "params_name": rec.params_name,
                    "exit_rule": rec.exit_rule,
                    "account_name": config.name,
                    "max_positions": config.max_positions,
                    "daily_new_limit": config.daily_new_limit,
                    "daily_budget_frac": config.daily_budget_frac,
                    "position_cap_frac": config.position_cap_frac,
                    "allocation_mode": config.allocation_mode,
                    "sample_count": int(rec.sample_count),
                    "trade_success_rate": float(rec.success_rate),
                    "avg_trade_return": float(rec.avg_return),
                    "avg_max_favorable": float(rec.avg_max_favorable),
                    "avg_max_adverse": float(rec.avg_max_adverse),
                    **p,
                }
            )
            if done % 50 == 0 or done == total_account:
                print(f"{tag} 账户层搜索进度: {done}/{total_account}")

    account_df = pd.DataFrame(account_rows)
    best = pick_best(account_df, min_sample_count=min_sample_count, min_trade_count=min_trade_count)
    summary = {
        "best": best.to_dict(),
        "top_trade_candidates": top_trade_df.head(5).to_dict("records"),
    }
    return trade_df, account_df, summary


def parse_params_name(name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in str(name).split("|"):
        key, value = item.split("=")
        out[key] = float(value)
    return out


def update_b2filter(best_type1: pd.Series, best_type4: pd.Series) -> None:
    path = ROOT / "utils/b2filter.py"
    text = path.read_text(encoding="utf-8")

    t1 = parse_params_name(str(best_type1["params_name"]))
    t4 = parse_params_name(str(best_type4["params_name"]))

    replacements = {
        "TYPE1_RET1_MIN = 0.04": f"TYPE1_RET1_MIN = {t1['ret1_min']:.2f}",
        "TYPE1_J_MAX = 90.0": f"TYPE1_J_MAX = {t1['j_max']:.1f}",
        "TYPE1_UPPER_SHADOW_BODY_RATIO = 0.3": f"TYPE1_UPPER_SHADOW_BODY_RATIO = {t1['upper_shadow_body_ratio']:.1f}",
        "TYPE1_START_NEAR_RATIO = 1.01": f"TYPE1_START_NEAR_RATIO = {t1['type1_near_ratio']:.2f}",
        "TYPE1_J_LOW_RANK20_MAX = 0.10": f"TYPE1_J_LOW_RANK20_MAX = {t1['type1_j_rank20_max']:.2f}",
        "TYPE4_RET1_MIN = 0.03": f"TYPE4_RET1_MIN = {t4['ret1_min']:.2f}",
        "TYPE4_J_MAX = 100.0": f"TYPE4_J_MAX = {t4['j_max']:.1f}",
        "TYPE4_UPPER_SHADOW_BODY_RATIO = 0.5": f"TYPE4_UPPER_SHADOW_BODY_RATIO = {t4['upper_shadow_body_ratio']:.1f}",
        "TYPE4_TOUCH_RATIO = 1.00": f"TYPE4_TOUCH_RATIO = {t4['type4_touch_ratio']:.2f}",
    }

    new_text = text
    for old, new in replacements.items():
        if old in new_text:
            new_text = new_text.replace(old, new)
    path.write_text(new_text, encoding="utf-8")


def main() -> None:
    dfs = load_all_data()
    type1_candidates = load_candidates(TYPE1_CANDIDATES_PATH)
    type4_candidates = load_candidates(TYPE4_CANDIDATES_PATH)

    type1_trade_df, type1_account_df, type1_summary = search_type(
        tag="type1",
        candidates=type1_candidates,
        dfs=dfs,
        param_grid=TYPE1_PARAM_GRID,
        exit_rules=TYPE1_EXIT_RULES,
        min_sample_count=900,
        min_trade_count=120,
    )
    type4_trade_df, type4_account_df, type4_summary = search_type(
        tag="type4",
        candidates=type4_candidates,
        dfs=dfs,
        param_grid=TYPE4_PARAM_GRID,
        exit_rules=TYPE4_EXIT_RULES,
        min_sample_count=550,
        min_trade_count=100,
    )

    type1_trade_df.to_csv(RESULT_DIR / "type1_trade_search.csv", index=False)
    type1_account_df.to_csv(RESULT_DIR / "type1_account_search.csv", index=False)
    type4_trade_df.to_csv(RESULT_DIR / "type4_trade_search.csv", index=False)
    type4_account_df.to_csv(RESULT_DIR / "type4_account_search.csv", index=False)

    best1 = pd.Series(type1_summary["best"])
    best4 = pd.Series(type4_summary["best"])
    update_b2filter(best1, best4)

    summary = {
        "type1": type1_summary,
        "type4": type4_summary,
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
