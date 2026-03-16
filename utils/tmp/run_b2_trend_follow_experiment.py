from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import b2filter


DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/b2_trend_follow_experiment_20260313")
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
MAX_SINGLE_WEIGHT = 0.2
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Combo:
    max_hold_days: int
    exit_mode: str

    @property
    def combo_name(self) -> str:
        return f"{self.max_hold_days}天上限__{self.exit_mode}"


def compute_equity_metrics(portfolio_df: pd.DataFrame) -> dict:
    if portfolio_df.empty:
        return {
            "annual_return": np.nan,
            "max_drawdown": np.nan,
            "equity_days": 0,
            "final_equity": np.nan,
        }
    eq = portfolio_df["equity"].astype(float)
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    days = len(eq)
    annual_return = (eq.iloc[-1] / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / days) - 1.0 if days > 0 else np.nan
    return {
        "annual_return": float(annual_return),
        "max_drawdown": float(drawdown.min()),
        "equity_days": int(days),
        "final_equity": float(eq.iloc[-1]),
    }


def max_consecutive_failures(flags: List[bool]) -> int:
    best = 0
    cur = 0
    for flag in flags:
        if flag:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def load_feature_map() -> Dict[str, pd.DataFrame]:
    fmap: Dict[str, pd.DataFrame] = {}
    files = sorted(DATA_DIR.glob("*.txt"))
    total = len(files)
    for idx, fp in enumerate(files, 1):
        df = b2filter.load_one_csv(str(fp))
        if df is None or df.empty:
            continue
        x = b2filter.add_features(df)
        x = x[(x["date"] < EXCLUDE_START) | (x["date"] > EXCLUDE_END)].reset_index(drop=True)
        if len(x) >= b2filter.MIN_BARS:
            fmap[str(x["code"].iloc[0])] = x
        if idx % 500 == 0 or idx == total:
            print(f"特征进度: {idx}/{total}")
    return fmap


def base_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["trend_ok"]
        & df["dual_start"]
        & df["small_upper_shadow"]
        & df["b2_volume_ok"]
        & df["b2_medium_volume"]
        & df["b2_j_ok"]
        & df["near_20d_high"]
        & (df["close_position"] >= b2filter.CLOSE_POSITION_MIN)
        & (df["ret1"] >= b2filter.RET1_MIN)
    )


def _make_trade(df: pd.DataFrame, signal_idx: int, exit_idx: int, exit_price: float, exit_reason: str) -> Optional[dict]:
    entry_idx = signal_idx + 1
    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(exit_price) or exit_price <= 0:
        return None
    ret = exit_price / entry_price - 1.0
    highs = df.iloc[entry_idx:min(len(df), exit_idx + 1)]["high"].astype(float)
    lows = df.iloc[entry_idx:min(len(df), exit_idx + 1)]["low"].astype(float)
    max_runup = float(highs.max() / entry_price - 1.0) if not highs.empty else np.nan
    max_drawdown = float(lows.min() / entry_price - 1.0) if not lows.empty else np.nan
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "first_eligible_exit_date": df.at[entry_idx + 1, "date"] if entry_idx + 1 < len(df) else pd.NaT,
        "exit_date": df.at[exit_idx, "date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "ret": ret,
        "holding_days": int(exit_idx - entry_idx),
        "success": ret > 0,
        "sort_score": float(df.at[signal_idx, "sort_score"]),
        "exit_reason": exit_reason,
        "max_runup": max_runup,
        "max_drawdown_during_trade": max_drawdown,
    }


def simulate_trade(df: pd.DataFrame, signal_idx: int, combo: Combo) -> Optional[dict]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    last_hold_idx = entry_idx + combo.max_hold_days - 1
    if last_hold_idx >= len(df):
        return None

    # 买入当日不能卖，最早从 T+2 开始检查退出
    first_eligible_idx = entry_idx + 1

    below_long_indices: List[int] = []
    for i in range(first_eligible_idx, last_hold_idx + 1):
        close_i = float(df.at[i, "close"])
        trend_i = float(df.at[i, "trend_line"])
        long_i = float(df.at[i, "long_line"])

        trend_break = close_i < trend_i
        if close_i < long_i:
            below_long_indices.append(i)
        else:
            below_long_indices = []

        long_break_confirm = False
        if len(below_long_indices) >= 3:
            watch_start = below_long_indices[-3]
            three_bar_low = float(df.loc[watch_start:below_long_indices[-1], "low"].min())
            if float(df.at[i, "low"]) < three_bar_low:
                long_break_confirm = True

        if combo.exit_mode == "跌破趋势线次日开盘":
            if trend_break and i + 1 < len(df):
                return _make_trade(df, signal_idx, i + 1, float(df.at[i + 1, "open"]), "跌破趋势线")

        elif combo.exit_mode == "三收破多空线再破低次日开盘":
            if long_break_confirm and i + 1 < len(df):
                return _make_trade(df, signal_idx, i + 1, float(df.at[i + 1, "open"]), "三收破多空线再破低")

        elif combo.exit_mode == "趋势线或多空线破位次日开盘":
            if (trend_break or long_break_confirm) and i + 1 < len(df):
                reason = "跌破趋势线" if trend_break else "三收破多空线再破低"
                return _make_trade(df, signal_idx, i + 1, float(df.at[i + 1, "open"]), reason)

    return _make_trade(df, signal_idx, last_hold_idx, float(df.at[last_hold_idx, "close"]), "到期收盘")


def build_portfolio_curve(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["signal_date", "portfolio_ret", "equity"])
    rows = []
    equity = INITIAL_CAPITAL
    tdf = trade_df.copy()
    tdf["signal_date"] = pd.to_datetime(tdf["signal_date"])
    for signal_date, group in tdf.groupby("signal_date", sort=True):
        g = group.sort_values(["sort_score", "code"], ascending=[False, True]).head(MAX_POSITIONS).copy()
        score = g["sort_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        if score.sum() <= 0:
            weights = np.repeat(1 / len(g), len(g))
        else:
            weights = (score / score.sum()).clip(upper=MAX_SINGLE_WEIGHT).to_numpy()
            weights = weights / weights.sum()
        basket_ret = float(np.sum(g["ret"].to_numpy() * weights))
        equity *= 1.0 + basket_ret
        rows.append({"signal_date": signal_date, "portfolio_ret": basket_ret, "equity": equity})
    return pd.DataFrame(rows)


def summarize_combo(combo: Combo, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    row = {
        "sample_count": int(len(trade_df)) if not trade_df.empty else 0,
        "avg_trade_return": float(trade_df["ret"].mean()) if not trade_df.empty else np.nan,
        "success_rate": float(trade_df["success"].mean()) if not trade_df.empty else np.nan,
        "max_consecutive_failures": int(max_consecutive_failures(trade_df["success"].tolist())) if not trade_df.empty else np.nan,
        "avg_max_runup": float(trade_df["max_runup"].mean()) if not trade_df.empty else np.nan,
        "avg_drawdown_during_trade": float(trade_df["max_drawdown_during_trade"].mean()) if not trade_df.empty else np.nan,
    }
    row.update(compute_equity_metrics(portfolio_df))
    row.update(asdict(combo))
    row["combo_name"] = combo.combo_name
    return row


def build_combos() -> List[Combo]:
    combos: List[Combo] = []
    for max_hold_days in [10, 15, 20, 30]:
        for exit_mode in [
            "固定持有到期收盘",
            "跌破趋势线次日开盘",
            "三收破多空线再破低次日开盘",
            "趋势线或多空线破位次日开盘",
        ]:
            combos.append(Combo(max_hold_days=max_hold_days, exit_mode=exit_mode))
    return combos


def validate(result_df: pd.DataFrame) -> None:
    finite_dd = result_df["max_drawdown"].dropna()
    if not finite_dd.empty and ((finite_dd < -1.0) | (finite_dd > 0.0)).any():
        raise ValueError("存在非法最大回撤")
    valid = result_df.dropna(subset=["annual_return", "final_equity"])
    inconsistent = valid[
        ((valid["final_equity"] > INITIAL_CAPITAL) & (valid["annual_return"] <= 0))
        | ((valid["final_equity"] < INITIAL_CAPITAL) & (valid["annual_return"] >= 0))
    ]
    if not inconsistent.empty:
        raise ValueError("年化与净值方向不一致")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fmap = load_feature_map()
    combos = build_combos()
    rows = []
    for idx, combo in enumerate(combos, 1):
        trades = []
        for code, df in fmap.items():
            idxs = np.flatnonzero(base_mask(df).fillna(False).to_numpy())
            for sidx in idxs:
                trade = simulate_trade(df, int(sidx), combo)
                if trade is None:
                    continue
                trade["code"] = code
                trades.append(trade)
        trade_df = pd.DataFrame(trades)
        portfolio_df = build_portfolio_curve(trade_df)
        rows.append(summarize_combo(combo, trade_df, portfolio_df))
        print(f"组合进度: {idx}/{len(combos)}")

    result_df = pd.DataFrame(rows)
    validate(result_df)
    result_df["drawdown_abs"] = result_df["max_drawdown"].abs()
    result_df = result_df.sort_values(
        ["annual_return", "drawdown_abs", "avg_trade_return", "success_rate"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    result_df.to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    best = result_df.iloc[0].to_dict()
    summary = {
        "data_dir": str(DATA_DIR),
        "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "entry_signal": "当前最优B2，仅使用T日已知信息",
        "entry": "T+1_open",
        "note": "买入当日不能卖；结构退出从 T+2 开始检查",
        "combos": [c.combo_name for c in combos],
        "best_combo": best,
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
