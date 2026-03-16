from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import b2filter


DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/b2_trendline_exit_with_n_rank_experiment_20260313")
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
MAX_SINGLE_WEIGHT = 0.2
TRADING_DAYS_PER_YEAR = 252
N_LOOKBACK = 80


@dataclass(frozen=True)
class Combo:
    max_hold_days: int
    exit_mode: str
    use_n_up: bool
    n_rank_window: int
    n_rank_threshold: float

    @property
    def combo_name(self) -> str:
        return (
            f"{self.max_hold_days}天上限__{self.exit_mode}__N向上_{self.use_n_up}"
            f"__J分位窗口_{self.n_rank_window}__J分位阈值_{self.n_rank_threshold:.2f}"
        )


def compute_equity_metrics(portfolio_df: pd.DataFrame) -> dict:
    if portfolio_df.empty:
        return {"annual_return": np.nan, "max_drawdown": np.nan, "equity_days": 0, "final_equity": np.nan}
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


def identify_low_zones(mask_series: pd.Series) -> List[Tuple[int, int]]:
    mask = mask_series.fillna(False).to_numpy(dtype=bool)
    zones: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            zones.append((start, i - 1))
            start = None
    if start is not None:
        zones.append((start, len(mask) - 1))
    return zones


def rolling_last_percentile(series: pd.Series, window: int) -> pd.Series:
    values = series.astype(float)

    def _pct_last(arr: np.ndarray) -> float:
        if len(arr) == 0 or not np.isfinite(arr[-1]):
            return np.nan
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid <= arr[-1]) / len(valid))

    return values.rolling(window, min_periods=window).apply(_pct_last, raw=True)


def build_n_up_feature(df: pd.DataFrame, rank_col: str, rank_threshold: float) -> pd.Series:
    out = np.zeros(len(df), dtype=bool)
    lows = df["low"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    rank_values = df[rank_col].astype(float)

    for idx in range(len(df)):
        left = max(0, idx - N_LOOKBACK + 1)
        sub_rank = rank_values.iloc[left : idx + 1].reset_index(drop=True)
        zones = identify_low_zones(sub_rank <= rank_threshold)
        if len(zones) < 2:
            continue
        z1, z2 = zones[-2], zones[-1]
        z1_start, z1_end = left + z1[0], left + z1[1]
        z2_start, z2_end = left + z2[0], left + z2[1]
        first_low = float(np.min(lows[z1_start : z1_end + 1]))
        second_low = float(np.min(lows[z2_start : z2_end + 1]))
        if not (second_low > first_low):
            continue
        mid_left = z1_end + 1
        mid_right = z2_start - 1
        if mid_right < mid_left:
            continue
        rebound_high = float(np.max(highs[mid_left : mid_right + 1]))
        if closes[idx] > rebound_high:
            out[idx] = True
    return pd.Series(out, index=df.index)


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
        if len(x) < b2filter.MIN_BARS:
            continue
        x["j_rank_20"] = rolling_last_percentile(x["J"], 20)
        x["j_rank_30"] = rolling_last_percentile(x["J"], 30)
        x["n_up_rank20_p10"] = build_n_up_feature(x, "j_rank_20", 0.10)
        x["n_up_rank20_p05"] = build_n_up_feature(x, "j_rank_20", 0.05)
        x["n_up_rank30_p10"] = build_n_up_feature(x, "j_rank_30", 0.10)
        x["n_up_rank30_p05"] = build_n_up_feature(x, "j_rank_30", 0.05)
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
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "first_eligible_exit_date": df.at[entry_idx + 1, "date"] if entry_idx + 1 < len(df) else pd.NaT,
        "exit_date": df.at[exit_idx, "date"],
        "ret": ret,
        "success": ret > 0,
        "holding_days": int(exit_idx - entry_idx),
        "sort_score": float(df.at[signal_idx, "sort_score"]),
        "max_runup": float(highs.max() / entry_price - 1.0) if not highs.empty else np.nan,
        "max_drawdown_during_trade": float(lows.min() / entry_price - 1.0) if not lows.empty else np.nan,
        "exit_reason": exit_reason,
    }


def simulate_trade(df: pd.DataFrame, signal_idx: int, combo: Combo) -> Optional[dict]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    last_hold_idx = entry_idx + combo.max_hold_days - 1
    if last_hold_idx >= len(df):
        return None
    first_eligible_idx = entry_idx + 1
    trend_break_streak = 0

    for i in range(first_eligible_idx, last_hold_idx + 1):
        close_i = float(df.at[i, "close"])
        trend_i = float(df.at[i, "trend_line"])
        trend_break = close_i < trend_i

        if trend_break:
            trend_break_streak += 1
        else:
            trend_break_streak = 0

        if combo.exit_mode == "跌破趋势线次日开盘":
            if trend_break and i + 1 < len(df):
                return _make_trade(df, signal_idx, i + 1, float(df.at[i + 1, "open"]), "跌破趋势线")

        elif combo.exit_mode == "连续2天跌破趋势线次日开盘":
            if trend_break_streak >= 2 and i + 1 < len(df):
                return _make_trade(df, signal_idx, i + 1, float(df.at[i + 1, "open"]), "连续2天跌破趋势线")

        elif combo.exit_mode == "跌破趋势线次日未收回卖":
            if trend_break and i + 1 < len(df):
                next_close = float(df.at[i + 1, "close"])
                next_trend = float(df.at[i + 1, "trend_line"])
                if next_close < next_trend and i + 1 < len(df):
                    sell_idx = i + 2
                    if sell_idx < len(df):
                        return _make_trade(df, signal_idx, sell_idx, float(df.at[sell_idx, "open"]), "跌破趋势线次日未收回")

        elif combo.exit_mode == "趋势线拐头下且跌破趋势线次日开盘":
            if i - 3 >= 0:
                slope3 = float(df.at[i, "trend_line"] / df.at[i - 3, "trend_line"] - 1.0) if float(df.at[i - 3, "trend_line"]) > 0 else np.nan
                if trend_break and np.isfinite(slope3) and slope3 < 0 and i + 1 < len(df):
                    return _make_trade(df, signal_idx, i + 1, float(df.at[i + 1, "open"]), "趋势线拐头下且跌破")

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
    for max_hold_days in [20, 30]:
        for exit_mode in [
            "固定持有到期收盘",
            "跌破趋势线次日开盘",
            "连续2天跌破趋势线次日开盘",
            "跌破趋势线次日未收回卖",
            "趋势线拐头下且跌破趋势线次日开盘",
        ]:
            combos.append(
                Combo(
                    max_hold_days=max_hold_days,
                    exit_mode=exit_mode,
                    use_n_up=False,
                    n_rank_window=20,
                    n_rank_threshold=0.10,
                )
            )
            for n_window in [20, 30]:
                for n_threshold in [0.10, 0.05]:
                    combos.append(
                        Combo(
                            max_hold_days=max_hold_days,
                            exit_mode=exit_mode,
                            use_n_up=True,
                            n_rank_window=n_window,
                            n_rank_threshold=n_threshold,
                        )
                    )
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
            mask = base_mask(df).copy()
            if combo.use_n_up:
                n_col = f"n_up_rank{combo.n_rank_window}_p{int(combo.n_rank_threshold * 100):02d}"
                mask &= df[n_col]
            idxs = np.flatnonzero(mask.fillna(False).to_numpy())
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
    summary = {
        "data_dir": str(DATA_DIR),
        "exclude_range": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
        "entry_signal": "当前最优B2，仅使用T日已知信息",
        "entry": "T+1_open",
        "note": "只研究趋势线退出；买入当日不能卖；N型结构改为J在20/30日历史分位 <=10%/5% 识别低位区，且仅使用T日之前信息识别",
        "best_combo": result_df.iloc[0].to_dict(),
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
