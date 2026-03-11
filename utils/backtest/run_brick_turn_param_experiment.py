from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
OUTPUT_DIR = "/Users/lidongyang/Desktop/Qstrategy/results/brick_turn_param_experiment"
MIN_BARS = 160
EPS = 1e-12
TRADING_DAYS_PER_YEAR = 252
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")


DATE_COL_CANDIDATES = ["date", "Date", "trade_date", "日期", "DATE"]
OPEN_COL_CANDIDATES = ["open", "Open", "开盘", "OPEN"]
HIGH_COL_CANDIDATES = ["high", "High", "最高", "HIGH"]
LOW_COL_CANDIDATES = ["low", "Low", "最低", "LOW"]
CLOSE_COL_CANDIDATES = ["close", "Close", "收盘", "CLOSE"]
VOL_COL_CANDIDATES = ["volume", "vol", "Volume", "成交量", "VOL"]
CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码", "CODE"]


@dataclass(frozen=True)
class Combo:
    ratio_threshold: float
    gain_limit: float
    hold_days: int
    entry_timing: str
    scheduled_exit_timing: str
    take_profit_threshold: Optional[float]
    take_profit_exec: str
    stop_loss_mode: str
    stop_loss_exec: str

    @property
    def combo_name(self) -> str:
        tp = "none" if self.take_profit_threshold is None else f"{self.take_profit_threshold:.2f}"
        sl = self.stop_loss_mode
        return (
            f"ratio_{self.ratio_threshold:.1f}"
            f"__gain_{self.gain_limit:.2f}"
            f"__hold_{self.hold_days}"
            f"__entry_{self.entry_timing}"
            f"__exit_{self.scheduled_exit_timing}"
            f"__tp_{tp}"
            f"__tp_exec_{self.take_profit_exec}"
            f"__sl_{sl}"
            f"__sl_exec_{self.stop_loss_exec}"
        )


def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > EPS)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    return series.ewm(alpha=m / n, adjust=False).mean()


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"缺少字段，候选字段={candidates}")
    return None


def read_csv_auto(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+|\t+", engine="python")


def load_one_csv(path: str) -> Optional[pd.DataFrame]:
    raw = read_csv_auto(path)
    date_col = pick_col(raw, DATE_COL_CANDIDATES)
    open_col = pick_col(raw, OPEN_COL_CANDIDATES)
    high_col = pick_col(raw, HIGH_COL_CANDIDATES)
    low_col = pick_col(raw, LOW_COL_CANDIDATES)
    close_col = pick_col(raw, CLOSE_COL_CANDIDATES)
    vol_col = pick_col(raw, VOL_COL_CANDIDATES)
    code_col = pick_col(raw, CODE_COL_CANDIDATES, required=False)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "open": pd.to_numeric(raw[open_col], errors="coerce"),
            "high": pd.to_numeric(raw[high_col], errors="coerce"),
            "low": pd.to_numeric(raw[low_col], errors="coerce"),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
            "volume": pd.to_numeric(raw[vol_col], errors="coerce"),
        }
    )
    if code_col:
        df["code"] = raw[code_col].astype(str).iloc[0]
    else:
        df["code"] = os.path.splitext(os.path.basename(path))[0]

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df[(df["date"] < EXCLUDE_START) | (df["date"] > EXCLUDE_END)].copy()
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ].copy()
    if len(df) < MIN_BARS:
        return None
    return df


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    streak = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        streak[i] = streak[i - 1] + 1 if green_flag[i] else 0
    return streak


def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x["ret1"] = x["close"].pct_change()
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["bull_bear_line"] = (
        x["close"].rolling(14).mean()
        + x["close"].rolling(28).mean()
        + x["close"].rolling(57).mean()
        + x["close"].rolling(114).mean()
    ) / 4.0
    x["trend_gt_bullbear"] = (x["trend_line"] > x["bull_bear_line"]).fillna(False)

    hhv4 = x["high"].rolling(4).max()
    llv4 = x["low"].rolling(4).min()
    den4 = (hhv4 - llv4).replace(0, np.nan)
    var1a = safe_div((hhv4 - x["close"]), den4) * 100 - 90
    var2a = tdx_sma(pd.Series(var1a, index=x.index), 4, 1) + 100
    var3a = safe_div((x["close"] - llv4), den4) * 100
    var4a = tdx_sma(pd.Series(var3a, index=x.index), 6, 1)
    var5a = tdx_sma(var4a, 6, 1) + 100
    var6a = var5a - var2a

    x["brick"] = np.where(var6a > 4, var6a - 4, 0.0)
    x["brick_prev"] = x["brick"].shift(1)
    x["brick_red_len"] = np.where(x["brick"] > x["brick_prev"], x["brick"] - x["brick_prev"], 0.0)
    x["brick_green_len"] = np.where(x["brick"] < x["brick_prev"], x["brick_prev"] - x["brick"], 0.0)
    x["today_red"] = x["brick_red_len"] > 0
    x["yesterday_green"] = x["brick_green_len"].shift(1).fillna(0) > 0
    x["today_red_turn"] = x["today_red"] & x["yesterday_green"]
    x["prev_bar_len"] = (x["brick"] - x["brick_prev"]).abs().shift(1)
    x["brick_ratio_vs_prev_bar"] = safe_div(x["brick_red_len"], x["prev_bar_len"])

    green_streak = calc_green_streak((x["brick_green_len"] > 0).to_numpy())
    x["prev_green_streak"] = pd.Series(green_streak, index=x.index).shift(1)
    x["vol_gt_prev3"] = (x["volume"] > x["volume"].shift(1).rolling(3).max()).fillna(False)
    x["sort_ratio_score"] = safe_div(x["brick_ratio_vs_prev_bar"], x["ret1"].abs())
    x["valid_signal_base"] = (
        x["today_red_turn"]
        & (x["prev_green_streak"] >= 3)
        & x["brick_ratio_vs_prev_bar"].notna()
        & x["ret1"].notna()
    )
    return x


def load_feature_map(data_dir: str) -> Dict[str, pd.DataFrame]:
    feature_map: Dict[str, pd.DataFrame] = {}
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".txt"))]
    for idx, file_name in enumerate(files, 1):
        df = load_one_csv(os.path.join(data_dir, file_name))
        if df is None:
            continue
        code = str(df["code"].iloc[0])
        feature_map[code] = build_feature_df(df)
        if idx % 500 == 0:
            print(f"特征进度: {idx}/{len(files)}")
    return feature_map


def build_combos() -> List[Combo]:
    combos: List[Combo] = []
    for ratio, gain_limit, hold_days, entry_timing, exit_timing, tp_threshold, tp_exec, sl_mode, sl_exec in product(
        [3.0, 4.0, 5.0],
        [0.03, 0.04, 0.05],
        [2, 3, 4],
        ["next_open"],
        ["next_open"],
        [0.02, 0.03, 0.04],
        ["next_open"],
        ["entry_low", "entry_low_x_0.99", "entry_low_x_0.98"],
        ["next_open"],
    ):
        combos.append(
            Combo(
                ratio_threshold=ratio,
                gain_limit=gain_limit,
                hold_days=hold_days,
                entry_timing=entry_timing,
                scheduled_exit_timing=exit_timing,
                take_profit_threshold=tp_threshold,
                take_profit_exec=tp_exec,
                stop_loss_mode=sl_mode,
                stop_loss_exec=sl_exec,
            )
        )
    return combos


def entry_index_from_signal(signal_idx: int, entry_timing: str, n: int) -> Optional[int]:
    if entry_timing == "close":
        return signal_idx
    if signal_idx + 1 < n:
        return signal_idx + 1
    return None


def entry_price_from_index(df: pd.DataFrame, idx: int, entry_timing: str) -> Optional[float]:
    price = float(df.at[idx, "close"] if entry_timing == "close" else df.at[idx, "open"])
    if np.isfinite(price) and price > 0:
        return price
    return None


def stop_loss_price(df: pd.DataFrame, signal_idx: int, mode: str) -> Optional[float]:
    low = float(df.at[signal_idx, "low"])
    if not np.isfinite(low) or low <= 0:
        return None
    if mode == "entry_low":
        return low
    if mode == "entry_low_x_0.99":
        return low * 0.99
    if mode == "entry_low_x_0.98":
        return low * 0.98
    if mode == "entry_low_x_0.97":
        return low * 0.97
    return None


def scheduled_exit_price(df: pd.DataFrame, entry_idx: int, combo: Combo) -> tuple[int, Optional[float], str]:
    n = len(df)
    if combo.scheduled_exit_timing == "close":
        exit_idx = min(entry_idx + combo.hold_days, n - 1)
        price = float(df.at[exit_idx, "close"])
    else:
        exit_idx = min(entry_idx + combo.hold_days + 1, n - 1)
        price = float(df.at[exit_idx, "open"])
    if not np.isfinite(price) or price <= 0:
        return exit_idx, None, "invalid_exit"
    return exit_idx, price, "time_exit"


def simulate_trade(df: pd.DataFrame, signal_idx: int, combo: Combo) -> Optional[dict]:
    n = len(df)
    entry_idx = entry_index_from_signal(signal_idx, combo.entry_timing, n)
    if entry_idx is None:
        return None
    entry_price = entry_price_from_index(df, entry_idx, combo.entry_timing)
    if entry_price is None:
        return None

    scheduled_idx, scheduled_price, scheduled_reason = scheduled_exit_price(df, entry_idx, combo)
    if scheduled_price is None:
        return None

    exit_idx = scheduled_idx
    exit_price = scheduled_price
    exit_reason = scheduled_reason
    sl_price = stop_loss_price(df, signal_idx, combo.stop_loss_mode)

    for j in range(entry_idx + 1, min(scheduled_idx, n - 1) + 1):
        if sl_price is not None:
            if combo.stop_loss_exec == "same_day_close":
                if float(df.at[j, "low"]) <= sl_price:
                    price = float(df.at[j, "close"])
                    if np.isfinite(price) and price > 0:
                        exit_idx = j
                        exit_price = price
                        exit_reason = "sl_same_day_close"
                        break
            else:
                if float(df.at[j, "low"]) <= sl_price:
                    next_idx = min(j + 1, n - 1)
                    if next_idx > entry_idx:
                        price = float(df.at[next_idx, "open"])
                        if np.isfinite(price) and price > 0:
                            exit_idx = next_idx
                            exit_price = price
                            exit_reason = "sl_next_open"
                            break

        target_price = entry_price * (1 + combo.take_profit_threshold)
        if float(df.at[j, "high"]) < target_price:
            continue
        if combo.take_profit_exec == "same_day_close":
            price = float(df.at[j, "close"])
            if np.isfinite(price) and price > 0:
                exit_idx = j
                exit_price = price
                exit_reason = "tp_same_day_close"
                break
        else:
            next_idx = min(j + 1, n - 1)
            if next_idx <= entry_idx:
                continue
            price = float(df.at[next_idx, "open"])
            if np.isfinite(price) and price > 0:
                exit_idx = next_idx
                exit_price = price
                exit_reason = "tp_next_open"
                break

    ret = exit_price / entry_price - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "ret": ret,
        "holding_days": int(exit_idx - entry_idx),
        "success": ret > 0,
        "trend_gt_bullbear": bool(df.at[signal_idx, "trend_gt_bullbear"]),
        "vol_gt_prev3": bool(df.at[signal_idx, "vol_gt_prev3"]),
        "exit_reason": exit_reason,
    }


def max_consecutive_failures(success_flags: Iterable[bool]) -> int:
    current = 0
    worst = 0
    for flag in success_flags:
        if flag:
            current = 0
        else:
            current += 1
            worst = max(worst, current)
    return worst


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
    equity_days = len(portfolio_df)
    final_equity = float(eq.iloc[-1])
    annual_return = (
        (final_equity / 1_000_000.0) ** (TRADING_DAYS_PER_YEAR / equity_days) - 1
        if equity_days > 0 and final_equity > 0
        else np.nan
    )
    return {
        "annual_return": float(annual_return),
        "max_drawdown": float(drawdown.min()) if equity_days > 0 else np.nan,
        "equity_days": int(equity_days),
        "final_equity": final_equity,
    }


def build_portfolio_curve(
    trade_df: pd.DataFrame,
    initial_cash: float = 1_000_000.0,
    max_positions: int = 10,
    win_rate_lookback_days: int = 15,
    low_win_rate_threshold: float = 0.5,
    reduced_exposure: float = 0.5,
) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["signal_date", "portfolio_ret", "equity"])
    rows = []
    equity = initial_cash
    all_dates = sorted(pd.to_datetime(trade_df["signal_date"]).dropna().unique().tolist())
    trade_df = trade_df.copy()
    trade_df["signal_date"] = pd.to_datetime(trade_df["signal_date"])
    trade_df["exit_date"] = pd.to_datetime(trade_df["exit_date"])

    for signal_date in all_dates:
        group = trade_df[trade_df["signal_date"] == signal_date].copy()
        g = group.copy()
        score = g["sort_score"].replace([np.inf, -np.inf], np.nan).fillna(-1e18)
        g = g.assign(_score=score).sort_values(["_score", "code"], ascending=[False, True]).head(max_positions)

        lookback_start = signal_date - pd.Timedelta(days=win_rate_lookback_days)
        recent_closed = trade_df[
            (trade_df["exit_date"] < signal_date)
            & (trade_df["exit_date"] >= lookback_start)
        ]
        exposure = 1.0
        if len(recent_closed) > 0:
            recent_win_rate = float(recent_closed["success"].mean())
            if recent_win_rate < low_win_rate_threshold:
                exposure = reduced_exposure
        basket_ret = float(g["ret"].mean()) if len(g) > 0 else 0.0
        invested_ret = exposure * basket_ret
        equity *= 1.0 + invested_ret
        rows.append(
            {
                "signal_date": signal_date,
                "portfolio_ret": invested_ret,
                "raw_basket_ret": basket_ret,
                "exposure": exposure,
                "equity": equity,
            }
        )
    return pd.DataFrame(rows)


def summarize_combo(combo: Combo, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    if trade_df.empty:
        base = {
            "sample_count": 0,
            "avg_trade_return": np.nan,
            "success_rate": np.nan,
            "max_consecutive_failures": np.nan,
        }
    else:
        base = {
            "sample_count": int(len(trade_df)),
            "avg_trade_return": float(trade_df["ret"].mean()),
            "success_rate": float(trade_df["success"].mean()),
            "max_consecutive_failures": int(max_consecutive_failures(trade_df["success"].tolist())),
        }
    base.update(compute_equity_metrics(portfolio_df))
    base.update(asdict(combo))
    base["combo_name"] = combo.combo_name
    return base


def run_experiment(feature_map: Dict[str, pd.DataFrame], combos: List[Combo]) -> pd.DataFrame:
    signal_cache: Dict[tuple[float, float], Dict[str, np.ndarray]] = {}
    rows: List[dict] = []

    for combo_idx, combo in enumerate(combos, 1):
        cache_key = (combo.ratio_threshold, combo.gain_limit)
        if cache_key not in signal_cache:
            per_code_signal_idx: Dict[str, np.ndarray] = {}
            for code, df in feature_map.items():
                mask = (
                    df["valid_signal_base"]
                    & (df["brick_ratio_vs_prev_bar"] >= combo.ratio_threshold)
                    & (df["ret1"] <= combo.gain_limit)
                )
                idxs = np.flatnonzero(mask.to_numpy())
                if len(idxs) > 0:
                    per_code_signal_idx[code] = idxs
            signal_cache[cache_key] = per_code_signal_idx

        trades: List[dict] = []
        for code, idxs in signal_cache[cache_key].items():
            df = feature_map[code]
            for signal_idx in idxs:
                trade = simulate_trade(df, int(signal_idx), combo)
                if trade is None:
                    continue
                trade["code"] = code
                trade["sort_score"] = float(df.at[int(signal_idx), "sort_ratio_score"])
                trades.append(trade)

        trade_df = (
            pd.DataFrame(trades).sort_values(["signal_date", "code"]).reset_index(drop=True)
            if trades
            else pd.DataFrame()
        )
        portfolio_df = build_portfolio_curve(trade_df, initial_cash=1_000_000.0, max_positions=10)
        rows.append(summarize_combo(combo, trade_df, portfolio_df))
        if combo_idx % 50 == 0 or combo_idx == len(combos):
            print(f"组合进度: {combo_idx}/{len(combos)}")

    return pd.DataFrame(rows).sort_values(
        ["annual_return", "avg_trade_return", "success_rate", "max_drawdown"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    experiment_design = {
        "strategy": "连续三天绿砖后翻红短线策略",
        "fixed_rules": [
            "连续三天绿砖后翻红",
            "不额外加入趋势线>多空线过滤",
            "不额外加入量能过滤，量能仅作为结果归因列保留",
        ],
        "grid": {
            "brick_ratio_threshold": [3.0, 4.0, 5.0],
            "daily_gain_limit": [0.03, 0.04, 0.05],
            "hold_days": [2, 3, 4],
            "entry_timing": ["next_open"],
            "scheduled_exit_timing": ["next_open"],
            "take_profit_threshold": [0.02, 0.03, 0.04],
            "take_profit_exec": ["next_open"],
            "stop_loss_mode": ["entry_low", "entry_low_x_0.99", "entry_low_x_0.98"],
            "stop_loss_exec": ["next_open"],
        },
        "notes": [
            "止盈触发条件定义为买入后任意一天最高价达到止盈阈值",
            "止损触发条件定义为买入后任意一天最低价跌破止损价",
            "买入当日禁止卖出，所以止盈检查从买入后的下一交易日开始",
            "买入当日禁止卖出，所以止损检查也从买入后的下一交易日开始",
            "年化收益率基于100万初始资金、每个信号日最多持仓10只、按砖长阈值与当日涨幅绝对值之比排序取前10的组合净值序列计算",
            "最佳组合筛选时要求样本数>=500，主排序指标为年化收益率",
        ],
    }
    (Path(OUTPUT_DIR) / "experiment_design.json").write_text(
        json.dumps(experiment_design, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    feature_map = load_feature_map(DATA_DIR)
    combos = build_combos()
    result_df = run_experiment(feature_map, combos)
    result_df.to_csv(Path(OUTPUT_DIR) / "combo_results.csv", index=False, encoding="utf-8-sig")

    eligible = result_df[result_df["sample_count"] >= 500].copy()
    best_row = eligible.iloc[0] if not eligible.empty else result_df.iloc[0]
    summary = {
        "best_combo": best_row.to_dict(),
        "top10": eligible.head(10).to_dict(orient="records") if not eligible.empty else result_df.head(10).to_dict(orient="records"),
    }
    (Path(OUTPUT_DIR) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
