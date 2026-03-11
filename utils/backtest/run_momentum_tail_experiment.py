from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
OUTPUT_DIR = "/Users/lidongyang/Desktop/Qstrategy/results/momentum_tail_experiment"
MIN_BARS = 160
EPS = 1e-12
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
MAX_SINGLE_WEIGHT = 0.2
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
    rebound_threshold: float
    gain_limit: float
    take_profit: float
    stop_mode: str

    @property
    def combo_name(self) -> str:
        return (
            f"rebound_{self.rebound_threshold:.2f}"
            f"__gain_{self.gain_limit:.2f}"
            f"__tp_{self.take_profit:.3f}"
            f"__stop_{self.stop_mode}"
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
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x["ret1"] = x["close"].pct_change()
    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["close_pullback_white"] = x["close"] < x["trend_line"] * 1.01
    x["close_above_white"] = x["close"] > x["trend_line"]
    x["gain_limit_base"] = x["ret1"].notna()

    x["ma15_vol"] = x["volume"].rolling(15).mean()
    x["is_yang"] = x["close"] > x["open"]
    x["yang_volume"] = np.where(x["is_yang"], x["volume"], np.nan)
    x["max_yang_vol_15"] = pd.Series(x["yang_volume"], index=x.index).shift(1).rolling(15).max()
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
    x["brick_red"] = x["brick_red_len"] > 0
    x["brick_green"] = x["brick_green_len"] > 0
    x["prev_green_streak"] = pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(1)

    # Horizontal filter via 10-day close regression slope.
    x["close_slope_10"] = (
        x["close"]
        .rolling(10)
        .apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if np.isfinite(s).all() else np.nan, raw=False)
    )
    x["not_sideways"] = np.abs(safe_div(x["close_slope_10"], x["close"].rolling(10).mean())) > 0.002

    # Momentum lead-up and shrink-volume pullback.
    x["up_leg_avg_vol"] = x["volume"].shift(4).rolling(3).mean()
    x["pullback_avg_vol"] = x["volume"].shift(1).rolling(3).mean()
    x["pullback_shrinking"] = x["pullback_avg_vol"] < x["up_leg_avg_vol"]
    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5_prev"])
    x["signal_vs_ma5_valid"] = x["signal_vs_ma5"].between(1.3, 2.2, inclusive="both")
    x["pattern_a"] = (
        (x["prev_green_streak"] >= 3)
        & x["brick_red"]
        & x["close_pullback_white"].shift(1).fillna(False)
        & x["close_above_white"]
    )
    x["pattern_b"] = (
        (pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(3) >= 3)
        & x["brick_red"]
        & x["brick_green"].shift(1).fillna(False)
        & x["brick_red"].shift(2).fillna(False)
        & x["close_pullback_white"].shift(1).fillna(False)
        & x["close_above_white"]
    )

    x["rebound_ratio"] = safe_div(x["brick_red_len"], x["brick_green_len"].shift(1))
    x["signal_base"] = (
        (x["pattern_a"] | x["pattern_b"])
        & x["pullback_shrinking"].fillna(False)
        & x["signal_vs_ma5_valid"].fillna(False)
        & x["not_sideways"].fillna(False)
        & x["gain_limit_base"]
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


def build_round1_combos() -> List[Combo]:
    return [
        Combo(rebound_threshold=rebound, gain_limit=gain, take_profit=tp, stop_mode=stop)
        for rebound, gain, tp, stop in product(
            [1.0, 1.2, 1.5, 2.0],
            [0.06, 0.08],
            [0.02, 0.03, 0.04],
            ["entry_low", "entry_low_x_0.99"],
        )
    ]


def unique_sorted(values: Sequence[float], floor: Optional[float] = None) -> List[float]:
    out = []
    for value in values:
        v = float(value)
        if floor is not None:
            v = max(v, floor)
        out.append(round(v, 4))
    return sorted(set(out))


def build_round2_combos(round1_best: Combo) -> List[Combo]:
    rebound_candidates = unique_sorted(
        [round1_best.rebound_threshold - 0.2, round1_best.rebound_threshold, round1_best.rebound_threshold + 0.2],
        floor=1.0,
    )
    if abs(round1_best.gain_limit - 0.06) < EPS:
        gain_candidates = [0.05, 0.06, 0.07]
    else:
        gain_candidates = [0.07, 0.08, 0.09]
    if abs(round1_best.take_profit - 0.02) < EPS:
        tp_candidates = [0.015, 0.02, 0.025]
    elif abs(round1_best.take_profit - 0.03) < EPS:
        tp_candidates = [0.025, 0.03, 0.035]
    else:
        tp_candidates = [0.035, 0.04, 0.045]
    if round1_best.stop_mode == "entry_low":
        stop_candidates = ["entry_low", "entry_low_x_0.995", "entry_low_x_0.99"]
    else:
        stop_candidates = ["entry_low_x_0.995", "entry_low_x_0.99", "entry_low_x_0.985"]
    return [
        Combo(rebound_threshold=rebound, gain_limit=gain, take_profit=tp, stop_mode=stop)
        for rebound, gain, tp, stop in product(rebound_candidates, gain_candidates, tp_candidates, stop_candidates)
    ]


def stop_loss_price(signal_low: float, mode: str) -> Optional[float]:
    if not np.isfinite(signal_low) or signal_low <= 0:
        return None
    if mode == "entry_low":
        return signal_low
    if mode == "entry_low_x_0.99":
        return signal_low * 0.99
    if mode == "entry_low_x_0.995":
        return signal_low * 0.995
    if mode == "entry_low_x_0.985":
        return signal_low * 0.985
    return None


def simulate_trade(df: pd.DataFrame, signal_idx: int, combo: Combo) -> Optional[dict]:
    n = len(df)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None
    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    signal_low = float(df.at[signal_idx, "low"])
    sl_price = stop_loss_price(signal_low, combo.stop_mode)
    target_price = entry_price * (1.0 + combo.take_profit)
    scheduled_exit_idx = min(entry_idx + 3 + 1, n - 1)  # hold 3 days, exit next open
    exit_idx = scheduled_exit_idx
    exit_price = float(df.at[exit_idx, "open"])
    exit_reason = "time_exit_next_open"

    for j in range(entry_idx + 1, min(entry_idx + 3, n - 1) + 1):
        if sl_price is not None and float(df.at[j, "low"]) <= sl_price:
            next_idx = min(j + 1, n - 1)
            if next_idx > entry_idx:
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = "stop_loss_next_open"
                    break
        if float(df.at[j, "high"]) >= target_price:
            next_idx = min(j + 1, n - 1)
            if next_idx > entry_idx:
                px = float(df.at[next_idx, "open"])
                if np.isfinite(px) and px > 0:
                    exit_idx = next_idx
                    exit_price = px
                    exit_reason = "take_profit_next_open"
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
        "exit_reason": exit_reason,
        "signal_low": signal_low,
        "pattern_a": bool(df.at[signal_idx, "pattern_a"]),
        "pattern_b": bool(df.at[signal_idx, "pattern_b"]),
    }


def build_signal_cache(feature_map: Dict[str, pd.DataFrame], combos: List[Combo]) -> Dict[tuple[float, float], Dict[str, np.ndarray]]:
    cache: Dict[tuple[float, float], Dict[str, np.ndarray]] = {}
    for combo in combos:
        key = (combo.rebound_threshold, combo.gain_limit)
        if key in cache:
            continue
        per_code: Dict[str, np.ndarray] = {}
        for code, df in feature_map.items():
            mask_a = df["pattern_a"] & (df["rebound_ratio"] >= combo.rebound_threshold)
            mask_b = df["pattern_b"] & (df["rebound_ratio"] >= 1.0)
            mask = df["signal_base"] & (df["ret1"] <= combo.gain_limit) & (mask_a | mask_b)
            idxs = np.flatnonzero(mask.to_numpy())
            if len(idxs) > 0:
                per_code[code] = idxs
        cache[key] = per_code
    return cache


def build_portfolio_curve(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["signal_date", "portfolio_ret", "equity"])
    rows = []
    equity = INITIAL_CAPITAL
    trade_df = trade_df.copy()
    trade_df["signal_date"] = pd.to_datetime(trade_df["signal_date"])
    for signal_date, group in trade_df.groupby("signal_date", sort=True):
        g = group.copy().sort_values(["sort_score", "code"], ascending=[False, True]).head(MAX_POSITIONS)
        score = g["sort_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        score = score.clip(lower=0.0)
        if score.sum() <= 0:
            weights = np.repeat(1 / len(g), len(g))
        else:
            weights = (score / score.sum()).clip(upper=MAX_SINGLE_WEIGHT).to_numpy()
            weights = weights / weights.sum()
        basket_ret = float(np.sum(g["ret"].to_numpy() * weights))
        equity *= 1.0 + basket_ret
        rows.append({"signal_date": signal_date, "portfolio_ret": basket_ret, "equity": equity})
    return pd.DataFrame(rows)


def compute_equity_metrics(portfolio_df: pd.DataFrame) -> dict:
    if portfolio_df.empty:
        return {"annual_return": np.nan, "max_drawdown": np.nan, "equity_days": 0, "final_equity": np.nan}
    eq = portfolio_df["equity"].astype(float)
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    final_equity = float(eq.iloc[-1])
    days = len(portfolio_df)
    annual_return = (final_equity / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / days) - 1 if final_equity > 0 and days > 0 else np.nan
    return {
        "annual_return": float(annual_return),
        "max_drawdown": float(drawdown.min()),
        "equity_days": int(days),
        "final_equity": final_equity,
    }


def max_consecutive_failures(success_flags: List[bool]) -> int:
    current = 0
    worst = 0
    for flag in success_flags:
        if flag:
            current = 0
        else:
            current += 1
            worst = max(worst, current)
    return worst


def summarize_combo(combo: Combo, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    if trade_df.empty:
        row = {
            "sample_count": 0,
            "avg_trade_return": np.nan,
            "success_rate": np.nan,
            "max_consecutive_failures": np.nan,
            "pattern_a_share": np.nan,
            "pattern_b_share": np.nan,
        }
    else:
        row = {
            "sample_count": int(len(trade_df)),
            "avg_trade_return": float(trade_df["ret"].mean()),
            "success_rate": float(trade_df["success"].mean()),
            "max_consecutive_failures": int(max_consecutive_failures(trade_df["success"].tolist())),
            "pattern_a_share": float(trade_df["pattern_a"].mean()),
            "pattern_b_share": float(trade_df["pattern_b"].mean()),
        }
    row.update(compute_equity_metrics(portfolio_df))
    row.update(asdict(combo))
    row["combo_name"] = combo.combo_name
    return row


def validate_result_df(result_df: pd.DataFrame, stage_name: str) -> None:
    non_empty = result_df["sample_count"].fillna(0).gt(0).any()
    if not non_empty:
        raise ValueError(f"{stage_name} 没有任何非零样本组合")
    finite_dd = result_df["max_drawdown"].dropna()
    if not finite_dd.empty and ((finite_dd < -1.0) | (finite_dd > 0.0)).any():
        raise ValueError(f"{stage_name} 存在非法最大回撤")
    finite_eq = result_df["final_equity"].dropna()
    if not finite_eq.empty and (finite_eq <= 0).any():
        raise ValueError(f"{stage_name} 存在非正最终净值")
    valid = result_df.dropna(subset=["annual_return", "final_equity"])
    bad_direction = valid[
        ((valid["final_equity"] > INITIAL_CAPITAL) & (valid["annual_return"] <= -EPS))
        | ((valid["final_equity"] < INITIAL_CAPITAL) & (valid["annual_return"] >= EPS))
    ]
    if not bad_direction.empty:
        raise ValueError(f"{stage_name} 年化收益率与最终净值方向不一致")


def select_best_row(result_df: pd.DataFrame, min_sample_count: int = 1000) -> pd.Series:
    eligible = result_df[result_df["sample_count"].fillna(0) >= min_sample_count].copy()
    if eligible.empty:
        raise ValueError(f"没有样本数 >= {min_sample_count} 的候选组合")
    eligible["drawdown_abs"] = eligible["max_drawdown"].abs()
    eligible = eligible.sort_values(
        ["annual_return", "drawdown_abs", "avg_trade_return", "success_rate"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    return eligible.iloc[0]


def run_experiment(feature_map: Dict[str, pd.DataFrame], combos: List[Combo], stage_name: str) -> pd.DataFrame:
    signal_cache = build_signal_cache(feature_map, combos)
    rows: List[dict] = []
    for idx, combo in enumerate(combos, 1):
        trades: List[dict] = []
        for code, signal_idxs in signal_cache[(combo.rebound_threshold, combo.gain_limit)].items():
            df = feature_map[code]
            for signal_idx in signal_idxs:
                trade = simulate_trade(df, int(signal_idx), combo)
                if trade is None:
                    continue
                trade["code"] = code
                trade["sort_score"] = float(df.at[int(signal_idx), "rebound_ratio"] / max(abs(float(df.at[int(signal_idx), "ret1"])), 0.01))
                trades.append(trade)
        trade_df = pd.DataFrame(trades).sort_values(["signal_date", "code"]).reset_index(drop=True) if trades else pd.DataFrame()
        portfolio_df = build_portfolio_curve(trade_df)
        rows.append(summarize_combo(combo, trade_df, portfolio_df))
        print(f"{stage_name} 组合进度: {idx}/{len(combos)}")
    result_df = pd.DataFrame(rows).sort_values(
        ["annual_return", "max_drawdown", "avg_trade_return", "success_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    validate_result_df(result_df, stage_name)
    return result_df


def persist_stage(output_dir: Path, stage_name: str, result_df: pd.DataFrame, best_row: pd.Series, extra: Optional[dict] = None) -> None:
    csv_path = output_dir / f"{stage_name}_combo_results.csv"
    json_path = output_dir / f"{stage_name}_summary.json"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = {
        "stage": stage_name,
        "best_combo": best_row.to_dict(),
        "top10": result_df.head(10).to_dict(orient="records"),
    }
    if extra:
        summary.update(extra)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    design = {
        "strategy": "超短线动量续冲策略",
        "rules": {
            "patterns": ["3绿1红", "3绿1红1绿1红"],
            "white_line": "知行趋势线",
            "pullback_white": "前一日 close < 白线 * 1.01",
            "stand_above_white": "信号日 close > 白线",
            "sideways_filter": "10日回归斜率绝对值/10日均价 > 0.002",
            "volume_shape": "回撤阶段平均量小于前一段上涨平均量",
            "volume_filter": "信号日量 / 前5日均量在 1.3~2.2",
            "round1_gain_limits": [0.06, 0.08],
            "round1_rebound_thresholds_for_3g1r": [1.0, 1.2, 1.5, 2.0],
            "rebound_threshold_for_3g1r1g1r": 1.0,
            "entry": "次日开盘",
            "holding": "固定3天，到期次日开盘卖出",
            "round1_take_profit_thresholds": [0.02, 0.03, 0.04],
            "round1_stop_modes": ["entry_low", "entry_low_x_0.99"],
            "capital": "100万，最多10只，按评分加权，单票上限20%",
            "exclude_date_range": ["2015-06-01", "2024-09-30"],
        },
    }
    (output_dir / "experiment_design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"样本过滤区间已生效: {EXCLUDE_START.date()} ~ {EXCLUDE_END.date()}")
    feature_map = load_feature_map(DATA_DIR)
    round1_combos = build_round1_combos()
    print(f"第一轮组合数: {len(round1_combos)}")
    round1_df = run_experiment(feature_map, round1_combos, "round1")
    round1_best = select_best_row(round1_df)
    persist_stage(output_dir, "round1", round1_df, round1_best)

    round1_center = Combo(
        rebound_threshold=float(round1_best["rebound_threshold"]),
        gain_limit=float(round1_best["gain_limit"]),
        take_profit=float(round1_best["take_profit"]),
        stop_mode=str(round1_best["stop_mode"]),
    )
    round2_combos = build_round2_combos(round1_center)
    print(f"第二轮组合数: {len(round2_combos)}")
    round2_df = run_experiment(feature_map, round2_combos, "round2")
    round2_best = select_best_row(round2_df)
    center_name = round1_center.combo_name
    round2_names = set(round2_df["combo_name"].tolist())
    if center_name not in round2_names:
        raise ValueError("第一轮最优组合不在第二轮中心点附近")
    persist_stage(
        output_dir,
        "round2",
        round2_df,
        round2_best,
        extra={"round1_center_combo_name": center_name},
    )

    final_recommended = round2_best
    final_stage = "round2"
    round2_is_better = (
        float(round2_best["annual_return"]) > float(round1_best["annual_return"]) + EPS
        and abs(float(round2_best["max_drawdown"])) <= abs(float(round1_best["max_drawdown"])) + EPS
    )
    if not round2_is_better:
        final_recommended = round1_best
        final_stage = "round1"

    summary = {
        "round1_best": round1_best.to_dict(),
        "round2_best": round2_best.to_dict(),
        "final_recommendation_stage": final_stage,
        "final_recommendation": final_recommended.to_dict(),
        "round2_substantially_better": bool(round2_is_better),
        "comparison": {
            "annual_return_diff": float(round2_best["annual_return"]) - float(round1_best["annual_return"]),
            "avg_trade_return_diff": float(round2_best["avg_trade_return"]) - float(round1_best["avg_trade_return"]),
            "success_rate_diff": float(round2_best["success_rate"]) - float(round1_best["success_rate"]),
            "max_drawdown_diff": float(round2_best["max_drawdown"]) - float(round1_best["max_drawdown"]),
            "sample_count_diff": int(round2_best["sample_count"]) - int(round1_best["sample_count"]),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
