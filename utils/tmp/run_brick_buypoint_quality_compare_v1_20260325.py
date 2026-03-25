from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
DATA_DIR = ROOT / "data" / "20260317" / "normal"
RESULT_ROOT = ROOT / "results"

FORMAL_SELECTED_PATH = ROOT / "results" / "brick_green_exit_compare_v1_full_20260323" / "selected_signals.csv"
RELAXED_CANDIDATE_PATH = ROOT / "results" / "brick_comprehensive_lab_full_20260325_r1" / "candidate_scored.csv"
RELAXED_BEST_CONFIG_PATH = ROOT / "results" / "brick_comprehensive_lab_full_20260325_r1" / "best_config.json"

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
INITIAL_CAPITAL = 1_000_000.0
TP_LIST = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
SL_MULT_LIST = [0.985, 0.99, 0.995]
MAX_HOLD_LIST = [2, 3, 4, 5]
FORWARD_HORIZONS = [3, 5]
HIT_LEVELS = [0.01, 0.02, 0.03, 0.05]


@dataclass
class TradeCandidate:
    strategy: str
    code: str
    signal_idx: int
    signal_date: pd.Timestamp
    entry_idx: int
    entry_date: pd.Timestamp
    entry_price: float
    signal_low: float
    score: float


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {
        "stage": stage,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update(extra)
    (result_dir / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_text_auto(path: Path) -> list[str]:
    for enc in ("gbk", "utf-8", "latin1"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except Exception:
            pass
    raise RuntimeError(f"无法读取文件: {path}")


def load_daily_df(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in read_text_auto(path)[2:]:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        rows.append(
            {
                "date": pd.to_datetime(parts[0], errors="coerce"),
                "open": pd.to_numeric(parts[1], errors="coerce"),
                "high": pd.to_numeric(parts[2], errors="coerce"),
                "low": pd.to_numeric(parts[3], errors="coerce"),
                "close": pd.to_numeric(parts[4], errors="coerce"),
                "volume": pd.to_numeric(parts[5], errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return df[(df["date"] < EXCLUDE_START) | (df["date"] > EXCLUDE_END)].reset_index(drop=True)


def build_file_code_lookup() -> tuple[dict[str, str], dict[str, str]]:
    exact_lookup: dict[str, str] = {}
    digit_lookup: dict[str, str] = {}
    for path in DATA_DIR.glob("*.txt"):
        stem = path.stem
        exact_lookup[stem] = stem
        if "#" not in stem:
            continue
        prefix, code = stem.split("#", 1)
        digit_lookup[code] = stem
        digit_lookup[code.lstrip("0") or "0"] = stem
        if prefix == "SH" and code.startswith("6"):
            digit_lookup.setdefault(code, stem)
        elif prefix == "SZ" and code[:1] in {"0", "2", "3"}:
            digit_lookup.setdefault(code, stem)
        elif prefix == "BJ" and code[:1] in {"4", "8", "9"}:
            digit_lookup.setdefault(code, stem)
    return exact_lookup, digit_lookup


def normalize_trade_code(raw_code: Any, exact_lookup: dict[str, str], digit_lookup: dict[str, str]) -> str | None:
    code = str(raw_code).strip()
    if not code:
        return None
    if code in exact_lookup:
        return exact_lookup[code]
    bare = code.split("#", 1)[-1]
    if bare in digit_lookup:
        return digit_lookup[bare]
    if bare.isdigit():
        bare6 = bare.zfill(6)
        if bare6 in digit_lookup:
            return digit_lookup[bare6]
        if bare6.startswith("6"):
            prefixed = f"SH#{bare6}"
        elif bare6[:1] in {"0", "2", "3"}:
            prefixed = f"SZ#{bare6}"
        elif bare6[:1] in {"4", "8", "9"}:
            prefixed = f"BJ#{bare6}"
        else:
            prefixed = bare6
        if prefixed in exact_lookup:
            return prefixed
    return None


def build_daily_map(codes: set[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for code in sorted(codes):
        path = DATA_DIR / f"{code}.txt"
        if not path.exists():
            continue
        df = load_daily_df(path)
        if df.empty:
            continue
        out[code] = df
    return out


def load_relaxed_best_meta() -> dict[str, Any]:
    return json.loads(RELAXED_BEST_CONFIG_PATH.read_text(encoding="utf-8"))


def load_relaxed_selected() -> pd.DataFrame:
    best = load_relaxed_best_meta()
    df = pd.read_csv(RELAXED_CANDIDATE_PATH)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = df[(df["signal_date"] < EXCLUDE_START) | (df["signal_date"] > EXCLUDE_END)].copy()
    df = df[df["candidate_pool"] == best["candidate_pool"]].copy()
    df = df[df["sim_score"] >= float(best["sim_gate"])].copy()
    df = df.sort_values(["signal_date", "rank_score", "code"], ascending=[True, False, True], kind="mergesort")
    df = df.groupby("signal_date", group_keys=False).head(int(best["daily_topn"])).reset_index(drop=True)
    df["strategy"] = "relaxed_fusion"
    df["source_score"] = pd.to_numeric(df["rank_score"], errors="coerce").fillna(0.0)
    return df


def load_formal_selected(date_start: pd.Timestamp, date_end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(FORMAL_SELECTED_PATH)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = df[(df["signal_date"] < EXCLUDE_START) | (df["signal_date"] > EXCLUDE_END)].copy()
    df = df[(df["signal_date"] >= date_start) & (df["signal_date"] <= date_end)].copy()
    df["strategy"] = "formal_best"
    df["source_score"] = pd.to_numeric(df["sort_score"], errors="coerce").fillna(0.0)
    return df


def restrict_to_common_signal_dates(formal_df: pd.DataFrame, relaxed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_dates = sorted(set(formal_df["signal_date"]) & set(relaxed_df["signal_date"]))
    if not common_dates:
        return formal_df.iloc[0:0].copy(), relaxed_df.iloc[0:0].copy()
    formal_df = formal_df[formal_df["signal_date"].isin(common_dates)].copy()
    relaxed_df = relaxed_df[relaxed_df["signal_date"].isin(common_dates)].copy()
    return formal_df, relaxed_df


def build_trade_candidates(
    raw_df: pd.DataFrame,
    daily_map: dict[str, pd.DataFrame],
    exact_lookup: dict[str, str],
    digit_lookup: dict[str, str],
) -> list[TradeCandidate]:
    out: list[TradeCandidate] = []
    for row in raw_df.itertuples(index=False):
        normalized_code = normalize_trade_code(getattr(row, "code", ""), exact_lookup, digit_lookup)
        if not normalized_code:
            continue
        code = normalized_code
        daily = daily_map.get(code)
        if daily is None or daily.empty:
            continue
        signal_idx = int(row.signal_idx)
        entry_idx = signal_idx + 1
        if signal_idx < 0 or entry_idx >= len(daily):
            continue
        signal_date = pd.Timestamp(row.signal_date)
        if pd.Timestamp(daily.at[signal_idx, "date"]) != signal_date:
            date_match = daily.index[daily["date"] == signal_date]
            if len(date_match) == 0:
                continue
            signal_idx = int(date_match[0])
            entry_idx = signal_idx + 1
            if entry_idx >= len(daily):
                continue
        entry_price = float(daily.at[entry_idx, "open"])
        signal_low = float(getattr(row, "signal_low", getattr(row, "entry_low_for_trade", np.nan)))
        if not (np.isfinite(entry_price) and entry_price > 0 and np.isfinite(signal_low) and signal_low > 0):
            continue
        out.append(
            TradeCandidate(
                strategy=str(row.strategy),
                code=code,
                signal_idx=signal_idx,
                signal_date=signal_date,
                entry_idx=entry_idx,
                entry_date=pd.Timestamp(daily.at[entry_idx, "date"]),
                entry_price=entry_price,
                signal_low=signal_low,
                score=float(getattr(row, "source_score", 0.0)),
            )
        )
    return out


def simulate_trade(
    candidate: TradeCandidate,
    daily: pd.DataFrame,
    tp_pct: float,
    sl_mult: float,
    max_hold_days: int,
) -> dict[str, Any]:
    tp_price = candidate.entry_price * (1.0 + tp_pct)
    stop_price = candidate.signal_low * sl_mult
    start_idx = candidate.entry_idx + 1
    last_idx = min(len(daily) - 1, candidate.entry_idx + max_hold_days)
    if start_idx > last_idx:
        exit_idx = min(len(daily) - 1, candidate.entry_idx)
        exit_price = float(daily.at[exit_idx, "close"])
        ret = exit_price / candidate.entry_price - 1.0
        return {
            "strategy": candidate.strategy,
            "code": candidate.code,
            "signal_date": candidate.signal_date,
            "entry_date": candidate.entry_date,
            "exit_date": pd.Timestamp(daily.at[exit_idx, "date"]),
            "entry_price": candidate.entry_price,
            "exit_price": exit_price,
            "return_pct": ret,
            "hold_days": int(exit_idx - candidate.entry_idx),
            "result": "time_exit",
        }
    triggered = False
    trigger_idx = None
    trigger_reason = None
    for idx in range(start_idx, last_idx + 1):
        high = float(daily.at[idx, "high"])
        low = float(daily.at[idx, "low"])
        tp_hit = np.isfinite(high) and high >= tp_price
        sl_hit = np.isfinite(low) and low <= stop_price
        if tp_hit or sl_hit:
            triggered = True
            trigger_idx = idx
            trigger_reason = "take_profit" if tp_hit else "stop_loss"
            break
    if triggered and trigger_idx is not None:
        next_idx = trigger_idx + 1
        if next_idx < len(daily):
            exit_idx = next_idx
            exit_price = float(daily.at[next_idx, "open"])
            exit_reason = f"{trigger_reason}_next_open"
        else:
            exit_idx = trigger_idx
            exit_price = float(daily.at[trigger_idx, "close"])
            exit_reason = f"{trigger_reason}_last_close_fallback"
    else:
        exit_idx = last_idx
        exit_price = float(daily.at[last_idx, "close"])
        exit_reason = "time_exit_close"
    ret = exit_price / candidate.entry_price - 1.0
    return {
        "strategy": candidate.strategy,
        "code": candidate.code,
        "signal_date": candidate.signal_date,
        "entry_date": candidate.entry_date,
        "exit_date": pd.Timestamp(daily.at[exit_idx, "date"]),
        "entry_price": candidate.entry_price,
        "exit_price": exit_price,
        "return_pct": ret,
        "hold_days": int(exit_idx - candidate.entry_idx),
        "result": exit_reason,
    }


def compute_forward_quality(candidate: TradeCandidate, daily: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "strategy": candidate.strategy,
        "code": candidate.code,
        "signal_date": candidate.signal_date,
        "entry_date": candidate.entry_date,
        "entry_price": candidate.entry_price,
        "score": candidate.score,
    }
    for horizon in FORWARD_HORIZONS:
        start_idx = candidate.entry_idx + 1
        end_idx = min(len(daily) - 1, candidate.entry_idx + horizon)
        if start_idx > end_idx:
            highs = np.array([], dtype=float)
            lows = np.array([], dtype=float)
            closes = np.array([], dtype=float)
        else:
            highs = daily.loc[start_idx:end_idx, "high"].astype(float).to_numpy()
            lows = daily.loc[start_idx:end_idx, "low"].astype(float).to_numpy()
            closes = daily.loc[end_idx:end_idx, "close"].astype(float).to_numpy()
        max_high_ret = float(np.nanmax(highs) / candidate.entry_price - 1.0) if highs.size else 0.0
        min_low_ret = float(np.nanmin(lows) / candidate.entry_price - 1.0) if lows.size else 0.0
        close_ret = float(closes[-1] / candidate.entry_price - 1.0) if closes.size else 0.0
        row[f"mfe_{horizon}d"] = max_high_ret
        row[f"mae_{horizon}d"] = min_low_ret
        row[f"close_ret_{horizon}d"] = close_ret
        for level in HIT_LEVELS:
            key = str(level).replace(".", "")
            row[f"hit_up_{key}_{horizon}d"] = bool(highs.size and np.nanmax(highs) >= candidate.entry_price * (1.0 + level))
            row[f"hit_dn_{key}_{horizon}d"] = bool(lows.size and np.nanmin(lows) <= candidate.entry_price * (1.0 - level))
    return row


def build_signal_basket_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "daily_ret", "equity"])
    daily = trades.groupby("signal_date", as_index=False)["return_pct"].mean()
    daily = daily.sort_values("signal_date").reset_index(drop=True)
    daily["equity"] = INITIAL_CAPITAL * (1.0 + daily["return_pct"]).cumprod()
    daily["daily_ret"] = daily["return_pct"]
    daily["date"] = daily["signal_date"]
    return daily[["date", "daily_ret", "equity"]]


def summarize_trades(trades: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
    if trades.empty:
        return {
            "strategy": strategy_name,
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "median_trade_return": 0.0,
            "avg_holding_days": 0.0,
            "profit_factor": 0.0,
            "total_return_signal_basket": 0.0,
            "annual_return_signal_basket": 0.0,
            "max_drawdown_signal_basket": 0.0,
            "final_equity_signal_basket": INITIAL_CAPITAL,
        }
    rets = trades["return_pct"].astype(float)
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if not losses.empty and abs(losses.sum()) > 1e-12 else float("inf")
    equity = build_signal_basket_curve(trades)
    dd = equity["equity"] / equity["equity"].cummax() - 1.0
    if len(equity) > 1:
        years = max((equity["date"].iloc[-1] - equity["date"].iloc[0]).days / 365.25, 1.0 / 365.25)
        annual = float((equity["equity"].iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0)
    else:
        annual = 0.0
    return {
        "strategy": strategy_name,
        "trade_count": int(len(trades)),
        "win_rate": float((rets > 0).mean()),
        "avg_trade_return": float(rets.mean()),
        "median_trade_return": float(rets.median()),
        "avg_holding_days": float(trades["hold_days"].mean()),
        "profit_factor": profit_factor,
        "total_return_signal_basket": float(equity["equity"].iloc[-1] / INITIAL_CAPITAL - 1.0) if not equity.empty else 0.0,
        "annual_return_signal_basket": annual,
        "max_drawdown_signal_basket": float(dd.min()) if not dd.empty else 0.0,
        "final_equity_signal_basket": float(equity["equity"].iloc[-1]) if not equity.empty else INITIAL_CAPITAL,
    }


def aggregate_forward_metrics(forward_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in forward_df.groupby("strategy"):
        row: dict[str, Any] = {"strategy": strategy, "sample_count": int(len(g))}
        for horizon in FORWARD_HORIZONS:
            row[f"mfe_{horizon}d_mean"] = float(g[f"mfe_{horizon}d"].mean())
            row[f"mfe_{horizon}d_median"] = float(g[f"mfe_{horizon}d"].median())
            row[f"mae_{horizon}d_mean"] = float(g[f"mae_{horizon}d"].mean())
            row[f"close_ret_{horizon}d_mean"] = float(g[f"close_ret_{horizon}d"].mean())
            for level in HIT_LEVELS:
                key = str(level).replace(".", "")
                row[f"hit_up_{key}_{horizon}d_rate"] = float(g[f"hit_up_{key}_{horizon}d"].mean())
                row[f"hit_dn_{key}_{horizon}d_rate"] = float(g[f"hit_dn_{key}_{horizon}d"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_rows(grid_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in grid_df.groupby("strategy"):
        best_return = g.sort_values(
            ["total_return_signal_basket", "win_rate", "avg_trade_return"],
            ascending=[False, False, False],
        ).head(1).copy()
        best_return["selection_metric"] = "best_total_return"
        best_win = g.sort_values(
            ["win_rate", "avg_trade_return", "total_return_signal_basket"],
            ascending=[False, False, False],
        ).head(1).copy()
        best_win["selection_metric"] = "best_win_rate"
        best_avg = g.sort_values(
            ["avg_trade_return", "win_rate", "total_return_signal_basket"],
            ascending=[False, False, False],
        ).head(1).copy()
        best_avg["selection_metric"] = "best_avg_trade_return"
        rows.extend([best_return, best_win, best_avg])
    return pd.concat(rows, ignore_index=True)


def run(mode: str, output_dir: Path, signal_limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(output_dir, "loading_candidates")

    relaxed_selected = load_relaxed_selected()
    if relaxed_selected.empty:
        raise RuntimeError("relaxed_fusion 经过统一可交易日期过滤后为空，无法比较买点质量。")
    date_start = relaxed_selected["signal_date"].min()
    date_end = relaxed_selected["signal_date"].max()
    formal_selected = load_formal_selected(date_start, date_end)
    if formal_selected.empty:
        raise RuntimeError("formal_best 在 relaxed_fusion 对应日期区间内为空，无法比较买点质量。")
    formal_selected, relaxed_selected = restrict_to_common_signal_dates(formal_selected, relaxed_selected)
    if formal_selected.empty or relaxed_selected.empty:
        raise RuntimeError("formal_best 与 relaxed_fusion 没有共同 signal_date，无法进行公平比较。")

    if signal_limit > 0:
        relaxed_selected = relaxed_selected.sort_values(["signal_date", "source_score"], ascending=[False, False]).head(signal_limit).copy()
        formal_selected = formal_selected.sort_values(["signal_date", "source_score"], ascending=[False, False]).head(signal_limit).copy()
        formal_selected, relaxed_selected = restrict_to_common_signal_dates(formal_selected, relaxed_selected)
        if formal_selected.empty or relaxed_selected.empty:
            raise RuntimeError("signal_limit 截断后两边没有共同 signal_date，无法进行公平比较。")

    combined_selected = pd.concat([formal_selected, relaxed_selected], ignore_index=True, sort=False)
    combined_selected.to_csv(output_dir / "selected_candidates_raw.csv", index=False, encoding="utf-8-sig")

    exact_lookup, digit_lookup = build_file_code_lookup()
    normalized_codes = {
        normalize_trade_code(code, exact_lookup, digit_lookup)
        for code in combined_selected["code"].astype(str).unique()
    }
    codes = {code for code in normalized_codes if code}
    daily_map = build_daily_map(codes)

    formal_candidates = build_trade_candidates(formal_selected, daily_map, exact_lookup, digit_lookup)
    relaxed_candidates = build_trade_candidates(relaxed_selected, daily_map, exact_lookup, digit_lookup)
    all_candidates = formal_candidates + relaxed_candidates
    update_progress(
        output_dir,
        "candidates_ready",
        formal_count=len(formal_candidates),
        relaxed_count=len(relaxed_candidates),
        date_start=str(date_start.date()),
        date_end=str(date_end.date()),
    )

    candidate_rows = [
        {
            "strategy": c.strategy,
            "code": c.code,
            "signal_idx": c.signal_idx,
            "signal_date": c.signal_date,
            "entry_idx": c.entry_idx,
            "entry_date": c.entry_date,
            "entry_price": c.entry_price,
            "signal_low": c.signal_low,
            "score": c.score,
        }
        for c in all_candidates
    ]
    pd.DataFrame(candidate_rows).to_csv(output_dir / "selected_candidates.csv", index=False, encoding="utf-8-sig")

    forward_rows = []
    for candidate in all_candidates:
        daily = daily_map.get(candidate.code)
        if daily is None or daily.empty:
            continue
        forward_rows.append(compute_forward_quality(candidate, daily))
    forward_df = pd.DataFrame(forward_rows)
    forward_df.to_csv(output_dir / "forward_quality_detail.csv", index=False, encoding="utf-8-sig")
    forward_summary = aggregate_forward_metrics(forward_df)
    forward_summary.to_csv(output_dir / "forward_quality_summary.csv", index=False, encoding="utf-8-sig")
    update_progress(output_dir, "forward_ready", rows=len(forward_df))

    grid_rows = []
    best_equity_files = []
    for strategy in ("formal_best", "relaxed_fusion"):
        strategy_candidates = [c for c in all_candidates if c.strategy == strategy]
        for tp_pct in TP_LIST:
            for sl_mult in SL_MULT_LIST:
                for hold_days in MAX_HOLD_LIST:
                    trade_rows = []
                    for candidate in strategy_candidates:
                        daily = daily_map.get(candidate.code)
                        if daily is None or daily.empty:
                            continue
                        trade = simulate_trade(candidate, daily, tp_pct, sl_mult, hold_days)
                        trade_rows.append(trade)
                    trade_df = pd.DataFrame(trade_rows)
                    summary = summarize_trades(trade_df, strategy)
                    summary.update(
                        {
                            "take_profit_pct": tp_pct,
                            "stop_loss_multiplier": sl_mult,
                            "max_hold_days": hold_days,
                            "mode": "next_open_after_trigger",
                        }
                    )
                    grid_rows.append(summary)
                    equity_df = build_signal_basket_curve(trade_df)
                    safe_name = f"{strategy}_tp{tp_pct:.3f}_sl{sl_mult:.3f}_hold{hold_days}".replace(".", "")
                    trade_df.to_csv(output_dir / f"trades_{safe_name}.csv", index=False, encoding="utf-8-sig")
                    equity_df.to_csv(output_dir / f"equity_{safe_name}.csv", index=False, encoding="utf-8-sig")
                    best_equity_files.append({"strategy": strategy, "tp": tp_pct, "sl": sl_mult, "hold": hold_days, "trade_count": len(trade_df)})
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(output_dir / "grid_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best_equity_files).to_csv(output_dir / "equity_file_index.csv", index=False, encoding="utf-8-sig")
    best_rows = select_best_rows(grid_df)
    best_rows.to_csv(output_dir / "best_by_metric.csv", index=False, encoding="utf-8-sig")
    update_progress(output_dir, "grid_ready", rows=len(grid_df))

    summary = {
        "mode": mode,
        "date_range": {
            "signal_start": str(date_start.date()),
            "signal_end": str(date_end.date()),
        },
        "candidate_counts": {
            "formal_best": len(formal_candidates),
            "relaxed_fusion": len(relaxed_candidates),
        },
        "forward_quality_leader_by_hit_up_003_5d": None,
        "forward_quality_leader_by_mfe_5d_mean": None,
        "best_total_return": {},
        "best_win_rate": {},
        "best_avg_trade_return": {},
        "notes": [
            "这轮实验只比较买点质量，统一使用次日开盘买入、买入当日不能卖、触发后次日开盘执行。",
            "formal_best 使用正式 BRICK baseline 的已选信号，relaxed_fusion 使用 BRICK 综合实验 final_test 冠军的已选信号。",
            "为了避免混入口径，这轮只在两者重叠的 final_test 时间区间内对比。",
        ],
    }
    if not forward_summary.empty:
        hit_col = "hit_up_003_5d_rate"
        mfe_col = "mfe_5d_mean"
        summary["forward_quality_leader_by_hit_up_003_5d"] = forward_summary.sort_values(hit_col, ascending=False).iloc[0].to_dict()
        summary["forward_quality_leader_by_mfe_5d_mean"] = forward_summary.sort_values(mfe_col, ascending=False).iloc[0].to_dict()
    for metric_name in ("best_total_return", "best_win_rate", "best_avg_trade_return"):
        row = best_rows[best_rows["selection_metric"] == metric_name].sort_values(
            ["strategy", "take_profit_pct", "stop_loss_multiplier", "max_hold_days"]
        )
        summary[metric_name] = row.to_dict(orient="records")

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(output_dir, "finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="BRICK formal_best vs relaxed融合冠军 买点质量统一对比")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--signal-limit", type=int, default=0, help="每个策略保留前N个信号，仅用于smoke")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULT_ROOT / f"brick_buypoint_quality_compare_v1_{args.mode}_{ts}"

    signal_limit = args.signal_limit
    if args.mode == "smoke" and signal_limit <= 0:
        signal_limit = 200

    run(args.mode, output_dir, signal_limit)


if __name__ == "__main__":
    main()
