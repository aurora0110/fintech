from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


BASE_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/run_momentum_tail_experiment.py")
RANKING_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest/compare_momentum_tail_ranking_models.py")
DEFAULT_DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
DEFAULT_RESULT_ROOT = Path("/Users/lidongyang/Desktop/Qstrategy/results")
TOP_N = 10
PCT_RANK_THRESHOLD = 0.50
EPS = 1e-12


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_PATH, "brick_green_exit_base")
ranking = load_module(RANKING_PATH, "brick_green_exit_ranking")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **extra) -> None:
    payload = {"stage": stage}
    payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def load_feature_map_limited(data_dir: Path, file_limit: int) -> Dict[str, pd.DataFrame]:
    feature_map: Dict[str, pd.DataFrame] = {}
    files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {".csv", ".txt"}])
    if file_limit > 0:
        files = files[:file_limit]
    total = len(files)
    for idx, path in enumerate(files, 1):
        df = base.load_one_csv(str(path))
        if df is None:
            continue
        code = str(df["code"].iloc[0])
        feature_map[code] = base.build_feature_df(df)
        if idx % 100 == 0 or idx == total:
            print(f"BRICK 对比特征加载进度: {idx}/{total}")
    return feature_map


def build_selected_signal_df(feature_map: Dict[str, pd.DataFrame], combo) -> pd.DataFrame:
    rows: List[dict] = []
    for code, df in feature_map.items():
        df2 = ranking.add_long_line(df)
        mask_a = df2["pattern_a"] & (df2["rebound_ratio"] >= combo.rebound_threshold)
        mask_b = df2["pattern_b"] & (df2["rebound_ratio"] >= 1.0)
        mask = (
            df2["signal_base"]
            & (df2["ret1"] <= combo.gain_limit)
            & (mask_a | mask_b)
            & (df2["trend_line"] > df2["long_line"])
        )
        signal_idxs = np.flatnonzero(mask.to_numpy())
        for signal_idx in signal_idxs:
            signal_idx = int(signal_idx)
            row = {
                "code": code,
                "signal_idx": signal_idx,
                "signal_date": df2.at[signal_idx, "date"],
                "pattern_a": bool(df2.at[signal_idx, "pattern_a"]),
                "pattern_b": bool(df2.at[signal_idx, "pattern_b"]),
                "trend_line": float(df2.at[signal_idx, "trend_line"]),
                "long_line": float(df2.at[signal_idx, "long_line"]),
                "ret1": float(df2.at[signal_idx, "ret1"]),
                "brick_red_len": float(df2.at[signal_idx, "brick_red_len"]),
                "brick_green_len_prev": float(df2["brick_green_len"].shift(1).iloc[signal_idx])
                if signal_idx >= 1 and pd.notna(df2["brick_green_len"].shift(1).iloc[signal_idx])
                else np.nan,
                "rebound_ratio": float(df2.at[signal_idx, "rebound_ratio"]),
                "signal_vs_ma5": float(df2.at[signal_idx, "signal_vs_ma5"]),
                "pullback_avg_vol": float(df2.at[signal_idx, "pullback_avg_vol"]),
                "up_leg_avg_vol": float(df2.at[signal_idx, "up_leg_avg_vol"]),
                "close": float(df2.at[signal_idx, "close"]),
                "open": float(df2.at[signal_idx, "open"]),
                "high": float(df2.at[signal_idx, "high"]),
                "low": float(df2.at[signal_idx, "low"]),
                "entry_low_for_trade": float(df2.at[signal_idx, "low"]),
            }
            row.update(ranking.compute_signal_features(df2, signal_idx))
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    signal_df = pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)
    signal_df = ranking.assign_rank_scores(signal_df)
    signal_df["sort_score"] = ranking.build_sort_score(signal_df, "shrink_focus")
    signal_df["daily_pct_rank"] = signal_df.groupby("signal_date")["sort_score"].rank(pct=True, method="first")
    signal_df = signal_df[signal_df["daily_pct_rank"] >= PCT_RANK_THRESHOLD].copy()
    signal_df = signal_df.sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True])
    signal_df = signal_df.groupby("signal_date", group_keys=False).head(TOP_N).reset_index(drop=True)
    return signal_df


def simulate_trade_green_exit(df: pd.DataFrame, signal_idx: int) -> Optional[dict]:
    n = len(df)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None
    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    exit_idx: Optional[int] = None
    exit_reason = "end_of_data_close"
    exit_price: Optional[float] = None

    # 买入当日不能卖，所以从 entry_idx + 1 开始检查是否转绿。
    for j in range(entry_idx + 1, n):
        if bool(df.at[j, "brick_green"]):
            if j + 1 < n:
                exit_idx = j + 1
                exit_price = float(df.at[exit_idx, "open"])
                exit_reason = "brick_turn_green_next_open"
            else:
                exit_idx = j
                exit_price = float(df.at[j, "close"])
                exit_reason = "brick_turn_green_last_close"
            break

    if exit_idx is None:
        exit_idx = n - 1
        exit_price = float(df.at[exit_idx, "close"])
        exit_reason = "end_of_data_close"

    if not np.isfinite(exit_price) or exit_price <= 0:
        return None

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
        "signal_low": float(df.at[signal_idx, "low"]),
        "pattern_a": bool(df.at[signal_idx, "pattern_a"]),
        "pattern_b": bool(df.at[signal_idx, "pattern_b"]),
    }


def build_trade_df_for_exit(
    feature_map: Dict[str, pd.DataFrame],
    selected_signal_df: pd.DataFrame,
    combo,
    exit_mode: str,
) -> pd.DataFrame:
    trades: List[dict] = []
    total = len(selected_signal_df)
    for idx, row in enumerate(selected_signal_df.itertuples(index=False), 1):
        df = feature_map[row.code]
        if exit_mode == "baseline":
            trade = base.simulate_trade(df, int(row.signal_idx), combo)
        elif exit_mode == "green_exit":
            trade = simulate_trade_green_exit(df, int(row.signal_idx))
        else:
            raise ValueError(f"未知卖出模式: {exit_mode}")
        if trade is None:
            continue
        trade["code"] = row.code
        trade["sort_score"] = float(row.sort_score)
        trade["rebound_ratio"] = float(row.rebound_ratio)
        trade["signal_vs_ma5"] = float(row.signal_vs_ma5)
        trade["ret1"] = float(row.ret1)
        trades.append(trade)
        if idx % 500 == 0 or idx == total:
            print(f"BRICK {exit_mode} 交易回放进度: {idx}/{total}")
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades).sort_values(["signal_date", "code"]).reset_index(drop=True)


def summarize_trade_and_portfolio(name: str, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    if trade_df.empty:
        return {
            "strategy": name,
            "trade_count": 0,
            "avg_trade_return": np.nan,
            "success_rate": np.nan,
            "avg_holding_days": np.nan,
            "profit_factor": np.nan,
            "max_consecutive_failures": np.nan,
            "annual_return_signal_basket": np.nan,
            "max_drawdown_signal_basket": np.nan,
            "final_equity_signal_basket": np.nan,
            "equity_days_signal_basket": 0,
        }
    row = {
        "strategy": name,
        "trade_count": int(len(trade_df)),
        "avg_trade_return": float(trade_df["ret"].mean()),
        "success_rate": float(trade_df["success"].mean()),
        "avg_holding_days": float(trade_df["holding_days"].mean()),
        "profit_factor": float(calc_profit_factor(trade_df["ret"])),
    }
    row["max_consecutive_failures"] = int(base.max_consecutive_failures(trade_df["success"].tolist()))
    row.update(base.compute_equity_metrics(portfolio_df))
    row["annual_return_signal_basket"] = row.pop("annual_return")
    row["max_drawdown_signal_basket"] = row.pop("max_drawdown")
    row["final_equity_signal_basket"] = row.pop("final_equity")
    row["equity_days_signal_basket"] = row.pop("equity_days")
    return row


def build_comparison_df(summary_rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return df
    return df[
        [
            "strategy",
            "trade_count",
            "avg_trade_return",
            "success_rate",
            "avg_holding_days",
            "profit_factor",
            "max_consecutive_failures",
            "annual_return_signal_basket",
            "max_drawdown_signal_basket",
            "final_equity_signal_basket",
            "equity_days_signal_basket",
        ]
    ]


def calc_profit_factor(ret_series: pd.Series) -> float:
    pos = float(ret_series[ret_series > 0].sum())
    neg = float(ret_series[ret_series < 0].sum())
    if abs(neg) < EPS:
        return np.nan
    return pos / abs(neg)


def run_chain(data_dir: Path, result_dir: Path, file_limit: int) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "starting", data_dir=str(data_dir), file_limit=file_limit)

    combo = base.Combo(
        rebound_threshold=1.2,
        gain_limit=0.08,
        take_profit=0.03,
        stop_mode="entry_low_x_0.99",
    )
    write_json(
        result_dir / "experiment_config.json",
        {
            "data_dir": str(data_dir),
            "file_limit": file_limit,
            "buy_logic": {
                "patterns": ["3绿1红", "3绿1红1绿1红"],
                "rebound_threshold_3g1r": combo.rebound_threshold,
                "rebound_threshold_3g1r1g1r": 1.0,
                "gain_limit": combo.gain_limit,
                "trend_filter": "趋势线 > 多空线",
                "signal_vs_ma5_filter": "1.3~2.2",
                "pullback_shrinking": True,
                "ranking_model": "shrink_focus",
                "rank_cutoff": "前50%",
                "top_n": TOP_N,
            },
            "baseline_exit": {
                "take_profit": combo.take_profit,
                "stop_mode": combo.stop_mode,
                "holding_rule": "固定持有3天，到期次日开盘卖出",
                "entry": "signal次日开盘",
            },
            "green_exit": {
                "rule": "买入当日不能卖；买入后首个砖块转绿日出现后，次日开盘卖出",
                "stop_loss": "none",
            },
        },
    )

    feature_map = load_feature_map_limited(data_dir, file_limit=file_limit)
    update_progress(result_dir, "features_ready", stock_count=len(feature_map))

    selected_signal_df = build_selected_signal_df(feature_map, combo)
    selected_signal_df.to_csv(result_dir / "selected_signals.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "signals_ready", selected_signal_count=int(len(selected_signal_df)))

    baseline_trade_df = build_trade_df_for_exit(feature_map, selected_signal_df, combo, "baseline")
    baseline_trade_df.to_csv(result_dir / "baseline_trades.csv", index=False, encoding="utf-8-sig")
    baseline_portfolio_df = base.build_portfolio_curve(baseline_trade_df)
    baseline_portfolio_df.to_csv(result_dir / "baseline_portfolio.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "baseline_ready", baseline_trade_count=int(len(baseline_trade_df)))

    green_trade_df = build_trade_df_for_exit(feature_map, selected_signal_df, combo, "green_exit")
    green_trade_df.to_csv(result_dir / "green_exit_trades.csv", index=False, encoding="utf-8-sig")
    green_portfolio_df = base.build_portfolio_curve(green_trade_df)
    green_portfolio_df.to_csv(result_dir / "green_exit_portfolio.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "green_exit_ready", green_trade_count=int(len(green_trade_df)))

    baseline_summary = summarize_trade_and_portfolio("baseline_tp3_sl099_hold3", baseline_trade_df, baseline_portfolio_df)
    green_summary = summarize_trade_and_portfolio("green_exit_next_open_no_sl", green_trade_df, green_portfolio_df)
    comparison_df = build_comparison_df([baseline_summary, green_summary])
    comparison_df.to_csv(result_dir / "comparison.csv", index=False, encoding="utf-8-sig")

    summary = {
        "baseline": baseline_summary,
        "green_exit": green_summary,
        "diff_green_minus_baseline": {
            "trade_count": int(green_summary["trade_count"] - baseline_summary["trade_count"]),
            "avg_trade_return": float(green_summary["avg_trade_return"] - baseline_summary["avg_trade_return"]),
            "success_rate": float(green_summary["success_rate"] - baseline_summary["success_rate"]),
            "avg_holding_days": float(green_summary["avg_holding_days"] - baseline_summary["avg_holding_days"]),
            "annual_return_signal_basket": float(
                green_summary["annual_return_signal_basket"] - baseline_summary["annual_return_signal_basket"]
            ),
            "max_drawdown_signal_basket": float(
                green_summary["max_drawdown_signal_basket"] - baseline_summary["max_drawdown_signal_basket"]
            ),
            "final_equity_signal_basket": float(
                green_summary["final_equity_signal_basket"] - baseline_summary["final_equity_signal_basket"]
            ),
        },
        "notes": [
            "本轮比较口径使用 BRICK 历史信号日篮子近似账户法，与 run_momentum_tail_experiment.py 一致。",
            "买点池、排序、前50%过滤和 top10 完全一致，只替换卖出规则。",
            "新卖法不保留止损，若直到样本结束都未转绿，则最后一个交易日收盘平仓。",
        ],
    }
    write_json(result_dir / "summary.json", summary)
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--file-limit", type=int, default=None)
    parser.add_argument("--result-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        suffix = "smoke" if args.mode == "smoke" else "full"
        result_dir = DEFAULT_RESULT_ROOT / f"brick_green_exit_compare_v1_{suffix}_20260323"
    file_limit = args.file_limit if args.file_limit is not None else (300 if args.mode == "smoke" else 0)
    run_chain(data_dir=data_dir, result_dir=result_dir, file_limit=file_limit)


if __name__ == "__main__":
    main()
