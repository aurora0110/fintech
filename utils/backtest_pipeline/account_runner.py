from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from core.engine import BacktestEngine, EngineConfig
from utils.backtest_pipeline.catalog import register_builtin_modules
from utils.backtest_pipeline.candidate_pools.base import CandidatePoolContext
from utils.backtest_pipeline.confirmers.base import ConfirmerContext
from utils.backtest_pipeline.inputs.main_style_loader import load_market_data
from utils.backtest_pipeline.rankers.base import RankerContext
from utils.backtest_pipeline.registry import CANDIDATE_POOL_REGISTRY, CONFIRMER_REGISTRY, RANKER_REGISTRY
from utils.backtest_pipeline.strategy_adapter import PipelineBacktestContext, PipelineSignalStrategy
from utils.backtest_pipeline.types import PipelineConfig


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _resolve_output_dir(config: PipelineConfig) -> Path:
    if config.output_dir is not None:
        return Path(config.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("/Users/lidongyang/Desktop/Qstrategy/results") / f"pipeline_backtest_{config.name}_{stamp}"


def materialize_selected_candidates(config: PipelineConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], list[pd.Timestamp]]:
    register_builtin_modules()
    stock_data, all_dates = load_market_data(config.data)

    candidate_cls = CANDIDATE_POOL_REGISTRY.get(config.strategy.candidate_pool).builder
    candidate_pool = candidate_cls()
    if getattr(candidate_pool, "requires_market_data", True):
        candidate_df = candidate_pool.generate(
            CandidatePoolContext(
                data_dir=config.data.data_dir,
                stock_data=stock_data,
                all_dates=all_dates,
                strategy_config=config.strategy,
            )
        )
    else:
        candidate_df = candidate_pool.generate(
            CandidatePoolContext(
                data_dir=config.data.data_dir,
                stock_data={},
                all_dates=[],
                strategy_config=config.strategy,
            )
        )

    if config.strategy.confirmer:
        confirmer_cls = CONFIRMER_REGISTRY.get(config.strategy.confirmer).builder
        confirmer = confirmer_cls()
        candidate_df = confirmer.apply(
            ConfirmerContext(candidate_df=candidate_df, stock_data=stock_data, params=config.strategy.params)
        )

    ranker_cls = RANKER_REGISTRY.get(config.ranker.name).builder
    ranker = ranker_cls()
    candidate_df = ranker.score(RankerContext(candidate_df=candidate_df, params=config.ranker.params))

    if not candidate_df.empty:
        valid_codes = set(stock_data.keys())
        candidate_df = candidate_df[candidate_df["code"].astype(str).isin(valid_codes)].copy()

        if "signal_date" in candidate_df.columns:
            keep_mask = []
            for row in candidate_df.itertuples(index=False):
                code = str(row.code)
                signal_date = pd.Timestamp(row.signal_date)
                keep_mask.append(signal_date in stock_data[code].index)
            candidate_df = candidate_df[pd.Series(keep_mask, index=candidate_df.index)].copy()

    if candidate_df.empty:
        if "rank_score" not in candidate_df.columns:
            candidate_df = candidate_df.copy()
            candidate_df["rank_score"] = pd.Series(dtype=float)
        selected = candidate_df.reset_index(drop=True)
    elif "signal_date" in candidate_df.columns and not candidate_df.empty:
        candidate_df = candidate_df.sort_values(["signal_date", "rank_score"], ascending=[True, False]).copy()
        selected = candidate_df.groupby("signal_date", group_keys=False).head(int(config.ranker.top_n)).reset_index(drop=True)
    else:
        selected = candidate_df.sort_values("rank_score", ascending=False).head(int(config.ranker.top_n)).reset_index(drop=True)

    return selected, stock_data, all_dates


def _trades_to_df(result) -> pd.DataFrame:
    if not result.trades:
        return pd.DataFrame(
            columns=["code", "entry_date", "exit_date", "entry_price", "exit_price", "shares", "pnl", "return_pct", "reason"]
        )
    rows = [
        {
            "code": item.code,
            "entry_date": item.entry_date,
            "exit_date": item.exit_date,
            "entry_price": item.entry_price,
            "exit_price": item.exit_price,
            "shares": item.shares,
            "pnl": item.pnl,
            "return_pct": item.return_pct,
            "reason": item.reason,
        }
        for item in result.trades
    ]
    return pd.DataFrame(rows)


def run_pipeline_backtest(config: PipelineConfig, result_dir: Path | None = None) -> dict[str, Any]:
    register_builtin_modules()
    output_dir = result_dir or _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "progress.json", {"stage": "starting", "pipeline": config.name})

    selected_df, stock_data, all_dates = materialize_selected_candidates(config)
    selected_df.to_csv(output_dir / "selected_candidates.csv", index=False, encoding="utf-8-sig")
    _write_json(
        output_dir / "progress.json",
        {
            "stage": "signals_ready",
            "candidate_count": int(len(selected_df)),
            "stock_count": int(len(stock_data)),
            "date_count": int(len(all_dates)),
        },
    )

    strategy = PipelineSignalStrategy(
        PipelineBacktestContext(
            candidate_df=selected_df,
            exit_config=config.exit,
            account_config=config.account,
        )
    )
    engine = BacktestEngine(
        EngineConfig(
            initial_capital=float(config.account.initial_capital),
            max_positions=int(config.account.max_positions),
        )
    )
    result = engine.run(strategy, stock_data, all_dates)

    equity_df = result.equity_curve.rename("equity").reset_index().rename(columns={"index": "date"})
    returns_df = result.daily_returns.rename("daily_return").reset_index().rename(columns={"index": "date"})
    trades_df = _trades_to_df(result)

    equity_df.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    returns_df.to_csv(output_dir / "daily_returns.csv", index=False, encoding="utf-8-sig")
    trades_df.to_csv(output_dir / "trades.csv", index=False, encoding="utf-8-sig")

    summary = {
        "pipeline": config.name,
        "candidate_pool": config.strategy.candidate_pool,
        "ranker": config.ranker.name,
        "exit": config.exit.name,
        "account": {
            "initial_capital": config.account.initial_capital,
            "max_positions": config.account.max_positions,
            "max_holding_days": config.account.max_holding_days,
            "default_stop_loss_pct": config.account.default_stop_loss_pct,
            "halve_stop_loss_below_long_line": config.account.halve_stop_loss_below_long_line,
        },
        "candidate_count": int(len(selected_df)),
        "trade_count": int(len(trades_df)),
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
        "replayable": True,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "progress.json", {"stage": "finished"})
    return {
        "result_dir": str(output_dir),
        "summary": summary,
    }
