from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.backtest_pipeline.catalog import register_builtin_modules
from utils.backtest_pipeline.candidate_pools.base import CandidatePoolContext
from utils.backtest_pipeline.confirmers.base import ConfirmerContext
from utils.backtest_pipeline.inputs.main_style_loader import load_market_data
from utils.backtest_pipeline.rankers.base import RankerContext
from utils.backtest_pipeline.registry import CANDIDATE_POOL_REGISTRY, CONFIRMER_REGISTRY, EXIT_REGISTRY, RANKER_REGISTRY
from utils.backtest_pipeline.types import (
    AccountPolicyConfig,
    DataConfig,
    ExitConfig,
    PipelineConfig,
    RankerConfig,
    StrategyConfig,
)


def _build_config(payload: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
        name=payload["name"],
        data=DataConfig(**payload["data"]),
        strategy=StrategyConfig(**payload["strategy"]),
        ranker=RankerConfig(**payload["ranker"]),
        exit=ExitConfig(**payload["exit"]),
        account=AccountPolicyConfig(**payload["account"]),
        output_dir=Path(payload["output_dir"]) if payload.get("output_dir") else None,
    )


def load_pipeline_config(config_path: str) -> PipelineConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return _build_config(payload)


def describe_pipeline(config: PipelineConfig) -> dict[str, Any]:
    register_builtin_modules()
    candidate_item = CANDIDATE_POOL_REGISTRY.get(config.strategy.candidate_pool)
    confirmer_item = CONFIRMER_REGISTRY.get(config.strategy.confirmer) if config.strategy.confirmer else None
    ranker_item = RANKER_REGISTRY.get(config.ranker.name)
    exit_item = EXIT_REGISTRY.get(config.exit.name)
    candidate_cls = candidate_item.builder
    candidate_pool = candidate_cls()
    if getattr(candidate_pool, "requires_market_data", True):
        stock_data, all_dates = load_market_data(config.data)
        stock_count = len(stock_data)
        date_count = len(all_dates)
    else:
        stock_count = 0
        date_count = 0
    return {
        "pipeline_name": config.name,
        "data_dir": config.data.data_dir,
        "stock_count": stock_count,
        "date_count": date_count,
        "strategy_family": config.strategy.strategy_family,
        "candidate_pool": candidate_item.name,
        "candidate_pool_description": candidate_item.description,
        "confirmer": confirmer_item.name if confirmer_item else None,
        "ranker": ranker_item.name,
        "exit": exit_item.name,
        "account": asdict(config.account),
    }


def materialize_pipeline_candidates(config: PipelineConfig) -> dict[str, Any]:
    register_builtin_modules()
    candidate_cls = CANDIDATE_POOL_REGISTRY.get(config.strategy.candidate_pool).builder
    candidate_pool = candidate_cls()
    if getattr(candidate_pool, "requires_market_data", True):
        stock_data, all_dates = load_market_data(config.data)
    else:
        stock_data, all_dates = {}, []
    candidate_df = candidate_pool.generate(
        CandidatePoolContext(
            data_dir=config.data.data_dir,
            stock_data=stock_data,
            all_dates=all_dates,
            strategy_config=config.strategy,
        )
    )

    if config.strategy.confirmer:
        confirmer_cls = CONFIRMER_REGISTRY.get(config.strategy.confirmer).builder
        confirmer = confirmer_cls()
        candidate_df = confirmer.apply(
            ConfirmerContext(
                candidate_df=candidate_df,
                stock_data=stock_data,
                params=config.strategy.params,
            )
        )

    ranker_cls = RANKER_REGISTRY.get(config.ranker.name).builder
    ranker = ranker_cls()
    candidate_df = ranker.score(
        RankerContext(
            candidate_df=candidate_df,
            params=config.ranker.params,
        )
    )

    if "signal_date" in candidate_df.columns:
        candidate_df = candidate_df.sort_values(["signal_date", "rank_score"], ascending=[True, False]).copy()
        topn = int(config.ranker.top_n)
        selected = (
            candidate_df.groupby("signal_date", group_keys=False)
            .head(topn)
            .reset_index(drop=True)
        )
    else:
        selected = candidate_df.sort_values("rank_score", ascending=False).head(config.ranker.top_n).reset_index(drop=True)

    preview_cols = [
        c for c in [
            "code",
            "signal_date",
            "entry_date",
            "candidate_pool",
            "base_score",
            "confirmer_score",
            "rank_score",
            "discovery_factor_score",
            "template_hard_full_fusion_score",
            "xgb_full_score",
        ] if c in selected.columns
    ]
    return {
        "pipeline_name": config.name,
        "candidate_count": int(len(candidate_df)),
        "selected_count": int(len(selected)),
        "top_n": int(config.ranker.top_n),
        "preview": selected[preview_cols].head(10).to_dict(orient="records"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模块化 backtest pipeline 入口骨架")
    parser.add_argument("config", help="pipeline json 配置文件路径")
    parser.add_argument("--mode", choices=["describe", "materialize"], default="describe")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    if args.mode == "materialize":
        summary = materialize_pipeline_candidates(config)
    else:
        summary = describe_pipeline(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
