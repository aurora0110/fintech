from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from core.engine import BaseSignalStrategy
from core.models import Position, Signal
from utils.backtest_pipeline.types import AccountPolicyConfig, ExitConfig


@dataclass
class PipelineBacktestContext:
    candidate_df: pd.DataFrame
    exit_config: ExitConfig
    account_config: AccountPolicyConfig


class PipelineSignalStrategy(BaseSignalStrategy):
    """把 pipeline 选股结果桥接到 core.engine 的统一账户层引擎。"""

    name = "pipeline.generic"

    def __init__(self, context: PipelineBacktestContext):
        self.context = context

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return stock_data

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        if self.context.candidate_df.empty:
            return signal_map

        valid_codes = set(stock_data.keys())
        for row in self.context.candidate_df.itertuples(index=False):
            code = str(row.code)
            if code not in valid_codes:
                continue
            signal_date = pd.Timestamp(row.signal_date)
            metadata = row._asdict()
            signal = Signal(
                date=signal_date,
                code=code,
                score=float(metadata.get("rank_score", metadata.get("base_score", 0.0)) or 0.0),
                reason=str(metadata.get("signal_type", metadata.get("candidate_pool", "pipeline"))),
                metadata=metadata,
            )
            signal_map.setdefault(signal_date, []).append(signal)
        return signal_map

    def build_position(self, signal: Signal, exec_row: pd.Series) -> Dict[str, float]:
        metadata = dict(signal.metadata)
        account = self.context.account_config
        exit_cfg = self.context.exit_config

        stop_price = metadata.get("stop_loss_price")
        if stop_price is not None:
            try:
                stop_price = float(stop_price)
            except Exception:
                stop_price = None
        if stop_price is None or pd.isna(stop_price):
            stop_pct = float(account.default_stop_loss_pct)
            long_line = exec_row.get("long_line")
            close_price = float(exec_row.get("close", exec_row.get("open", 0.0)) or 0.0)
            if (
                account.halve_stop_loss_below_long_line
                and long_line is not None
                and pd.notna(long_line)
                and close_price > 0
                and close_price < float(long_line)
            ):
                stop_pct *= 0.5
            open_price = float(exec_row.get("open", 0.0) or 0.0)
            stop_price = open_price * (1.0 - stop_pct) if open_price > 0 else None

        take_profit_price = None
        tp_pct = exit_cfg.params.get("take_profit_pct")
        if exit_cfg.name in {"exit.fixed_tp", "exit.model_plus_tp"} and tp_pct is not None:
            try:
                open_price = float(exec_row.get("open", 0.0) or 0.0)
                tp_pct = float(tp_pct)
                if open_price > 0 and tp_pct > 0:
                    take_profit_price = open_price * (1.0 + tp_pct)
            except Exception:
                take_profit_price = None

        out = {
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "max_holding_days": int(account.max_holding_days),
        }
        if "model_score_col" in exit_cfg.params:
            out["model_score_col"] = str(exit_cfg.params["model_score_col"])
        if "model_threshold" in exit_cfg.params:
            out["model_threshold"] = float(exit_cfg.params["model_threshold"])
        return out

    def should_exit(
        self,
        current_date: pd.Timestamp,
        position: Position,
        row: pd.Series,
        stock_df: pd.DataFrame,
    ) -> Optional[str]:
        # 先跑通统一引擎：固定止盈与最长持有完全支持；模型卖法支持读取行情行上的分数字段。
        base_reason = super().should_exit(current_date, position, row, stock_df)
        if base_reason is not None:
            return base_reason

        exit_cfg = self.context.exit_config
        if exit_cfg.name not in {"exit.model_only", "exit.model_plus_tp"}:
            return None

        score_col = position.metadata.get("model_score_col")
        threshold = position.metadata.get("model_threshold")
        if score_col is None or threshold is None:
            return None
        if score_col in row.index:
            try:
                score = float(row[score_col])
                if pd.notna(score) and score >= float(threshold):
                    return "model_exit"
            except Exception:
                return None
        return None
