from __future__ import annotations

from utils.backtest_pipeline.exits.base import BaseExitModule


class FixedTakeProfitExit(BaseExitModule):
    name = "exit.fixed_tp"


class ModelOnlyExit(BaseExitModule):
    name = "exit.model_only"


class ModelPlusTakeProfitExit(BaseExitModule):
    name = "exit.model_plus_tp"


class PartialTakeProfitExit(BaseExitModule):
    """预留给 10% 卖一半、20% 再卖一半这类分批卖。"""

    name = "exit.partial_tp"
