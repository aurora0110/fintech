from __future__ import annotations

from utils.backtest_pipeline.exits.base import BaseExitModule


class BrickFixedTakeProfitGridExit(BaseExitModule):
    strategy_family = "brick"
    name = "exit.fixed_tp_grid"


class BrickPartialTakeProfitGridExit(BaseExitModule):
    strategy_family = "brick"
    name = "exit.partial_tp_grid"
