from __future__ import annotations

from utils.backtest_pipeline.validators.base import BaseValidator


class RollingWindowValidator(BaseValidator):
    strategy_family = "brick"
    name = "validator.rolling_window"
