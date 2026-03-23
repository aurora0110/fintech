from __future__ import annotations

from utils.backtest_pipeline.portfolio.base import PortfolioPolicy


EQUAL_WEIGHT_POLICY = PortfolioPolicy(
    name="portfolio.equal_weight",
    description="等权分配，可配合 max_positions / 单票上限。",
)
