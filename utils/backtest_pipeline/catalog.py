from __future__ import annotations

from utils.backtest_pipeline.candidate_pools.b1 import B1LowCrossPool, B1TxtConfirmedPool
from utils.backtest_pipeline.candidate_pools.b2 import B2MainPool, B2Type1Pool, B2Type4Pool
from utils.backtest_pipeline.candidate_pools.b3 import B3FollowThroughPool, B3WeeklyAlignedPool
from utils.backtest_pipeline.candidate_pools.brick import BrickFormalBestPool, BrickMainPool
from utils.backtest_pipeline.candidate_pools.pin import PinStructureSupportPool, PinTrendWashPool
from utils.backtest_pipeline.confirmers.b1 import B1SemanticBonusConfirmer
from utils.backtest_pipeline.confirmers.b2 import B2StartupQualityConfirmer
from utils.backtest_pipeline.confirmers.b3 import B3FollowThroughConfirmer
from utils.backtest_pipeline.confirmers.brick import BrickTurnQualityConfirmer
from utils.backtest_pipeline.confirmers.pin import PinNeedleQualityConfirmer
from utils.backtest_pipeline.exits.generic import (
    BrickHalfTakeProfitThenGreenExit,
    FixedTakeProfitExit,
    ModelOnlyExit,
    ModelPlusTakeProfitExit,
    PartialTakeProfitExit,
)
from utils.backtest_pipeline.rankers.generic import (
    FactorDiscoveryRanker,
    FusionRanker,
    LightGBMRanker,
    NaiveBayesRanker,
    ReinforcementLearningRanker,
    SimilarityRanker,
    XGBoostRanker,
)
from utils.backtest_pipeline.registry import (
    ACCOUNT_POLICY_REGISTRY,
    CANDIDATE_POOL_REGISTRY,
    CONFIRMER_REGISTRY,
    EXIT_REGISTRY,
    RANKER_REGISTRY,
)


def register_builtin_modules() -> None:
    if CANDIDATE_POOL_REGISTRY.names():
        return

    CANDIDATE_POOL_REGISTRY.register("b1.low_cross", "b1", B1LowCrossPool, "B1 主冠军低位回踩池")
    CANDIDATE_POOL_REGISTRY.register("b1.txt_confirmed", "b1", B1TxtConfirmedPool, "B1 文本模板确认池")
    CANDIDATE_POOL_REGISTRY.register("b2.main", "b2", B2MainPool, "B2 主桥接候选池")
    CANDIDATE_POOL_REGISTRY.register("b2.type1", "b2", B2Type1Pool, "B2 第一类候选池")
    CANDIDATE_POOL_REGISTRY.register("b2.type4", "b2", B2Type4Pool, "B2 第四类候选池")
    CANDIDATE_POOL_REGISTRY.register("b3.follow_through", "b3", B3FollowThroughPool, "B3 承接确认候选池")
    CANDIDATE_POOL_REGISTRY.register("b3.weekly_aligned", "b3", B3WeeklyAlignedPool, "B3 周线共振候选池")
    CANDIDATE_POOL_REGISTRY.register("pin.trend_wash", "pin", PinTrendWashPool, "单针趋势洗盘候选池")
    CANDIDATE_POOL_REGISTRY.register("pin.structure_support", "pin", PinStructureSupportPool, "单针结构支撑候选池")
    CANDIDATE_POOL_REGISTRY.register("brick.main", "brick", BrickMainPool, "砖型主候选池")
    CANDIDATE_POOL_REGISTRY.register("brick.formal_best", "brick", BrickFormalBestPool, "BRICK 历史最优正式候选池")

    CONFIRMER_REGISTRY.register("b1.semantic_bonus", "b1", B1SemanticBonusConfirmer, "B1 关键K/缩半量/倍量柱加分")
    CONFIRMER_REGISTRY.register("b2.startup_quality", "b2", B2StartupQualityConfirmer, "B2 启动质量确认")
    CONFIRMER_REGISTRY.register("b3.follow_through_quality", "b3", B3FollowThroughConfirmer, "B3 小涨缩量承接确认")
    CONFIRMER_REGISTRY.register("pin.needle_quality", "pin", PinNeedleQualityConfirmer, "单针下影线/量能/结构确认")
    CONFIRMER_REGISTRY.register("brick.turn_quality", "brick", BrickTurnQualityConfirmer, "砖型反包质量确认")

    RANKER_REGISTRY.register("ranker.similarity", "generic", SimilarityRanker, "纯相似度排序")
    RANKER_REGISTRY.register("ranker.factor_discovery", "generic", FactorDiscoveryRanker, "自动特征发现因子排序")
    RANKER_REGISTRY.register("ranker.xgboost", "generic", XGBoostRanker, "XGBoost 排序")
    RANKER_REGISTRY.register("ranker.lightgbm", "generic", LightGBMRanker, "LightGBM 排序")
    RANKER_REGISTRY.register("ranker.naive_bayes", "generic", NaiveBayesRanker, "贝叶斯排序")
    RANKER_REGISTRY.register("ranker.reinforcement_learning", "generic", ReinforcementLearningRanker, "强化学习排序占位")
    RANKER_REGISTRY.register("ranker.fusion", "generic", FusionRanker, "相似度/因子/模型融合排序")

    EXIT_REGISTRY.register("exit.fixed_tp", "generic", FixedTakeProfitExit, "固定止盈")
    EXIT_REGISTRY.register("exit.model_only", "generic", ModelOnlyExit, "模型卖出")
    EXIT_REGISTRY.register("exit.model_plus_tp", "generic", ModelPlusTakeProfitExit, "模型+固定止盈")
    EXIT_REGISTRY.register("exit.partial_tp", "generic", PartialTakeProfitExit, "分批固定止盈")
    EXIT_REGISTRY.register("exit.brick_half_tp_then_green", "brick", BrickHalfTakeProfitThenGreenExit, "BRICK 3.5%半仓止盈+转绿卖剩余")

    ACCOUNT_POLICY_REGISTRY.register(
        "portfolio.equal_weight",
        "generic",
        lambda: None,
        "等权组合占位，后续由账户层执行器消费。",
    )
