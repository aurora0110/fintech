"""模块化回测 pipeline 第一版骨架。

这层的目标不是立刻替换现有所有实验脚本，而是把：
- 数据输入
- 候选池
- 确认因子
- 排序/模型
- 卖出
- 账户层

收敛成稳定可注册、可拼接、可枚举的模块体系。
"""

from .types import (
    AccountPolicyConfig,
    CandidateRecord,
    DataConfig,
    ExitConfig,
    PipelineConfig,
    RankerConfig,
    StrategyConfig,
)

__all__ = [
    "AccountPolicyConfig",
    "CandidateRecord",
    "DataConfig",
    "ExitConfig",
    "PipelineConfig",
    "RankerConfig",
    "StrategyConfig",
]
