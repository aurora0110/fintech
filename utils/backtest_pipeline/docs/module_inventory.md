# Backtest Pipeline 模块清单

## 根目录

### [__init__.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/__init__.py)
- 对外导出基础配置类型。

### [types.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/types.py)
- 定义 pipeline 的核心数据结构：
  - `DataConfig`
  - `StrategyConfig`
  - `RankerConfig`
  - `ExitConfig`
  - `AccountPolicyConfig`
  - `PipelineConfig`
  - `CandidateRecord`

### [registry.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/registry.py)
- 定义统一注册表：
  - `DATA_INPUT_REGISTRY`
  - `CANDIDATE_POOL_REGISTRY`
  - `CONFIRMER_REGISTRY`
  - `RANKER_REGISTRY`
  - `EXIT_REGISTRY`
  - `ACCOUNT_POLICY_REGISTRY`

### [catalog.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/catalog.py)
- 内置模块注册入口。
- 当前负责把 B1/B2/B3/单针/BRICK 的候选池、确认因子和通用排序器/卖法注册进去。

### [compatibility.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/compatibility.py)
- 定义每个策略家族当前“真实可跑”的实验矩阵约束。
- 包括：
  - 候选池
  - 确认因子
  - 排序器
  - `topN`
  - 卖法参数

### [experiment_matrix.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/experiment_matrix.py)
- 负责：
  - 生成实验矩阵
  - 读取机器可读实验账本
  - 输出已做/未做覆盖报告

### [runner.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/runner.py)
- 当前统一入口骨架。
- 负责：
  - 读取配置
  - 注册模块
  - 加载数据
  - 输出当前 pipeline 的结构摘要
  - `materialize` 真实生成候选与 `topN`
  - `backtest` 调用统一账户层执行器落交易/净值/summary

### [account_runner.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/account_runner.py)
- 统一账户层执行器（最小可运行版）
- 负责：
  - 复用 pipeline 配置生成 `selected_candidates`
  - 调用 `core.engine.BacktestEngine`
  - 输出：
    - `selected_candidates.csv`
    - `equity_curve.csv`
    - `daily_returns.csv`
    - `trades.csv`
    - `summary.json`

### [strategy_adapter.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/strategy_adapter.py)
- 把 pipeline 候选结果桥接成 `core.engine` 可执行的 `SignalStrategy`
- 当前支持：
  - 通用止损
  - 多空线下止损减半
  - `fixed_tp`
  - `model_only`
  - `model_plus_tp`
  - 最大持有天数

## inputs

### [inputs/main_style_loader.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/inputs/main_style_loader.py)
- 参考 `main.py` 和 `core/data_loader.py`
- 负责：
  - 统一读取行情目录
  - 应用默认剔除区间
  - 输出 `stock_data + all_dates`

## candidate_pools

### [candidate_pools/base.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/candidate_pools/base.py)
- 候选池基类和上下文

### [candidate_pools/b1.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/candidate_pools/b1.py)
- `B1LowCrossPool`
- `B1TxtConfirmedPool`

### [candidate_pools/b2.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/candidate_pools/b2.py)
- `B2MainPool`
- `B2Type1Pool`
- `B2Type4Pool`
- 当前状态：
  - `b2.main`：基于 `b2filter.add_features()` 的主候选池
  - `b2.type1`：贴近多空线 + `J` 进入 20 日 10% 低位
  - `b2.type4`：趋势线上穿多空线后的第一次回踩趋势线

### [candidate_pools/b3.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/candidate_pools/b3.py)
- `B3FollowThroughPool`
- `B3WeeklyAlignedPool`
- 当前状态：
  - `b3.follow_through`：主承接池
  - `b3.weekly_aligned`：在主承接基础上加入周线一致性背景加分

### [candidate_pools/pin.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/candidate_pools/pin.py)
- `PinTrendWashPool`
- `PinStructureSupportPool`
- 当前状态：
  - `pin.trend_wash`：A/B/C 综合单针池
  - `pin.structure_support`：收敛到 `C型(结构支撑)` 的单针子池

### [candidate_pools/brick.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/candidate_pools/brick.py)
- `BrickMainPool`
- `BrickFormalBestPool`

## confirmers

### [confirmers/base.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/confirmers/base.py)
- 确认因子基类和上下文

### [confirmers/b1.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/confirmers/b1.py)
- `B1SemanticBonusConfirmer`

### [confirmers/b2.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/confirmers/b2.py)
- `B2StartupQualityConfirmer`
- 当前状态：
  - 已按 `收盘质量 / 量能质量 / 趋势领先 / J空间 / type1/type4 bonus` 加分

### [confirmers/b3.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/confirmers/b3.py)
- `B3FollowThroughConfirmer`
- 当前状态：
  - 已按 `ret1 / 振幅 / 缩量 / prev_b2 / 周线背景` 加分

### [confirmers/pin.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/confirmers/pin.py)
- `PinNeedleQualityConfirmer`
- 当前状态：
  - 已按 `A/B/C 子型 + along_trend_up / n_up_any / keyk_support_active` 加分

### [confirmers/brick.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/confirmers/brick.py)
- `BrickTurnQualityConfirmer`

## rankers

### [rankers/base.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/rankers/base.py)
- 排序器基类和上下文

### [rankers/generic.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/rankers/generic.py)
- `SimilarityRanker`
- `FactorDiscoveryRanker`
- `XGBoostRanker`
- `LightGBMRanker`
- `NaiveBayesRanker`
- `ReinforcementLearningRanker`
- `FusionRanker`

## exits

### [exits/base.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/exits/base.py)
- 卖出模块基类

### [exits/generic.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/exits/generic.py)
- `FixedTakeProfitExit`
- `ModelOnlyExit`
- `ModelPlusTakeProfitExit`
- `PartialTakeProfitExit`
- `BrickHalfTakeProfitThenGreenExit`

## portfolio

### [portfolio/base.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/portfolio/base.py)
- 账户层策略描述结构

### [portfolio/policies.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/portfolio/policies.py)
- 当前内置：
  - `EQUAL_WEIGHT_POLICY`

## configs

### [configs/b1_reference_pipeline.json](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/configs/b1_reference_pipeline.json)
- B1 参考配置

### [configs/b1_reference_pipeline_smoke.json](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/configs/b1_reference_pipeline_smoke.json)
- B1 小输入验证配置

### [configs/b3_reference_pipeline.json](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/configs/b3_reference_pipeline.json)
- B3 参考配置

### [configs/pin_reference_pipeline.json](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/configs/pin_reference_pipeline.json)
- 单针参考配置

### [configs/brick_formal_best_pipeline.json](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/configs/brick_formal_best_pipeline.json)
- BRICK 历史正式最优策略配置

### [configs/brick_formal_best_pipeline_smoke.json](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/configs/brick_formal_best_pipeline_smoke.json)
- BRICK 历史正式最优策略小样本配置

## docs

### [docs/pipeline_design.md](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/pipeline_design.md)
- 解释为什么要按策略拆家族，以及模块化原则

### [docs/usage.md](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/usage.md)
- 使用方法和当前内置模块说明

### [docs/module_inventory.md](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/module_inventory.md)
- 这份模块清单文档

### [docs/experiment_ledger.md](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/experiment_ledger.md)
- 人读版实验纪要

### [docs/experiment_ledger.json](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/experiment_ledger.json)
- 机器可读实验账本
- 用于和实验矩阵精确比对
