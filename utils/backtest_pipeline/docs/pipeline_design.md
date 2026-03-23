# Backtest Pipeline 设计说明

## 目标

把当前散落在 `main.py`、`utils/*.py`、`utils/tmp/*.py`、`utils/backtest/*.py` 的研究和回测能力，
收敛成一套可拼接、可枚举、可扩展的模块化 pipeline。

## 总体流程

1. 数据输入
2. 候选池生成
3. 确认因子加分/扣分
4. 排序器/模型打分
5. 卖出模块
6. 账户层执行
7. 结果审计与覆盖报告

## 核心原则

### 1. 按策略家族拆，不混策略语义

- `B1`
- `B2`
- `B3`
- `PIN`
- `BRICK`

每个家族都应该有自己的：
- 候选池模块
- 确认因子模块
- 可选排序器适配
- 可选卖法适配

不要把 `B1` 的主语义和 `B2` / `BRICK` 的主语义混成一个大池。

### 2. 候选池不是“一个策略一个文件”，而是“一个策略多个可命名池”

例如 `B1`：
- `b1.low_cross`
- `b1.txt_confirmed`
- 后续可以加：
  - `b1.trend_pullback`
  - `b1.longline_pullback`

也就是：
- 先按策略拆家族
- 再在策略内部定义多个候选池变体

### 3. 确认因子默认是加分/扣分，不轻易做硬过滤

例如：
- `关键K`
- `缩半量`
- `倍量柱`

在当前 B1 语义下应默认视为加分项，而不是独立买点主类型。

### 4. 新方法永远作为模块追加，不覆盖旧方法

例如模型层后续可继续增加：
- `ranker.naive_bayes`
- `ranker.catboost`
- `ranker.reinforcement_learning`

新增后只需要：
1. 加一个模块文件
2. 在 `catalog.py` 注册

不应该再回到“开一堆新脚本、手改老脚本”的方式。

## 后续最优先落地顺序

1. `B1` 先做成可运行样板
2. 把当前冠军链映射成标准 pipeline 配置
3. 做实验矩阵生成器
4. 再把 `B2 / BRICK / PIN / B3` 逐步接入

## 本版骨架已包含

- 基础配置数据结构：`types.py`
- 模块注册表：`registry.py`
- 输入模块：`inputs/main_style_loader.py`
- 候选池骨架：`candidate_pools/`
- 确认因子骨架：`confirmers/`
- 排序器骨架：`rankers/`
- 卖出模块骨架：`exits/`
- 默认模块注册：`catalog.py`
- 示例配置：`configs/b1_reference_pipeline.json`

## 当前仍未完成

这版先落“架构骨架”，还没有把现有 B1/B2/BRICK 实验逻辑完整迁移进来。

下一步应优先补：
- pipeline 运行器
- 实验矩阵生成器
- 覆盖报告
- 当前冠军 B1 逻辑的真实模块化实现
