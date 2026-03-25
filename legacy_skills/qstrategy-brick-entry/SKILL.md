---
name: qstrategy-brick-entry
description: 用于 Qstrategy 的 BRICK 买点定义与变体选择。当用户要使用、比较或修改 BRICK 买点时使用。
---

# BRICK 买点

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 用户要用 BRICK 选股
- 用户要比较 BRICK 不同买点变体
- 用户要做正式 baseline vs 研究冠军对比

## 可选变体

### `formal_best`

- 口径：**正式**
- 含义：当前正式 BRICK baseline
- 核心：
  - `趋势线 > 多空线`
  - `3绿1红 / 3绿1红1绿1红`
  - 量能区间
  - 反包质量
  - `shrink_focus` 排序
  - 前 `50%` 再取 `top10`

### `relaxed_similarity`

- 口径：**研究**
- 含义：BRICK relaxed 纯相似度冠军
- 当前参考：
  - `green4_enhance`
  - `cluster_100`
  - `len21`
  - `close_vol_concat`
  - `pipeline_corr_dtw`
  - `gate0.75`
  - `top10`

### `relaxed_full_fusion`

- 口径：**研究**
- 含义：BRICK relaxed 综合冠军
- 当前参考：
  - `relaxed_base`
  - `sample_300`
  - `len21`
  - `close_norm`
  - `pipeline_corr_dtw`
  - `RandomForest`
  - 融合权重 `0.4 / 0.2 / 0.4`
  - `top10`

### `green4_low_variant`

- 口径：**研究**
- 含义：把 `green4 + low层` 当增强条件的 BRICK 变体
- 适合做稳定性和成功率增强研究

## 输出要求

必须说明：

- 用的是哪个变体
- 该变体是正式还是研究口径
- 是否含相似度 / 因子 / 机器学习

## 使用方法

推荐直接这样说：

- “用 `qstrategy-brick-entry` 的 `formal_best` 变体”
- “用 `qstrategy-brick-entry` 的 `relaxed_full_fusion` 变体”
- “比较 `formal_best` 和 `relaxed_full_fusion` 的买点质量”

## 修改点

允许改：

- 候选池
- 排序方式
- 模板构建
- gate / topN
- 模型家族

不要在这里写卖法和资金分配。
