---
name: brick-entry
description: 用于 Qstrategy 的 BRICK 买点定义与变体选择。当用户要使用、比较或修改 BRICK 买点时使用。
---

# BRICK 买点

先读取 [conventions](/Users/lidongyang/Desktop/Qstrategy/skills/conventions/SKILL.md)。

## 可选变体

### `formal_best`

- 口径：正式
- 当前正式 BRICK baseline
- 核心：
  - `趋势线 > 多空线`
  - `3绿1红 / 3绿1红1绿1红`
  - 量能区间
  - 反包质量
  - `shrink_focus`
  - 前 `50%` 再取 `top10`

### `relaxed_similarity`

- 口径：研究
- 当前 relaxed 纯相似度冠军参考

### `relaxed_full_fusion`

- 口径：研究
- 当前 relaxed 综合冠军参考

### `green4_low_variant`

- 口径：研究
- 用于 `green4 + low层` 增强条件实验

## 使用方法

- “用 `brick-entry` 的 `formal_best`”
- “比较 `brick-entry(formal_best)` 和 `brick-entry(relaxed_full_fusion)`”

## 修改点

- 候选池
- 排序方式
- 模板构建
- gate / topN
- 模型家族

不要在这里写卖法和资金分配。
