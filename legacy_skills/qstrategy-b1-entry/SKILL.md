---
name: qstrategy-b1-entry
description: 用于 Qstrategy 的 B1 买点定义与变体选择。当用户要使用、比较或修改 B1 买点时使用。
---

# B1 买点

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 用户要用 B1 选股
- 用户要比较 B1 不同买点变体
- 用户要复用 B1 当前正式冠军

## 可选变体

### `template_fusion`

- 口径：**正式**
- 含义：当前 B1 正式冠军
- 核心：
  - `J` 低位
  - 主买点：
    - 回踩趋势线
    - 回踩多空线
  - `关键K / 缩半量 / 倍量柱` 为加分项
  - 模板相似度 + 近似反例 + 因子 + `XGBoost`
  - `top3`

### `factor_discovery_old`

- 口径：**正式候选 / 旧冠军**
- 含义：自动特征发现因子 + `low_cross` 的旧冠军

### `trend_only`

- 口径：**研究**
- 含义：只保留回踩趋势线的模板变体

### `long_only`

- 口径：**研究**
- 含义：只保留回踩多空线的模板变体

## 输出要求

必须说明：

- 当前变体是否正式
- 主买点语义
- 辅助确认项是不是硬门槛

## 使用方法

推荐直接这样说：

- “用 `qstrategy-b1-entry` 的 `template_fusion`”
- “比较 `template_fusion` 和 `factor_discovery_old`”

## 修改点

允许改：

- 模板库
- 反例库
- 因子版本
- 模型版本
- `topN`
