---
name: b1-entry
description: 用于 Qstrategy 的 B1 买点定义与变体选择。当用户要使用、比较或修改 B1 买点时使用。
---

# B1 买点

先读取 [conventions](/Users/lidongyang/Desktop/Qstrategy/skills/conventions/SKILL.md)。

## 可选变体

### `template_fusion`

- 口径：正式
- 当前 B1 正式冠军
- 核心：
  - `J` 低位
  - 回踩趋势线 / 回踩多空线
  - `关键K / 缩半量 / 倍量柱` 为加分项
  - 模板相似度 + 近似反例 + 因子 + `XGBoost`
  - `top3`

### `factor_discovery_old`

- 口径：正式候选 / 旧冠军

### `trend_only`

- 口径：研究

### `long_only`

- 口径：研究

## 使用方法

- “用 `b1-entry` 的 `template_fusion`”
- “比较 `b1-entry(template_fusion)` 和 `b1-entry(factor_discovery_old)`”
