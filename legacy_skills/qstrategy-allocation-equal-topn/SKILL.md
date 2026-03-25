---
name: qstrategy-allocation-equal-topn
description: 用于 Qstrategy 的等权 topN 资金分配模块。当用户要公平比较买点、卖点或候选池时使用。
---

# 等权 TopN 资金分配

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 公平比较不同买点
- 公平比较不同卖法
- 不希望仓位干扰结论

## 核心规则

- 按分数取 `topN`
- 每只票等权
- 最大持仓固定

## 使用方法

- “用 `qstrategy-allocation-equal-topn`，topN=10”

## 修改点

- `topN`
- 最大持仓
- 是否允许同日重叠
