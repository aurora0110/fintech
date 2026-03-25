---
name: qstrategy-allocation-score-weighted
description: 用于 Qstrategy 的按评分加权资金分配模块。当用户要研究高分信号是否应放大仓位时使用。
---

# 评分加权资金分配

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 用户要研究“高分信号给更大仓位”是否更优
- 用户要把仓位与模型分数绑定

## 核心规则

- 先排序
- 再按分数区间或相对权重分配资金
- 需要同时限制单票上限

## 使用方法

- “用 `qstrategy-allocation-score-weighted`，高分信号加仓”

## 修改点

- 权重函数
- 单票上限
- 最大持仓数
