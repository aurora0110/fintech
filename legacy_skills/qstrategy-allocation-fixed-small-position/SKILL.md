---
name: qstrategy-allocation-fixed-small-position
description: 用于 Qstrategy 的固定小仓位资金分配模块。当用户要避免净值扩仓影响实验时使用。
---

# 固定小仓位资金分配

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- B3、单针等样本偏少策略
- 不希望因为净值增长自动放大仓位

## 核心规则

- 每次买入固定金额
- 不随账户收益加仓
- 更适合看策略本身质量

## 使用方法

- “用 `qstrategy-allocation-fixed-small-position`，每笔固定1万元”
