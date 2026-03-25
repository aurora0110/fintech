---
name: alloc-equal
description: 用于 Qstrategy 的等权 topN 资金分配模块。当用户要公平比较买点、卖点或候选池时使用。
---

# 等权 TopN 分配

先读取 [conventions](/Users/lidongyang/Desktop/Qstrategy/skills/conventions/SKILL.md)。

## 核心规则

- 按分数取 `topN`
- 每只票等权
- 最大持仓固定

## 使用方法

- “用 `alloc-equal`，topN=10”
