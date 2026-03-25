---
name: alloc-score
description: 用于 Qstrategy 的按评分加权资金分配模块。当用户要研究高分信号是否应放大仓位时使用。
---

# 评分加权分配

先读取 [conventions](/Users/lidongyang/Desktop/Qstrategy/skills/conventions/SKILL.md)。

## 核心规则

- 先排序
- 再按分数权重分配资金
- 同时限制单票上限

## 使用方法

- “用 `alloc-score`，高分信号给更大仓位”
