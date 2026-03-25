---
name: exit-model-tp
description: 用于 Qstrategy 的模型卖点加固定止盈卖法模块。当用户要使用类似 B1 当前正式卖法时使用。
---

# 模型 + 固定止盈

先读取 [conventions](/Users/lidongyang/Desktop/Qstrategy/skills/conventions/SKILL.md)。

## 核心结构

- 固定止盈阈值
- 模型卖点分数
- 谁先触发谁优先

## 当前正式参考

- B1：
  - `model_plus_tp`
  - `TP20%`
  - `xgb_score_v2`

## 使用方法

- “用 `exit-model-tp`，TP=20%，score=xgb_score_v2”
