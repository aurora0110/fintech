---
name: qstrategy-exit-model-plus-tp
description: 用于 Qstrategy 的模型卖点加固定止盈卖法模块。当用户要使用 B1 当前正式卖法或类似组合卖法时使用。
---

# 模型 + 固定止盈卖法

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 用户要使用 B1 当前正式卖法
- 用户要比较“模型卖”与“固定止盈卖”的组合

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

- “用 `qstrategy-exit-model-plus-tp`，TP=20%，score=xgb_score_v2”

## 修改点

- 固定止盈比例
- 模型版本
- 阈值
