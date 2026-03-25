---
name: qstrategy-b3-entry
description: 用于 Qstrategy 的 B3 买点定义与变体选择。当用户要使用、比较或继续收敛 B3 买点时使用。
---

# B3 买点

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 用户要继续研究 B3 链式结构
- 用户要切换不同 B3 变体

## 可选变体

### `follow_through`

- 口径：**研究**
- 当前主承接池

### `weekly_aligned`

- 口径：**研究**
- 在主承接基础上叠加周线背景

### `small_position_reference`

- 口径：**研究**
- 固定小仓位参考版本
- 历史结论里 `12%止盈` 强于最佳固定持有

## 当前状态

- B3 还没有统一正式冠军
- 但链式结构方向和小仓位口径已有明确研究价值

## 使用方法

- “用 `qstrategy-b3-entry` 的 `follow_through`”
- “比较 `follow_through` 和 `weekly_aligned`”

## 修改点

- `prev_b2`
- 缩量承接
- 周线对齐
- 链式前提条件
