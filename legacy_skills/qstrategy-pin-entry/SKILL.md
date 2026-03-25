---
name: qstrategy-pin-entry
description: 用于 Qstrategy 的单针策略买点定义与变体选择。当用户要使用、比较或继续收敛单针买点时使用。
---

# 单针买点

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 用户要使用单针策略
- 用户要比较 A/B/C 三型
- 用户要研究单针增强因子

## 可选变体

### `trend_wash`

- 口径：**研究**
- A/B/C 综合单针池

### `structure_support`

- 口径：**研究**
- 偏 `C型(结构支撑)` 的单针池

### `abc_split`

- 口径：**研究**
- 单独比较 A/B/C 三型

## 当前已知结论

- 最佳持有天数：`3天`
- 更均衡的止盈止损参考：
  - `5%止盈 + signal日最低价*0.99止损`

## 使用方法

- “用 `qstrategy-pin-entry` 的 `trend_wash`”
- “比较单针 A/B/C”

## 修改点

- A/B/C 定义
- 下影线比例
- 趋势斜率
- 位置过滤
