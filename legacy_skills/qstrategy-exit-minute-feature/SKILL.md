---
name: qstrategy-exit-minute-feature
description: 用于 Qstrategy 的分钟线卖出特征模块。当用户要研究分钟线止盈止损或盘中退出特征时使用。
---

# 分钟线卖出特征

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 用户要研究盘中卖点
- 用户要研究 1分钟 / 5分钟 特征

## 当前候选方法

- `kdj_reversal`
- `ema_cross`
- `vwap_break`
- `consecutive_down`
- `trailing_stop`

## 使用方法

- “用 `qstrategy-exit-minute-feature` 的 `kdj_reversal`”
- “比较 `kdj_reversal` 和 `vwap_break`”

## 输出要求

- 必须说明用到了哪些线：
  - 日线
  - 1分钟线
  - 5分钟线
- 必须说明是否为 smoke 结果
