---
name: exit-minute
description: 用于 Qstrategy 的分钟线卖出特征模块。当用户要研究分钟线止盈止损或盘中退出特征时使用。
---

# 分钟线卖出

先读取 [conventions](/Users/lidongyang/Desktop/Qstrategy/skills/conventions/SKILL.md)。

## 当前候选方法

- `kdj_reversal`
- `ema_cross`
- `vwap_break`
- `consecutive_down`
- `trailing_stop`

## 使用方法

- “用 `exit-minute` 的 `kdj_reversal`”
- “比较 `exit-minute(kdj_reversal)` 和 `exit-minute(vwap_break)`”
