---
name: exit-fixed
description: 用于 Qstrategy 的固定止盈止损卖法模块。当用户要统一 TP/SL 口径比较不同买点时使用。
---

# 固定止盈止损

先读取 [conventions](/Users/lidongyang/Desktop/Qstrategy/skills/conventions/SKILL.md)。

## 参数

- `take_profit_pct`
- `stop_loss_rule`
- `max_hold_days`
- `exec_mode`
  - `next_open_after_trigger`
  - `intraday_trigger_price`
- `same_day_priority`
  - `take_profit_first`
  - `stop_loss_first`

## 使用方法

- “用 `exit-fixed`，TP=3%，SL=0.99，hold=3”
- “统一用 `exit-fixed` 比较买点质量”
