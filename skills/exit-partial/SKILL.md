---
name: exit-partial
description: 用于 Qstrategy 的分批止盈卖法模块。当用户要研究一次性卖出和分批卖出的差异时使用。
---

# 分批止盈

先读取 [conventions](/Users/lidongyang/Desktop/Qstrategy/skills/conventions/SKILL.md)。

## 参数

- `first_sell_ratio`
- `first_tp_pct`
- `remaining_exit_mode`
  - `day3_exit`
  - `green_next_open`
  - `green_next_open_profit_only`
  - `second_tp`
- `second_tp_pct`
- `stop_loss_rule`

## 使用方法

- “用 `exit-partial`，first_tp=3.5%，先卖一半，剩余转绿卖”
