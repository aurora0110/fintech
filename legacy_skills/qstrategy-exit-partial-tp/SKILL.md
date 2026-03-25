---
name: qstrategy-exit-partial-tp
description: 用于 Qstrategy 的分批止盈卖法模块。当用户要研究一次性卖出和分批卖出的差异时使用。
---

# 分批止盈卖法

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 用户要比较一次性卖出和分批卖出
- 用户要研究半仓止盈、剩余仓位怎么处理

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

- “用 `qstrategy-exit-partial-tp`，first_tp=3.5%，卖一半，剩余转绿卖”

## 输出要求

- 必须明确第一笔卖多少
- 剩余仓位的退出条件是什么
- 是否只对盈利仓启用后续规则
