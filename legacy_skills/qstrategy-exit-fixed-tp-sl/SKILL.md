---
name: qstrategy-exit-fixed-tp-sl
description: 用于 Qstrategy 的固定止盈止损卖法模块。当用户要统一 TP/SL 口径比较不同买点时使用。
---

# 固定止盈止损卖法

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 用户要统一卖法比较买点质量
- 用户要做 TP/SL 网格
- 用户要先用最简单卖法做 baseline

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

## 默认建议

- 买入当日不能卖
- 先固定 TP/SL/hold，再比较买点本身

## 使用方法

- “用 `qstrategy-exit-fixed-tp-sl`，TP=3%，SL=0.99，hold=3”
- “在统一固定止盈止损下比较”

## 输出要求

- 必须写清 TP/SL 的触发价格
- 必须写清实际成交价格
- 必须说明是不是次日开盘执行
