---
name: pin-entry
description: 用于 Qstrategy 的单针策略买点定义与变体选择。当用户要使用、比较或继续收敛单针买点时使用。
---

# 单针买点

先读取 [conventions](/Users/lidongyang/Desktop/Qstrategy/skills/conventions/SKILL.md)。

## 可选变体

### `trend_wash`

- 口径：研究
- A/B/C 综合单针池

### `structure_support`

- 口径：研究
- 偏 `C型(结构支撑)`

### `abc_split`

- 口径：研究
- A/B/C 三型对比

## 已知结论

- 最佳持有天数：`3天`
- 均衡止盈止损参考：`5%止盈 + signal日低点*0.99止损`
