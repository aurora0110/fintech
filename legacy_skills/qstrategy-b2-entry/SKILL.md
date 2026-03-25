---
name: qstrategy-b2-entry
description: 用于 Qstrategy 的 B2 买点定义与变体选择。当用户要使用、比较或继续收敛 B2 买点时使用。
---

# B2 买点

先读取 [qstrategy-backtest-conventions](/Users/lidongyang/Desktop/Qstrategy/skills/qstrategy-backtest-conventions/SKILL.md)。

## 适用场景

- 用户要继续研究 B2
- 用户要在 `type1 / type4 / main` 之间切换
- 用户要把 B2 明确收敛成正式主线

## 可选变体

### `main`

- 口径：**研究**
- 当前最宽的 B2 主候选池

### `type1`

- 口径：**研究**
- 贴近多空线
- `J` 在 20 日 10% 低位附近

### `type4`

- 口径：**研究**
- 趋势线上穿多空线后的第一次回踩趋势线

## 当前状态

- B2 还没有像 B1/BRICK 那样收敛出统一正式冠军
- 上影线、出货标签更多还是辅助研究结论

## 使用方法

- “用 `qstrategy-b2-entry` 的 `type1`”
- “比较 B2 `type1` 和 `type4`”

## 修改点

- 上影线阈值
- 出货过滤
- 类型定义
- 时间位置约束
