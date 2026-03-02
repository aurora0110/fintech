---
name: a-share-quant-rules
description: 为A股量化研究与策略开发提供交易制度约束、回测实现规范和风险控制规则。若任务涉及A股选股、择时、下单逻辑、回测框架、交易可执行性验证（如涨跌停、T+1、停牌、一字板、最小交易单位、费用滑点）或将研究信号转为可实盘策略时使用本技能。
---

# A-Share Quant Rules

## Quick Start

1. 识别标的市场与板块属性（主板、创业板、科创板、北交所、ST、IPO阶段）。
2. 先读取 [trading-rules.md](references/trading-rules.md) 设定交易可行域。
3. 再读取 [modeling-policies.md](references/modeling-policies.md) 约束信号、回测和风控实现。
4. 输出策略时显式区分“研究收益”和“可执行收益”。

## Workflow

### 1. Build Constraints First

- 将所有制度约束参数化：涨跌停比例、T+1、停牌不可交易、最小成交单位、费用、滑点、集合竞价/连续竞价窗口。
- 不允许先写策略再补交易规则。

### 2. Encode Tradability in Signal-to-Order Mapping

- 买入信号触发时，先判断当日是否可买（非涨停封单、非停牌、满足成交量约束）。
- 卖出信号触发时，先判断是否可卖（非跌停封单、持仓满足T+1）。
- 若不可成交，记录“应交易未成交”的原因标签。

### 3. Enforce Position and Risk Limits

- 单票权重、行业暴露、风格因子暴露、换手上限、组合回撤阈值必须写成硬约束。
- 对触发风控的订单，执行降仓或禁开仓规则，而不是仅做事后统计。

### 4. Report With Execution Diagnostics

- 至少输出：信号命中率、订单成交率、因制度约束丢失的收益、分板块收益拆解、成本前后收益对比。
- 明确标注所有假设（撮合价、滑点模型、费用模型、复权口径、停复牌处理）。

## Output Contract

- 默认交付以下结构，除非用户要求其他格式：
  - `Strategy assumptions`: 板块范围、交易频率、可交易时段。
  - `Rule constraints`: 逐条列出制度约束及代码层实现方式。
  - `Execution model`: 信号到订单、订单到成交的流程。
  - `Risk controls`: 仓位、止损、回撤、流动性过滤。
  - `Backtest diagnostics`: 收益、回撤、换手、成交率、交易失败原因。

## References

- 交易制度细节：`references/trading-rules.md`
- 建模与回测策略规范：`references/modeling-policies.md`

## Guardrails

- 不得将A股制度约束简化为“次日可交易且必然成交”。
- 不得忽略涨跌停和停牌对成交概率的影响。
- 不得把未来信息泄露到当期可交易决策。
- 不得省略交易成本并直接比较“毛收益”。
