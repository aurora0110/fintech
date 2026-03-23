# Backtest Pipeline 使用说明

## 当前目标

`backtest_pipeline` 用来把回测拆成固定模块，再用配置拼接组合，避免每次拍脑门加脚本。

## 入口

当前第一版入口是：

- [runner.py](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/runner.py)

现在它先做两件事：
1. 读取配置
2. 校验并描述这条 pipeline 的模块组合

后续会继续补：
- 真正的候选池生成
- 排序器执行
- 账户层回测
- 结果快照与覆盖报告

## 运行方式

示例：

```bash
python3 /Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/runner.py \
  /Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/configs/b1_reference_pipeline_smoke.json
```

## 配置结构

每条 pipeline 配置由 5 块组成：

1. `data`
- 数据目录
- 剔除区间
- 最小历史长度

2. `strategy`
- 策略家族：`b1 / b2 / b3 / pin / brick`
- 候选池模块
- 确认因子模块
- 该策略自己的参数

3. `ranker`
- 排序器模块
- `top_n`
- 模型/融合参数

4. `exit`
- 卖出模块
- 止盈止损参数

5. `account`
- 初始资金
- 最大持仓
- 最长持股
- 冷却与暂停规则
- 默认止损
- 多空线下止损减半

## 当前内置候选池

### B1
- `b1.low_cross`
- `b1.txt_confirmed`

### B2
- `b2.type1`
- `b2.type4`

### B3
- `b3.follow_through`
- `b3.weekly_aligned`

### 单针
- `pin.trend_wash`
- `pin.structure_support`

### BRICK
- `brick.main`

## 当前内置确认因子

### B1
- `b1.semantic_bonus`

### B2
- `b2.startup_quality`

### B3
- `b3.follow_through_quality`

### 单针
- `pin.needle_quality`

### BRICK
- `brick.turn_quality`

## 当前内置排序器

- `ranker.similarity`
- `ranker.factor_discovery`
- `ranker.xgboost`
- `ranker.lightgbm`
- `ranker.naive_bayes`
- `ranker.reinforcement_learning`
- `ranker.fusion`

## 当前内置卖出模块

- `exit.fixed_tp`
- `exit.model_only`
- `exit.model_plus_tp`
- `exit.partial_tp`

## 当前原则

1. 候选池先按策略家族拆开，不混 B1/B2/BRICK 语义。
2. 同一策略内部允许多个候选池变体。
3. `关键K / 缩半量 / 倍量柱` 这类默认放确认因子，不直接变成独立主买点。
4. 新方法一律作为模块追加，不覆盖旧方法。

## 实验矩阵与覆盖报告

当前已经支持自动生成“哪些组合理论上该测、哪些已经做过”的覆盖报告。

命令：

```bash
python3 /Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/experiment_matrix.py
```

默认会生成：

- [coverage_report.json](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/coverage_report.json)
- [coverage_report.md](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/coverage_report.md)

如果只看某个策略家族，例如 `B1`：

```bash
python3 /Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/experiment_matrix.py \
  --family b1
```

账本来源：

- 人读版：[experiment_ledger.md](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/experiment_ledger.md)
- 机器版：[experiment_ledger.json](/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/experiment_ledger.json)

后续每完成一轮有明确结论的实验，都应该同步补进这两份账本，优先保证 `json` 版可用于自动比对。
