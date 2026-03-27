# 回测脚本目录

存放独立回测脚本，按策略分类。

## 内容

```
backtest/
├── backtest_b1_strategy.py/.md    # B1策略回测
├── backtest_b2_strategy.py/.md    # B2策略回测
├── backtest_b3_strategy.py/.md    # B3策略回测
├── backtest_brick_strategy.py/.md # BRICK策略回测
├── backtest_pin_strategy.py/.md   # 单针策略回测
├── compare_*.py                   # 策略对比实验
├── run_*.py                        # 参数搜索实验
└── similar_filter_strategy.md      # 相似过滤策略说明
```

## 策略说明

| 策略 | 目的 |
|------|------|
| B1 | 低位低风险介入点 |
| B2 | 正式启动确认 |
| B3 | B2后二次确认承接 |
| BRICK | 连续压制后反包 |
| 单针 | 强趋势洗盘再起 |

## 主要实验脚本

| 脚本 | 说明 |
|------|------|
| `run_brick_b2_signal_expansion.py` | BRICK+B2信号扩展实验 |
| `run_pin_combo_experiment.py` | 单针组合实验 |
| `compare_momentum_tail_*.py` | 动量尾部对比系列 |
