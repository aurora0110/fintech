# 多因子研究目录

存放因子研究、因子优化和组合权重搜索相关代码。

## 内容

```
multi_factor_research/
├── factor_calculator.py          # 因子计算
├── factor_pruning.md              # 因子剪枝说明
├── combo_search.py               # 组合搜索
├── weight_optimizer.py            # 权重优化
├── data_processor.py              # 数据处理
├── research_metrics.py            # 研究指标
├── run_*.py                        # 各类实验脚本
└── *.md                           # 分析报告
```

## 主要实验

| 脚本 | 说明 |
|------|------|
| `run_factor_scoring.py` | 因子打分实验 |
| `run_weighted_portfolio_backtest.py` | 加权组合回测 |
| `run_trend_*.py` | 趋势相关实验系列 |
| `run_repair_*.py` | 修复模型实验系列 |
| `run_sideways_special_experiment.py` | 震荡行情特殊处理 |

## 因子研究流程

1. `factor_calculator.py` - 计算候选因子
2. `analyze_factor_stability.py` - 分析因子稳定性
3. `weight_optimizer.py` - 优化因子权重
4. `combo_search.py` - 搜索最优组合
