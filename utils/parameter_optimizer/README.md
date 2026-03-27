# 参数优化目录

存放策略参数优化相关代码和结果。

## 内容

```
parameter_optimizer/
├── optimizer.py           # 优化器主程序
├── full_optimization*.py  # 全量参数优化
├── build_cache*.py       # 构建缓存
├── score_optimizer.py     # 分数优化
├── single_factor_test.py  # 单因子测试
├── optimal_*.json         # 最优参数结果
└── analysis_report.md     # 分析报告
```

## 主要文件

| 文件 | 说明 |
|------|------|
| `optimizer.py` | 参数优化器 |
| `full_optimization*.py` | 全量参数搜索 |
| `score_optimizer.py` | 排序分数优化 |
| `single_factor_test.py` | 单因子显著性测试 |
| `optimal_factor_search.py` | 最优因子搜索 |
| `optimal_*.json` | 搜索结果 |
| `analysis_report.md` | 分析报告 |

## 用途

对策略参数（如止盈止损阈值、持有天数、排序权重等）进行网格搜索或贝叶斯优化，找到最优参数组合。
