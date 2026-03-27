# 回测管道目录

模块化的回测流水线系统，支持配置化组合回测。

## 内容

```
backtest_pipeline/
├── candidate_pools/      # 候选股票池
│   ├── b1.py            # B1候选池
│   ├── b2.py            # B2候选池
│   ├── b3.py            # B3候选池
│   ├── brick.py         # BRICK候选池
│   └── pin.py           # 单针候选池
├── confirmers/           # 二次确认器
├── configs/              # 管道配置文件
│   ├── b*_reference_pipeline*.json
│   ├── brick_comprehensive_lab*.json
│   └── brick_formal_best_pipeline*.json
├── exits/                # 退出规则
│   ├── brick.py          # BRICK专用退出
│   └── generic.py        # 通用退出
├── inputs/               # 数据输入
├── portfolio/            # 组合管理
├── rankers/             # 排序器
├── validators/           # 验证器
├── docs/                # 文档
│   ├── experiment_ledger.json  # 实验台账
│   └── module_inventory.md     # 模块清单
├── runner.py            # 管道运行器
├── catalog.py           # 模块目录
└── registry.py         # 模块注册表
```

## 流水线组件

| 组件 | 作用 |
|------|------|
| `candidate_pools/` | 按策略规则筛选候选股票 |
| `confirmers/` | 对候选股做二次确认过滤 |
| `rankers/` | 对候选股排序打分 |
| `exits/` | 止盈止损退出规则 |
| `portfolio/` | 组合资金管理 |
| `validators/` | 结果验证 |

## 配置说明

| 配置文件 | 说明 |
|----------|------|
| `brick_formal_best_pipeline.json` | BRICK正式最佳管道 |
| `brick_comprehensive_lab_*.json` | BRICK综合实验室 |
| `b*_reference_pipeline_smoke.json` | 各策略冒烟测试配置 |
