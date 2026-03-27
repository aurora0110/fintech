# 核心引擎目录

存放量化策略系统的核心模块。

## 内容

```
core/
├── __init__.py
├── data_loader.py    # 数据加载器
├── engine.py         # 回测引擎
├── market_rules.py   # 市场规则（涨跌停、T+1等）
├── metrics.py        # 绩效指标计算
└── models.py         # 模型定义
```

## 用途说明

| 文件 | 说明 |
|------|------|
| `data_loader.py` | 从txt/csv文件加载K线数据，清洗、排序、去重 |
| `engine.py` | 回测执行引擎，模拟买卖、持仓管理 |
| `market_rules.py` | A股市场规则：±10%涨跌停、T+1、100股整数等 |
| `metrics.py` | 计算胜率、夏普、最大回撤等绩效指标 |
| `models.py` | 策略模型定义接口 |
