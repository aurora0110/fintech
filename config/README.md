# 配置目录

存放策略运行所需的各类配置文件。

## 内容

```
config/
├── daily_records.yaml          # 每日选股记录
├── holding.yaml                # 当前持仓记录
├── technical_config.yaml       # 技术指标配置
└── qfactor/                    # 量化因子配置
    ├── B1factor/b1.py         # B1因子定义
    ├── B2factor/b2.py         # B2因子定义
    ├── B3factor/b3.py         # B3因子定义
    ├── decisionsupportfactor/  # 决策支持因子
    ├── didifactor/            # DIDI因子
    └── liefactor/             # LIE因子
```

## 用途说明

| 文件 | 说明 |
|------|------|
| `holding.yaml` | 当前持仓股票及成本价 |
| `daily_records.yaml` | 每日选股结果记录 |
| `technical_config.yaml` | 止盈止损、技术指标参数 |
| `qfactor/` | 各策略（B1/B2/B3/BRICK/单针）的量化因子定义 |
