# 工具目录

存放策略过滤器、指标计算、持仓管理、数据验证等核心工具模块。

## 内容

```
utils/
├── b1filter.py              # B1策略过滤器
├── b2filter.py              # B2策略过滤器
├── b3filter.py              # B3策略过滤器
├── brick_filter.py          # BRICK策略过滤器
├── pinfilter.py             # 单针策略过滤器
├── b1filter_similar_filter.py   # B1相似度过滤
├── lgbm_p3_shallower_core10_daily_top9_filter.py  # LGBM每日Top9过滤
├── technical_indicators.py  # 技术指标计算
├── stoploss.py              # 止损模块
├── takeprofit.py            # 止盈模块
├── holdprint.py             # 持仓打印
├── selectprint.py           # 选股结果打印
├── stockDataValidator.py    # 数据验证
├── strategy_feature_cache.py # 策略特征缓存
├── market_risk_tags.py      # 市场风险标签
├── shared_market_features.py # 共享市场特征
├── dashboard.py             # 看板
├── main.py                  # 主入口
├── backtest/                # 回测脚本
├── backtest_pipeline/       # 回测流水线
├── multi_factor_research/   # 多因子研究
├── parameter_optimizer/     # 参数优化
├── dashboard/               # 看板脚本
└── tmp/                     # 临时实验脚本
```

## 策略过滤器

| 文件 | 策略 |
|------|------|
| `b1filter.py` | B1 - 低位介入 |
| `b2filter.py` | B2 - 启动确认 |
| `b3filter.py` | B3 - 二次确认 |
| `brick_filter.py` | BRICK - 反包 |
| `pinfilter.py` | 单针 - 洗盘再起 |
| `b1filter_similar_filter.py` | B1相似度ML |

## 辅助模块

| 文件 | 用途 |
|------|------|
| `technical_indicators.py` | 计算MA、MACD、RSI等技术指标 |
| `stoploss.py` / `takeprofit.py` | 止盈止损逻辑 |
| `holdprint.py` / `selectprint.py` | 结果展示 |
| `stockDataValidator.py` | 清洗验证K线数据 |
| `strategy_feature_cache.py` | 特征缓存加速 |
| `dashboard.py` | Web看板 |
