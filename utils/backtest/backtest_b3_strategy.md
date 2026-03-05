## B3策略详细方案

### 买入条件（a日满足以下全部10条）：

| 序号 | 条件 |
|------|------|
| 1 | a日J值 < 80 |
| 2 | a日是阳线（收盘价 > 开盘价） |
| 3 | a日涨幅 < 2%（收盘价-前一日收盘价）/前一日收盘价 |
| 4 | a日振幅 < 4%（最高价-最低价）/最低价 |
| 5 | a日成交量 < a-1日成交量（缩量） |
| 6 | a-1日是阳线 |
| 7 | a-1日成交量 >= a-2日成交量 × 1.8倍（倍量） |
| 8 | a-2日J值 < 30 |
| 9 | a-3日是阴线 |
| 10 | a-4日是阴线 |

### 买入执行
- **信号发出**：a日收盘后
- **买入时机**：a+1日以开盘价买入

### 止损价
```
min(买入日最低价, 信号日最低价) × 0.95
```

---

## 策略对比结果

### 策略1：固定持有天数

| 持有天数 | 最终倍数 | 年化收益率 | 交易次数 | 成功率 | 最大回撤 | 平均持有天数 |
|----------|----------|------------|----------|--------|----------|--------------|
| 2天 | 0.94x | -5.70% | 48 | 58.33% | -19.90% | 2.0天 |
| 3天 | 0.73x | -27.34% | 39 | 46.15% | -27.95% | 2.8天 |
| 30天 | 1.01x | 1.38% | 14 | 64.29% | -16.54% | 13.0天 |
| 60天 | 0.91x | -9.81% | 11 | 54.55% | -16.47% | 16.0天 |

### 策略2：止盈止损

| 止盈比例 | 最终倍数 | 年化收益率 | 交易次数 | 成功率 | 最大回撤 | 平均持有天数 |
|----------|----------|------------|----------|--------|----------|--------------|
| 7% | 0.97x | -2.77% | 13 | 53.85% | -23.22% | 11.7天 |
| 8% | 0.92x | -8.55% | 8 | 50.00% | -23.15% | 15.1天 |
| 9% | 1.01x | 0.96% | 9 | 55.56% | -25.32% | 12.0天 |
| 10% | 1.01x | 0.91% | 6 | 50.00% | -24.64% | 25.2天 |
| **11%** | **1.19x** | **19.36%** | 6 | **66.67%** | -12.26% | 21.2天 |
| 12% | 1.10x | 10.60% | 2 | 50.00% | -13.96% | 13.5天 |

---

## 结论

| 指标 | 最佳固定持有(30天) | 最佳止盈(11%) |
|------|-------------------|---------------|
| 最终倍数 | 1.01x | **1.19x** |
| 年化收益率 | 1.38% | **19.36%** |
| 交易次数 | 14 | 6 |
| 成功率 | 64.29% | **66.67%** |
| 最大回撤 | -16.54% | **-12.26%** |

**11%止盈策略**在所有指标上都优于最佳固定持有策略！

toolName: search_by_regex
            
status: success
          
query: initial_capital = 
          


        
            
toolName: search_by_regex
            
status: success
          
query: allocation = total_value
          


        
            
toolName: edit_file_search_replace
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/backtest/backtest_b3_strategy.py
          
changes: undefined
          


        
            
toolName: search_by_regex
            
status: success
          
query: initial_capital = 1000000
          


        
            
toolName: search_by_regex
            
status: success
          
query: initial_capital = 1000000
          


        
            
toolName: view_files
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/backtest/backtest_b3_strategy.py
          


        
            
toolName: edit_file_search_replace
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/backtest/backtest_b3_strategy.py
          
changes: undefined
          


        
            
toolName: edit_file_search_replace
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/backtest/backtest_b3_strategy.py
          
changes: undefined
          


        
            
toolName: search_by_regex
            
status: success
          
query: allocation = total_value \* 0.1
          


        
            
toolName: edit_file_search_replace
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/backtest/backtest_b3_strategy.py
          
changes: undefined
          


        
            
toolName: search_by_regex
            
status: success
          
query: cost <= total_value:
          


        
            
toolName: edit_file_search_replace
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/backtest/backtest_b3_strategy.py
          
changes: undefined
          


        
            
toolName: search_by_regex
            
status: success
          
query: cost <= total_value:
          


        
            
toolName: edit_file_search_replace
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/backtest/backtest_b3_strategy.py
          
changes: undefined
          


        
            
toolName: run_command
            
status: success
          
command: cd /Users/lidongyang/Desktop/Qstrategy && python3 utils/backtest/backtest_b3_strategy.py 2>&1 | tail -35 undefined
          
回测完成！

## 对比结果

### 修改前（10%仓位，总资金的10%）
| 指标 | 最佳固定持有(2天) | 最佳止盈(11%) |
|------|-------------------|---------------|
| 最终倍数 | 2.00x | 1.98x |
| 年化收益率 | 104.93% | 103.44% |
| 交易次数 | 70 | 68 |
| 最大回撤 | -0.32% | -1.62% |

### 修改后（1%仓位，固定100万的1%）
| 指标 | 最佳固定持有(60天) | 最佳止盈(12%) |
|------|-------------------|---------------|
| 最终倍数 | 1.03x | 1.16x |
| 年化收益率 | 2.76% | 16.22% |
| 交易次数 | 944 | 664 |
| 最大回撤 | -4.89% | -5.14% |

### 关键差异

| 对比项 | 修改前 | 修改后 |
|--------|--------|--------|
| 仓位 | 10%动态 | 1%固定 |
| 收益率 | 很高(100%+) | 较低(最高16%) |
| 交易次数 | 70次 | 600-1000次 |
| 最大回撤 | 很小(<2%) | 稍大(~5%) |

**说明**：修改前是用当时总资金的10%买入，赚钱后继续加仓；修改后是每次只用100万的1%（即1万元）固定买入，即使赚钱也不增加仓位，所以收益率较低但交易次数更多。