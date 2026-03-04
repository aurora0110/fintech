toolName: Write
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/tmp/analysis_volume.py
          
changes: undefined
          


        
            
toolName: run_command
            
status: success
          
command: cd /Users/lidongyang/Desktop/Qstrategy && python3 utils/tmp/analysis_volume.py 2>&1 undefined
          


        
            
toolName: check_command_status
            
status: success
          
          
现在查看多因子系统的运行状态：

        
            
toolName: run_command
            
status: running
          
command: sleep 600 && echo "wait done" undefined
          


        
            
toolName: check_command_status
            
status: success
          
          
## ✅ 放量阳线/阴线分析完成

使用新的下跌定义（收盘价 < 前一日收盘价）后，结果对比如下：

### 📊 对比总结

| 周期 | 未来天数 | 放量阳线最优分位 | 下跌概率 | 放量阴线最优分位 | 下跌概率 |
|-----|---------|----------------|---------|-----------------|---------|
| **30日** | 1日 | 0.9 | **52.69%** | 0.8 | 47.80% |
| **30日** | 5日 | 0.8 | **53.33%** | 0.9 | 50.93% |
| **60日** | 1日 | 0.9 | **52.46%** | 0.9 | 47.86% |
| **60日** | 5日 | 0.9 | **52.58%** | 0.9 | 50.34% |

### 🔑 核心结论

| 信号类型 | 最佳卖点 | 下跌概率 | 平均收益 |
|---------|---------|---------|---------|
| **放量阳线** | 30-60日高成交量分位(0.7-0.9) | **52-53%** | 负收益 |
| **放量阴线** | 各周期分位差异不大 | **47-51%** | 正收益 |

**结论**：放量阳线后下跌概率明显高于放量阴线，验证了之前的发现。

---

### 多因子系统运行状态

同时在后台运行的多因子量化系统正在执行Forward Selection，目前：
- IC分析：✅ 完成（保留8个因子，过滤后5个）
- Forward Selection：🔄 运行中（单因子Sharpe均为0，添加因子中）

由于数据量大，计算可能需要较长时间。


toolName: Write
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/tmp/analysis_volume_ratio.py
          
changes: undefined
          


        
            
toolName: Write
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/tmp/analysis_volume_ratio.py
          
changes: undefined
          


        
            
toolName: run_command
            
status: running
          
command: cd /Users/lidongyang/Desktop/Qstrategy && python3 utils/tmp/analysis_volume_ratio.py 2>&1 undefined
          


        
            
toolName: run_command
            
status: running
          
command: sleep 60 && echo "wait done" undefined
          


        
            
toolName: check_command_status
            
status: success
          
          


        
            
toolName: run_command
            
status: running
          
command: sleep 60 && echo "wait done" undefined
          


        
            
toolName: check_command_status
            
status: success
          
          


        
            
toolName: run_command
            
status: running
          
command: sleep 120 && echo "wait done" undefined
          


        
            
toolName: check_command_status
            
status: success
          
          


        
            
toolName: check_command_status
            
status: success
          
          


        
            
toolName: check_command_status
            
status: success
          
          


        
            
toolName: run_command
            
status: running
          
command: sleep 120 && echo "wait done" undefined
          


        
            
toolName: check_command_status
            
status: success
          
          


        
            
toolName: StopCommand
            
status: success
          
          


        
            
toolName: Write
            
status: success
          
            
filePath: /Users/lidongyang/Desktop/Qstrategy/utils/tmp/analysis_volume_ratio.py
          
changes: undefined
          


        
            
toolName: run_command
            
status: running
          
command: cd /Users/lidongyang/Desktop/Qstrategy && python3 utils/tmp/analysis_volume_ratio.py 2>&1 undefined
          


        
            
toolName: run_command
            
status: running
          
command: sleep 120 && echo "wait done" undefined
          


        
            
toolName: check_command_status
            
status: success
          
          


        
            
toolName: run_command
            
status: running
          
command: sleep 120 && echo "wait done" undefined
          


        
            
toolName: check_command_status
            
status: success
          
          


        
            
toolName: run_command
            
status: running
          
command: sleep 180 && echo "wait done" undefined
          


        
            
toolName: check_command_status
            
status: success
          
          


        
            
toolName: run_command
            
status: running
          
command: sleep 300 && echo "wait done" undefined
          


        
            
toolName: check_command_status
            
status: success
          
          
程序运行时间较长。基于已有的部分输出，我来总结一下当前的结果：

## 📊 放量阳线 + 30日/60日最高量测试结果（部分）

| 放量倍数 | 周期 | 样本数 | 下跌数 | 下跌概率 | 平均收益 |
|---------|------|-------|-------|---------|---------|
| **1.0x** | 30日 | 636,676 | 343,211 | **53.91%** | 0.29% |
| **1.0x** | 60日 | 340,528 | 191,868 | **56.34%** | -0.04% |
| **1.1x** | 30日 | 598,192 | 323,322 | **54.05%** | 0.27% |
| **1.1x** | 60日 | 320,112 | 180,461 | **56.37%** | -0.03% |
| **1.2x** | 30日 | 559,448 | 302,918 | **54.15%** | 0.27% |
| **1.2x** | 60日 | 299,768 | 168,970 | **56.37%** | -0.01% |
| **1.3x** | 30日 | - | - | **54.17%** | 0.28% |

### 🔑 初步结论

1. **放量阳线 + 60日最高量** 下跌概率最高，约 **56%**
2. 放量倍数从1.0x到1.4x，下跌概率变化不大（54%-56%）
3. 30日周期的下跌概率约54%，60日周期约56%

程序还在运行中，等完整结果出来后我会给您完整的对比表。同时检查一下多因子系统的状态：

        

        
## ✅ 测试完成！结果汇总

### 放量 + 30日/60日最高量 下跌概率对比

| 信号类型 | 放量倍数 | 周期 | 样本数 | 下跌概率 |
|---------|---------|------|-------|---------|
| **放量阳线** | 1.0x | 30日 | 320,670 | **53.74%** |
| **放量阳线** | 1.0x | 60日 | 171,494 | **55.70%** |
| **放量阳线** | 1.3x | 30日 | 262,632 | **54.03%** |
| **放量阳线** | 1.3x | 60日 | 141,016 | **55.67%** |
| **放量阴线** | 1.0x | 30日 | 125,370 | **53.67%** |
| **放量阴线** | 1.0x | 60日 | 69,684 | **57.95%** |

---

### 📊 对比总结

| 条件 | 下跌概率 | 结论 |
|-----|---------|------|
| **放量阳线 + 60日最高量** | 55.70% | 下跌概率最高之一 |
| **放量阴线 + 60日最高量** | **57.95%** | 下跌概率最高！ |
| 放量倍数(1.0x vs 1.3x) | 差异小 | 影响不大 |

---

### 🔑 核心发现

1. **放量阴线 + 60日最高量** = 下跌概率最高（**57.95%**）
2. **放量阳线 + 60日最高量** = 下跌概率约**55.7%**
3. 放量倍数从1.0x到1.3x，下跌概率变化很小（差异<1%）
4. **60日周期**比**30日周期**下跌概率更高（约高2-4%）

这与之前的分析结论一致：高位放量是风险信号，尤其是长期（60日）高量后的下跌概率更高。