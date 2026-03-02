# 单策略结构分析报告

---

## 一、实验信息

- 实验编号：{experiment_id}
- 策略名称：{strategy_name}
- 策略版本：{version}
- 运行日期：{run_date}
- 数据区间：{start_date} ~ {end_date}
- 样本股票数量：{stock_count}

---

## 二、策略定义

### 入场规则
{entry_rule_description}

### 出场规则
{exit_rule_description}

### 持仓逻辑
{holding_logic}

### 风控设置
{risk_control_description}

---

## 三、单笔交易统计

- 总交易次数：{total_trades}
- 胜率：{win_rate:.2%}
- 平均单笔收益：{avg_return:.4%}
- 平均盈利：{avg_win:.4%}
- 平均亏损：{avg_loss:.4%}
- 盈亏比：{profit_loss_ratio:.2f}
- 单笔期望值：{expectation:.4%}
- 最大单笔盈利：{max_win:.4%}
- 最大单笔亏损：{max_loss:.4%}
- 收益标准差：{std_return:.4%}

---

## 四、收益结构诊断

- 正收益笔数：{win_count}
- 负收益笔数：{loss_count}
- 连续最大盈利次数：{max_consecutive_wins}
- 连续最大亏损次数：{max_consecutive_losses}

### 结构判断

- 是否正期望：{is_positive_expectation}
- 是否高胜率低盈亏比：{is_high_win_low_pl}
- 是否依赖少数大盈利：{tail_dependency}
- 是否风险集中于单笔极端亏损：{extreme_loss_risk}

---

## 五、结构评估结论

{structure_evaluation_text}

---

## 六、优化方向建议

{optimization_suggestions}