# Experiment Ledger

用于记录已经做过、且已经形成明确结论的实验，避免后续重复做同一轮。

## 记录规则

- 只登记已经有结果文件和明确结论的实验
- 每条至少记录：
  - 实验名
  - 主对象
  - 核心方法
  - 最终结论
- 结果目录
- 如果暂时无法还原到 pipeline 精确组合，至少登记：
  - 主对象
  - 核心结论
  - 参考文档或结果目录

## 已登记实验

### 1. B1 全量主链（自动特征发现冠军）
- 主对象：`B1`
- 核心方法：`自动特征发现因子评分 + pool_low_cross + top5 + fixed_tp 30%`
- 结论：全量账户层冠军，`final_multiple = 1.12698`
- 结果目录：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/results\/b1_buy_sell_model_account_v2_full_20260321_102049](/Users/lidongyang/Desktop/Qstrategy/results/b1_buy_sell_model_account_v2_full_20260321_102049)

### 2. B1 文本模板库 + 近似反例融合冠军
- 主对象：`B1`
- 核心方法：`fusion_template_hard_full_fusion_pool_txt_confirmed_top3 + model_plus_tp + xgb_score_v2 + TP20%`
- 结论：打破旧冠军，`final_multiple = 1.20154`，当前 B1 总冠军
- 结果目录：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/results\/b1_txt_template_account_v2_full_20260322_201735](/Users/lidongyang/Desktop/Qstrategy/results/b1_txt_template_account_v2_full_20260322_201735)

### 3. B1 分簇子类型整链
- 主对象：`B1`
- 核心方法：`subtype_cosine_pool_uptrend_top3 + ET 卖点模型`
- 结论：有效，但未超过主链冠军，`final_multiple = 1.11233`
- 结果目录：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/results\/b1_cluster_subtype_account_v1_full_20260321_171945](/Users/lidongyang/Desktop/Qstrategy/results/b1_cluster_subtype_account_v1_full_20260321_171945)

### 4. B1 子类型分流卖法
- 主对象：`B1`
- 核心方法：主链最优买点 + 子类型分流卖法
- 结论：子类型卖法有效，但仍低于全局固定止盈 30%
- 结果目录：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/results\/b1_subtype_exit_experiment_v1_full_20260321_191920](/Users/lidongyang/Desktop/Qstrategy/results/b1_subtype_exit_experiment_v1_full_20260321_191920)

### 5. B1 正例模板拆分：回踩趋势线 vs 回踩多空线
- 主对象：`B1`
- 核心方法：两类模板库分开建模
- 结论：
  - 趋势线模板强于多空线模板
  - 但两类合并后的总模板库冠军仍然最强
- 结果目录：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/results\/b1_txt_template_split_account_v1_trend_full_20260322_203745](/Users/lidongyang/Desktop/Qstrategy/results/b1_txt_template_split_account_v1_trend_full_20260322_203745)
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/results\/b1_txt_template_split_account_v1_long_full_20260322_204001](/Users/lidongyang/Desktop/Qstrategy/results/b1_txt_template_split_account_v1_long_full_20260322_204001)

### 6. B2 完美图上影线统计
- 主对象：`B2`
- 核心方法：用 `完美图/B2` 成功样本统计 `上影线/实体`
- 结论：`1/3` 是相对合理的硬门槛候选，`1/4` 偏严
- 参考文档：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/Qstrategy实验纪要补充_20260320.md](/Users/lidongyang/Desktop/Qstrategy/Qstrategy实验纪要补充_20260320.md)

### 7. B2 出货卖点量化回测
- 主对象：`B2`
- 核心方法：点状出货 / 区间出货卖点标签
- 结论：当前这版出货标签对 B2 没有形成稳定增益，暂不适合直接并入正式卖出
- 参考文档：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/Qstrategy实验纪要补充_20260320.md](/Users/lidongyang/Desktop/Qstrategy/Qstrategy实验纪要补充_20260320.md)

### 8. B3 逻辑放宽与案例回放
- 主对象：`B3`
- 核心方法：放宽前序 B1/B2 链式定义
- 结论：把“前前日 B1”改成 `J<13 且前一日 J 更高` 后，典型 B3 案例能够被稳定覆盖
- 参考文档：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/Qstrategy实验纪要补充_20260320.md](/Users/lidongyang/Desktop/Qstrategy/Qstrategy实验纪要补充_20260320.md)

### 9. B3 上影线与振幅阈值放宽
- 主对象：`B3`
- 核心方法：上影线 `1/4 -> 1/3`，振幅 `<4% -> <5%`
- 结论：案例覆盖更符合主观样本，但真正关键仍是链式结构本身
- 参考文档：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/Qstrategy实验纪要补充_20260320.md](/Users/lidongyang/Desktop/Qstrategy/Qstrategy实验纪要补充_20260320.md)

### 10. B3 固定小仓位口径回测
- 主对象：`B3`
- 核心方法：固定资金 1% 仓位，对比固定持有与止盈策略
- 结论：修正为真实小仓位后，`12%止盈` 仍优于最佳固定持有
- 参考文档：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/utils\/backtest\/backtest_b3_strategy.md](/Users/lidongyang/Desktop/Qstrategy/utils/backtest/backtest_b3_strategy.md)

### 11. 单针二阶增强因子实验
- 主对象：`单针`
- 核心方法：基础单针 + 二阶结构因子增强
- 结论：`下影线占比 <= 0.05` 与 `趋势线3日斜率 > 0.8%` 是当前最强增强方向
- 结果目录：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/results\/pin_secondary_factor_experiment](/Users/lidongyang/Desktop/Qstrategy/results/pin_secondary_factor_experiment)

### 12. 单针最佳持有天数实验
- 主对象：`单针`
- 核心方法：固定最优单针信号，比较 `1/2/3/4/5` 天持有
- 结论：当前最佳持有天数是 `3天`
- 结果目录：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/results\/pin_hold_days_comparison](/Users/lidongyang/Desktop/Qstrategy/results/pin_hold_days_comparison)

### 13. 单针止盈止损 5 天 AB
- 主对象：`单针`
- 核心方法：对比固定止盈、信号日最低价止损等短持有退出方式
- 结论：更均衡的版本收敛到 `10%止盈 + 信号日最低价止损 + T+2起执行`
- 结果目录：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/results\/pin_tp_sl_comparison](/Users/lidongyang/Desktop/Qstrategy/results/pin_tp_sl_comparison)
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/Qstrategy实验纪要补充_20260320.md](/Users/lidongyang/Desktop/Qstrategy/Qstrategy实验纪要补充_20260320.md)

### 14. 单针 A/B/C 三分型重建
- 主对象：`单针`
- 核心方法：缩量回踩型 / 强趋势加速型 / 结构支撑型
- 结论：A+B+C 版本在当时样本口径下实现了 `15/15` 成功样本覆盖，且未新增失败案例
- 参考文档：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/Qstrategy实验纪要补充_20260320.md](/Users/lidongyang/Desktop/Qstrategy/Qstrategy实验纪要补充_20260320.md)

### 15. BRICK 完美案例反推过滤器
- 主对象：`BRICK`
- 核心方法：用完美案例反推 `brick_filter.py` 的错杀点并重训过滤器
- 结论：去掉过严前置后，完美案例覆盖从 `7/21` 提升到 `13/21`
- 结果目录：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/results\/brick_filter](/Users/lidongyang/Desktop/Qstrategy/results/brick_filter)
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/utils\/backtest\/backtest_brick_strategy.md](/Users/lidongyang/Desktop/Qstrategy/utils/backtest/backtest_brick_strategy.md)

### 16. BRICK 成功/失败共性与二值排序
- 主对象：`BRICK`
- 核心方法：成功案例共性统计、反例共性统计、二值加减分排序
- 结论：BRICK 更像中高位活动区里的压制后转强，`2倍量` 适合加分不适合硬过滤，二值排序更符合人工复盘语义
- 参考文档：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/Qstrategy实验纪要补充_20260320.md](/Users/lidongyang/Desktop/Qstrategy/Qstrategy实验纪要补充_20260320.md)

### 17. 风险标签对 B3 / 单针 / BRICK 的适配性
- 主对象：`B3 / 单针 / BRICK`
- 核心方法：加入风险标签后测试硬过滤与扣分提示
- 结论：单针更适合过滤“近20日高位巨量阴线”，B3 适合轻量过滤“顶部出货区”，BRICK 优先过滤“近20日假突破”或核心出货组合
- 参考文档：
  - [\/Users\/lidongyang\/Desktop\/Qstrategy\/Qstrategy实验纪要补充_20260320.md](/Users/lidongyang/Desktop/Qstrategy/Qstrategy实验纪要补充_20260320.md)

## 仍待继续结构化

以下实验虽然已经登记，但还没有全部还原成可直接映射到 `backtest_pipeline` 的精确 `combo`：
- `B2`
- `B3`
- `单针`
- `BRICK`

后续应继续把这些历史实验补成：
- 精确候选池
- 精确排序器
- 精确卖法参数
- 对应结果目录
