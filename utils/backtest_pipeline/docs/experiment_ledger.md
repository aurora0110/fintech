# Experiment Ledger

- `b1` | `b1-factor-discovery-old-champion` | B1 全量主链旧冠军 | completed
  结论：自动特征发现因子评分 + low_cross + top5 + 固定止盈30%，全量账户层 final_multiple=1.12698。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/b1_buy_sell_model_account_v2_full_20260321_102049
- `b1` | `b1-template-hard-current-champion` | B1 文本模板 + 近似反例融合冠军 | completed
  结论：当前 B1 总冠军，final_multiple=1.20154，回撤更低。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/b1_txt_template_account_v2_full_20260322_201735
- `b1` | `b1-template-split-trend` | B1 回踩趋势线模板拆分 | completed
  结论：趋势线模板拆分有效，强于多空线模板，但仍低于总模板冠军。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/b1_txt_template_split_account_v1_trend_full_20260322_203745
- `b1` | `b1-template-split-long` | B1 回踩多空线模板拆分 | completed
  结论：多空线模板有效，但明显弱于趋势线模板和总模板冠军。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/b1_txt_template_split_account_v1_long_full_20260322_204001
- `b1` | `b1-subtype-chain` | B1 分簇子类型整链 | completed
  结论：分型思路成立，但当前全量仍未超过主链冠军。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/b1_cluster_subtype_account_v1_full_20260321_171945
- `b2` | `b2-upper-shadow-stat` | B2 完美图上影线统计 | completed
  结论：成功 B2 中大多数上影线/实体落在 1/3 以内，1/4 偏严。
- `b2` | `b2-distribution-exit-tags` | B2 出货卖点量化回测 | completed
  结论：当前点状出货/区间出货标签对 B2 没有形成稳定增益，暂不宜直接并入正式卖出。
- `b3` | `b3-logic-relax` | B3 逻辑放宽与案例回放 | completed
  结论：把前前日 B1 放宽为 J<13 且前一日 J 更高后，典型 B3 案例能够被稳定覆盖。
- `b3` | `b3-threshold-relax` | B3 上影线与振幅阈值放宽 | completed
  结论：上影线放宽到 1/3、振幅放宽到 <5% 后案例覆盖更符合主观样本，但关键仍是链式结构。
- `b3` | `b3-fixed-small-position` | B3 固定小仓位口径回测 | completed
  结论：在固定小仓位口径下，12%止盈仍优于最佳固定持有。
- `pin` | `pin-secondary-factor` | 单针二阶增强因子实验 | completed
  结论：单针当前最强增强方向是下影线占比<=0.05 与 趋势线3日斜率>0.8%。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/pin_secondary_factor_experiment
- `pin` | `pin-hold-days` | 单针最佳持有天数实验 | completed
  结论：固定最优单针信号后，最佳持有天数为 3 天。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/pin_hold_days_comparison
- `pin` | `pin-tp-sl-ab` | 单针止盈止损 5 天 AB | completed
  结论：更均衡的版本收敛到 10%止盈 + 信号日最低价止损 + T+2起执行。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/pin_tp_sl_comparison
- `pin` | `pin-abc-rebuild` | 单针 A/B/C 三分型重建 | completed
  结论：A+B+C 版本在当时样本口径下实现 15/15 成功样本覆盖，且未新增失败案例。
- `brick` | `brick-perfect-case-filter-rebuild` | BRICK 完美案例反推过滤器 | completed
  结论：去掉过严前置后，完美案例覆盖从 7/21 提升到 13/21。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/brick_filter
- `brick` | `brick-binary-ranking` | BRICK 成功/失败共性与二值排序 | completed
  结论：BRICK 更像中高位活动区里的压制后转强，2倍量适合加分不适合硬过滤，二值排序更符合人工复盘语义。
- `mixed` | `risk-tags-b3-pin-brick` | 风险标签对 B3 / 单针 / BRICK 的适配性 | completed
  结论：单针更适合过滤近20日高位巨量阴线，B3 适合轻量过滤顶部出货区，BRICK 优先过滤近20日假突破或核心出货组合。
- `brick` | `brick-formal-best-half-tp-green-compare` | BRICK 正式冠军卖法对比：3%止盈 vs 3.5%半仓止盈后等转绿 | completed
  结论：BRICK 正式冠军买点迁入 pipeline 后，对比结果显示 3.5%半仓止盈后等砖块转绿的卖法没有超过原 3%止盈策略；新卖法胜率更低、持有更久、组合净值更差。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/brick_pipeline_half_tp_green_compare_v1_full_20260323
- `brick` | `brick-formal-best-half-tp-green-profit-only-compare` | BRICK 正式冠军卖法对比：仅盈利仓启用 3.5% 半仓止盈后等转绿 | completed
  结论：仅对盈利仓启用 3.5% 半仓止盈后等转绿，明显优于“所有仓位都让剩余半仓等转绿”的版本，但仍未超过原 3%止盈+0.99止损+3天到期的 BRICK 正式冠军。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/brick_pipeline_half_tp_green_compare_v2_full_20260323
- `brick` | `brick-comprehensive-lab-brick_comprehensive_lab_smoke_20260325_r4` | BRICK 相似度/因子/机器学习/全融合综合实验 | completed
  结论：BRICK 综合实验当前总榜第一为 similarity_only|brick.green4_enhance|sample_300|len21|close_norm|cosine|gate0.80|top10，final_test total_return=0.6698。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/brick_comprehensive_lab_smoke_20260325_r4
- `brick` | `brick-comprehensive-lab-brick_comprehensive_lab_smoke_20260325_r5` | BRICK 相似度/因子/机器学习/全融合综合实验 | completed
  结论：BRICK 综合实验当前总榜第一为 similarity_only|brick.green4_enhance|sample_300|len21|close_norm|cosine|gate0.80|top10，final_test total_return=0.6698。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/brick_comprehensive_lab_smoke_20260325_r5
- `brick` | `brick-comprehensive-lab-brick_comprehensive_lab_debug_20260325_r1` | BRICK 相似度/因子/机器学习/全融合综合实验 | completed
  结论：BRICK 综合实验当前总榜第一为 similarity_plus_ml|brick.green4_low_enhance|sample_300|len21|close_norm|pipeline_corr_dtw|gate0.80|top10|xgboost，final_test total_return=0.1229。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/brick_comprehensive_lab_debug_20260325_r1
- `brick` | `brick-comprehensive-lab-brick_comprehensive_lab_debug_20260325_r2` | BRICK 相似度/因子/机器学习/全融合综合实验 | completed
  结论：BRICK 综合实验当前总榜第一为 similarity_plus_ml|brick.green4_low_enhance|sample_300|len21|close_norm|pipeline_corr_dtw|gate0.80|top10|xgboost，final_test total_return=0.1229。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/brick_comprehensive_lab_debug_20260325_r2
- `brick` | `brick-comprehensive-lab-brick_comprehensive_lab_smoke_20260325_r6` | BRICK 相似度/因子/机器学习/全融合综合实验 | completed
  结论：BRICK 综合实验当前总榜第一为 similarity_only|brick.green4_enhance|sample_300|len21|close_norm|cosine|gate0.80|top10，final_test total_return=0.6698。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/brick_comprehensive_lab_smoke_20260325_r6
- `brick` | `brick-comprehensive-lab-brick_comprehensive_lab_smoke_20260325_r7` | BRICK 相似度/因子/机器学习/全融合综合实验 | completed
  结论：BRICK 综合实验当前总榜第一为 similarity_only|brick.green4_enhance|sample_300|len21|close_norm|cosine|gate0.80|top10，final_test total_return=0.6698。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/brick_comprehensive_lab_smoke_20260325_r7
- `brick` | `brick-comprehensive-lab-brick_comprehensive_lab_full_20260325_r1` | BRICK 相似度/因子/机器学习/全融合综合实验 | completed
  结论：BRICK 综合实验当前总榜第一为 full_fusion|brick.relaxed_base|sample_300|len21|close_norm|pipeline_corr_dtw|gate0.80|top10|random_forest|w0.4_0.2_0.4，final_test total_return=2.9061。
  目录：/Users/lidongyang/Desktop/Qstrategy/results/brick_comprehensive_lab_full_20260325_r1
