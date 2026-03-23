# Experiment Coverage Report

- 总组合数：`876`
- 已登记完成：`2`
- 待覆盖：`874`

## 按策略家族汇总

| 策略 | 总数 | 已做 | 待做 |
| --- | ---: | ---: | ---: |
| `b1` | 784 | 2 | 782 |
| `b2` | 30 | 0 | 30 |
| `b3` | 16 | 0 | 16 |
| `brick` | 30 | 0 | 30 |
| `pin` | 16 | 0 | 16 |

## 待做组合

| 策略 | 候选池 | 确认因子 | 排序器 | TopN | 卖法 | 参数 |
| --- | --- | --- | --- | ---: | --- | --- |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.factor_discovery` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.similarity` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.xgboost` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.lightgbm` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.naive_bayes` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `-` | `ranker.fusion` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "sim_corr_close_vol_concat"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.low_cross` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.similarity` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.factor_discovery` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.xgboost` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.lightgbm` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.naive_bayes` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `-` | `ranker.fusion` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.similarity` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "template_similarity_score"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.factor_discovery` | 8 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "factor_discovery"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.xgboost` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "xgboost"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.lightgbm` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "lightgbm"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.naive_bayes` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"score_col": "naive_bayes"}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 3 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.5}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "xgb_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "lgb_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "rf_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_only` | `{"exit": {"score_model": "et_score_v2"}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "lgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.2, "partial_take_profit_pct": 0.1}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b1` | `b1.txt_confirmed` | `b1.semantic_bonus` | `ranker.fusion` | 5 | `exit.partial_tp` | `{"exit": {"final_take_profit_pct": 0.3, "partial_take_profit_pct": 0.2}, "ranker": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b2` | `b2.main` | `b2.startup_quality` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `b3.follow_through_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `b3.follow_through_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `b3.follow_through_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `b3.follow_through_quality` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `b3.follow_through_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `b3.follow_through_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `b3.follow_through_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `b3` | `b3.follow_through` | `b3.follow_through_quality` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `-` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 8 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `brick` | `brick.main` | `brick.turn_quality` | `ranker.factor_discovery` | 8 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `-` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `-` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `-` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `-` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `pin.needle_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `pin.needle_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `pin.needle_quality` | `ranker.factor_discovery` | 3 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `pin.needle_quality` | `ranker.factor_discovery` | 3 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `pin.needle_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.1}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `pin.needle_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `pin.needle_quality` | `ranker.factor_discovery` | 5 | `exit.fixed_tp` | `{"exit": {"take_profit_pct": 0.3}, "ranker": {"score_col": "base_score"}}` |
| `pin` | `pin.trend_wash` | `pin.needle_quality` | `ranker.factor_discovery` | 5 | `exit.model_plus_tp` | `{"exit": {"score_model": "xgb_score_v2", "take_profit_pct": 0.2}, "ranker": {"score_col": "base_score"}}` |
