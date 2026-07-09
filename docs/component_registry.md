# Qstrategy Component Registry

用途：记录当前项目中可复用的标准积木、冻结实验积木、待标准化能力和禁止复用的历史实现。新增回测、实验脚本、策略逻辑、指标、退出方式、统计分析或回测修复前，必须先查本文件。

治理说明：

本文件为 Qstrategy 项目级组件注册表治理初版，用于记录当前组件复用边界、状态和输入契约。文中 `FORMAL` / `TESTED` / `FROZEN_EXPERIMENT` 等状态来自当前治理审计和已有验证记录。若对应代码尚未纳入版本管理，或尚未通过独立提交审查，不应将其视为已发布稳定接口。任何组件在正式策略、OOS 验证或候选升级中使用前，仍需以对应代码、测试、冻结文档和最新审计结论为准。

状态说明：

- `FORMAL`：正式标准实现，可以直接复用。
- `FROZEN`：口径已冻结，禁止随意修改。
- `FROZEN_EXPERIMENT`：来自实验，但已通过质量检查，可作为实验积木使用。
- `NEEDS_STANDARDIZATION`：存在多个版本，尚未选出标准实现。
- `DEPRECATED`：已有替代实现，不应继续新增依赖。
- `INVALID`：存在已确认错误，禁止复用。
- `HISTORICAL_ONLY`：仅用于历史结果复原。

复用规则：

- `DIRECT_CALL`：直接调用标准实现。
- `PARAMETERIZED_CALL`：参数化调用标准实现。
- `THIN_ADAPTER`：允许薄适配字段名、格式、路径，不允许复制核心逻辑。
- `REFERENCE_ONLY`：只作为历史参考，不直接 import。
- `DO_NOT_REUSE`：禁止复用。

## 核心积木

| capability_id | capability_name | standard_implementation | function_or_class | status | input_contract | output_contract | known_scope | known_limitations | quality_status | replacement_for | reuse_rule | owner_experiment | last_verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DATA_DAILY_READ | 标准日线读取 | `utils/common/io.py` | `read_stock_daily`, `extract_stock_code_from_filename`, `read_stock_name_from_file` | FORMAL | txt/csv日线文件路径 | 标准化日线DataFrame、股票代码、名称 | 股票/ETF日线读取 | 成交量不做复权；ETF末尾注释行需调用方确认清洗 | TESTED | 多个脚本内重复读数 | DIRECT_CALL | common-library | 2026-07-06 |
| INDICATORS_CORE | 通用指标计算 | `utils/common/indicators.py` | MA/EMA/MACD/KDJ/RSI/趋势线/多空线相关函数 | FORMAL | 标准OHLCV序列 | 指标Series/DataFrame | 股票/ETF/策略实验 | 使用前必须核对窗口和字段口径 | TESTED | 脚本内重复指标公式 | DIRECT_CALL | common-library | 2026-07-06 |
| BRICK_CALC | BRICK底层计算 | `utils/common/brick.py` | `calc_brick_values` | FORMAL | close序列、BRICK参数 | brick颜色、长度、结构字段 | BRICK信号生成 | 参数必须由实验配置冻结，不得在公共函数里硬编码 | TESTED | 多脚本重复BRICK公式 | DIRECT_CALL | common-library | 2026-07-06 |
| COST_LOT_EQUAL | 成本、整手、等分资金底层 | `utils/common/backtest.py` | `calc_commission`, `calc_stamp_tax`, `round_lot_size`, `equal_weight_cash` | FORMAL | 金额、费率、股数、仓位数 | 手续费、印花税、整手股数、等分资金 | A股账户层 | 不包含完整账户撮合和涨跌停判断 | TESTED | 脚本内重复手续费/整手 | DIRECT_CALL | common-library | 2026-07-06 |
| RESULT_REPORT_UTILS | 结果目录和报告辅助 | `utils/common/report.py` | 目录/CSV/Markdown辅助函数 | FORMAL | 输出路径、表格、文本 | 标准结果文件 | 实验落盘 | 具体报告结构仍由实验定义 | AVAILABLE | 手写目录/CSV保存 | DIRECT_CALL | common-library | 2026-07-06 |
| VALIDATION_UTILS | 字段、样本、权益校验 | `utils/common/validation.py` | 字段/样本/权益检查函数 | FORMAL | DataFrame、字段列表、权益序列 | 校验结果 | 回测质量检查 | 尚未覆盖复用治理字段 | AVAILABLE | 手写字段检查 | DIRECT_CALL | common-library | 2026-07-06 |
| EXIT_EXECUTION_GAP | 跳空修正退出成交 | `utils/backtest_pipeline/exits/execution.py` | `take_profit_gap_corrected`, `stop_loss_gap_corrected`, `corrected_intraday_trail_exit`, `close_trail_next_open_exit`, `next_valid_open`, `fixed_close_exit` | FORMAL | 标准OHLCV DataFrame/row、entry_idx、entry_price、阈值 | `ExitFill` | 止盈止损、移动止盈、next valid open、固定收盘退出 | 不处理涨跌停不可成交；账户层需另接撮合 | TESTED | 旧ETF/B3脚本内重复退出成交 | DIRECT_CALL | reuse-first-governance-20260706 | 2026-07-06 |

## 策略与实验积木

| capability_id | capability_name | standard_implementation | function_or_class | status | input_contract | output_contract | known_scope | known_limitations | quality_status | replacement_for | reuse_rule | owner_experiment | last_verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BRICK_466_CANDIDATES | BRICK 466 BASE_MINIMAL_POOL候选 | `utils/tmp/run_brick_validation_runner_20260623.py` | `build_candidates_for_param` + `BrickParams(4,6,6)` | FROZEN_EXPERIMENT | `stock_data`, 参数、validation区间 | BRICK候选DataFrame和feature_map | BRICK研究实验 | 脚本是实验runner，import前需确认无顶层全量副作用 | QUALITY_PASS | 多脚本重复BRICK候选生成 | PARAMETERIZED_CALL | brick_validation_runner_20260623 | 2026-06-28 |
| BRICK_FIXED_TN_PLAN | BRICK固定T+N收盘交易计划 | `utils/tmp/run_brick_validation_runner_20260623.py` | `build_trade_plan(..., "fixed_TN_close")` | FROZEN_EXPERIMENT | 候选、feature_map、sell_rule、成本、fallback | 交易计划DataFrame | BRICK固定持有实验 | T+N定义为`signal_idx + N`收盘，不是买入后N日 | QUALITY_PASS | 多脚本重复固定持有 | PARAMETERIZED_CALL | brick_validation_runner_20260623 | 2026-07-06 |
| BRICK_ACCOUNT_RUNNER | BRICK账户层runner | `utils/tmp/run_brick_validation_runner_20260623.py` | `run_account` | FROZEN_EXPERIMENT | trade_plan、feature_map、stock_data、资金参数 | executed trades、equity curve | BRICK主线validation | 与完整涨跌停/停牌顺延仍有近似；复用前需报告口径 | QUALITY_PASS | 多脚本重复账户撮合 | PARAMETERIZED_CALL | brick_validation_runner_20260623 | 2026-06-28 |
| BRICK_W_RANKER | BRICK W排序 | `utils/tmp/run_brick_w_top5_robustness_20260625.py` | `W_WEIGHTS`, `w_score`, `rank_by_score` | FROZEN_EXPERIMENT | 候选DataFrame、W权重 | rank_score、daily_rank | BRICK W Top5主线 | W权重不得在普通实验中修改 | QUALITY_PASS | 多个W排序重写 | DIRECT_CALL | brick_w_top5_robustness_20260625 | 2026-06-28 |
| BRICK_INDEX_ENV | 指数修复环境标签 | `utils/tmp/run_brick_index_env_w_top5_20260626.py` | `load_index_data`, `attach_index_env` | FROZEN_EXPERIMENT | validation_end、trade_plan | 指数MACD环境字段 | BRICK index_repair过滤 | 指数数据补齐在结果目录，不覆盖原始数据 | QUALITY_PASS | 多脚本重复指数环境 | DIRECT_CALL | brick_index_env_w_top5_20260626 | 2026-06-28 |
| BRICK_POSITION_FILTER | trend_dev偏离过滤 | `utils/tmp/run_brick_position_w_top5_20260626.py` | `add_position_columns` | FROZEN_EXPERIMENT | feature DataFrame | trend_dev、trend_dev_gt_8等 | BRICK avoid_trend_dev_gt_8 | 早期样本长周期字段可能缺失 | QUALITY_PASS | 多脚本重复位置字段 | DIRECT_CALL | brick_position_w_top5_20260626 | 2026-06-28 |
| BRICK_FINAL_CANDIDATE | BRICK最终validation候选 | `utils/brick_final_candidate.py` | `FINAL_CANDIDATE_NAME`, `FINAL_BRICK_PARAMS`, latest candidate helpers | FROZEN | 最新日候选输入；必须有本地上证指数快照覆盖`signal_date`；不允许运行期外部取数 | 每日候选输出 | BRICK_FINAL_CANDIDATE_V1 | 不等于历史账户回测runner；不得改W权重和正式候选名；缺少本地指数快照时必须fail-fast | VALIDATION_READY | 旧BRICK候选口径 | DIRECT_CALL | brick_validation_final_20260628 | 2026-07-09 |

## 待标准化能力

| capability_id | capability_name | standard_implementation | function_or_class | status | input_contract | output_contract | known_scope | known_limitations | quality_status | replacement_for | reuse_rule | owner_experiment | last_verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BUYABLE_T1_CHECK | T+1可买检查 | 多个实验runner | 待统一 | NEEDS_STANDARDIZATION | signal row、feature DataFrame | buyable bool、skip_reason、entry open | 股票/ETF账户层 | 停牌、涨停、零量口径散落多脚本 | MIXED | 多脚本买入检查 | THIN_ADAPTER | pending | 2026-07-06 |
| LIMIT_UP_DOWN_FILL | 涨跌停不可成交处理 | 多个实验runner | 待统一 | NEEDS_STANDARDIZATION | OHLCV、涨跌停规则、板块/日期 | 可买/可卖/顺延 | A股账户层 | 当前历史脚本多为近似处理 | MIXED | 分散涨跌停判断 | THIN_ADAPTER | pending | 2026-07-06 |
| PROFIT_FACTOR | Profit Factor统计 | 多个summary/account函数 | 待统一 | NEEDS_STANDARDIZATION | 收益序列或账户成交 | PF值和scope | 信号层/账户层/窗口层 | 必须标明gross/net、signal/account scope | MIXED | 多脚本PF | THIN_ADAPTER | pending | 2026-07-06 |
| MAX_DRAWDOWN | 最大回撤统计 | 多个account_metrics/curve函数 | 待统一 | NEEDS_STANDARDIZATION | 净值曲线或单笔序列 | MDD和scope | 账户收盘净值/盘中压力/单笔 | 字段名`max_drawdown`易混用 | MIXED | 多脚本MDD | THIN_ADAPTER | pending | 2026-07-06 |
| WALK_FORWARD | walk-forward统计 | 多个实验脚本 | 待统一 | NEEDS_STANDARDIZATION | equity/trade/period config | 分段表现 | validation稳定性 | 当前多为固定窗口统计，不等于严格样本外 | MIXED | 多脚本WF | THIN_ADAPTER | pending | 2026-07-06 |
| RETURN_CONCENTRATION | 收益集中度 | 多个实验脚本 | 待统一 | NEEDS_STANDARDIZATION | account trades、行业映射、时间字段 | topN贡献、月/行业/股票贡献 | 策略稳定性评估 | 字段名和分组口径需统一 | MIXED | 多脚本集中度 | THIN_ADAPTER | pending | 2026-07-06 |
| QUALITY_CHECK_FRAMEWORK | quality_check统一框架 | 多个实验脚本 | 待统一 | NEEDS_STANDARDIZATION | 实验配置、计划、结果 | quality_check.csv | 所有实验 | 需新增复用治理字段 | MIXED | 手写quality_check | THIN_ADAPTER | pending | 2026-07-06 |
| BOOTSTRAP_STATS | bootstrap统计 | 多个实验脚本 | 待统一 | NEEDS_STANDARDIZATION | 样本收益/指标、seed、抽样次数 | CI、差异检验 | 信号层统计验证 | seed、抽样口径不统一 | MIXED | 多脚本bootstrap | THIN_ADAPTER | pending | 2026-07-06 |

## 禁止复用或仅历史复原

| capability_id | capability_name | standard_implementation | function_or_class | status | input_contract | output_contract | known_scope | known_limitations | quality_status | replacement_for | reuse_rule | owner_experiment | last_verified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LEGACY_INTRADAY_TRAIL_GAP_BUG | 旧版盘中动态锁盈 | 旧B3/BRICK退出实验脚本 | legacy intraday trail | INVALID | OHLCV、trail阈值 | 旧交易结果 | 历史复原 | 跳空低开可能按更高trail_price成交 | INVALID | corrected_intraday_trail_exit | DO_NOT_REUSE | b3_brick_exit_style legacy | 2026-07-03 |
| OPTIMISTIC_TRAIL_SENSITIVITY | optimistic移动止盈/锁盈 | ETF卖法敏感性脚本 | optimistic trail/lock variants | HISTORICAL_ONLY | OHLCV、trail/lock阈值 | 乐观敏感性结果 | ETF卖法探索 | 只能做敏感性，不作保守冠军 | HISTORICAL_ONLY | corrected/close trail | REFERENCE_ONLY | brick_etf_sell_rule_robustness_20260629 | 2026-06-29 |

## 更新规则

必须更新本注册表的情况：

- 新增正式公共组件；
- 实验组件冻结；
- 组件被废弃；
- 发现已确认错误；
- 标准实现迁移；
- 输入输出契约变化；
- 第三次复用后完成公共化。

普通一次性分析字段不需要登记。未经确认，不得为了登记而重构历史脚本。
