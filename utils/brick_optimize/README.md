# BRICK Case Rank Optimize Workflow

这个目录集中放置 `case_rank_lgbm_top20` 这条工作流相关的核心脚本，避免继续散落在 `utils/tmp` 中。

## 执行顺序

### 1. 维护正样本案例
- 目录：`/Users/lidongyang/Desktop/Qstrategy/data/完美图/砖型图`
- 作用：新增或修正完美砖型图案例。

### 2. 案例语义与候选基础
- `brick_case_semantics_v1_20260326.py`
- `brickfilter_case_first_v1_20260326.py`
- `brickfilter_case_recall_v1_20260327.py`

作用：
- 解析完美案例图片
- 构建 `brick.case_first` 候选池
- 计算案例召回与相似度 enrich 特征

### 3. 重新搜索冠军排序模型
- `run_brick_case_rank_model_search_v1_20260327.py`

作用：
- 构建 `candidate_dataset.csv`
- 搜索 `heuristic / logreg / rf / xgb / lgbm`
- 以 `recall@20 + MRR` 选出新的冠军模型

主要输出：
- `candidate_dataset.csv`
- `best_config_by_model.csv`
- `model_validation_summary.csv`
- `model_full_coverage_summary.csv`
- `best_model_top20_candidates.csv`
- `summary.json`

### 4. 重建全年日级出票流
- `run_brick_case_rank_daily_stream_v1_20260328.py`
- `run_brick_case_rank_daily_stream_v2_20260328.py`

推荐优先使用 `v2`。

作用：
- 读取第 3 步的冠军模型
- 在全市场、全交易日生成日级打分流

主要输出：
- `daily_scored_candidates.csv`
- `daily_top20_candidates.csv`
- `daily_top50_candidates.csv`
- `daily_top100_candidates.csv`
- `date_coverage_summary.csv`
- `summary.json`

### 5. 主流程日常接入
- 过滤器：`/Users/lidongyang/Desktop/Qstrategy/utils/brickfilter_case_rank_lgbm_top20.py`
- 主入口：`/Users/lidongyang/Desktop/Qstrategy/main.py`

作用：
- 读取第 3 步训练集和冠军模型
- 自动计算正样本 `10%` 分位阈值
- 优先复用第 4 步的 `daily_scored_candidates.csv`
- 当日只保留高于阈值、且按分数排名前 20 的股票

主流程验证：
- 运行：`python3 /Users/lidongyang/Desktop/Qstrategy/main.py`
- 检查：
  - 终端打印 `【BRICK case_rank_lgbm_top20 策略】`
  - 结果 JSON 中出现 `brick_case_rank_lgbm_top20_list`

### 6. 正式账户层回测
- `run_brick_case_rank_final_spec_search_v1_20260327.py`
- `run_brick_case_rank_phase1_fixed_exit_search_v1_20260328.py`
- `run_brick_case_rank_phase1_fixed_exit_search_v2_20260329.py`
- `run_brick_case_rank_phase2_atr_search_v2_20260402.py`

作用：
- 在新的全年日级输入流上重做固定退出、ATR 动态止盈等正式回测

## 使用说明

### 最小更新链路
当你新增完美砖型图案例后，最少要按这个顺序重跑：

1. `run_brick_case_rank_model_search_v1_20260327.py`
2. `run_brick_case_rank_daily_stream_v2_20260328.py`
3. `python3 /Users/lidongyang/Desktop/Qstrategy/main.py`

### 如果要重新做正式策略研究
在上面 3 步完成后，再继续：

4. `run_brick_case_rank_phase1_fixed_exit_search_v2_20260329.py`
5. `run_brick_case_rank_phase2_atr_search_v2_20260402.py`

## 结果目录

### 模型搜索结果
- `/Users/lidongyang/Desktop/Qstrategy/results/brick_case_rank_model_search_v1_*`

### 日级出票流结果
- `/Users/lidongyang/Desktop/Qstrategy/results/brick_case_rank_daily_stream_v1_*`
- `/Users/lidongyang/Desktop/Qstrategy/results/brick_case_rank_daily_stream_v2_*`

### 正式回测结果
- `/Users/lidongyang/Desktop/Qstrategy/results/brick_case_rank_phase1_fixed_exit_search_v2_*`
- `/Users/lidongyang/Desktop/Qstrategy/results/brick_case_rank_phase2_atr_search_v2_*`

## 兼容说明

`utils/tmp` 下同名文件目前保留为薄包装入口，只负责转发到本目录，避免旧脚本和历史命令立即失效。后续如果确认不再需要兼容层，可以再统一清理。
