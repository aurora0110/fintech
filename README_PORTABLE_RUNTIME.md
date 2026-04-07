# Portable Runtime Notes

## 目标

让另一台电脑在拉取仓库后，能够直接安装依赖并运行：

- `/Users/lidongyang/Desktop/Qstrategy/main.py`
- `BRICK`
- `BRICK_RELAXED_FUSION`
- `BRICK_CASE_RANK_LGBM_TOP20`
- 以及 `B3 / 单针 / 止损止盈` 等主流程过滤器

## 安装依赖

```bash
pip install -r requirements.txt
```

## 目录约定

默认按仓库根目录相对路径寻找：

- `data/`
- `results/`
- `config/`

也可以用环境变量覆盖：

- `QSTRATEGY_DATA_DIR`
- `QSTRATEGY_RESULTS_DIR`
- `QSTRATEGY_CONFIG_DIR`

## 运行主流程

```bash
QSTRATEGY_NON_INTERACTIVE=1 python3 main.py
```

可选环境变量：

- `QSTRATEGY_MAIN_WORKERS`
  - 控制主流程并行进程数
- `QSTRATEGY_NON_INTERACTIVE=1`
  - 跳过持仓交互输入

## BRICK 相关说明

### 1. `brick_filter.py`

这是逐票规则过滤器，只依赖：

- `data/<YYYYMMDD>/normal/*.txt`

### 2. `brickfilter_relaxed_fusion.py`

这是横截面排序器，优先读取历史实验产物：

- `results/brick_comprehensive_lab_full_20260325_r1`

如果另一台电脑没有这份结果目录：

- `main.py` 不会中断
- 该过滤器会自动返回空列表

也可以显式指定结果目录：

- `QSTRATEGY_BRICK_RELAXED_FUSION_RESULT_DIR`

### 3. `brickfilter_case_rank_lgbm_top20.py`

这条线优先依赖两类结果目录：

- 模型搜索结果目录
- 日级流结果目录

如果缺少这些目录：

- `main.py` 不会中断
- 该过滤器会自动返回空列表

也可以显式指定：

- `QSTRATEGY_BRICK_CASE_RANK_MODEL_RESULT_DIR`
- `QSTRATEGY_BRICK_CASE_RANK_DAILY_STREAM_RESULT_DIR`

## 数据说明

本仓库默认不提交：

- `data/`
- `results/`

因此如果要在另一台电脑上得到与本机接近的筛选结果，需要同步：

- 历史日线数据目录 `data/`
- 如需启用两个 BRICK 横截面排序器，还需要同步对应的 `results/` 实验产物

如果只同步代码而不带这些数据：

- `main.py` 仍然可以启动
- 但依赖实验产物的横截面过滤器会返回空结果
