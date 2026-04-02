from __future__ import annotations

"""
BRICK case_rank_lgbm_top20
==========================

这条过滤器不是普通单票规则，而是：

1. 先在目标交易日从全市场构建 `brick.case_first` 候选池；
2. 再用冻结的完美案例排序冠军模型打分；
3. 用“完美砖型图正样本分数的 10% 分位”做阈值；
4. 最后仅保留高于阈值的股票，并按分数降序取当日最多 20 只。

为什么把“训练步骤”写在这里
--------------------------
因为这条策略后续会随着你继续补充完美砖型案例而持续迭代。
这个文件的目标之一，就是让你未来只看这里也能想起来：

- 训练集怎么造
- 模型怎么选
- 日级流怎么重建
- 阈值怎么更新
- 最后 `main.py` 用的到底是哪一份结果

新增完美案例后的完整更新流程
--------------------------
当你在：

- `/Users/lidongyang/Desktop/Qstrategy/data/完美图/砖型图`

继续补充新的完美砖型图案例后，按下面顺序重做即可。

整个工作流涉及的代码文件（按先后顺序）
--------------------------------
第 0 步：案例与语义基础

1. 完美案例目录
   - `/Users/lidongyang/Desktop/Qstrategy/data/完美图/砖型图`
   - 这是新增/维护正样本案例的地方

2. 案例语义与特征抽取
   - `/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/brick_case_semantics_v1_20260326.py`
   - 负责：
     - 完美案例图片解析
     - `code_key`
     - 分型语义
     - 风险分布特征

3. 候选池构建器
   - `/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/brickfilter_case_first_v1_20260326.py`
   - 负责：
     - 全市场候选构建
     - `build_candidates_for_date`
     - `build_candidate_cache_for_dates`

4. 案例召回与 enrich
   - `/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/brickfilter_case_recall_v1_20260327.py`
   - 负责：
     - 候选 enrich
     - 相似度相关字段
     - `recall_score`

第 1 步：重新搜索冠军排序模型

5. 模型搜索主脚本
   - `/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_model_search_v1_20260327.py`
   - 负责：
     - 读取第 0 步候选与完美案例
     - 构建 `candidate_dataset.csv`
     - 比较 `heuristic / logreg / rf / xgb / lgbm`
     - 输出冠军模型和参数

第 2 步：把冠军模型应用到全年日级流

6. 日级流基础版脚本
   - `/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_daily_stream_v1_20260328.py`
   - 负责：
     - 读取第 1 步冠军模型
     - 在全市场、全交易日上重建日级打分流

7. 日级流优化版脚本
   - `/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_daily_stream_v2_20260328.py`
   - 负责：
     - 第 2 步的优化实现
     - 分块落盘
     - 进度条
     - 当前正式日级流默认应优先使用这版

第 3 步：主流程接入与日常扫描

8. 当前过滤器本体
   - `/Users/lidongyang/Desktop/Qstrategy/utils/brickfilter_case_rank_lgbm_top20.py`
   - 负责：
     - 读取第 1 步训练集和冠军模型
     - 自动计算正样本 `10%` 分位阈值
     - 优先复用第 2 步日级打分流
     - 对目标扫描日筛出高于阈值且当日前 20 的股票

9. 主流程入口
   - `/Users/lidongyang/Desktop/Qstrategy/main.py`
   - 负责：
     - 加载本过滤器
     - 在日常扫描时把结果展示并写入结果 JSON

10. 结果展示
    - `/Users/lidongyang/Desktop/Qstrategy/utils/selectprint.py`
    - 负责：
      - 在命令行里把 `BRICK_CASE_RANK_LGBM_TOP20` 正确打印出来

第 4 步：如果要重做正式账户层实验

11. 固定止盈/止损基线
    - `/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_phase1_fixed_exit_search_v2_20260329.py`

12. ATR 动态止盈
    - `/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_phase2_atr_search_v2_20260402.py`

13. histvol / shapevol / 利润保护 / 补位
    - 这些是第 4 步以后继续扩展的账户层实验脚本
    - 它们都必须建立在第 2 步新的全年日级输出之上

Step 1. 重新跑完美案例排序模型搜索
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
脚本：

- `/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_model_search_v1_20260327.py`

目的：

- 基于更新后的完美砖型图案例，重新生成 `candidate_dataset.csv`
- 在 `heuristic / logreg / rf / xgb / lgbm` 中重新搜索冠军模型
- 评价标准不是账户收益，而是：
  - `recall@20`
  - `MRR`
  - 完美案例在当天候选中的排序能力

核心产物：

- `candidate_dataset.csv`
- `best_config_by_model.csv`
- `model_validation_summary.csv`
- `model_full_coverage_summary.csv`
- `best_model_top20_candidates.csv`
- `summary.json`

你最应该看：

- `summary.json`
  - `best_model_name`
  - `best_model_params`
  - `best_model_summary`

如果这里冠军模型发生变化（比如不再是 `lgbm`），这条过滤器就要跟着更新。

Step 2. 重新跑全年日级出票流
~~~~~~~~~~~~~~~~~~~~~~~~~~
脚本：

- `/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_daily_stream_v2_20260328.py`

目的：

- 把 Step 1 得到的新冠军模型，应用到全市场、全交易日候选池上
- 重新生成真实全年日级输出，而不是只在完美案例日期上输出

核心产物：

- `daily_scored_candidates.csv`
- `daily_top20_candidates.csv`
- `daily_top50_candidates.csv`
- `daily_top100_candidates.csv`
- `summary.json`

你最应该看：

- `summary.json`
  - 日期覆盖是否正常
  - `2025/2026` 是否仍有候选
- `daily_scored_candidates.csv`
  - 这是本过滤器当前优先复用的预计算日级打分流

Step 3. 阈值自动随训练集更新
~~~~~~~~~~~~~~~~~~~~~~~~~
本文件不会手写一个固定阈值。

当前实现会自动：

1. 读取 Step 1 生成的 `candidate_dataset.csv`
2. 用 Step 1 的冠军模型参数重新拟合冻结模型
3. 对训练集重新打分
4. 取其中 `label = 1` 的正样本分数
5. 计算这些正样本分数的 `10%` 分位数，作为新阈值

也就是说：

- 只要你重跑了 Step 1
- 这条过滤器下次运行时就会自动拿到新的阈值

Step 4. 若日级预计算流已更新，本过滤器直接复用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
如果扫描数据目录仍是当前快照：

- `/Users/lidongyang/Desktop/Qstrategy/data/20260324`

则本过滤器会优先直接读取：

- Step 2 生成的 `daily_scored_candidates.csv`

避免每次在 `main.py` 里重新做全市场候选构建和模型打分。

如果以后你把正式快照目录切到新的日期，例如：

- `/Users/lidongyang/Desktop/Qstrategy/data/20260415`

那记得同步更新本文件里的：

- `SNAPSHOT_DATA_DIR`
- `PREBUILT_DAILY_SCORED_CSV`

让它继续优先复用最新的日级流。

Step 5. 若冠军模型结构变化，需要同步改这里
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
当前这条过滤器默认假设：

- 冠军模型来自 `run_brick_case_rank_model_search_v1_20260327.py`
- 特征列与 `rank_model.FEATURE_COLS` 一致
- 模型仍能通过 `phase0._load_frozen_best_model()` 重建

如果后续你改了：

- 特征工程
- 模型搜索脚本
- 冠军模型类型
- 训练数据位置

那至少要检查并同步更新本文件中的这些常量：

- `MODEL_SEARCH_RESULT`
- `TRAIN_DATASET_CSV`
- `PHASE0_BASE_SCRIPT`
- `PHASE0_RESULT_DIR`
- `PREBUILT_DAILY_SCORED_CSV`
- `SNAPSHOT_DATA_DIR`

这 6 个位置如果不对，过滤器会继续跑，但语义就可能变旧。

Step 6. 验证主流程
~~~~~~~~~~~~~~~~
当 Step 1 和 Step 2 都更新后，最后验证：

- `python3 /Users/lidongyang/Desktop/Qstrategy/main.py`

应确认两件事：

1. `main.py` 能正常打印：
   - `【BRICK case_rank_lgbm_top20 策略】`
2. 结果 JSON 中含有：
   - `brick_case_rank_lgbm_top20_list`

这就说明：

- 新案例
- 新模型
- 新阈值
- 新日级流
- 新主流程接入

已经整条链打通。

当前这条策略的语义边界
--------------------
这条过滤器本质上是：

- “当日全市场横截面排序器”

而不是：

- “单只股票绝对规则检查器”

因此它和 `brick_filter.py` 最大的区别是：

- `brick_filter.py` 可以逐票 `check(file)`
- 这条线必须 `scan_dir(...)` 后统一排序取前 20

不要把它误改回逐票独立判断，否则“每天最多 20 只”的语义会坏掉。
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
PHASE0_BASE_SCRIPT = ROOT / "utils" / "brick_optimize" / "run_brick_case_rank_daily_stream_v1_20260328.py"
SNAPSHOT_DATA_DIR = ROOT / "data" / "20260324"

STRATEGY_NAME = "BRICK_CASE_RANK_LGBM_TOP20"
TOP_N = 20
POSITIVE_SCORE_QUANTILE = 0.10

_MODULE_CACHE: dict[str, Any] = {}
_RUNTIME_CACHE: Optional[dict[str, Any]] = None


def _find_latest_finished_result_dir(prefix: str, required_files: list[str], env_name: str) -> Path:
    explicit = os.environ.get(env_name, "").strip()
    if explicit:
        path = Path(explicit)
        missing = [name for name in required_files if not (path / name).exists()]
        if missing:
            raise RuntimeError(f"{env_name} 指向的目录缺少文件: {missing} -> {path}")
        return path

    candidates: list[tuple[float, Path]] = []
    for path in (ROOT / "results").glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        missing = [name for name in required_files if not (path / name).exists()]
        if missing:
            continue
        summary_mtime = (path / "summary.json").stat().st_mtime if (path / "summary.json").exists() else 0.0
        progress_mtime = (path / "progress.json").stat().st_mtime if (path / "progress.json").exists() else 0.0
        sort_key = max(summary_mtime, progress_mtime, path.stat().st_mtime)
        candidates.append((sort_key, path))
    if not candidates:
        raise RuntimeError(f"未找到可用结果目录: {prefix}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _resolve_model_search_result_dir() -> Path:
    return _find_latest_finished_result_dir(
        prefix="brick_case_rank_model_search_v1_",
        required_files=["candidate_dataset.csv", "summary.json"],
        env_name="QSTRATEGY_BRICK_CASE_RANK_MODEL_RESULT_DIR",
    )


def _resolve_phase0_result_dir() -> Path:
    return _find_latest_finished_result_dir(
        prefix="brick_case_rank_daily_stream_v2_full_",
        required_files=["daily_scored_candidates.csv", "summary.json"],
        env_name="QSTRATEGY_BRICK_CASE_RANK_DAILY_STREAM_RESULT_DIR",
    )


def _load_module(path: Path, module_name: str):
    cache_key = f"{module_name}:{path}"
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


phase0 = _load_module(PHASE0_BASE_SCRIPT, "brick_case_rank_lgbm_top20_phase0_base")
rank_model = phase0.rank_model
case_first = phase0.case_first
case_recall = phase0.case_recall


def strategy_name() -> str:
    return STRATEGY_NAME


def strategy_description() -> str:
    return (
        "BRICK case_rank_lgbm_top20：使用冻结的完美砖型案例排序冠军模型，"
        "以完美案例正样本分数的10%分位作为阈值，只保留高于阈值的候选，"
        "再按当日模型分数排序取前20。"
    )


def operation_suggestion() -> str:
    return (
        "这条线不是单票绝对规则，而是全市场横截面排序器；"
        "应和 relaxed_fusion 一样按日统一跑，再看前20，而不是逐票误判。"
    )


def execution_rule_summary() -> str:
    return (
        "这里只负责给出当日 ranked candidates；"
        "具体买卖仍应使用账户层统一回测或后续执行策略。"
    )


def _code_key(value: Any) -> str:
    text = str(value)
    match = re.search(r"(\d{6})", text)
    return match.group(1) if match else text


def _resolve_target_date(data_dir: str | Path) -> pd.Timestamp:
    data_path = Path(data_dir)
    if data_path.name == "normal" and re.fullmatch(r"\d{8}", data_path.parent.name or ""):
        return pd.to_datetime(data_path.parent.name, format="%Y%m%d")

    all_dates = phase0.collect_trading_dates(Path(data_dir), max_workers=1)
    if not all_dates:
        raise RuntimeError(f"未能从 {data_dir} 解析出目标交易日")
    return pd.Timestamp(all_dates[-1])


def _normalize_scan_data_dir(data_dir: str | Path) -> Path:
    data_path = Path(data_dir)
    if data_path.name == "normal":
        return data_path.parent
    return data_path


def _load_prebuilt_scored_for_date(target_date: pd.Timestamp, data_dir: str | Path) -> pd.DataFrame:
    scan_data_dir = _normalize_scan_data_dir(data_dir)
    try:
        if scan_data_dir.resolve() != SNAPSHOT_DATA_DIR.resolve():
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    prebuilt_daily_scored_csv = _resolve_phase0_result_dir() / "daily_scored_candidates.csv"
    if not prebuilt_daily_scored_csv.exists():
        return pd.DataFrame()
    cols = ["code", "signal_date", "signal_low", "signal_close", "model_score", "sort_score"]
    df = pd.read_csv(prebuilt_daily_scored_csv, usecols=cols, parse_dates=["signal_date"])
    df = df[df["signal_date"] == pd.Timestamp(target_date)].copy()
    if df.empty:
        return df
    return df.sort_values(["model_score", "code"], ascending=[False, True]).reset_index(drop=True)


def _load_runtime_bundle() -> dict[str, Any]:
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE

    model_search_result_dir = _resolve_model_search_result_dir()
    phase0_result_dir = _resolve_phase0_result_dir()
    dataset = pd.read_csv(model_search_result_dir / "candidate_dataset.csv")
    dataset["signal_date"] = pd.to_datetime(dataset["signal_date"])
    dataset = rank_model._prepare_features(dataset)

    best_name, best_params, model = phase0._load_frozen_best_model()
    dataset["model_score"] = phase0._score_with_model(model, dataset)
    positive = dataset[pd.to_numeric(dataset["label"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if positive.empty:
        raise RuntimeError("训练样本中没有正样本，无法计算 case_rank 阈值")

    threshold = float(pd.to_numeric(positive["model_score"], errors="coerce").quantile(POSITIVE_SCORE_QUANTILE))
    _RUNTIME_CACHE = {
        "model_search_result_dir": str(model_search_result_dir),
        "phase0_result_dir": str(phase0_result_dir),
        "best_model_name": best_name,
        "best_model_params": best_params,
        "model": model,
        "threshold": threshold,
        "positive_count": int(len(positive)),
    }
    return _RUNTIME_CACHE


def _score_candidates_for_date(
    target_date: pd.Timestamp,
    data_dir: str | Path,
    max_workers: int,
) -> pd.DataFrame:
    prebuilt = _load_prebuilt_scored_for_date(target_date, data_dir)
    if not prebuilt.empty:
        return prebuilt

    scan_data_dir = _normalize_scan_data_dir(data_dir)
    try:
        candidate_df = case_first.build_candidates_for_date(
            target_date=target_date,
            data_dir=scan_data_dir,
            max_workers=max_workers,
            required_lens=case_recall.CASE_SEQ_LENS,
        )
    except PermissionError:
        candidate_df = case_first.build_candidates_for_date(
            target_date=target_date,
            data_dir=scan_data_dir,
            max_workers=1,
            required_lens=case_recall.CASE_SEQ_LENS,
        )
    if candidate_df.empty:
        return pd.DataFrame()

    enriched = case_recall.enrich_candidates_for_date(pd.Timestamp(target_date), candidate_df, scan_data_dir)
    if enriched.empty:
        return pd.DataFrame()

    bundle = _load_runtime_bundle()
    enriched = rank_model._prepare_features(enriched)
    enriched["model_score"] = phase0._score_with_model(bundle["model"], enriched)
    enriched["sort_score"] = pd.to_numeric(enriched["model_score"], errors="coerce").fillna(0.0)
    enriched["signal_date"] = pd.to_datetime(enriched["signal_date"])
    return enriched.sort_values(["model_score", "code"], ascending=[False, True]).reset_index(drop=True)


def scan_dir(
    data_dir: str | Path,
    hold_list: Optional[list[Any]] = None,
    max_workers: int = 8,
) -> list[list[str]]:
    del hold_list

    target_date = _resolve_target_date(data_dir)
    scored = _score_candidates_for_date(target_date=target_date, data_dir=data_dir, max_workers=max_workers)
    if scored.empty:
        return []

    bundle = _load_runtime_bundle()
    threshold = float(bundle["threshold"])
    filtered = scored[pd.to_numeric(scored["model_score"], errors="coerce").fillna(-1.0) >= threshold].copy()
    if filtered.empty:
        return []

    filtered = filtered.sort_values(["model_score", "code"], ascending=[False, True]).head(TOP_N).reset_index(drop=True)

    out: list[list[str]] = []
    for row in filtered.itertuples(index=False):
        code = _code_key(getattr(row, "code"))
        stop_ref = round(float(getattr(row, "signal_low")), 2)
        close_ref = round(float(getattr(row, "signal_close")), 2)
        score = round(float(getattr(row, "model_score")), 6)
        note = (
            f"signal_date={pd.Timestamp(getattr(row, 'signal_date')).strftime('%Y-%m-%d')}; "
            f"q10={threshold:.6f}; rank_top{TOP_N}"
        )
        out.append([code, str(stop_ref), str(close_ref), str(score), note])
    return out


def debug_summary() -> dict[str, Any]:
    bundle = _load_runtime_bundle()
    return {
        "strategy_name": STRATEGY_NAME,
        "model_search_result_dir": bundle["model_search_result_dir"],
        "phase0_result_dir": bundle["phase0_result_dir"],
        "threshold_quantile": POSITIVE_SCORE_QUANTILE,
        "threshold": float(bundle["threshold"]),
        "positive_count": int(bundle["positive_count"]),
        "best_model_name": str(bundle["best_model_name"]),
        "best_model_params": bundle["best_model_params"],
    }


def retrain_workflow_summary() -> dict[str, Any]:
    """
    给未来维护时快速查看“新增完美案例后如何完整更新模型”的摘要。
    """
    return {
        "step_1_model_search_script": "/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_model_search_v1_20260327.py",
        "step_1_key_outputs": [
            "candidate_dataset.csv",
            "best_config_by_model.csv",
            "model_validation_summary.csv",
            "model_full_coverage_summary.csv",
            "best_model_top20_candidates.csv",
            "summary.json",
        ],
        "step_2_daily_stream_script": "/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_daily_stream_v2_20260328.py",
        "step_2_key_outputs": [
            "daily_scored_candidates.csv",
            "daily_top20_candidates.csv",
            "daily_top50_candidates.csv",
            "daily_top100_candidates.csv",
            "summary.json",
        ],
        "step_3_threshold_rule": "用 Step 1 的 candidate_dataset.csv 中 label=1 正样本，重新打分后取 model_score 的 10% 分位",
        "step_4_main_validation": [
            "python3 /Users/lidongyang/Desktop/Qstrategy/main.py",
            "确认结果 JSON 中含有 brick_case_rank_lgbm_top20_list",
        ],
        "result_dir_resolution": "默认自动读取最新成功的模型搜索结果目录与最新成功的日级出票流目录；必要时可通过环境变量 QSTRATEGY_BRICK_CASE_RANK_MODEL_RESULT_DIR 和 QSTRATEGY_BRICK_CASE_RANK_DAILY_STREAM_RESULT_DIR 显式覆盖。",
    }


def retrain_workflow_text() -> str:
    """
    返回便于人工直接阅读的工作流摘要文本。
    """
    summary = retrain_workflow_summary()
    lines = [
        "case_rank_lgbm_top20 更新流程：",
        f"1. 模型搜索：{summary['step_1_model_search_script']}",
        f"   输出：{', '.join(summary['step_1_key_outputs'])}",
        f"2. 全年日级出票流：{summary['step_2_daily_stream_script']}",
        f"   输出：{', '.join(summary['step_2_key_outputs'])}",
        f"3. 阈值规则：{summary['step_3_threshold_rule']}",
        "4. 主流程验证：",
    ]
    lines.extend([f"   - {item}" for item in summary["step_4_main_validation"]])
    return "\n".join(lines)


def workflow_code_index() -> dict[str, Any]:
    """
    工作流涉及的代码文件索引，按先后顺序列出。
    """
    return {
        "phase_0_case_and_candidate_foundation": [
            "/Users/lidongyang/Desktop/Qstrategy/data/完美图/砖型图",
            "/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/brick_case_semantics_v1_20260326.py",
            "/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/brickfilter_case_first_v1_20260326.py",
            "/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/brickfilter_case_recall_v1_20260327.py",
        ],
        "phase_1_model_search": [
            "/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_model_search_v1_20260327.py",
        ],
        "phase_2_daily_stream": [
            "/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_daily_stream_v1_20260328.py",
            "/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_daily_stream_v2_20260328.py",
        ],
        "phase_3_live_filter_and_entry": [
            "/Users/lidongyang/Desktop/Qstrategy/utils/brickfilter_case_rank_lgbm_top20.py",
            "/Users/lidongyang/Desktop/Qstrategy/main.py",
            "/Users/lidongyang/Desktop/Qstrategy/utils/selectprint.py",
        ],
        "phase_4_account_level_research": [
            "/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_phase1_fixed_exit_search_v2_20260329.py",
            "/Users/lidongyang/Desktop/Qstrategy/utils/brick_optimize/run_brick_case_rank_phase2_atr_search_v2_20260402.py",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(debug_summary(), ensure_ascii=False, indent=2))
