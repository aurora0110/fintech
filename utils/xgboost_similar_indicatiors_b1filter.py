from __future__ import annotations

"""
B1 当前冠军策略封装版
====================

文件定位
--------
这个文件放在 `b1filter.py` 同级目录，目的是把当前已经验证过的 B1 最优思路
收敛成一个单独的过滤器模块，便于：

1. 在主扫描流程里直接调用；
2. 让“当前冠军策略”有一个稳定文件名，不必每次再去翻实验目录；
3. 避免把冠军逻辑复制多份，后续难维护。

当前策略来源
------------
本文件对应的策略原型，来自这轮全量账户层冠军：

- 结果目录：
  `/Users/lidongyang/Desktop/Qstrategy/results/b1_txt_template_account_v2_full_20260322_201735`
- 冠军组合：
  `fusion_template_hard_full_fusion_pool_txt_confirmed_top3`
  `+ model_plus_tp`
  `+ xgb_score_v2`
  `+ TP20%`

账户层指标：
- 最终净值：1.20154
- 最大回撤：-2.47%
- 交易数：27
- 胜率：77.78%
- 平均单笔收益：7.94%
- 盈亏比：7.60

买点语义
--------
当前 B1 冠军买点，不是简单的 “J<13 就买”，而是：

1. `J` 回到低位；
2. 主形态属于：
   - 上涨后回踩趋势线；
   - 或上涨后回踩多空线；
3. 前面没有明显出货破坏；
4. `关键K / 缩半量 / 倍量柱` 只作为加分项，不是主类型；
5. 再用以下信息做融合评分：
   - 正例模板相似度；
   - 自动近似反例 / 硬反例对比；
   - 自动发现因子；
   - 买点机器学习分数。

卖点口径
--------
注意：本文件只负责“买点筛选和打分”，不直接执行卖出。

这条冠军策略在账户层对应的卖法是：
- `20%` 固定止盈；
- 同时启用 `XGBoost` 卖点模型；
- 两者谁先触发，就在次日开盘退出。

止损口径
--------
- 默认止损：买入 K 线最低价 * 0.90
- 若买入时价格仍位于多空线下方，则按项目永久规则收紧为：
  买入 K 线最低价 * 0.95

实现说明
--------
这里不再重复复制一份完整冠军逻辑，而是复用已经稳定运行的：
- `utils/b1filter_similar_indicators_ml.py`

这样做有两个原因：
1. 当前冠军买点逻辑已经在那份文件里实现并验证过；
2. 这里做“冠军策略专用封装”，可以避免后续逻辑双份漂移。

返回格式
--------
与项目里其他 filter 模块保持一致：
- `[-1]`：不入选
- `[1, stop_loss_price, close_price, score, note]`：入选

备注
----
虽然文件名里带 `xgboost`，但当前买点融合分不是“只用 XGBoost”。
更准确地说，它是：
- 模板相似度
- 反例对比
- 因子
- ML 分数
的融合买点策略；
账户层冠军卖点部分则明确使用了 `xgb_score_v2`。
"""

from utils import b1filter_similar_indicators_ml as _champion_impl


STRATEGY_NAME = "B1_XGBOOST_SIMILAR_INDICATIORS_CHAMPION"

STRATEGY_SUMMARY = (
    "B1 当前冠军买点：J 低位 + 回踩趋势线/多空线 + 无明显出货，"
    "再融合正例模板相似度、近似反例对比、自动发现因子和 ML 分数。"
)

SELL_RULE_SUMMARY = (
    "账户层对应卖法：20% 固定止盈 + XGBoost 卖点模型并行，"
    "谁先触发谁在次日开盘卖出。"
)


def strategy_description() -> str:
    """返回这条冠军策略的简要中文说明。"""
    return f"{STRATEGY_SUMMARY} {SELL_RULE_SUMMARY}"


def check(file_path: str, hold_list=None, feature_cache=None):
    """
    复用当前已经验证通过的 B1 冠军买点实现。

    参数：
    - `file_path`：单只股票日线文件路径
    - `hold_list`：与其他 filter 接口保持一致，当前不使用
    - `feature_cache`：可选的 StrategyFeatureCache，避免重复算特征

    返回：
    - `[-1]`：当前不满足冠军买点
    - `[1, stop_loss_price, close_price, score, note]`：满足冠军买点
    """
    return _champion_impl.check(file_path=file_path, hold_list=hold_list, feature_cache=feature_cache)
