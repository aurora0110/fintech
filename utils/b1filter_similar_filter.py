from __future__ import annotations

"""
B1 相似度冠军过滤器
=================

用途
----
这个文件把当前已经验证过的 B1 冠军买点，收敛成一个稳定的主扫描过滤器模块，
放在 `b1filter.py` 同级目录下，便于：

1. 在 `main.py` 中直接调用；
2. 让当前冠军策略有一个稳定、易理解的文件名；
3. 后续新增案例后，可以继续在不改主入口的前提下更新内部实现。

策略来源
--------
当前对应的是 B1 全量冠军买点：

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
这条策略不是简单的 `J` 超卖信号，而是：

1. `J` 回到低位；
2. 形态属于：
   - 上涨后回踩趋势线；
   - 或上涨后回踩多空线；
3. 前面没有明显出货破坏；
4. `关键K / 缩半量 / 倍量柱` 只是加分项；
5. 最终综合以下信息做买点评分：
   - 成功案例模板相似度；
   - 近似反例 / 硬反例对比；
   - 自动发现因子；
   - 买点机器学习分数。

卖点与止损口径
--------------
这个文件只负责买点筛选，不直接执行卖出。

对应账户层冠军卖法是：
- `20%` 固定止盈；
- `XGBoost` 卖点模型并行；
- 谁先触发，就次日开盘卖出。

止损规则：
- 默认：买入 K 线最低价 * 0.90
- 若买入时价格位于多空线下方，则按项目永久规则收紧为：
  买入 K 线最低价 * 0.95

运行建议
--------
这条策略是“低频高质量”思路：
- 更适合作为重点关注列表，而不是盲目放宽成高频扫票；
- 若当天命中多只，优先选择综合分更高的前几名；
- 若后续账户层执行，优先沿用冠军卖法，不要随意改成纯固定持有。

实现说明
--------
这里不重复复制冠军逻辑，而是直接复用已经验证过的实现：
- `utils/xgboost_similar_indicatiors_b1filter.py`

这样可以避免主逻辑出现双份漂移。
"""

from utils import xgboost_similar_indicatiors_b1filter as _champion_impl


STRATEGY_NAME = "B1_SIMILAR_FILTER_CHAMPION"

STRATEGY_SUMMARY = (
    "B1 当前冠军买点：J 低位，且上涨后回踩趋势线或多空线，"
    "前面无明显出货，再融合模板相似度、反例对比、因子和 ML 分数。"
)

OPERATION_SUGGESTION = (
    "操作建议：优先把这条策略当作高质量候选排序器使用；"
    "若当天命中多只，优先看综合分最高的前 3 名；"
    "账户层优先沿用 20% 固定止盈 + XGBoost 卖点模型并行的冠军卖法。"
)


def strategy_description() -> str:
    """返回策略摘要。"""
    return STRATEGY_SUMMARY


def operation_suggestion() -> str:
    """返回运行时建议。"""
    return OPERATION_SUGGESTION


def check(file_path: str, hold_list=None, feature_cache=None):
    """
    复用当前已经验证通过的 B1 冠军买点实现。

    返回格式与项目其他过滤器保持一致：
    - `[-1]`：不入选
    - `[1, stop_loss_price, close_price, score, note]`：入选
    """
    return _champion_impl.check(file_path=file_path, hold_list=hold_list, feature_cache=feature_cache)
