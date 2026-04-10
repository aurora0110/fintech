# BRICK 独立交易看板

这套看板只覆盖：

- `brick_list`
- `brick_case_rank_lgbm_top20_list`

并且使用独立账本：

- 不回写 `config/holding.yaml`
- 不影响 `main.py` 当前持仓逻辑

## 启动方式

先安装依赖：

```bash
pip install -r /Users/lidongyang/Desktop/Qstrategy/requirements.txt
```

再启动：

```bash
streamlit run /Users/lidongyang/Desktop/Qstrategy/utils/dashboard/brick_panel/app.py
```

## 功能

- 自动从 `results/YYYYMMDD.json` 导入 `brick` 与 `brick_case_rank` 信号
- 记录独立持仓与历史交易
- 展示：
  - 当前持仓
  - 历史交易
  - 策略累计收益
  - 最近 5 个交易日选股篮子到最新收盘价的等权收益

## 预计卖出价展示

- 止损价：买入 K 线最低价
- ATR 止盈价：`entry_price + ATR20 * 2.0`
- 固定止盈价：
  - `entry_price * 1.03`
  - `entry_price * 1.08`

## 注意

- 当前持仓超过 3 个交易日会在页面中标红提醒
- 第一版只做手动记账，不做自动卖出
