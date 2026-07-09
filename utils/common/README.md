# utils/common

本目录用于存放 Qstrategy 的公共函数库，目标是把重复出现的底层能力统一沉淀到单一位置。

当前内容：

- `io.py`：日线读取、列标准化、文件名股票代码解析
- `indicators.py`：MA、EMA、MACD、KDJ、RSI、趋势线、多空线、偏离度、PIN值
- `dates.py`：交易日偏移辅助
- `brick.py`：BRICK 底层计算、连砖结构、最大量阴线识别
- `backtest.py`：手续费、印花税、100股整数倍、等分资金
- `report.py`：结果目录、CSV 保存、Markdown 报告
- `validation.py`：字段、样本数、权益合法性检查

使用原则：

1. 这里只放底层公共函数，不放具体策略排序和买卖逻辑。
2. 新增脚本优先 import 本目录函数，不再复制旧实现。
3. 若业务口径与公共函数不同，必须显式新建函数名，不得偷偷改公共口径。

