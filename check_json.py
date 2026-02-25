"""
JSON筛选结果检查工具

功能说明：
1. 读取股票筛选结果的JSON文件
2. 打印各个策略的筛选结果数量，包括：
   - B1买入条件
   - B3买入条件
   - 持有股票监控
   - 卖出股票列表
3. 显示B1买入条件的前5个股票代码，用于确认数据格式

使用场景：
- 快速检查筛选结果的数量分布
- 验证JSON数据格式是否正确
- 确认筛选策略的执行效果
"""
import json

# 读取JSON文件
with open('/Users/lidongyang/Desktop/Qstrategy/results/20260206.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 打印各个策略的筛选结果数量
print('B1买入条件:', len(data['b1_list']))
print('B3买入条件:', len(data['b3_list']))
print('持有股票监控:', len(data['hold_list']))
print('卖出股票列表:', len(data['sell_list']))

# 打印前几个股票代码，确认数据格式
print('\nB1买入条件前5个股票:')
for item in data['b1_list'][:5]:
    print(item[0])
