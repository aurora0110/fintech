from datetime import datetime
from dateutil.relativedelta import relativedelta

"""
入参
"""
# 票代码
symbol = "sh000001"
symbol_list = ["sh515450","sh563300","sh512580","sh588000","sz159985","sh520990","sh510300","sh510050","sh518880"
               ,"sh512660","sh512100","sh512170","sh513180","sz159920","sh512980","sh515180","sz159938","sh512880"
               ,"sh512070"]
# 存量数据路径和计算年份
file_path = '/Users/lidongyang/Desktop/MYINVESTSTRATEGY/'

# 均线窗口
windows=[20, 30, 60, 120]

# 绘制各条均线颜色
colors = ['orange', 'green', 'red', 'blue']

# 获取当前时间
now = datetime.now()
end_date = now.strftime("%Y%m%d")

years_ago =  now - relativedelta(years=5) # 获取5年前的日期
start_date = years_ago.strftime("%Y%m%d")

# 回测参数，amount--投入总金额，ineterval_days--最小操作时间，shares--可操作份数
amount = 20000
ineterval_days = 1
shares = 10

print(f"起始日期：{start_date}, 结束日期：{end_date}")