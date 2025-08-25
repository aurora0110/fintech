
from calendar import month
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from pathlib import Path
import getData

# ---- 列名适配（中英 & 不区分大小写）----
COL_MAP = {
    "日期":"date","时间":"date","trade_date":"date","date":"date",
    "开盘":"open","开盘价":"open","open":"open",
    "最高":"high","最高价":"high","high":"high",
    "最低":"low","最低价":"low","low":"low",
    "收盘":"close","收盘价":"close","close":"close","pct_chg":"pct_chg",
    "成交量":"volume","成交额":"amount","volume":"volume","amount":"amount",
    "代码":"code","股票代码":"code","ts_code":"code","code":"code",
    "名称":"name","股票名称":"name","name":"name"
}
file_path = '/Users/lidongyang/Desktop/涨停20250822.csv'
file_path = Path(file_path)
stock_symbol_list = []
df = pd.read_csv(file_path)
# 遍历 DataFrame 的每一行
for _, row in df.iterrows():
    code = str(row[0]).strip().zfill(6)   # 股票代码，转为字符串
    name = str(row[1]).strip()   # 股票名称
    if code.startswith('#'):
        continue
    else:
        stock_symbol_list.append(f'{code}')

print(stock_symbol_list)
# 获取当前时间
now = datetime.now()
end_date = now.strftime("%Y%m%d")

stock_years_ago =  now - relativedelta(months=6)
stock_start_date = stock_years_ago.strftime("%Y%m%d")

data_new = getData.batch_download_stock_data(stock_symbol_list, "all", stock_start_date, end_date, 5)
out = pd.concat(data_new, names=["code"]).reset_index(level=0).rename(columns={"level_0": "code"})
out.to_csv("positive_combined.csv", index=False)
