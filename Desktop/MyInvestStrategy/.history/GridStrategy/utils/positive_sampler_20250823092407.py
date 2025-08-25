import os, glob, re
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from GridStrategy.main import file_path
import getData
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys, termios
import pandas as pd
from pathlib import Path

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
file_path = '/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/data/'

stock_symbol_list = []

# 获取当前时间
now = datetime.now()
end_date = now.strftime("%Y%m%d")

stock_years_ago =  now - relativedelta(years=1) # 获取n年前的日期
stock_start_date = stock_years_ago.strftime("%Y%m%d")

data_new = getData.batch_download_etf_data(stock_symbol_list, "all", stock_start_date, end_date, 5)
print(data_new)
#for key, value in data_new.items():
 #   getData.save_2_csv(value, key, file_path)