import akshare as ak
import pandas as pd
from pathlib import Path
import os
import config
from datetime import datetime

def download_stock_data(symbol, type, start_date, end_date, adjust):
    """
    获取市场某只股票数据
    :param symbol: 股票代码
    :param type: etf or stock
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param adjust：qfq（计算技术指标） or hfq（计算分红）
    :return: 股票数据
    """
    try:
        if type == "stock":
            stock_data = ak.stock_zh_a_hist("000001", "daily", start_date, end_date, "qfq") # symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"
            # stock_data = ak.stock_zh_a_hist_tx(symbol = "sz000001", start_date = "19980101", end_date = "20500101", adjust = "",timeout = None)
            #stock_data = ak.stock_zh_a_hist(symbol= "000001",period= "daily",start_date= "19980101",end_date= "20500101",adjust= "",timeout = None)
            print('stock_data:', stock_data)
            return stock_data
        else:
            print("No This Type")
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        return None

def download_etf_data(symbol):
    """
    获取市场全部ETF基金目录和实时数据
    :param symbol: choice of {"封闭式基金", "ETF基金", "LOF基金"}
    param type: etf 
    """
    try:
        if symbol == "etf":
            etf_data = ak.fund_etf_category_sina(symbol="ETF基金")
            return etf_data

        elif symbol == "lof":
            symbol = symbol[2:]
            lof_data = ak.fund_lof_spot_em(symbol="LOF基金")
            return lof_data
        
        else:
            print("No This Symbol")
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        return None

def download_etfInfo_data(type, symbol):
    """
    获取单只基金实时数据
    param type: etf 
    param symbol: 基金代码，带市场前缀，如sh000001
    return: 基金实时数据
    """
    try:
        if type == "etf":
            etf_data = ak.fund_etf_spot_sina(symbol=symbol)
            return etf_data
        else:
            print("No This Type")
    except Exception as e:
        print(f"获取{symbol}实时数据失败: {str(e)}")
        return None

def download_etfHistory_data(type, symbol, start_date, end_date):
    """
    获取单只基金历史数据
    :param symbol: 基金代码，新浪sina接口带市场前缀，如sh000001，东方财富em接口不加
    :return: 基金历史数据
    """
    try:
        if type == "etf":
            symbol = symbol[2:]
            etf_data = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            print(etf_data)
            return etf_data
        elif type == "lof":
            symbol = symbol[2:]
            lof_data = ak.fund_lof_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            return lof_data
        else:
            print("No This Type")
    except Exception as e:
        print(f"获取{type} {symbol}历史数据失败: {str(e)}")
        return None
    
def batch_download_etf_data(symbol_list, days):
    """
    批量获取基金最近N天的数据
    :param symbol_list: 基金代码列表，带市场前缀，如['sh000001', 'sh000002']
    :param days: 天数
    :return: 基金实时数据
    """
    all_data = {}

    for symbol in symbol_list:
        history_etf_data = download_etfHistory_data(type="etf", symbol=symbol)
        history_lof_data = download_etfHistory_data(type="lof", symbol=symbol)
        if history_etf_data is not None or history_lof_data is not None: 
            if days == "all": 
                if history_etf_data is not None:
                    # 获取所有时间段etf数据
                    all_data[symbol] = history_etf_data
                else:
                    # 获取所有时间段lof数据
                    all_data[symbol] = history_lof_data
            else:
                if history_etf_data is not None:
                    # 获取最近N天数据
                    history_data = history_data.tail(days)
                    all_data[symbol] = history_data
                else:
                    history_data = history_data.tail(days)
                    all_data[symbol] = history_data
        else:
            print(f"批量获取{symbol}历史数据失败")
    return all_data

def save_2_csv(data, symbol):
    """
    保存数据到单个csv文件
    :param data: 数据
    :param symbol: 文件名
    :return: None
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    print(f"正在保存{symbol}数据到csv文件...")
    data.to_csv(f"{symbol}.csv")

def read_from_csv(file_path):
    """
    从csv文件读取数据
    :param target: 文件名
    :return: 数据
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件{path}不存在")
    
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"读取文件{path}失败，错误信息：{e}")
        return pd.read_csv(path, encoding="utf-8")

if __name__ == '__main__':
   
    # 获取etf历史数据使用实例
    category_name = "全市场etf目录0612"
    symbol = config.symbol # symbol 调用em东方财富接口不用加前缀，调用sina新浪接口要加上市场前缀 sh sz
    symbol_list = config.symbol_list
    history_etf_data = batch_download_etf_data(symbol_list,days="all")
    for key, value in history_etf_data.items():
        save_2_csv(value, key)


