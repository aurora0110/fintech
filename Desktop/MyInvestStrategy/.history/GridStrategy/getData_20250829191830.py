import akshare as ak
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import ssl, certifi, urllib.request

_ctx = ssl.create_default_context(cafile=certifi.where())  # 用 certifi 的 CA 列表
_opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx))
urllib.request.install_opener(_opener)  # 让 pandas/urllib 用这个带 CA 的 opener



def download_stock_data(symbol, start_date, end_date, adjust='qfq'):
    """
    获取市场某只股票数据
    :param symbol: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param adjust：qfq（计算技术指标） or hfq（计算分红）
    :return: 股票数据
    """
    try:
        #ak.stock_zh_a_spot 获取全市场行情
        stock_data = ak.stock_zh_a_hist(symbol, "daily", start_date, end_date, "qfq") # symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq"
        #stock_data = ak.stock_zh_a_hist_tx(symbol = "sz000001", start_date = "19980101", end_date = "20500101", adjust = "",timeout = None)
        #stock_data = ak.stock_zh_a_hist(symbol= "000001",period= "daily",start_date= "19980101",end_date= "20500101",adjust= "",timeout = None)
        print('stock_data:', stock_data)
        return stock_data
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        return None

def download_stock_category():
    df = ak.stock_info_a_code_name().assign(market="A股")
    return df

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
    
def batch_download_stock_data(symbol_list, days, start_date, end_date, year_interval=5):
    """
    批量获取股票最近N天的数据
    :param symbol_list: 股票代码列表，不带市场前缀，如['000001', '000002']
    :param days: 天数
    :param year_interval: 时间间隔，默认5年
    :return: 股票实时数据
    """
    
    all_data = {}

    for symbol in symbol_list:
        history_stock_data = download_stock_data(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")

        if history_stock_data is not None or history_stock_data is not None: 
            if days == "all": 
                if history_stock_data is not None:
                    # 获取所有时间段etf数据
                    all_data[symbol] = history_stock_data
                else:
                    # 获取所有时间段lof数据
                    all_data[symbol] = history_stock_data
            else:
                if history_stock_data is not None:
                    # 获取最近N天数据
                    history_data = history_data.tail(days)
                    all_data[symbol] = history_data
                else:
                    history_data = history_data.tail(days)
                    all_data[symbol] = history_data
        else:
            print(f"批量获取stock：{symbol}历史数据失败")
    return all_data
    
def batch_download_etf_data(symbol_list, days, start_date, end_date, year_interval=5):
    """
    批量获取基金最近N天的数据
    :param symbol_list: 基金代码列表，带市场前缀，如['sh000001', 'sh000002']
    :param days: 类型
    :param year_interval: 时间间隔，默认5年
    :return: 基金实时数据
    """
    all_data = {}

    for symbol in symbol_list:
        history_etf_data = download_etfHistory_data(type="etf", symbol=symbol, start_date=start_date, end_date=end_date)
        #history_lof_data = download_etfHistory_data(type="lof", symbol=symbol, start_date=start_date, end_date=end_date)

        if history_etf_data is not None: 
            if days == "all": 
                if history_etf_data is not None:
                    # 获取所有时间段etf数据
                    all_data[symbol] = history_etf_data
        else:
            print(f"批量获取fund：{symbol}历史数据失败")
    return all_data

def download_daily_trade_volume(symbol, retry):   
    '''
    使用腾讯接口，批量获取当日交易量信息和买卖笔数
    '''
    for _ in range(retry):
        try:
            # 自动识别市场前缀
            market = 'sh' if symbol.startswith('6') else 'sz'
            df = ak.stock_zh_a_tick_tx_js(symbol=f"{market}{symbol}")
            # 清洗数据
            df = df[['成交时间','成交价格','成交金额','成交量','性质']]
            df['成交量'] = df['成交量'].astype(int)
            print(f"已保存{market}{symbol}数据当日成交数据")
            return df
        except Exception as e:
            print(f"第{_+1}次获取{market}{symbol}数据失败，重试中...")
            time.sleep(1)

    raise Exception(f"获取{market}{symbol}数据失败，重试次数已用完")

# ========= 估值接口 =========
def download_csindex_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    中证估值（优先用于：中证/上证部分宽基与行业）。
    返回尽量规范化的列：date, PE, PB（PB可能缺失）
    """
    df = ak.stock_zh_index_hist_csindex(symbol=code, start_date=start_date, end_date=end_date)
    df2 = ak.stock_index_pe_lg(symbol="沪深300")
    print(f"{code}数据下载中:{df}")
    print(df2)
    return df

def save_2_csv(data, symbol, file_path):
    """
    保存数据到单个csv文件
    :param data: 数据
    :param symbol: 文件名
    :return: None
    """
    print(f"正在保存{symbol}数据到csv文件...")
    data.to_csv(file_path + f"{symbol}.csv")

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
    
def download_total_cap(symbol):
    '''
    获取总市值和总股本
    '''
    stock_info = ak.stock_individual_info_em(symbol=symbol)

    # 提取总市值和总股本字段（单位是元）
    market_cap1 = stock_info[stock_info['item'] == '总市值'].iloc[0]['value']
    market_cap2 = stock_info[stock_info['item'] == '总股本'].iloc[0]['value']
    print(f"{symbol} 的总市值为：{int2chinese.int_to_chinese_num(int(market_cap1))}，{symbol} 的总股本为：{int2chinese.int_to_chinese_num(int(market_cap2))}")
    return int(market_cap1), int(market_cap2)

if __name__ == '__main__':
    df = csindex_valuation('000300')
    print(df)



