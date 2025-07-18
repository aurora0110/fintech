import akshare as ak
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import time
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        # 添加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stock_data = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
                stock_data['股票代码'] = symbol  # 添加股票代码列
                return stock_data
            except Exception as e:
                if attempt == max_retries - 1:  # 最后一次尝试仍然失败
                    print(f"股票 {symbol} 获取失败: {str(e)}")
                    return None
                time.sleep(1)  # 等待1秒后重试
    except Exception as e:
        print(f"股票 {symbol} 发生异常: {str(e)}")
        return None

def download_stock_category():
    df_a = ak.stock_info_a_code_name().assign(market="A股")
    df_list = df_a['code'].tolist()
    return df_list

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
    
def batch_download_stock_data(symbol_list, max_workers, start_date, end_date, adjust='qfq'):
    """
    批量获取股票最近N天的数据
    :param symbol_list: 股票代码列表，不带市场前缀，如['000001', '000002']
    :param days: 天数
    :param year_interval: 时间间隔，默认5年
    :return: 股票实时数据
    """
    max_workers = 20
    all_data = []
    failed_symbols = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建任务字典 {future: symbol} 以便跟踪
        future_to_symbol = {
            executor.submit(
                fetch_single_stock, 
                symbol, 
                start_date, 
                end_date, 
                adjust
            ): symbol for symbol in symbol_list
        }
        
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(future_to_symbol), total=len(symbol_list), desc="下载进度"):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result is not None:
                    all_data.append(result)
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                print(f"股票 {symbol} 处理异常: {str(e)}")
                failed_symbols.append(symbol)
    
    # 输出下载结果统计
    print(f"\n下载完成: 成功 {len(all_data)} 只, 失败 {len(failed_symbols)} 只")
    if failed_symbols:
        print("失败的股票代码:", failed_symbols)
    
    # 合并所有成功下载的数据
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

    
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
                if history_etf_data is not None:
                    # 获取最近N天数据
                    history_data = history_data.tail(days)
                    all_data[symbol] = history_data
                else:
                    history_data = history_data.tail(days)
                    all_data[symbol] = history_data
        else:
            print(f"批量获取fund：{symbol}历史数据失败")
    return all_data

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

if __name__ == '__main__':
   '''
    # 获取etf历史数据使用实例
    category_name = "全市场etf目录0612"
    symbol = config.symbol # symbol 调用em东方财富接口不用加前缀，调用sina新浪接口要加上市场前缀 sh sz
    symbol_list = config.stock_symbol_list
    stock_start_date = config.stock_start_date
    end_date = config.end_date
    #istory_etf_data = batch_download_etf_data(symbol_list,days="all")
    #for key, value in history_etf_data.items():
     #   save_2_csv(value, key)

    # 获取A股全部股票目录
    batch_download_stock_data(days="all", start_date=stock_start_date, end_date=end_date, year_interval=1)
    # 获取某只股票的历史记录
    stock_data = batch_download_stock_data(symbol_list, days="all", start_date=stock_start_date, end_date=end_date, year_interval=1)
    for key, value in stock_data.items():
        save_2_csv(value, key)
    '''


