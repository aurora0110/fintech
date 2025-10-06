import pandas as pd
import os
import glob
from  pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import getData

'''
把导出的B1/B2数据合并成一个文件
'''

# 从本地文件导入股票代码
def import_b2_data(code_path):
    code_path = Path(code_path)
    stock_symbol_list = []
    etf_list = []
    df = pd.read_csv(code_path)
    # 遍历 DataFrame 的每一行
    for _, row in df.iterrows():
        code = str(row[0]).strip().zfill(6)   # 股票代码，转为字符串
        name = str(row[1]).strip()   # 股票名称

        # 判断是否含有 "ETF"
        if "ETF" in name.upper():   # 转大写防止大小写问题
            etf_list.append(f'{code}')    # 带引号的字符串
        else:
            stock_symbol_list.append(f'{code}')
    return stock_symbol_list

# 合并单独的股票数据到一个文件里
def merge_stock_data(stock_symbol_list, folder_path, output_file):
    """
    合并data文件夹中的所有股票CSV文件
    
    参数:
    folder_path: 包含CSV文件的文件夹路径
    output_file: 输出文件名
    """
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        print("在指定文件夹中未找到CSV文件")
        return
    
    # 存储所有数据的列表
    all_data = []
    
    # 读取每个CSV文件
    for file_path in csv_files:
        if file_path.split('/')[-1].split('.')[0] in stock_symbol_list:
            try:
                # 读取CSV文件，跳过索引列
                df = pd.read_csv(file_path)
                
                # 如果第一列是Unnamed: 0索引列，可以删除
                if df.columns[0] == 'Unnamed: 0':
                    df = df.drop(df.columns[0], axis=1)
                
                # 添加到列表
                all_data.append(df)
                print(f"已读取: {os.path.basename(file_path)} - {len(df)} 行数据")
                
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
    
    if all_data:
        # 合并所有数据
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # 保存到单个CSV文件
        merged_df.to_csv(output_file, index=False)
        print(f"\n合并完成！共合并 {len(csv_files)} 个文件，总计 {len(merged_df)} 行数据")
        print(f"结果已保存到: {output_file}")
        
        return merged_df
    else:
        print("没有成功读取任何文件")
        return None

# 使用示例
if __name__ == "__main__":
    # B1/B2数据文件路径
    file_path = '/Users/lidongyang/Desktop/临时条件股20250829.csv'
    # 数据保存路径
    out_path = '/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/b2data/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    stock_symbol_list = import_b2_data(file_path)
    # 获取当前时间
    now = datetime.now()
    end_date = now.strftime("%Y%m%d")

    stock_years_ago =  now - relativedelta(months=6) # 获取n年前的日期
    stock_start_date = stock_years_ago.strftime("%Y%m%d")

    print('开始下载股票数据...')
    stock_data = getData.batch_download_stock_data(stock_symbol_list, days="all", start_date=stock_start_date, end_date=end_date, year_interval=1)
    for key, value in stock_data.items():
        getData.save_2_csv(value, key, out_path)
    # 执行合并
    merged_data = merge_stock_data(stock_symbol_list, out_path, out_path + 'merged_stocks' + datetime.now().strftime("%Y%m%d") + '.csv')
    
    # 显示合并后的数据前几行
    if merged_data is not None:
        print("\n合并后数据的前5行:")
        print(merged_data.head())