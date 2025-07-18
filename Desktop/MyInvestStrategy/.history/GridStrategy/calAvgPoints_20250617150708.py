from datetime import datetime
import numpy as np
import pandas as pd
import getData as gd

def x_years_avg_points(file_path, years):
    '''
      开放输入年份以计算平均值点
    '''
    # input: five years of data
    # output: extrema points
    data = public_calculate(file_path, years)
    return data

def public_calculate(file_path, years):
    current_date = datetime.now().date()
    data = gd.read_from_csv(file_path) # dataframe格式
    data['日期'] = pd.to_datetime(data['日期']) # 转换日期为datetime格式

    end_date = data['日期'].max() # 获取最新日期
    start_date = end_date - pd.DateOffset(years=years) # 获取一年前的日期

    target_data = data[(data['日期'] >= start_date) & (data['日期'] <= end_date)] # 获取一年内的数据
    print(type(target_data['收盘']))

    if not target_data.empty:
        avg_price = target_data['收盘'].mean() # 文件中有收盘 最高 最低，取每日的收盘
    else:
        max_price = min_price = '无数据'
    
    result = []
    result.append({
        '时间':f'{years}年内',
        '平均价': avg_price
    })

    result_df = pd.DataFrame(result) # 转换为dataframe输出
    print(f'计算极值点结果：{result_df}')
    return result_df

if __name__ == '__main__':
    file_path = '/Users/lidongyang/Desktop/vscodePython/sh51030020250612.csv'
    x_years_avg_points(file_path, 1)