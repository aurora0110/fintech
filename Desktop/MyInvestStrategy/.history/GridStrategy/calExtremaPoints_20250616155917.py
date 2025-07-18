from datetime import datetime
import numpy as np
import pandas as pd
import getData as gd

'''
  计算单个品种，多年内的极值点
'''

def one_years_extrema_points(file_path):
    # input: one year of data
    # output: extrema points
    data = public_calculate(file_path, 1)
    return data

def two_years_extrema_points(file_path):
    # input: two years of data
    # output: extrema points
    data = public_calculate(file_path, 2)
    return data

def three_years_extrema_points(file_path):
    # input: three years of data
    # output: extrema points
    data = public_calculate(file_path, 3)
    return data


def four_years_extrema_points(file_path):
    # input: four years of data
    # output: extrema points
    data = public_calculate(file_path, 4)
    return data

def five_years_extrema_points(file_path):
    # input: five years of data
    # output: extrema points
    data = public_calculate(file_path, 5)
    return data

def x_years_extrema_points(file_path, years):
    '''
      开放输入年份以计算极值点
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

    if not target_data.empty:
        max_price = target_data['最高'].max() # 文件中有收盘 最高 最低，取每日的最高
        min_price = target_data['最低'].min() # 文件中有收盘 最高 最低，取每日的最低
    else:
        max_price = min_price = '无数据'
    
    result = []
    result.append({
        '时间':f'{years}年内',
        '最高价': max_price,
        '最低价': min_price
    })

    result_df = pd.DataFrame(result) # 转换为dataframe输出
    print(f'计算极值点结果：{result_df}')
    return result_df

if __name__ == '__main__':
    file_path = '/Users/lidongyang/Desktop/vscodePython/sh51030020250612.csv'