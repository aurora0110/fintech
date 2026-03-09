#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据拼接工具
将指定目录中的所有股票数据文件拼接成一个统一的TXT文件
"""

import os
import pandas as pd
from typing import List, Dict


def get_stock_files(data_dir: str) -> List[str]:
    """获取目录中的所有股票数据文件"""
    files = []
    for f in os.listdir(data_dir):
        if f.lower().endswith(('.txt', '.csv')):
            files.append(os.path.join(data_dir, f))
    return sorted(files)


def read_stock_data(file_path: str) -> pd.DataFrame:
    """读取单个股票数据文件"""
    try:
        # 尝试多种分隔符读取
        df = pd.read_csv(file_path, sep=r"\s+|\t+", engine="python")
        
        # 获取股票代码（从文件名提取）
        filename = os.path.basename(file_path)
        stock_code = os.path.splitext(filename)[0]
        
        # 添加股票代码列
        df['stock_code'] = stock_code
        
        return df
    except Exception as e:
        print(f"读取文件失败: {file_path}, 错误: {e}")
        return pd.DataFrame()


def concatenate_all_data(data_dir: str, output_file: str) -> None:
    """拼接所有股票数据到单个文件"""
    print(f"开始处理目录: {data_dir}")
    
    # 获取所有股票文件
    stock_files = get_stock_files(data_dir)
    print(f"找到 {len(stock_files)} 个股票数据文件")
    
    if not stock_files:
        print("未找到任何股票数据文件")
        return
    
    # 读取并拼接所有数据
    all_data = []
    
    for i, file_path in enumerate(stock_files, 1):
        print(f"处理进度: {i}/{len(stock_files)} - {os.path.basename(file_path)}")
        
        df = read_stock_data(file_path)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("未成功读取任何数据")
        return
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 按股票代码和日期排序
    combined_df = combined_df.sort_values(['stock_code', '日期']).reset_index(drop=True)
    
    print(f"合并后数据量: {len(combined_df)} 行")
    print(f"股票数量: {combined_df['stock_code'].nunique()} 只")
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 按股票代码分组写入
        for stock_code, group in combined_df.groupby('stock_code'):
            # 写入股票代码行
            f.write(f"{stock_code}\n")
            
            # 写入数据行（不包含股票代码列）
            data_rows = group.drop('stock_code', axis=1)
            for _, row in data_rows.iterrows():
                # 将行数据转换为字符串并写入
                row_str = '\t'.join([str(x) for x in row.values])
                f.write(f"{row_str}\n")
            
            # 股票数据之间空一行
            f.write("\n")
    
    print(f"数据已成功写入: {output_file}")
    
    # 显示统计信息
    print("\n统计信息:")
    print(f"总股票数: {combined_df['stock_code'].nunique()}")
    print(f"总数据行数: {len(combined_df)}")
    print(f"数据时间范围: {combined_df['日期'].min()} 到 {combined_df['日期'].max()}")


def main():
    """主函数"""
    # 输入目录
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/20260226/normal"
    
    # 输出文件
    output_file = "/Users/lidongyang/Desktop/Qstrategy/data/combined_stock_data.txt"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 执行拼接
    concatenate_all_data(data_dir, output_file)


if __name__ == "__main__":
    main()