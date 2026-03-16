import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
import time
import sys
warnings.filterwarnings('ignore')

class StockDataValidator:
    def __init__(self):
        # 定义必填字段（适配通达信导出格式）
        self.required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']
        # 价格字段（用于逻辑校验）
        self.price_cols = ['开盘', '最高', '最低', '收盘']
        # 数值字段（禁止负数）
        self.numeric_cols = ['开盘', '最高', '最低', '收盘', '成交量', '成交额']

    def load_data(self, file_path):
        """加载单个股票数据文件（逐行解析，兼容列数混乱/分隔符不一致的文件）"""
        try:
            # 步骤1：读取文件所有行（兼容编码）
            encodings = ['gbk', 'gb2312', 'utf-8', 'latin-1']
            lines = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                    break
                except UnicodeDecodeError:
                    continue
            
            if lines is None:
                return None, "文件编码格式不支持，无法读取"
            
            # 步骤2：逐行过滤+清洗，只保留有效数据行
            data_rows = []
            for line in lines:
                # 跳过ST股票数据
                if any(keyword in line for keyword in ['ST', '*ST', '*']):
                    break

                # 跳过标题行（含中文列名的行）
                if any(keyword in line for keyword in ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']):
                    continue
                
                # 清洗行数据：把任意分隔符（空格/制表符）换成单个空格，再拆分
                clean_line = ' '.join(line.split())  # 合并多个空格/制表符为一个
                parts = clean_line.split(' ')  # 按单个空格拆分
                
                # 只保留7列的有效数据行（日期+4个价格+成交量+成交额）
                if len(parts) == 7:
                    # 校验日期格式（YYYY/MM/DD），过滤无效行
                    if '/' in parts[0] and len(parts[0].split('/')) == 3:
                        data_rows.append(parts)
            
            # 检查是否有有效数据
            if not data_rows:
                return None, "文件中未找到有效数据行（7列的日期+价格+成交量数据）"
            
            # 步骤3：转换为DataFrame并处理类型
            df = pd.DataFrame(data_rows, columns=self.required_cols)
            
            # 数值列转换（容错，失败设为NaN）
            for col in self.numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 日期转换（容错）
            df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce')
            
            # 步骤4：删除无效行（日期或收盘价为空的行）
            df = df.dropna(subset=['日期', '收盘'])
            
            # 重置索引
            df = df.reset_index(drop=True)
            
            return df, None
        except Exception as e:
            return None, f"文件读取失败：{str(e)}"
    
    def check_completeness(self, df, file_name):
        """校验数据完整性"""
        errors = []
        # 检查必填字段是否存在
        missing_cols = [col for col in self.required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"缺失必填字段：{','.join(missing_cols)}")
        # 检查缺失值
        null_stats = df.isnull().sum()
        null_cols = null_stats[null_stats > 0].to_dict()
        if null_cols:
            errors.append(f"字段缺失值：{null_cols}")
        # 检查空行
        empty_rows = df[df[self.price_cols].isnull().all(axis=1)].shape[0]
        if empty_rows > 0:
            errors.append(f"存在空行：共{empty_rows}行")
        # 检查行数小于22行
        if len(df) <= 22:
            errors.append(f"数据行数过少：共{len(df)}行")
        return errors
    
    def check_logic(self, df, file_name):
        """校验业务逻辑合理性"""
        errors = []
        # 检查价格逻辑：最高≥开盘/收盘≥最低
        logic1 = df[(df['最高'] < df['开盘']) | (df['最高'] < df['收盘'])].shape[0]
        logic2 = df[(df['最低'] > df['开盘']) | (df['最低'] > df['收盘'])].shape[0]
        if logic1 > 0:
            errors.append(f"价格逻辑错误（最高<开盘/收盘）：{logic1}行")
        if logic2 > 0:
            errors.append(f"价格逻辑错误（最低>开盘/收盘）：{logic2}行")
        # 检查日期重复
        duplicate_dates = df['日期'].duplicated().sum()
        if duplicate_dates > 0:
            errors.append(f"日期重复：{duplicate_dates}行")
        # 检查日期连续性（允许交易日间隙，不允许无序）
        if not df['日期'].is_monotonic_increasing:
            errors.append("日期顺序混乱（非递增）")
        return errors
    
    def check_numeric(self, df, file_name):
        """校验数值有效性（仅检查最近一天/最后一行数据）"""
        errors = []
        
        # 先判断数据是否为空，避免索引报错
        if df.empty:
            errors.append("数据为空，无法校验最近一天数值")
            return errors
        
        # 提取最后一行数据（最近一天）
        last_row = df.iloc[-1]
        
        # 1. 检查最近一天的非负数值列是否存在负数
        for col in self.numeric_cols:
            # 先判断列是否存在，避免KeyError
            if col not in df.columns:
                errors.append(f"缺失列：{col}，无法校验最近一天数值")
                continue
            
            # 提取最后一行该列的值，并转换为数值类型（避免非数值干扰）
            try:
                col_value = float(last_row[col])
            except (ValueError, TypeError):
                errors.append(f"{col}最近一天的值非数值类型：{last_row[col]}")
                continue
            
            # 检查是否为负数
            if col_value < 0:
                errors.append(f"{col}最近一天存在负数：{col_value}")
        
        # 2. 检查最近一天的成交量是否为0
        if '成交量' in df.columns:
            try:
                volume_value = float(last_row['成交量'])
                if volume_value == 0:
                    errors.append(f"成交量最近一天为0：{volume_value}")
            except (ValueError, TypeError):
                errors.append(f"成交量最近一天的值非数值类型：{last_row['成交量']}")
        else:
            errors.append("缺失列：成交量，无法校验最近一天成交量")
        
        return errors

    def validate_single_file(self, file_path):
        """校验单个文件"""
        file_name = Path(file_path).name
        #print(f"正在校验：{file_name}")
        
        # 加载数据
        df, load_error = self.load_data(file_path)
        if load_error:
            return {
                '文件名': file_name,
                '状态': '失败',
                '异常信息': [load_error],
                '数据行数': 0
            }
        
        # 执行各类校验
        completeness_errors = self.check_completeness(df, file_name)
        logic_errors = self.check_logic(df, file_name)
        numeric_errors = self.check_numeric(df, file_name)
        
        # 汇总异常
        all_errors = completeness_errors + logic_errors + numeric_errors
        status = '正常' if len(all_errors) == 0 else '异常'

        if status == '正常':
            parent_dir = os.path.dirname(file_path)
            new_dir = os.path.join(parent_dir, "normal")

            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            new_file_path = os.path.join(new_dir, file_name)

            df.to_csv(
                new_file_path,
                sep='\t',  # 按制表符分隔，与原文件格式一致
                index=False,  # 不保存索引
                header=True,  # 保存列名（日期、开盘、最高、最低、收盘、成交量、成交额）
                date_format='%Y/%m/%d'  # 日期格式保持 YYYY/MM/DD
            )
        
        return {
            '文件名': file_name,
            '状态': status,
            '异常信息': all_errors if all_errors else ['无'],
            '数据行数': len(df)
        }
    
    def validate_batch_files(self, folder_path):
        """批量校验文件夹内所有txt文件"""
        file_paths = list(Path(folder_path).glob('*.txt'))
        if not file_paths:
            print("未找到任何txt文件")
            return pd.DataFrame()
        
        # 批量执行校验
        results = []

        for file_path in file_paths:
            result = self.validate_single_file(str(file_path))
            results.append(result)
        
        # 生成报告
        report_df = pd.DataFrame(results)
        return report_df

def main(data_dir_before):
    # 配置参数
    FOLDER_PATH = data_dir_before  # 替换为实际文件夹路径
    REPORT_PATH = data_dir_before + f"/股票数据异常校验报告_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    
    # 初始化校验器并执行
    validator = StockDataValidator()
    report = validator.validate_batch_files(FOLDER_PATH)
    
    # 保存报告
    report.to_csv(REPORT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n校验完成！报告已保存至：{REPORT_PATH}")
    
    # 打印汇总信息
    print(f"\n汇总统计：")
    print(f"总文件数：{len(report)}")
    if '状态' not in report.columns:
        print("正常文件数：0")
        print("异常文件数：0")
        return report

    print(f"正常文件数：{len(report[report['状态']=='正常'])}")
    print(f"异常文件数：{len(report[report['状态']=='异常'])}")
    
    # 打印异常文件详情
    abnormal_files = report[report['状态']=='异常']
    if len(abnormal_files) > 0:
        print(f"\n异常文件详情：")
        for _, row in abnormal_files.iterrows():
            print(f"- {row['文件名']}：{row['异常信息']}")

    return report

if __name__ == "__main__":
    main()
