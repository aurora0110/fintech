import pandas as pd
import numpy as np
import os
import time
import time
import math
import copy
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

class InvestmentTracker:
    def __init__(self):
        self.filename = 'investment_gridrecord.csv'
        self.columns = ['种类', '档位', '买入价', '卖出价', '买入数量', '买入金额', '卖出数量', '卖出金额', '盈利金额', '盈利比例']
        self.init_data_file()

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("网格投资记录工具V1.0")
        self.root.geometry("600x800")

        # 创建输入组件
        self.create_widgets()

        self.init_data_file()


    # 初始化数据文件
    def init_data_file(self):
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filename, index=False)

    def create_widgets(self):
        # 品种输入
        ttk.Label(self.root, text="种类").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.stock_entry = ttk.Entry(self.root)
        self.stock_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
        # 档位输入
        ttk.Label(self.root, text="档位").grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.level_entry = ttk.Entry(self.root)
        self.level_entry.grid(row=2, column=1, padx=10, pady=10, sticky='ew')
        # 数量输入
        ttk.Label(self.root, text="买入价").grid(row=3, column=0, padx=10, pady=10, sticky='w')
        self.buyprice_entry = ttk.Entry(self.root)
        self.buyprice_entry.grid(row=3, column=1, padx=10, pady=10, sticky='ew')
        # 总价输入
        ttk.Label(self.root, text="卖出价").grid(row=4, column=0, padx=10, pady=10, sticky='w')
        self.sellprice_entry = ttk.Entry(self.root)
        self.sellprice_entry.grid(row=4, column=1, padx=10, pady=10, sticky='ew')
        # 操作收益输入
        ttk.Label(self.root, text="买入数量").grid(row=5, column=0, padx=10, pady=10, sticky='w')
        self.buynum_entry = ttk.Entry(self.root)
        self.buynum_entry.grid(row=5, column=1, padx=10, pady=10, sticky='ew')
        # 操作输入
        ttk.Label(self.root, text="买入金额").grid(row=6, column=0, padx=10, pady=10, sticky='w')
        self.buyamout_entry = ttk.Entry(self.root)
        self.buyamout_entry.grid(row=6, column=1, padx=10, pady=10, sticky='ew')
        # 操作输入
        ttk.Label(self.root, text="卖出数量").grid(row=7, column=0, padx=10, pady=10, sticky='w')
        self.sellnum_entry = ttk.Entry(self.root)
        self.sellnum_entry.grid(row=7, column=1, padx=10, pady=10, sticky='ew')
        # 操作输入
        ttk.Label(self.root, text="卖出金额").grid(row=8, column=0, padx=10, pady=10, sticky='w')
        self.sellamount_entry = ttk.Entry(self.root)
        self.sellamount_entry.grid(row=8, column=1, padx=10, pady=10, sticky='ew')
        # 操作输入
        ttk.Label(self.root, text="盈利金额").grid(row=9, column=0, padx=10, pady=10, sticky='w')
        self.profit_entry = ttk.Entry(self.root)
        self.profit_entry.grid(row=9, column=1, padx=10, pady=10, sticky='ew')
        # 操作输入
        ttk.Label(self.root, text="盈利比例").grid(row=10, column=0, padx=10, pady=10, sticky='w')
        self.ratio_entry = ttk.Entry(self.root)
        self.ratio_entry.grid(row=10, column=1, padx=10, pady=10, sticky='ew')
        # 提交按钮
        ttk.Button(self.root, text="提交", command=self.save_record).grid(row=11, column=1, padx=10, pady=10, sticky='ew')
        # 配置网格列权重
        self.root.columnconfigure(1, weight=1)

    # 验证输入有效性
    def validate_input(self):
        print("validate_input")
        try:
            float(self.price_entry.get())
            float(self.profit_entry.get())
            if not self.stock_entry.get():
                raise ValueError("品种不能为空")
            return True
        except ValueError as e :
            messagebox.showerror("输入错误", "价格和操作收益必须为数字")
            return False
    
    # 保存记录到Excel文件
    def save_record(self):
        print("save_record")

        new_record = {
            "种类": self.stock_entry.get(),
            "档位": self.level_entry.get(),
            "买入价": self.buyprice_entry.get(),
            "卖出价": self.sellprice_entry.get(),
            "买入数量": self.buynum_entry.get(),
            "买入金额": self.buyamout_entry.get(),
            "卖出数量": self.sellnum_entry.get(),
            "卖出金额": self.sellamount_entry.get(),
            "盈利金额": self.profit_entry.get(),
            "盈利比例": self.ratio_entry.get(),
            "日期": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            # 读取现有数据
            print("读取")
            df = pd.read_csv(self.filename)
            # 添加新纪录
            print("添加")
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            # 保存到文件
            print("保存")
            df.to_csv(self.filename, index=False)
            print("记录已保存到Excel文件")
            messagebox.showinfo("保存成功", "记录已保存到Excel文件")
        except Exception as e:
            print(e)
            messagebox.showerror("保存失败", f"保存记录时出错: {str(e)}")

    # 获取清空输入字段
    def clear_fields(self):
        self.stock_entry.delete(0, tk.END)
        self.price_entry.delete(0, tk.END)
        self.profit_entry.delete(0, tk.END)

    # 提交记录
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    record = InvestmentTracker()
    record.run()