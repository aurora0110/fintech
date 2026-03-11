import time
import random
from utils import b1filter, pinfilter, brick_filter, holdprint, selectprint, stockDataValidator, stoploss, takeprofit
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import json
import os

if __name__ == '__main__':
    start_time = time.time()
    # 买入卖出信号
    b1_list = []
    pin_list = []
    brick_list = []
    sell_list = []

    # 获取今天日期
    today_str = datetime.today().strftime("%Y%m%d")
    print("今天的日期:", today_str)  # 输出示例：20260123

    # 打印持仓
    hold_path = "/Users/lidongyang/Desktop/Qstrategy/config/holding.yaml"
    hold_list = holdprint.show(hold_path)

    # 校验处理数据
    data_dir_before = "/Users/lidongyang/Desktop/Qstrategy/data/" + today_str
    data_dir_after = "/Users/lidongyang/Desktop/Qstrategy/data/" + today_str + "/normal"
    if not Path(data_dir_before).exists():
        Path(data_dir_before).mkdir(parents=True, exist_ok=True)
    if not Path(data_dir_after).exists():
        Path(data_dir_after).mkdir(parents=True, exist_ok=True)
    stockDataValidator.main(data_dir_before)
    #print("处理前的数据目录：", data_dir_before, "处理后的数据目录：", data_dir_before)

    file_paths = list(Path(data_dir_after).glob('*.txt'))
    if not file_paths:
        print("未找到任何txt文件")
        
    # 批量执行校验
    for file_path in file_paths:
        #print(file_path)
        # 步骤1：取最后一个/后的文件名 → SZ#300319.txt
        file_name_full = str(file_path).split('/')[-1]
        # 步骤2：去掉.txt后缀 → SZ#300319
        file_name_no_suffix = file_name_full.replace('.txt', '')
        # 步骤3：取#后的股票代码 → 300319
        file_name = file_name_no_suffix.split('#')[-1]

        # 针对持仓进行止损止盈校验
        stoploss_result = stoploss.check(str(file_path), hold_list)
        takeprofit_result = takeprofit.check(str(file_path), hold_list)
        if stoploss_result[0] == 1:
            sell_list.append([file_name, stoploss_result[1]])
        if takeprofit_result[0] == 1:
            sell_list.append([file_name, takeprofit_result[1]])

        # 针对B1进行校验
        b1_result = b1filter.check(str(file_path), hold_list)
        if b1_result[0] == 1:
            b1_list.append([file_name, str(b1_result[1].round(2)), str(b1_result[2].round(2)), str(b1_result[3]), str(b1_result[4])])
        # 针对B2进行校验
        #b2_result = b2filter.check(str(file_path), hold_list)
        #if b2_result[0][0] == 1:
            #b2_list.append([file_name, str(b2_result[0][1]), str(b2_result[0][2].round(2))])
        # 针对B3进行校验
        #b3_result = b3filter.check(str(file_path), hold_list, b2_result[1])
        #if b3_result[0] == 1:
            #b3_list.append([file_name, str(b3_result[1]), str(b3_result[2]), str(b3_result[3])])
        # 针对单针进行校验
        pin_result = pinfilter.check(str(file_path))
        if pin_result:
            pin_list.append(file_name)
        # 针对brick策略进行校验
        brick_result = brick_filter.check(str(file_path), hold_list)
        if brick_result[0] == 1:
            brick_list.append([
                file_name,
                str(round(brick_result[1], 2)),
                str(round(brick_result[2], 2)),
                str(brick_result[3]),
                str(brick_result[4]),
            ])

    sell_list.sort()
    b1_list.sort()
    pin_list.sort()
    brick_list.sort()

    print("💡持有列表：", sell_list, '\n', "💡b1列表：", b1_list, '\n', "💡单针列表：", pin_list, '\n', "💡brick列表：", brick_list, '\n')
    selectprint.show(sell_list, "持有")
    selectprint.show(b1_list, "B1")
    selectprint.show(pin_list, "单针")
    selectprint.show(brick_list, "BRICK")
    
    # 保存筛选结果到文件，供dashboard.py读取
    # 创建结果目录
    result_dir = "/Users/lidongyang/Desktop/Qstrategy/results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 保存结果到JSON文件
    result_file = os.path.join(result_dir, f"{today_str}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'b1_list': b1_list,
            'sell_list': sell_list,
            'pin_list': pin_list,
            'brick_list': brick_list,
            'hold_list': hold_list
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n筛选结果已保存到：{result_file}")
    print("\n提示：运行 dashboard.py 可以查看这些结果的可视化展示")
    
    end_time = time.time()
    print("程序运行时间：", end_time - start_time, "秒")
