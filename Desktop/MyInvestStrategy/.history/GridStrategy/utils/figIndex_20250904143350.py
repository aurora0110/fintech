# -*- coding: utf-8 -*-
"""
功能清单
1) 用 stock_zh_index_spot_em 下载目录（上证/深证/中证/国证），只输出你关心的指数
2) 下载“指定指数”的历史数据（日线，东财接口）
3) 绘制：中证全指(000985) PE × 深综指(399106)（双轴）
4) 绘制：全A PB × 深综指（PB用申万“市场表征-全A”近似，双轴）
5) 绘制：中证全指（PE或PB就近）× 深综指（双轴）
6) 绘制：你关心的“行业指数” PE、PB（优先中证估值，缺失则用申万月报）
7) 绘制：你关心的“宽基指数” PE、PB（中证估值为主；海外/港股若无估值则跳过估值，仅保留点位对比提示）

沪深300（上证000300）、上证180（上证000010）、深证100（深证399330）、科创50（上证000688）、创业板指（深证399006）、上证50（上证000016）、中证500（上证000905）、中证1000（上证000852）、恒生科技、恒生医疗、标普500、恒生、纳斯达克；
行业我只关心：全指医药（上证000991）、全指金融（上证000992）、全指消费（上证000990）、中证环保（上证000827）、全指信息（上证000993）、中证医疗（深证399989）、食品饮料（上证000807）、中证红利（上证000922）、中证军工（深证399967）、中证传媒（深证399971）、中国互联

输出：./output/ 下若干 CSV 与 PNG

1、下载数据
"""
# utils/figIndex.py 顶部
from pathlib import Path
import sys
# 把“GridStrategy 的父目录（MyInvestStrategy）”加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from GridStrategy import getData, config
import warnings, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
from GridStrategy import getData, config
import ssl, certifi, urllib.request
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from cycler import cycler
import matplotlib.cm as cm

n = 40
cmap = cm.get_cmap('tab20', n)  # 或 'nipy_spectral'
plt.rcParams['axes.prop_cycle'] = cycler(color=[cmap(i) for i in range(n)])
_ctx = ssl.create_default_context(cafile=certifi.where())  # 用 certifi 的 CA 列表
_opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx))
urllib.request.install_opener(_opener)  # 让 pandas/urllib 用这个带 CA 的 opener
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial Unicode MS','Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

def fig(file_path):
    df = getData.read_from_csv(file_path)
    x_col = "日期"
    y_col = "滚动市盈率"
    data = df[[x_col, y_col]].copy()
    data = data.dropna(subset=[x_col, y_col])  # 去掉缺失
        # 2) 尝试把 A 列解析成日期；如果不是日期，再尝试转为数字
    x_try_dt = pd.to_datetime(data[x_col], errors="coerce")
    if x_try_dt.notna().mean() > 0.8:      # 大多数能转成功就当作日期用
        data[x_col] = x_try_dt
        x_is_date = True
    else:
        # 不是日期，就尽量转成数值（若仍失败就保留原样）
        data[x_col] = pd.to_numeric(data[x_col], errors="ignore")
        x_is_date = False
    # 3) Y 列转数值（非数值会变成 NaN 并被丢弃）
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[y_col])
    return data[x_col],data[y_col],x_is_date

if __name__ == '__main__':
    index_list = config.index_list
    index_list_name = config.index_list_name
    industry_list = config.industry_list
    industry_list_name = config.industry_list_name
    downloadIndexNewDataSwitch = config.downloadIndexNewDataSwitch
    index_start_date = config.index_start_date
    end_date = config.end_date
    index_save_path = '/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/indexdata/'


    none_list = []
     # 下载最新数据并保存成csv文件
    if downloadIndexNewDataSwitch:
        for x in index_list:
            print(f"{x}开始下载")
            index_data = getData.download_csindex_data(x, start_date=index_start_date, end_date=end_date)
            if index_data is not None:
                getData.save_2_csv(index_data, x, index_save_path)
            else:
                none_list.append(x)
        for x in industry_list:
            print(f"{x}开始下载")
            index_data = getData.download_csindex_data(x, start_date=index_start_date, end_date=end_date)
            if index_data is not None:
                getData.save_2_csv(index_data, x, index_save_path)
            else:
                none_list.append(x)
    print(none_list)
    X = None
    vecs = []
    for x in index_list:
        file_path = index_save_path + x + '.csv'
        x_col, y_col, x_is_date = fig(file_path)
        vecs.append(y_col)
        X = x_col
    # 5) 画图（不指定颜色，遵循默认配色）
    plt.figure(figsize=(50, 10))
    lines = plt.plot(
        X, vecs[0], X, vecs[1], X, vecs[2], X, vecs[3], X, vecs[4],
        X, vecs[5],X, vecs[6],X, vecs[7],X, vecs[8],
        linewidth=1.8
    )

    # 给每条线设置标签
    labels = [f"{x}" for x in index_list_name]   # 自己改成更有意义的名字
    for line, lab in zip(lines, labels):
        line.set_label(lab)
    plt.legend(loc="best")
    plt.title("各宽基指数PE图（近1年）")
    plt.xlabel("日期")
    plt.ylabel("PE")
    plt.grid(True, alpha=0.25)
    ax = plt.gca()

    # -- 注解法A：放在坐标轴比例坐标（不随数据变动，稳）
    # ax.text(0.02, 0.95, "格物致知",transform=ax.transAxes, ha="left", va="top",fontsize=16,bbox=dict(boxstyle="round,pad=0.2", fc='w', alpha=0.7))

    # 横轴（日期）刻度更精细
    if x_is_date:
        major = mdates.MonthLocator(interval=1)                # 每月一个主刻度
        minor = mdates.WeekdayLocator(byweekday=mdates.MO)     # 每周一一个次刻度
        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(major))
        ax.xaxis.set_minor_locator(minor)

    # 纵轴刻度更细（按你的PE范围调整步长）
    ax.yaxis.set_major_locator(MultipleLocator(5.0))   # 主刻度间隔 1
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))    # 每个主刻度再分 2 份

    # 刻度“更小”
    ax.tick_params(axis="both", which="major", labelsize=9, length=4)
    ax.tick_params(axis="both", which="minor", labelsize=0, length=2)

    plt.tight_layout()

    vecs2 = []
    x_vecs2 = []
    for x in industry_list:
        file_path = index_save_path + x + '.csv'
        x_col, y_col, x_is_date = fig(file_path)
        vecs2.append(y_col)
        x_vecs2.append(x_col)
    print(len(vecs2))
    # 5) 画图（不指定颜色，遵循默认配色
    plt.figure(figsize=(35, 10))

    lines2 = plt.plot(x_vecs2[0], vecs2[0], x_vecs2[1], vecs2[1], x_vecs2[2], vecs2[2], x_vecs2[3], vecs2[3], x_vecs2[4], vecs2[4],x_vecs2[5], vecs2[5], x_vecs2[6], vecs2[6], x_vecs2[7], vecs2[7], x_vecs2[8], vecs2[8], x_vecs2[9], vecs2[9],x_vecs2[10], vecs2[10], x_vecs2[11], vecs2[11],x_vecs2[12], vecs2[12],x_vecs2[13], vecs2[13],x_vecs2[14], vecs2[14],x_vecs2[15], vecs2[15],linewidth=1.8)
    # 给每条线设置标签
    labels2 = [f"{x}" for x in industry_list_name]   # 自己改成更有意义的名字
    for line, lab in zip(lines2, labels2):
        line.set_label(lab)
    plt.legend(loc="best")
    plt.title("各行业指数PE图（近1年）")
    plt.xlabel("Chart by 格物致知")
    plt.ylabel("PE")
    plt.grid(True, alpha=0.25)

    ax2 = plt.gca()

    # 注解：坐标轴比例坐标
    # ax2.text(0.02, 0.95, "格物致知",transform=ax2.transAxes, ha="left", va="top",fontsize=16,bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.7))

    # 横轴更精细
    if x_is_date:
        major2 = mdates.MonthLocator(interval=1)
        minor2 = mdates.WeekdayLocator(byweekday=mdates.MO)
        ax2.xaxis.set_major_locator(major2)
        ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(major2))
        ax2.xaxis.set_minor_locator(minor2)

    # 纵轴更细
    ax2.yaxis.set_major_locator(MultipleLocator(5.0))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # 刻度更小
    ax2.tick_params(axis="both", which="major", labelsize=9, length=4)
    ax2.tick_params(axis="both", which="minor", labelsize=0, length=2)

    plt.tight_layout()
    plt.show()
    