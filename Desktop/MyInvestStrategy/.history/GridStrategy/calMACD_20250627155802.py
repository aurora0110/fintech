import figure as fig
import numpy as np
import matplotlib.pyplot as plt

# @Description: 计算MACD 指数，计算价格的变化趋势，DIF(快白线)、DEA（慢黄线）、MACD，用于判断价格趋势的强弱 DIF穿过DEA可能需要买入 反之可能需要是卖出
# @params: data pd.DataFrame
# @params: days int
# @params: price_type str
# @return: data pd.DataFrame
def cal_macd(data, days_short=12, days_long=26, price_type='收盘'):
    data[f'EMA_{days_short}'] = data[price_type].ewm(span=days_short, adjust=False).mean()
    data[f'EMA_{days_long}'] = data[price_type].ewm(span=days_long, adjust=False).mean()

    data['DIF'] = data[f'EMA_{days_short}'] - data[f'EMA_{days_long}']
    data['DEA'] = data['DIF'].ewm(span=9, adjust=False).mean()
    # 计算MACD柱
    data['MACD'] = 2 * (data['DIF'] - data['DEA'])

    return data

def plot_moving_averages(data, ticker, colors, type):

    plt.figure(figsize=(14, 8))
    x_axis = data['date'].tolist()

    i = 0 
    for key, value in data.items():
        if i < len(colors):
            plt.plot(x_axis, value, label=key, color=colors[i], linewidth=1.5)
            i += 1
        else:
            pass
    
    # 手动设置步长
    step = 100
    # 选择x轴上的刻度
    selected_ticks = x_axis[::step]
    plt.xticks(selected_ticks, rotation=45)

    plt.xlabel('date')
    plt.ylabel('price')
    plt.text(2, 5, f'{type}', fontsize=10, color='gray') # 增加图片注释
    plt.grid(True, linestyle='--', linewidth=0.2, alpha=1)  # 增加网格线
    plt.title(f'{ticker} {type}')
    plt.legend(loc='best')
    plt.show()

