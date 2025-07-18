import matplotlib.pyplot as plt

'''
@Param input_date 完整日期 
@Param input_price 完整价格 
@Param date_point 突出日期 
@Param price_point 突出价格 
@Param color 颜色 
@Param stock_name 股票名称 
@Param op 买入或卖出

example:
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 35]
highlight_x = [3, 5]
highlight_y = [25, 35]
fi.highlight_point(x, y, highlight_x, highlight_y, 'red', '111', 'buy')
'''
# 绘制highlight点图，可用于突出卖出买入点
def highlight_point(input_date, input_price, date_point, price_point, color, stock_name, op):
    plt.plot(input_date, input_price, label = stock_name)
    plt.scatter(date_point, price_point, color = color, label = op)
    plt.legend()
    plt.show()

def figure_line(input_date, input_data, label1, color1):
    plt.figure(figsize=(12, 6))
    plt.plot(input_date, input_data, label = label1, color = color1)
    plt.legend()
    plt.show()

def figure_line(input_date, input_data1, input_data2, label1, label2, color1, color2):
    plt.figure(figsize=(12, 6))
    plt.plot(input_date, input_data1, label = label1, color = color1)
    plt.plot(input_date, input_data2, label = label2, color = color2)
    plt.legend()
    plt.show()

def figure_line(input_date, input_data1, input_data2, input_data3, label1, label2, label3, color1, color2, color3):
    plt.figure(figsize=(12, 6))
    plt.plot(input_date, input_data1, label = label1, color = color1)
    plt.plot(input_date, input_data2, label = label2, color = color2)
    plt.plot(input_date, input_data3, label = label3, color = color3)
    plt.legend()
    plt.show()

def figure_line(input_date, input_data1, input_data2, input_data3, input_data4, label1, label2, label3, label4, color1, color2, color3, color4):
    plt.figure(figsize=(12, 6))
    plt.plot(input_date, input_data1, label = label1, color = color1)
    plt.plot(input_date, input_data2, label = label2, color = color2)
    plt.plot(input_date, input_data3, label = label3, color = color3)
    plt.plot(input_date, input_data4, label = label4, color = color4)
    plt.legend()
    plt.show()

