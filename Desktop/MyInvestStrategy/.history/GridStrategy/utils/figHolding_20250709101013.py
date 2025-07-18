import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Seaborn主题风格，美观
sns.set(style="whitegrid")

def draw_stock_holdings_from_csv(csv_path):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 检查列名
    if '股票代码' not in df.columns or '持仓份额' not in df.columns:
        raise ValueError("CSV 文件应包含 '股票代码' 和 '持仓份额' 两列")

    # 按持仓份额降序排列，图更美观
    df = df.sort_values(by='持仓份额', ascending=True)

    # 设置颜色：为每个股票分配不同颜色
    palette = sns.color_palette("husl", len(df))  # HUSL 色盘，亮丽且区分度高

    # 创建水平条形图
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df['股票代码'], df['持仓份额'], color=palette)

    # 添加数值标注
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height()/2, f'{int(width)}', va='center', fontsize=10)

    # 设置标题与轴标签
    plt.title("各股票持仓份额", fontsize=14)
    plt.xlabel("持仓份额")
    plt.ylabel("股票代码")

    plt.tight_layout()
    plt.show()

# 使用方法
draw_stock_holdings_from_csv("/Users/lidongyang/Desktop/MyInvestStrategy/持仓份额.csv")
