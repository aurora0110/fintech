import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def draw_stock_holdings_from_csv(csv_path, title="我的股票持仓图", save_path=None):
    # 读取 CSV
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    if '股票代码' not in df.columns or '持仓份额' not in df.columns:
        print("CSV 文件缺少必要列：'股票代码' 或 '持仓份额'")
        return

    # 排序
    df = df.sort_values(by='持仓份额', ascending=False)

    # 百分比标签
    total = df['持仓份额'].sum()
    df['占比'] = df['持仓份额'] / total * 100
    labels = [f"{p:.1f}%" for p in df['占比']]

    # 美化样式
    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = ['SimHei', 'Arial']
    
    # 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df['股票代码'], df['持仓份额'], color=sns.color_palette("viridis", len(df)))

    for bar, label in zip(bars, labels):
        plt.text(bar.get_width() + max(df['持仓份额']) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 label,
                 va='center',
                 fontsize=11)

    plt.xlabel("持仓份额", fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图表已保存为：{save_path}")
    
    plt.show()

# 使用示例
draw_stock_holdings_from_csv("/Users/lidongyang/Library/Mobile Documents/com~apple~Numbers/Documents/持仓份额.csv")
