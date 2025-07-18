import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import holdingConfig
from datetime import date

# 使用 seaborn 样式
sns.set(style="whitegrid")
current_date = date.today()
def draw_pretty_position_chart(stock_codes, holdings, title="My Holding Chart " + str(current_date), save_path=None):
    # 排序（从大到小）
    sorted_data = sorted(zip(stock_codes, holdings), key=lambda x: x[1], reverse=True)
    stock_codes_sorted, holdings_sorted = zip(*sorted_data)

    # 计算百分比
    total = sum(holdings_sorted)
    percent_labels = [f"{h/total*100:.1f}%" for h in holdings_sorted]

    # 设置中文字体（如在中文系统下自动生效）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # SimHei 是黑体

    plt.figure(figsize=(10, 6))
    bars = plt.barh(stock_codes_sorted, holdings_sorted, color=sns.color_palette("viridis", len(stock_codes)))

    # 添加百分比标签
    for bar, label in zip(bars, percent_labels):
        plt.text(bar.get_width() + max(holdings_sorted)*0.01,  # 距离右边一点
                 bar.get_y() + bar.get_height()/2,
                 label,
                 va='center',
                 fontsize=11)

    plt.xlabel("持仓份额", fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.gca().invert_yaxis()  # 最大的排最上面
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图像已保存为：{save_path}")

    plt.show()

# 示例输入
stock_codes = holdingConfig.stock_codes_20250709
holdings = holdingConfig.holdings_20250709

draw_pretty_position_chart(stock_codes, holdings)
