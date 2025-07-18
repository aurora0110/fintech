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
    percent_labels = [f"{h / total * 100:.1f}%" for h in holdings_sorted]

    # 设置中文字体（支持中文显示）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        stock_codes_sorted,
        holdings_sorted,
        color=sns.color_palette("viridis", len(stock_codes))
    )

    # 添加百分比标签
    for bar, label in zip(bars, percent_labels):
        ax.text(
            bar.get_width() + max(holdings_sorted) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va='center',
            fontsize=11
        )

    # 添加右上角注释文本
    ax.text(
        0.95, 0.95,
        "数据来源：个人持仓\n单位：元",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=12,
        color='darkred'
    )

    # 其他设置
    ax.set_xlabel("持仓金额", fontsize=12)
    ax.set_ylabel("股票代码", fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.invert_yaxis()

    # 调整边距防止注释被裁掉
    plt.subplots_adjust(top=0.9, right=0.95)

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图像已保存至：{save_path}")

    plt.show()

# 示例输入
stock_codes = holdingConfig.stock_codes_20250709
holdings = holdingConfig.holdings_20250709

draw_pretty_position_chart(stock_codes, holdings)
