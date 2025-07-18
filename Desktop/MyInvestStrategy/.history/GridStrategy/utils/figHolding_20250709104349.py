import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import holdingConfig
from datetime import date

sns.set(style="whitegrid")
current_date = date.today()

def draw_pretty_position_chart(stock_codes, holdings, title="My Holding Chart " + str(current_date), save_path=None):
    sorted_data = sorted(zip(stock_codes, holdings), key=lambda x: x[1], reverse=True)
    stock_codes_sorted, holdings_sorted = zip(*sorted_data)

    total = sum(holdings_sorted)
    percent_labels = [f"{h / total * 100:.1f}%" for h in holdings_sorted]

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        stock_codes_sorted,
        holdings_sorted,
        color=sns.color_palette("viridis", len(stock_codes))
    )

    for bar, label in zip(bars, percent_labels):
        ax.text(
            bar.get_width() + max(holdings_sorted) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va='center',
            fontsize=11
        )

    # ✅ 注释文本（右上角）
    # 使用 `fig.text` 而不是 `ax.text`，避免被坐标轴裁切
    fig.text(
        0.98, 0.98,  # 相对于整个画布的右上角
        "input strategy",
        ha='right',
        va='top',
        fontsize=12,
        color='darkred'
    )

    ax.set_xlabel("amount", fontsize=12)
    ax.set_ylabel("code", fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
        # 设置坐标轴线加粗
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(2)

    # 设置刻度线加粗
    ax.tick_params(axis='both', which='major', width=2, length=6)

    # 设置刻度字体加粗
    ax.tick_params(labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.invert_yaxis()

    # 关键：关闭 tight_layout，以防遮挡 fig.text
    # 或者用 constrained_layout 来自动布局更好
    fig.tight_layout()  # 不推荐
    fig.subplots_adjust(left=0.15, right=0.85, top=0.90)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图像已保存至：{save_path}")

    plt.show()

# 示例输入
stock_codes = holdingConfig.stock_codes_20250709
holdings = holdingConfig.holdings_20250709

draw_pretty_position_chart(stock_codes, holdings)
