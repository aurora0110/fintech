# 读取配置文件
import yaml
import random
import secrets

def show(hold_path):
    hold_list = []
    with open(hold_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 提取配置参数
    strategy_name = config["strategy"]["name"]
    strategy_time = config["strategy"]["update_time"]
    strategy_hold = config["hold_stocks"]
    strategy_watch = config["watch_stocks"]
    strategy_period = config["strategy"]["active_period"]
    strategy_note = config["strategy"]["note"]


    # 构造工整的输出内容
    output = f"当前持仓策略：{strategy_name}\n"
    output += f"策略更新时间：{strategy_time}\n"
    output += "=" * 80 + "\n"  # 分隔线
    output += "当前持仓明细：\n"

    # 逐条格式化持仓信息
    for idx, stock in enumerate(strategy_hold, 1):
        output += f"\n【持仓{idx}】\n"
        output += f"  股票代码：{stock['stock_code']}\n"
        output += f"  股票名称：{stock['stock_name']}\n"
        output += f"  买入类型：{stock['type']}\n"
        output += f"  止损价格：{stock['stop_loss_price']:.2f}元\n"
        output += f"  持仓天数：{stock['hold_days']}天\n"
        output += f"  持仓占比：{stock.get('position_ratio', '无')}\n"  # 处理可选字段
        output += f"  备注信息：{stock['note']}\n"
        output += "-" * 60 + "\n"  # 每条持仓分隔线
        hold_list.append([stock['stock_code'],stock['stock_name'],stock['type'],stock['stop_loss_price'],stock['hold_days'],stock['position_ratio'],stock['note'], stock['take_profit_price']])

    for idx, stock in enumerate(strategy_watch, 1):
        output += f"\n【关注{idx}】\n"
        output += f"  股票代码：{stock['stock_code']}\n"
        output += f"  股票名称：{stock['stock_name']}\n"
        output += f"  关注类型：{stock['type']}\n"
        output += f"  备注信息：{stock['note']}\n"
        output += "-" * 60 + "\n"  # 每条持仓分隔线

    # 初始化总占比
    total_ratio = 0.0
    # 获取持仓股票列表
    hold_stocks = config.get("hold_stocks", [])
    # 遍历每个持仓股票
    for stock in hold_stocks:
        # 提取持仓占比字段（处理字段不存在的情况）
        ratio_str = stock.get("position_ratio", "0%")
        # 去除百分号，转换为浮点数
        try:
            ratio = float(ratio_str.replace("%", ""))
            total_ratio += ratio
        except ValueError:
            # 处理格式异常的情况（比如字段不是数字+%）
            print(f"警告：股票 {stock.get('stock_code', '未知代码')} 的持仓占比格式错误：{ratio_str}，已按0%处理")
    
    # 打印最终结果
    print(output)

    if total_ratio > 50 and strategy_period <= -2.3:
        print("💡当前总持仓占比为：", total_ratio, "%", "\n", "⚠️当前活跃市值处于下行区间，总持仓占比超过50%，请降低仓位！", "\n", "-" * 60, "\n")
    elif total_ratio < 50 and strategy_period >= 4:
        print("💡当前总持仓占比为：", total_ratio, "%", "\n", "⚠️当前活跃市值处于上行区间，总持仓占比低于50%，请增加仓位！", "\n", "-" * 60, "\n")
    else:
        print("💡当前总持仓占比为：", total_ratio, "%", "\n", "⚠️当前活跃市值区间与持仓占比比例正常，无需调整仓位。", "\n", "-" * 60, "\n")
    print(f"💡{secrets.choice(strategy_note)}", "\n", "-" * 60 , "\n")
    # 返回持仓明细
    return hold_list