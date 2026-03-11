import re
import re

def show(input_list, type):
    # 1. 打印监控头部（优化格式，增加[]标注）
    print('=' * 30 + f" 💡{type}监控 [开始] " + '=' * 30)
    
    # 2. 处理空列表（无数据时提示）
    if not input_list:
        print(f"【提示】💡{type}监控暂无数据")
        print('=' * 30 + f" 💡{type}监控 [结束] " + '=' * 30)
        return
    else:
        print(f"共筛选出{len(input_list)}条数据")

    # 3. 遍历列表格式化打印
    for x in input_list:
        # 原有逻辑：先获取第一个元素的第二个值，判断类型（修正原代码s[0]的错误）
        # 原代码s = input_list[0][1] 会导致所有循环都取第一个元素，这里移到循环内并修正
        s = x[1] if len(x) >= 2 else ""
        is_all_chinese = re.fullmatch(r'[\u4e00-\u9fa5]+', str(s)) is not None
        is_all_digit = re.fullmatch(r'^-?\d+(\.\d+)?$', str(s)) is not None  # 兼容小数/负数
        
        # 3.1 处理B1监控（长度4：代码+止损价+收盘价+盈亏比）
        if len(x) >= 4:
            code = x[0]
            stop_price = x[1]
            close_price = x[2]
            profit = x[3]
            
            # 价格统一保留2位小数（兼容数值/字符串）
            def format_price(price):
                try:
                    return f"{float(price):.1f}"
                except (ValueError, TypeError):
                    return price
            
            stop_p = format_price(stop_price)
            close_p = format_price(close_price)
            # 简化盈亏比提示文本
            profit_str = "请人工判断" if "请人工判断" in str(profit) else profit
            
            # 格式化输出（列对齐，用|分隔）
            print(f"股票代码{code:<6} | 止损价：{stop_p:<8} | 当日收盘价：{close_p:<10} | 盈亏比粗估：{profit_str}")
        
        # 3.2 处理持有监控（中文操作：止损/持有等）
        elif is_all_chinese and len(x) >= 2:
            print(f"股票代码{x[0]:<6} | 类型：{x[1][-3:-1]:<4} | 操作：{x[1]}")
        
        # 3.3 异常数据提示
        else:
            print(f"【异常】股票代码{x[0]}：数据格式错误，请检查 → 原始数据：{x}")
    
    # 4. 打印监控尾部
    print('=' * 30 + f" 💡{type}监控 [结束] " + '=' * 30 + "\n")
