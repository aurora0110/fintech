import re
import re


def _display_code(code, code_name_map):
    code_str = str(code)
    if code_name_map and code_str in code_name_map:
        return f"{code_str}({code_name_map[code_str]})"
    return code_str


def show(input_list, type, code_name_map=None):
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

        if type == "BRICK" and len(x) >= 9:
            code = _display_code(x[0], code_name_map)
            stop_price = x[1]
            close_price = x[2]
            total_score = x[3]
            positive_score = x[4]
            negative_score = x[5]
            positive_detail = x[6]
            negative_detail = x[7]
            note = x[8]

            def format_price(price):
                try:
                    return f"{float(price):.1f}"
                except (ValueError, TypeError):
                    return price

            stop_p = format_price(stop_price)
            close_p = format_price(close_price)
            print(
                f"股票代码{code:<6} | 止损价：{stop_p:<8} | 当日收盘价：{close_p:<8} | 总分：{total_score:<3} | "
                f"加分：{positive_score:<2} | 扣分：{negative_score:<2} | 加分原因：{positive_detail} | "
                f"扣分原因：{negative_detail} | 备注：{note}"
            )
            continue

        if type == "BRICK_RELAXED_FUSION" and len(x) >= 5:
            code = _display_code(x[0], code_name_map)
            stop_price = x[1]
            close_price = x[2]
            score = x[3]
            note = x[4]

            def format_price(price):
                try:
                    return f"{float(price):.1f}"
                except (ValueError, TypeError):
                    return price

            stop_p = format_price(stop_price)
            close_p = format_price(close_price)
            print(f"股票代码{code:<6} | 止损价(参考)：{stop_p:<8} | 当日收盘价：{close_p:<10} | 综合分：{score} | 备注：{note}")
            continue

        if type == "BRICK_CASE_RANK_LGBM_TOP20" and len(x) >= 5:
            code = _display_code(x[0], code_name_map)
            stop_price = x[1]
            close_price = x[2]
            score = x[3]
            note = x[4]

            def format_price(price):
                try:
                    return f"{float(price):.1f}"
                except (ValueError, TypeError):
                    return price

            stop_p = format_price(stop_price)
            close_p = format_price(close_price)
            print(f"股票代码{code:<6} | 止损参考：{stop_p:<8} | 当日收盘价：{close_p:<10} | 模型分：{score} | 备注：{note}")
            continue
        
        # 3.1 处理 B1 相似度 + 因子 + ML 融合监控
        if type == "B1_SIM_ML" and len(x) >= 5:
            code = _display_code(x[0], code_name_map)
            stop_price = x[1]
            close_price = x[2]
            score = x[3]
            note = x[4]

            def format_price(price):
                try:
                    return f"{float(price):.1f}"
                except (ValueError, TypeError):
                    return price

            stop_p = format_price(stop_price)
            close_p = format_price(close_price)
            print(f"股票代码{code:<6} | 止损价：{stop_p:<8} | 当日收盘价：{close_p:<10} | 综合分：{score} | 备注：{note}")
            continue

        # 3.2 处理 B1 / B3 / 其他同类监控（长度4：代码+止损价+收盘价+盈亏比）
        if len(x) >= 4:
            code = _display_code(x[0], code_name_map)
            stop_price = x[1]
            close_price = x[2]
            profit = x[3]
            note = x[4] if len(x) >= 5 else ""
            
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
            if note:
                print(f"股票代码{code:<6} | 止损价：{stop_p:<8} | 当日收盘价：{close_p:<10} | 盈亏比粗估：{profit_str} | 备注：{note}")
            else:
                print(f"股票代码{code:<6} | 止损价：{stop_p:<8} | 当日收盘价：{close_p:<10} | 盈亏比粗估：{profit_str}")
        
        # 3.2 处理持有监控（中文操作：止损/持有等）
        elif is_all_chinese and len(x) >= 2:
            print(f"股票代码{_display_code(x[0], code_name_map):<16} | 类型：{x[1][-3:-1]:<4} | 操作：{x[1]}")

        # 3.3 处理单针监控（代码 + 类型）
        elif len(x) >= 2 and type == "单针":
            code = _display_code(x[0], code_name_map)
            recommendation = x[2] if len(x) >= 3 else ""
            note = x[3] if len(x) >= 4 else ""
            if recommendation and note:
                print(f"股票代码{code:<16} | 单针类型：{x[1]} | 推荐顺序：{recommendation} | 备注：{note}")
            elif recommendation:
                print(f"股票代码{code:<16} | 单针类型：{x[1]} | 推荐顺序：{recommendation}")
            elif note:
                print(f"股票代码{code:<16} | 单针类型：{x[1]} | 备注：{note}")
            else:
                print(f"股票代码{code:<16} | 单针类型：{x[1]}")
        
        # 3.4 异常数据提示
        else:
            print(f"【异常】股票代码{x[0]}：数据格式错误，请检查 → 原始数据：{x}")
    
    # 4. 打印监控尾部
    print('=' * 30 + f" 💡{type}监控 [结束] " + '=' * 30 + "\n")
