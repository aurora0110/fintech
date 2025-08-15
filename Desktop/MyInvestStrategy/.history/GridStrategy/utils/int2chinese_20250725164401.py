
def int_to_chinese_num(num):
    '''
    转换整数为中文数字
    '''
    if not isinstance(num, int):
        return "请输入整数"

    # 处理负数情况
    sign = ""
    if num < 0:
        sign = "-"
        num = abs(num)

    if num == 0:
        return "零"

    digit_map = ["", "十", "百", "千"]
    unit_map = ["", "万", "亿", "兆"]  # 可扩展更高单位
    num_str = str(num)
    length = len(num_str)
    result = []

    # 每4位一组（中文数字以万为单位）
    for i in range(0, length, 4):
        segment = num_str[max(0, length - i - 4): length - i]
        segment_len = len(segment)
        segment_str = ""

        # 处理每一段（千、百、十、个位）
        for j in range(segment_len):
            digit = int(segment[j])
            if digit == 0:
                continue  # 零不单独显示，除非在中间（如 1001 → 一千零一）
            # 添加数字和单位（如 "3" + "百" → "3百"）
            segment_str += str(digit) + digit_map[segment_len - j - 1]

        # 添加段单位（万、亿等）
        if segment_str:  # 如果该段不为空
            segment_str += unit_map[i // 4]
        result.append(segment_str)

    # 拼接所有段（从高到低）
    chinese_num = "".join(reversed(result))

    # 处理连续的零（如 "1001" → "一千零一"）
    chinese_num = chinese_num.replace("零零", "零").strip("零")
    
    # 加上符号（如果是负数）
    return sign + chinese_num if chinese_num else "零"