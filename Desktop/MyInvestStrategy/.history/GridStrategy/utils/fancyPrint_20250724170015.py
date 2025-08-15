import os
import shutil
from colorama import Fore, Style, init

init()  # 初始化colorama

def fancy_print(message, sep_char="=", padding=2, 
               text_color=Fore.RED, sep_color=Fore.GREEN, 
               center=True):
    """
    增强版动态分隔符打印，支持颜色和居中
    
    参数：
        message: 要打印的消息
        sep_char: 分隔符字符
        padding: 消息两侧空格数
        text_color: 文字颜色(Fore.XXX)
        sep_color: 分隔符颜色
        center: 是否居中
    """
    try:
        term_width = shutil.get_terminal_size().columns
    except:
        term_width = 80
    
    # 计算可用宽度
    available_width = term_width - 2 * padding
    
    # 处理居中
    if center and len(message) < available_width:
        spaces = (available_width - len(message)) // 2
        message = " " * spaces + message
    
    # 构建分隔线
    sep_length = max(5, term_width)
    separator = sep_char * sep_length
    
    # 应用颜色
    colored_sep = sep_color + separator + Style.RESET_ALL
    colored_msg = text_color + message + Style.RESET_ALL
    
    # 格式化输出
    print(f"\n{colored_sep}")
    print(" " * padding + colored_msg)
    print(f"{colored_sep}\n")

# 示例用法
fancy_print("系统警告", sep_char="!", text_color=Fore.RED)
fancy_print("数据处理完成", sep_char="*", sep_color=Fore.GREEN)