import calMA
import calKDJ
import config
import shakeoutMonitoring as som

'''
sfzf ➕ 单针下20 ➕ bbi
'''

if __name__ == '__main__':
    # 读取文件路径
    file_path = config.file_path 
    windows = config.windows
    colors = config.colors
    symbol_list = config.symbol_list
    start_date = config.start_date
    end_date = config.end_date

        # 获取当前时间
    now = datetime.now()
    now_date = now.strftime("%Y%m%d")