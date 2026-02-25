"""
B1交易策略回测工具

功能说明：
1. 实现基于KDJ指标、MACD指标和趋势线的B1交易策略
2. 支持多种止损条件设置，如价格止损、时间止损、连续下跌止损等
3. 支持多种止盈条件设置，如J值止盈、涨幅止盈、偏离趋势线止盈
4. 实现锁仓逻辑和锁仓卖出条件
5. 支持参数组合生成和测试，找出最优参数组合
6. 计算策略的成功率、年化收益率等绩效指标
7. 生成详细的回测报告，包括表现最好和最差的股票

交易规则：
1. 基本条件：当知行短期趋势线>知行多空线且MACD DIFF线>0时可以买入
2. 买入条件：当J值<a时，第二天按照开盘价全仓买入
3. 止损条件：价格止损、时间止损、连续下跌止损、成交量异常止损等
4. 止盈条件：J值止盈、涨幅止盈、偏离趋势线止盈
5. 锁仓逻辑：连续盈利止盈o次且累计收益率大于p%后锁仓
"""
import os
import pandas as pd
import numpy as np
import itertools
from technical_indicators import calculate_trend, calculate_kdj, calculate_macd

class B1StrategyBacktest:
    def __init__(self, data_dir, results_file):
        """
        初始化回测器
        :param data_dir: 数据目录路径
        :param results_file: 结果保存文件路径
        """
        self.data_dir = data_dir
        self.results_file = results_file
        self.results = []
        self.initial_capital = 1000000  # 每只股票的初始资金为100万元
        self.data_cache = {}  # 股票数据缓存
        self.tech_indicators_cache = {}  # 技术指标缓存
    
    def load_stock_data(self, file_path):
        """
        加载单个股票数据
        :param file_path: 文件路径
        :return: 股票数据DataFrame
        """
        try:
            # 读取文件（空格分隔）
            with open(file_path, 'r', encoding='gbk') as f:
                lines = f.readlines()
            
            # 解析数据
            data = []
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                # 跳过表头
                if i == 0 and '开盘' in line:
                    continue
                # 按空格分隔
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        # 日期格式：2021/08/02
                        date_str = parts[0]
                        # 价格和成交量
                        open_price = float(parts[1])
                        high_price = float(parts[2])
                        low_price = float(parts[3])
                        close_price = float(parts[4])
                        volume = float(parts[5])
                        
                        data.append([date_str, open_price, high_price, low_price, close_price, volume])
                    except ValueError:
                        # 跳过无法解析的行
                        continue
            
            # 创建DataFrame
            df = pd.DataFrame(data, columns=['日期', '开盘', '最高', '最低', '收盘', '成交量'])
            
            # 确保数据按日期排序
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)
            
            return df
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {str(e)}")
            return None
    
    def run_strategy(self, df, stock_code, params):
        """
        运行回测策略
        :param df: 股票数据DataFrame
        :param stock_code: 股票代码
        :param params: 策略参数字典
        :return: 回测结果字典
        """
        # 提取参数
        a = params['a']  # J值买入阈值
        b = params['b']  # 时间止损天数
        c = params['c']  # 时间止损最小涨幅百分比
        d = params['d']  # 连续下跌止损天数
        e = params['e']  # 成交量异常天数
        f = params['f']  # J值止盈卖出比例
        g = params['g']  # 涨幅止盈阈值
        h = params['h']  # 涨幅止盈卖出比例
        i = params['i']  # 偏离趋势线止盈卖出比例
        j = params['j']  # 止盈后暂停天数
        n = params['n']  # J值止盈阈值
        p = params['p']  # 锁仓累计收益率阈值
        lock_exit_condition = params.get('lock_exit_condition', 'k')  # 锁仓卖出条件
        
        # 计算技术指标（使用缓存）
        if stock_code in self.tech_indicators_cache:
            df = self.tech_indicators_cache[stock_code]
        else:
            df = calculate_trend(df)
            df = calculate_kdj(df)
            df = calculate_macd(df)
            self.tech_indicators_cache[stock_code] = df
        
        # 初始化变量
        position = 0  # 当前仓位（0-1之间）
        entry_price = 0  # 买入价格（开盘价）
        entry_date = None
        entry_low = 0  # 买入当根K线的最低价
        prev_low = 0  # 买入前一天的最低价
        exit_price = 0
        exit_date = None
        trades = []
        buy_records = []  # 记录每次买入的价格和仓位
        
        # 标记趋势线金叉死叉
        df['趋势线金叉'] = (df['知行短期趋势线'] > df['知行多空线']) & (df['知行短期趋势线'].shift(1) <= df['知行多空线'].shift(1))
        df['趋势线死叉'] = (df['知行短期趋势线'] < df['知行多空线']) & (df['知行短期趋势线'].shift(1) >= df['知行多空线'].shift(1))
        
        # 标记MACD DIFF线>0
        df['MACD_DIFF>0'] = df['MACD_DIFF'] > 0
        
        # 计算最近e日内成交量
        df['最近e日成交量'] = df['成交量'].rolling(window=e).mean()
        
        # 标记可以开始买入的点
        df['可以买入'] = False
        can_buy = False
        
        # 连续盈利止盈次数和累计收益率
        consecutive_win = 0
        cumulative_return = 0
        
        # 锁仓相关变量
        locked_position = 0
        lock_entry_price = 0
        lock_entry_date = None
        
        # 止盈后暂停天数计数器
        stop_days = 0
        
        # 连续下跌天数计数器
        consecutive_down_days = 0
        
        # 连续收盘价小于知行多空线计数器
        consecutive_below_multiple_days = 0
        
        # 标记是否需要等待收盘价重新打上知行多空线
        wait_for_break_above_multiple = False
        
        # 锁仓卖出条件相关变量
        below_trend_line_days = 0
        consecutive_below_trend_line_days = 0
        
        for i in range(len(df)):
            # 检查趋势线金叉
            if df['趋势线金叉'].iloc[i]:
                can_buy = True
                df.loc[i, '可以买入'] = True
            
            # 检查趋势线死叉
            if df['趋势线死叉'].iloc[i]:
                can_buy = False
                wait_for_break_above_multiple = False
                # 死叉时全仓平仓
                if position > 0:
                    exit_price = df['收盘'].iloc[i]
                    exit_date = df['日期'].iloc[i]
                    
                    # 计算总盈利
                    total_profit = 0
                    for record in buy_records:
                        buy_price = record['price']
                        buy_position = record['position']
                        if buy_price > 0:
                            profit = (exit_price - buy_price) / buy_price * buy_position
                            total_profit += profit
                            trades.append({
                                'entry_date': record['date'],
                                'exit_date': exit_date,
                                'entry_price': buy_price,
                                'exit_price': exit_price,
                                'profit': (exit_price - buy_price) / buy_price,
                                'position': buy_position,
                                'type': '趋势线死叉卖出'
                            })
                    
                    # 更新累计收益率
                    cumulative_return += total_profit
                    
                    # 重置仓位
                    position = 0
                    buy_records = []
                
                # 锁仓也需要平仓
                if locked_position > 0:
                    exit_price = df['收盘'].iloc[i]
                    exit_date = df['日期'].iloc[i]
                    
                    profit = (exit_price - lock_entry_price) / lock_entry_price * locked_position
                    cumulative_return += profit
                    
                    trades.append({
                        'entry_date': lock_entry_date,
                        'exit_date': exit_date,
                        'entry_price': lock_entry_price,
                        'exit_price': exit_price,
                        'profit': (exit_price - lock_entry_price) / lock_entry_price,
                        'position': locked_position,
                        'type': '趋势线死叉锁仓卖出'
                    })
                    
                    locked_position = 0
                    lock_entry_price = 0
                    lock_entry_date = None
            
            # 检查是否需要等待收盘价重新打上知行多空线
            if wait_for_break_above_multiple:
                if df['收盘'].iloc[i] > df['知行多空线'].iloc[i]:
                    wait_for_break_above_multiple = False
                    can_buy = True
                else:
                    continue
            
            # 检查止盈后暂停天数
            if stop_days > 0:
                stop_days -= 1
                continue
            
            # 检查买入条件：可以买入，仓位为0，前一天J值<a，MACD DIFF>0
            if can_buy and position == 0 and i > 0:
                if df['J'].iloc[i-1] < a and df['MACD_DIFF>0'].iloc[i]:
                    # 用当天的开盘价买入
                    buy_price = df['开盘'].iloc[i]
                    # 避免买入价格为0
                    if buy_price > 0:
                        # 全仓买入
                        buy_position = 1.0
                        entry_date = df['日期'].iloc[i]
                        entry_low = df['最低'].iloc[i]  # 买入当根K线的最低价
                        prev_low = df['最低'].iloc[i-1]  # 买入前一天的最低价
                        
                        # 记录买入
                        buy_records.append({
                            'date': entry_date,
                            'price': buy_price,
                            'position': buy_position,
                            'buy_index': i,  # 记录买入的索引位置
                            'buy_price': buy_price  # 记录买入价格用于计算涨幅
                        })
                        
                        # 更新仓位
                        position = buy_position
                        entry_price = buy_price
            
            # 检查止损条件
            if position > 0 and i > 0:
                current_close = df['收盘'].iloc[i]
                current_open = df['开盘'].iloc[i]
                current_volume = df['成交量'].iloc[i]
                
                # 1. 价格止损：收盘价小于买入当天K线最低价与前一天K线最低价的最小值
                if current_close < min(entry_low, prev_low):
                    exit_price = current_close
                    exit_date = df['日期'].iloc[i]
                    
                    # 全仓平仓
                    total_profit = 0
                    for record in buy_records:
                        buy_price = record['price']
                        buy_position = record['position']
                        if buy_price > 0:
                            profit = (exit_price - buy_price) / buy_price * buy_position
                            total_profit += profit
                            trades.append({
                                'entry_date': record['date'],
                                'exit_date': exit_date,
                                'entry_price': buy_price,
                                'exit_price': exit_price,
                                'profit': (exit_price - buy_price) / buy_price,
                                'position': buy_position,
                                'type': '价格止损卖出'
                            })
                    
                    # 更新累计收益率
                    cumulative_return += total_profit
                    if total_profit > 0:
                        consecutive_win += 1
                    else:
                        consecutive_win = 0
                    
                    # 重置仓位
                    position = 0
                    buy_records = []
                    continue
                
                # 2. 时间止损：b天未上涨c%
                for record in buy_records.copy():
                    buy_index = record['buy_index']
                    buy_price = record['price']
                    days_held = i - buy_index
                    
                    if days_held == b:
                        # 计算涨幅
                        change_pct = (current_close - buy_price) / buy_price * 100
                        
                        # 如果涨幅未达到要求，平仓
                        if change_pct < c:
                            exit_price = current_close
                            exit_date = df['日期'].iloc[i]
                            
                            # 计算盈利
                            profit = (exit_price - buy_price) / buy_price * record['position']
                            cumulative_return += profit
                            
                            # 记录交易
                            trades.append({
                                'entry_date': record['date'],
                                'exit_date': exit_date,
                                'entry_price': buy_price,
                                'exit_price': exit_price,
                                'profit': (exit_price - buy_price) / buy_price,
                                'position': record['position'],
                                'type': f'时间止损卖出({b}天未涨{c}%)'
                            })
                            
                            # 更新仓位
                            position -= record['position']
                            buy_records.remove(record)
                            
                            if profit > 0:
                                consecutive_win += 1
                            else:
                                consecutive_win = 0
                
                if position == 0:
                    buy_records = []
                    continue
                
                # 3. 连续两天收盘价小于知行多空线
                if current_close < df['知行多空线'].iloc[i]:
                    consecutive_below_multiple_days += 1
                else:
                    consecutive_below_multiple_days = 0
                
                if consecutive_below_multiple_days >= 2:
                    exit_price = current_close
                    exit_date = df['日期'].iloc[i]
                    
                    # 全仓平仓
                    total_profit = 0
                    for record in buy_records:
                        buy_price = record['price']
                        buy_position = record['position']
                        if buy_price > 0:
                            profit = (exit_price - buy_price) / buy_price * buy_position
                            total_profit += profit
                            trades.append({
                                'entry_date': record['date'],
                                'exit_date': exit_date,
                                'entry_price': buy_price,
                                'exit_price': exit_price,
                                'profit': (exit_price - buy_price) / buy_price,
                                'position': buy_position,
                                'type': '连续低于多空线止损卖出'
                            })
                    
                    # 更新累计收益率
                    cumulative_return += total_profit
                    if total_profit > 0:
                        consecutive_win += 1
                    else:
                        consecutive_win = 0
                    
                    # 重置仓位
                    position = 0
                    buy_records = []
                    
                    # 标记需要等待收盘价重新打上知行多空线
                    wait_for_break_above_multiple = True
                    continue
                
                # 4. 连续d天收盘价小于前一天的最低价
                if current_close < df['收盘'].iloc[i-1]:
                    consecutive_down_days += 1
                else:
                    consecutive_down_days = 0
                
                if consecutive_down_days >= d:
                    exit_price = current_close
                    exit_date = df['日期'].iloc[i]
                    
                    # 全仓平仓
                    total_profit = 0
                    for record in buy_records:
                        buy_price = record['price']
                        buy_position = record['position']
                        if buy_price > 0:
                            profit = (exit_price - buy_price) / buy_price * buy_position
                            total_profit += profit
                            trades.append({
                                'entry_date': record['date'],
                                'exit_date': exit_date,
                                'entry_price': buy_price,
                                'exit_price': exit_price,
                                'profit': (exit_price - buy_price) / buy_price,
                                'position': buy_position,
                                'type': f'连续{d}天下跌止损卖出'
                            })
                    
                    # 更新累计收益率
                    cumulative_return += total_profit
                    if total_profit > 0:
                        consecutive_win += 1
                    else:
                        consecutive_win = 0
                    
                    # 重置仓位
                    position = 0
                    buy_records = []
                    consecutive_down_days = 0
                    continue
                
                # 5. 成交量异常止损：前一日收盘价<当日收盘价<当日开盘价且当日成交量大于最近e日内成交量
                if i > 0:
                    prev_close = df['收盘'].iloc[i-1]
                    if prev_close < current_close < current_open and current_volume > df['最近e日成交量'].iloc[i]:
                        exit_price = current_close
                        exit_date = df['日期'].iloc[i]
                        
                        # 全仓平仓
                        total_profit = 0
                        for record in buy_records:
                            buy_price = record['price']
                            buy_position = record['position']
                            if buy_price > 0:
                                profit = (exit_price - buy_price) / buy_price * buy_position
                                total_profit += profit
                                trades.append({
                                    'entry_date': record['date'],
                                    'exit_date': exit_date,
                                    'entry_price': buy_price,
                                    'exit_price': exit_price,
                                    'profit': (exit_price - buy_price) / buy_price,
                                    'position': buy_position,
                                    'type': '成交量异常止损卖出'
                                })
                        
                        # 更新累计收益率
                        cumulative_return += total_profit
                        if total_profit > 0:
                            consecutive_win += 1
                        else:
                            consecutive_win = 0
                        
                        # 重置仓位
                        position = 0
                        buy_records = []
                        continue
            
            # 检查止盈条件
            if position > 0 and i > 0:
                current_close = df['收盘'].iloc[i]
                current_trend_line = df['知行短期趋势线'].iloc[i]
                
                # 计算平均买入价格
                avg_buy_price = sum([record['price'] * record['position'] for record in buy_records]) / position if position > 0 else 0
                
                if avg_buy_price > 0:
                    # 1. J值止盈：J值>n则卖出f%的总仓位
                    if df['J'].iloc[i] > n:
                        sell_position = position * f / 100
                        if sell_position > 0:
                            exit_price = current_close
                            exit_date = df['日期'].iloc[i]
                            
                            # 按比例卖出每个买入记录
                            sold_position = 0
                            for record in buy_records.copy():
                                if sold_position >= sell_position:
                                    break
                                
                                # 计算本次卖出的仓位
                                sell_amount = min(record['position'], sell_position - sold_position)
                                
                                # 计算盈利
                                profit = (exit_price - record['price']) / record['price'] * sell_amount
                                cumulative_return += profit
                                
                                # 记录交易
                                trades.append({
                                    'entry_date': record['date'],
                                    'exit_date': exit_date,
                                    'entry_price': record['price'],
                                    'exit_price': exit_price,
                                    'profit': (exit_price - record['price']) / record['price'],
                                    'position': sell_amount,
                                    'type': f'J值止盈卖出({f}%)'
                                })
                                
                                # 更新记录
                                record['position'] -= sell_amount
                                sold_position += sell_amount
                                
                                # 如果该记录仓位为0，移除
                                if record['position'] <= 0:
                                    buy_records.remove(record)
                            
                            # 更新仓位
                            position -= sell_position
                            
                            # 止盈后暂停j天
                            stop_days = j
                            
                            # 更新连续盈利次数
                            if profit > 0:
                                consecutive_win += 1
                            else:
                                consecutive_win = 0
                    
                    # 2. 涨幅止盈：上涨幅度达到g%以上则卖出h%的总仓位并重新计算上涨幅度
                    for record in buy_records:
                        buy_price = record['buy_price']
                        if buy_price > 0:
                            change_pct = (current_close - buy_price) / buy_price * 100
                            if change_pct >= g:
                                sell_position = position * h / 100
                                if sell_position > 0:
                                    exit_price = current_close
                                    exit_date = df['日期'].iloc[i]
                                    
                                    # 按比例卖出每个买入记录
                                    sold_position = 0
                                    for r in buy_records.copy():
                                        if sold_position >= sell_position:
                                            break
                                        
                                        # 计算本次卖出的仓位
                                        sell_amount = min(r['position'], sell_position - sold_position)
                                        
                                        # 计算盈利
                                        profit = (exit_price - r['price']) / r['price'] * sell_amount
                                        cumulative_return += profit
                                        
                                        # 记录交易
                                        trades.append({
                                            'entry_date': r['date'],
                                            'exit_date': exit_date,
                                            'entry_price': r['price'],
                                            'exit_price': exit_price,
                                            'profit': (exit_price - r['price']) / r['price'],
                                            'position': sell_amount,
                                            'type': f'涨幅止盈卖出({h}%)'
                                        })
                                        
                                        # 更新记录
                                        r['position'] -= sell_amount
                                        r['buy_price'] = current_close  # 重新计算上涨幅度的基准价格
                                        sold_position += sell_amount
                                        
                                        # 如果该记录仓位为0，移除
                                        if r['position'] <= 0:
                                            buy_records.remove(r)
                                    
                                    # 更新仓位
                                    position -= sell_position
                                    
                                    # 止盈后暂停j天
                                    stop_days = j
                                    
                                    # 更新连续盈利次数
                                    if profit > 0:
                                        consecutive_win += 1
                                    else:
                                        consecutive_win = 0
                    
                    # 3. 偏离趋势线止盈：收盘价高于知行短期趋势线10%、15%、25%以上
                    if current_trend_line > 0:
                        deviation_pct = (current_close - current_trend_line) / current_trend_line * 100
                        if deviation_pct >= 25:
                            sell_position = position * i / 100
                            sell_type = '偏离趋势线25%止盈卖出'
                        elif deviation_pct >= 15:
                            sell_position = position * i / 100
                            sell_type = '偏离趋势线15%止盈卖出'
                        elif deviation_pct >= 10:
                            sell_position = position * i / 100
                            sell_type = '偏离趋势线10%止盈卖出'
                        else:
                            sell_position = 0
                            sell_type = ''
                        
                        if sell_position > 0:
                            exit_price = current_close
                            exit_date = df['日期'].iloc[i]
                            
                            # 按比例卖出每个买入记录
                            sold_position = 0
                            for record in buy_records.copy():
                                if sold_position >= sell_position:
                                    break
                                
                                # 计算本次卖出的仓位
                                sell_amount = min(record['position'], sell_position - sold_position)
                                
                                # 计算盈利
                                profit = (exit_price - record['price']) / record['price'] * sell_amount
                                cumulative_return += profit
                                
                                # 记录交易
                                trades.append({
                                    'entry_date': record['date'],
                                    'exit_date': exit_date,
                                    'entry_price': record['price'],
                                    'exit_price': exit_price,
                                    'profit': (exit_price - record['price']) / record['price'],
                                    'position': sell_amount,
                                    'type': f'{sell_type}({i}%)'
                                })
                                
                                # 更新记录
                                record['position'] -= sell_amount
                                sold_position += sell_amount
                                
                                # 如果该记录仓位为0，移除
                                if record['position'] <= 0:
                                    buy_records.remove(record)
                            
                            # 更新仓位
                            position -= sell_position
                            
                            # 止盈后暂停j天
                            stop_days = j
                            
                            # 更新连续盈利次数
                            if profit > 0:
                                consecutive_win += 1
                            else:
                                consecutive_win = 0
            
            # 检查锁仓条件：连续盈利止盈o次，且累计收益率大于p%
            if position > 0 and consecutive_win >= 2 and cumulative_return * 100 >= p:
                # 锁仓：将当前仓位转为锁仓仓位
                lock_entry_price = sum([record['price'] * record['position'] for record in buy_records]) / position
                lock_entry_date = df['日期'].iloc[i]
                
                # 记录锁仓
                trades.append({
                    'entry_date': lock_entry_date,
                    'exit_date': None,
                    'entry_price': lock_entry_price,
                    'exit_price': None,
                    'profit': 0,
                    'position': position,
                    'type': f'锁仓({consecutive_win}次连续盈利，累计收益{p}%)'
                })
                
                # 更新锁仓仓位
                locked_position = position
                position = 0
                buy_records = []
                consecutive_win = 0
            
            # 检查锁仓卖出条件
            if locked_position > 0:
                current_close = df['收盘'].iloc[i]
                current_trend_line = df['知行短期趋势线'].iloc[i]
                
                if lock_exit_condition == 'k':
                    # k: 收盘价跌破知行短期趋势线2天后平仓
                    if current_close < current_trend_line:
                        below_trend_line_days += 1
                    else:
                        below_trend_line_days = 0
                    
                    if below_trend_line_days >= 2:
                        exit_price = current_close
                        exit_date = df['日期'].iloc[i]
                        
                        # 计算盈利
                        profit = (exit_price - lock_entry_price) / lock_entry_price * locked_position
                        cumulative_return += profit
                        
                        # 记录交易
                        trades.append({
                            'entry_date': lock_entry_date,
                            'exit_date': exit_date,
                            'entry_price': lock_entry_price,
                            'exit_price': exit_price,
                            'profit': (exit_price - lock_entry_price) / lock_entry_price,
                            'position': locked_position,
                            'type': '锁仓卖出(k条件)'
                        })
                        
                        # 重置锁仓仓位
                        locked_position = 0
                        lock_entry_price = 0
                        lock_entry_date = None
                        below_trend_line_days = 0
                
                elif lock_exit_condition == 'l':
                    # l: 连续两天收盘价小于知行趋势线平仓
                    if current_close < current_trend_line:
                        consecutive_below_trend_line_days += 1
                    else:
                        consecutive_below_trend_line_days = 0
                    
                    if consecutive_below_trend_line_days >= 2:
                        exit_price = current_close
                        exit_date = df['日期'].iloc[i]
                        
                        # 计算盈利
                        profit = (exit_price - lock_entry_price) / lock_entry_price * locked_position
                        cumulative_return += profit
                        
                        # 记录交易
                        trades.append({
                            'entry_date': lock_entry_date,
                            'exit_date': exit_date,
                            'entry_price': lock_entry_price,
                            'exit_price': exit_price,
                            'profit': (exit_price - lock_entry_price) / lock_entry_price,
                            'position': locked_position,
                            'type': '锁仓卖出(l条件)'
                        })
                        
                        # 重置锁仓仓位
                        locked_position = 0
                        lock_entry_price = 0
                        lock_entry_date = None
                        consecutive_below_trend_line_days = 0
                
                elif lock_exit_condition == 'm':
                    # m: 连续两天收盘价小于知行趋势线先卖出一半剩余的仓位，收盘价跌破知行短期趋势线2天后卖出剩余所有的仓位
                    if current_close < current_trend_line:
                        consecutive_below_trend_line_days += 1
                        below_trend_line_days += 1
                    else:
                        consecutive_below_trend_line_days = 0
                        below_trend_line_days = 0
                    
                    if consecutive_below_trend_line_days == 2:
                        # 卖出一半仓位
                        sell_position = locked_position / 2
                        exit_price = current_close
                        exit_date = df['日期'].iloc[i]
                        
                        # 计算盈利
                        profit = (exit_price - lock_entry_price) / lock_entry_price * sell_position
                        cumulative_return += profit
                        
                        # 记录交易
                        trades.append({
                            'entry_date': lock_entry_date,
                            'exit_date': exit_date,
                            'entry_price': lock_entry_price,
                            'exit_price': exit_price,
                            'profit': (exit_price - lock_entry_price) / lock_entry_price,
                            'position': sell_position,
                            'type': '锁仓卖出(m条件-一半)'
                        })
                        
                        # 更新锁仓仓位
                        locked_position -= sell_position
                    
                    if below_trend_line_days >= 2 and locked_position > 0:
                        # 卖出剩余所有仓位
                        exit_price = current_close
                        exit_date = df['日期'].iloc[i]
                        
                        # 计算盈利
                        profit = (exit_price - lock_entry_price) / lock_entry_price * locked_position
                        cumulative_return += profit
                        
                        # 记录交易
                        trades.append({
                            'entry_date': lock_entry_date,
                            'exit_date': exit_date,
                            'entry_price': lock_entry_price,
                            'exit_price': exit_price,
                            'profit': (exit_price - lock_entry_price) / lock_entry_price,
                            'position': locked_position,
                            'type': '锁仓卖出(m条件-剩余)'
                        })
                        
                        # 重置锁仓仓位
                        locked_position = 0
                        lock_entry_price = 0
                        lock_entry_date = None
                        below_trend_line_days = 0
        
        # 计算回测结果
        if trades:
            total_profit = sum([trade['profit'] for trade in trades])
            num_trades = len(trades)
            win_rate = len([trade for trade in trades if trade['profit'] > 0]) / num_trades
            avg_profit = total_profit / num_trades
            
            # 计算最大回撤
            cumulative_profit = 1
            max_cumulative = 1
            max_drawdown = 0
            
            for trade in trades:
                cumulative_profit *= (1 + trade['profit'])
                if cumulative_profit > max_cumulative:
                    max_cumulative = cumulative_profit
                current_drawdown = (max_cumulative - cumulative_profit) / max_cumulative
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
            
            # 计算年化收益率
            # 按照用户操作思路：使用股票数据的完整时间范围
            # 计算公式：(最终总金额 - 初始金额) / 初始金额 / 年数
            if len(df) > 0:
                # 使用股票数据的完整时间范围
                first_date = df['日期'].iloc[0]
                last_date = df['日期'].iloc[-1]
                total_days = (last_date - first_date).days
                
                if total_days > 0:
                    # 计算年数
                    years = total_days / 365.0
                    
                    if years > 0:
                        # 计算年化收益率
                        # 初始金额 = 100万
                        initial_amount = 1.0  # 归一化处理
                        # 最终总金额 = cumulative_profit（已经归一化）
                        final_amount = cumulative_profit
                        # 年化收益率 = (最终总金额 - 初始金额) / 初始金额 / 年数
                        annual_return = (final_amount - initial_amount) / initial_amount / years
                    else:
                        annual_return = 0
                else:
                    annual_return = 0
            else:
                annual_return = 0
            
            # 计算成功率（胜率）
            success_rate = win_rate
            
            return {
                'stock_code': stock_code,
                'num_trades': num_trades,
                'total_profit': total_profit,
                'avg_profit_per_trade': avg_profit,
                'success_rate': success_rate,
                'win_rate': win_rate,  # 保留原字段以保持兼容性
                'max_drawdown': max_drawdown,
                'cumulative_return': cumulative_profit - 1,
                'annual_return': annual_return,
                'trades': trades,
                'params': params
            }
        else:
            return {
                'stock_code': stock_code,
                'num_trades': 0,
                'total_profit': 0,
                'avg_profit_per_trade': 0,
                'success_rate': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'cumulative_return': 0,
                'annual_return': 0,
                'trades': [],
                'params': params
            }
    
    def generate_params(self):
        """
        生成所有参数组合
        :return: 参数组合列表
        """
        # 固定参数值
        a = -5  # J值买入阈值
        e = 30  # 成交量异常天数
        n = 100  # J值止盈阈值
        p = 70  # 锁仓累计收益率阈值
        f = 10  # J值止盈卖出比例
        g = 8  # 涨幅止盈阈值
        h = 10  # 涨幅止盈卖出比例
        i = 10  # 偏离趋势线止盈卖出比例
        j = 1  # 止盈后暂停天数
        
        # 参数范围
        b_values = [3, 4]  # 时间止损天数
        c_values = [1, 2]  # 时间止损最小涨幅百分比
        d_values = [2, 3]  # 连续下跌止损天数
        lock_exit_conditions = ['k', 'l', 'm']  # 锁仓卖出条件
        
        # 生成参数组合（使用itertools.product优化）
        params_list = []
        param_values = list(itertools.product(
            [a], b_values, c_values, d_values, [e],
            [f], [g], [h], [i], [j],
            [n], [p], lock_exit_conditions
        ))
        
        for params in param_values:
            param_dict = {
                'a': params[0],
                'b': params[1],
                'c': params[2],
                'd': params[3],
                'e': params[4],
                'f': params[5],
                'g': params[6],
                'h': params[7],
                'i': params[8],
                'j': params[9],
                'n': params[10],
                'p': params[11],
                'lock_exit_condition': params[12]
            }
            params_list.append(param_dict)
        
        return params_list
    
    def run(self, params_list=None, test_single_stock=False, test_mode=False):
        """
        运行整个回测
        :param params_list: 参数组合列表，如果为None则生成所有参数组合
        :param test_single_stock: 是否只测试单只股票（用于计算时间）
        :param test_mode: 是否为测试模式（只运行前几个股票和参数组合）
        """
        # 获取所有股票文件
        stock_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        print(f"找到 {len(stock_files)} 个股票文件")
        
        # 生成参数组合
        if params_list is None:
            params_list = self.generate_params()
        print(f"生成 {len(params_list)} 组参数组合")
        
        # 测试模式：只运行前5个参数组合和前10个股票
        if test_mode:
            print("测试模式：只运行前5个参数组合和前10个股票")
            params_list = params_list[:5]
            stock_files = stock_files[:10]
        
        # 如果是测试单只股票
        if test_single_stock and stock_files:
            print("测试单只股票的回测时间...")
            import time
            start_time = time.time()
            
            # 只取第一只股票
            test_stock = stock_files[0]
            print(f"测试股票: {test_stock}")
            
            # 遍历所有参数组合
            for param_idx, params in enumerate(params_list):
                print(f"测试参数组合 {param_idx+1}/{len(params_list)}: {params}")
                
                # 处理测试股票
                stock_code = test_stock.split('#')[-1].replace('.txt', '')
                file_path = os.path.join(self.data_dir, test_stock)
                
                # 加载数据（使用缓存）
                if test_stock in self.data_cache:
                    df = self.data_cache[test_stock]
                else:
                    df = self.load_stock_data(file_path)
                    if df is not None:
                        self.data_cache[test_stock] = df
                
                if df is not None and len(df) > 60:
                    # 运行策略
                    result = self.run_strategy(df, stock_code, params)
                    self.results.append(result)
            
            end_time = time.time()
            single_stock_time = end_time - start_time
            print(f"\n单只股票回测时间: {single_stock_time:.2f} 秒")
            
            # 估计5400只股票的总时间
            total_time_seconds = single_stock_time * 5400
            total_hours = total_time_seconds / 3600
            print(f"估计5400只股票回测总时间: {total_hours:.2f} 小时")
            return self.results
        
        print(f"预计总计算量: {len(params_list) * len(stock_files)} 次策略运行")
        
        # 计算预计运行时间（假设每次策略运行平均需要0.1秒）
        avg_time_per_run = 0.1  # 秒
        total_seconds = len(params_list) * len(stock_files) * avg_time_per_run
        
        # 转换为小时:分钟:秒格式
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        
        #print(f"预计运行时间: {hours}小时{minutes}分钟{seconds}秒")
        print("开始运行回测...")
        
        # 遍历所有参数组合
        for param_idx, params in enumerate(params_list):
            print(f"\n测试参数组合 {param_idx+1}/{len(params_list)}: {params}")
            
            # 遍历所有股票
            for i, file_name in enumerate(stock_files):
                if i % 100 == 0:
                    print(f"处理进度: {i}/{len(stock_files)}")
                
                # 提取股票代码
                stock_code = file_name.split('#')[-1].replace('.txt', '')
                
                # 加载数据（使用缓存）
                file_path = os.path.join(self.data_dir, file_name)
                if file_name in self.data_cache:
                    df = self.data_cache[file_name]
                else:
                    df = self.load_stock_data(file_path)
                    if df is not None:
                        self.data_cache[file_name] = df
                
                if df is not None and len(df) > 60:  # 确保数据足够长
                    # 运行策略
                    result = self.run_strategy(df, stock_code, params)
                    self.results.append(result)
        
        # 保存结果
        if self.results_file:
            # 展开结果，保存为DataFrame
            expanded_results = []
            for result in self.results:
                expanded_result = {
                    'stock_code': result['stock_code'],
                    'num_trades': result['num_trades'],
                    'total_profit': result['total_profit'],
                    'avg_profit_per_trade': result['avg_profit_per_trade'],
                    'success_rate': result['success_rate'],
                    'win_rate': result['win_rate'],
                    'max_drawdown': result['max_drawdown'],
                    'cumulative_return': result['cumulative_return'],
                    'annual_return': result['annual_return'],
                }
                # 添加参数
                expanded_result.update(result['params'])
                expanded_results.append(expanded_result)
            
            results_df = pd.DataFrame(expanded_results)
            results_df.to_csv(self.results_file, index=False, encoding='utf-8-sig')
            print(f"回测结果已保存到 {self.results_file}")
        
        # 打印总体结果
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """
        打印回测总结
        """
        if not self.results:
            print("没有回测结果")
            return
        
        results_df = pd.DataFrame(self.results)
        
        # 计算总体统计
        total_trades = results_df['num_trades'].sum()
        total_profit = results_df['total_profit'].sum()
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        
        # 计算有交易的股票数量
        stocks_with_trades = len(results_df[results_df['num_trades'] > 0])
        
        # 计算盈利股票比例
        profitable_stocks = len(results_df[results_df['total_profit'] > 0])
        profitable_stocks_ratio = profitable_stocks / len(results_df) if len(results_df) > 0 else 0
        
        # 计算平均成功率和年化收益率
        avg_success_rate = results_df['success_rate'].mean()
        avg_annual_return = results_df['annual_return'].mean()
        
        print("\n回测总结:")
        print(f"总股票数: {len(results_df)}")
        print(f"有交易的股票数: {stocks_with_trades}")
        print(f"总交易次数: {total_trades}")
        print(f"总盈利: {total_profit:.4f}")
        print(f"平均每笔交易盈利: {avg_profit_per_trade:.4f}")
        print(f"盈利股票比例: {profitable_stocks_ratio:.2%}")
        print(f"平均成功率: {avg_success_rate:.2%}")
        print(f"平均年化收益率: {avg_annual_return:.4f}")
        
        # 按参数组合分组，计算每组的平均年化收益率和成功率
        if 'params' in results_df.columns:
            # 提取参数组合作为分组键
            def get_param_key(params):
                return tuple(sorted(params.items()))
            
            results_df['param_key'] = results_df['params'].apply(get_param_key)
            
            # 分组计算
            grouped = results_df.groupby('param_key').agg({
                'annual_return': 'mean',
                'success_rate': 'mean',
                'num_trades': 'sum'
            }).reset_index()
            
            # 排序，找出年化收益率和成功率最高的五组参数
            top_params = grouped.sort_values(['annual_return', 'success_rate'], ascending=False).head(5)
            
            print("\n年化收益率和成功率最高的五组参数:")
            for i, row in top_params.iterrows():
                # 解析参数
                param_dict = dict(row['param_key'])
                print(f"\n第{i+1}组参数:")
                print(f"年化收益率: {row['annual_return']:.4f}")
                print(f"成功率: {row['success_rate']:.2%}")
                print(f"交易次数: {row['num_trades']}")
                print(f"参数: {param_dict}")
                
                # 找出该参数组合下表现最好的前100只股票和最差的100只股票
                param_results = results_df[results_df['param_key'] == row['param_key']]
                
                if len(param_results) > 0:
                    # 表现最好的前100只股票
                    top_stocks = param_results.sort_values('total_profit', ascending=False).head(100)
                    print("\n表现最好的前100只股票:")
                    for _, stock_row in top_stocks.iterrows():
                        print(f"{stock_row['stock_code']}: 总盈利 {stock_row['total_profit']:.4f}, 成功率 {stock_row['success_rate']:.2%}, 年化收益率 {stock_row['annual_return']:.4f}")
                    
                    # 表现最差的100只股票
                    bottom_stocks = param_results.sort_values('total_profit', ascending=True).head(100)
                    print("\n表现最差的100只股票:")
                    for _, stock_row in bottom_stocks.iterrows():
                        print(f"{stock_row['stock_code']}: 总盈利 {stock_row['total_profit']:.4f}, 成功率 {stock_row['success_rate']:.2%}, 年化收益率 {stock_row['annual_return']:.4f}")
        
        # 打印表现最好的前10只股票
        print("\n表现最好的前10只股票:")
        top_stocks = results_df.sort_values('total_profit', ascending=False).head(10)
        for _, row in top_stocks.iterrows():
            print(f"{row['stock_code']}: 总盈利 {row['total_profit']:.4f}, 交易次数 {row['num_trades']}, 成功率 {row['success_rate']:.2%}, 年化收益率 {row['annual_return']:.4f}")
        
        # 打印年化收益率最高的前10只股票
        print("\n年化收益率最高的前10只股票:")
        top_annual_stocks = results_df[results_df['num_trades'] > 5].sort_values('annual_return', ascending=False).head(10)
        for _, row in top_annual_stocks.iterrows():
            print(f"{row['stock_code']}: 年化收益率 {row['annual_return']:.4f}, 成功率 {row['success_rate']:.2%}, 交易次数 {row['num_trades']}")

if __name__ == "__main__":
    # 配置参数
    data_dir = "c:\\Users\\lidon\\Desktop\\Qstrategy\\data\\20260207\\"
    results_file = "c:\\Users\\lidon\\Desktop\\Qstrategy\\backtest_b1_results.csv"
    
    # 运行回测（测试模式）
    backtest = B1StrategyBacktest(data_dir, results_file)
    backtest.run(test_single_stock=False, test_mode=True)
