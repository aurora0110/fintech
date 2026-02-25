"""
综合交易策略回测工具

功能说明：
1. 实现基于多技术指标的综合交易策略回测
2. 支持多参数组合测试，包括时间止损、涨幅要求、停止交易天数和仓位管理
3. 计算详细的绩效指标，如成功率、年化收益率、最大回撤等
4. 支持批量股票回测和参数组合分析
5. 实现复杂的交易规则，包括多种止损和止盈条件

交易规则：
- 买入条件：J<-5，DIF>0，趋势线金叉后
- 止损条件：
  1. 收盘价小于买入当天和前一天K线最低价的最小值
  2. a天未上涨b%
  3. 连续两天收盘价小于知行多空线
  4. 收益率达到5%以上且连续两天收盘价小于前一天最低价
  5. 出现60天最大成交量阴线
  6. 特殊K线形态
- 止盈条件：
  1. J值>100，卖出1/d仓位
  2. 单日或两日涨幅>8%，卖出1/d仓位
  3. 偏离趋势线>10%，卖出1/d仓位
- 锁仓条件：连续盈利后仓位小于1/5时锁仓
- 锁仓卖出：收盘价跌破趋势线2天后

参数说明：
- a: 时间止损天数 [3,4,5]
- b: 最小涨幅要求 [%] [1,2,3,4]
- c: 止盈后停止交易天数 [1,2,3,4,5]
- d: 止盈仓位比例分母 [3,4,5,6,7,8,9,10]
"""
import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.append('/Users/lidongyang/Desktop/Qstrategy')
from utils.technical_indicators import calculate_trend, calculate_kdj

class ComprehensiveBacktest:
    def __init__(self, data_dir, initial_capital=1000000):
        """
        初始化回测器
        :param data_dir: 数据目录路径
        :param initial_capital: 初始资金
        """
        self.data_dir = data_dir
        self.initial_capital = initial_capital
    
    def load_stock_data(self, file_path):
        """
        加载单个股票数据
        :param file_path: 文件路径
        :return: 股票数据DataFrame
        """
        try:
            # 读取文件（空格分隔）
            with open(file_path, 'r', encoding='utf-8') as f:
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
    
    def calculate_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        """
        计算MACD指标
        :param df: 股票数据DataFrame
        :param fast_period: 快速EMA周期
        :param slow_period: 慢速EMA周期
        :param signal_period: 信号EMA周期
        :return: 新增MACD指标列的DataFrame
        """
        # 计算EMA
        df['EMA12'] = df['收盘'].ewm(span=fast_period, adjust=False).mean()
        df['EMA26'] = df['收盘'].ewm(span=slow_period, adjust=False).mean()
        
        # 计算DIF和DEA
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['DEA'] = df['DIF'].ewm(span=signal_period, adjust=False).mean()
        
        # 计算MACD柱状图
        df['MACD'] = 2 * (df['DIF'] - df['DEA'])
        
        return df
    
    def run_strategy(self, df, stock_code, params):
        """
        运行回测策略
        :param df: 股票数据DataFrame
        :param stock_code: 股票代码
        :param params: 策略参数 (a, b, c, d)
        :return: 回测结果字典
        """
        a, b, c, d = params
        
        # 计算技术指标
        df = calculate_trend(df)
        df = calculate_kdj(df)
        df = self.calculate_macd(df)
        
        # 初始化变量
        position = 0  # 当前仓位（0-1之间）
        capital = self.initial_capital
        entry_price = 0  # 买入价格（开盘价）
        entry_date = None
        entry_low = 0  # 买入当根K线的最低价
        prev_low = 0  # 买入前一天的最低价
        exit_price = 0
        exit_date = None
        trades = []
        buy_records = []  # 记录每次买入的价格和仓位
        can_buy = False
        days_since_buy = 0
        profit_days = 0
        locked_position = 0
        last_buy_date = None
        stop_trading_days = 0
        
        # 标记趋势线金叉死叉
        df['趋势线金叉'] = (df['知行短期趋势线'] > df['知行多空线']) & (df['知行短期趋势线'].shift(1) <= df['知行多空线'].shift(1))
        df['趋势线死叉'] = (df['知行短期趋势线'] < df['知行多空线']) & (df['知行短期趋势线'].shift(1) >= df['知行多空线'].shift(1))
        
        # 计算最近60天最大成交量
        df['60天最大成交量'] = df['成交量'].rolling(window=60, min_periods=1).max()
        
        # 计算最近30天成交量
        df['30天成交量'] = df['成交量'].rolling(window=30, min_periods=1).mean()
        
        for i in range(len(df)):
            # 检查停止交易天数
            if stop_trading_days > 0:
                stop_trading_days -= 1
            
            # 检查趋势线金叉
            if df['趋势线金叉'].iloc[i]:
                can_buy = True
            
            # 检查趋势线死叉
            if df['趋势线死叉'].iloc[i]:
                can_buy = False
                # 死叉时全仓平仓
                if position > 0:
                    exit_price = df['收盘'].iloc[i]
                    exit_date = df['日期'].iloc[i]
                    
                    # 计算盈利
                    total_profit = (exit_price - entry_price) / entry_price * position * capital
                    capital += total_profit
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': (exit_price - entry_price) / entry_price,
                        'position': position,
                        'type': '趋势线死叉卖出',
                        'capital': capital
                    })
                    
                    # 重置仓位
                    position = 0
                    locked_position = 0
                    buy_records = []
            
            # 检查买入条件：J<-5，DIF>0，趋势线金叉后，且不在停止交易期
            if can_buy and position == 0 and stop_trading_days == 0 and i > 0:
                if df['J'].iloc[i-1] < -5 and df['DIF'].iloc[i-1] > 0:
                    # 用当天的开盘价买入
                    buy_price = df['开盘'].iloc[i]
                    if buy_price > 0:
                        # 全仓买入
                        position = 1.0
                        entry_price = buy_price
                        entry_date = df['日期'].iloc[i]
                        entry_low = df['最低'].iloc[i]  # 买入当根K线的最低价
                        prev_low = df['最低'].iloc[i-1]  # 买入前一天的最低价
                        last_buy_date = entry_date
                        days_since_buy = 0
                        profit_days = 0
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': None,
                            'entry_price': entry_price,
                            'exit_price': None,
                            'profit': 0,
                            'position': position,
                            'type': '买入',
                            'capital': capital
                        })
            
            # 检查持仓
            if position > 0:
                days_since_buy += 1
                current_close = df['收盘'].iloc[i]
                current_date = df['日期'].iloc[i]
                
                # 计算当前收益率
                current_return = (current_close - entry_price) / entry_price
                
                # 检查止损条件1：收盘价小于买入当天K线和前一天K线的最小值
                if current_close < min(entry_low, prev_low):
                    exit_price = current_close
                    exit_date = current_date
                    
                    # 计算盈利
                    total_profit = current_return * position * capital
                    capital += total_profit
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': current_return,
                        'position': position,
                        'type': '止损卖出1',
                        'capital': capital
                    })
                    
                    # 重置仓位
                    position = 0
                    locked_position = 0
                    buy_records = []
                    continue
                
                # 检查止损条件2：a天未上涨b%
                if days_since_buy == a:
                    price_change = (current_close - entry_price) / entry_price * 100
                    if price_change < b:
                        exit_price = current_close
                        exit_date = current_date
                        
                        # 计算盈利
                        total_profit = current_return * position * capital
                        capital += total_profit
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': current_return,
                            'position': position,
                            'type': f'止损卖出2({a}天未涨{b}%)',
                            'capital': capital
                        })
                        
                        # 重置仓位
                        position = 0
                        locked_position = 0
                        buy_records = []
                        continue
                
                # 检查止损条件3：连续两天收盘价小于知行多空线
                if i >= 2:
                    if df['收盘'].iloc[i] < df['知行多空线'].iloc[i] and df['收盘'].iloc[i-1] < df['知行多空线'].iloc[i-1]:
                        exit_price = current_close
                        exit_date = current_date
                        
                        # 计算盈利
                        total_profit = current_return * position * capital
                        capital += total_profit
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': current_return,
                            'position': position,
                            'type': '止损卖出3(连续两天收盘价小于多空线)',
                            'capital': capital
                        })
                        
                        # 重置仓位
                        position = 0
                        locked_position = 0
                        buy_records = []
                        continue
                
                # 检查止损条件4：买入收益率达到5%以上且连续两天收盘价小于前一天的最低价
                if current_return > 0.05 and i >= 2:
                    if df['收盘'].iloc[i] < df['最低'].iloc[i-1] and df['收盘'].iloc[i-1] < df['最低'].iloc[i-2]:
                        exit_price = current_close
                        exit_date = current_date
                        
                        # 计算盈利
                        total_profit = current_return * position * capital
                        capital += total_profit
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': current_return,
                            'position': position,
                            'type': '止损卖出4(盈利5%以上且连续两天收盘价小于前一天最低价)',
                            'capital': capital
                        })
                        
                        # 重置仓位
                        position = 0
                        locked_position = 0
                        buy_records = []
                        continue
                
                # 检查止损条件5：出现最近60个交易日交易量最大的阴线
                if current_close < df['开盘'].iloc[i] and df['成交量'].iloc[i] == df['60天最大成交量'].iloc[i]:
                    exit_price = current_close
                    exit_date = current_date
                    
                    # 计算盈利
                    total_profit = current_return * position * capital
                    capital += total_profit
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': current_return,
                        'position': position,
                        'type': '止损卖出5(60天最大成交量阴线)',
                        'capital': capital
                    })
                    
                    # 重置仓位
                    position = 0
                    locked_position = 0
                    buy_records = []
                    continue
                
                # 检查止损条件6：前一日收盘价小于当日收盘价小于当日开盘价且当日成交量大于最近30日内成交量
                if i >= 1:
                    if df['收盘'].iloc[i-1] < current_close < df['开盘'].iloc[i] and df['成交量'].iloc[i] > df['30天成交量'].iloc[i]:
                        exit_price = current_close
                        exit_date = current_date
                        
                        # 计算盈利
                        total_profit = current_return * position * capital
                        capital += total_profit
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': current_return,
                            'position': position,
                            'type': '止损卖出6(特殊K线形态)',
                            'capital': capital
                        })
                        
                        # 重置仓位
                        position = 0
                        locked_position = 0
                        buy_records = []
                        continue
                
                # 检查止盈条件1：J值>100
                if df['J'].iloc[i] > 100:
                    # 卖出1/d总仓位
                    sell_position = min(1/d, position)
                    if sell_position > 0:
                        exit_price = current_close
                        exit_date = current_date
                        
                        # 计算盈利
                        profit = current_return * sell_position * capital
                        capital += profit
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': current_return,
                            'position': sell_position,
                            'type': '止盈卖出1(J>100)',
                            'capital': capital
                        })
                        
                        # 更新仓位
                        position -= sell_position
                        
                        # 检查是否需要锁仓
                        if position < 1/5:
                            locked_position = position
                            position = 0
                        
                        # 设置停止交易天数
                        stop_trading_days = c
                        continue
                
                # 检查止盈条件2：单日或两日总共上涨8%以上
                if i >= 1:
                    # 计算两日涨幅（避免除以零）
                    if i >= 2 and df['收盘'].iloc[i-2] > 0:
                        two_day_change = (current_close - df['收盘'].iloc[i-2]) / df['收盘'].iloc[i-2] * 100
                    else:
                        two_day_change = 0
                    # 计算单日涨幅（避免除以零）
                    if df['收盘'].iloc[i-1] > 0:
                        one_day_change = (current_close - df['收盘'].iloc[i-1]) / df['收盘'].iloc[i-1] * 100
                    else:
                        one_day_change = 0
                    if one_day_change > 8 or two_day_change > 8:
                        # 卖出1/d总仓位
                        sell_position = min(1/d, position)
                        if sell_position > 0:
                            exit_price = current_close
                            exit_date = current_date
                            
                            # 计算盈利
                            profit = current_return * sell_position * capital
                            capital += profit
                            
                            trades.append({
                                'entry_date': entry_date,
                                'exit_date': exit_date,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'profit': current_return,
                                'position': sell_position,
                                'type': '止盈卖出2(涨幅>8%)',
                                'capital': capital
                            })
                            
                            # 更新仓位
                            position -= sell_position
                            
                            # 检查是否需要锁仓
                            if position < 1/5:
                                locked_position = position
                                position = 0
                            
                            # 设置停止交易天数
                            stop_trading_days = c
                            continue
                
                # 检查止盈条件3：偏离知行短期趋势线10%、15%、25%以上
                trend_line_price = df['知行短期趋势线'].iloc[i]
                deviation = (current_close - trend_line_price) / trend_line_price * 100
                if deviation > 10:
                    # 卖出1/d总仓位
                    sell_position = min(1/d, position)
                    if sell_position > 0:
                        exit_price = current_close
                        exit_date = current_date
                        
                        # 计算盈利
                        profit = current_return * sell_position * capital
                        capital += profit
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': current_return,
                            'position': sell_position,
                            'type': '止盈卖出3(偏离趋势线>10%)',
                            'capital': capital
                        })
                        
                        # 更新仓位
                        position -= sell_position
                        
                        # 检查是否需要锁仓
                        if position < 1/5:
                            locked_position = position
                            position = 0
                        
                        # 设置停止交易天数
                        stop_trading_days = c
                        continue
                
                # 检查锁仓卖出条件：收盘价跌破知行短期趋势线2天后
                if locked_position > 0 and i >= 2:
                    if df['收盘'].iloc[i-1] < df['知行短期趋势线'].iloc[i-1] and df['收盘'].iloc[i-2] < df['知行短期趋势线'].iloc[i-2]:
                        exit_price = current_close
                        exit_date = current_date
                        
                        # 计算盈利
                        profit = current_return * locked_position * capital
                        capital += profit
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': current_return,
                            'position': locked_position,
                            'type': '锁仓卖出(跌破趋势线2天)',
                            'capital': capital
                        })
                        
                        # 重置锁仓
                        locked_position = 0
                        continue
        
        # 计算回测结果
        if trades:
            # 过滤掉未完成的交易
            completed_trades = [trade for trade in trades if trade['exit_date'] is not None]
            num_trades = len(completed_trades)
            
            if num_trades > 0:
                total_profit = sum([trade['profit'] for trade in completed_trades])
                win_rate = len([trade for trade in completed_trades if trade['profit'] > 0]) / num_trades
                avg_profit = total_profit / num_trades
                
                # 计算最大回撤
                cumulative_profit = 1
                max_cumulative = 1
                max_drawdown = 0
                
                for trade in completed_trades:
                    cumulative_profit *= (1 + trade['profit'])
                    if cumulative_profit > max_cumulative:
                        max_cumulative = cumulative_profit
                    current_drawdown = (max_cumulative - cumulative_profit) / max_cumulative
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                
                # 计算年化收益率
                first_trade_date = completed_trades[0]['entry_date']
                last_trade_date = completed_trades[-1]['exit_date']
                total_days = (last_trade_date - first_trade_date).days
                final_capital = capital
                
                if total_days > 0:
                    total_return = (final_capital - self.initial_capital) / self.initial_capital
                    annual_return = (pow(1 + total_return, 365 / total_days)) - 1
                else:
                    annual_return = 0
                
                # 计算成功率（胜率）
                success_rate = win_rate
                
                return {
                    'stock_code': stock_code,
                    'params': params,
                    'num_trades': num_trades,
                    'total_profit': total_profit,
                    'avg_profit_per_trade': avg_profit,
                    'success_rate': success_rate,
                    'win_rate': win_rate,
                    'max_drawdown': max_drawdown,
                    'cumulative_return': cumulative_profit - 1,
                    'annual_return': annual_return,
                    'final_capital': final_capital,
                    'trades': completed_trades
                }
            else:
                return {
                    'stock_code': stock_code,
                    'params': params,
                    'num_trades': 0,
                    'total_profit': 0,
                    'avg_profit_per_trade': 0,
                    'success_rate': 0,
                    'win_rate': 0,
                    'max_drawdown': 0,
                    'cumulative_return': 0,
                    'annual_return': 0,
                    'final_capital': self.initial_capital,
                    'trades': []
                }
        else:
            return {
                'stock_code': stock_code,
                'params': params,
                'num_trades': 0,
                'total_profit': 0,
                'avg_profit_per_trade': 0,
                'success_rate': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'cumulative_return': 0,
                'annual_return': 0,
                'final_capital': self.initial_capital,
                'trades': []
            }
    
    def run(self, params_list):
        """
        运行整个回测
        :param params_list: 参数组合列表
        :return: 回测结果列表
        """
        # 获取所有股票文件
        stock_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        print(f"找到 {len(stock_files)} 个股票文件")
        
        all_results = []
        
        # 遍历所有参数组合
        for params in params_list:
            print(f"\n测试参数组合: a={params[0]}, b={params[1]}, c={params[2]}, d={params[3]}")
            
            # 遍历所有股票
            for i, file_name in enumerate(stock_files):
                if i % 100 == 0:
                    print(f"处理进度: {i}/{len(stock_files)}")
                
                # 提取股票代码
                stock_code = file_name.split('#')[-1].replace('.txt', '')
                
                # 加载数据
                file_path = os.path.join(self.data_dir, file_name)
                df = self.load_stock_data(file_path)
                
                if df is not None and len(df) > 60:  # 确保数据足够长
                    # 运行策略
                    result = self.run_strategy(df, stock_code, params)
                    all_results.append(result)
        
        return all_results

def generate_params():
    """
    生成所有参数组合
    :return: 参数组合列表
    """
    params = []
    # 测试全部参数组合
    for a in [3, 4, 5]:
        for b in [1, 2, 3, 4]:
            for c in [1, 2, 3, 4, 5]:
                for d in [3, 4, 5, 6, 7, 8, 9, 10]:
                    params.append((a, b, c, d))
    return params

def analyze_results(results):
    """
    分析回测结果
    :param results: 回测结果列表
    """
    if not results:
        print("没有回测结果")
        return
    
    results_df = pd.DataFrame(results)
    
    # 按参数分组分析
    params_groups = results_df.groupby('params')
    
    print("\n各参数组合策略表现:")
    print("-" * 120)
    print(f"{'参数组合':<20} {'股票数':<8} {'总交易数':<10} {'成功率':<10} {'年化收益率':<15} {'平均每笔盈利':<15} {'最终资金(万)':<15}")
    print("-" * 120)
    
    best_params = None
    best_annual_return = -float('inf')
    
    for params, group in params_groups:
        # 计算统计数据
        num_stocks = len(group)
        total_trades = group['num_trades'].sum()
        success_rate = group['success_rate'].mean()
        annual_return = group['annual_return'].mean()
        avg_profit_per_trade = group['avg_profit_per_trade'].mean()
        final_capital = group['final_capital'].mean() / 10000
        
        # 打印结果
        print(f"{str(params):<20} {num_stocks:<8} {total_trades:<10} {success_rate:<10.2%} {annual_return:<15.4f} {avg_profit_per_trade:<15.4f} {final_capital:<15.2f}")
        
        # 找出年化收益率最高的参数组合
        if annual_return > best_annual_return:
            best_annual_return = annual_return
            best_params = params
    
    print("-" * 120)
    print(f"\n最佳参数组合: {best_params}")
    print(f"年化收益率: {best_annual_return:.4f}")
    
    # 保存详细结果
    results_df.to_csv("/Users/lidongyang/Desktop/Qstrategy/comprehensive_results.csv", index=False, encoding='utf-8-sig')
    print("\n详细结果已保存到 comprehensive_results.csv")

if __name__ == "__main__":
    # 配置参数
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/20260207/normal/"
    
    # 生成参数组合
    params_list = generate_params()
    print(f"生成 {len(params_list)} 个参数组合")
    
    # 运行回测
    backtest = ComprehensiveBacktest(data_dir)
    results = backtest.run(params_list)
    
    # 分析结果
    analyze_results(results)
