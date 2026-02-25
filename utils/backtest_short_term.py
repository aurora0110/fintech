"""
短期交易策略回测工具

功能说明：
1. 实现基于KDJ指标和趋势线的短期交易策略
2. 支持时间止损条件设置，如3天未上涨1%就平仓等
3. 计算策略的成功率、年化收益率等绩效指标
4. 支持批量股票回测和结果分析

交易规则：
- 买入条件：J值<-5，且在趋势线金叉后
- 卖出条件：J值>100或涨幅>8%
- 止损条件：收盘价跌破买入当天K线最低价或前一天最低价
- 时间止损：可设置不同时间周期和涨幅要求
"""
import os
import pandas as pd
import numpy as np
from .technical_indicators import calculate_trend, calculate_kdj

class ShortTermBacktest:
    def __init__(self, data_dir, results_file=None):
        """
        初始化回测器
        :param data_dir: 数据目录路径
        :param results_file: 结果保存文件路径
        """
        self.data_dir = data_dir
        self.results_file = results_file
        self.results = []
    
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
    
    def run_strategy(self, df, stock_code, time_stop_conditions=None):
        """
        运行回测策略
        :param df: 股票数据DataFrame
        :param stock_code: 股票代码
        :param time_stop_conditions: 时间止损条件列表，格式为[(days, pct)]
        :return: 回测结果字典
        """
        # 计算技术指标
        df = calculate_trend(df)
        df = calculate_kdj(df)
        
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
        
        # 标记可以开始买入的点
        df['可以买入'] = False
        can_buy = False
        
        for i in range(len(df)):
            # 检查趋势线金叉
            if df['趋势线金叉'].iloc[i]:
                can_buy = True
                df.loc[i, '可以买入'] = True
            
            # 检查趋势线死叉
            if df['趋势线死叉'].iloc[i]:
                can_buy = False
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
                    
                    # 重置仓位
                    position = 0
                    buy_records = []
            
            # 检查买入条件：前一天J值<-5，今天可以买入，且仓位未满
            if can_buy and position < 1 and i > 0:
                if df['J'].iloc[i-1] < -5:
                    # 用当天的开盘价买入
                    buy_price = df['开盘'].iloc[i]
                    # 避免买入价格为0
                    if buy_price > 0:
                        # 计算买入仓位（每次买入25%，直到满仓）
                        buy_position = min(0.25, 1 - position)
                        entry_date = df['日期'].iloc[i]
                        entry_low = df['最低'].iloc[i]  # 买入当根K线的最低价
                        prev_low = df['最低'].iloc[i-1]  # 买入前一天的最低价
                        
                        # 记录买入
                        buy_records.append({
                            'date': entry_date,
                            'price': buy_price,
                            'position': buy_position,
                            'buy_index': i  # 记录买入的索引位置
                        })
                        
                        # 更新仓位
                        position += buy_position
            
            # 检查止损条件：买入后的收盘价跌破买入当根K线的最低价或者前一天的最低价
            if position > 0 and i > 0:
                current_close = df['收盘'].iloc[i]
                if current_close < entry_low or current_close < prev_low:
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
                                'type': '止损卖出'
                            })
                    
                    # 重置仓位
                    position = 0
                    buy_records = []
            
            # 检查卖出条件：J值>100或者涨幅>8%
            if position > 0 and i > 0:
                current_close = df['收盘'].iloc[i]
                # 计算平均买入价格
                avg_buy_price = sum([record['price'] * record['position'] for record in buy_records]) / position if position > 0 else 0
                
                if avg_buy_price > 0:
                    # 计算涨幅
                    change_pct = (current_close - avg_buy_price) / avg_buy_price * 100
                    
                    # 卖出条件：J值>100或者涨幅>8%
                    if df['J'].iloc[i] > 100 or change_pct > 8:
                        exit_price = current_close
                        exit_date = df['日期'].iloc[i]
                        
                        # 卖出25%的仓位
                        sell_position = min(0.25, position)
                        
                        # 按比例卖出每个买入记录
                        sold_position = 0
                        for record in buy_records.copy():
                            if sold_position >= sell_position:
                                break
                            
                            # 计算本次卖出的仓位
                            sell_amount = min(record['position'], sell_position - sold_position)
                            
                            # 计算盈利
                            profit = (exit_price - record['price']) / record['price'] * sell_amount
                            
                            # 记录交易
                            trades.append({
                                'entry_date': record['date'],
                                'exit_date': exit_date,
                                'entry_price': record['price'],
                                'exit_price': exit_price,
                                'profit': (exit_price - record['price']) / record['price'],
                                'position': sell_amount,
                                'type': '止盈卖出'
                            })
                            
                            # 更新记录
                            record['position'] -= sell_amount
                            sold_position += sell_amount
                            
                            # 如果该记录仓位为0，移除
                            if record['position'] <= 0:
                                buy_records.remove(record)
                        
                        # 更新仓位
                        position -= sell_position
            
            # 检查时间止损条件
            if position > 0 and time_stop_conditions:
                current_close = df['收盘'].iloc[i]
                
                # 检查每个买入记录
                for record in buy_records.copy():
                    buy_index = record['buy_index']
                    buy_price = record['price']
                    days_held = i - buy_index
                    
                    # 检查是否满足时间止损条件
                    for days, min_change in time_stop_conditions:
                        if days_held == days:
                            # 计算涨幅
                            change_pct = (current_close - buy_price) / buy_price * 100
                            
                            # 如果涨幅未达到要求，平仓
                            if change_pct < min_change:
                                exit_price = current_close
                                exit_date = df['日期'].iloc[i]
                                
                                # 记录交易
                                trades.append({
                                    'entry_date': record['date'],
                                    'exit_date': exit_date,
                                    'entry_price': buy_price,
                                    'exit_price': exit_price,
                                    'profit': (exit_price - buy_price) / buy_price,
                                    'position': record['position'],
                                    'type': f'时间止损卖出({days}天未涨{min_change}%)'
                                })
                                
                                # 更新仓位
                                position -= record['position']
                                buy_records.remove(record)
                                break
        
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
            # 计算总交易天数
            if len(trades) > 0:
                first_trade_date = trades[0]['entry_date']
                last_trade_date = trades[-1]['exit_date']
                total_days = (last_trade_date - first_trade_date).days
                if total_days > 0:
                    annual_return = (cumulative_profit ** (365 / total_days)) - 1
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
                'trades': trades
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
                'trades': []
            }
    
    def run(self):
        """
        运行整个回测
        """
        # 获取所有股票文件
        stock_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        print(f"找到 {len(stock_files)} 个股票文件")
        
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
                result = self.run_strategy(df, stock_code)
                self.results.append(result)
        
        # 保存结果
        if self.results_file:
            results_df = pd.DataFrame(self.results)
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
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/20260207/normal/"
    results_file = "/Users/lidongyang/Desktop/Qstrategy/backtest_results.csv"
    
    # 运行回测
    backtest = ShortTermBacktest(data_dir, results_file)
    backtest.run()
