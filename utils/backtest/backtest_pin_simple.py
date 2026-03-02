"""
PIN策略简化版回测
- 买入条件：PIN(True) + 知行短期趋势线 > 多空线 + 收盘价 > 多空线
- 买入：第二天开盘价
- 卖出：第三天收盘价
- 全仓
- 不止损
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_pin_conditions(df):
    """计算PIN条件"""
    N1, N2 = 3, 21
    
    llv_l_n1 = df['最低'].rolling(window=N1, min_periods=1).min()
    hhv_c_n1 = df['收盘'].rolling(window=N1, min_periods=1).max()
    df['短期'] = (df['收盘'] - llv_l_n1) / (hhv_c_n1 - llv_l_n1 + 0.0001) * 100
    
    llv_l_n2 = df['最低'].rolling(window=N2, min_periods=1).min()
    hhv_l_n2 = df['收盘'].rolling(window=N2, min_periods=1).max()
    df['长期'] = (df['收盘'] - llv_l_n2) / (hhv_l_n2 - llv_l_n2 + 0.0001) * 100
    
    df['PIN信号'] = (df['短期'] <= 30) & (df['长期'] >= 85)
    
    return df


def calculate_trend(df):
    """计算趋势线"""
    df['知行短期趋势线'] = df['收盘'].ewm(span=10, adjust=False).mean()
    df['知行短期趋势线'] = df['知行短期趋势线'].ewm(span=10, adjust=False).mean()
    
    df['MA14'] = df['收盘'].rolling(window=14).mean()
    df['MA28'] = df['收盘'].rolling(window=28).mean()
    df['MA57'] = df['收盘'].rolling(window=57).mean()
    df['MA114'] = df['收盘'].rolling(window=114).mean()
    df['知行多空线'] = (df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']) / 4
    
    return df


def load_stock_data(file_path):
    """加载股票数据"""
    try:
        df = pd.read_csv(file_path, sep='\s+', encoding='utf-8')
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        return df
    except:
        return None


def run_backtest(data_dir, results_file, test_mode=False):
    """运行回测"""
    stock_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    print(f"找到 {len(stock_files)} 个股票文件")
    
    if test_mode:
        stock_files = stock_files[:10]
    
    all_trades = []
    initial_capital = 1_000_000
    capital = initial_capital
    
    equity_curve = []
    dates_equity = []
    
    for file_name in tqdm(stock_files, desc="回测中"):
        file_path = os.path.join(data_dir, file_name)
        df = load_stock_data(file_path)
        
        if df is None or len(df) < 50:
            continue
        
        df = calculate_trend(df)
        df = calculate_pin_conditions(df)
        
        for i in range(len(df) - 3):
            current = df.iloc[i]
            next_day = df.iloc[i + 1]
            third_day = df.iloc[i + 2]
            
            trend_line = current.get('知行短期趋势线', 0)
            duokongxian = current.get('知行多空线', 0)
            close = current['收盘']
            pin_signal = current.get('PIN信号', False)
            
            buy_condition = (
                pin_signal and 
                not pd.isna(trend_line) and 
                not pd.isna(duokongxian) and
                trend_line > duokongxian and 
                close > duokongxian
            )
            
            if buy_condition:
                entry_price = next_day['开盘']
                if entry_price <= 0:
                    continue
                
                exit_price = third_day['收盘']
                if exit_price <= 0:
                    continue
                
                profit_pct = (exit_price - entry_price) / entry_price * 100
                profit = capital * profit_pct / 100
                
                capital += profit
                
                all_trades.append({
                    'stock': file_name.replace('.txt', ''),
                    'entry_date': next_day['日期'],
                    'exit_date': third_day['日期'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                })
        
        equity_curve.append(capital)
        if len(df) > 0:
            dates_equity.append(df.iloc[-1]['日期'])
    
    print(f"\n总交易次数: {len(all_trades)}")
    print(f"最终资金: {capital:.2f}")
    print(f"总收益率: {(capital - initial_capital) / initial_capital * 100:.2f}%")
    
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        wins = len(trades_df[trades_df['profit_pct'] > 0])
        success_rate = wins / len(trades_df) * 100
        avg_profit = trades_df['profit_pct'].mean()
        
        print(f"成功率: {success_rate:.2f}%")
        print(f"平均收益率: {avg_profit:.2f}%")
        
        trades_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"交易记录已保存到: {results_file}")
    
    return all_trades, equity_curve, dates_equity


if __name__ == "__main__":
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
    results_file = "/Users/lidongyang/Desktop/Qstrategy/backtest_pin_results.csv"
    
    run_backtest(data_dir, results_file, test_mode=False)
