import os
import pandas as pd
import numpy as np

data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

def calculate_indicators(df):
    close = df['CLOSE']
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA30'] = close.rolling(30).mean()
    df['MA60'] = close.rolling(60).mean()
    
    low_9 = df['LOW'].rolling(9).min()
    high_9 = df['HIGH'].rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-6) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

def check_data_anomaly(df):
    anomaly_dates = set()
    if len(df) < 2:
        return anomaly_dates
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row['OPEN']) or pd.isna(row['HIGH']) or pd.isna(row['LOW']) or pd.isna(row['CLOSE']):
            anomaly_dates.add(df.index[i])
            continue
        if row['HIGH'] == row['LOW'] and row['LOW'] == row['CLOSE']:
            anomaly_dates.add(df.index[i])
            continue
        if row['VOLUME'] <= 0:
            anomaly_dates.add(df.index[i])
            continue
    return anomaly_dates

# 加载数据
print("加载数据...")
files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
stock_data = {}
daily_signals = {}
daily_scores = {}
loaded_count = 0

for file in files:
    stock_code = file.replace('.txt', '')
    path = os.path.join(data_dir, file)
    
    try:
        df = pd.read_csv(path, sep='\t', encoding='utf-8')
        df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期')
        df = df.sort_index()
        
        anomaly_dates = check_data_anomaly(df)
        if len(anomaly_dates) > len(df) * 0.1:
            continue
        
        df = calculate_indicators(df)
        stock_data[stock_code] = df
        loaded_count += 1
        
        for i in range(2, len(df)):
            row = df.iloc[i]
            if pd.isna(row['J']):
                continue
            if row['J'] >= -5:
                continue
            date = df.index[i]
            if date not in daily_signals:
                daily_signals[date] = []
                daily_scores[date] = []
            daily_signals[date].append(stock_code)
            daily_scores[date].append((stock_code, 1, {'J<-5': 1}))
        
        if loaded_count >= 500:
            print(f"已达到测试数量 500，停止加载")
            break
    
    except Exception as e:
        continue

print(f"成功加载 {loaded_count} 只股票")

# 收集所有日期
all_dates = []
for df in stock_data.values():
    all_dates.extend(df.index.tolist())
all_dates = sorted(set(all_dates))

print(f"总交易日: {len(all_dates)}")
print(f"有信号的天数: {len(daily_signals)}")
print(f"总信号数: {sum(len(v) for v in daily_signals.values())}")

# 模拟回测逻辑
initial_capital = 1000000
fee_rate = 0.0003
slippage = 0.001
cash = float(initial_capital)
positions = []
pause_trading_days = 0
trade_count = 0

date_to_idx = {date: idx for idx, date in enumerate(all_dates)}

debug_info = {
    'skip_no_signal': 0,
    'skip_paused': 0,
    'skip_no_candidates': 0,
    'skip_open_zero': 0,
    'skip_no_shares': 0,
    'skip_cash_insufficient': 0,
    'success_buy': 0
}

for i, current_date in enumerate(all_dates):
    if pause_trading_days > 0:
        pause_trading_days -= 1
        debug_info['skip_paused'] += 1
    
    current_idx = date_to_idx[current_date]
    
    # 处理持仓
    new_positions = []
    for pos in positions:
        stock = pos["stock"]
        df = stock_data[stock]
        
        if current_date not in df.index:
            new_positions.append(pos)
            continue
        
        row = df.loc[current_date]
        
        if pd.isna(row["CLOSE"]) or row["CLOSE"] <= 0:
            new_positions.append(pos)
            continue
        
        # T+1 卖出限制
        entry_idx = pos["entry_idx"]
        if current_idx < entry_idx + 1:
            new_positions.append(pos)
            continue
        
        # 止损检查
        if row["LOW"] <= pos["stop_price"]:
            # 卖出
            trade_count += 1
            cash += pos['shares'] * row["CLOSE"] * (1 - fee_rate - slippage)
            continue
        
        # 止盈检查 (J>100)
        if not pos.get('j_take_profit_done', False):
            if not pd.isna(row.get('J')) and row['J'] > 100:
                trade_count += 1
                cash += pos['shares'] * row["CLOSE"] * (1 - fee_rate - slippage)
                continue
        
        # 涨幅>8%止盈
        change_pct = (row["CLOSE"] - pos["entry_price"]) / pos["entry_price"] * 100
        if not pos.get('gain_take_profit_done', False):
            if change_pct > 8:
                trade_count += 1
                cash += pos['shares'] * row["CLOSE"] * (1 - fee_rate - slippage)
                continue
        
        new_positions.append(pos)
    
    positions = new_positions
    
    # 买入逻辑
    if current_date in daily_signals and pause_trading_days == 0:
        candidates = daily_scores.get(current_date, [])
        
        if not candidates:
            debug_info['skip_no_candidates'] += 1
        else:
            candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
            top_stock = candidates_sorted[0][0]
            
            df = stock_data[top_stock]
            row = df.loc[current_date]
            open_price = row['OPEN']
            
            if open_price <= 0:
                debug_info['skip_open_zero'] += 1
            else:
                # 使用90%的资金
                invest_amount = cash * 0.9
                shares = int(invest_amount / open_price / 100) * 100
                
                if shares <= 0:
                    debug_info['skip_no_shares'] += 1
                else:
                    cost = shares * open_price * (1 + fee_rate + slippage)
                    if cost > cash:
                        debug_info['skip_cash_insufficient'] += 1
                    else:
                        cash -= cost
                        stop_price = open_price * 0.95
                        positions.append({
                            'stock': top_stock,
                            'shares': shares,
                            'entry_price': open_price,
                            'entry_idx': current_idx,
                            'stop_price': stop_price,
                            'j_take_profit_done': False,
                            'gain_take_profit_done': False
                        })
                        debug_info['success_buy'] += 1
    else:
        debug_info['skip_no_signal'] += 1

print("\n" + "=" * 50)
print("调试信息:")
print("=" * 50)
for k, v in debug_info.items():
    print(f"  {k}: {v}")
print(f"\n最终交易次数: {trade_count}")
print(f"最终现金: {cash:.2f}")
