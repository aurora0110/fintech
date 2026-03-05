import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"


def load_stock_data(max_stocks=5000):
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')][:max_stocks]
    stock_data = {}
    
    for file in files:
        stock_code = file.replace('.txt', '')
        
        if stock_code.startswith('BJ'):
            continue
        
        path = os.path.join(DATA_DIR, file)
        
        try:
            df = pd.read_csv(path, sep='\t', encoding='utf-8')
            df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            df = df.sort_index()
            
            if len(df) < 120:
                continue
            
            stock_data[stock_code] = df
        except:
            continue
    
    print(f"成功加载 {len(stock_data)} 只股票")
    return stock_data


def calculate_indicators(df):
    close = df['CLOSE']
    high = df['HIGH']
    low = df['LOW']
    
    for window in [5, 10, 20, 30, 60, 120]:
        df[f'MA{window}'] = close.rolling(window).mean()
    
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-6) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    df['知行短期趋势线'] = close.ewm(span=5, adjust=False).mean()
    df['知行多空线'] = close.ewm(span=10, adjust=False).mean()
    
    return df


def run_strategy_fixed(stock_data, j_thresh, struct_name, pos_name, ema_name, holding_days, use_stop_loss=False):
    """固定持有期回测
    use_stop_loss=False: 无止损，次日开盘买入，固定持有期后次日开盘卖出
    use_stop_loss=True: 有止损，次日开盘买入，固定持有期后次日开盘卖出，或触及止损则次日开盘卖出
    """
    all_returns = []
    stop_loss_count = 0
    
    for stock_code, df in stock_data.items():
        df = calculate_indicators(df)
        
        if len(df) < 122:
            continue
        
        close = df['CLOSE'].values
        open_prices = df['OPEN'].values
        low_prices = df['LOW'].values
        j_vals = df['J'].values
        ma5 = df['MA5'].values
        ma20 = df['MA20'].values
        ma30 = df['MA30'].values
        ma60 = df['MA60'].values
        ema5 = df['知行短期趋势线'].values
        ema10 = df['知行多空线'].values
        
        for idx in range(121, len(df) - holding_days - 1):
            j = j_vals[idx]
            if pd.isna(j) or j >= -j_thresh:
                continue
            
            ma5_v, ma20_v, ma30_v = ma5[idx], ma20[idx], ma30[idx]
            if pd.isna(ma5_v) or pd.isna(ma20_v) or pd.isna(ma30_v):
                continue
            
            if struct_name == 'G底部反转':
                ma60_v = ma60[idx]
                if pd.isna(ma60_v):
                    continue
                if not (ma5_v > ma20_v and ma20_v < ma60_v):
                    continue
            
            if pos_name == '站MA20':
                if pd.isna(close[idx]) or pd.isna(ma20[idx]):
                    continue
                if not (close[idx] > ma20[idx]):
                    continue
            
            if ema_name == '有EMA':
                ema5_v, ema10_v = ema5[idx], ema10[idx]
                if pd.isna(ema5_v) or pd.isna(ema10_v):
                    continue
                if not (ema5_v > ema10_v):
                    continue
            
            signal_low = low_prices[idx]
            entry_open = open_prices[idx + 1]
            
            if pd.isna(entry_open) or entry_open <= 0:
                continue
            
            if use_stop_loss:
                buy_day_low = low_prices[idx + 1]
                stop_loss_price = min(signal_low, buy_day_low) * 0.95
                
                stopped = False
                for day in range(1, holding_days + 1):
                    if idx + day + 1 >= len(df):
                        break
                    
                    day_close = close[idx + day]
                    if pd.isna(day_close):
                        continue
                    
                    if day_close < stop_loss_price:
                        exit_open = open_prices[idx + day + 1]
                        if pd.isna(exit_open) or exit_open <= 0:
                            continue
                        ret = (exit_open - entry_open) / entry_open * 100
                        all_returns.append(ret)
                        stop_loss_count += 1
                        stopped = True
                        break
                
                if stopped:
                    continue
            
            exit_open = open_prices[idx + holding_days + 1]
            if pd.isna(exit_open) or exit_open <= 0:
                continue
            
            ret = (exit_open - entry_open) / entry_open * 100
            all_returns.append(ret)
    
    return all_returns, stop_loss_count


def main():
    print("="*80)
    print("固定持有期策略：开盘价买入 + 止损对比")
    print("="*80)
    print("""
入场: 信号次日开盘价买入
出场: 固定持有期后次日开盘卖出，或触及止损则次日开盘卖出
止损价: min(信号日最低价, 买入日最低价) * 0.95
""")
    
    print("加载股票数据...")
    stock_data = load_stock_data(max_stocks=5000)
    
    holdings = [3, 5, 7]
    
    print("\n" + "="*80)
    print("【J<-5 + G底部反转 + 站MA20】")
    print("="*80)
    
    print(f"\n{'持有期':<8} {'无止损收益':<12} {'有止损收益':<12} {'止损次数':<10} {'收益变化'}")
    print("-"*60)
    
    results = []
    
    for holding in holdings:
        returns_no_sl, _ = run_strategy_fixed(stock_data, 5, 'G底部反转', '站MA20', '无', holding, use_stop_loss=False)
        returns_with_sl, sl_count = run_strategy_fixed(stock_data, 5, 'G底部反转', '站MA20', '无', holding, use_stop_loss=True)
        
        if len(returns_no_sl) >= 50 and len(returns_with_sl) >= 50:
            avg_no_sl = np.mean(returns_no_sl)
            avg_with_sl = np.mean(returns_with_sl)
            change = avg_with_sl - avg_no_sl
            
            print(f"{holding}日{'':<4} {avg_no_sl:.2f}%{'':<7} {avg_with_sl:.2f}%{'':<7} {sl_count}{'':<7} {change:+.2f}%")
            
            results.append({
                'holding': holding,
                'no_sl': avg_no_sl,
                'with_sl': avg_with_sl,
                'sl_count': sl_count,
                'change': change,
                'win_rate_no_sl': (np.array(returns_no_sl) > 0).mean() * 100,
                'win_rate_with_sl': (np.array(returns_with_sl) > 0).mean() * 100,
                'count_no_sl': len(returns_no_sl),
                'count_with_sl': len(returns_with_sl),
            })
    
    print("\n" + "="*80)
    print("【J<-5 + G底部反转 + 有EMA】")
    print("="*80)
    
    print(f"\n{'持有期':<8} {'无止损收益':<12} {'有止损收益':<12} {'止损次数':<10} {'收益变化'}")
    print("-"*60)
    
    results2 = []
    
    for holding in holdings:
        returns_no_sl, _ = run_strategy_fixed(stock_data, 5, 'G底部反转', '无', '有EMA', holding, use_stop_loss=False)
        returns_with_sl, sl_count = run_strategy_fixed(stock_data, 5, 'G底部反转', '无', '有EMA', holding, use_stop_loss=True)
        
        if len(returns_no_sl) >= 50 and len(returns_with_sl) >= 50:
            avg_no_sl = np.mean(returns_no_sl)
            avg_with_sl = np.mean(returns_with_sl)
            change = avg_with_sl - avg_no_sl
            
            print(f"{holding}日{'':<4} {avg_no_sl:.2f}%{'':<7} {avg_with_sl:.2f}%{'':<7} {sl_count}{'':<7} {change:+.2f}%")
            
            results2.append({
                'holding': holding,
                'no_sl': avg_no_sl,
                'with_sl': avg_with_sl,
                'sl_count': sl_count,
                'change': change,
                'win_rate_no_sl': (np.array(returns_no_sl) > 0).mean() * 100,
                'win_rate_with_sl': (np.array(returns_with_sl) > 0).mean() * 100,
                'count_no_sl': len(returns_no_sl),
                'count_with_sl': len(returns_with_sl),
            })
    
    print("\n" + "="*80)
    print("【详细对比表】")
    print("="*80)
    
    print("\n--- 策略1: J<-5 + G底部反转 + 站MA20 ---")
    print(f"{'持有期':<8} {'无止损':<12} {'有止损':<12} {'止损触发':<10} {'胜率变化':<12} {'结论'}")
    print("-"*75)
    
    for r in results:
        wr_change = r['win_rate_with_sl'] - r['win_rate_no_sl']
        if r['change'] > 0:
            conlusion = "✓ 止损有效"
        elif r['change'] > -0.3:
            conlusion = "≈ 持平"
        else:
            conlusion = "✗ 止损有害"
        
        print(f"{r['holding']}日{'':<4} {r['no_sl']:.2f}%{'':<7} {r['with_sl']:.2f}%{'':<7} {r['sl_count']}次{'':<5} {wr_change:+.1f}%{'':<7} {conlusion}")
    
    print("\n--- 策略2: J<-5 + G底部反转 + 有EMA ---")
    print(f"{'持有期':<8} {'无止损':<12} {'有止损':<12} {'止损触发':<10} {'胜率变化':<12} {'结论'}")
    print("-"*75)
    
    for r in results2:
        wr_change = r['win_rate_with_sl'] - r['win_rate_no_sl']
        if r['change'] > 0:
            conlusion = "✓ 止损有效"
        elif r['change'] > -0.3:
            conlusion = "≈ 持平"
        else:
            conlusion = "✗ 止损有害"
        
        print(f"{r['holding']}日{'':<4} {r['no_sl']:.2f}%{'':<7} {r['with_sl']:.2f}%{'':<7} {r['sl_count']}次{'':<5} {wr_change:+.1f}%{'':<7} {conlusion}")
    
    print("\n" + "="*80)
    print("【总结】")
    print("="*80)


if __name__ == "__main__":
    main()
