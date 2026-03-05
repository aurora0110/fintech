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


def run_strategy_drawdown(stock_data, j_thresh, struct_name, pos_name, ema_name, holding_days, stop_loss_ratio=None):
    """回测并计算回撤统计
    stop_loss_ratio: 止损比例，如0.95表示止损价为min(信号日最低,买入日最低)*0.95
                     如果为None表示无止损
    """
    all_returns = []
    all_trade_drawdowns = []
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
            elif struct_name == 'B短期多头':
                if not (ma5_v > ma20_v > ma30_v):
                    continue
            elif struct_name == 'D趋势启动':
                if not (ma5_v > ma20_v and ma20_v > ma30_v):
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
            
            exit_ret = None
            trade_drawdowns = []
            
            if stop_loss_ratio is not None:
                buy_day_low = low_prices[idx + 1]
                stop_loss_price = min(signal_low, buy_day_low) * stop_loss_ratio
                
                stopped = False
                for day in range(1, holding_days + 1):
                    if idx + day + 1 >= len(df):
                        break
                    
                    day_close = close[idx + day]
                    if pd.isna(day_close):
                        continue
                    
                    drawdown = (day_close - entry_open) / entry_open * 100
                    trade_drawdowns.append(drawdown)
                    
                    if day_close < stop_loss_price:
                        exit_open = open_prices[idx + day + 1]
                        if pd.isna(exit_open) or exit_open <= 0:
                            continue
                        exit_ret = (exit_open - entry_open) / entry_open * 100
                        all_returns.append(exit_ret)
                        stop_loss_count += 1
                        stopped = True
                        break
                
                if stopped:
                    if trade_drawdowns:
                        min_dd = min(trade_drawdowns)
                        all_trade_drawdowns.append(min_dd)
                    continue
            
            exit_open = open_prices[idx + holding_days + 1]
            if pd.isna(exit_open) or exit_open <= 0:
                continue
            
            exit_ret = (exit_open - entry_open) / entry_open * 100
            all_returns.append(exit_ret)
            
            if stop_loss_ratio is not None:
                for day in range(1, holding_days + 1):
                    if idx + day + 1 >= len(df):
                        break
                    day_close = close[idx + day]
                    if pd.isna(day_close):
                        continue
                    drawdown = (day_close - entry_open) / entry_open * 100
                    trade_drawdowns.append(drawdown)
                
                if trade_drawdowns:
                    min_dd = min(trade_drawdowns)
                    all_trade_drawdowns.append(min_dd)
    
    return {
        'returns': all_returns,
        'drawdowns': all_trade_drawdowns,
        'stop_loss_count': stop_loss_count
    }


def main():
    print("="*100)
    print("策略回撤分析与止损比例优化")
    print("="*100)
    print("""
入场: 信号次日开盘价买入
出场: 固定持有期后次日开盘卖出
止损方式对比:
  1. 无止损
  2. 固定止损价: min(信号日最低价, 买入日最低价) * x
""")
    
    print("加载股票数据...")
    stock_data = load_stock_data(max_stocks=5000)
    
    top_strategies = [
        (10, 'G底部反转', '站MA20', '无'),
        (10, 'G底部反转', '无', '无'),
        (5, 'G底部反转', '无', '有EMA'),
        (5, 'B短期多头', '无', '有EMA'),
    ]
    
    holdings = [5, 7, 10, 15, 20, 30, 60]
    stop_loss_ratios = [None, 0.99, 0.98, 0.95, 0.90, 0.85, 0.80]
    
    results = []
    
    print("\n回测中 (回撤分析)...")
    
    for j_thresh, struct, pos, ema in top_strategies:
        for holding in holdings:
            for sl_ratio in stop_loss_ratios:
                result = run_strategy_drawdown(stock_data, j_thresh, struct, pos, ema, holding, sl_ratio)
                
                if len(result['returns']) >= 50:
                    arr = np.array(result['returns'])
                    drawdowns = np.array(result['drawdowns']) if result['drawdowns'] else np.array([0])
                    
                    results.append({
                        'j': j_thresh,
                        'struct': struct,
                        'pos': pos,
                        'ema': ema,
                        'holding': holding,
                        'sl_ratio': sl_ratio,
                        'count': len(arr),
                        'avg_return': arr.mean(),
                        'win_rate': (arr > 0).mean() * 100,
                        'max_drawdown': drawdowns.min(),
                        'avg_drawdown': drawdowns.mean(),
                        'sl_count': result['stop_loss_count'],
                    })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("【各策略回撤统计 (无止损)】")
    print("="*100)
    
    df_no_sl = df[df['sl_ratio'].isna()].copy()
    
    print(f"\n{'持有期':<8} {'策略':<45} {'信号数':<8} {'平均收益':<10} {'胜率':<8} {'最大回撤':<12} {'平均回撤'}")
    print("-"*100)
    
    for holding in holdings:
        print(f"\n--- 持有期: {holding}日 ---")
        df_h = df_no_sl[df_no_sl['holding'] == holding].sort_values('avg_return', ascending=False)
        
        for _, r in df_h.head(3).iterrows():
            strategy = f"J<-{r['j']}+{r['struct']}+{r['pos']}+{r['ema']}"
            print(f"{r['holding']}日{'':<2} {strategy:<45} {r['count']:<8} {r['avg_return']:.2f}%{'':<5} {r['win_rate']:.1f}%{'':<3} {r['max_drawdown']:.2f}%{'':<5} {r['avg_drawdown']:.2f}%")
    
    print("\n" + "="*100)
    print("【止损比例优化: J<-10 + G底部反转 + 站MA20 + 无】")
    print("="*100)
    
    df_sl_opt = df[(df['j'] == 10) & (df['struct'] == 'G底部反转') & 
                   (df['pos'] == '站MA20') & (df['ema'] == '无')]
    
    print(f"\n{'持有期':<8} {'止损比例':<12} {'信号数':<8} {'平均收益':<10} {'胜率':<8} {'最大回撤':<12} {'平均回撤':<10} {'止损次数'}")
    print("-"*110)
    
    for holding in holdings:
        print(f"\n--- 持有期: {holding}日 ---")
        df_h = df_sl_opt[df_sl_opt['holding'] == holding]
        
        for sl_name in ['无止损', 'x=0.99', 'x=0.98', 'x=0.95', 'x=0.90', 'x=0.85', 'x=0.80']:
            if sl_name == '无止损':
                r = df_h[df_h['sl_ratio'].isna()].iloc[0] if len(df_h[df_h['sl_ratio'].isna()]) > 0 else None
            else:
                ratio = float(sl_name.split('=')[1])
                r = df_h[df_h['sl_ratio'] == ratio].iloc[0] if len(df_h[df_h['sl_ratio'] == ratio]) > 0 else None
            
            if r is not None:
                print(f"{r['holding']}日{'':<2} {sl_name:<12} {r['count']:<8} {r['avg_return']:.2f}%{'':<5} {r['win_rate']:.1f}%{'':<3} {r['max_drawdown']:.2f}%{'':<5} {r['avg_drawdown']:.2f}%{'':<5} {r['sl_count']}")
    
    print("\n" + "="*100)
    print("【最优止损比例分析】")
    print("="*100)
    
    print(f"\n{'持有期':<8} {'最优止损比例':<15} {'最优收益':<12} {'vs无止损'}")
    print("-"*50)
    
    for holding in holdings:
        df_h = df_sl_opt[df_sl_opt['holding'] == holding]
        
        best_idx = df_h['avg_return'].idxmax()
        best = df_h.loc[best_idx]
        
        no_sl = df_h[df_h['sl_ratio'].isna()]['avg_return'].values[0]
        change = best['avg_return'] - no_sl
        
        sl_name = '无止损' if pd.isna(best['sl_ratio']) else f"x={best['sl_ratio']}"
        print(f"{holding}日{'':<2} {sl_name:<15} {best['avg_return']:.2f}%{'':<7} {change:+.2f}%")
    
    print("\n" + "="*100)
    print("【总结: 各策略最优止损比例】")
    print("="*100)
    
    print(f"\n{'策略':<45} {'持有期':<8} {'最优止损':<12} {'最优收益':<12} {'最大回撤'}")
    print("-"*80)
    
    for j_thresh, struct, pos, ema in top_strategies:
        strategy = f"J<-{j_thresh}+{struct}+{pos}+{ema}"
        
        for holding in holdings:
            df_s = df[(df['j'] == j_thresh) & (df['struct'] == struct) & 
                      (df['pos'] == pos) & (df['ema'] == ema) & (df['holding'] == holding)]
            
            if len(df_s) == 0:
                continue
            
            best_idx = df_s['avg_return'].idxmax()
            best = df_s.loc[best_idx]
            
            sl_name = '无止损' if pd.isna(best['sl_ratio']) else f"x={best['sl_ratio']}"
            print(f"{strategy:<45} {holding}日{'':<2} {sl_name:<12} {best['avg_return']:.2f}%{'':<5} {best['max_drawdown']:.2f}%")


if __name__ == "__main__":
    main()
