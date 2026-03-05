import os
import pandas as pd

data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

files = [f for f in os.listdir(data_dir) if f.endswith('.txt')][:200]

results = []
for file in files:
    stock_code = file.replace('.txt', '')
    path = os.path.join(data_dir, file)
    
    try:
        df = pd.read_csv(path, sep='\t', encoding='utf-8')
        df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期')
        df = df.sort_index()
        
        if len(df) >= 2:
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            prev_close = prev_row['CLOSE']
            close = last_row['CLOSE']
            high = last_row['HIGH']
            
            if prev_close > 0:
                change_pct = (close - prev_close) / prev_close * 100
                
                if change_pct >= 9.9:
                    results.append({
                        'stock': stock_code,
                        'prev_close': prev_close,
                        'close': close,
                        'high': high,
                        'change_pct': change_pct
                    })
    except:
        continue

print(f"今日涨停股票数量: {len(results)}")
print("\n股票列表:")
for r in results[:30]:
    print(f"{r['stock']}: 昨收={r['prev_close']:.2f}, 今收={r['close']:.2f}, 涨幅={r['change_pct']:.2f}%")
