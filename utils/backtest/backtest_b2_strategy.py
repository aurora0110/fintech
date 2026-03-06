import os
import pandas as pd
import numpy as np
from datetime import datetime


# ==================== 增强功能配置参数 ====================

REBALANCE_CONFIG = {
    'enabled': False,
    'period': 'monthly',
    'top_n': 10,
    'force_rebalance': True,
}

MULTI_START_CONFIG = {
    'enabled': False,
    'start_dates': ['2022-01-01', '2023-01-01', '2024-01-01'],
}

ROLLING_WINDOW_CONFIG = {
    'enabled': False,
    'window_years': 2,
    'frequency': 'monthly',
}

TURNOVER_CONFIG = {
    'enabled': True,
}


# ==================== 增强功能函数 ====================

def get_rebalance_dates(all_dates, period='monthly'):
    """生成调仓日期列表"""
    dates = pd.to_datetime(all_dates)
    rebalance_dates = []
    
    if period == 'monthly':
        df_dates = pd.DataFrame({'date': dates})
        df_dates['month'] = df_dates['date'].dt.to_period('M')
        rebalance_dates = df_dates.groupby('month').first()['date'].tolist()
    elif period == '20d':
        for i in range(0, len(dates), 20):
            if i < len(dates):
                rebalance_dates.append(dates[i])
    elif period == '5d':
        for i in range(0, len(dates), 5):
            if i < len(dates):
                rebalance_dates.append(dates[i])
    elif period.endswith('d'):
        interval = int(period.replace('d', ''))
        for i in range(0, len(dates), interval):
            if i < len(dates):
                rebalance_dates.append(dates[i])
    
    return [d for d in rebalance_dates if d in all_dates]


def calculate_b2_factor_scores(stock_data, current_date, volume_multiplier=2.0):
    """计算B2策略因子得分"""
    scores = {}
    
    for stock_code, df in stock_data.items():
        if current_date not in df.index:
            continue
        
        try:
            idx = df.index.get_loc(current_date)
        except KeyError:
            continue
        
        if idx < 2:
            continue
        
        row = df.iloc[idx-1]
        current_row = df.iloc[idx]
        
        trend_ok = row['知行多空线'] <= row['知行短期趋势线']
        j_ok = (row['J'].shift(1) <= 20) & (row['J'] <= 50) if pd.notna(row['J']) else False
        volatility_ok = (row['CLOSE'] / row['CLOSE'].shift(1) - 1) * 100 >= 4 if pd.notna(row['CLOSE']) else False
        volume_ok = row['VOLUME'] > row['VOLUME'].shift(1) * volume_multiplier if pd.notna(row['VOLUME']) else False
        bullish_ok = row['CLOSE'] > row['OPEN']
        
        if trend_ok and j_ok and volatility_ok and volume_ok and bullish_ok:
            score = 1.0
            scores[stock_code] = score
    
    return scores


def run_rebalance_backtest_b2(stock_data, all_dates, volume_multiplier, rebalance_config, turnover_config):
    """带强制调仓的B2策略回测"""
    max_positions = rebalance_config.get('top_n', 10)
    rebalance_dates = get_rebalance_dates(all_dates, rebalance_config.get('period', 'monthly'))
    
    initial_capital = 1000000
    fee_rate = 0.0005
    slippage = 0.001
    
    cash = float(initial_capital)
    positions = {}
    equity_curve = []
    
    turnover_records = []
    rebalance_count = 0
    
    pending_signals = {}
    
    for i, current_date in enumerate(all_dates):
        if rebalance_config.get('enabled') and current_date in rebalance_dates:
            scores = calculate_b2_factor_scores(stock_data, current_date, volume_multiplier)
            pending_signals[current_date] = scores
        
        if i > 0 and all_dates[i-1] in pending_signals:
            exec_date = all_dates[i-1]
            scores = pending_signals[exec_date]
            
            sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            target_stocks = [s[0] for s in sorted_stocks[:max_positions]]
            
            current_value = cash + sum([pos['shares'] * stock_data[pos['stock']].loc[exec_date, 'CLOSE'] 
                                      if pos['stock'] in stock_data and exec_date in stock_data[pos['stock']].index
                                      and not pd.isna(stock_data[pos['stock']].loc[exec_date, 'CLOSE'])
                                      else 0 for pos in positions.values()])
            
            sell_value = 0
            buy_value = 0
            stocks_to_sell = [s for s in positions.keys() if s not in target_stocks]
            for stock in stocks_to_sell:
                if stock in stock_data and exec_date in stock_data[stock].index:
                    price = stock_data[stock].loc[exec_date, 'CLOSE']
                    if price > 0 and stock in positions:
                        shares = positions[stock]['shares']
                        proceeds = shares * price * (1 - fee_rate - slippage)
                        sell_value += proceeds
                        cash += proceeds
                        del positions[stock]
            
            total_after_sell = cash + sum([pos['shares'] * stock_data[pos['stock']].loc[exec_date, 'CLOSE'] 
                                         if pos['stock'] in stock_data and exec_date in stock_data[pos['stock']].index
                                         and not pd.isna(stock_data[pos['stock']].loc[exec_date, 'CLOSE'])
                                         else 0 for pos in positions.values()])
            
            stocks_to_buy = [s for s in target_stocks if s not in positions]
            target_prices = {}
            for stock in stocks_to_buy:
                if stock in stock_data and current_date in stock_data[stock].index:
                    price = stock_data[stock].loc[current_date, 'OPEN']
                    if price > 0:
                        target_prices[stock] = price
            
            scale_factor = 1.0
            if stocks_to_buy:
                needed_cash = sum([target_prices.get(s, 0) * int(total_after_sell / len(stocks_to_buy) / max(target_prices.get(s, 1), 1) / 100) * 100 
                                 for s in stocks_to_buy if s in target_prices])
                if needed_cash > cash and needed_cash > 0:
                    scale_factor = cash / needed_cash * 0.95
            
            target_value_per_stock = (total_after_sell * scale_factor) / max_positions
            
            for stock in stocks_to_buy:
                if stock not in target_prices:
                    continue
                price = target_prices[stock]
                
                shares = int(target_value_per_stock / price / 100) * 100
                if shares > 0:
                    cost = shares * price * (1 + fee_rate + slippage)
                    if cost <= cash:
                        cash -= cost
                        buy_value += cost
                        positions[stock] = {
                            'stock': stock,
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': current_date
                        }
            
            while len(positions) < max_positions and stocks_to_buy:
                remaining_stocks = [s for s in target_stocks if s not in positions]
                if not remaining_stocks:
                    break
                next_stock = remaining_stocks[0]
                if next_stock in target_prices:
                    price = target_prices[next_stock]
                    shares = int(target_value_per_stock / price / 100) * 100
                    if shares > 0:
                        cost = shares * price * (1 + fee_rate + slippage)
                        if cost <= cash:
                            cash -= cost
                            buy_value += cost
                            positions[next_stock] = {
                                'stock': next_stock,
                                'shares': shares,
                                'entry_price': price,
                                'entry_date': current_date
                            }
                stocks_to_buy.remove(next_stock)
            
            if turnover_config.get('enabled') and current_value > 0:
                turnover = (sell_value + buy_value) / current_value
                turnover_records.append(turnover)
            
            if sell_value + buy_value > 0:
                rebalance_count += 1
        
        total_value = cash
        for stock, pos in positions.items():
            if stock in stock_data and current_date in stock_data[stock].index:
                price = stock_data[stock].loc[current_date, 'CLOSE']
                if not pd.isna(price) and price > 0:
                    total_value += pos['shares'] * price
        
        equity_curve.append(total_value)
    
    if len(equity_curve) < 2:
        return None
    
    final_multiple = equity_curve[-1] / initial_capital
    returns_arr = np.array([(equity_curve[i+1] - equity_curve[i]) / equity_curve[i] 
                           for i in range(len(equity_curve)-1) if equity_curve[i] > 0])
    
    sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(252) if returns_arr.std() > 0 else 0
    
    years = len(all_dates) / 252
    CAGR = (final_multiple ** (1 / years)) - 1 if years > 0 and final_multiple > 0 else -1
    
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / running_max
    max_dd = drawdown.min()
    
    avg_turnover = np.mean(turnover_records) if turnover_records else 0
    
    return {
        'desc': f'强制调仓: {rebalance_config.get("period")}',
        'trade_count': rebalance_count,
        'CAGR': CAGR * 100,
        'final_multiple': final_multiple,
        'max_dd': max_dd * 100,
        'sharpe': sharpe,
        'turnover': avg_turnover,
        'rebalance_count': rebalance_count,
    }


def run_multi_start_backtest_b2(stock_data, all_dates, config, volume_multiplier):
    """B2策略多起点回测"""
    results = []
    
    for start_date_str in config.get('start_dates', []):
        start_date = pd.to_datetime(start_date_str)
        valid_dates = [d for d in all_dates if d >= start_date]
        if not valid_dates:
            continue
        
        start_date = valid_dates[0]
        
        if len(valid_dates) < 60:
            continue
        
        result = run_rebalance_backtest_b2(
            stock_data, valid_dates, volume_multiplier,
            {'enabled': True, 'period': config.get('period', 'monthly'), 'top_n': config.get('top_n', 10)},
            {'enabled': True}
        )
        
        if result:
            result['start_date'] = start_date
            results.append(result)
    
    if not results:
        return None
    
    returns = [r['CAGR'] for r in results]
    sharpes = [r['sharpe'] for r in results]
    max_dds = [r['max_dd'] for r in results]
    
    return {
        'results': results,
        'summary': {
            'avg_annual_return': np.mean(returns),
            'std_annual_return': np.std(returns),
            'min_annual_return': np.min(returns),
            'max_annual_return': np.max(returns),
            'avg_sharpe': np.mean(sharpes),
            'avg_max_dd': np.mean(max_dds),
            'count': len(results)
        }
    }


def run_rolling_window_test_b2(stock_data, all_dates, config, volume_multiplier):
    """B2策略滚动窗口测试"""
    window_years = config.get('window_years', 2)
    frequency = config.get('frequency', 'monthly')
    window_days = window_years * 252
    results = []
    
    if frequency == 'monthly':
        df_dates = pd.DataFrame({'date': all_dates})
        df_dates['month'] = df_dates['date'].dt.to_period('M')
        start_indices = df_dates.groupby('month').first().index
        start_dates = [all_dates[i] for i in range(len(all_dates)) 
                     if i in start_indices or (i > 0 and df_dates.iloc[i]['month'] != df_dates.iloc[i-1]['month'])]
    else:
        start_dates = all_dates[::20]
    
    for start_date in start_dates:
        start_idx = all_dates.index(start_date)
        end_idx = start_idx + window_days
        
        if end_idx >= len(all_dates):
            continue
        
        window_dates = all_dates[start_idx:end_idx]
        
        result = run_rebalance_backtest_b2(
            stock_data, window_dates, volume_multiplier,
            {'enabled': True, 'period': config.get('period', 'monthly'), 'top_n': config.get('top_n', 10)},
            {'enabled': True}
        )
        
        if result:
            result['start_date'] = start_date
            result['end_date'] = window_dates[-1]
            results.append(result)
    
    if not results:
        return None
    
    returns = [r['CAGR'] for r in results]
    
    return {
        'results': results,
        'distribution': {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'median': np.median(returns),
            'count': len(results)
        }
    }


# ================= 涨跌幅限制 =================

def is_chi_next_or_star(stock_code):
    """判断是否为创业板或科创板股票"""
    code = stock_code.upper()
    if code.startswith("300") or code.startswith("688"):
        return True
    return False

def limit_price_change(price, prev_price, stock_code, direction="both"):
    """
    限制涨跌幅
    direction: "up"=涨停, "down"=跌停, "both"=双向限制
    """
    if prev_price <= 0 or price <= 0:
        return price
    
    change_pct = (price - prev_price) / prev_price
    
    if is_chi_next_or_star(stock_code):
        max_change = 0.20  # 创业板/科创板 20%
    else:
        max_change = 0.10  # 主板 10%
    
    if direction == "up":
        max_change = max_change
    elif direction == "down":
        max_change = -max_change
    else:
        max_change = max_change
    
    if change_pct > max_change:
        return prev_price * (1 + max_change)
    elif change_pct < -max_change:
        return prev_price * (1 - max_change)
    
    return price


# ================= 日志系统 =================

def write_log(content):

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    log_file = os.path.join(script_dir, script_name + ".txt")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_record = f"\n========== {timestamp} ==========\n"
    new_record += content + "\n\n"

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            old = f.read()
    else:
        old = ""

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(new_record + old)


# ================= 数据加载 =================

def check_data_anomaly(df):
    anomaly_reasons = []
    
    if len(df) < 2:
        return True, anomaly_reasons
    
    for i in range(len(df)):
        row = df.iloc[i]
        open_p = row['OPEN']
        high = row['HIGH']
        low = row['LOW']
        close = row['CLOSE']
        volume = row['VOLUME']
        
        if pd.isna(open_p) or pd.isna(high) or pd.isna(low) or pd.isna(close):
            anomaly_reasons.append(f"{df.index[i]} 数据缺失")
            continue
        
        if high == low == close:
            anomaly_reasons.append(f"{df.index[i]} 一字板")
            continue
        
        if i > 0:
            prev_close = df.iloc[i-1]['CLOSE']
            if prev_close > 0:
                change_pct = (close - prev_close) / prev_close * 100
                if change_pct > 20 or change_pct < -20:
                    anomaly_reasons.append(f"{df.index[i]} 涨跌幅异常: {change_pct:.2f}%")
                    continue
        
        if volume <= 0:
            anomaly_reasons.append(f"{df.index[i]} 成交量为0")
            continue
        
        if i >= 60:
            rolling_vol = df.iloc[i-60:i]['VOLUME']
            avg_vol = rolling_vol.mean()
            if avg_vol > 0 and volume > avg_vol * 5:
                anomaly_reasons.append(f"{df.index[i]} 成交量异常放大: {volume/avg_vol:.2f}倍")
                continue
        
        if open_p < low or open_p > high:
            anomaly_reasons.append(f"{df.index[i]} 开盘价不在高低区间")
            continue
        
        if high > 0 and low > 0:
            amplitude = (high - low) / low * 100
            if i > 0:
                prev_vol = df.iloc[i-1]['VOLUME']
                vol_change = volume / prev_vol if prev_vol > 0 else 1
                if amplitude > 15 and vol_change < 1.2:
                    anomaly_reasons.append(f"{df.index[i]} 振幅异常但无放量")
                    continue
        
        if i >= 1:
            for j in range(1, min(6, i+1)):
                if i - j >= 0:
                    prev_change = (df.iloc[i-j]['CLOSE'] - df.iloc[i-j-1]['CLOSE']) / df.iloc[i-j-1]['CLOSE'] * 100
                    if prev_change < -15:
                        anomaly_reasons.append(f"{df.index[i]} 连续异常大跌")
                        break
    
    if len(df) >= 2:
        first_price = df.iloc[0]['CLOSE']
        last_price = df.iloc[-1]['CLOSE']
        if first_price > 0 and last_price > 0:
            total_change = (last_price - first_price) / first_price
            if abs(total_change) > 100:
                anomaly_reasons.append("全历史价格比例变化异常")
        
        if last_price < 0.5:
            anomaly_reasons.append("股价极低")
        
        if i >= 1:
            avg_vol = df.iloc[:-1]['VOLUME'].mean()
            if avg_vol > 0:
                turnover = df.iloc[-1]['VOLUME'] * close
                avg_turnover = avg_vol * first_price
                if avg_turnover > 0 and turnover < avg_turnover * 0.001:
                    anomaly_reasons.append("成交额过低")
    
    return len(anomaly_reasons) > 0, anomaly_reasons


def load_stock(path):
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            header=1,
            encoding="gbk"
        )
        df.columns = ["日期", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "成交额"][:len(df.columns)]
        df = df[["日期", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        df = df[pd.to_numeric(df["OPEN"], errors="coerce").notna()]
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期")
        df.set_index("日期", inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        is_anomaly, reasons = check_data_anomaly(df)
        if is_anomaly:
            return None
        
        return df
    except Exception as e:
        print(f"加载失败 {path}: {e}")
        return None


# ================= 技术指标（不改） =================

def calculate_indicators(df):

    df['知行短期趋势线'] = df['CLOSE'].ewm(span=10).mean()
    df['知行短期趋势线'] = df['知行短期趋势线'].ewm(span=10).mean()

    df['MA14'] = df['CLOSE'].rolling(14).mean()
    df['MA28'] = df['CLOSE'].rolling(28).mean()
    df['MA57'] = df['CLOSE'].rolling(57).mean()
    df['MA114'] = df['CLOSE'].rolling(114).mean()

    df['知行多空线'] = (
        df['MA14'] + df['MA28'] +
        df['MA57'] + df['MA114']
    ) / 4

    df['HHV9'] = df['HIGH'].rolling(9).max()
    df['LLV9'] = df['LOW'].rolling(9).min()

    rng = df['HHV9'] - df['LLV9']
    df['RSV'] = (df['CLOSE'] - df['LLV9']) / rng * 100
    df['RSV'] = df['RSV'].fillna(50)

    df['K'] = df['RSV'].ewm(alpha=1/3).mean()
    df['D'] = df['K'].ewm(alpha=1/3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    return df


def generate_signal(df, volume_multiplier):

    trend_ok = df['知行多空线'] <= df['知行短期趋势线']
    j_ok = (df['J'].shift(1) <= 20) & (df['J'] <= 50)
    volatility_ok = (df['CLOSE'] / df['CLOSE'].shift(1) - 1) * 100 >= 4
    # 修改：成交量条件从放大2倍改为相较前一天放量即可（> 1）
    volume_ok = df['VOLUME'] > df['VOLUME'].shift(1) * volume_multiplier
    # 当 volume_multiplier = 1.0 时，只要放量即可（不要求具体倍数）
    if volume_multiplier <= 1.0:
        volume_ok = df['VOLUME'] > df['VOLUME'].shift(1)
    bullish_ok = df['CLOSE'] > df['OPEN']

    entity = (df['CLOSE'] - df['OPEN']).abs()
    upper_shadow = df['HIGH'] - np.maximum(df['OPEN'], df['CLOSE'])
    shadow_ratio = np.where(entity > 0, upper_shadow / entity, 0)
    shadow_ok = shadow_ratio < 0.3

    return (
        trend_ok & j_ok & volatility_ok &
        volume_ok & bullish_ok & shadow_ok
    ).fillna(False)


# ================= 主回测 =================

def run_backtest(data_dir,
                 volume_multiplier=2.0,
                 profit_target=8,
                 stop_loss=2,
                 initial_capital=1_000_000,
                 max_positions=4,
                 volume_strategy="60day_all",
                 rolling_window=60,
                 volume_ratio=2.0,
                 top_n=1):

    commission = 0.0005
    stamp = 0.001

    stock_data = {}
    daily_signals = {}

    # 🔥 新增风控状态
    loss_streak = {}
    cooldown_flag = {}
    cooldown_count = {}  # 修复冷却机制：记录剩余冷却次数
    
    # 🔥 每年收益率统计
    yearly_returns = {}  # {年份: [交易收益率列表]}

    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    total = len(files)

    for idx, file in enumerate(files, 1):

        print(f"[股票加载 {idx}/{total}] {file}")

        df = load_stock(os.path.join(data_dir, file))
        if df is None or len(df) < 130:
            print(f"  跳过：数据不足或加载失败，行数={len(df) if df is not None else 0}")
            continue

        df = calculate_indicators(df)
        df["signal"] = generate_signal(df, volume_multiplier)
        
        signal_count = df["signal"].sum()
        if signal_count > 0:
            print(f"  信号数={signal_count}, 数据行数={len(df)}")

        stock_data[file] = df
        loss_streak[file] = 0
        cooldown_flag[file] = False
        cooldown_count[file] = 0  # 初始化冷却次数

        for date in df.index[df["signal"]]:
            daily_signals.setdefault(date, []).append(file)

    all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))

    cash = initial_capital
    positions = []
    equity_curve = []
    
    pending_buy_signals = {}
    pending_exit_signals = {}
    first_trade_date = None
    last_trade_date = None
    
    for date in daily_signals:
        date_idx = all_dates.index(date)
        if date_idx + 1 < len(all_dates):
            next_date = all_dates[date_idx + 1]
            if next_date not in pending_buy_signals:
                pending_buy_signals[next_date] = []
            pending_buy_signals[next_date].extend(daily_signals[date])
    
    for current_date in all_dates:

        new_positions = []

        # ===== 处理持仓 =====
        for pos in positions:

            df = stock_data[pos["stock"]]

            if current_date not in df.index:
                new_positions.append(pos)
                continue

            row = df.loc[current_date]

            open_p = row["OPEN"]
            high = row["HIGH"]
            low = row["LOW"]
            close = row["CLOSE"]

            exit_flag = False
            stop_triggered = False

            idx = df.index.get_loc(current_date)
            
            # 止盈条件（修复未来函数：用前一日成交量判断，次日开盘卖出）
            if idx >= rolling_window:
                volume_exit = False
                
                # 策略 1：滚动窗口 + 阴量/阳量比例
                if volume_strategy == "rolling_ratio":
                    if idx >= rolling_window:
                        prev_volume = df.iloc[idx - 1]["VOLUME"]
                        rolling_volumes = df.iloc[idx - rolling_window:idx]["VOLUME"]
                        if close < open_p:  # 阴量
                            ratio = prev_volume / rolling_volumes.mean()
                            if ratio >= volume_ratio:
                                volume_exit = True
                
                # 策略 2：当日成交量为 rolling_window 日内最高 + 阴量
                elif volume_strategy == "60day_max":
                    prev_volume = df.iloc[idx - 1]["VOLUME"]
                    rolling_max = df.iloc[idx - rolling_window:idx]["VOLUME"].max()
                    if prev_volume >= rolling_max and close < open_p:
                        volume_exit = True
                
                # 策略 3：当日成交量为 rolling_window 日内前 top_n 高 + 阴量
                elif volume_strategy == "60day_topn":
                    prev_volume = df.iloc[idx - 1]["VOLUME"]
                    rolling_volumes = df.iloc[idx - rolling_window:idx]["VOLUME"].sort_values(ascending=False)
                    if len(rolling_volumes) >= top_n:
                        threshold = rolling_volumes.iloc[top_n - 1]
                        if prev_volume >= threshold and close < open_p:
                            volume_exit = True
                
                # 原始策略：当日成交量高于全部 + 阴量（用前一日数据）
                elif volume_strategy == "60day_all":
                    prev_volume = df.iloc[idx - 1]["VOLUME"]
                    rolling_max = df.iloc[idx - rolling_window:idx]["VOLUME"].max()
                    if prev_volume > rolling_max and close < open_p:
                        volume_exit = True
                
                if volume_exit:
                    # 修复未来函数：标记为需要止盈，次日开盘卖出
                    pos["volume_exit_flag"] = True

            # 修复未来函数：成交量止盈信号次日开盘卖出
            if pos.get("volume_exit_flag", False):
                exit_price = open_p
                exit_flag = True

            # 固定止盈：8% 止盈
            if not exit_flag:
                target_price = pos["entry_price"] * (1 + profit_target/100)
                if open_p >= target_price:
                    exit_price = open_p
                    exit_flag = True
                elif high >= target_price:
                    exit_price = target_price
                    exit_flag = True

            # 固定止损：4% 止损（处理跳空：开盘价低于止损价时按开盘价卖出）
            if not exit_flag:
                stop_price_fixed = pos["entry_price"] * (1 - stop_loss/100)
                if open_p <= stop_price_fixed:
                    exit_price = open_p
                    exit_flag = True
                    stop_triggered = True
                elif low <= stop_price_fixed:
                    exit_price = stop_price_fixed
                    exit_flag = True
                    stop_triggered = True

            # B2 信号日最低价止损（仅当固定止损未触发时）
            # 止损价改为买入日的最低价，跳空低开以当天收盘价卖出
            if not exit_flag and low < pos["stop_price"]:
                # 跳空低开：按收盘价卖出（更保守）
                if open_p < pos["stop_price"]:
                    exit_price = close
                else:
                    exit_price = pos["stop_price"]
                exit_flag = True
                stop_triggered = True

            if exit_flag:

                gross = (exit_price - pos["entry_price"]) / pos["entry_price"]
                net = gross - commission*2 - stamp
                cash += pos["invested"] * (1 + net)

                stock = pos["stock"]
                
                # 🔥 记录每年收益率
                year = current_date.year
                if year not in yearly_returns:
                    yearly_returns[year] = []
                yearly_returns[year].append(gross)

                # ===== 更新连续止损统计 =====
                if stop_triggered:
                    loss_streak[stock] += 1
                    # 修复冷却机制：连续止损 3 次后暂停 2 次交易，增强回撤控制
                    if loss_streak[stock] >= 3:
                        cooldown_flag[stock] = True
                        cooldown_count[stock] = 2  # 暂停 2 次交易
                        loss_streak[stock] = 0
                else:
                    loss_streak[stock] = 0

            else:
                pos["current_price"] = close
                new_positions.append(pos)

        positions = new_positions

        # ===== 计算净值 =====
        total_equity = cash
        for pos in positions:
            total_equity += pos["invested"] * (
                pos["current_price"] / pos["entry_price"]
            )

        equity_curve.append(total_equity)
        
        if positions and first_trade_date is None:
            first_trade_date = current_date
        last_trade_date = current_date

        # ===== 执行前一天发出的买入信号（次日以开盘价买入） =====
        if current_date not in pending_buy_signals:
            continue

        available_slots = max_positions - len(positions)
        if available_slots <= 0:
            continue

        signals_today = pending_buy_signals[current_date]

        for stock in signals_today:

            if available_slots <= 0:
                break

            # 🔥 冷却机制（修复：暂停 2 次交易而非 1 次）
            if cooldown_flag[stock] and cooldown_count[stock] > 0:
                cooldown_count[stock] -= 1
                if cooldown_count[stock] == 0:
                    cooldown_flag[stock] = False
                continue

            df = stock_data[stock]
            idx = df.index.get_loc(current_date)

            if idx + 1 >= len(df):
                continue

            entry_price = df.iloc[idx+1]["OPEN"]
            stop_price = df.iloc[idx]["LOW"]  # B2 信号发出时的最低点
            
            prev_close = df.iloc[idx]["CLOSE"]
            limit_pct = df.iloc[idx].get('limit_pct', 0.10)
            if pd.notna(prev_close) and prev_close > 0:
                limit_up = prev_close * (1 + limit_pct)
                if entry_price >= limit_up:
                    continue
            
            if entry_price <= 0:
                continue

            # 修复资金分配：按当前现金动态分配，而非固定初始资金
            invested = cash / available_slots
            cash -= invested

            positions.append({
                "stock": stock,
                "entry_price": entry_price,
                "entry_date": current_date,
                "invested": invested,
                "current_price": entry_price,
                "stop_price": stop_price
            })

            available_slots -= 1

    equity_curve = np.array(equity_curve)
    
    # 边界保护：如果没有交易数据
    if len(equity_curve) == 0:
        print("错误：没有交易数据，请检查数据源和信号条件")
        return

    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    # 修复 CAGR 边界保护 + 年化时间计算（使用真实日期差）
    final_multiple = equity_curve[-1] / initial_capital
    
    # 使用真实日期差计算年数（修复年化失真）
    if first_trade_date and last_trade_date:
        total_years = (last_trade_date - first_trade_date).days / 365.25
        total_years = max(total_years, 0.01)
    else:
        start_date = all_dates[0]
        end_date = all_dates[-1]
        total_years = (end_date - start_date).days / 365.25
    
    # CAGR 边界保护
    if total_years > 0 and final_multiple > 0:
        CAGR = final_multiple ** (1 / total_years) - 1
    else:
        CAGR = 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = np.min(drawdowns)

    sharpe = 0
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = (np.mean(daily_returns) /
                  np.std(daily_returns)) * np.sqrt(252)

    result = (
        "===== 加入连续止损暂停机制 =====\n\n"
        f"CAGR: {CAGR:.4f}\n"
        f"最大回撤: {max_dd:.4f}\n"
        f"年化夏普: {sharpe:.4f}\n"
        f"最终资金倍数: {final_multiple:.4f}"
    )

    print(result)
    
    # 🔥 打印每年收益率分布
    print("\n" + "=" * 50)
    print("每年收益率分布")
    print("=" * 50)
    for year in sorted(yearly_returns.keys()):
        returns = yearly_returns[year]
        if returns:
            avg_return = np.mean(returns) * 100
            win_rate = np.mean([r > 0 for r in returns]) * 100
            print(f"{year}年: 交易次数={len(returns)}, 平均收益率={avg_return:.2f}%, 胜率={win_rate:.1f}%")
    
    print("\n" + "=" * 50)
    print("持有期间收益率分布统计")
    print("=" * 50)
    
    all_trade_returns = []
    for year_returns in yearly_returns.values():
        all_trade_returns.extend(year_returns)
    
    if all_trade_returns:
        returns_arr = np.array(all_trade_returns)
        returns_pct = returns_arr * 100
        
        bins = [(-200, -10), (-10, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 10), (10, 200)]
        print(f"{'收益率区间':<15} {'交易次数':<10} {'占比':<10}")
        print("-" * 35)
        for low, high in bins:
            count = np.sum((returns_pct >= low) & (returns_pct < high))
            pct = count / len(returns_pct) * 100 if len(returns_pct) > 0 else 0
            print(f"[{low:>5}%, {high:<5}%)    {count:<10} {pct:.2f}%")
        
        median_return = np.median(returns_arr) * 100
        print(f"\n收益率中位数: {median_return:.2f}%")
        print(f"平均收益率: {np.mean(returns_arr) * 100:.2f}%")
        
        if len(returns_arr) >= 2:
            returns_sorted = np.sort(returns_arr)
            print(f"25分位数: {returns_sorted[len(returns_arr)//4] * 100:.2f}%")
            print(f"75分位数: {returns_sorted[len(returns_arr)*3//4] * 100:.2f}%")
        
        var_95 = np.percentile(returns_arr, 5) * 100
        cvar_95 = returns_arr[returns_arr <= np.percentile(returns_arr, 5)].mean() * 100
        print(f"VaR(95%): {var_95:.2f}%")
        print(f"CVaR(95%): {cvar_95:.2f}%")
        
        print(f"\n最大盈利: {returns_arr.max() * 100:.2f}%")
        print(f"最大亏损: {returns_arr.min() * 100:.2f}%")
        print(f"收益标准差: {returns_arr.std() * 100:.2f}%")
    
    write_log(result)


if __name__ == "__main__":

    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

    run_backtest(
        data_dir=data_dir,
        volume_multiplier=1.0,  # 修改：只要放量即可（不要求2倍）
        profit_target=8,
        stop_loss=2,
        initial_capital=1_000_000,
        max_positions=4
    )