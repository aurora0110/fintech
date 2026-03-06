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


def calculate_brick_factor_scores(stock_data, current_date):
    """计算砖块策略因子得分"""
    scores = {}
    
    for stock_code, df in stock_data.items():
        if current_date not in df.index:
            continue
        
        try:
            idx = df.index.get_loc(current_date)
        except KeyError:
            continue
        
        if idx < 5:
            continue
        
        row = df.iloc[idx]
        
        if pd.isna(row.get('BRICK')) or row.get('BRICK') != 1:
            continue
        
        score = 1.0
        scores[stock_code] = score
    
    return scores


def run_rebalance_backtest_brick(stock_data, all_dates, fixed_investment, rebalance_config, turnover_config):
    """带强制调仓的砖块策略回测"""
    max_positions = rebalance_config.get('top_n', 10)
    rebalance_dates = get_rebalance_dates(all_dates, rebalance_config.get('period', 'monthly'))
    
    initial_capital = 1000000
    fee_rate = 0.0003
    slippage = 0.001
    
    cash = float(initial_capital)
    positions = {}
    equity_curve = []
    
    turnover_records = []
    rebalance_count = 0
    
    pending_signals = {}
    
    for i, current_date in enumerate(all_dates):
        if rebalance_config.get('enabled') and current_date in rebalance_dates:
            scores = calculate_brick_factor_scores(stock_data, current_date)
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


def run_multi_start_backtest_brick(stock_data, all_dates, config, fixed_investment):
    """砖块策略多起点回测"""
    results = []
    
    for start_date_str in config.get('start_dates', []):
        start_date = pd.to_datetime(start_date_str)
        valid_dates = [d for d in all_dates if d >= start_date]
        if not valid_dates:
            continue
        
        start_date = valid_dates[0]
        
        if len(valid_dates) < 60:
            continue
        
        result = run_rebalance_backtest_brick(
            stock_data, valid_dates, fixed_investment,
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


def run_rolling_window_test_brick(stock_data, all_dates, config, fixed_investment):
    """砖块策略滚动窗口测试"""
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
        
        result = run_rebalance_backtest_brick(
            stock_data, window_dates, fixed_investment,
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


# ===========================
# 涨跌幅限制
# ===========================

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


def tongdaxin_sma(series, n, m=1):
    result = np.zeros(len(series))
    prev_sma = 0
    for i in range(len(series)):
        val = series.iloc[i]
        if i < n - 1:
            sma = series.iloc[:i+1].sum() / (i + 1)
        else:
            sma = (val * m + prev_sma * (n - m)) / n
        result[i] = sma
        prev_sma = sma
    return pd.Series(result, index=series.index)


def brick_chart_indicator(df):
    df = df.copy()

    df['HHV_H4'] = df['HIGH'].rolling(window=4).max()
    df['LLV_L4'] = df['LOW'].rolling(window=4).min()

    df['VAR1A'] = (df['HHV_H4'] - df['CLOSE']) / (df['HHV_H4'] - df['LLV_L4']) * 100 - 90
    df['VAR1A'] = df['VAR1A'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df['VAR2A'] = tongdaxin_sma(df['VAR1A'], 4, 1) + 100

    df['VAR3A'] = (df['CLOSE'] - df['LLV_L4']) / (df['HHV_H4'] - df['LLV_L4']) * 100
    df['VAR3A'] = df['VAR3A'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df['VAR4A'] = tongdaxin_sma(df['VAR3A'], 6, 1)
    df['VAR5A'] = tongdaxin_sma(df['VAR4A'], 6, 1) + 100

    df['VAR6A'] = df['VAR5A'] - df['VAR2A']

    df['砖型图数值'] = np.where(df['VAR6A'] > 4, df['VAR6A'] - 4, 0)
    df['砖型图变化量'] = df['砖型图数值'] - df['砖型图数值'].shift(1)

    df['当日柱体长度'] = df['砖型图变化量'].abs()
    df['前日柱体长度'] = df['当日柱体长度'].shift(1)
    df['当日红柱'] = df['砖型图变化量'] > 0
    df['前日绿柱'] = df['砖型图变化量'].shift(1) < 0

    df['买入信号'] = np.where(
        (df['前日绿柱'] == True) &
        (df['当日红柱'] == True) &
        (df['当日柱体长度'] >= df['前日柱体长度'] * 0.66),
        1,
        0
    )

    df = df.dropna()

    return df


def calculate_trend(df):
    df['知行短期趋势线'] = df['CLOSE'].ewm(span=10, adjust=False).mean()
    df['知行短期趋势线'] = df['知行短期趋势线'].ewm(span=10, adjust=False).mean()

    df['MA14'] = df['CLOSE'].rolling(window=14).mean()
    df['MA28'] = df['CLOSE'].rolling(window=28).mean()
    df['MA57'] = df['CLOSE'].rolling(window=57).mean()
    df['MA114'] = df['CLOSE'].rolling(window=114).mean()

    df['知行多空线'] = (df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']) / 4

    return df


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
            names=["日期", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "成交额"],
            skiprows=1,
            encoding='gbk'
        )
        df = df.iloc[1:]
        df["日期"] = pd.to_datetime(df["日期"], errors='coerce')
        df = df[df["日期"].notna()]
        
        for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        df = df.sort_values("日期")
        df.set_index("日期", inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        if len(df) < 10:
            return None
        
        is_anomaly, reasons = check_data_anomaly(df)
        if is_anomaly:
            return None
        
        return df
    except:
        return None


def run_backtest(data_dir, initial_capital=1_000_000, max_positions=4):
    stock_data = {}
    daily_signals = {}
    
    # 使用固定金额仓位（每次投入固定金额，不受复利影响）
    fixed_investment = 100000  # 每次投入10万元

    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    total = len(files)

    print(f"加载 {total} 只股票...")

    for idx, file in enumerate(files, 1):
        if idx % 500 == 0:
            print(f"[加载 {idx}/{total}]")

        df = load_stock(os.path.join(data_dir, file))
        if df is None or len(df) < 120:
            continue

        df = calculate_trend(df)
        df = brick_chart_indicator(df)

        stock_data[file] = df

        for date in df.index[df["买入信号"] == 1]:
            if date in df.index:
                row = df.loc[date]
                if row['知行多空线'] <= row['知行短期趋势线']:
                    # 不使用未来函数，直接用当日收盘价
                    if row['CLOSE'] >= row['知行多空线']:
                        daily_signals.setdefault(date, []).append(file)

    # 获取所有股票的日期并集作为时间轴
    all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))

    print(f"总交易日: {len(all_dates)}")
    print(f"有信号的天数: {len(daily_signals)}")

    total_signals = sum(len(v) for v in daily_signals.values())
    print(f"总信号数量: {total_signals}")

    # 将日期转换为索引位置
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}

    cash = float(initial_capital)
    positions = []
    equity_curve = []
    stopped = False
    trade_count = 0
    abnormal_count = 0
    stock_returns = {}  # 记录每只股票的收益率
    
    # 🔥 每年收益率统计
    yearly_returns = {}  # {年份: [交易收益率列表]}

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
        if stopped:
            break
            
        current_idx = date_to_idx[current_date]
        new_positions = []

        for pos in positions:
            df = stock_data[pos["stock"]]

            if current_date not in df.index:
                new_positions.append(pos)
                continue

            row = df.loc[current_date]

            # 防止价格异常
            if pd.isna(row["CLOSE"]) or row["CLOSE"] <= 0:
                new_positions.append(pos)
                continue

            open_p = row["OPEN"]
            high = row["HIGH"]
            low = row["LOW"]
            close = row["CLOSE"]

            # 用股票自己的索引计算持有天数
            df = stock_data[pos["stock"]]
            stock_idx = df.index.get_loc(current_date)
            stock_entry_idx = int(pos["entry_idx"])
            holding_days = stock_idx - stock_entry_idx

            entry_price = pos["entry_price"]
            
            # 计算当天涨跌幅（相对买入价）
            open_change = (open_p - entry_price) / entry_price  # 开盘价相对买入价的涨幅
            high_change = (high - entry_price) / entry_price   # 最高价相对买入价的涨幅
            low_change = (low - entry_price) / entry_price     # 最低价相对买入价的跌幅
            
            # 固定止盈止损
            TP_RATE = 0.03   # 止盈3%
            SL_RATE = -0.02  # 止损-2%
            
            exit_flag = False
            exit_price = close
            
            # 开盘价优先原则：开盘价已穿过止盈/止损价，按开盘价成交
            if open_change >= TP_RATE:
                exit_price = open_p  # 开盘高开3%以上，以开盘价卖出
                exit_flag = True
            elif open_change <= SL_RATE:
                exit_price = open_p  # 开盘低开2%以上，以开盘价卖出
                exit_flag = True
            # 检查是否触及止盈（盘中）
            elif high_change >= TP_RATE:
                exit_price = entry_price * (1 + TP_RATE)  # 以止盈价卖出
                exit_flag = True
            # 检查是否触及止损（盘中）
            elif low_change <= SL_RATE:
                exit_price = entry_price * (1 + SL_RATE)  # 以止损价卖出
                exit_flag = True
            # 检查是否持有满3天
            elif holding_days >= 3:
                exit_price = close  # 第3天收盘价卖出
                exit_flag = True

            if exit_flag:
                gross = (exit_price - pos["entry_price"]) / pos["entry_price"]

                # 4️⃣ 异常数据保护：超出合理范围的收益判定为数据异常，整笔交易作废，资金原样退回
                is_abnormal = False
                if np.isnan(gross) or np.isinf(gross):
                    is_abnormal = True
                elif abs(gross) > 2:  # 持有3天收益超过200%视为异常
                    is_abnormal = True
                
                if is_abnormal:
                    # 异常交易：资金原样退回，不计入收益
                    cash += pos["invested"]
                    abnormal_count += 1
                else:
                    # 正常交易：计算收益
                    trade_count += 1
                    if not np.isnan(gross) and not np.isnan(pos["invested"]):
                        cash += pos["invested"] * (1 + gross)
                        # 记录股票收益率
                        stock = pos["stock"]
                        if stock not in stock_returns:
                            stock_returns[stock] = []
                        stock_returns[stock].append(gross)
                        
                        # 🔥 记录每年收益率
                        year = current_date.year
                        if year not in yearly_returns:
                            yearly_returns[year] = []
                        yearly_returns[year].append(gross)
            else:
                pos["current_price"] = close
                new_positions.append(pos)

        positions = new_positions

        # 计算总权益
        total_equity = cash if not np.isnan(cash) else 0
        for pos in positions:
            price_ratio = pos["current_price"] / pos["entry_price"]
            # 防止价格比例异常
            if np.isnan(price_ratio) or price_ratio <= 0:
                price_ratio = 1.0
            if not np.isnan(pos["invested"]):
                total_equity += pos["invested"] * price_ratio

        if np.isnan(total_equity):
            stopped = True
        elif total_equity <= 0:
            stopped = True
            
        if stopped:
            equity_curve.append(total_equity)
            break

        equity_curve.append(total_equity)
        
        if positions and first_trade_date is None:
            first_trade_date = current_date
        last_trade_date = current_date

        if current_date not in pending_buy_signals:
            continue

        available_slots = max_positions - len(positions)
        if available_slots <= 0:
            continue

        if current_date not in pending_buy_signals:
            continue

        signals_today = pending_buy_signals[current_date]

        existing_stocks = {pos["stock"] for pos in positions}
        available_signals = [s for s in signals_today if s not in existing_stocks]

        if not available_signals:
            continue

        # 防止资金不足或异常
        if cash <= 0 or np.isnan(cash) or np.isinf(cash):
            continue

        # 每次投入固定金额
        # 根据信号数量平均分配
        num_to_buy = min(len(available_signals), available_slots)
        
        # 每只股票投入固定金额
        fixed_invested = fixed_investment

        count = 0
        for stock in available_signals:
            if count >= available_slots:
                break
            
            # 1️⃣ 检查资金是否足够
            if cash < fixed_invested:
                break

            df = stock_data[stock]
            idx = df.index.get_loc(current_date)

            if idx + 1 >= len(df):
                continue

            # 获取当天收盘价和明天开盘价，计算跳空幅度
            today_close = df.iloc[idx]["CLOSE"]
            entry_price = df.iloc[idx + 1]["OPEN"]

            # 防止价格异常
            if entry_price <= 0 or np.isnan(entry_price):
                continue
            if today_close <= 0 or np.isnan(today_close):
                continue
                
            # 跳空高开过滤：开盘价相对当天收盘价涨幅>=3%不买入
            gap_up = (entry_price - today_close) / today_close
            if gap_up >= 0.03:
                continue

            invested = fixed_invested

            positions.append({
                "stock": stock,
                "entry_price": entry_price,
                "entry_date": current_date,
                "entry_idx": idx + 1,
                "invested": invested,
                "current_price": entry_price
            })

            cash -= invested
            count += 1

    equity_curve = np.array(equity_curve)

    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    if first_trade_date and last_trade_date:
        total_years = (last_trade_date - first_trade_date).days / 365.25
        total_years = max(total_years, 0.01)
    else:
        total_years = len(equity_curve) / 252
    
    final_multiple = equity_curve[-1] / initial_capital
    
    # 2️⃣ CAGR 边界保护
    if total_years <= 0 or final_multiple <= 0:
        CAGR = -1.0
    else:
        CAGR = final_multiple ** (1 / total_years) - 1

    running_max = np.maximum.accumulate(equity_curve)
    
    # 3️⃣ 防止 running_max = 0 导致除零
    running_max_safe = np.where(running_max == 0, 1, running_max)
    drawdowns = (equity_curve - running_max_safe) / running_max_safe
    max_dd = np.min(drawdowns)

    sharpe = 0
    if np.std(daily_returns) > 0 and len(daily_returns) > 0:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

    print("\n" + "=" * 50)
    print("砖型图策略回测结果（统一资金曲线版）")
    print("=" * 50)
    print(f"初始资金: {initial_capital:,.0f}")
    print(f"最终资金: {equity_curve[-1]:,.2f}")
    print(f"最终倍数: {final_multiple:.4f}")
    print(f"年化收益率(CAGR): {CAGR:.4f}")
    print(f"最大回撤: {max_dd:.4f}")
    print(f"年化夏普: {sharpe:.4f}")
    print(f"正常交易: {trade_count}, 异常交易: {abnormal_count}")

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
    
    # 保存结果到文件
    result_file = "/Users/lidongyang/Desktop/Qstrategy/utils/backtest/backtest_brick_strategy.txt"
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_content = f"========== {timestamp} ==========\n"
    new_content += f"最终资金: {equity_curve[-1]:,.2f}\n"
    new_content += f"最终倍数: {final_multiple:.4f}\n"
    new_content += f"年化收益率(CAGR): {CAGR:.4f}\n"
    new_content += f"最大回撤: {max_dd:.4f}\n"
    new_content += f"年化夏普: {sharpe:.4f}\n"
    new_content += f"正常交易: {trade_count}, 异常交易: {abnormal_count}\n"
    new_content += "\n"
    
    # 读取现有内容，追加到后面
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            old_content = f.read()
    except:
        old_content = ""
    
    # 新内容写在最前面
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(new_content + old_content)

    # 计算每只股票的平均收益率
    stock_avg_returns = {}
    for stock, returns in stock_returns.items():
        if returns:
            stock_avg_returns[stock] = np.mean(returns)

    # 排序
    sorted_stocks = sorted(stock_avg_returns.items(), key=lambda x: x[1], reverse=True)

    # 收益率最高的20只
    print("\n" + "=" * 60)
    print("收益率最高的20只股票:")
    print("=" * 60)
    print(f"{'股票代码':<20} {'平均收益率':<15}")
    print("-" * 60)
    for stock, avg_return in sorted_stocks[:20]:
        print(f"{stock:<20} {avg_return*100:>10.2f}%")

    # 收益率最低的20只
    print("\n" + "=" * 60)
    print("收益率最低的20只股票:")
    print("=" * 60)
    print(f"{'股票代码':<20} {'平均收益率':<15}")
    print("-" * 60)
    for stock, avg_return in sorted_stocks[-20:]:
        print(f"{stock:<20} {avg_return*100:>10.2f}%")

    print("=" * 50)


if __name__ == "__main__":
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
    run_backtest(data_dir)
