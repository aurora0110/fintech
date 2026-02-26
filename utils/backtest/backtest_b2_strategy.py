import os
import pandas as pd
import numpy as np
from datetime import datetime


def log_backtest_result(script_name, results_summary, buy_conditions, sell_conditions, stop_loss, data_path, method):
    results_dir = "/Users/lidongyang/Desktop/Qstrategy/results"
    os.makedirs(results_dir, exist_ok=True)

    backtest_folder = os.path.join(results_dir, "backtest")
    os.makedirs(backtest_folder, exist_ok=True)

    txt_file = os.path.join(backtest_folder, f"{script_name}.txt")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    log_content = f"""
{'='*80}
记录时间: {timestamp}
{'='*80}

【买入条件】
{buy_conditions}

【卖出策略】
{sell_conditions}

【止损条件】
{stop_loss}

【回测数据路径】
{data_path}

【回测方法】
{method}

【回测结果】
{results_summary}

"""

    with open(txt_file, 'a', encoding='utf-8') as f:
        f.write(log_content)

    print(f"回测结果已记录到: {txt_file}")


def calculate_trend_vectorized(df, M1=14, M2=28, M3=57, M4=114):
    df['知行短期趋势线'] = df['CLOSE'].ewm(span=10, adjust=False).mean()
    df['知行短期趋势线'] = df['知行短期趋势线'].ewm(span=10, adjust=False).mean()

    df['MA14'] = df['CLOSE'].rolling(window=M1, min_periods=1).mean()
    df['MA28'] = df['CLOSE'].rolling(window=M2, min_periods=1).mean()
    df['MA57'] = df['CLOSE'].rolling(window=M3, min_periods=1).mean()
    df['MA114'] = df['CLOSE'].rolling(window=M4, min_periods=1).mean()
    df['知行多空线'] = (df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']) / 4

    return df


def calculate_kdj_vectorized(df, N=9, M1=3, M2=3):
    df['HHV9'] = df['HIGH'].rolling(window=N, min_periods=1).max()
    df['LLV9'] = df['LOW'].rolling(window=N, min_periods=1).min()

    rng = df['HHV9'] - df['LLV9']
    rng = rng.replace(0, np.nan)

    df['RSV'] = (df['CLOSE'] - df['LLV9']) / rng * 100
    df['RSV'] = df['RSV'].fillna(50)

    df['K'] = df['RSV'].ewm(alpha=1/M1, adjust=False, min_periods=1).mean()
    df['D'] = df['K'].ewm(alpha=1/M2, adjust=False, min_periods=1).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    df['K'] = df['K'].fillna(50)
    df['D'] = df['D'].fillna(50)
    df['J'] = df['J'].fillna(50)

    return df


def precompute_b2_signals(df):
    if len(df) < 4:
        return pd.Series([False] * len(df)), pd.Series([0.0] * len(df))

    df_calc = df[['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']].copy()

    df_calc = calculate_trend_vectorized(df_calc)
    df_calc = calculate_kdj_vectorized(df_calc)

    trend_ok = df_calc['知行多空线'] <= df_calc['知行短期趋势线']

    j_yesterday = df_calc['J'].shift(1)
    j_today = df_calc['J']
    j_ok = (j_yesterday <= 50) & (j_today <= 80)

    close_yesterday = df_calc['CLOSE'].shift(1)
    close_yesterday = close_yesterday.replace(0, np.nan)
    change_pct = ((df_calc['CLOSE'] - close_yesterday) / close_yesterday * 100).fillna(0)
    volatility_ok = change_pct >= 4

    volume_yesterday = df_calc['VOLUME'].shift(1)
    volume_ok = df_calc['VOLUME'] > volume_yesterday

    volume_3x_ok = df_calc['VOLUME'] > volume_yesterday * 3

    high_col = df_calc['HIGH']
    open_col = df_calc['OPEN']
    close_col = df_calc['CLOSE']

    length = (high_col - open_col).abs()
    shadow_length = high_col - close_col

    with np.errstate(divide='ignore', invalid='ignore'):
        shadow_ratio = np.where(length > 0, shadow_length / length, 0)
    shadow_ok = shadow_ratio < 0.3

    b2_signal = trend_ok & j_ok & volatility_ok & volume_ok & volume_3x_ok & shadow_ok

    stop_loss_prices = df_calc['LOW'].copy()

    return b2_signal.fillna(False), stop_loss_prices.fillna(0)


class B2StrategyBacktest:
    def __init__(self, data_dir, results_file=None):
        self.data_dir = data_dir
        self.results_file = results_file
        self.results = []
        self.initial_capital = 1000000
        self.data_cache = {}

    def load_stock_data(self, file_path):
        encodings = ['gbk', 'utf-8', 'latin-1', 'gb18030']
        df = None

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()

                data = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    if i == 0 and '开盘' in line:
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            date_str = parts[0]
                            open_price = float(parts[1])
                            high_price = float(parts[2])
                            low_price = float(parts[3])
                            close_price = float(parts[4])
                            volume = float(parts[5])

                            data.append([date_str, open_price, high_price, low_price, close_price, volume])
                        except ValueError:
                            continue

                df = pd.DataFrame(data, columns=['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'])
                df['日期'] = pd.to_datetime(df['日期'])
                break
            except (UnicodeDecodeError, LookupError):
                continue
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {str(e)}")
                return None

        return df

    def precompute_stock_signals(self, df):
        b2_signals, stop_loss_prices = precompute_b2_signals(df)

        df_signals = pd.DataFrame({
            'b2_signal': b2_signals,
            'stop_loss_price': stop_loss_prices
        })

        return df_signals

    def run_strategy(self, df, stock_code, strategy_type):
        if len(df) < 130:
            return self.calculate_metrics([], stock_code, strategy_type)

        df_signals = self.precompute_stock_signals(df)

        trades = []
        position = None

        start_idx = 120
        end_idx = len(df) - 6

        for i in range(start_idx, end_idx):
            if position is None:
                for j in range(i, min(i + 6, len(df) - 1)):
                    if df_signals.iloc[j]['b2_signal']:
                        signal_low = df_signals.iloc[j]['stop_loss_price']

                        buy_date = df.iloc[j + 1]['日期']
                        buy_price = df.iloc[j + 1]['OPEN']

                        position = {
                            'entry_date': buy_date,
                            'entry_price': buy_price,
                            'stop_loss_price': signal_low,
                            'buy_idx': j + 1,
                            'holding_days': 0,
                            'sold': False
                        }
                        break

            if position and not position['sold']:
                buy_idx = position['buy_idx']

                for hold_day in range(1, 6):
                    current_idx = buy_idx + hold_day
                    if current_idx >= len(df):
                        break

                    current_data = df.iloc[current_idx]

                    if hold_day == position['holding_days'] + 1:
                        position['holding_days'] = hold_day

                    entry_price = position['entry_price']

                    high_slice = df.iloc[buy_idx:current_idx + 1]['HIGH']
                    high_price = high_slice.max() if len(high_slice) > 0 else current_data['HIGH']

                    current_close = current_data['CLOSE']
                    current_open = current_data['OPEN']

                    max_rise = (high_price - entry_price) / entry_price * 100

                    should_sell = False
                    sell_reason = ''
                    exit_price = 0

                    profit_target = 0
                    if '3%止盈' in strategy_type:
                        profit_target = 3
                    elif '4%止盈' in strategy_type:
                        profit_target = 4
                    elif '5%止盈' in strategy_type:
                        profit_target = 5
                    elif '6%止盈' in strategy_type:
                        profit_target = 6
                    elif '7%止盈' in strategy_type:
                        profit_target = 7
                    elif '8%止盈' in strategy_type:
                        profit_target = 8
                    elif '9%止盈' in strategy_type:
                        profit_target = 9

                    if profit_target > 0 and max_rise >= profit_target:
                        should_sell = True
                        sell_reason = f'{profit_target}%止盈'
                        exit_price = current_close

                    if current_close <= position['stop_loss_price']:
                        should_sell = True
                        sell_reason = '止损'
                        exit_price = current_close

                    if strategy_type == '2天收盘卖' and hold_day == 1 and not should_sell:
                        should_sell = True
                        sell_reason = '2天收盘卖'
                        exit_price = current_close

                    if strategy_type == '2天开盘卖' and hold_day == 1 and not should_sell:
                        should_sell = True
                        sell_reason = '2天开盘卖'
                        exit_price = current_open

                    if strategy_type == '3天收盘卖' and hold_day == 2 and not should_sell:
                        should_sell = True
                        sell_reason = '3天收盘卖'
                        exit_price = current_close

                    if strategy_type == '3天开盘卖' and hold_day == 2 and not should_sell:
                        should_sell = True
                        sell_reason = '3天开盘卖'
                        exit_price = current_open

                    if strategy_type == '4天收盘卖' and hold_day == 3 and not should_sell:
                        should_sell = True
                        sell_reason = '4天收盘卖'
                        exit_price = current_close

                    if strategy_type == '4天开盘卖' and hold_day == 3 and not should_sell:
                        should_sell = True
                        sell_reason = '4天开盘卖'
                        exit_price = current_open

                    if strategy_type == '5天收盘卖' and hold_day == 4 and not should_sell:
                        should_sell = True
                        sell_reason = '5天收盘卖'
                        exit_price = current_close

                    if strategy_type == '5天开盘卖' and hold_day == 4 and not should_sell:
                        should_sell = True
                        sell_reason = '5天开盘卖'
                        exit_price = current_open

                    if strategy_type == '6天收盘卖' and hold_day == 5 and not should_sell:
                        should_sell = True
                        sell_reason = '6天收盘卖'
                        exit_price = current_close

                    if strategy_type == '6天开盘卖' and hold_day == 5 and not should_sell:
                        should_sell = True
                        sell_reason = '6天开盘卖'
                        exit_price = current_open

                    if should_sell:
                        profit = (exit_price - entry_price) / entry_price
                        holding_time = (current_data['日期'] - position['entry_date']).days
                        trades.append({
                            'stock_code': stock_code,
                            'entry_date': position['entry_date'],
                            'entry_price': entry_price,
                            'exit_date': current_data['日期'],
                            'exit_price': exit_price,
                            'profit': profit,
                            'holding_days': holding_time,
                            'reason': sell_reason
                        })
                        position = None
                        break

        return self.calculate_metrics(trades, stock_code, strategy_type)

    def calculate_metrics(self, trades, stock_code, strategy_type):
        if not trades:
            return {
                'stock_code': stock_code,
                'strategy': strategy_type,
                'num_trades': 0,
                'total_profit': 0,
                'success_rate': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'max_consecutive_losses': 0,
                'avg_consecutive_losses': 0,
                'sharpe_ratio': 0,
                'avg_daily_trades': 0,
                'turnover_rate': 0,
                'avg_holding_time': 0,
                'trade_density': 0,
                'trades': []
            }

        trades_df = pd.DataFrame(trades)
        num_trades = len(trades_df)
        total_profit = trades_df['profit'].sum()
        success_rate = len(trades_df[trades_df['profit'] > 0]) / num_trades if num_trades > 0 else 0

        if num_trades > 1:
            cumulative_profit = (1 + trades_df['profit']).cumprod()
            running_max = cumulative_profit.expanding().max()
            drawdowns = (cumulative_profit - running_max) / running_max
            max_drawdown = drawdowns.min() * 100
            avg_drawdown = drawdowns.mean() * 100
        else:
            max_drawdown = 0
            avg_drawdown = 0

        consecutive_losses = 0
        max_consecutive_losses = 0
        for profit in trades_df['profit']:
            if profit <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        avg_consecutive_losses = max_consecutive_losses / num_trades if num_trades > 0 else 0

        first_date = trades_df['entry_date'].min()
        last_date = trades_df['exit_date'].max()
        days = (last_date - first_date).days if last_date > first_date else 1
        trading_days = min(days, 252)
        avg_daily_trades = num_trades / trading_days if trading_days > 0 else num_trades

        avg_profit_pct = trades_df['profit'].mean() * 100 if num_trades > 0 else 0
        annual_return = avg_profit_pct * avg_daily_trades * 252 if trading_days > 0 else 0

        returns = trades_df['profit'].values
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        avg_holding_time = trades_df['holding_days'].mean() if 'holding_days' in trades_df.columns else 0

        total_invested = trades_df['entry_price'].sum()
        turnover_rate = (total_invested / self.initial_capital) * 100 if total_invested > 0 else 0

        trade_density = num_trades / trading_days if trading_days > 0 else 0

        return {
            'stock_code': stock_code,
            'strategy': strategy_type,
            'num_trades': num_trades,
            'total_profit': total_profit,
            'success_rate': success_rate,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_consecutive_losses': avg_consecutive_losses,
            'sharpe_ratio': sharpe_ratio,
            'avg_daily_trades': avg_daily_trades,
            'turnover_rate': turnover_rate,
            'avg_holding_time': avg_holding_time,
            'trade_density': trade_density,
            'trades': trades
        }

    def run(self, test_mode=False):
        stock_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        print(f"找到 {len(stock_files)} 个股票文件")

        if test_mode:
            print("测试模式：只运行前10个股票")
            stock_files = stock_files[:10]

        strategy_types = [
            '2天收盘卖', '2天开盘卖',
            '3天收盘卖', '3天开盘卖',
            '4天收盘卖', '4天开盘卖',
            '5天收盘卖', '5天开盘卖',
            '6天收盘卖', '6天开盘卖',
            '3%止盈', '4%止盈', '5%止盈', '6%止盈', '7%止盈', '8%止盈', '9%止盈'
        ]

        total_strategies = len(strategy_types)
        print(f"预计总计算量: {len(stock_files)} × {total_strategies} 种策略")
        print("开始运行回测...")

        all_results = []

        for strategy_type in strategy_types:
            print(f"\n{'='*60}")
            print(f"运行策略: {strategy_type}")
            print(f"{'='*60}")

            self.results = []

            for i, file_name in enumerate(stock_files):
                if i % 100 == 0:
                    print(f"处理进度: {i}/{len(stock_files)}")

                stock_code = file_name.split('#')[-1].replace('.txt', '')

                file_path = os.path.join(self.data_dir, file_name)
                if file_name in self.data_cache:
                    df = self.data_cache[file_name]
                else:
                    df = self.load_stock_data(file_path)
                    if df is not None:
                        self.data_cache[file_name] = df

                if df is not None and len(df) > 120:
                    result = self.run_strategy(df, stock_code, strategy_type)
                    self.results.append(result)

            self.print_summary(strategy_type)

            if self.results_file and self.results:
                results_df = pd.DataFrame(self.results)
                strategy_file = self.results_file.replace('.csv', f'_{strategy_type}.csv')
                results_df.to_csv(strategy_file, index=False, encoding='utf-8-sig')
                print(f"回测结果已保存到 {strategy_file}")

            all_results.extend(self.results)

        self.print_strategy_comparison()

        self.log_results()

        return all_results

    def log_results(self):
        buy_conditions = """1. 趋势条件: 知行多空线 <= 知行短期趋势线
2. J值条件: 昨日J值 <= 50 且 今日J值 <= 80
3. 涨幅条件: 今日涨幅 >= 4%
4. 成交量放量: 今日成交量 > 昨日成交量
5. 成交量3倍: 今日成交量 > 昨日成交量 * 3
6. 上影线条件: 上影线长度 / 实体长度 < 30%"""

        sell_conditions = """策略1: 持有1天后（第2天）以收盘价卖出
策略2: 持有1天后（第2天）以开盘价卖出
策略3: 持有2天后（第3天）以收盘价卖出
策略4: 持有2天后（第3天）以开盘价卖出
策略5: 持有3天后（第4天）以收盘价卖出
策略6: 持有3天后（第4天）以开盘价卖出
策略7: 持有4天后（第5天）以收盘价卖出
策略8: 持有4天后（第5天）以开盘价卖出
策略9: 持有5天后（第6天）以收盘价卖出
策略10: 持有5天后（第6天）以开盘价卖出
策略11: 买入后任何日期最高价涨幅>=3%则止盈卖出
策略12: 买入后任何日期最高价涨幅>=4%则止盈卖出
策略13: 买入后任何日期最高价涨幅>=5%则止盈卖出
策略14: 买入后任何日期最高价涨幅>=6%则止盈卖出
策略15: 买入后任何日期最高价涨幅>=7%则止盈卖出
策略16: 买入后任何日期最高价涨幅>=8%则止盈卖出
策略17: 买入后任何日期最高价涨幅>=9%则止盈卖出"""

        stop_loss = "止损价为发出买入信号当天的最低价"

        results_summary = ""
        for strategy_type in ['2天收盘卖', '2天开盘卖', '3%止盈', '4%止盈', '5%止盈']:
            strategy_file = self.results_file.replace('.csv', f'_{strategy_type}.csv')
            if os.path.exists(strategy_file):
                df = pd.read_csv(strategy_file)
                if not df.empty:
                    avg_success = df['success_rate'].mean()
                    avg_annual = df['annual_return'].mean()
                    results_summary += f"\n{strategy_type}: 成功率={avg_success:.2%}, 平均年化={avg_annual:.2f}%"

        log_backtest_result(
            "backtest_b2_strategy",
            results_summary,
            buy_conditions,
            sell_conditions,
            stop_loss,
            self.data_dir,
            "单股串行"
        )

    def print_summary(self, strategy_type):
        if not self.results:
            print("没有回测结果")
            return

        results_df = pd.DataFrame(self.results)

        total_stocks = len(results_df)
        traded_stocks = len(results_df[results_df['num_trades'] > 0])
        total_trades = results_df['num_trades'].sum()
        total_profit = results_df['total_profit'].sum()
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        profitable_stocks = len(results_df[results_df['total_profit'] > 0])
        profitable_stocks_ratio = profitable_stocks / total_stocks if total_stocks > 0 else 0
        avg_success_rate = results_df['success_rate'].mean()
        avg_annual_return = results_df['annual_return'].mean()
        max_annual_return = results_df['annual_return'].max() if not results_df.empty else 0
        max_drawdown = results_df['max_drawdown'].max() if not results_df.empty else 0
        avg_drawdown = results_df['avg_drawdown'].mean() if not results_df.empty else 0
        max_consecutive_losses = results_df['max_consecutive_losses'].max() if not results_df.empty else 0
        avg_consecutive_losses = results_df['avg_consecutive_losses'].mean() if not results_df.empty else 0
        avg_sharpe = results_df['sharpe_ratio'].mean() if not results_df.empty else 0
        avg_turnover = results_df['turnover_rate'].mean() if not results_df.empty else 0
        avg_holding = results_df['avg_holding_time'].mean() if not results_df.empty else 0
        avg_density = results_df['trade_density'].mean() if not results_df.empty else 0

        print(f"\n回测总结 ({strategy_type}):")
        print(f"总股票数: {total_stocks}")
        print(f"有交易的股票数: {traded_stocks}")
        print(f"总交易次数: {total_trades}")
        print(f"总盈利: {total_profit:.4f}")
        print(f"平均每笔交易盈利: {avg_profit_per_trade:.4f}")
        print(f"盈利股票比例: {profitable_stocks_ratio:.2%}")
        print(f"平均成功率: {avg_success_rate:.2%}")
        print(f"平均年化收益率: {avg_annual_return:.4f}%")
        print(f"最大年化收益率: {max_annual_return:.4f}%")
        print(f"最大回撤: {max_drawdown:.4f}%")
        print(f"平均回撤: {avg_drawdown:.4f}%")
        print(f"最大连续失败次数: {max_consecutive_losses}")
        print(f"平均连续失败次数: {avg_consecutive_losses:.2f}")
        print(f"夏普比率: {avg_sharpe:.4f}")
        print(f"平均每天交易次数: {results_df['avg_daily_trades'].mean():.4f}")
        print(f"平均换手率: {avg_turnover:.4f}%")
        print(f"平均持仓时间: {avg_holding:.2f}天")
        print(f"交易密度: {avg_density:.4f}")

    def print_strategy_comparison(self):
        print(f"\n{'='*120}")
        print("策略对比总结")
        print(f"{'='*120}")

        strategy_files = []
        for root, dirs, files in os.walk('/Users/lidongyang/Desktop/Qstrategy'):
            for f in files:
                if f.startswith('backtest_b2_results_') and f.endswith('.csv'):
                    strategy_files.append(os.path.join(root, f))

        if not strategy_files:
            return

        print(f"{'策略名称':<20} {'成功率':>8} {'平均年化':>12} {'最大年化':>12} {'最大回撤':>10} {'平均回撤':>10} {'夏普比率':>10} {'换手率':>10} {'持仓时间':>10} {'交易密度':>10}")
        print("-" * 120)

        for file_path in strategy_files:
            df = pd.read_csv(file_path)
            if df.empty:
                continue

            strategy_name = os.path.basename(file_path).replace('backtest_b2_results_', '').replace('.csv', '')

            avg_success_rate = df['success_rate'].mean()
            avg_annual_return = df['annual_return'].mean()
            max_annual_return = df['annual_return'].max()
            max_drawdown = df['max_drawdown'].max()
            avg_drawdown = df['avg_drawdown'].mean()
            avg_sharpe = df['sharpe_ratio'].mean()
            avg_turnover = df['turnover_rate'].mean()
            avg_holding = df['avg_holding_time'].mean()
            avg_density = df['trade_density'].mean()

            print(f"{strategy_name:<20} {avg_success_rate:>6.2%} {avg_annual_return:>12.4f} {max_annual_return:>12.4f} {max_drawdown:>10.2f}% {avg_drawdown:>10.2f}% {avg_sharpe:>10.4f} {avg_turnover:>10.4f} {avg_holding:>10.2f} {avg_density:>10.4f}")


if __name__ == "__main__":
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/20260225"
    results_file = "/Users/lidongyang/Desktop/Qstrategy/backtest_b2_results.csv"

    backtest = B2StrategyBacktest(data_dir, results_file)
    backtest.run(test_mode=False)
