import os
import pandas as pd
import numpy as np


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


def brick_chart_indicator(df, ratio):
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
        (df['当日柱体长度'] >= df['前日柱体长度'] * ratio),
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

        if len(df) < 120:
            return None
        return df
    except:
        return None


def run_backtest(data_dir, initial_capital=1_000_000, max_positions=4, ratio=0.66):
    stock_data = {}
    daily_signals = {}
    fixed_investment = 100000

    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    for file in files:
        df = load_stock(os.path.join(data_dir, file))
        if df is None:
            continue

        df = calculate_trend(df)
        df = brick_chart_indicator(df, ratio)

        stock_data[file] = df

        for date in df.index[df["买入信号"] == 1]:
            if date in df.index:
                row = df.loc[date]
                if row['知行多空线'] <= row['知行短期趋势线']:
                    if row['CLOSE'] >= row['知行多空线']:
                        daily_signals.setdefault(date, []).append(file)

    all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}

    cash = float(initial_capital)
    positions = []
    equity_curve = []
    stopped = False
    trade_count = 0
    abnormal_count = 0

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

            if pd.isna(row["CLOSE"]) or row["CLOSE"] <= 0:
                new_positions.append(pos)
                continue

            open_p = row["OPEN"]
            high = row["HIGH"]
            low = row["LOW"]
            close = row["CLOSE"]

            df = stock_data[pos["stock"]]
            stock_idx = df.index.get_loc(current_date)
            stock_entry_idx = int(pos["entry_idx"])
            holding_days = stock_idx - stock_entry_idx

            entry_price = pos["entry_price"]

            open_change = (open_p - entry_price) / entry_price
            high_change = (high - entry_price) / entry_price
            low_change = (low - entry_price) / entry_price

            TP_RATE = 0.03
            SL_RATE = -0.02

            exit_flag = False
            exit_price = close

            if open_change >= TP_RATE:
                exit_price = open_p
                exit_flag = True
            elif open_change <= SL_RATE:
                exit_price = open_p
                exit_flag = True
            elif high_change >= TP_RATE:
                exit_price = entry_price * (1 + TP_RATE)
                exit_flag = True
            elif low_change <= SL_RATE:
                exit_price = entry_price * (1 + SL_RATE)
                exit_flag = True
            elif holding_days >= 3:
                exit_price = close
                exit_flag = True

            if exit_flag:
                gross = (exit_price - pos["entry_price"]) / pos["entry_price"]

                is_abnormal = False
                if np.isnan(gross) or np.isinf(gross):
                    is_abnormal = True
                elif abs(gross) > 2:
                    is_abnormal = True

                if is_abnormal:
                    cash += pos["invested"]
                    abnormal_count += 1
                else:
                    trade_count += 1
                    if not np.isnan(gross) and not np.isnan(pos["invested"]):
                        cash += pos["invested"] * (1 + gross)
            else:
                pos["current_price"] = close
                new_positions.append(pos)

        positions = new_positions

        total_equity = cash if not np.isnan(cash) else 0
        for pos in positions:
            price_ratio = pos["current_price"] / pos["entry_price"]
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

        if current_date not in daily_signals:
            continue

        available_slots = max_positions - len(positions)
        if available_slots <= 0:
            continue

        signals_today = daily_signals[current_date]

        existing_stocks = {pos["stock"] for pos in positions}
        available_signals = [s for s in signals_today if s not in existing_stocks]

        if not available_signals:
            continue

        if cash <= 0 or np.isnan(cash) or np.isinf(cash):
            continue

        num_to_buy = min(len(available_signals), available_slots)
        fixed_invested = fixed_investment

        count = 0
        for stock in available_signals:
            if count >= available_slots:
                break
            if cash < fixed_invested:
                break

            df = stock_data[stock]
            idx = df.index.get_loc(current_date)

            if idx + 1 >= len(df):
                continue

            today_close = df.iloc[idx]["CLOSE"]
            entry_price = df.iloc[idx + 1]["OPEN"]

            if entry_price <= 0 or np.isnan(entry_price):
                continue
            if today_close <= 0 or np.isnan(today_close):
                continue

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

    total_years = len(equity_curve) / 252
    final_multiple = equity_curve[-1] / initial_capital

    if total_years <= 0 or final_multiple <= 0:
        CAGR = -1.0
    else:
        CAGR = final_multiple ** (1 / total_years) - 1

    running_max = np.maximum.accumulate(equity_curve)
    running_max_safe = np.where(running_max == 0, 1, running_max)
    drawdowns = (equity_curve - running_max_safe) / running_max_safe
    max_dd = np.min(drawdowns)

    sharpe = 0
    if np.std(daily_returns) > 0 and len(daily_returns) > 0:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

    return {
        "ratio": ratio,
        "final_multiple": final_multiple,
        "CAGR": CAGR,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "trade_count": trade_count,
        "abnormal_count": abnormal_count
    }


if __name__ == "__main__":
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/backtest_data"

    ratios = [0.66, 1, 1.5, 2]

    print("=" * 80)
    print(f"{'柱体比例':<12} {'最终倍数':<12} {'年化收益':<12} {'最大回撤':<12} {'夏普比率':<12} {'交易笔数':<10}")
    print("=" * 80)

    for ratio in ratios:
        result = run_backtest(data_dir, ratio=ratio)
        print(f"{ratio:<12} {result['final_multiple']:<12.4f} {result['CAGR']:<12.4f} {result['max_dd']:<12.4f} {result['sharpe']:<12.4f} {result['trade_count']:<10}")

    print("=" * 80)
