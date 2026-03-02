import streamlit as st
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta

'''
记录每日挑选
'''

st.set_page_config(
    page_title='每日选大富翁记录',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

CONFIG_FILE = "/Users/lidongyang/Desktop/Qstrategy/config/daily_records.yaml"
DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data"

def load_config():
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        return {}

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        return True
    except Exception as e:
        st.error(f"保存失败: {e}")
        return False

def get_data_folders():
    folders = []
    for f in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, f)
        if os.path.isdir(path) and f.isdigit() and len(f) == 8:
            folders.append(f)
    return sorted(folders, reverse=True)

def get_latest_date_folder():
    folders = get_data_folders()
    if folders:
        return folders[0]
    return None

def load_stock_price(stock_code, date_folder):
    if not date_folder:
        return None

    if stock_code.startswith('6'):
        file_path = os.path.join(DATA_DIR, date_folder, f"SH#{stock_code}.txt")
    else:
        file_path = os.path.join(DATA_DIR, date_folder, f"SZ#{stock_code}.txt")

    if not os.path.exists(file_path):
        return None

    try:
        encodings = ['gbk', 'utf-8', 'gb2312', 'latin1']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep='\s+', header=None,
                               names=['日期', '开盘', '最高', '最低', '收盘', '成交量'])
                if len(df) > 0:
                    return df['收盘'].iloc[-1]
            except:
                continue
    except:
        pass
    return None

def calculate_change(record_date, stock_code):
    record_date_str = record_date.replace('-', '')
    latest_folder = get_latest_date_folder()

    if not latest_folder:
        return None, None

    if record_date_str > latest_folder:
        return None, None

    entry_price = None
    exit_price = None

    if stock_code.startswith('6'):
        entry_file = os.path.join(DATA_DIR, record_date_str, f"SH#{stock_code}.txt")
        exit_file = os.path.join(DATA_DIR, latest_folder, f"SH#{stock_code}.txt")
    else:
        entry_file = os.path.join(DATA_DIR, record_date_str, f"SZ#{stock_code}.txt")
        exit_file = os.path.join(DATA_DIR, latest_folder, f"SZ#{stock_code}.txt")

    encodings = ['gbk', 'utf-8', 'gb2312', 'latin1']

    for encoding in encodings:
        try:
            if os.path.exists(entry_file):
                df = pd.read_csv(entry_file, encoding=encoding, sep='\s+', header=None,
                               names=['日期', '开盘', '最高', '最低', '收盘', '成交量'])
                if len(df) > 0:
                    entry_price = df['收盘'].iloc[-1]
            break
        except:
            continue

    for encoding in encodings:
        try:
            if os.path.exists(exit_file):
                df = pd.read_csv(exit_file, encoding=encoding, sep='\s+', header=None,
                               names=['日期', '开盘', '最高', '最低', '收盘', '成交量'])
                if len(df) > 0:
                    exit_price = df['收盘'].iloc[-1]
            break
        except:
            continue

    if entry_price and exit_price:
        change_pct = (exit_price - entry_price) / entry_price * 100
        return round(change_pct, 2), latest_folder
    return None, latest_folder

st.title("📊 每日选大富翁记录")

config = load_config()

tab1, tab2 = st.tabs(["📈 查看记录", "➕ 添加记录"])

with tab1:
    dates = list(config.get('records', {}).keys())
    dates = sorted(dates, reverse=True)

    if dates:
        selected_date = st.selectbox("选择日期", dates, format_func=lambda x: x)

        if selected_date:
            st.subheader(f"📅 {selected_date} 选大富翁记录")

            strategies = ['B1', 'B2', 'B3', 'pin', 'brick']
            strategy_names = {'B1': 'B1策略', 'B2': 'B2策略', 'B3': 'B3策略', 'pin': 'Pin策略', 'brick': 'Brick策略'}

            for strategy in strategies:
                stocks = config['records'].get(selected_date, {}).get(strategy, [])
                if stocks:
                    st.markdown(f"### {strategy_names.get(strategy, strategy)}")

                    data = []
                    total_change = 0
                    valid_count = 0

                    for stock_code in stocks:
                        change_pct, _ = calculate_change(selected_date, stock_code)
                        if change_pct is not None:
                            status = "✅ 涨" if change_pct > 0 else "❌ 跌"
                            data.append({
                                "大富翁代码": stock_code,
                                "涨跌幅": f"{change_pct:.2f}%",
                                "状态": status
                            })
                            total_change += change_pct
                            valid_count += 1
                        else:
                            data.append({
                                "大富翁代码": stock_code,
                                "涨跌幅": "无数据",
                                "状态": "⚠️"
                            })

                    if data:
                        st.table(pd.DataFrame(data))

                    if valid_count > 0:
                        avg_change = total_change / valid_count
                        success_rate = len([d for d in data if d["涨跌幅"] != "无数据" and d["涨跌幅"].startswith("✅")]) / valid_count * 100
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("平均涨跌幅", f"{avg_change:.2f}%")
                        with col2:
                            st.metric("成功率", f"{success_rate:.1f}%")
                    else:
                        st.info("暂无涨跌幅数据")
                    st.divider()
    else:
        st.info("暂无记录")

with tab2:
    st.subheader("➕ 添加选大富翁记录")

    today = datetime.now().strftime('%Y-%m-%d')
    record_date = st.text_input("记录日期", value=today)

    if record_date:
        if 'records' not in config:
            config['records'] = {}

        if record_date not in config['records']:
            config['records'][record_date] = {}

        strategies = ['B1', 'B2', 'B3', 'pin', 'brick']
        strategy_names = {'B1': 'B1策略', 'B2': 'B2策略', 'B3': 'B3策略', 'pin': 'Pin策略', 'brick': 'Brick策略'}

        for strategy in strategies:
            st.markdown(f"### {strategy_names.get(strategy, strategy)}")
            current_stocks = config['records'][record_date].get(strategy, [])
            stocks_str = ', '.join(current_stocks) if current_stocks else ''
            new_stocks = st.text_area(f"{strategy}大富翁代码 (用逗号分隔)", value=stocks_str, key=f"add_{strategy}")

            stock_list = [s.strip() for s in new_stocks.split('\n') if s.strip()]
            config['records'][record_date][strategy] = stock_list
            st.divider()

        if st.button("💾 保存记录"):
            if save_config(config):
                st.success("保存成功！")
                st.rerun()

st.divider()
st.caption(f"配置文件: {CONFIG_FILE}")
