import streamlit as st
import pandas as pd
import yaml
import json
import os
from datetime import datetime

st.set_page_config(
    page_title='持仓管理面板',
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

CONFIG_FILE = "/Users/lidongyang/Desktop/Qstrategy/config/holding.yaml"

if 'holding_loaded' not in st.session_state:
    st.session_state.holding_loaded = False
    st.session_state.hold_edit_idx = None
    st.session_state.watch_edit_idx = None

def load_config():
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"加载配置失败: {e}")
        return None

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        return True
    except Exception as e:
        st.error(f"保存配置失败: {e}")
        return False

def format_buy_date(dates):
    if isinstance(dates, list):
        return ', '.join(str(d) for d in dates)
    return str(dates) if dates else ''

st.title("📊 持仓管理面板")

config = load_config()
if config is None:
    st.stop()

tab1, tab2, tab3 = st.tabs(["📈 当前持仓", "👀 关注大富翁", "➕ 添加/编辑"])

with tab1:
    st.subheader("当前持仓列表")
    hold_stocks = config.get('hold_stocks', [])

    if hold_stocks:
        col_display = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 2])
        with col_display[0]:
            st.write("**大富翁代码**")
        with col_display[1]:
            st.write("**大富翁名称**")
        with col_display[2]:
            st.write("**类型**")
        with col_display[3]:
            st.write("**止损价**")
        with col_display[4]:
            st.write("**止盈价**")
        with col_display[5]:
            st.write("**持仓天数**")
        with col_display[6]:
            st.write("**持仓占比**")
        with col_display[7]:
            st.write("**买入日期**")
        with col_display[8]:
            st.write("**操作**")

        for idx, stock in enumerate(hold_stocks):
            col_row = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 2])
            with col_row[0]:
                st.write(stock.get('stock_code', ''))
            with col_row[1]:
                st.write(stock.get('stock_name', ''))
            with col_row[2]:
                st.write(stock.get('type', ''))
            with col_row[3]:
                st.write(stock.get('stop_loss_price', ''))
            with col_row[4]:
                st.write(stock.get('take_profit_price', ''))
            with col_row[5]:
                st.write(stock.get('hold_days', ''))
            with col_row[6]:
                st.write(stock.get('position_ratio', ''))
            with col_row[7]:
                st.write(format_buy_date(stock.get('buy_date', [])))
            with col_row[8]:
                if st.button(f"🗑️ 删除", key=f"del_hold_{idx}"):
                    hold_stocks.pop(idx)
                    config['hold_stocks'] = hold_stocks
                    if save_config(config):
                        st.rerun()

        st.divider()

        st.subheader("✏️ 编辑持仓")

        options = [f"{s.get('stock_code', '')} - {s.get('stock_name', '')}" for s in hold_stocks]
        if not options:
            options = ["无持仓"]

        edit_idx = st.selectbox(
            "选择要编辑的持仓",
            range(len(hold_stocks)) if hold_stocks else [None],
            format_func=lambda x: options[x] if x is not None and 0 <= x < len(options) else "无持仓",
            key="hold_edit_selectbox"
        )

        if edit_idx is not None and 0 <= edit_idx < len(hold_stocks):
            stock = hold_stocks[edit_idx]

            col1, col2, col3 = st.columns(3)
            with col1:
                new_stock_code = st.text_input("大富翁代码", stock.get('stock_code', ''), key=f"hold_edit_code_{edit_idx}")
                new_stock_name = st.text_input("大富翁名称", stock.get('stock_name', ''), key=f"hold_edit_name_{edit_idx}")
                new_type = st.text_input("类型", stock.get('type', ''), key=f"hold_edit_type_{edit_idx}")
            with col2:
                new_stop_loss = st.number_input("止损价", value=float(stock.get('stop_loss_price', 0)), step=0.01, key=f"hold_edit_sl_{edit_idx}")
                new_take_profit = st.number_input("止盈价", value=float(stock.get('take_profit_price', 0)), step=0.01, key=f"hold_edit_tp_{edit_idx}")
                new_hold_days = st.number_input("持仓天数", value=int(stock.get('hold_days', 0)), step=1, key=f"hold_edit_hd_{edit_idx}")
            with col3:
                new_position_ratio = st.text_input("持仓占比", stock.get('position_ratio', ''), key=f"hold_edit_pr_{edit_idx}")
                new_note = st.text_area("备注", stock.get('note', ''), key=f"hold_edit_note_{edit_idx}")

            buy_date_str = st.text_area(
                "买入日期 (用逗号分隔)",
                format_buy_date(stock.get('buy_date', [])),
                key=f"hold_edit_bd_{edit_idx}"
            )

            if st.button("💾 保存修改", key=f"hold_save_{edit_idx}"):
                buy_dates = [d.strip() for d in buy_date_str.split(',') if d.strip()]
                hold_stocks[edit_idx] = {
                    'stock_code': new_stock_code,
                    'stock_name': new_stock_name,
                    'type': new_type,
                    'stop_loss_price': new_stop_loss,
                    'take_profit_price': new_take_profit,
                    'hold_days': new_hold_days,
                    'position_ratio': new_position_ratio,
                    'buy_date': buy_dates,
                    'note': new_note
                }
                config['hold_stocks'] = hold_stocks
                if save_config(config):
                    st.success("保存成功！")
                    st.rerun()
    else:
        st.info("暂无持仓")

with tab2:
    st.subheader("关注大富翁列表")
    watch_stocks = config.get('watch_stocks', [])

    if watch_stocks:
        col_display = st.columns([1, 1, 1, 1, 1, 2])
        with col_display[0]:
            st.write("**大富翁代码**")
        with col_display[1]:
            st.write("**大富翁名称**")
        with col_display[2]:
            st.write("**类型**")
        with col_display[3]:
            st.write("**止损价**")
        with col_display[4]:
            st.write("**持仓天数**")
        with col_display[5]:
            st.write("**操作**")

        for idx, stock in enumerate(watch_stocks):
            col_row = st.columns([1, 1, 1, 1, 1, 2])
            with col_row[0]:
                st.write(stock.get('stock_code', ''))
            with col_row[1]:
                st.write(stock.get('stock_name', ''))
            with col_row[2]:
                st.write(stock.get('type', ''))
            with col_row[3]:
                st.write(stock.get('stop_loss_price', ''))
            with col_row[4]:
                st.write(stock.get('hold_days', ''))
            with col_row[5]:
                if st.button(f"🗑️ 删除", key=f"del_watch_{idx}"):
                    watch_stocks.pop(idx)
                    config['watch_stocks'] = watch_stocks
                    if save_config(config):
                        st.rerun()

        st.divider()

        st.subheader("✏️ 编辑关注")

        watch_options = [f"{s.get('stock_code', '')} - {s.get('stock_name', '')}" for s in watch_stocks]
        if not watch_options:
            watch_options = ["无关注"]

        watch_edit_idx = st.selectbox(
            "选择要编辑的关注",
            range(len(watch_stocks)) if watch_stocks else [None],
            format_func=lambda x: watch_options[x] if x is not None and 0 <= x < len(watch_options) else "无关注",
            key="watch_edit_selectbox"
        )

        if watch_edit_idx is not None and 0 <= watch_edit_idx < len(watch_stocks):
            stock = watch_stocks[watch_edit_idx]

            col1, col2 = st.columns(2)
            with col1:
                new_stock_code = st.text_input("大富翁代码", stock.get('stock_code', ''), key=f"watch_edit_code_{watch_edit_idx}")
                new_stock_name = st.text_input("大富翁名称", stock.get('stock_name', ''), key=f"watch_edit_name_{watch_edit_idx}")
                new_type = st.text_input("类型", stock.get('type', ''), key=f"watch_edit_type_{watch_edit_idx}")
            with col2:
                new_stop_loss = st.number_input("止损价", value=float(stock.get('stop_loss_price', 0)), step=0.01, key=f"watch_edit_sl_{watch_edit_idx}")
                new_hold_days = st.number_input("持仓天数", value=int(stock.get('hold_days', 0)), step=1, key=f"watch_edit_hd_{watch_edit_idx}")

            new_note = st.text_area("备注", stock.get('note', ''), key=f"watch_edit_note_{watch_edit_idx}")

            if st.button("💾 保存关注修改", key=f"watch_save_{watch_edit_idx}"):
                watch_stocks[watch_edit_idx] = {
                    'stock_code': new_stock_code,
                    'stock_name': new_stock_name,
                    'type': new_type,
                    'stop_loss_price': new_stop_loss,
                    'hold_days': new_hold_days,
                    'note': new_note
                }
                config['watch_stocks'] = watch_stocks
                if save_config(config):
                    st.success("保存成功！")
                    st.rerun()
    else:
        st.info("暂无关注大富翁")

with tab3:
    st.subheader("➕ 添加新持仓")

    add_type = st.radio("添加类型", ["持仓大富翁", "关注大富翁"], horizontal=True, key="add_type_radio")

    if add_type == "持仓大富翁":
        col1, col2, col3 = st.columns(3)
        with col1:
            new_stock_code = st.text_input("大富翁代码", key="tab3_add_code")
            new_stock_name = st.text_input("大富翁名称", key="tab3_add_name")
            new_type = st.text_input("类型", key="tab3_add_type")
        with col2:
            new_stop_loss = st.number_input("止损价", value=0.0, step=0.01, key="tab3_add_sl")
            new_take_profit = st.number_input("止盈价", value=0.0, step=0.01, key="tab3_add_tp")
            new_hold_days = st.number_input("持仓天数", value=0, step=1, key="tab3_add_hd")
        with col3:
            new_position_ratio = st.text_input("持仓占比", key="tab3_add_pr")
            new_note = st.text_area("备注", key="tab3_add_note")

        buy_date_str = st.text_input("买入日期 (用逗号分隔，多个日期)", key="tab3_add_bd")

        if st.button("➕ 添加持仓", key="tab3_add_hold_btn"):
            if not new_stock_code:
                st.error("大富翁代码不能为空")
            else:
                buy_dates = [d.strip() for d in buy_date_str.split(',') if d.strip()]
                new_stock = {
                    'stock_code': new_stock_code,
                    'stock_name': new_stock_name,
                    'type': new_type,
                    'stop_loss_price': new_stop_loss,
                    'take_profit_price': new_take_profit,
                    'hold_days': new_hold_days,
                    'position_ratio': new_position_ratio,
                    'buy_date': buy_dates,
                    'note': new_note
                }
                if 'hold_stocks' not in config:
                    config['hold_stocks'] = []
                config['hold_stocks'].append(new_stock)
                if save_config(config):
                    st.success("添加成功！")
                    st.rerun()

    else:
        col1, col2 = st.columns(2)
        with col1:
            new_stock_code = st.text_input("大富翁代码", key="tab3_watch_add_code")
            new_stock_name = st.text_input("大富翁名称", key="tab3_watch_add_name")
            new_type = st.text_input("类型", key="tab3_watch_add_type")
        with col2:
            new_stop_loss = st.number_input("止损价", value=0.0, step=0.01, key="tab3_watch_add_sl")
            new_hold_days = st.number_input("持仓天数", value=0, step=1, key="tab3_watch_add_hd")

        new_note = st.text_area("备注", key="tab3_watch_add_note")

        if st.button("➕ 添加关注", key="tab3_add_watch_btn"):
            if not new_stock_code:
                st.error("大富翁代码不能为空")
            else:
                new_stock = {
                    'stock_code': new_stock_code,
                    'stock_name': new_stock_name,
                    'type': new_type,
                    'stop_loss_price': new_stop_loss,
                    'hold_days': new_hold_days,
                    'note': new_note
                }
                if 'watch_stocks' not in config:
                    config['watch_stocks'] = []
                config['watch_stocks'].append(new_stock)
                if save_config(config):
                    st.success("添加成功！")
                    st.rerun()

st.divider()
st.caption(f"配置文件: {CONFIG_FILE}")
