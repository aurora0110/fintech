import streamlit as st
import pandas as pd
import os
import yaml

CONFIG_FILE = "c:/Users/lidon/Desktop/Qstrategy/config/holding.yaml"

def load_holding_data():
    """加载持仓数据"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"读取配置文件失败: {str(e)}")
        return {}

def save_holding_data(data):
    """保存持仓数据"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True)
        return True
    except Exception as e:
        print(f"写入配置文件失败: {str(e)}")
        return False

st.set_page_config(page_title="持仓管理 Dashboard", layout="wide")
st.title("📊 持仓管理 Dashboard")

data = load_holding_data()

if data:
    st.success("✅ 配置文件加载成功")
    st.json(data)
else:
    st.warning("⚠️ 配置文件为空或加载失败")

st.caption(f"配置文件: {CONFIG_FILE}")
