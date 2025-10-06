import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.subplots as sp
import plotly.graph_objs as go
from pathlib import Path
import getData
from typing import List, Dict, Any, Tuple, Optional
import os

def int_to_chinese_num(num):
    '''
    转换整数为中文数字
    '''
    if not isinstance(num, int):
        return "请输入整数"

    # 处理负数情况
    sign = ""
    if num < 0:
        sign = "-"
        num = abs(num)

    if num == 0:
        return "零"

    digit_map = ["", "十", "百", "千"]
    unit_map = ["", "万", "亿", "兆"]  # 可扩展更高单位
    num_str = str(num)
    length = len(num_str)
    result = []

    # 每4位一组（中文数字以万为单位）
    for i in range(0, length, 4):
        segment = num_str[max(0, length - i - 4): length - i]
        segment_len = len(segment)
        segment_str = ""

        # 处理每一段（千、百、十、个位）
        for j in range(segment_len):
            digit = int(segment[j])
            if digit == 0:
                continue  # 零不单独显示，除非在中间（如 1001 → 一千零一）
            # 添加数字和单位（如 "3" + "百" → "3百"）
            segment_str += str(digit) + digit_map[segment_len - j - 1]

        # 添加段单位（万、亿等）
        if segment_str:  # 如果该段不为空
            segment_str += unit_map[i // 4]
        result.append(segment_str)

    # 拼接所有段（从高到低）
    chinese_num = "".join(reversed(result))

    # 处理连续的零（如 "1001" → "一千零一"）
    chinese_num = chinese_num.replace("零零", "零").strip("零")
    
    # 加上符号（如果是负数）
    return sign + chinese_num if chinese_num else "零"

def to_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """将指定列转为数值，非法值置为 NaN。"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# 计算技术指标
class StockAnalyzer:
    '''
    计算技术指标
    '''
    def __init__(self, ticker, file_path, start_date=None, end_date=None, kdj_days = 9, kdj_m1=3, kdj_m2=3, windows = [20, 30, 60, 120]):
        self.ticker = ticker
        self.file_path = file_path
        self.end_date = pd.to_datetime(end_date or datetime.now().strftime('%Y-%m-%d'))
        self.start_date = pd.to_datetime(start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        self.stock_data = self.get_data()
        self.windows = windows # 均线窗口

        self.data_ma = {}
        self.data_bbi = {}
        self.data_price = {}
        self.data_macd = {}
        self.data_kdj = {}
        self.data_shakeout = {}

        self.kdj_days = kdj_days
        self.kdj_m1 = kdj_m1
        self.kdj_m2 = kdj_m2
    
    def get_data(self):
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件{path}不存在")
        try:
            data = pd.read_csv(self.file_path, engine=None)
            data['日期'] = pd.to_datetime(data['日期'])
            start_date = pd.to_datetime(self.start_date)
            end_date = pd.to_datetime(self.end_date)
            data = data[(data['日期'] >= start_date) & (data['日期'] <= end_date)]
            return data
        except Exception as e:
            print(f"读取文件{path}失败，错误信息：{e}")
        return pd.read_csv(path, encoding="utf-8")

    def calculate_all_indicators(self):
        self.calculate_moving_averages()
        self.calculate_bbi()
        self.calculate_price()
        self.calculate_macd()
        self.calculate_kdj()
        self.calculate_shakeout()

    def calculate_kdj(self):
        df = self.stock_data.copy()
        # 使用 datetime 比较
        df['日期'] = pd.to_datetime(df['日期'])
        df = df[(df['日期'] >= self.start_date) & (df['日期'] <= self.end_date)]
        df['high_kdj'] = df['最高'].rolling(window = self.kdj_days, min_periods = 1).max()
        df['low_kdj'] = df['最低'].rolling(window = self.kdj_days, min_periods = 1).min()
        df['RSV'] = (df['收盘'] - df['low_kdj']) / (df['high_kdj'] - df['low_kdj']) * 100
        df['K'] = df['RSV'].ewm(alpha=1/self.kdj_m1, adjust=False).mean()
        df['D'] = df['K'].ewm(alpha=1/self.kdj_m2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        data_list = df[['日期','J', '收盘', '最低', '最高']].to_numpy().tolist()
        # 定义区间阈值和筛选函数
        thresholds = {
            'low_0': lambda j: j <= 0,
            'low_5': lambda j: j <= -5,
            'low_10': lambda j: j <= -10,
            'low_15': lambda j: j <= -15,
            'low_20': lambda j: j <= -20,
            'low_25': lambda j: j <= -25,
            'high_80': lambda j: j >= 80,
            'high_90': lambda j: j >= 90,
            'high_100': lambda j: j >= 100,
            'high_110': lambda j: j >= 110,
            'high_120': lambda j: j >= 120,
        }
        fast_down_j_label = False
        # 初始化结果字典
        ret_kdj_dict = {key: [] for key in thresholds}

        # 遍历每一行数据
        for row in data_list:
            date, j_val, close, high, low = row[0], float(row[1]), row[2], row[3], row[4]
            row_data = [date, round(j_val, 3), close, high, low]

            for key, condition in thresholds.items():
                if condition(j_val):
                    ret_kdj_dict[key].append(row_data)

        # 筛选三天内J快速下降
        if data_list[-1][1] < 10 and data_list[-3][1] > 80:
            fast_down_j_label = True
        # 整理为列表（按 thresholds 的顺序）
        ret_kdj = [ret_kdj_dict[key] for key in thresholds]

        # 返回结构化结果
        return {
            'K': df['K'],
            'D': df['D'],
            'J': df['J'],
            'ret_kdj': ret_kdj,
            'fast_down_j_label': fast_down_j_label
        }
    # 计算RSI
    def calculate_rsi(self):
        df = self.stock_data.copy()
        # 使用 datetime 比较
        df['日期'] = pd.to_datetime(df['日期'])
        df = df[(df['日期'] >= self.start_date) & (df['日期'] <= self.end_date)]

        delta = df['收盘'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df['RSI'] = rsi
        return df
    # 计算RSI动量
    def calculate_rsi_momentum(mom_len: int = 5, method: str = "roc") -> pd.Series:
        """
        RSI 动量：对 RSI 再做动量（ROC 或差分）
        method: 'roc' -> (RSI/RSI_{t-m}-1); 'diff' -> RSI - RSI_{t-m}
        """
        r = StockAnalyzer.calculate_rsi()
        if method == "roc":
            out = r / r.shift(mom_len) - 1.0
        elif method == "diff":
            out = r - r.shift(mom_len)
        else:
            raise ValueError("method must be 'roc' or 'diff'")
        return out
    # 计算价格动量
    def momentum(series: pd.Series, n: int = 10, method: str = "roc") -> pd.Series:
        """
        价格动量：
        method: 
        - 'roc'  -> (P / P_{t-n} - 1)   百分比动量
        - 'diff' -> (P - P_{t-n})       差值动量
        """
        s = pd.to_numeric(series, errors="coerce")
        if method == "roc":
            return s / s.shift(n) - 1.0
        elif method == "diff":
            return s - s.shift(n)
        else:
            raise ValueError("method must be 'roc' or 'diff'")

    def calculate_moving_averages(self):
        windows = self.windows
        result = {}
        for window in windows:
            result[f'MA_{window}'] = self.stock_data['收盘'].rolling(window=window).mean()
        result['date'] = self.stock_data['日期']
        self.data_ma = pd.DataFrame(result)
        return self.data_ma

    def calculate_bbi(self):
        data = self.stock_data.copy()
        data['avg_price'] = (data['收盘'] + data['最高'] + data['最低']) / 3
        data['ma3'] = data['avg_price'].rolling(3).mean()
        data['ma6'] = data['avg_price'].rolling(6).mean()
        data['ma12'] = data['avg_price'].rolling(12).mean()
        data['ma24'] = data['avg_price'].rolling(24).mean()
        data['bbi'] = (data['ma3'] + data['ma6'] + data['ma12'] + data['ma24']) / 4
        return {'date': data['日期'], 'bbi': data['bbi']}

    def calculate_price(self):
        data = self.stock_data.copy()
        data['avg_price'] = (data['收盘'] + data['最高'] + data['最低']) / 3
        data['close_price'] = data['收盘']
        data['open_price'] = data['开盘']
        return {'date': data['日期'], 'avg_price': data['avg_price'], 'close_price': data['close_price'], 'open_price':data['open_price']}

    def calculate_macd(self, fast=12, slow=26, signal=9):
        df = self.stock_data.copy()
        ema_fast = df['收盘'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['收盘'].ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd = (dif - dea) * 2
        return {'date': df['日期'], 'DIF': dif, 'DEA': dea, 'MACD': macd}

    def calculate_shakeout(self):
        '''
        | 指标名称       | 条件                  | 含义解释              |
        | ---------- | ------------------- | ----------------- |
        | **四线归零买**  | 短期、中期、中长期、长期都 <= 6  | 四个指标都超卖，极端低点，可能反弹 |
        | **白线下20买** | 短期 <= 20 且 长期 >= 60 | 短期超卖，长期仍强，可能是回调买点 |
        | **白穿红线买**  | 短期上穿长期 且 长期 < 20    | 动量金叉且低位，反转可能性大    |
        | **白穿黄线买**  | 短期上穿中期 且 中期 < 30    | 动量拐头，初步反弹信号       |
        '''
        N1 = 3 # 短期指标
        N2 = 21 # 长期指标
        df = self.stock_data.copy()
        #df = df[(df['日期'] >= str(self.start_date)) & (df['日期'] <= str(self.end_date))]


        # 计算函数
        def momentum_indicator(C, L, n):
            return 100 * (C - L.rolling(n).min()) / (C.rolling(n).max() - L.rolling(n).min())

        # 计算短中长期指标
        df['短期'] = momentum_indicator(df['收盘'], df['最低'], N1)
        df['中期'] = momentum_indicator(df['收盘'], df['最低'], 10)
        df['中长期'] = momentum_indicator(df['收盘'], df['最低'], 20)
        df['长期'] = momentum_indicator(df['收盘'], df['最低'], N2)

        # 买点条件
        df['四线归零买'] = np.where(
            (df['短期'] <= 6) & (df['中期'] <= 6) & (df['中长期'] <= 6) & (df['长期'] <= 6),
            1, 0)

        df['白线下20买'] = np.where(
            (df['短期'] <= 20) & (df['长期'] >= 80),
            1, 0)

        df['白线下20买_小V'] = np.where(
            (df['长期'] - df['短期'] >= 40) & (df['长期'] >= 60),
            1, 0)

        # 白穿红线买（金叉）
        df['白穿红线买'] = np.where(
            (df['短期'] > df['长期']) & (df['短期'].shift(1) <= df['长期'].shift(1)) & (df['长期'] < 20),
            1, 0)

        # 白穿黄线买（金叉）
        df['白穿黄线买'] = np.where(
            (df['短期'] > df['中期']) & (df['短期'].shift(1) <= df['中期'].shift(1)) & (df['中期'] < 30),
            1, 0)

        # 输出最后几行查看
        #print(df[['短期', '中期', '中长期', '长期', '四线归零买', '白线下20买', '白穿红线买', '白穿黄线买']].tail())

        return df
    # 计算ATR
    def rolling_atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
        """
        计算 ATR% = ATR / Close，使用简单均值（非Wilder），反映价格波动幅度。
        TR = max(high-low, |high-prev_close|, |low-prev_close|)
        """
        h = df["最高"].values
        l = df["最低"].values
        c = df["收盘"].values
        pc = np.r_[np.nan, c[:-1]]
        tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
        atr = pd.Series(tr).rolling(n, min_periods=n).mean()
        return atr / df["收盘"]
    # 计算布林带贷款
    def boll_bandwidth(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
        """
        布林带带宽： (Upper - Lower) / Middle, 其中 Upper/Lower = MA ± k*STD
        """
        mid = df["收盘"].rolling(n, min_periods=n).mean()
        std = df["收盘"].rolling(n, min_periods=n).std()
        upper = mid + k * std
        lower = mid - k * std
        return (upper - lower) / mid
    # 计算zigzag
    def percent_zigzag(high: np.ndarray,
                    low: np.ndarray,
                    pct: float = 0.05) -> List[Dict[str, Any]]:
        """
        百分比 ZigZag（使用 high/low 捕捉极值）。
        当从当前极值向相反方向反转幅度 >= pct 即确认拐点。
        返回按时间排序的 pivot 列表：
        [{'idx': i, 'type': 'peak'/'trough', 'price': price}, ...]
        """
        n = len(high)
        if n == 0:
            return []
        pivots: List[Dict[str, Any]] = []
        direction = 0
        up_max, up_i = high[0], 0
        dn_min, dn_i = low[0], 0

        for i in range(1, n):
            if high[i] > up_max:
                up_max, up_i = high[i], i
            if low[i] < dn_min:
                dn_min, dn_i = low[i], i

            if direction >= 0:
                if (up_max - low[i]) / max(up_max, 1e-12) >= pct:
                    pivots.append({"idx": int(up_i), "type": "peak", "price": float(up_max)})
                    direction = -1
                    dn_min, dn_i = low[i], i
            if direction <= 0:
                if (high[i] - dn_min) / max(dn_min, 1e-12) >= pct:
                    pivots.append({"idx": int(dn_i), "type": "trough", "price": float(dn_min)})
                    direction = 1
                    up_max, up_i = high[i], i

        pivots.sort(key=lambda x: x["idx"])
        # 清理连续同类，保留更“极端”的
        cleaned = []
        for p in pivots:
            if not cleaned:
                cleaned.append(p)
            else:
                if cleaned[-1]["type"] != p["type"]:
                    cleaned.append(p)
                else:
                    if p["type"] == "peak":
                        if p["price"] >= cleaned[-1]["price"]:
                            cleaned[-1] = p
                    else:
                        if p["price"] <= cleaned[-1]["price"]:
                            cleaned[-1] = p
        return cleaned
    # 计算VCP压缩
    def calculate_vcp(self, col_date: str = "date", col_open: str = "开盘",col_high: str = "最高",col_low: str = "最低",col_close: str = "收盘",col_volume: str = "成交量",
        # ZigZag & 收缩判定
        zigzag_pct: float = 0.06,
        min_contractions: int = 2,
        max_contractions: int = 4,
        min_drop: float = 0.06,           # 每段回撤至少 6%
        tighten_ratio_max: float = 0.85,  # 收缩递减：C2 <= C1*0.85
        # 压缩（基底期）与干涸
        bb_len: int = 20,
        atr_len: int = 14,
        bw_low_pct: float = 0.35,         # 布林带带宽历史低分位阈值
        atr_low_pct: float = 0.35,        # ATR% 历史低分位阈值
        dryup_days: int = 5,
        vol_ma_len: int = 50,
        dryup_thresh: float = 0.60,       # 近 dryup_days 均量 / MA50 < 0.6
        # 枢轴/突破/再扩张
        near_pivot_tol: float = 0.03,     # 现价在枢轴下方 ≤3% 视为“接近枢轴”
        breakout_vol_mult: float = 1.5,   # 突破放量阈值（相对 MA50）
        reexpansion_win: int = 3,         # 突破附近最近 N 天
        reexpansion_bw_increase: float = 0.20,  # 近N天 带宽 >= 基底中位数*1.20
        reexpansion_atr_increase: float = 0.10, # 近N天 ATR% >= 基底中位数*1.10
        # 范围限制
        window: int = 220                 # 只看最近 N 根，便于分位计算
    ):
        """
        读取 CSV → 计算 VCP “压缩”评分。
        得分构成（总分=100）：
        1) 基底压缩质量：布林带 20 + ATR% 10 = 30
        2) 连续收缩结构：20
        3) 量能枯竭：15
        4) 接近枢轴：10
        5) 突破放量：10
        6) 再扩张（突破附近带宽/ATR%抬升）：15
        """
        df = self.stock_data.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        rename_map = {}
        for k_std, v_user in {
            "日期": col_date, "开盘": col_open, "最高": col_high,
            "最低": col_low, "收盘": col_close, "成交量": col_volume
        }.items():
            if v_user.lower() in df.columns:
                rename_map[v_user.lower()] = k_std
        df = df.rename(columns=rename_map)

        need = {"开盘", "最高", "最低", "收盘", "成交量"}
        if not need.issubset(df.columns):
            return {"ok": False, "message": f"CSV 需包含列：{need}，实际列：{set(df.columns)}"}

        if "日期" in df.columns:
            try:
                df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
            except Exception:
                pass

        for c in ["开盘", "最高", "最低", "收盘", "成交量"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[ "最高", "最低", "收盘", "成交量"]).reset_index(drop=True)

        if len(df) < max(bb_len + 5, atr_len + 5, vol_ma_len + 5, 80):
            return {"ok": False, "message": "数据长度不足，无法稳定计算。"}

        # 仅截取最近 window 根
        if window and len(df) > window:
            df = df.iloc[-window:].reset_index(drop=True)

        # —— 指标计算 ——
        df["atr_pct"] = StockAnalyzer.rolling_atr_pct(df, n=atr_len)
        df["bb_bw"] = StockAnalyzer.boll_bandwidth(df, n=bb_len, k=2.0)
        df["vol_ma"] = df["交易量"].rolling(vol_ma_len, min_periods=1).mean()

        # —— ZigZag 收缩段（peak->trough） ——
        pivots = StockAnalyzer.percent_zigzag(df["最高"].values, df["最低"].values, pct=zigzag_pct)
        contractions: List[Dict[str, Any]] = []
        for i in range(len(pivots) - 1):
            a, b = pivots[i], pivots[i + 1]
            if a["type"] == "peak" and b["type"] == "trough" and b["idx"] > a["idx"]:
                drop = (a["price"] - b["price"]) / max(a["price"], 1e-12)
                contractions.append({
                    "peak_idx": a["idx"], "trough_idx": b["idx"],
                    "peak": float(a["price"]), "trough": float(b["price"]),
                    "drop_pct": float(drop)
                })
        if len(contractions) < min_contractions:
            return {"ok": False, "message": "ZigZag 收缩段不足。", "pivots": pivots}

        m = min(max_contractions, len(contractions))
        seg = contractions[-m:]
        drops = [c["drop_pct"] for c in seg]

        base_start = seg[0]["peak_idx"]
        base_end   = seg[-1]["trough_idx"]
        if base_end <= base_start:
            return {"ok": False, "message": "基底区间非法。"}

        # —— 1) 连续收缩结构 —— 
        contraction_ok = True
        if any(d < min_drop for d in drops):
            contraction_ok = False
        for j in range(len(drops) - 1):
            if drops[j + 1] > drops[j] * tighten_ratio_max:
                contraction_ok = False
                break

        # —— 2) 基底压缩质量（低分位）——
        base_bw = df.loc[base_start:base_end, "bb_bw"].dropna()
        hist_bw = df["bb_bw"].dropna()
        base_atr = df.loc[base_start:base_end, "atr_pct"].dropna()
        hist_atr = df["atr_pct"].dropna()

        bw_ok = atr_ok = False
        bw_med = atr_med = np.nan
        bw_q = atr_q = np.nan
        if len(base_bw) and len(hist_bw):
            bw_med = float(np.nanmedian(base_bw.values))
            bw_q   = float(np.nanpercentile(hist_bw.values, bw_low_pct * 100))
            bw_ok  = (bw_med <= bw_q)
        if len(base_atr) and len(hist_atr):
            atr_med = float(np.nanmedian(base_atr.values))
            atr_q   = float(np.nanpercentile(hist_atr.values, atr_low_pct * 100))
            atr_ok  = (atr_med <= atr_q)

        # —— 3) 量能枯竭 —— 
        tail = df.loc[max(base_end - dryup_days + 1, 0):base_end]
        vol_ma_ref = float(df.loc[:base_end, "vol_ma"].iloc[-1])
        dryup_ratio = float(tail["volume"].mean() / max(vol_ma_ref, 1e-9))
        dryup_ok = (dryup_ratio < dryup_thresh)

        # —— 4) 枢轴/接近枢轴 —— 
        pivot_idx = seg[-1]["peak_idx"]
        pivot_price = float(df.loc[pivot_idx, "最高"])
        last_close = float(df["close"].iloc[-1])
        dist_to_pivot = (pivot_price - last_close) / max(pivot_price, 1e-9)  # >0 表示在枢轴下方
        near_pivot = (0 <= dist_to_pivot <= near_pivot_tol)

        # —— 5) 突破 + 放量 —— 
        breakout = (last_close > pivot_price * (1 + 1e-4))
        breakout_vol_ok = False
        if breakout:
            breakout_vol_ok = (df["交易量"].iloc[-1] > breakout_vol_mult * df["vol_ma"].iloc[-1])

        # —— 6) 再扩张（突破附近带宽/ATR% 从基底中位数回升）——
        recent_bw  = float(df["bb_bw"].tail(reexpansion_win).mean())
        recent_atr = float(df["atr_pct"].tail(reexpansion_win).mean())
        bw_reexp_ok  = (not np.isnan(bw_med))  and (recent_bw  >= bw_med  * (1.0 + reexpansion_bw_increase))
        atr_reexp_ok = (not np.isnan(atr_med)) and (recent_atr >= atr_med * (1.0 + reexpansion_atr_increase))
        reexpansion_ok = bool(breakout and (bw_reexp_ok or atr_reexp_ok))

        # -------------------- 打分（总分 100） --------------------
        # 1) 基底压缩（30）
        score_bw   = 20.0 if bw_ok  else 0.0
        score_atr  = 10.0 if atr_ok else 0.0
        score_comp = score_bw + score_atr

        # 2) 连续收缩结构（20）
        score_contraction = 0.0
        if contraction_ok:
            avg_drop = float(np.mean(drops))         # 平均回撤(作为力度参考)
            drop_score = min(1.0, avg_drop / 0.15)   # 15% 饱和
            tighten_bonus = 1.0                      # 通过即满（结构性）
            score_contraction = 20.0 * (0.6 * drop_score + 0.4 * tighten_bonus)

        # 3) 干涸（15）
        dryup_score = max(0.0, min(1.0, (0.6 - dryup_ratio) / 0.6))  # ratio<=0.6 得满分
        score_dryup = 15.0 * dryup_score

        # 4) 接近枢轴（10）
        score_near = 0.0
        if near_pivot:
            score_near = 10.0 * max(0.0, min(1.0, (near_pivot_tol - dist_to_pivot) / max(near_pivot_tol, 1e-9)))

        # 5) 突破放量（10）
        score_breakout = 10.0 if (breakout and breakout_vol_ok) else 0.0

        # 6) 再扩张（15）
        score_reexp = 15.0 if reexpansion_ok else 0.0

        total_score = round(score_comp + score_contraction + score_dryup + score_near + score_breakout + score_reexp, 3)

        # -------------------- 输出 --------------------
        out = {
            "ok": True,
            "message": "success",
            "score": total_score,
            "score_breakdown": {
                "compression_bw": round(score_bw, 3),
                "compression_atr": round(score_atr, 3),
                "contraction": round(score_contraction, 3),
                "dryup": round(score_dryup, 3),
                "near_pivot": round(score_near, 3),
                "breakout": round(score_breakout, 3),
                "reexpansion": round(score_reexp, 3),
            },
            "flags": {
                "contraction_ok": bool(contraction_ok),
                "bw_low_ok": bool(bw_ok),
                "atr_low_ok": bool(atr_ok),
                "dryup_ok": bool(dryup_ok),
                "near_pivot": bool(near_pivot),
                "breakout": bool(breakout),
                "breakout_vol_ok": bool(breakout_vol_ok),
                "reexpansion_ok": bool(reexpansion_ok),
                "bw_reexp_ok": bool(bw_reexp_ok),
                "atr_reexp_ok": bool(atr_reexp_ok),
            },
            "components": {
                "drops": [round(d, 6) for d in drops],
                "avg_drop": float(np.mean(drops)),
                "pivot_price": round(pivot_price, 6),
                "last_close": round(last_close, 6),
                "dist_to_pivot": round(dist_to_pivot, 6),
                "dryup_ratio": round(dryup_ratio, 6),
                "base_bw_median": float(bw_med) if not np.isnan(bw_med) else None,
                "base_atr_median": float(atr_med) if not np.isnan(atr_med) else None,
                "hist_bw_pct_threshold": float(bw_q) if not np.isnan(bw_q) else None,
                "hist_atr_pct_threshold": float(atr_q) if not np.isnan(atr_q) else None,
                "recent_bw": recent_bw,
                "recent_atr": recent_atr,
            },
            "pivots_tail": pivots[-12:],  # 便于审阅
            "params": {
                "zigzag_pct": zigzag_pct,
                "min_drop": min_drop,
                "tighten_ratio_max": tighten_ratio_max,
                "bb_len": bb_len,
                "atr_len": atr_len,
                "bw_low_pct": bw_low_pct,
                "atr_low_pct": atr_low_pct,
                "dryup_days": dryup_days,
                "vol_ma_len": vol_ma_len,
                "dryup_thresh": dryup_thresh,
                "near_pivot_tol": near_pivot_tol,
                "breakout_vol_mult": breakout_vol_mult,
                "reexpansion_win": reexpansion_win,
                "reexpansion_bw_increase": reexpansion_bw_increase,
                "reexpansion_atr_increase": reexpansion_atr_increase,
                "window": window
            }
        }
        return out

    # 计算均线结构
    def evaluate_ma_structure(
        close: pd.Series,
        windows: Tuple[int, ...] = (5, 10, 20, 50, 60),
        slope_lookback: int = 3,           # 仍用于“所有MA上行”判定
        max_extension: Optional[float] = 0.12,
        # —— 低点抬高参数 ——
        use_low_series: Optional[pd.Series] = None,  # 建议传入 low；不传则用 close
        hl_lookaround: int = 3,   # 摆动低点窗口半宽
        hl_count: int = 3         # 至少最近 hl_count 个摆动低点依次抬高
    ) -> Dict[str, Any]:
        """
        扩展版“优秀均线结构”评分（总分=100）：
        价格在均线上方：20
        均线严格多头排列：30
        均线整体上行（所有MA）：20
        MA5 严格抬头（见下述双条件）：15
        低点不断抬高：15
        不过度乖离：5

        MA5 严格抬头（必须同时满足）：
        1) MA5[t] > MA5[t-1]
        2) (MA5[t]-MA5[t-3]) > (MA5[t-3]-MA5[t-6])
        """
        # —— 数据准备 ——
        s = pd.to_numeric(close, errors="coerce")
        if 5 not in windows:
            windows = tuple(sorted(set((5,) + tuple(windows))))  # 强制包含MA5
        ma: Dict[int, pd.Series] = {w: s.rolling(w, min_periods=w).mean() for w in windows}
        last = s.iloc[-1]

        # —— 价格在全部 MA 之上 ——
        above_all = all(
            (not pd.isna(ma[w].iloc[-1])) and (last > ma[w].iloc[-1])
            for w in windows
        )

        # —— 多头排列（严格：短 > 长） ——
        ordered = True
        last_vals: List[float] = []
        for w in windows:
            v = ma[w].iloc[-1]
            if pd.isna(v):
                ordered = False
                break
            last_vals.append(v)
        if ordered:
            ordered = all(last_vals[i] > last_vals[i+1] for i in range(len(last_vals)-1))

        # —— 所有 MA 上行：MA[-1] > MA[-1 - slope_lookback] ——
        slopes_ok = True
        for w in windows:
            series = ma[w]
            if len(series) <= slope_lookback or pd.isna(series.iloc[-1]) or pd.isna(series.iloc[-1 - slope_lookback]):
                slopes_ok = False
                break
            if series.iloc[-1] <= series.iloc[-1 - slope_lookback]:
                slopes_ok = False
                break

        # —— MA5 严格抬头（今天>昨天 & 近3日斜率>前3日斜率） ——
        ma5 = ma[5]
        ma5_turning_strict = False
        ma5_today = ma5.iloc[-1] if len(ma5) > 0 else np.nan
        ma5_yest  = ma5.iloc[-2] if len(ma5) > 1 else np.nan
        slope_recent_3 = np.nan
        slope_prev_3   = np.nan
        if len(ma5) >= 7 and not any(pd.isna([ma5_today, ma5_yest])):
            slope_recent_3 = ma5.iloc[-1] - ma5.iloc[-4]  # t - (t-3)
            slope_prev_3   = ma5.iloc[-4] - ma5.iloc[-7]  # (t-3) - (t-6)
            ma5_turning_strict = (ma5_today > ma5_yest) and (slope_recent_3 > slope_prev_3)

        # —— 低点不断抬高（摆动低点） ——
        base_series = pd.to_numeric(use_low_series, errors="coerce") if use_low_series is not None else s
        swing_low_idx: List[int] = []
        swing_low_val: List[float] = []
        for i in range(hl_lookaround, len(base_series) - hl_lookaround):
            win = base_series.iloc[i - hl_lookaround:i + hl_lookaround + 1]
            # 当前点既是窗口最小值且位于中心（通过 idxmin 校验）
            if base_series.iloc[i] == win.min() and win.idxmin() == base_series.index[i]:
                swing_low_idx.append(i)
                swing_low_val.append(float(base_series.iloc[i]))

        higher_lows_ok = False
        higher_lows_used: List[float] = []
        if len(swing_low_val) >= hl_count:
            last_lows = swing_low_val[-hl_count:]
            higher_lows_ok = all(last_lows[j] < last_lows[j+1] for j in range(len(last_lows)-1))
            higher_lows_used = last_lows

        # —— 乖离不过度（相对最短均线） ——
        not_overextended = True
        ext_ratio = np.nan
        if max_extension is not None:
            short_w = min(windows)
            short_ma = ma[short_w].iloc[-1]
            if pd.isna(short_ma) or short_ma == 0:
                not_overextended = False
            else:
                ext_ratio = abs(last / short_ma - 1.0)
                not_overextended = (ext_ratio <= max_extension)

        # —— 打分（总分 100） ——
        score = 0.0
        score += 20 if above_all else 0
        score += 30 if ordered else 0
        score += 20 if slopes_ok else 0
        score += 15 if ma5_turning_strict else 0
        score += 15 if higher_lows_ok else 0
        score += 5  if not_overextended else 0

        return {
            "score": round(score, 2),
            "flags": {
                "above_all": bool(above_all),
                "ordered": bool(ordered),
                "slopes_ok": bool(slopes_ok),
                "ma5_turning_strict": bool(ma5_turning_strict),
                "higher_lows_ok": bool(higher_lows_ok),
                "not_overextended": bool(not_overextended),
            },
            "latest": {
                "close": float(last),
                **{f"ma{w}": float(ma[w].iloc[-1]) if not pd.isna(ma[w].iloc[-1]) else np.nan for w in windows},
                "extension_vs_ma_short": float(ext_ratio) if not np.isnan(ext_ratio) else None,
                "higher_lows_used": higher_lows_used,
                "ma5_today": float(ma5_today) if not pd.isna(ma5_today) else None,
                "ma5_yesterday": float(ma5_yest) if not pd.isna(ma5_yest) else None,
                "ma5_slope_recent_3": float(slope_recent_3) if not pd.isna(slope_recent_3) else None,
                "ma5_slope_prev_3": float(slope_prev_3) if not pd.isna(slope_prev_3) else None,
            },
            "params": {
                "windows": windows,
                "slope_lookback": slope_lookback,
                "max_extension": max_extension,
                "use_low_series": "low" if use_low_series is not None else "close",
                "hl_lookaround": hl_lookaround,
                "hl_count": hl_count,
                "ma5_turning_rule": "MA5[t]>MA5[t-1] and (MA5[t]-MA5[t-3])>(MA5[t-3]-MA5[t-6])",
            }
        }
    
    # 计算RS
    def compute_rs(stock_close: pd.Series, bench_close: pd.Series) -> pd.Series:
        """
        RS 线：个股 / 基准（自动对齐索引并前向填充基准缺口）
        """
        sc = pd.to_numeric(stock_close, errors="coerce")
        bc = pd.to_numeric(bench_close, errors="coerce")
        df = pd.concat({"s": sc, "b": bc}, axis=1).sort_index().dropna(how="all")
        df["b"] = df["b"].ffill()
        rs = df["s"] / df["b"].replace(0, np.nan)
        return rs
    
    # 计算RS结构
    def evaluate_rs_structure(
        stock_close: pd.Series,
        bench_close: pd.Series,
        windows: Tuple[int, ...] = (20, 50, 100),
        slope_lookback: int = 3,
        nhigh_lookback: int = 60,
    ) -> Dict[str, Any]:
        """
        RS 结构（相对强于大盘/行业）：
        - RS 在其均线上方
        - RS 的均线多头有序 & 上行
        - RS 创出近 nhigh_lookback 高点（可选，加分）

        评分（总分100）：
        RS 在均线上方：30
        RS 均线多头排列：40
        RS 均线上行：20
        RS 接近/突破近高：10
        """
        rs = StockAnalyzer.compute_rs(stock_close, bench_close)
        ma = {w: rs.rolling(w, min_periods=w).mean() for w in windows}
        last = rs.iloc[-1]

        above_all = all(last > ma[w].iloc[-1] for w in windows if not pd.isna(ma[w].iloc[-1]))

        ordered = True
        last_vals = []
        for w in windows:
            v = ma[w].iloc[-1]
            if pd.isna(v):
                ordered = False
                break
            last_vals.append(v)
        if ordered:
            ordered = all(last_vals[i] > last_vals[i+1] for i in range(len(last_vals)-1))

        slopes_ok = True
        for w in windows:
            v_now = ma[w].iloc[-1]
            v_prev = ma[w].iloc[-1 - slope_lookback] if len(ma[w]) > slope_lookback else np.nan
            if pd.isna(v_now) or pd.isna(v_prev) or v_now <= v_prev:
                slopes_ok = False
                break

        # 近高（不必须）：最近是否创 nhigh_lookback 新高
        near_high = False
        high_val = rs.rolling(nhigh_lookback, min_periods=1).max().iloc[-1]
        if not pd.isna(high_val):
            # 允许与近高差距 1%
            near_high = (last >= 0.99 * high_val)

        score = 0.0
        score += 30 if above_all else 0
        score += 40 if ordered else 0
        score += 20 if slopes_ok else 0
        score += 10 if near_high else 0

        return {
            "score": round(score, 2),
            "flags": {
                "rs_above_all_ma": bool(above_all),
                "rs_ma_ordered": bool(ordered),
                "rs_ma_slopes_ok": bool(slopes_ok),
                "rs_near_high": bool(near_high),
            },
            "latest": {
                "rs": float(last),
                **{f"rs_ma{w}": float(ma[w].iloc[-1]) if not pd.isna(ma[w].iloc[-1]) else np.nan for w in windows},
                "rs_recent_high": float(high_val) if not pd.isna(high_val) else np.nan,
            },
            "windows": windows,
            "slope_lookback": slope_lookback,
            "nhigh_lookback": nhigh_lookback,
        }
    
    # 计算RS动量
    def rs_momentum(
        stock_close: pd.Series,
        bench_close: pd.Series,
        n: int = 20,
        method: str = "roc"
    ) -> pd.Series:
        """
        RS 动量：对 RS 线做动量（默认 ROC）。
        """
        rs = StockAnalyzer.compute_rs(stock_close, bench_close)
        if method == "roc":
            return rs / rs.shift(n) - 1.0
        elif method == "diff":
            return rs - rs.shift(n)
        else:
            raise ValueError("method must be 'roc' or 'diff'")

    def plot_moving_averages(self, colors=['red', 'blue', 'green']):
        if not hasattr(self, 'ma_data'):
            raise ValueError("请先调用 calculate_moving_averages 方法！")
        x_axis = self.ma_data['date']
        plt.figure(figsize=(14, 8))
        for i, column in enumerate(self.ma_data.columns):
            if 'MA_' in column:
                plt.plot(x_axis, self.ma_data[column], label=column, color=colors[i % len(colors)], linewidth=1.5)
        step = 50
        selected_ticks = x_axis[::step]
        plt.xticks(selected_ticks, rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{self.ticker} Moving Averages')
        plt.grid(True, linestyle='--', linewidth=0.2, alpha=1)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_all(self, data_ma, data_bbi, data_price, data_macd, data_kdj, data_shakeout, ticker, windows):
        # x_axis 是形如 ['2020-01-01', ...] 的字符串列表
        x_axis = pd.to_datetime(data_ma['date'])

        fig = sp.make_subplots(
            rows=6, cols=2,
            specs=[[{}, {}],    # 第一行两列
                [{"colspan": 2}, None],  # 第二行一整行（跨2列）
                [{"colspan": 2}, None],  # 第三行一整行
                [{"colspan": 2}, None],  # 第四行一整行
                [{}, {}],
                [{"colspan": 2}, None]], # 第五行一整行（跨2列）
            shared_xaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
            subplot_titles=[
                f'MA {windows[0]} {windows[1]} {windows[2]}', 'BBI',
                'Avg & Close Price',
                'KDJ-J Highlighted Points',
                'MACD',
                'KDJ-J Highlighted Points -10~90', 'KDJ-J Highlighted Points-15~100',
                'Shakeout Monitoring'
            ]
        )

        fig.update_layout(
        xaxis3=dict(matches='x'),
        xaxis4=dict(matches='x'),
        xaxis5=dict(matches='x'),
        xaxis6=dict(matches='x')
    )

        # 第一行，左图 MA
        fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[0]}'], name=f'MA_{windows[0]}', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[1]}'], name=f'MA_{windows[1]}', line=dict(color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[2]}'], name=f'MA_{windows[2]}', line=dict(color='green')), row=1, col=1)

        # 第一行，右图 BBI
        fig.add_trace(go.Scatter(x=x_axis, y=data_bbi['bbi'], name='BBI', line=dict(color='orange')), row=1, col=2)

        # 第二行，整行 price
        fig.add_trace(go.Scatter(x=x_axis, y=data_price['avg_price'], name='avg_price', line=dict(color='gray')), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_price['close_price'], name='close_price', line=dict(color='green')), row=2, col=1)

        # 第三行，整行 KDJ-J + 高亮点
        fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='gray')), row=3, col=1)

        mask_low = data_kdj['J'] <= -5
        fig.add_trace(go.Scatter(
            x=x_axis[mask_low],
            y=data_kdj['J'][mask_low],
            mode='markers+text',
            name='J <= -5',
            marker=dict(color='red', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=3, col=1)

        mask_high = data_kdj['J'] > 80
        fig.add_trace(go.Scatter(
            x=x_axis[mask_high],
            y=data_kdj['J'][mask_high],
            mode='markers+text',
            name='J > 80',
            marker=dict(color='green', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_high]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=3, col=1)

        # 第四行，整行 MACD
        fig.add_trace(go.Scatter(x=x_axis, y=data_macd['DIF'], name='DIF', line=dict(color='green')), row=4, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_macd['DEA'], name='DEA', line=dict(color='gray')), row=4, col=1)

        # 第五行，左图 -10～90
        fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='gray')), row=5, col=1)

        mask_low = data_kdj['J'] <= -10
        fig.add_trace(go.Scatter(
            x=x_axis[mask_low],
            y=data_kdj['J'][mask_low],
            mode='markers+text',
            name='J <= -10',
            marker=dict(color='red', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=5, col=1)

        mask_high = data_kdj['J'] > 90
        fig.add_trace(go.Scatter(
            x=x_axis[mask_high],
            y=data_kdj['J'][mask_high],
            mode='markers+text',
            name='J > 90',
            marker=dict(color='green', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_high]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=5, col=1)

        # 第五行，右图 -15～100
        fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='gray')), row=5, col=2)

        mask_low = data_kdj['J'] <= -15
        fig.add_trace(go.Scatter(
            x=x_axis[mask_low],
            y=data_kdj['J'][mask_low],
            mode='markers+text',
            name='J <= -15',
            marker=dict(color='red', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=5, col=1)

        mask_high = data_kdj['J'] > 100
        fig.add_trace(go.Scatter(
            x=x_axis[mask_high],
            y=data_kdj['J'][mask_high],
            mode='markers+text',
            name='J > 100',
            marker=dict(color='green', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_high]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=5, col=2)

        # 第六行，整行 shakeout monitoring
        fig.add_trace(go.Scatter(x=x_axis, y=data_shakeout['短期'], name='短期', line=dict(color='green')), row=6, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_shakeout['长期'], name='长期', line=dict(color='red')), row=6, col=1)

        # 添加掩码
        mask = (data_shakeout['短期'] < 20) & (data_shakeout['长期'] > 60)

        # 筛选符合条件的 x 和 y 值
        x_highlight = x_axis[mask]
        y_highlight = data_shakeout['短期'][mask]
        # 添加高亮点
        fig.add_trace(go.Scatter(
            x=x_highlight,
            y=y_highlight,
            mode='markers+text',
            name='短期<20 & 长期>60',
            marker=dict(color='cyan', size=10, symbol='circle'),
            text=[f'{v:.1f}' for v in y_highlight],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=6, col=1)

        # 在第六行子图（row=6, col=1）上绘制 y=20, 60, 80 三条横线 红线在60 80之间 白线在20以下
        for y_val in [60, 80]:
            fig.add_shape(
                type="line",
                x0=x_axis.min(),
                x1=x_axis.max(),
                y0=y_val,
                y1=y_val,
                line=dict(color="green", width=3, dash="solid"),
                xref="x8",  # row=6 col=1 的 subplot x 轴
                yref="y8"   # row=6 col=1 的 subplot y 轴
            )

        fig.add_shape(
                type="line",
                x0=x_axis.min(),
                x1=x_axis.max(),
                y0=20,
                y1=20,
                line=dict(color="yellow", width=3, dash="solid"),
                xref="x8",  # row=6 col=1 的 subplot x 轴
                yref="y8"   # row=6 col=1 的 subplot y 轴
            )
        # 更新布局
        fig.update_layout(
            height=1200,
            width=1400,
            title=f'{ticker}',
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,1)',
            template='plotly_white'
        )

        fig.show()
    
    def save_moving_averages(self, filename=None):
        if not hasattr(self, 'ma_data'):
            raise ValueError("请先调用 calculate_moving_averages 方法！")
        filename = filename or f'{self.ticker}_moving_averages.csv'
        self.ma_data.to_csv(filename, index=False)
        print(f"数据已保存至：{filename}")

# 计算买入卖出信号
class StockMonitor:
    '''
    监控买入卖出信号
    '''
    def __init__(self, ticker, file_path,  file_volume_path, start_date=None, end_date=None, lookback_period=10, min_signal_count=3):
        self.ticker = ticker
        self.file_path = file_path
        self.file_volume_path = file_volume_path
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_period = lookback_period # 连续n天内出现单针下20的信号
        self.min_signal_count = min_signal_count # 出现n次单针下20的信号
    
    def fastdown_J(self):
        '''
        买入逻辑
        '''
        analyzer = StockAnalyzer(self.ticker, self.file_path)
        data_kdj = analyzer.calculate_kdj()
        label = False
        if (data_kdj['J'].iloc[-3] - data_kdj['J'].iloc[-1]) >= 70 and data_kdj['J'].iloc[-1] < 10:
            label = True
        '''
        label = all(
        any([
            (data_kdj['J'].iloc[-1] < 10 and data_kdj['J'].iloc[period] > 70),  
            (data_kdj['J'].iloc[-1] < 5 and data_kdj['J'].iloc[period] > 65),  
            (data_kdj['J'].iloc[-1] < 0 and data_kdj['J'].iloc[period] > 60) 
        ])
        for period in [-1, -2, -3] )
        '''
        print(f"{self.ticker}:J值是否快速下降：{'true✅' if label else 'false❌'}，最近3️⃣天的J值：{round(data_kdj['J'].iloc[-3],1)}，{round(data_kdj['J'].iloc[-2],1)}，{round(data_kdj['J'].iloc[-1],1)}")

        return label

    def continuous_shakeout(self):
        '''
        买入逻辑
        '''
        analyzer = StockAnalyzer(self.ticker, self.file_path)
        data_shakeout = analyzer.calculate_shakeout()
        label = False
        label = all(
        any([
            data_shakeout.iloc[period].get("四线归零买", False) == 1,
            data_shakeout.iloc[period].get("白线下20买", False) == 1,
            data_shakeout.iloc[period].get("白穿红线买", False) == 1,
            data_shakeout.iloc[period].get("白穿黄线买", False) == 1,
            data_shakeout.iloc[period].get("白线下20买_小V", False) == 1
        ])
        for period in [-1, -2] )

        return label

    def check_signal_frequency(self):
        '''
        买入逻辑，检查最近10天内是否至少有3个周期满足任意买入信号
        '''
        analyzer = StockAnalyzer(self.ticker, self.file_path)
        data_shakeout = analyzer.calculate_shakeout()
        signal_count = 0
        for period in range(-1, -self.lookback_period-1, -1): 
            if any([
                data_shakeout.iloc[period].get("四线归零买", 0) == 1,
                data_shakeout.iloc[period].get("白线下20买", 0) == 1,
                data_shakeout.iloc[period].get("白穿红线买", 0) == 1,
                data_shakeout.iloc[period].get("白穿黄线买", 0) == 1,
                data_shakeout.iloc[period].get("白线下20买_小V", 0) == 1
            ]):
                signal_count += 1
                if signal_count >= self.min_signal_count:  # 达到最小信号数就提前返回
                    return True
        return signal_count >= self.min_signal_count

    def volume_contraction(self):
        '''
        缩量判断，检查当前量是否缩到最近30天中低于最高量的1/4
        '''
        df = pd.read_csv(self.file_path, engine=None)
        sorted_df = df.sort_values('日期')
        print(f"缩量判断排序后的日期：{sorted_df}")
        latest_vol = sorted_df.iloc[-1]['成交量']
        window_max = sorted_df.iloc[-30:]['成交量'].max()  # 最近30个交易日（含最新日）
        is_ok = latest_vol < window_max / 4
        return is_ok


    def bs_abnormal_monitor(self):
        '''
        * 买入、卖出逻辑
        * 监控异常交易量、价格、买卖笔数，比如当日绿线，但是买入笔数大于卖出笔数，可能是有人在低位收筹码
        * 开盘收盘价格是从000001.csv（历史价格）文件中获取的，开盘收盘总价和总量是从000001_volume.csv，因为_volume中才有卖盘和卖盘信息，如果想看历史数据可以去通达信导出
        '''
        # 获取的是当天最新的数据分析!
        df = pd.read_csv(self.file_volume_path, engine=None)
        sellvolume_amount = df.loc[df["性质"].eq("卖盘"), "成交量"].sum()
        sellprice_amount = df.loc[df["性质"].eq("卖盘"), "成交金额"].sum()
        buyprice_amount = df.loc[df["性质"].eq("买盘"), "成交金额"].sum()
        buyvolume_amount = df.loc[df["性质"].eq("买盘"), "成交量"].sum()

        label = False
        abnormal_type = 'none'
        # 获取历史上最新的数据
        analyzer = StockAnalyzer(self.ticker, self.file_path)
        price_dict = analyzer.calculate_price()
        open_price = price_dict['open_price'].iloc[-1]
        close_price = price_dict['close_price'].iloc[-1]

        if (close_price < open_price) and (buyvolume_amount > sellvolume_amount):
            print(f"❗️当日绿线📉，但是买入量大于卖出量，可能是有人偷偷在低位收筹码❗️")
            label = True
            abnormal_type = 'buy'
        elif(close_price > open_price) and (buyvolume_amount < sellvolume_amount):
            print(f"❗️当日红线📈，但是买入量小于卖出量，可能是有人偷偷在高位卖筹码❗️")
            label = True
            abnormal_type = 'sell'
        else:
            print(f"成交量无异常")

        if (close_price < open_price) and (buyprice_amount > sellprice_amount):
            print(f"❗️当日绿线📉，但是买入总额大于卖出总额，可能是有人偷偷在低位收筹码❗️")
            label = True
            abnormal_type = 'buy'
        elif(close_price > open_price) and (buyprice_amount < sellprice_amount):
            print(f"❗️当日红线📈，但是买入总额小于卖出总额，可能是有人偷偷在高位卖筹码❗️")
            label = True
            abnormal_type = 'sell'
        else:
            print(f"成交总额无异常")

        # 获取总市值和总股本
        market_cap, share_cap = getData.download_total_cap(self.ticker)

        print(f"当日开盘价：{open_price}，收盘价：{close_price}， {'📈' if close_price > open_price else '📉'}， 卖出总额：{sellprice_amount}={int_to_chinese_num(sellprice_amount)}，买入总额：{buyprice_amount}={int_to_chinese_num(buyprice_amount)}，净买入总额：{buyprice_amount-sellprice_amount}={int_to_chinese_num(buyprice_amount-sellprice_amount)}，卖出总量：{sellvolume_amount}={int_to_chinese_num(sellvolume_amount)}，买入总量：{buyvolume_amount}={int_to_chinese_num(buyvolume_amount)}，净买入总量：{buyvolume_amount-sellvolume_amount}={int_to_chinese_num(buyvolume_amount-sellvolume_amount)}")
        print(f"当前交易占总股本比重:{round(abs(buyprice_amount-sellprice_amount) / int(share_cap),3)}，占总市值比重为:{round(abs(buyprice_amount-sellprice_amount) / int(market_cap),3)}")
        return {'open_price': open_price, 'close_price': close_price, 'sellprice_amount': sellprice_amount, 'buyprice_amount': buyprice_amount, 'sellvolume_amount': sellvolume_amount
                , 'buyvolume_amount': buyvolume_amount, 'label':label, 'abnormal_type':abnormal_type, 'market_cap_percentage':round(abs(buyprice_amount-sellprice_amount) / int(share_cap),3)
                , 'share_cap_percentage':round(abs(buyprice_amount-sellprice_amount) / int(market_cap),3)}

    def below_bbi_monitor(self):
        '''
        卖出信号，跌破BBI两根卖出信号，即收盘价低于BBI
        '''
        # 获取的是当天最新的价格数据
        df = getData.read_from_csv(self.file_path)

        analyzer = StockAnalyzer(self.ticker, self.file_path)
        bbi = analyzer.calculate_bbi()

        bbi_label = False

        if (df['开盘'].iloc[-1] > bbi['bbi'].iloc[-1] > df['收盘'].iloc[-1]) and (df['开盘'].iloc[-2] > bbi['bbi'].iloc[-2] > df['收盘'].iloc[-2]):
            bbi_label = True
            print(f"❗️跌破BBI两根卖出信号❗️")
        else:
            print(f"未跌破BBI两根卖出信号")

        return bbi_label

    def twin_tails_monitor(self):
        '''
        双马尾，根据价格判断----看一下这个逻辑能不能筛选出来娜娜的图形

        检查DataFrame中最近period个交易日是否符合以下模式：
        1. 只有两天达到最高价，且差值在1%以内；
        2. 这两天之间相差15天以上；
        3. 周期内最高价和最低价落差超过30%。

        参数:
            df (pd.DataFrame): 包含 '日期', '最高', '最低' 三列
            period (int): 检查周期（默认60天）

        返回:
            bool: 是否满足条件
        '''
        period_days_60 = 60 # 设置计算周期为60天

        df = getData.read_from_csv(self.file_path)
        required_cols = {'日期','最高价','最低价'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"输入DataFrame必须包含列：{required_cols}")
        
        recent_df_60 = df.tail(period_days_60).copy()

        if recent_df_60.shape[0] < period_days_60:
            print("数据不足指定周期，跳过检查。")
            return False

        # 最高价和最低价
        highest_price_60 = recent_df_60['最高价'].max()
        lowest_price_60 = recent_df_60['最低价'].min()

        # 落差 > 15%
        if (highest_price_60 - lowest_price_60) / lowest_price_60 < 0.15:
            return False
        
        # 取第二高，判断与最高价的差距
        second_high = recent_df_60['最高价'].nlargest(2).iloc[1]
        diff_pct = abs(highest_price_60 - second_high) / highest_price_60
        if diff_pct > 0.05:
            return False
        
        # 找出达到最高价的行
        highest_rows = recent_df_60[recent_df_60['最高价'] == highest_price_60]
        # 两个最高价的日期间隔要超过15天
        day_diff = abs((highest_rows['日期'].iloc[1] - highest_rows['日期'].iloc[0]).days)
        if day_diff <= 5:
            return False
        
        return True
    
    def position_building_monitor(self):
        '''
        建仓波，准备撅小土包，连续7天及以上小红或者14天内红的占一多半
        '''
        df = getData.read_from_csv(self.file_path)
        # 条件1相关布尔序列：收盘高于开盘，且振幅（high-low）/open < 5%
        condition1 = (df['收盘'] > df['开盘']) & ((df['最高'] - df['最低']) / df['最低'] < 0.05)
        # 条件2：14 日窗口内，收盘高于开盘的天数 >= 10
        condition2 = (df['收盘'] > df['开盘'])

        # 条件 1：最近连续 5 天及以上满足 cond1
        consecutive = 0
        for val in reversed(condition1.tolist()):  # 从最近日期向前看
            if val:
                consecutive += 1
            else:
                break
        if consecutive >= 5:
            return True

        # ----------------------
        # 条件 2：最近 14 个交易日内有 10 天以上满足 cond2
        if condition2.tail(14).sum() >= 10:
            return True

        return False
    
    def heavy_cannos_monitor(self):
        '''
        两门重炮，取5天为1个周期，金额和成交量必须同时被两边包住
        '''
        df = getData.read_from_csv(self.file_path)
        df_volume = getData.read_from_csv(self.file_volume_path)
        period_df = df.iloc[-5:].copy()
        period_df_volume = df_volume.iloc[-5:].copy()

        today = df.iloc[-1]
        today_volume = df_volume.iloc[-1]

        today_high = today['最高价']
        today_low = today['最低价']
        today_volume_amount = today_volume['成交金额']
    
        for i in range(len(period_df) - 1):  # 最后一行是今天，不比较
            day = period_df.iloc[i]
            day_volume = period_df_volume.iloc[i]
            
            # 判断价格与成交量的相似性
            high_similar = abs(day['最高价'] - today_high) / today_high <= 0.05
            low_similar = abs(day['最低价'] - today_low) / today_low <= 0.05
            volume_similar = abs(day_volume['成交金额'] - today_volume_amount) / today_volume_amount <= 0.2
            
            if high_similar and low_similar and volume_similar:
                # 找出这两天中较低的成交量
                vol_min = min(day_volume['成交金额'], today_volume_amount)
                
                # 这两天之间的所有天的成交量必须都低于这两天的最小成交量，不一定是完全小于，可能中间的成交量略大于最小的
                intermediate = period_df.iloc[i+1:-1]
                if all(intermediate['成交金额'] <= vol_min) and all(intermediate['最高价'] < today_high) and all(intermediate['最低价'] > today_low):
                    return True  # 满足条件
    
        return False  # 所有天都不满足

    def tradeing_sideways(self):
        '''
        横盘监控，采用比较5天周期内最高价和最低价，如果最高价和最低价相差不超过10%，则认为横盘
        '''
        df = getData.read_from_csv(self.file_path)
        period_df = df.iloc[-5:].copy()
        highest_price = period_df['最高'].max()
        lowest_price = period_df['最低'].min()
        if (highest_price - lowest_price) / lowest_price <= 0.1:
            return True

        return False

    def anti_premature_exit_scoreing_strategy(self):
        '''
        放卖飞策略

        规则口径一览
	1.	收盘价上升：默认判 close[-1] > close[-2]
	2.	未低于BBI：BBI = (MA3+MA6+MA12+MA24)/4，判 close[-1] ≥ BBI[-1]
	3.	无巨量阴线：在最近 giant_recent_days 天内，是否存在 阴线 且 当日成交量 > 前 1 天的最大成交量。
	4.	趋势向上：MA20[-1] > MA20[-1-3] 且 MA10[-1] > MA20[-1] 且 close[-1] > MA10[-1]
	5.	KDJ 的 J 无死叉：当天没有 K 由上向下穿 D，且 J[-1] ≥ J[-2]
        '''
        df = getData.read_from_csv(self.file_path)

        count = 0
        close_price = df['收盘']
        open_price = df['开盘']
        volume = df['成交量']
        analyzer = StockAnalyzer(self.ticker, self.file_path)
        bbi_price = analyzer.calculate_bbi()
        kdj = analyzer.calculate_kdj()
        k,d,j = kdj['K'].iloc[-1], kdj['D'].iloc[-1], kdj['J'].iloc[-1]
        k_prev,d_prev,j_prev = kdj['K'].iloc[-2], kdj['D'].iloc[-2], kdj['J'].iloc[-2]

        if close_price.iloc[-1] > close_price.iloc[-2]:
            count += 1

        if close_price.iloc[-1] >= bbi_price['bbi'].iloc[-1]:
            count += 1
        
        if close_price.iloc[-1] < open_price.iloc[-1] and volume.iloc[-1] > volume.iloc[-2]:
            count += 1
        
        if (k_prev > d_prev and k > d and j < j_prev):
            count += 1

        print(f"放卖飞🕊️策略：{self.ticker} 评分：{count}，是否上升趋势📈请结合图形判断")

# 监控太子的异动和放量
# 示例调用
if __name__ == "__main__":
    
    ticker = '600036.SS'
    file_path = '/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/data/000001.csv'  # 替换为你的路径
    file_volume_path = '/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/data/000001_volume.csv'


    analyzer = StockAnalyzer(ticker, file_path)
    ma = analyzer.calculate_moving_averages()
    bbi = analyzer.calculate_bbi()
    kdj = analyzer.calculate_kdj()
    macd = analyzer.calculate_macd()
    price = analyzer.calculate_price()
    shakeout = analyzer.calculate_shakeout()

    StockMonitor(ticker, file_path, file_volume_path).below_bbi_monitor()
    
        #analyzer.plot_all(ma, bbi, price, macd, kdj, shakeout, '000001', windows=[20, 30, 60, 120])
    
