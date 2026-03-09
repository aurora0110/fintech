import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
OUTPUT_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/body_k_compare_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_COL_CANDIDATES = ["date", "Date", "trade_date", "日期"]
OPEN_COL_CANDIDATES = ["open", "Open", "开盘"]
HIGH_COL_CANDIDATES = ["high", "High", "最高"]
LOW_COL_CANDIDATES = ["low", "Low", "最低"]
CLOSE_COL_CANDIDATES = ["close", "Close", "收盘"]
VOL_COL_CANDIDATES = ["volume", "vol", "Volume", "成交量"]
CODE_COL_CANDIDATES = ["code", "ts_code", "symbol", "代码"]

# 如果你的原始文件里本来就有“趋势线 / 多空线”列，会优先读取它们；
# 如果没有，就自动退化为：
# 趋势线 = ma10
# 多空线 = ma20
TRENDLINE_COL_CANDIDATES = ["trend_line", "趋势线", "trendline"]
BULLBEAR_COL_CANDIDATES = ["bull_bear_line", "多空线", "bullbear_line"]

MIN_BARS = 80

# 卖出参数：保持你前面那套最优候选不变
TAKE_PROFIT_MODE = "close_profit_next_open"   # 收盘达到阈值，下一天收盘卖
TAKE_PROFIT_THRESHOLD = 0.035                 # 3.5%
STOP_MODE = "prev_day_low"                    # 前一天最低价止损
MAX_HOLD_DAYS = 3

# K线长短划分：用“当日总振幅 / 过去20日振幅中位数”做自适应划分
# <= 1.0 记为短K，> 1.0 记为长K
KLINE_LEN_THRESHOLD = 1.0


# =========================
# 基础工具
# =========================

def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"缺少字段，候选字段={candidates}")
    return None


def robust_read_csv(path: str) -> pd.DataFrame:
    """
    兼容逗号分隔、制表符、空格分隔。
    """
    # 先尝试标准 csv
    try:
        df = pd.read_csv(path)
        if df.shape[1] >= 5:
            return df
    except Exception:
        pass

    # 再尝试正则分隔
    try:
        df = pd.read_csv(path, sep=r"\s+|\t+|,", engine="python")
        return df
    except Exception as e:
        raise ValueError(f"读取失败: {e}")


def load_one_csv(path: str) -> pd.DataFrame:
    df = robust_read_csv(path)

    date_col = pick_col(df, DATE_COL_CANDIDATES)
    open_col = pick_col(df, OPEN_COL_CANDIDATES)
    high_col = pick_col(df, HIGH_COL_CANDIDATES)
    low_col = pick_col(df, LOW_COL_CANDIDATES)
    close_col = pick_col(df, CLOSE_COL_CANDIDATES)
    vol_col = pick_col(df, VOL_COL_CANDIDATES)
    code_col = pick_col(df, CODE_COL_CANDIDATES, required=False)

    trendline_col = pick_col(df, TRENDLINE_COL_CANDIDATES, required=False)
    bullbear_col = pick_col(df, BULLBEAR_COL_CANDIDATES, required=False)

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "open": pd.to_numeric(df[open_col], errors="coerce"),
        "high": pd.to_numeric(df[high_col], errors="coerce"),
        "low": pd.to_numeric(df[low_col], errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
        "volume": pd.to_numeric(df[vol_col], errors="coerce"),
    })

    if trendline_col:
        out["trend_line_raw"] = pd.to_numeric(df[trendline_col], errors="coerce")
    else:
        out["trend_line_raw"] = np.nan

    if bullbear_col:
        out["bull_bear_line_raw"] = pd.to_numeric(df[bullbear_col], errors="coerce")
    else:
        out["bull_bear_line_raw"] = np.nan

    out["code"] = df[code_col].iloc[0] if code_col else os.path.splitext(os.path.basename(path))[0]

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    out = out.sort_values("date").reset_index(drop=True)

    # 基础价格清洗
    out = out[
        (out["open"] > 0) &
        (out["high"] > 0) &
        (out["low"] > 0) &
        (out["close"] > 0) &
        (out["volume"] >= 0)
    ].copy()

    # high/low 纠偏
    out["high"] = out[["open", "high", "close", "low"]].max(axis=1)
    out["low"] = out[["open", "high", "close", "low"]].min(axis=1)

    if len(out) < MIN_BARS:
        return pd.DataFrame()

    return out


def load_all_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    data_map = {}
    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".csv", ".txt"))]
    total = len(files)

    for i, f in enumerate(files, 1):
        path = os.path.join(data_dir, f)
        try:
            df = load_one_csv(path)
            if not df.empty:
                code = str(df["code"].iloc[0])
                data_map[code] = df
        except Exception as e:
            print(f"加载失败: {f} -> {e}")

        if i % 500 == 0 or i == total:
            print(f"加载进度: {i}/{total}")

    return data_map


# =========================
# 特征
# =========================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["body"] = (x["close"] - x["open"]).abs()
    x["range"] = x["high"] - x["low"]
    x["range"] = x["range"].where(x["range"] > 0, np.nan)

    x["body_pct_range"] = x["body"] / x["range"]
    x["close_pos"] = (x["close"] - x["low"]) / x["range"]

    x["ma5"] = x["close"].rolling(5).mean()
    x["ma10"] = x["close"].rolling(10).mean()
    x["ma20"] = x["close"].rolling(20).mean()

    x["vol_ma5"] = x["volume"].rolling(5).mean()
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["vol_ratio_5_20"] = x["vol_ma5"] / x["vol_ma20"].replace(0, np.nan)

    x["prev_body"] = x["body"].shift(1)
    x["body_vs_prev"] = x["body"] / x["prev_body"].replace(0, np.nan)

    # K线长度相对过去20日中位数
    x["range_med20"] = x["range"].rolling(20).median()
    x["kline_len_ratio"] = x["range"] / x["range_med20"].replace(0, np.nan)

    # 趋势线 / 多空线
    # 优先使用原文件列；若没有，则回退为 ma10 / ma20
    x["trend_line"] = x["trend_line_raw"]
    x["bull_bear_line"] = x["bull_bear_line_raw"]

    x["trend_line"] = x["trend_line"].fillna(x["ma10"])
    x["bull_bear_line"] = x["bull_bear_line"].fillna(x["ma20"])

    x["trend_gt_bullbear"] = (x["trend_line"] > x["bull_bear_line"]).astype(int)

    return x


# =========================
# 买入信号：只对比两组
# =========================

def build_compare_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    基本买入条件固定：
    1) 趋势线 > 多空线
    2) 阳线
    3) 收盘靠近高位
    4) 5日均量 / 20日均量 >= 1.0
    5) 长柱体 > 前一天柱体长度 * 2

    然后仅对比：
    A. 长柱体 + 短K线
    B. 长柱体 + 长K线
    """
    x = df.copy()

    base_cond = (
        (x["trend_gt_bullbear"] == 1) &
        (x["close"] > x["open"]) &
        (x["close_pos"] >= 0.7) &
        (x["vol_ratio_5_20"] >= 1.0) &
        (x["body_vs_prev"] > 2.0)
    )

    short_k_cond = base_cond & (x["kline_len_ratio"] <= KLINE_LEN_THRESHOLD)
    long_k_cond = base_cond & (x["kline_len_ratio"] > KLINE_LEN_THRESHOLD)

    x["signal_shortk"] = short_k_cond.astype(int)
    x["signal_longk"] = long_k_cond.astype(int)

    return x


# =========================
# 卖出模拟：保持原最优候选
# =========================

def simulate_exit(
    close_series: pd.Series,
    high_series: pd.Series,
    low_series: pd.Series,
    entry_idx: int,
    mode: str,
    threshold: float,
    hold_days: int,
    stop_mode: str,
    prev_day_low: float,
) -> Tuple[float, int]:
    n = len(close_series)

    entry_close = close_series.iloc[entry_idx]
    if pd.isna(entry_close) or entry_close <= 0:
        return np.nan, 0

    stop_price = prev_day_low
    if pd.isna(stop_price) or stop_price <= 0:
        stop_price = low_series.iloc[entry_idx]

    max_j = min(entry_idx + hold_days, n - 1)

    for j in range(entry_idx + 1, max_j + 1):
        low_j = low_series.iloc[j]
        high_j = high_series.iloc[j]
        close_j = close_series.iloc[j]

        if pd.isna(low_j) or pd.isna(high_j) or pd.isna(close_j):
            continue

        # 先止损
        if low_j <= stop_price:
            return stop_price / entry_close - 1, j - entry_idx

        if mode == "close_profit_next_open":
            if close_j / entry_close - 1 >= threshold:
                sell_day = min(j + 1, n - 1)
                sell_close = close_series.iloc[sell_day]
                if pd.isna(sell_close) or sell_close <= 0:
                    return np.nan, sell_day - entry_idx
                return sell_close / entry_close - 1, sell_day - entry_idx

        elif mode == "high_same_day_take_profit":
            if high_j / entry_close - 1 >= threshold:
                return threshold, j - entry_idx

    final_close = close_series.iloc[max_j]
    if pd.isna(final_close) or final_close <= 0:
        return np.nan, max_j - entry_idx

    return final_close / entry_close - 1, max_j - entry_idx


def calc_trade_stats(trades_df: pd.DataFrame) -> pd.Series:
    if trades_df.empty:
        return pd.Series({
            "样本数": 0,
            "平均每笔收益": np.nan,
            "胜率": np.nan,
            "收益标准差": np.nan,
            "平均持有天数": np.nan,
            "盈亏比": np.nan,
            "近似夏普": np.nan,
        })

    r = trades_df["ret"].dropna()
    if len(r) == 0:
        return pd.Series({
            "样本数": 0,
            "平均每笔收益": np.nan,
            "胜率": np.nan,
            "收益标准差": np.nan,
            "平均持有天数": np.nan,
            "盈亏比": np.nan,
            "近似夏普": np.nan,
        })

    win = r[r > 0]
    loss = r[r <= 0]

    avg_win = win.mean() if len(win) > 0 else np.nan
    avg_loss = loss.mean() if len(loss) > 0 else np.nan

    if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0:
        pl_ratio = abs(avg_win / avg_loss)
    else:
        pl_ratio = np.nan

    std = r.std(ddof=0)
    sharpe = r.mean() / std if pd.notna(std) and std > 0 else np.nan

    return pd.Series({
        "样本数": len(r),
        "平均每笔收益": r.mean(),
        "胜率": (r > 0).mean(),
        "收益标准差": std,
        "平均持有天数": trades_df["hold_days"].dropna().mean(),
        "盈亏比": pl_ratio,
        "近似夏普": sharpe,
    })


# =========================
# 生成交易
# =========================

def run_group_backtest(
    data_map: Dict[str, pd.DataFrame],
    signal_col: str,
    group_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_trades = []
    daily_rows = []

    for code, raw in data_map.items():
        x = add_features(raw)
        x = build_compare_signals(x).reset_index(drop=True)

        sig_idx = x.index[x[signal_col] == 1].tolist()
        if len(sig_idx) == 0:
            continue

        for i in sig_idx:
            if i < 1 or i + 1 >= len(x):
                continue

            prev_day_low = x.loc[i - 1, "low"]
            ret, hold_days = simulate_exit(
                close_series=x["close"],
                high_series=x["high"],
                low_series=x["low"],
                entry_idx=i,
                mode=TAKE_PROFIT_MODE,
                threshold=TAKE_PROFIT_THRESHOLD,
                hold_days=MAX_HOLD_DAYS,
                stop_mode=STOP_MODE,
                prev_day_low=prev_day_low,
            )

            if pd.isna(ret):
                continue

            row = {
                "code": code,
                "date": x.loc[i, "date"],
                "group": group_name,
                "ret": ret,
                "hold_days": hold_days,
                "entry_close": x.loc[i, "close"],
                "prev_day_low": prev_day_low,
                "body": x.loc[i, "body"],
                "prev_body": x.loc[i, "prev_body"],
                "body_vs_prev": x.loc[i, "body_vs_prev"],
                "range": x.loc[i, "range"],
                "kline_len_ratio": x.loc[i, "kline_len_ratio"],
                "vol_ratio_5_20": x.loc[i, "vol_ratio_5_20"],
                "close_pos": x.loc[i, "close_pos"],
                "trend_line": x.loc[i, "trend_line"],
                "bull_bear_line": x.loc[i, "bull_bear_line"],
            }
            all_trades.append(row)

    trades_df = pd.DataFrame(all_trades)

    if not trades_df.empty:
        daily_df = trades_df.groupby("date").agg(
            当日样本数=("ret", "count"),
            当日平均收益=("ret", "mean"),
            当日胜率=("ret", lambda s: (s > 0).mean()),
        ).reset_index()
        daily_df["group"] = group_name
    else:
        daily_df = pd.DataFrame(columns=["date", "当日样本数", "当日平均收益", "当日胜率", "group"])

    return trades_df, daily_df


# =========================
# 汇总比较
# =========================

def compare_groups(short_trades: pd.DataFrame, long_trades: pd.DataFrame) -> pd.DataFrame:
    s1 = calc_trade_stats(short_trades)
    s2 = calc_trade_stats(long_trades)

    out = pd.DataFrame([
        {"分组": "长柱体+短K线", **s1.to_dict()},
        {"分组": "长柱体+长K线", **s2.to_dict()},
    ])

    return out


def generate_conclusion(compare_df: pd.DataFrame) -> pd.DataFrame:
    if compare_df.empty or len(compare_df) < 2:
        return pd.DataFrame(columns=["结论项", "结果", "原因"])

    a = compare_df[compare_df["分组"] == "长柱体+短K线"].iloc[0]
    b = compare_df[compare_df["分组"] == "长柱体+长K线"].iloc[0]

    # 先看平均每笔收益，再看胜率
    if pd.notna(a["平均每笔收益"]) and pd.notna(b["平均每笔收益"]):
        if (a["平均每笔收益"] > b["平均每笔收益"]) and (a["胜率"] >= b["胜率"]):
            better = "长柱体+短K线"
            reason = f"平均每笔收益={a['平均每笔收益']:.4f} > {b['平均每笔收益']:.4f}，胜率={a['胜率']:.2%} vs {b['胜率']:.2%}"
        elif (b["平均每笔收益"] > a["平均每笔收益"]) and (b["胜率"] >= a["胜率"]):
            better = "长柱体+长K线"
            reason = f"平均每笔收益={b['平均每笔收益']:.4f} > {a['平均每笔收益']:.4f}，胜率={b['胜率']:.2%} vs {a['胜率']:.2%}"
        else:
            better = "结果分化，需要结合收益/胜率/盈亏比综合判断"
            reason = (
                f"短K：平均每笔收益={a['平均每笔收益']:.4f}, 胜率={a['胜率']:.2%}, 盈亏比={a['盈亏比']:.2f}; "
                f"长K：平均每笔收益={b['平均每笔收益']:.4f}, 胜率={b['胜率']:.2%}, 盈亏比={b['盈亏比']:.2f}"
            )
    else:
        better = "无法判断"
        reason = "有效样本不足或收益为空"

    return pd.DataFrame([
        {
            "结论项": "哪组表现更好",
            "结果": better,
            "原因": reason
        },
        {
            "结论项": "本次固定买入条件",
            "结果": "趋势线>多空线 + 阳线 + 收盘靠近高位 + 5日均量/20日均量>=1.0 + 当日柱体>前一天柱体*2",
            "原因": "只比较长柱体+短K线 vs 长柱体+长K线，其他条件不变"
        },
        {
            "结论项": "本次固定卖出条件",
            "结果": "收盘达到3.5%则下一日收盘卖；跌破前一天最低价止损；最长持有3天",
            "原因": "保持你前面要求的最优候选卖出逻辑不变"
        }
    ])


# =========================
# 主流程
# =========================

def main():
    print("开始加载数据...")
    data_map = load_all_data(DATA_DIR)
    print(f"有效股票数: {len(data_map)}")

    if len(data_map) == 0:
        print("没有可用数据。")
        return

    print("开始回测：长柱体+短K线...")
    short_trades, short_daily = run_group_backtest(
        data_map=data_map,
        signal_col="signal_shortk",
        group_name="长柱体+短K线"
    )

    print("开始回测：长柱体+长K线...")
    long_trades, long_daily = run_group_backtest(
        data_map=data_map,
        signal_col="signal_longk",
        group_name="长柱体+长K线"
    )

    compare_df = compare_groups(short_trades, long_trades)
    conclusion_df = generate_conclusion(compare_df)

    # 保存
    short_trades.to_csv(os.path.join(OUTPUT_DIR, "长柱体+短K线_逐笔交易.csv"), index=False, encoding="utf-8-sig")
    long_trades.to_csv(os.path.join(OUTPUT_DIR, "长柱体+长K线_逐笔交易.csv"), index=False, encoding="utf-8-sig")
    short_daily.to_csv(os.path.join(OUTPUT_DIR, "长柱体+短K线_按日汇总.csv"), index=False, encoding="utf-8-sig")
    long_daily.to_csv(os.path.join(OUTPUT_DIR, "长柱体+长K线_按日汇总.csv"), index=False, encoding="utf-8-sig")
    compare_df.to_csv(os.path.join(OUTPUT_DIR, "长短K线对比结果.csv"), index=False, encoding="utf-8-sig")
    conclusion_df.to_csv(os.path.join(OUTPUT_DIR, "自动结论.csv"), index=False, encoding="utf-8-sig")

    print("\n================= 对比结果 =================")
    if compare_df.empty:
        print("没有有效结果，请检查数据。")
    else:
        print(compare_df.to_string(index=False))

    print("\n================= 自动结论 =================")
    if conclusion_df.empty:
        print("暂无结论。")
    else:
        print(conclusion_df.to_string(index=False))

    print("\n结果文件已保存到：")
    print(OUTPUT_DIR)
    print("\n重点看这几个文件：")
    print("1. 长短K线对比结果.csv")
    print("2. 自动结论.csv")
    print("3. 长柱体+短K线_逐笔交易.csv")
    print("4. 长柱体+长K线_逐笔交易.csv")


if __name__ == "__main__":
    main()