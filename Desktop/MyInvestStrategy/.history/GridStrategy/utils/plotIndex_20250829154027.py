# -*- coding: utf-8 -*-
"""
用 AkShare 的 stock_zh_index_spot_em 获取指数清单（如：上证系列指数/深证系列指数/中证系列指数），
再抓取历史行情与估值，输出 5 类图：
1) 近5年的各行业指数 PE（申万月报）
2) 全A（中证全指）PE × 深综指（399106）双轴
3) 全A PB × 深综指 双轴
4) 各宽基指数 PE（沪深300/中证500/中证1000/上证50/中证全指）
5) 宽基估值 × 深综指 多面板

注意：stock_zh_index_spot_em 是“现货快照”，我们用它来找“代码与名称”，
历史K线与估值仍用更合适的接口（EM日线、中证估值、申万月报）。
"""

import warnings, datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak

pd.set_option('display.max_rows', None)        # 行不截断
pd.set_option('display.max_columns', None)     # 列不截断
pd.set_option('display.width', None)           # 允许自动换行到任意宽
pd.set_option('display.max_colwidth', None)    # 列内容不截断（文本很长时有用）

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial Unicode MS','Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ========= 配置 =========
YEARS = 5
TODAY = dt.date.today()
START_5Y = (TODAY.replace(year=TODAY.year-5)).strftime("%Y%m%d")

# 你想用的“spot类别”与“名称关键词”（用于匹配spot里的指数行）
SPOT_WATCH = {
    "上证系列指数": ["上证医药"],                # 000001
    "深证系列指数": ["新能电池"],  # 399106 / 399001 / 399006
    "中证系列指数": ["中证基建"],  # 000300 / 000905 / 000852 / 000985 / 000016  ,"中证500","中证1000","中证全指","上证50"
}

# 宽基指数 -> 中证估值代码（PE/PB用）
CSI_VAL_CODES = {
    "上证医药": "000037",
    "新能电池": "980032",
    "中证基建": "930608"
}

# 深综指（指数点位）用来和全A估值对比
SZ_COMPOSITE_CODE = "399106"  # 深证综合
# ========= 工具 =========
def get_spot_list(category: str) -> pd.DataFrame:
    """从 stock_zh_index_spot_em 拉某个大类的指数清单"""
    file_path = '/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/indexdata/'
    df = ak.stock_zh_index_daily_em(symbol=category, start_date=start_date, end_date=TODAY.strftime("%Y%m%d"))
    # 常见列：'序号','代码','名称','涨跌幅','最新价',... 不同版本略有差异
    df.columns = [str(c).strip() for c in df.columns]
    print(f"正在保存{category}数据到csv文件...")
    df.to_csv(file_path + f"{category}.csv")
    # 只保留我们常用的列
    keep_cols = [c for c in df.columns if c in ["代码","名称","最新价","涨跌幅","今开","最高","最低","成交量"]]
    return df[keep_cols].copy()

def em_symbol_from_code(code: str) -> str:
    """将 6位代码映射为 Eastmoney 的 symbol：000xxx -> sh000xxx；399xxx -> sz399xxx"""
    code = str(code)
    if code.startswith("000"):
        return "sh" + code
    elif code.startswith("399"):
        return "sz" + code
    else:
        # 兜底：常见的国证/中证若是399开头仍走sz；其余尝试sh
        return ("sz" if code.startswith("3") else "sh") + code

def get_index_price_em(symbol_em: str, start=START_5Y, end="20500101") -> pd.DataFrame:
    """用 Eastmoney 接口抓指数日线（建议），symbol 形如 sz399106 / sh000300"""
    df = ak.stock_zh_index_daily_em(symbol=symbol_em, start_date=start, end_date=end)
    df = df.rename(columns={"date":"日期","open":"开盘","high":"最高","low":"最低","close":"收盘","volume":"成交量","amount":"成交额"})
    df["日期"] = pd.to_datetime(df["日期"])
    return df.sort_values("日期")

def get_csindex_valuation(code: str) -> pd.DataFrame:
    """中证估值（宽基）：返回日期-PE1-PE2"""
    df = ak.stock_zh_index_value_csindex(symbol=code)
    df = df.rename(columns={"日期":"date","市盈率1":"PE1","市盈率2":"PE2"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date","PE1","PE2"]].sort_values("date")

def get_sw_monthly_valuation(level="一级行业", last_years=5) -> pd.DataFrame:
    """申万指数分析-月报（行业估值，月频）"""
    months = pd.period_range(end=TODAY, periods=last_years*12, freq="M").astype(str)
    out = []
    for m in months:
        try:
            d = ak.index_analysis_monthly_sw(symbol=level, date=m.replace('-',''))
            d = d.rename(columns={"发布日期":"日期"})
            d["日期"] = pd.to_datetime(d["日期"])
            out.append(d[["日期","代码","名称","市盈率","市净率"]])
        except Exception:
            pass
    if not out:
        return pd.DataFrame()
    big = pd.concat(out, ignore_index=True).drop_duplicates(subset=["日期","代码"])
    return big.sort_values(["名称","日期"])

def get_allA_pe_series() -> pd.DataFrame:
    """全A估值：用中证全指(000985)替代"""
    df = get_csindex_valuation("000985").rename(columns={"date":"日期","PE1":"PE"})
    return df[["日期","PE"]]

def normalize_to_100(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    base = s.iloc[0] if len(s)>0 else np.nan
    return s / base * 100.0 if pd.notna(base) and base!=0 else s

# ========= 用 spot 获取代码 → 抓历史 =========
def build_symbol_map_from_spot(watch: dict) -> dict:
    """
    返回 {名称子串: {'code':'000xxx','em':'sh000xxx/sz399xxx'}} 的映射
    """
    symbol_map = {}
    for cat, name_keys in watch.items():
        try:
            spot = get_spot_list(cat)
        except Exception as e:
            print(f"[WARN] 获取 {cat} 失败：{e}")
            continue
        for key in name_keys:
            hit = spot.loc[spot["名称"].astype(str).str.contains(key, regex=False, na=False)]
            if hit.empty:
                print(f"[WARN] {cat} 未找到：{key}")
                continue
            row = hit.iloc[0]
            code = str(row["代码"]).zfill(6)
            symbol_map[key] = {"code": code, "em": em_symbol_from_code(code)}
    return symbol_map

# ========= 绘图（5类） =========
def plot_industry_pe(sw_monthly: pd.DataFrame, fn="chart_1_industry_PE.png"):
    if sw_monthly.empty:
        print("[WARN] 行业PE为空，跳过图1"); return
    # 选近5年覆盖最好的前若干行业（不然线太多）
    names = (sw_monthly.groupby("名称")["日期"].max()
             .sort_values(ascending=False).index.tolist())[:10]
    dfp = sw_monthly[sw_monthly["名称"].isin(names)]
    plt.figure(figsize=(12,7), dpi=140)
    for name, g in dfp.groupby("名称"):
        plt.plot(g["日期"], g["市盈率"], label=name)
    plt.title("申万行业 PE（月频，近5年）")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(fn)
    print("保存：", fn)

def plot_allA_PE_vs_SZ(symbol_map: dict, fn="chart_2_allA_PE_vs_SZ.png"):
    # 全A（中证全指）PE
    allA = get_allA_pe_series()
    # 深综指（用 spot 找到 em 符号）
    key_sz = "深证综合指数" if "深证综合指数" in symbol_map else "深证成指"
    if key_sz not in symbol_map:
        # 兜底：手写
        sz_em = em_symbol_from_code(SZ_COMPOSITE_CODE)
    else:
        sz_em = symbol_map[key_sz]["em"]
    sz = get_index_price_em(sz_em, START_5Y)
    if allA.empty or sz.empty:
        print("[WARN] 全A或深综指为空，跳过图2"); return
    m = allA.merge(sz[["日期","收盘"]], on="日期", how="inner")
    if m.empty:
        print("[WARN] 对齐失败，跳过图2"); return
    fig, ax1 = plt.subplots(figsize=(12,6), dpi=140)
    ax1.plot(m["日期"], m["PE"], label="全A(中证全指) PE", linewidth=1.5)
    ax1.set_ylabel("PE"); ax1.set_xlabel("日期")
    ax2 = ax1.twinx()
    ax2.plot(m["日期"], m["收盘"], label="深综指", alpha=0.8)
    ax2.set_ylabel("深综指点位")
    ax1.set_title("全A PE × 深综指（近5年，双轴）"); ax1.grid(alpha=0.25)
    lines, labels = [], []
    for ax in [ax1, ax2]:
        L = ax.get_lines(); lines += L; labels += [l.get_label() for l in L]
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout(); plt.savefig(fn); print("保存：", fn)

def plot_allA_PB_vs_SZ(fn="chart_3_allA_PB_vs_SZ.png"):
    # 通过申万“市场表征”月报获取全A PB（月频）
    months = pd.period_range(end=TODAY, periods=YEARS*12, freq="M").astype(str)
    out = []
    for m in months:
        try:
            d = ak.index_analysis_monthly_sw(symbol="市场表征", date=m.replace('-',''))
            d = d.rename(columns={"发布日期":"日期"})
            d["日期"] = pd.to_datetime(d["日期"])
            d = d[d["名称"].str.contains("全A")]
            out.append(d[["日期","名称","市净率"]].rename(columns={"市净率":"PB"}))
        except Exception:
            pass
    if not out:
        print("[WARN] 全A PB获取失败，跳过图3"); return
    allA = pd.concat(out).drop_duplicates(subset=["日期"]).sort_values("日期")
    sz = get_index_price_em(em_symbol_from_code(SZ_COMPOSITE_CODE), START_5Y)
    m = allA.merge(sz[["日期","收盘"]], on="日期", how="inner")
    if m.empty:
        print("[WARN] 对齐失败，跳过图3"); return
    fig, ax1 = plt.subplots(figsize=(12,6), dpi=140)
    ax1.plot(m["日期"], m["PB"], label="全A PB", linewidth=1.5)
    ax2 = ax1.twinx(); ax2.plot(m["日期"], m["收盘"], label="深综指", alpha=0.8)
    ax1.set_ylabel("PB"); ax2.set_ylabel("深综指点位")
    ax1.set_title("全A PB × 深综指（近5年，双轴）"); ax1.grid(alpha=0.25)
    lines, labels = [], []
    for ax in [ax1, ax2]:
        L = ax.get_lines(); lines += L; labels += [l.get_label() for l in L]
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout(); plt.savefig(fn); print("保存：", fn)

def plot_broad_PE(symbol_map: dict, fn1="chart_4_broad_index_PE.png", fn2="chart_5_broad_PE_vs_SZ_panels.png"):
    # 宽基估值（PE）— 用中证估值接口
    pe_map = {}
    for name, csi_code in CSI_VAL_CODES.items():
        try:
            v = get_csindex_valuation(csi_code).rename(columns={"date":"日期","PE1":"PE"})
            pe_map[name] = v[["日期","PE"]]
        except Exception:
            print(f"[WARN] 中证估值拉取失败：{name}")
    if not pe_map:
        print("[WARN] 宽基PE为空，跳过图4/5"); return
    # 图4：多线
    plt.figure(figsize=(12,6), dpi=140)
    for name, df in pe_map.items():
        plt.plot(df["日期"], df["PE"], label=name)
    plt.title("宽基指数 PE（中证估值）")
    plt.legend(ncol=3, fontsize=9); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(fn1); print("保存：", fn1)

    # 图5：每个宽基PE × 深综指（归一）小面板
    sz = get_index_price_em(em_symbol_from_code(SZ_COMPOSITE_CODE), START_5Y)
    if sz.empty:
        print("[WARN] 深综指为空，跳过图5"); return
    import math
    n = len(pe_map); cols = 2; rows = math.ceil(n/cols) + 1
    fig = plt.figure(figsize=(12, 3.2*rows), dpi=140)
    i = 1
    for name, df in pe_map.items():
        ax = fig.add_subplot(rows, cols, i)
        ax.plot(df["日期"], df["PE"], label=f"{name} PE")
        ax.set_title(name); ax.grid(alpha=0.2)
        i += 1
    ax = fig.add_subplot(rows, 1, rows)
    ax.plot(sz["日期"], normalize_to_100(sz["收盘"]), label="深综指(归一)")
    ax.set_title("深综指（归一）"); ax.grid(alpha=0.2)
    fig.tight_layout(); plt.savefig(fn2); print("保存：", fn2)

# ========= 主流程 =========
if __name__ == "__main__":
    print("AkShare 版本：", getattr(ak, "__version__", "unknown"))

    # A) 通过 spot 获取“我们要看的代码”
    symbol_map = build_symbol_map_from_spot(SPOT_WATCH)
    # 示例：你也可以直接看 spot：
    # stock_zh_index_spot_em_df = ak.stock_zh_index_spot_em(symbol="上证系列指数")
    # print(stock_zh_index_spot_em_df.head())

    # B) 行业估值（申万月报） → 图1
    #sw = get_sw_monthly_valuation(level="一级行业", last_years=YEARS)
    #plot_industry_pe(sw, "chart_1_industry_PE.png")

    # C) 全A PE vs 深综指 → 图2
    plot_allA_PE_vs_SZ(symbol_map, "chart_2_allA_PE_vs_SZ.png")

    # D) 全A PB vs 深综指 → 图3
    plot_allA_PB_vs_SZ("chart_3_allA_PB_vs_SZ.png")

    # E) 宽基指数 PE & 面板 → 图4 / 图5
    plot_broad_PE(symbol_map, "chart_4_broad_index_PE.png", "chart_5_broad_PE_vs_SZ_panels.png")

    print("完成。输出：chart_1~chart_5 共5张PNG")