import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ======== Matplotlib 中文与负号 ========
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial Unicode MS','Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ======== 配置区（可以按需修改）========
YEARS = 5
END_DATE = datetime.today().date()
START_DATE = END_DATE - timedelta(days=365*YEARS)

# 行业（申万一级）优先尝试的中文关键词（按你常看的行业改）
INDUSTRY_KEYS = [
    "银行","非银金融","食品饮料","家用电器","医药生物","电子","计算机","汽车",
    "煤炭","有色金属","电力及公用事业","社会服务","建筑材料","电力设备","机械设备"
]

# 宽基指数（基金大全估值里常见的中文名关键词）
BROAD_INDEX_KEYS = ["沪深300","中证500","中证1000","上证50","中证800"]

# 市场对比指数（“深综指”=399106；若想改成深成指=399001）
SZ_INDEX_CODE = "399106"

# ======== 常用函数 ========
def _dt(s):
    """转日期"""
    return pd.to_datetime(s, errors="coerce")

def _clip_5y(df, date_col="date"):
    """裁剪近5年"""
    if date_col not in df.columns:
        return df
    df = df.copy()
    m = (df[date_col] >= pd.Timestamp(START_DATE)) & (df[date_col] <= pd.Timestamp(END_DATE))
    return df.loc[m].reset_index(drop=True)

def list_funddb_names():
    """列出基金大全支持的指数估值名称，便于排错"""
    try:
        names = ak.index_value_name_funddb()
        # 规范列名
        cols = [c.strip() for c in names.columns]
        names.columns = cols
        return names
    except Exception as e:
        print("[WARN] 无法获取估值名称清单：", e)
        return pd.DataFrame()

def fetch_funddb_series(keyword: str, indicator: str):
    """
    从基金大全（funddb）估值接口，按关键字模糊匹配一个指数，然后取其历史估值（市盈率/市净率）
    indicator 可选：'市盈率' 或 '市净率'
    """
    names = list_funddb_names()
    if names.empty:
        return None

    # 估值名称列在不同 ak 版本里字段名可能不同，尝试多个
    cand_col = None
    for c in ["name","指数名称","symbol","指数名称(或symbol)"]:
        if c in names.columns:
            cand_col = c
            break
    if cand_col is None:
        cand_col = names.columns[0]

    # 过滤出包含关键字的候选
    mask_kw = names[cand_col].astype(str).str.contains(keyword, regex=False, na=False)
    mask_ind = pd.Series([True]*len(names))
    # 有些版本会提供“指标”或“indicator”列
    for ind_col in ["指标","indicator","Indicator"]:
        if ind_col in names.columns:
            mask_ind = names[ind_col].astype(str).str.contains(indicator, regex=False, na=False)
            break

    picks = names.loc[mask_kw & mask_ind]
    if picks.empty:
        # 宽松匹配：只按关键字
        picks = names.loc[mask_kw]

    if picks.empty:
        print(f"[WARN] 没在 funddb 名单中找到匹配：keyword={keyword}, indicator={indicator}")
        return None

    # 用第一条匹配（也可以打印让用户确认）
    chosen = str(picks.iloc[0][cand_col])
    # 试多种 indicator 书写
    indicators_try = [indicator, f"{indicator}TTM", f"{indicator}(TTM)", f"{indicator}（TTM）"]
    for ind in indicators_try:
        try:
            df = ak.index_value_hist_funddb(symbol=chosen, indicator=ind)
            if not df.empty:
                # 规范列名
                df.columns = [str(c).strip() for c in df.columns]
                # 常见形态：date / value
                # 各版本列名可能是：'date','pe','pb','close','估值','中位数' 等；统一为 date, value
                if "date" not in df.columns:
                    # 尝试中文列名
                    for dcol in ["日期","time","时间"]:
                        if dcol in df.columns:
                            df = df.rename(columns={dcol:"date"})
                            break
                # 值列：优先 pe/pb/估值
                vcol = None
                for vc in ["pe","pb","value","估值","PE","PB","pe_ttm","pb_lf"]:
                    if vc in df.columns:
                        vcol = vc
                        break
                if vcol is None:
                    # 尝试最后一列
                    vcol = df.columns[-1]
                df = df[["date", vcol]].rename(columns={vcol:"value"})
                df["date"] = _dt(df["date"])
                df = df.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)
                return df
        except Exception:
            continue

    print(f"[WARN] {keyword} 的 {indicator} 序列抓取失败（funddb）。")
    return None

def fetch_sz_index(code: str):
    """抓取深证指数日线（AkShare: index_zh_a_hist）"""
    try:
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=START_DATE.strftime("%Y%m%d"),
                                end_date=END_DATE.strftime("%Y%m%d"), adjust="")
        # 规范列
        ren = {"日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"volume","成交额":"amount"}
        df = df.rename(columns=ren)
        df["date"] = _dt(df["date"])
        df = df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[WARN] 抓取深市指数 {code} 失败：", e)
        return None

def align_left_right(left: pd.DataFrame, right: pd.DataFrame, on="date"):
    """按日期对齐并去重"""
    if left is None or right is None or left.empty or right.empty:
        return None
    df = pd.merge(left, right[["date","close"]], on=on, how="inner", suffixes=("","_idx"))
    df = df.dropna(subset=["value","close"]).sort_values("date").reset_index(drop=True)
    return df

def normalize(series: pd.Series):
    """把序列归一化到起点=100"""
    s = series.astype(float)
    if s.empty or s.isna().all():
        return s
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return s
    return s / base * 100.0

# ======== 1) 各行业 PE（近5年，多线）========
def plot_industry_pe():
    got = []
    for kw in INDUSTRY_KEYS:
        ser = fetch_funddb_series(kw, "市盈率")
        if ser is None or ser.empty:
            continue
        ser = _clip_5y(ser)
        if len(ser) < 30:
            continue
        got.append((kw, ser))
    if not got:
        print("[WARN] 行业PE一个都没抓到，检查关键词或 funddb 名称。")
        return
    plt.figure(figsize=(12,7))
    for kw, ser in got:
        plt.plot(ser["date"], ser["value"], label=kw)
    plt.title(f"行业指数市盈率（近{YEARS}年）")
    plt.xlabel("日期"); plt.ylabel("PE（TTM/近似）")
    plt.legend(ncol=3, fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_1_industry_PE.png", dpi=150)
    print("已保存：chart_1_industry_PE.png")

# ======== 2) 全A（中证全指）PE × 深综指（双轴）========
def plot_allA_pe_vs_sz():
    allA_pe = fetch_funddb_series("中证全指", "市盈率")
    sz = fetch_sz_index(SZ_INDEX_CODE)
    allA_pe = _clip_5y(allA_pe) if allA_pe is not None else None
    sz = _clip_5y(sz) if sz is not None else None
    df = align_left_right(allA_pe, sz)
    if df is None or df.empty:
        print("[WARN] 全A PE × 深综指 对齐失败。")
        return
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(df["date"], df["value"], label="全A（中证全指）PE", linewidth=1.5)
    ax1.set_ylabel("PE")
    ax1.set_xlabel("日期")
    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["close"], label="深综指（右轴）", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("深综指点位")
    ax1.set_title(f"全A（中证全指）PE × 深综指（近{YEARS}年，双轴）")
    ax1.grid(alpha=0.3)
    # 合并图例
    lines, labels = [], []
    for ax in [ax1, ax2]:
        L = ax.get_lines()
        lines += L
        labels += [l.get_label() for l in L]
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout()
    plt.savefig("chart_2_allA_PE_vs_SZ.png", dpi=150)
    print("已保存：chart_2_allA_PE_vs_SZ.png")

# ======== 3) 全A PB × 深综指（双轴）========
def plot_allA_pb_vs_sz():
    allA_pb = fetch_funddb_series("中证全指", "市净率")
    sz = fetch_sz_index(SZ_INDEX_CODE)
    allA_pb = _clip_5y(allA_pb) if allA_pb is not None else None
    sz = _clip_5y(sz) if sz is not None else None
    df = align_left_right(allA_pb, sz)
    if df is None or df.empty:
        print("[WARN] 全A PB × 深综指 对齐失败。")
        return
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(df["date"], df["value"], label="全A（中证全指）PB", linewidth=1.5)
    ax1.set_ylabel("PB")
    ax1.set_xlabel("日期")
    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["close"], label="深综指（右轴）", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("深综指点位")
    ax1.set_title(f"全A（中证全指）PB × 深综指（近{YEARS}年，双轴）")
    ax1.grid(alpha=0.3)
    lines, labels = [], []
    for ax in [ax1, ax2]:
        L = ax.get_lines()
        lines += L
        labels += [l.get_label() for l in L]
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout()
    plt.savefig("chart_3_allA_PB_vs_SZ.png", dpi=150)
    print("已保存：chart_3_allA_PB_vs_SZ.png")

# ======== 4) 各宽基指数 PE（多线）========
def plot_broad_index_pe():
    got = []
    for kw in BROAD_INDEX_KEYS:
        ser = fetch_funddb_series(kw, "市盈率")
        if ser is None or ser.empty:
            continue
        ser = _clip_5y(ser)
        if len(ser) < 30:
            continue
        got.append((kw, ser))
    if not got:
        print("[WARN] 宽基指数 PE 抓取为空。")
        return
    plt.figure(figsize=(12,7))
    for kw, ser in got:
        plt.plot(ser["date"], ser["value"], label=kw)
    plt.title(f"各宽基指数 PE（近{YEARS}年）")
    plt.xlabel("日期"); plt.ylabel("PE（TTM/近似）")
    plt.legend(ncol=3, fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_4_broad_index_PE.png", dpi=150)
    print("已保存：chart_4_broad_index_PE.png")

# ======== 5) 宽基估值 × 深综指（多面板，双轴）========
def plot_broad_vs_sz_panels():
    sz = fetch_sz_index(SZ_INDEX_CODE)
    sz = _clip_5y(sz) if sz is not None else None
    if sz is None or sz.empty:
        print("[WARN] 深综指抓取失败，无法绘制多面板。")
        return

    n = len(BROAD_INDEX_KEYS)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5*nrows), sharex=True)
    axes = axes.flatten()

    i = 0
    for kw in BROAD_INDEX_KEYS:
        ser = fetch_funddb_series(kw, "市盈率")
        if ser is None or ser.empty:
            continue
        ser = _clip_5y(ser)
        df = align_left_right(ser, sz)
        if df is None or df.empty:
            continue
        ax1 = axes[i]
        ax1.plot(df["date"], df["value"], label=f"{kw} PE", linewidth=1.4)
        ax1.set_ylabel("PE")
        ax2 = ax1.twinx()
        ax2.plot(df["date"], normalize(df["close"]), label="深综指(归一化)", linewidth=1.0, alpha=0.8)
        ax1.grid(alpha=0.25)
        ax1.set_title(f"{kw} PE × 深综指（右轴归一）")
        # 合并图例（局部）
        lines, labels = [], []
        for ax in [ax1, ax2]:
            L = ax.get_lines(); lines += L; labels += [l.get_label() for l in L]
        ax1.legend(lines, labels, loc="upper left", fontsize=9)
        i += 1

    # 清理多余子图
    while i < len(axes):
        fig.delaxes(axes[i])
        i += 1

    plt.tight_layout()
    plt.savefig("chart_5_broad_PE_vs_SZ_panels.png", dpi=150)
    print("已保存：chart_5_broad_PE_vs_SZ_panels.png")

# ======== 主流程 ========
if __name__ == "__main__":
    print("开始绘图（最近5年）...")
    plot_industry_pe()
    plot_allA_pe_vs_sz()
    plot_allA_pb_vs_sz()
    plot_broad_index_pe()
    plot_broad_vs_sz_panels()
    print("全部完成。输出文件：")
    print(" - chart_1_industry_PE.png")
    print(" - chart_2_allA_PE_vs_SZ.png")
    print(" - chart_3_allA_PB_vs_SZ.png")
    print(" - chart_4_broad_index_PE.png")
    print(" - chart_5_broad_PE_vs_SZ_panels.png")