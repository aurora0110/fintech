import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==== Matplotlib 中文/负号 ====
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial Unicode MS','Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ==== 配置 ====
YEARS = 5
END_DATE = datetime.today().date()
START_DATE = END_DATE - timedelta(days=365*YEARS)

# 常用行业关键词（funddb 估值名会做模糊匹配；匹配不到就跳过）
INDUSTRY_KEYS = [
    "银行","非银金融","食品饮料","家用电器","医药生物","电子","计算机","汽车",
    "煤炭","有色金属","电力及公用事业","社会服务","建筑材料","电力设备","机械设备"
]
BROAD_INDEX_KEYS = ["沪深300","中证500","中证1000","上证50","中证800"]

# 深综指：399106；如想改用深成指，换成 "399001"
SZ_INDEX_CODE = "399106"

# ==== 工具函数 ====
def _dt(s): return pd.to_datetime(s, errors="coerce")

def _clip_5y(df, date_col="date"):
    if df is None or df.empty or date_col not in df.columns:
        return df
    df = df.copy()
    m = (df[date_col] >= pd.Timestamp(START_DATE)) & (df[date_col] <= pd.Timestamp(END_DATE))
    return df.loc[m].reset_index(drop=True)

def has_attr(name: str) -> bool:
    return hasattr(ak, name)

def fetch_sz_index_daily(code: str):
    """
    深市指数日线（AkShare 旧版没有 adjust 参数）
    """
    try:
        df = ak.index_zh_a_hist(symbol=code, period="daily",
                                start_date=START_DATE.strftime("%Y%m%d"),
                                end_date=END_DATE.strftime("%Y%m%d"))
        ren = {"日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"volume","成交额":"amount"}
        df = df.rename(columns=ren)
        df["date"] = _dt(df["date"])
        df = df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)
        return df
    except TypeError:
        # 某些极旧版只支持 start_date，不支持 end_date；或参数名不同
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=START_DATE.strftime("%Y%m%d"))
        ren = {"日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"volume","成交额":"amount"}
        df = df.rename(columns=ren)
        df["date"] = _dt(df["date"])
        df = df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)
        return _clip_5y(df)
    except Exception as e:
        print(f"[WARN] 抓取指数 {code} 失败：{e}")
        return None

def try_funddb_hist(symbol_kw: str, indicator_kw: str):
    """
    估值序列（优先 funddb），兼容：没有 funddb 接口时返回 None
    """
    if not has_attr("index_value_hist_funddb"):
        print("[WARN] 你的 AkShare 版本缺少 funddb 估值接口；先画价格线，升级后自动启用估值。")
        return None
    # 直接试“symbol_kw + indicator_kw”的组合
    # funddb 的 symbol 需要完整名称；这里做一轮宽松尝试
    candidates = [
        symbol_kw,
        symbol_kw + " 指数",
        symbol_kw + "（指数）",
        symbol_kw + "指数",
    ]
    indicators_try = [indicator_kw, f"{indicator_kw}TTM", f"{indicator_kw}(TTM)", f"{indicator_kw}（TTM）"]
    for sym in candidates:
        for ind in indicators_try:
            try:
                df = ak.index_value_hist_funddb(symbol=sym, indicator=ind)
                if df is not None and not df.empty:
                    df.columns = [str(c).strip() for c in df.columns]
                    # 统一列名
                    if "date" not in df.columns:
                        for dcol in ["日期","time","时间"]:
                            if dcol in df.columns:
                                df = df.rename(columns={dcol:"date"})
                                break
                    vcol = None
                    for vc in ["pe","pb","value","估值","PE","PB","pe_ttm","pb_lf"]:
                        if vc in df.columns:
                            vcol = vc
                            break
                    if vcol is None:
                        vcol = df.columns[-1]
                    out = df[["date", vcol]].rename(columns={vcol:"value"})
                    out["date"] = _dt(out["date"])
                    out = out.dropna(subset=["date","value"]).sort_values("date").reset_index(drop=True)
                    return out
            except Exception:
                continue
    print(f"[WARN] funddb 未匹配到：{symbol_kw} - {indicator_kw}")
    return None

def align_left_right(left: pd.DataFrame, right: pd.DataFrame, on="date"):
    if left is None or right is None or left.empty or right.empty: return None
    df = pd.merge(left, right[["date","close"]], on=on, how="inner", suffixes=("","_idx"))
    return df.dropna(subset=["value","close"]).sort_values("date").reset_index(drop=True)

def normalize(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    if s.empty or s.isna().all(): return s
    base = s.iloc[0]
    if pd.isna(base) or base == 0: return s
    return s / base * 100.0

# ==== 1) 行业 PE ====
def plot_industry_pe():
    got = []
    for kw in INDUSTRY_KEYS:
        ser = try_funddb_hist(kw, "市盈率")
        if ser is None: 
            continue
        ser = _clip_5y(ser)
        if len(ser) < 30: 
            continue
        got.append((kw, ser))
    if not got:
        print("[WARN] 行业PE没有取到（估值接口缺失或关键词未匹配）。")
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

# ==== 2) 全A(中证全指) PE × 深综指 ====
def plot_allA_pe_vs_sz():
    allA_pe = try_funddb_hist("中证全指", "市盈率")
    sz = fetch_sz_index_daily(SZ_INDEX_CODE)
    allA_pe = _clip_5y(allA_pe) if allA_pe is not None else None
    sz = _clip_5y(sz) if sz is not None else None
    df = align_left_right(allA_pe, sz)
    if df is None or df.empty:
        print("[WARN] 全A PE × 深综指 对齐失败（可能缺估值接口或指数取数失败）。")
        return
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(df["date"], df["value"], label="全A（中证全指）PE", linewidth=1.5)
    ax1.set_ylabel("PE"); ax1.set_xlabel("日期")
    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["close"], label="深综指（右轴）", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("深综指点位")
    ax1.set_title(f"全A（中证全指）PE × 深综指（近{YEARS}年，双轴）")
    ax1.grid(alpha=0.3)
    lines, labels = [], []
    for ax in [ax1, ax2]:
        L = ax.get_lines(); lines += L; labels += [l.get_label() for l in L]
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout(); plt.savefig("chart_2_allA_PE_vs_SZ.png", dpi=150)
    print("已保存：chart_2_allA_PE_vs_SZ.png")

# ==== 3) 全A PB × 深综指 ====
def plot_allA_pb_vs_sz():
    allA_pb = try_funddb_hist("中证全指", "市净率")
    sz = fetch_sz_index_daily(SZ_INDEX_CODE)
    allA_pb = _clip_5y(allA_pb) if allA_pb is not None else None
    sz = _clip_5y(sz) if sz is not None else None
    df = align_left_right(allA_pb, sz)
    if df is None or df.empty:
        print("[WARN] 全A PB × 深综指 对齐失败（可能缺估值接口或指数取数失败）。")
        return
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(df["date"], df["value"], label="全A（中证全指）PB", linewidth=1.5)
    ax1.set_ylabel("PB"); ax1.set_xlabel("日期")
    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["close"], label="深综指（右轴）", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("深综指点位")
    ax1.set_title(f"全A（中证全指）PB × 深综指（近{YEARS}年，双轴）")
    ax1.grid(alpha=0.3)
    lines, labels = [], []
    for ax in [ax1, ax2]:
        L = ax.get_lines(); lines += L; labels += [l.get_label() for l in L]
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout(); plt.savefig("chart_3_allA_PB_vs_SZ.png", dpi=150)
    print("已保存：chart_3_allA_PB_vs_SZ.png")

# ==== 4) 各宽基指数 PE ====
def plot_broad_index_pe():
    got = []
    for kw in BROAD_INDEX_KEYS:
        ser = try_funddb_hist(kw, "市盈率")
        if ser is None: 
            continue
        ser = _clip_5y(ser)
        if len(ser) < 30: 
            continue
        got.append((kw, ser))
    if not got:
        print("[WARN] 宽基 PE 未取到（估值接口缺失或关键词未匹配）。")
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

# ==== 5) 宽基估值 × 深综指（面板） ====
def plot_broad_vs_sz_panels():
    sz = fetch_sz_index_daily(SZ_INDEX_CODE)
    sz = _clip_5y(sz) if sz is not None else None
    if sz is None or sz.empty:
        print("[WARN] 深综指抓取失败，无法绘制多面板。")
        return
    n = len(BROAD_INDEX_KEYS); ncols = 2; nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5*nrows), sharex=True)
    axes = axes.flatten()
    used = 0
    for kw in BROAD_INDEX_KEYS:
        ser = try_funddb_hist(kw, "市盈率")
        if ser is None or ser.empty: 
            continue
        ser = _clip_5y(ser)
        df = align_left_right(ser, sz)
        if df is None or df.empty: 
            continue
        ax1 = axes[used]
        ax1.plot(df["date"], df["value"], label=f"{kw} PE", linewidth=1.4)
        ax1.set_ylabel("PE")
        ax2 = ax1.twinx()
        ax2.plot(df["date"], normalize(df["close"]), label="深综指(归一化)", linewidth=1.0, alpha=0.8)
        ax1.grid(alpha=0.25); ax1.set_title(f"{kw} PE × 深综指（右轴归一）")
        lines, labels = [], []
        for ax in [ax1, ax2]:
            L = ax.get_lines(); lines += L; labels += [l.get_label() for l in L]
        ax1.legend(lines, labels, loc="upper left", fontsize=9)
        used += 1
        if used >= len(axes): break
    # 清理多余子图
    for j in range(used, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout(); plt.savefig("chart_5_broad_PE_vs_SZ_panels.png", dpi=150)
    print("已保存：chart_5_broad_PE_vs_SZ_panels.png")

# ==== 主流程 ====
if __name__ == "__main__":
    print("开始绘图（最近5年）...")
    plot_industry_pe()
    plot_allA_pe_vs_sz()
    plot_allA_pb_vs_sz()
    plot_broad_index_pe()
    plot_broad_vs_sz_panels()
    print("完成。输出：chart_1~chart_5 共 5 张 PNG。")