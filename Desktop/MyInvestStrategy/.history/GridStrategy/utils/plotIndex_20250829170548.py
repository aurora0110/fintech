# -*- coding: utf-8 -*-
"""
功能清单
1) 用 stock_zh_index_spot_em 下载目录（上证/深证/中证/国证），只输出你关心的指数
2) 下载“指定指数”的历史数据（日线，东财接口）
3) 绘制：中证全指(000985) PE × 深综指(399106)（双轴）
4) 绘制：全A PB × 深综指（PB用申万“市场表征-全A”近似，双轴）
5) 绘制：中证全指（PE或PB就近）× 深综指（双轴）
6) 绘制：你关心的“行业指数” PE、PB（优先中证估值，缺失则用申万月报）
7) 绘制：你关心的“宽基指数” PE、PB（中证估值为主；海外/港股若无估值则跳过估值，仅保留点位对比提示）

沪深300（上证000300）、上证180（上证000010）、深证100（深证399330）、科创50（上证000688）、创业板指（深证399006）、上证50（上证000016）、中证500（上证000905）、中证1000（上证000852）、恒生科技、恒生医疗、标普500；行业我只关心：全指医药（上证000991）、
全指金融（中证932075）、全指消费、中证环保、全指信息、养老产业、中证医疗、食品饮料、中证红利、中证军工、中证传媒、中国互联

输出：./output/ 下若干 CSV 与 PNG
"""

import warnings, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
# 放在你的 plotIndex.py 顶部（import 之后）
import ssl, certifi, urllib.request
_ctx = ssl.create_default_context(cafile=certifi.where())  # 用 certifi 的 CA 列表
_opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx))
urllib.request.install_opener(_opener)  # 让 pandas/urllib 用这个带 CA 的 opener

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial Unicode MS','Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ========= 目录与时间窗 =========
OUTDIR = Path("./output"); OUTDIR.mkdir(parents=True, exist_ok=True)
YEARS = 5
TODAY = dt.date.today()
START_5Y = (TODAY.replace(year=TODAY.year-5)).strftime("%Y%m%d")

# ========= 你关心的“宽基”和“行业” =========
FOCUS_BROAD = [
    "全市场", "沪深300", "上证180", "深证100", "科创50",
    "创业板指", "上证50", "中证500", "恒生", "标普500",
    "十年国债市盈率", "中证1000", "日经225", "恒生科技",
    "恒生医疗", "标普500"  #（重复不影响，会去重）
]
FOCUS_INDUSTRY = [
    "全指医药","全指金融","全指消费","中证环保","全指信息",
    "养老产业","中证医疗","食品饮料","中证红利","中证军工",
    "中证传媒","中国互联"
]

# ========= spot 分类（做代码发现）=========
SPOT_CATEGORIES = ["上证系列指数", "深证系列指数", "中证系列指数"]

# ========= 常用工具 =========
def save_csv(df: pd.DataFrame, filename: str):
    fp = OUTDIR / filename
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {fp}")

def to_em_symbol(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("000"):  # 上证口径常用 000xxx
        return "sh" + code
    elif code.startswith("399"):  # 深证 399xxx
        return "sz" + code
    # 兜底：根据首位判断
    return ("sz" if code.startswith("3") else "sh") + code

def get_index_history(code: str, start=START_5Y, end="20500101") -> pd.DataFrame:
    """日线行情（东财接口）"""
    sym = to_em_symbol(code)
    df = ak.stock_zh_index_daily_em(symbol=sym, start_date=start, end_date=end)
    df = df.rename(columns={"date":"日期","open":"开盘","high":"最高","low":"最低","close":"收盘",
                            "volume":"成交量","amount":"成交额"})
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期")
    print(f"------------{code}---------:")
    save_csv(df, f"history_{code}.csv")
    return df

def normalize_100(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.empty or pd.isna(s.iloc[0]) or s.iloc[0] == 0:
        return s
    return s / s.iloc[0] * 100.0

# ========= 用 spot 做“名称→代码”映射 =========
def fetch_spot_catalogs(categories):
    cat_list = []
    for cat in categories:
        try:
            df = ak.stock_zh_index_spot_em(symbol=cat).copy()
            df.columns = [str(c).strip() for c in df.columns]
            # 统一关键列
            if "指数代码" not in df.columns or "指数名称" not in df.columns:
                # 兼容处理
                for a,b in [("代码","指数代码"),("名称","指数名称")]:
                    if a in df.columns and b not in df.columns:
                        df[b] = df[a]
            df["category"] = cat
            cat_list.append(df)
            save_csv(df, f"catalog_{cat}.csv")
        except Exception as e:
            print(f"[WARN] 获取 {cat} 目录失败：{e}")
    if not cat_list:
        raise RuntimeError("一个 spot 目录都没抓到，检查网络/AkShare 版本。")
    big = pd.concat(cat_list, ignore_index=True)
    # 精简字段
    keep = [c for c in big.columns if c in ["指数代码","指数名称","category","最新价","涨跌幅","今开","最高","最低","成交量","成交额","更新时间","市场"]]
    big = big[keep].drop_duplicates(subset=["指数代码","指数名称"])
    save_csv(big, "catalog_all.csv")
    return big

def build_symbol_map_by_focus(full_catalog: pd.DataFrame, focus_names: list) -> dict:
    """
    按“包含关系”匹配名称，返回 {目标名: {'code': 6位代码, 'name': 实际命中名称}}。
    若未命中则返回 None。
    """
    res = {}
    for target in focus_names:
        hit = full_catalog.loc[full_catalog["指数名称"].astype(str).str.contains(target, regex=False, na=False)]
        if hit.empty:
            print(f"[WARN] 目录未找到：{target}")
            res[target] = None
        else:
            row = hit.iloc[0]
            res[target] = {"code": str(row["指数代码"]).zfill(6), "name": str(row["指数名称"])}
    return res

# ========= 估值接口 =========
def csindex_valuation(code: str) -> pd.DataFrame:
    """
    中证估值（优先用于：中证/上证部分宽基与行业）。
    返回尽量规范化的列：date, PE, PB（PB可能缺失）
    """
    df = ak.stock_zh_index_value_csindex(symbol=code)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "日期" in df.columns:
        df = df.rename(columns={"日期":"date"})
    df["date"] = pd.to_datetime(df["date"])
    # 识别 PE/PB 列
    pe_col = None; pb_col = None
    for c in df.columns:
        cl = c.lower()
        if ("pe" in cl) or ("市盈" in c):
            pe_col = pe_col or c
        if ("pb" in cl) or ("市净" in c):
            pb_col = pb_col or c
    out = pd.DataFrame({"date": df["date"]})
    if pe_col is not None:
        out["PE"] = pd.to_numeric(df[pe_col], errors="coerce")
    if pb_col is not None:
        out["PB"] = pd.to_numeric(df[pb_col], errors="coerce")
    return out.sort_values("date")

def sw_monthly_valuation(level="一级行业", months=YEARS*12) -> pd.DataFrame:
    """
    申万指数分析-月报：行业估值（PE/PB，月频），用于兜底那些“中证估值没有覆盖到的行业名”
    """
    end = dt.date.today()
    rng = pd.period_range(end=end, periods=months, freq="M").astype(str)
    out = []
    for m in rng:
        try:
            d = ak.index_analysis_monthly_sw(symbol=level, date=m.replace('-',''))
            d = d.rename(columns={"发布日期":"日期"})
            d["日期"] = pd.to_datetime(d["日期"])
            keep = ["日期","指数代码","指数名称"]
            if "市盈率" in d.columns: keep.append("市盈率")
            if "市净率" in d.columns: keep.append("市净率")
            out.append(d[keep])
        except Exception:
            pass
    if not out:
        return pd.DataFrame()
    big = pd.concat(out, ignore_index=True)
    if "市盈率" in big.columns: big = big.rename(columns={"市盈率":"PE"})
    if "市净率" in big.columns: big = big.rename(columns={"市净率":"PB"})
    big = big.drop_duplicates(subset=["日期","指数代码"]).sort_values(["指数名称","日期"])
    return big

def allA_pe_series() -> pd.DataFrame:
    """全市场估值：用中证全指(000985)"""
    df = csindex_valuation("000985")
    if df.empty: return df
    df = df.loc[df["date"] >= pd.Timestamp(START_5Y)]
    print('--------------中证全指————————————————————')
    return df.rename(columns={"date":"日期"})

# ========= 绘图 =========
def plot_allA_PE_vs_SZ(sz_code="399106"):
    """(3) 中证全指 PE × 深综指点位（双轴）"""
    csi = allA_pe_series()
    sz  = get_index_history(sz_code, START_5Y)
    if csi.empty or sz.empty or "PE" not in csi.columns:
        print("[WARN] 全A PE 或 深综指为空，跳过图3")
        return
    m = csi.merge(sz[["日期","收盘"]], on="日期", how="inner")
    if m.empty:
        print("[WARN] 对齐为空，跳过图3")
        return
    fig, ax1 = plt.subplots(figsize=(12,6), dpi=140)
    ax1.plot(m["日期"], m["PE"], label="中证全指 PE", linewidth=1.6)
    ax1.set_ylabel("PE"); ax1.set_xlabel("日期"); ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(m["日期"], m["收盘"], label="深综指(右轴)", alpha=0.8)
    ax2.set_ylabel("深综指点位")
    ax1.set_title("中证全指 PE × 深综指（近5年，双轴）")
    lines, labels = [], []
    for ax in [ax1, ax2]:
        L = ax.get_lines(); lines += L; labels += [l.get_label() for l in L]
    ax1.legend(lines, labels, loc="upper left")
    fn = OUTDIR/"chart_pe_allA_vs_sz.png"; plt.tight_layout(); plt.savefig(fn)
    print(f"[SAVE] {fn}")

def plot_allA_PB_vs_SZ(sz_code="399106"):
    """(4) 全A PB × 深综指（PB用申万‘市场表征-全A’近似，双轴）"""
    months = pd.period_range(end=TODAY, periods=YEARS*12, freq="M").astype(str)
    out = []
    for m in months:
        try:
            d = ak.index_analysis_monthly_sw(symbol="市场表征", date=m.replace('-',''))
            d = d.rename(columns={"发布日期":"日期"})
            d["日期"] = pd.to_datetime(d["日期"])
            hit = d.loc[d["指数名称"].astype(str).str.contains("全A", regex=False, na=False)]
            if "市净率" in hit.columns:
                out.append(hit[["日期","指数名称","市净率"]].rename(columns={"市净率":"PB"}))
        except Exception:
            pass
    if not out:
        print("[WARN] 全A PB获取失败，跳过图4"); return
    allA_pb = pd.concat(out).drop_duplicates(subset=["日期"]).sort_values("日期")
    sz = get_index_history(sz_code, START_5Y)
    m = allA_pb.merge(sz[["日期","收盘"]], on="日期", how="inner")
    if m.empty:
        print("[WARN] 对齐为空，跳过图4"); return
    fig, ax1 = plt.subplots(figsize=(12,6), dpi=140)
    ax1.plot(m["日期"], m["PB"], label="全A PB", linewidth=1.6)
    ax2 = ax1.twinx()
    ax2.plot(m["日期"], m["收盘"], label="深综指(右轴)", alpha=0.8)
    ax1.set_ylabel("PB"); ax2.set_ylabel("深综指点位")
    ax1.set_title("全A PB × 深综指（近5年，双轴）"); ax1.grid(alpha=0.25)
    lines, labels = [], []
    for ax in [ax1, ax2]:
        L = ax.get_lines(); lines += L; labels += [l.get_label() for l in L]
    ax1.legend(lines, labels, loc="upper left")
    fn = OUTDIR/"chart_pb_allA_vs_sz.png"; plt.tight_layout(); plt.savefig(fn)
    print(f"[SAVE] {fn}")

def plot_allA_valuation_vs_SZ(sz_code="399106"):
    """(5) 中证全指（PE优先，PB次之）× 深综指"""
    csi = allA_pe_series()
    sz  = get_index_history(sz_code, START_5Y)
    if csi.empty or sz.empty:
        print("[WARN] 全A或深综指为空，跳过图5"); return
    col = "PE" if "PE" in csi.columns else ("PB" if "PB" in csi.columns else None)
    if col is None:
        print("[WARN] 中证全指无 PE/PB 列，跳过图5"); return
    m = csi.merge(sz[["日期","收盘"]], on="日期", how="inner")
    fig, ax1 = plt.subplots(figsize=(12,6), dpi=140)
    ax1.plot(m["日期"], m[col], label=f"中证全指 {col}", linewidth=1.6)
    ax2 = ax1.twinx(); ax2.plot(m["日期"], m["收盘"], label="深综指(右轴)", alpha=0.8)
    ax1.set_ylabel(col); ax2.set_ylabel("深综指点位")
    ax1.set_title(f"中证全指 {col} × 深综指（近5年，双轴）"); ax1.grid(alpha=0.25)
    lines, labels = [], []
    for ax in [ax1, ax2]:
        L = ax.get_lines(); lines += L; labels += [l.get_label() for l in L]
    ax1.legend(lines, labels, loc="upper left")
    fn = OUTDIR/"chart_allA_valuation_vs_sz.png"; plt.tight_layout(); plt.savefig(fn)
    print(f"[SAVE] {fn}")

def plot_focus_industry_pe_pb(full_catalog: pd.DataFrame):
    """(6) 只画你关心的行业的 PE/PB（优先中证估值；若没有则尝试申万月报近似）"""
    # 先尝试用 spot 找代码 → csindex_valuation
    pe_lines, pb_lines = [], []
    for name in FOCUS_INDUSTRY:
        hit = full_catalog.loc[full_catalog["指数名称"].astype(str).str.contains(name, regex=False, na=False)]
        if hit.empty:
            print(f"[WARN] 行业未命中：{name}"); continue
        code = str(hit.iloc[0]["指数代码"]).zfill(6)
        try:
            v = csindex_valuation(code)
            v = v.loc[v["date"] >= pd.Timestamp(START_5Y)]
            if not v.empty and "PE" in v.columns:
                pe_lines.append((name, v.rename(columns={"date":"日期"})[["日期","PE"]]))
            if not v.empty and "PB" in v.columns:
                pb_lines.append((name, v.rename(columns={"date":"日期"})[["日期","PB"]]))
        except Exception as e:
            print(f"[WARN] 行业 {name} 估值失败：{e}")

    # 若 PE/PB 太少，补一份申万月报（名称不完全一致，仅作兜底）
    if len(pe_lines) + len(pb_lines) == 0:
        sw = sw_monthly_valuation(level="一级行业", months=YEARS*12+6)
        if not sw.empty:
            # 简易匹配：用“包含关系”找最接近的行业名字
            for name in FOCUS_INDUSTRY:
                hit = sw.loc[sw["指数名称"].astype(str).str.contains(name.replace("全指",""), regex=False, na=False)]
                if hit.empty: 
                    continue
                g = hit.sort_values("日期")
                if "PE" in g.columns:
                    pe_lines.append((name, g[["日期","PE"]].rename(columns={"PE":"PE"})))
                if "PB" in g.columns:
                    pb_lines.append((name, g[["日期","PB"]].rename(columns={"PB":"PB"})))

    # 画 PE
    if pe_lines:
        plt.figure(figsize=(12,7), dpi=140)
        for nm, df in pe_lines:
            plt.plot(df["日期"], df["PE"], label=nm)
        plt.title("关注行业 PE（近5年）")
        plt.legend(ncol=3, fontsize=9); plt.grid(alpha=0.25)
        fn = OUTDIR/"chart_focus_industry_PE.png"; plt.tight_layout(); plt.savefig(fn)
        print(f"[SAVE] {fn}")
    else:
        print("[WARN] 行业 PE 曲线为空")

    # 画 PB
    if pb_lines:
        plt.figure(figsize=(12,7), dpi=140)
        for nm, df in pb_lines:
            plt.plot(df["日期"], df["PB"], label=nm)
        plt.title("关注行业 PB（近5年）")
        plt.legend(ncol=3, fontsize=9); plt.grid(alpha=0.25)
        fn = OUTDIR/"chart_focus_industry_PB.png"; plt.tight_layout(); plt.savefig(fn)
        print(f"[SAVE] {fn}")
    else:
        print("[WARN] 行业 PB 曲线为空")

def plot_focus_broad_pe_pb(full_catalog: pd.DataFrame):
    """(7) 只画你关心的宽基的 PE、PB（中证估值）。海外/港股若无估值则跳过该条估值。"""
    pe_lines, pb_lines = [], []
    # 统一“全市场”为中证全指
    name_alias = {"全市场":"中证全指", "科创50":"科创50", "标普500":"标普500", "日经225":"日经225", "恒生":"恒生指数"}
    dedup = []
    for nm in FOCUS_BROAD:
        if nm not in dedup: dedup.append(nm)
    for name in dedup:
        key = name_alias.get(name, name)
        hit = full_catalog.loc[full_catalog["指数名称"].astype(str).str.contains(key, regex=False, na=False)]
        if hit.empty:
            print(f"[WARN] 宽基未命中：{name}"); continue
        code = str(hit.iloc[0]["指数代码"]).zfill(6)
        try:
            v = csindex_valuation(code)  # 中证估值不一定覆盖海外指数
            v = v.loc[v["date"] >= pd.Timestamp(START_5Y)]
            if not v.empty and "PE" in v.columns:
                pe_lines.append((name, v.rename(columns={"date":"日期"})[["日期","PE"]]))
            if not v.empty and "PB" in v.columns:
                pb_lines.append((name, v.rename(columns={"date":"日期"})[["日期","PB"]]))
            if v.empty:
                print(f"[INFO] {name} 可能非中证口径，未提供估值，仅可绘点位/跳过估值。")
        except Exception as e:
            print(f"[WARN] 宽基 {name} 估值失败：{e}")

    # 画 PE
    if pe_lines:
        plt.figure(figsize=(12,6), dpi=140)
        for nm, df in pe_lines:
            plt.plot(df["日期"], df["PE"], label=nm)
        plt.title("关注宽基 PE（中证估值，近5年）")
        plt.legend(ncol=3, fontsize=9); plt.grid(alpha=0.25)
        fn = OUTDIR/"chart_focus_broad_PE.png"; plt.tight_layout(); plt.savefig(fn)
        print(f"[SAVE] {fn}")
    else:
        print("[WARN] 宽基 PE 曲线为空（海外/港股大多无中证估值）")

    # 画 PB
    if pb_lines:
        plt.figure(figsize=(12,6), dpi=140)
        for nm, df in pb_lines:
            plt.plot(df["日期"], df["PB"], label=nm)
        plt.title("关注宽基 PB（中证估值，近5年）")
        plt.legend(ncol=3, fontsize=9); plt.grid(alpha=0.25)
        fn = OUTDIR/"chart_focus_broad_PB.png"; plt.tight_layout(); plt.savefig(fn)
        print(f"[SAVE] {fn}")
    else:
        print("[WARN] 宽基 PB 曲线为空")

# ========= 主流程 =========
if __name__ == "__main__":
    print("AkShare 版本：", getattr(ak, "__version__", "unknown"))
    # 1) 下载目录（spot）
    catalog = fetch_spot_catalogs(SPOT_CATEGORIES)
    # 1.1）仅输出你关心的 index 的目录（方便你核对）
    focus_all = list(set(FOCUS_BROAD + FOCUS_INDUSTRY))
    mask_focus = pd.Series(False, index=catalog.index)
    for nm in focus_all:
        mask_focus |= catalog["指数名称"].astype(str).str.contains(nm, regex=False, na=False)
    focus_catalog = catalog.loc[mask_focus].copy()
    save_csv(focus_catalog, "catalog_focus_only.csv")

    # 2) 示例：下载“深综指、沪深300、科创50”历史（日线）
    for c in ["399106","000300","000688"]:  # 000688 若不存在会自动跳过
        try:
            get_index_history(c, START_5Y)
        except Exception as e:
            print(f"[WARN] 历史抓取失败 {c}：{e}")

    # 3~5) 全A × 深综指系列
    plot_allA_PE_vs_SZ("399106")
    plot_allA_PB_vs_SZ("399106")
    plot_allA_valuation_vs_SZ("399106")

    # 6) 关注行业 PE/PB
    plot_focus_industry_pe_pb(catalog)

    # 7) 关注宽基 PE/PB
    plot_focus_broad_pe_pb(catalog)

    print("=== 检查日期范围 ===")

    print("完成。所有 CSV 与 PNG 已在 ./output")