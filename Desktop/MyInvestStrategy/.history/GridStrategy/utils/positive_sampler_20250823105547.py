from calendar import month
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from pathlib import Path
import getData

# ---- 列名适配（中英 & 不区分大小写）----
COL_MAP = {
    "日期":"date","时间":"date","trade_date":"date","date":"date",
    "开盘":"open","开盘价":"open","open":"open",
    "最高":"high","最高价":"high","high":"high",
    "最低":"low","最低价":"low","low":"low",
    "收盘":"close","收盘价":"close","close":"close",
    "涨跌幅":"pct_chg","涨跌幅(%)":"pct_chg","pct_chg":"pct_chg",
    "昨收":"pre_close","前收":"pre_close","昨收盘":"pre_close","pre_close":"pre_close",
    "成交量":"volume","成交额":"amount","volume":"volume","amount":"amount",
    "代码":"code","股票代码":"code","ts_code":"code","code":"code",
    "名称":"name","股票名称":"name","name":"name"
}

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    # 列名标准化 + 基本类型
    df = df.rename(columns={c: COL_MAP.get(str(c).strip(), COL_MAP.get(str(c).strip().lower(), str(c))) for c in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open","high","low","close","volume","amount","pre_close","pct_chg"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ===== 读取待下载列表（代码、名称） =====
# 从本地文件导入股票代码
file_path = '/Users/lidongyang/Desktop/涨停20250822.csv'
file_path = Path(file_path)

stock_symbol_list = []
df_list = pd.read_csv(file_path)
for _, row in df_list.iterrows():
    code = str(row[0]).strip().zfill(6)   # 股票代码
    if code.startswith('#'):
        continue
    stock_symbol_list.append(code)

print("待下载代码：", stock_symbol_list)

# ===== 时间区间 =====
now = datetime.now()
end_date = now.strftime("%Y%m%d")
stock_years_ago = now - relativedelta(months=6)
stock_start_date = stock_years_ago.strftime("%Y%m%d")

# ===== 批量下载（返回 dict[str, DataFrame]）=====
data_new = getData.batch_download_stock_data(
    stock_symbol_list, "all", stock_start_date, end_date, 5
)

# ===== 关键修复：先删除子表自带的“代码列”，避免与之后的 code 冲突 =====
CODE_COL_CAND = {"code","ts_code","股票代码","代码"}
cleaned = {}
for k, df in data_new.items():
    x = df.copy()
    # 先做列名标准化，再删
    x = _norm_cols(x)
    drop_cols = [c for c in x.columns if str(c).strip().lower() in CODE_COL_CAND or str(c) in CODE_COL_CAND]
    x = x.drop(columns=drop_cols, errors="ignore")
    cleaned[k] = x

# ===== 拼接为总表，把字典 key 变成唯一的 code 列 =====
out = pd.concat(cleaned, names=["code"]).reset_index(level=0).rename(columns={"level_0": "code"})

# 保险：如果仍有重复列名（极端情况），合并同名列并只保留第一个
if out.columns.duplicated().any():
    for col in out.columns[out.columns.duplicated()].unique():
        # 用第一个同名列填充缺失，再删除后续重复列
        first = out.columns.get_loc(col)
        dup_idx = [i for i, c in enumerate(out.columns) if c == col and i != first]
        for di in dup_idx:
            out.iloc[:, first] = out.iloc[:, first].fillna(out.iloc[:, di])
        out = out.drop(out.columns[dup_idx], axis=1)

# ===== 再次统一列类型（保证稳妥）=====
out = _norm_cols(out)

# ===== 处理 pct_chg：若无则用 close/pre_close 计算；若是字符串或 0.xx 小数，归一化到百分比 =====
if "pct_chg" not in out.columns and {"close","pre_close"}.issubset(out.columns):
    out["pct_chg"] = (out["close"] / out["pre_close"] - 1.0) * 100.0

if "pct_chg" in out.columns:
    if out["pct_chg"].dtype == object:
        out["pct_chg"] = out["pct_chg"].astype(str).str.replace("%", "", regex=False)
        out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="coerce")
    # 若最大绝对值很小（≤3），视作 0.xx 小数，放大到百分比
    try:
        if out["pct_chg"].abs().max() <= 3:
            out["pct_chg"] = out["pct_chg"] * 100.0
    except Exception:
        pass

# ===== 新增“是否涨停”列（0/1），规则：pct_chg > 9 则 1，否则 0 =====
out["是否涨停"] = 0
if "pct_chg" in out.columns:
    out.loc[out["pct_chg"] > 9, "是否涨停"] = 1
out["是否涨停"] = out["是否涨停"].fillna(0).astype("int8")  # 强制 0/1

# ===== 排序并输出 CSV（确保“是否涨停”写入）=====
preferred_order = [
    "code","name","date","open","high","low","close","pre_close","pct_chg","volume","amount","是否涨停"
]
cols = [c for c in preferred_order if c in out.columns] + [c for c in out.columns if c not in preferred_order]

# 这里不会再报 'code' not unique，因为我们已去重并只保留一个 'code'
out_final = out[cols].sort_values(["code","date"], ascending=[True, True], na_position="last")

# 强保证
assert "是否涨停" in out_final.columns, "未生成‘是否涨停’列，请检查前面逻辑。"
assert out_final.columns.is_unique, "列名仍不唯一，请检查上游数据源的重复列。"

out_final.to_csv("positive_combined.csv", index=False, encoding="utf-8-sig")
print("已写出：positive_combined.csv，行数 =", len(out_final))