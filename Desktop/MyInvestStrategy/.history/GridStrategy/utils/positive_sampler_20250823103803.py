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

# ===== 读取待下载列表（代码、名称） =====
file_path = '/Users/lidongyang/Desktop/涨停20250822.csv'
file_path = Path(file_path)

stock_symbol_list = []
df_list = pd.read_csv(file_path)
for _, row in df_list.iterrows():
    code = str(row[0]).strip().zfill(6)   # 股票代码
    name = str(row[1]).strip()            # 股票名称（未使用，但保留）
    if code.startswith('#'):
        continue
    stock_symbol_list.append(code)

print("待下载代码：", stock_symbol_list)

# ===== 时间区间 =====
now = datetime.now()
end_date = now.strftime("%Y%m%d")
stock_years_ago = now - relativedelta(months=6)
stock_start_date = stock_years_ago.strftime("%Y%m%d")

# ===== 批量下载（假设返回：dict[str, DataFrame]）=====
data_new = getData.batch_download_stock_data(
    stock_symbol_list, "all", stock_start_date, end_date, 5
)

# ===== 拼接为总表，把字典的 key 变成 code 列 =====
out = pd.concat(data_new, names=["code"]).reset_index(level=0).rename(columns={"level_0": "code"})

# ===== 列名标准化 & 类型统一 =====
out = out.rename(columns={c: COL_MAP.get(str(c).strip(), COL_MAP.get(str(c).strip().lower(), str(c))) for c in out.columns})

if "date" in out.columns:
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
for c in ("open","high","low","close","volume","amount","pre_close","pct_chg"):
    if c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

# ===== 处理 pct_chg：若无则用 close/pre_close 计算；若是字符串或小数，归一化到“百分比数值” =====
if "pct_chg" not in out.columns and {"close","pre_close"}.issubset(out.columns):
    out["pct_chg"] = (out["close"] / out["pre_close"] - 1.0) * 100.0

if "pct_chg" in out.columns:
    # 若原始是字符串（含 %），先去掉 %
    if out["pct_chg"].dtype == object:
        out["pct_chg"] = out["pct_chg"].astype(str).str.replace("%", "", regex=False)
        out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="coerce")
    # 若看起来是小数（最大绝对值 <= 3% 视为 0.xx 小数），放大到百分比
    try:
        if out["pct_chg"].abs().max() <= 3:
            out["pct_chg"] = out["pct_chg"] * 100.0
    except Exception:
        pass

# ===== 新增“是否涨停”列（0/1），规则：pct_chg > 9 则 1，否则 0 =====
# 若没有 pct_chg，默认不涨停（0）
out["是否涨停"] = 0
if "pct_chg" in out.columns:
    out.loc[out["pct_chg"] > 9, "是否涨停"] = 1
out["是否涨停"] = out["是否涨停"].fillna(0).astype("int8")  # 强制为 0/1 整型

# ===== 排序并输出 CSV（确保“是否涨停”写入）=====
# 常见列排序，其他列放后面
preferred_order = [
    "code","name","date","open","high","low","close","pre_close","pct_chg","volume","amount","是否涨停"
]
cols = [c for c in preferred_order if c in out.columns] + [c for c in out.columns if c not in preferred_order]
out_final = out[cols].sort_values(["code","date"], ascending=[True, True], na_position="last")

# 强保证：必须包含“是否涨停”列
assert "是否涨停" in out_final.columns, "未生成‘是否涨停’列，请检查前面逻辑。"

out_final.to_csv("positive_combined.csv", index=False, encoding="utf-8-sig")
print("已写出：positive_combined.csv，行数 =", len(out_final), "，列：", list(out_final.columns))