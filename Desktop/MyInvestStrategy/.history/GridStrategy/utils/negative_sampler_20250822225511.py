#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Negative Sample Sampler for A-share Daily Data
----------------------------------------------
从一批股票的日线CSV中筛选 6 类“负样本”（失败样本），用于校准/训练评分系统。

输入：
  - 默认读取某目录下所有 *.csv（可用 --input_dir 或 --input_glob 指定）
  - CSV 列名可为中文或英文，自动识别：
      日期/时间, 开盘/最高/最低/收盘/成交量, （可选：成交额）
输出：
  - negatives_combined.csv                 # 所有负样本合并（带类型与理由）
  - neg_fake_breakout.csv                  # 1) 假突破
  - neg_stall_on_volume.csv                # 2) 放量滞涨
  - neg_vol_expand.csv                     # 3) 波动扩张
  - neg_divergence.csv                     # 4) 量价背离（OBV不创新高）
  - neg_overextension.csv                  # 5) 过度乖离（相对MA10）
  - neg_low_turnover_limitup_dump.csv      # 6) 低换手涨停/一字后次日砸

用法：
  python negative_sampler.py --input_dir ./all_csv --out_dir ./neg_out
  或
  python negative_sampler.py --input_glob "./data/*.csv" --out_dir ./neg_out

参数（可调）：见 argparse 部分。

说明：
  这是基于日线的启发式筛选，避免未来函数，尽量在 T 日用 T 前的数据判定。
  为兼容不同数据源，使用成交量中位数/均值等“相对量化”阈值。
"""

import os, glob, argparse
import pandas as pd
import numpy as np

# ---------------- 工具函数 ---------------- #

COL_MAP = {
    "日期":"date","时间":"date",
    "开盘":"open","开盘价":"open",
    "最高":"high","最高价":"high",
    "最低":"low","最低价":"low",
    "收盘":"close","收盘价":"close",
    "成交量":"volume","成交额":"amount",
    "date":"date","open":"open","high":"high","low":"low","close":"close",
    "volume":"volume","amount":"amount"
}

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    m = {c: COL_MAP.get(str(c).strip(), COL_MAP.get(str(c).strip().lower(), str(c).strip().lower()))
         for c in df.columns}
    df = df.rename(columns=m)
    return df

def read_csv_auto(path: str, encs=("utf-8","utf-8-sig","gbk","ansi","latin1")) -> pd.DataFrame:
    last_err = None
    for e in encs:
        try:
            df = pd.read_csv(path, encoding=e)
            return df
        except Exception as ex:
            last_err = ex
    raise RuntimeError(f"读取失败：{path}；最后错误：{last_err}")

def closing_range(H, L, C) -> float:
    rng = H - L
    if rng <= 0 or np.isnan(rng): return 0.5
    return float((C - L) / rng)

def true_range(h, l, c, c_prev):
    return np.maximum(h - l, np.maximum(abs(h - c_prev), abs(l - c_prev)))

def obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    # On Balance Volume
    delta = close.diff().fillna(0.0)
    sign = np.sign(delta)
    return (sign * volume).cumsum()

def pct(a, b):
    return (a / np.maximum(b, 1e-12)) - 1.0

def ensure_num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = ensure_num(df, ["open","high","low","close","volume"])
    df = df.dropna(subset=["date","open","high","low","close","volume"]).sort_values("date").reset_index(drop=True)
    return df

# ---------------- 6 类负样本判定 ---------------- #

def detect_fake_breakout(df: pd.DataFrame,
                         lookback_high: int = 20,
                         lookback_vol: int = 60,
                         breakout_buf: float = 0.002,
                         cr_thresh: float = 0.55,
                         vol_mult: float = 1.2,
                         require_nextday_fall: bool = True):
    """
    假突破：T日冲击前高，但收盘弱（CR<阈值或没站上前高）；可选：次日跌回枢轴下方。
    """
    d = df.copy()
    h, l, c, o, v = d["high"].values, d["low"].values, d["close"].values, d["open"].values, d["volume"].values
    dates = d["date"].values

    # 过去窗口的前高、成交量中位数
    prev_high = pd.Series(h).shift(1).rolling(lookback_high).max().values
    vol_base = pd.Series(v).shift(1).rolling(lookback_vol).median().values

    out = []
    for i in range(max(lookback_high, lookback_vol)+1, len(d) - (1 if require_nextday_fall else 0)):
        ph = prev_high[i]
        vb = vol_base[i]
        if np.isnan(ph) or np.isnan(vb): 
            continue
        attempt = (h[i] >= ph * (1 + breakout_buf))
        big_vol = (v[i] >= vb * vol_mult)
        cr = closing_range(h[i], l[i], c[i])
        weak_close = (cr < cr_thresh) or (c[i] < ph)  # 没能站上
        if attempt and big_vol and weak_close:
            if require_nextday_fall:
                nxt_close = c[i+1]
                fall_back = nxt_close < ph  # 次日收盘回枢轴下
                if not fall_back:
                    continue
            out.append({
                "date": dates[i], "type": "fake_breakout",
                "prev_high": ph, "high": h[i], "close": c[i],
                "CR": cr, "vol": v[i], "vol_base": vb
            })
    return pd.DataFrame(out)

def detect_stall_on_volume(df: pd.DataFrame,
                           window: int = 3,
                           vol_boost: float = 1.2,
                           upper_shadow_sum_min: float = 2.0,
                           net_gain_max: float = 0.02):
    """
    放量滞涨：近N日量放大，但价格几乎不前进 + 上影偏长。
    """
    d = df.copy()
    out = []
    for i in range(window, len(d)):
        seg = d.iloc[i-window:i]
        v_mean = seg["volume"].mean()
        v_prev5 = d["volume"].iloc[max(0, i-window-5):i-window].mean() if i-window-5 >= 0 else np.nan
        if np.isnan(v_prev5) or v_prev5 <= 0: 
            continue
        vol_ok = v_mean >= v_prev5 * vol_boost

        # 上影之和（相对日内范围）
        upper_sum = ((seg["high"] - seg[["open","close"]].max(axis=1)) / (seg["high"] - seg["low"]).replace(0, np.nan)).clip(lower=0).sum()

        # 价格净进展（从窗口首日收盘到窗口末日收盘）
        net_gain = (seg["close"].iloc[-1] / seg["close"].iloc[0]) - 1.0

        if vol_ok and upper_sum >= upper_shadow_sum_min and net_gain <= net_gain_max:
            last = seg.iloc[-1]
            out.append({
                "date": last["date"], "type": "stall_on_volume",
                "upper_shadow_sum": float(upper_sum),
                "vol_mean_lastN": float(v_mean), "vol_mean_prev5": float(v_prev5),
                "net_gain": float(net_gain)
            })
    return pd.DataFrame(out)

def detect_vol_expand(df: pd.DataFrame,
                      ratio_thresh: float = 1.2,
                      atr_window_short: int = 5,
                      atr_window_long: int = 60):
    """
    波动扩张：近5日 TR% / 近60日 TR%中位数 > 阈值。
    """
    d = df.copy()
    c_prev = d["close"].shift(1).fillna(d["close"])
    tr = pd.Series(true_range(d["high"], d["low"], d["close"], c_prev)) / d["close"].replace(0, np.nan)
    m5 = tr.rolling(atr_window_short).mean()
    m60 = tr.rolling(atr_window_long).median()
    ratio = m5 / m60.replace(0, np.nan)

    mask = ratio >= ratio_thresh
    res = d.loc[mask, ["date"]].copy()
    res["type"] = "vol_expand"
    res["tr5_over_tr60med"] = ratio[mask].astype(float).values
    return res

def detect_divergence(df: pd.DataFrame, lookback: int = 60):
    """
    量价背离：价格创近lookback高，但OBV未创高。
    """
    d = df.copy()
    obv = obv_series(d["close"], d["volume"])
    price_roll_max = d["close"].shift(1).rolling(lookback).max()
    obv_roll_max = obv.shift(1).rolling(lookback).max()

    cond_price_new_high = d["close"] >= price_roll_max
    cond_obv_not_high   = obv < obv_roll_max  # 未创高

    mask = cond_price_new_high & cond_obv_not_high
    res = d.loc[mask, ["date"]].copy()
    res["type"] = "divergence_obv"
    res["close"] = d["close"][mask].values
    res["obv"] = obv[mask].astype(float).values
    res["obv_prev_max"] = obv_roll_max[mask].astype(float).values
    return res

def detect_overextension(df: pd.DataFrame,
                         ma_window: int = 10,
                         mid_thr: float = 0.15,
                         big_thr: float = 0.20):
    """
    过度乖离：|Close/MA10 - 1| > 15%（中度）/ 20%（重度）。
    """
    d = df.copy()
    ma = d["close"].rolling(ma_window).mean()
    dev = d["close"] / ma.replace(0, np.nan) - 1.0
    cond = dev.abs() > mid_thr
    res = d.loc[cond, ["date"]].copy()
    res["type"] = np.where(dev.abs() > big_thr, "overext_big", "overext_mid")
    res["dev_vs_ma10"] = dev[cond].astype(float).values
    return res

def detect_low_turnover_limitup_dump(df: pd.DataFrame,
                                     limit_ret: float = 0.095,
                                     low_rvol_mult: float = 0.8,
                                     vol_window: int = 20,
                                     nextday_drop: float = -0.03):
    """
    低换手涨停/一字，次日砸：
      - 当日涨幅≥9.5% 且收盘==最高（涨停或接近涨停）
      - 当日相对量不高（RVOL < 0.8）作为“低换手”近似
      - 次日收盘跌幅≤ -3%
    """
    d = df.copy()
    d["ret"] = d["close"].pct_change()
    vol_base = d["volume"].shift(1).rolling(vol_window).median()

    out = []
    for i in range(vol_window+1, len(d)-1):
        day = d.iloc[i]
        r = day["ret"]
        if pd.isna(r): 
            continue
        limit_like = (r >= limit_ret) and (day["close"] >= day["high"] * 0.999)  # 收盘≈最高
        low_rvol = (day["volume"] / max(vol_base.iloc[i], 1e-9)) <= low_rvol_mult if vol_base.iloc[i] > 0 else False
        if limit_like and low_rvol:
            next_drop = d["close"].iloc[i+1] / day["close"] - 1.0
            if next_drop <= nextday_drop:
                out.append({
                    "date": day["date"], "type": "low_turnover_limitup_dump",
                    "ret": float(r), "close": float(day["close"]), "high": float(day["high"]),
                    "rvol": float(day["volume"] / max(vol_base.iloc[i], 1e-9)),
                    "next_ret": float(next_drop)
                })
    return pd.DataFrame(out)

# ---------------- 主流程 ---------------- #

def scan_file(path: str, code: str, args) -> pd.DataFrame:
    raw = read_csv_auto(path)
    df = prepare(raw)
    if len(df) < 90:
        return pd.DataFrame(columns=["code","date","type","reason"])

    res = []

    # 1) 假突破
    fb = detect_fake_breakout(df,
                              lookback_high=args.fb_lookback_high,
                              lookback_vol=args.fb_lookback_vol,
                              breakout_buf=args.fb_buf,
                              cr_thresh=args.fb_cr,
                              vol_mult=args.fb_rvol,
                              require_nextday_fall=not args.fb_no_nextday)
    if len(fb):
        fb["code"] = code
        res.append(fb)

    # 2) 放量滞涨
    stall = detect_stall_on_volume(df,
                                   window=args.stall_win,
                                   vol_boost=args.stall_vol_boost,
                                   upper_shadow_sum_min=args.stall_upper_sum,
                                   net_gain_max=args.stall_net_gain)
    if len(stall):
        stall["code"] = code
        res.append(stall)

    # 3) 波动扩张
    vex = detect_vol_expand(df,
                            ratio_thresh=args.vex_ratio,
                            atr_window_short=args.vex_short,
                            atr_window_long=args.vex_long)
    if len(vex):
        vex["code"] = code
        res.append(vex)

    # 4) 量价背离
    div = detect_divergence(df, lookback=args.div_lookback)
    if len(div):
        div["code"] = code
        res.append(div)

    # 5) 过度乖离
    overx = detect_overextension(df,
                                 ma_window=args.over_ma,
                                 mid_thr=args.over_mid,
                                 big_thr=args.over_big)
    if len(overx):
        overx["code"] = code
        res.append(overx)

    # 6) 低换手涨停/一字后次日砸
    ltd = detect_low_turnover_limitup_dump(df,
                                           limit_ret=args.ltd_limit_ret,
                                           low_rvol_mult=args.ltd_rvol,
                                           vol_window=args.ltd_vol_win,
                                           nextday_drop=args.ltd_next_drop)
    if len(ltd):
        ltd["code"] = code
        res.append(ltd)

    if not res:
        return pd.DataFrame(columns=["code","date","type"])
    out = pd.concat(res, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["date","code","type"]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser(description="筛选 6 类负样本")
    ap.add_argument("--input_dir", type=str, default="/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/data", help="包含CSV的目录（和 --input_glob 二选一）")
    ap.add_argument("--input_glob", type=str, default=None, help="CSV通配符（如 './data/*.csv'）")
    ap.add_argument("--out_dir", type=str, default="/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/data", help="输出目录")

    # 假突破参数
    ap.add_argument("--fb_lookback_high", type=int, default=20)
    ap.add_argument("--fb_lookback_vol", type=int, default=60)
    ap.add_argument("--fb_buf", type=float, default=0.002, help="突破判定缓冲（前高 * (1+buf)）")
    ap.add_argument("--fb_cr", type=float, default=0.55, help="收盘强度阈值")
    ap.add_argument("--fb_rvol", type=float, default=1.2, help="相对量阈值（相对60日中位数）")
    ap.add_argument("--fb_no_nextday", action="store_true", help="不要求次日跌回枢轴下方")

    # 放量滞涨
    ap.add_argument("--stall_win", type=int, default=3)
    ap.add_argument("--stall_vol_boost", type=float, default=1.2)
    ap.add_argument("--stall_upper_sum", type=float, default=2.0)
    ap.add_argument("--stall_net_gain", type=float, default=0.02)

    # 波动扩张
    ap.add_argument("--vex_ratio", type=float, default=1.2)
    ap.add_argument("--vex_short", type=int, default=5)
    ap.add_argument("--vex_long", type=int, default=60)

    # 量价背离
    ap.add_argument("--div_lookback", type=int, default=60)

    # 过度乖离
    ap.add_argument("--over_ma", type=int, default=10)
    ap.add_argument("--over_mid", type=float, default=0.15)
    ap.add_argument("--over_big", type=float, default=0.20)

    # 低换手涨停/一字后次日砸
    ap.add_argument("--ltd_limit_ret", type=float, default=0.095, help="涨停近似阈值")
    ap.add_argument("--ltd_rvol", type=float, default=0.8, help="相对量下限（越小越低换手）")
    ap.add_argument("--ltd_vol_win", type=int, default=20)
    ap.add_argument("--ltd_next_drop", type=float, default=-0.03)

    args = ap.parse_args()

    # 收集文件
    if args.input_glob:
        files = glob.glob(args.input_glob)
    elif args.input_dir:
        files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    else:
        raise SystemExit("请用 --input_dir 或 --input_glob 指定数据源")

    if not files:
        raise SystemExit("没有找到任何CSV文件")

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    for p in sorted(files):
        code = os.path.splitext(os.path.basename(p))[0]
        try:
            neg = scan_file(p, code, args)
            if len(neg):
                all_rows.append(neg)
        except Exception as ex:
            print(f"[WARN] 处理失败 {p}: {ex}")

    if not all_rows:
        print("没有筛到负样本。")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # 拆分为 6 类
    def save_mask(name, mask):
        out = combined.loc[mask].copy()
        if len(out):
            out.to_csv(os.path.join(args.out_dir, name), index=False)
            print(f"保存 {name}：{len(out)} 条")
        else:
            print(f"{name}：0 条")

    combined.to_csv(os.path.join(args.out_dir, "negatives_combined.csv"), index=False)
    save_mask("neg_fake_breakout.csv", combined["type"]=="fake_breakout")
    save_mask("neg_stall_on_volume.csv", combined["type"]=="stall_on_volume")
    save_mask("neg_vol_expand.csv", combined["type"]=="vol_expand")
    save_mask("neg_divergence.csv", combined["type"]=="divergence_obv")
    save_mask("neg_overextension.csv", combined["type"].str.startswith("overext"))
    save_mask("neg_low_turnover_limitup_dump.csv", combined["type"]=="low_turnover_limitup_dump")

    # 小计按类型输出
    count = combined.groupby("type").size().sort_values(ascending=False)
    print("\n各类型统计：")
    print(count.to_string())

if __name__ == "__main__":
    main()

