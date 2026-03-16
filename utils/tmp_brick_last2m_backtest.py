from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


MOD_PATH = "/Users/lidongyang/Desktop/Qstrategy/utils/brick_filter.py"
INPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260312/normal")
HOLD_DAYS = 3
TAKE_PROFIT = 0.03


def load_brick_module():
    spec = importlib.util.spec_from_file_location("brick_filter_mod", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_brick_module()
    signal_df = mod.build_signal_df(INPUT_DIR)
    if signal_df.empty:
        print("未发现任何信号")
        return

    selected = mod.apply_selection(signal_df).copy()
    selected["date"] = pd.to_datetime(selected["date"])
    end_date = selected["date"].max()
    start_date = end_date - pd.DateOffset(months=2)
    selected = (
        selected[selected["date"] >= start_date]
        .sort_values(["date", "sort_score", "code"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    series = {}
    files = sorted([x for x in INPUT_DIR.iterdir() if x.suffix.lower() in {".txt", ".csv"}])
    for p in files:
        df = mod.load_one_csv(str(p))
        if df is None or df.empty:
            continue
        code = str(df["code"].iloc[0])
        series[code] = (
            df[["date", "open", "high", "low", "close"]]
            .sort_values("date")
            .reset_index(drop=True)
        )

    rows = []
    for row in selected.itertuples(index=False):
        code = row.code
        df = series.get(code)
        if df is None:
            continue
        hit = df.index[df["date"] == pd.Timestamp(row.date)]
        if len(hit) == 0:
            continue
        i = int(hit[0])
        entry_i = i + 1
        if entry_i >= len(df):
            continue

        entry_date = pd.Timestamp(df.at[entry_i, "date"])
        entry_price = float(df.at[entry_i, "open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        stop_price = float(row.signal_low) * 0.99
        tp_price = entry_price * (1.0 + TAKE_PROFIT)
        exit_reason = "到期平仓"
        exit_i = None
        max_check_i = min(entry_i + HOLD_DAYS - 1, len(df) - 2)

        for j in range(entry_i + 1, max_check_i + 1):
            day_low = float(df.at[j, "low"])
            day_high = float(df.at[j, "high"])
            tp_hit = np.isfinite(day_high) and day_high >= tp_price
            sl_hit = np.isfinite(day_low) and day_low <= stop_price
            # 同日同时触发时，采用保守口径，优先记止损。
            if tp_hit and sl_hit:
                exit_reason = "止损"
                exit_i = j + 1
                break
            if sl_hit:
                exit_reason = "止损"
                exit_i = j + 1
                break
            if tp_hit:
                exit_reason = "止盈"
                exit_i = j + 1
                break

        if exit_i is None:
            exit_i = entry_i + HOLD_DAYS
            if exit_i >= len(df):
                continue

        exit_date = pd.Timestamp(df.at[exit_i, "date"])
        exit_price = float(df.at[exit_i, "open"])
        if not np.isfinite(exit_price) or exit_price <= 0:
            continue

        trade_return = exit_price / entry_price - 1.0
        rows.append(
            {
                "信号日期": pd.Timestamp(row.date).date().isoformat(),
                "股票": code,
                "买入日期": entry_date.date().isoformat(),
                "卖出日期": exit_date.date().isoformat(),
                "平仓方式": exit_reason,
                "收益率": trade_return,
                "收益率%": round(trade_return * 100, 2),
            }
        )

    trades = pd.DataFrame(rows)
    print(f"区间: {start_date.date()} 到 {end_date.date()}")
    print(f"交易笔数: {len(trades)}")
    if trades.empty:
        return

    for d, g in trades.groupby("信号日期"):
        print(f"\\n{d}")
        print(g[["股票", "买入日期", "卖出日期", "平仓方式", "收益率%"]].to_string(index=False))

    success_rate = float((trades["收益率"] > 0).mean())
    print("\\n汇总:")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均收益率: {trades['收益率'].mean():.2%}")
    print(f"止盈笔数: {(trades['平仓方式'] == '止盈').sum()}")
    print(f"止损笔数: {(trades['平仓方式'] == '止损').sum()}")
    print(f"到期平仓笔数: {(trades['平仓方式'] == '到期平仓').sum()}")


if __name__ == "__main__":
    main()
