from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.market_rules import detect_board
from utils.multi_factor_research.data_processor import _read_price_file, _valid_price_frame


def _position_key(row: pd.Series) -> tuple[str, str, float]:
    return (
        str(row["code"]),
        pd.Timestamp(row["entry_date"]).strftime("%Y-%m-%d"),
        round(float(row["entry_price"]), 6),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description="分析分批止盈中第一段止盈后的后续走势")
    parser.add_argument("data_dir")
    parser.add_argument(
        "--trades-csv",
        default="results/weighted_portfolio_backtest_v18_constrained_card_1000w/tiered/trades.csv",
    )
    parser.add_argument(
        "--output-json",
        default="results/tiered_after_step1_analysis/汇总结果.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/tiered_after_step1_analysis/逐仓结果.csv",
    )
    args = parser.parse_args()

    trades = pd.read_csv(args.trades_csv)
    if trades.empty:
        raise ValueError("交易明细为空，无法分析。")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    trades["entry_price"] = pd.to_numeric(trades["entry_price"], errors="coerce")
    trades["exit_price"] = pd.to_numeric(trades["exit_price"], errors="coerce")
    trades["shares"] = pd.to_numeric(trades["shares"], errors="coerce")
    trades = trades.dropna(subset=["entry_date", "exit_date", "entry_price", "exit_price", "shares"]).copy()
    trades["持仓键"] = trades.apply(_position_key, axis=1)

    codes = sorted(set(trades["code"].astype(str)))
    stock_data = {}
    for code in codes:
        path = Path(args.data_dir) / f"{code}.txt"
        if not path.exists():
            continue
        df = _read_price_file(str(path))
        if df is None or not _valid_price_frame(df):
            continue
        df["code"] = code
        df["board"] = detect_board(code)
        stock_data[code] = df

    rows: list[dict] = []
    position_groups = trades.groupby("持仓键", sort=False)
    total_positions = 0
    positions_with_step1 = 0

    for key, group in position_groups:
        total_positions += 1
        group = group.sort_values(["exit_date", "reason"]).reset_index(drop=True)
        code, entry_date_str, entry_price = key
        stock_df = stock_data.get(code)
        if stock_df is None:
            continue
        stock_df = stock_df.copy()
        stock_df["date"] = pd.to_datetime(stock_df["date"])

        first_step = group[group["reason"] == "第一段止盈"]
        if first_step.empty:
            continue
        positions_with_step1 += 1

        first_step_row = first_step.sort_values("exit_date").iloc[0]
        first_step_date = pd.Timestamp(first_step_row["exit_date"])
        first_step_price = float(first_step_row["exit_price"])
        final_exit_date = pd.Timestamp(group["exit_date"].max())

        after_group = group[group["exit_date"] > first_step_date].copy()
        price_window = stock_df[
            (stock_df["date"] > first_step_date) & (stock_df["date"] <= final_exit_date)
        ].copy()

        if price_window.empty:
            max_future_high = first_step_price
            min_future_low = first_step_price
            final_close = first_step_price
        else:
            max_future_high = float(price_window["high"].max())
            min_future_low = float(price_window["low"].min())
            final_close = float(price_window.iloc[-1]["close"])

        rows.append(
            {
                "代码": code,
                "买入日期": entry_date_str,
                "买入价": entry_price,
                "第一段止盈日期": first_step_date.strftime("%Y-%m-%d"),
                "第一段止盈价": first_step_price,
                "最终离场日期": final_exit_date.strftime("%Y-%m-%d"),
                "第一段后最高价": max_future_high,
                "第一段后最低价": min_future_low,
                "第一段后最终收盘价": final_close,
                "第一段后最高涨幅": max_future_high / first_step_price - 1.0,
                "第一段后最大回撤到买入价以下": float(min_future_low <= entry_price),
                "第一段后最终收盘低于买入价": float(final_close <= entry_price),
                "第一段后再涨5%": float(max_future_high >= first_step_price * 1.05),
                "第一段后再涨10%": float(max_future_high >= first_step_price * 1.10),
                "第一段后再涨20%": float(max_future_high >= first_step_price * 1.20),
                "第一段后触发第二段": float((after_group["reason"] == "第二段止盈").any()),
                "第一段后触发第三段": float((after_group["reason"] == "第三段止盈").any()),
                "第一段后触发第四段": float((after_group["reason"] == "第四段止盈").any()),
                "第一段后触发第五段": float((after_group["reason"] == "第五段止盈").any()),
                "第一段后有止损卖出": float((after_group["reason"] == "止损卖出").any()),
                "第一段后有换仓卖出": float((after_group["reason"] == "换仓卖出").any()),
                "第一段后有趋势线破位卖出": float((after_group["reason"] == "趋势线破位卖出").any()),
                "第一段后有多空线压制卖出": float((after_group["reason"] == "多空线压制卖出").any()),
                "第一段后有滴滴止损": float((after_group["reason"] == "滴滴止损").any()),
                "第一段后有持有到期卖出": float((after_group["reason"] == "持有到期卖出").any()),
            }
        )

    detail = pd.DataFrame(rows)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if detail.empty:
        summary = {
            "总持仓数": total_positions,
            "触发第一段止盈的持仓数": 0,
            "说明": "没有找到触发第一段止盈的持仓。",
        }
    else:
        summary = {
            "总持仓数": total_positions,
            "触发第一段止盈的持仓数": int(positions_with_step1),
            "第一段止盈触发率": _safe_ratio(positions_with_step1, total_positions),
            "第一段后再涨5%比例": float(detail["第一段后再涨5%"].mean()),
            "第一段后再涨10%比例": float(detail["第一段后再涨10%"].mean()),
            "第一段后再涨20%比例": float(detail["第一段后再涨20%"].mean()),
            "第一段后回到买入价以下比例": float(detail["第一段后最大回撤到买入价以下"].mean()),
            "第一段后最终收盘低于买入价比例": float(detail["第一段后最终收盘低于买入价"].mean()),
            "第一段后触发第二段比例": float(detail["第一段后触发第二段"].mean()),
            "第一段后触发第三段比例": float(detail["第一段后触发第三段"].mean()),
            "第一段后触发第四段比例": float(detail["第一段后触发第四段"].mean()),
            "第一段后触发第五段比例": float(detail["第一段后触发第五段"].mean()),
            "第一段后出现止损卖出比例": float(detail["第一段后有止损卖出"].mean()),
            "第一段后出现换仓卖出比例": float(detail["第一段后有换仓卖出"].mean()),
            "第一段后出现趋势线破位卖出比例": float(detail["第一段后有趋势线破位卖出"].mean()),
            "第一段后出现多空线压制卖出比例": float(detail["第一段后有多空线压制卖出"].mean()),
            "第一段后出现滴滴止损比例": float(detail["第一段后有滴滴止损"].mean()),
            "第一段后出现持有到期卖出比例": float(detail["第一段后有持有到期卖出"].mean()),
            "第一段后平均最高涨幅": float(detail["第一段后最高涨幅"].mean()),
            "第一段后最高涨幅中位数": float(detail["第一段后最高涨幅"].median()),
        }

    detail.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
