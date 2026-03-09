from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import load_price_directory
from utils.multi_factor_research.factor_calculator import (
    FACTOR_COLUMNS,
    FACTOR_NAME_MAP,
    PENALTY_COLUMNS,
    build_prepared_stock_data,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable: Iterable, **kwargs):
        return iterable


HORIZONS = [5, 10, 20]
ALL_FACTORS = FACTOR_COLUMNS + PENALTY_COLUMNS


def _write_status(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_repair_dataset(prepared_stock_data: dict[str, pd.DataFrame], status_path: Path) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    total = len(prepared_stock_data)
    for idx, (code, df) in enumerate(tqdm(prepared_stock_data.items(), desc="构建修复型单因子样本", unit="股"), start=1):
        if idx == 1 or idx % 200 == 0 or idx == total:
            _write_status(
                status_path,
                {
                    "阶段": "构建修复型单因子样本",
                    "已处理股票数": idx,
                    "股票总数": total,
                    "完成组数": 0,
                },
            )
        signal_mask = (
            df["J"].lt(-5)
            & df["trend_line"].gt(df["bull_bear_line"])
            & df["close"].ge(df["bull_bear_line"])
        )
        signal_mask &= df["bearish_max_volume_60_penalty"].fillna(0.0).lt(0.5)

        signal_idx = np.flatnonzero(signal_mask.to_numpy(dtype=bool))
        signal_idx = signal_idx[signal_idx + 1 < len(df)]
        if signal_idx.size == 0:
            continue

        entry_idx = signal_idx + 1
        entry_price = df["open"].to_numpy(dtype=float)[entry_idx]
        close_arr = df["close"].to_numpy(dtype=float)
        high_arr = df["high"].to_numpy(dtype=float)
        low_arr = df["low"].to_numpy(dtype=float)
        dates = df.index.to_numpy()

        chunk = pd.DataFrame(
            {
                "股票代码": code,
                "信号日期": dates[signal_idx],
                "买入日期": dates[entry_idx],
                "J值": df["J"].to_numpy(dtype=float)[signal_idx],
                "趋势线": df["trend_line"].to_numpy(dtype=float)[signal_idx],
                "多空线": df["bull_bear_line"].to_numpy(dtype=float)[signal_idx],
                "信号收盘价": close_arr[signal_idx],
                "买入价": entry_price,
                "信号K线最低价": low_arr[signal_idx],
            }
        )

        for factor in ALL_FACTORS:
            chunk[factor] = df[factor].to_numpy(dtype=float)[signal_idx]

        for horizon in HORIZONS:
            ret = np.full(signal_idx.size, np.nan, dtype=float)
            max_ret = np.full(signal_idx.size, np.nan, dtype=float)
            min_ret = np.full(signal_idx.size, np.nan, dtype=float)
            valid = entry_idx + horizon - 1 < len(df)
            if valid.any():
                exit_idx = entry_idx[valid] + horizon - 1
                ret[valid] = close_arr[exit_idx] / entry_price[valid] - 1.0
                for src in np.flatnonzero(valid):
                    end_idx = entry_idx[src] + horizon
                    max_ret[src] = high_arr[entry_idx[src]:end_idx].max() / entry_price[src] - 1.0
                    min_ret[src] = low_arr[entry_idx[src]:end_idx].min() / entry_price[src] - 1.0
            chunk[f"{horizon}日收益率"] = ret
            chunk[f"{horizon}日上涨概率"] = (ret > 0).astype(float)
            chunk[f"{horizon}日最高涨幅"] = max_ret
            chunk[f"{horizon}日最低跌幅"] = min_ret

        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _summarize_group(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "样本数": 0,
            "5日上涨概率": 0.0,
            "10日上涨概率": 0.0,
            "20日上涨概率": 0.0,
            "5日平均收益率": 0.0,
            "10日平均收益率": 0.0,
            "20日平均收益率": 0.0,
            "20日平均最高涨幅": 0.0,
            "20日平均最低跌幅": 0.0,
        }
    return {
        "样本数": int(len(df)),
        "5日上涨概率": float(df["5日上涨概率"].mean()),
        "10日上涨概率": float(df["10日上涨概率"].mean()),
        "20日上涨概率": float(df["20日上涨概率"].mean()),
        "5日平均收益率": float(df["5日收益率"].mean()),
        "10日平均收益率": float(df["10日收益率"].mean()),
        "20日平均收益率": float(df["20日收益率"].mean()),
        "20日平均最高涨幅": float(df["20日最高涨幅"].mean()),
        "20日平均最低跌幅": float(df["20日最低跌幅"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="修复型单因子边际提升实验")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/repair_single_factor_experiment_v1")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "状态.json"
    dataset_path = output_root / "修复型单因子样本.csv"
    result_path = output_root / "单因子边际结果.csv"
    summary_path = output_root / "汇总结果.json"

    _write_status(status_path, {"阶段": "加载行情数据", "完成组数": 0})
    stock_data, _ = load_price_directory(args.data_dir)
    _write_status(status_path, {"阶段": "预处理因子", "股票数": len(stock_data), "完成组数": 0})
    prepared = build_prepared_stock_data(stock_data, burst_window=20)

    dataset = _build_repair_dataset(prepared, status_path)
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    baseline = _summarize_group(dataset)
    rows: list[dict] = []
    total = len(ALL_FACTORS)
    for idx, factor in enumerate(tqdm(ALL_FACTORS, desc="分析修复型单因子", unit="因子"), start=1):
        _write_status(
            status_path,
            {
                "阶段": "分析修复型单因子",
                "完成组数": idx - 1,
                "总组数": total,
                "当前因子": FACTOR_NAME_MAP.get(factor, factor),
            },
        )
        active_mask = dataset[factor].fillna(0.0) > 0
        active = _summarize_group(dataset.loc[active_mask])
        inactive = _summarize_group(dataset.loc[~active_mask])

        row = {
            "因子代码": factor,
            "因子名称": FACTOR_NAME_MAP.get(factor, factor),
            "因子来源": "原加分因子" if factor in FACTOR_COLUMNS else "原扣分因子",
            "激活样本数": active["样本数"],
            "非激活样本数": inactive["样本数"],
            "基线10日上涨概率": baseline["10日上涨概率"],
            "激活10日上涨概率": active["10日上涨概率"],
            "非激活10日上涨概率": inactive["10日上涨概率"],
            "10日上涨概率提升": active["10日上涨概率"] - baseline["10日上涨概率"],
            "基线20日上涨概率": baseline["20日上涨概率"],
            "激活20日上涨概率": active["20日上涨概率"],
            "非激活20日上涨概率": inactive["20日上涨概率"],
            "20日上涨概率提升": active["20日上涨概率"] - baseline["20日上涨概率"],
            "基线10日平均收益率": baseline["10日平均收益率"],
            "激活10日平均收益率": active["10日平均收益率"],
            "非激活10日平均收益率": inactive["10日平均收益率"],
            "10日平均收益率提升": active["10日平均收益率"] - baseline["10日平均收益率"],
            "基线20日平均收益率": baseline["20日平均收益率"],
            "激活20日平均收益率": active["20日平均收益率"],
            "非激活20日平均收益率": inactive["20日平均收益率"],
            "20日平均收益率提升": active["20日平均收益率"] - baseline["20日平均收益率"],
            "激活20日平均最高涨幅": active["20日平均最高涨幅"],
            "激活20日平均最低跌幅": active["20日平均最低跌幅"],
        }
        if row["20日平均收益率提升"] > 0 and row["20日上涨概率提升"] > 0:
            row["结论"] = "正向"
        elif row["20日平均收益率提升"] < 0 and row["20日上涨概率提升"] < 0:
            row["结论"] = "负向"
        else:
            row["结论"] = "待观察"
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(
        ["20日平均收益率提升", "20日上涨概率提升"], ascending=[False, False]
    )
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    summary = {
        "基础条件": "J<-5 且 趋势线>多空线 且 收盘价>=多空线",
        "样本数": baseline["样本数"],
        "基线5日上涨概率": baseline["5日上涨概率"],
        "基线10日上涨概率": baseline["10日上涨概率"],
        "基线20日上涨概率": baseline["20日上涨概率"],
        "基线10日平均收益率": baseline["10日平均收益率"],
        "基线20日平均收益率": baseline["20日平均收益率"],
        "20日平均收益率提升前三因子": result_df.head(3)[["因子名称", "20日平均收益率提升"]].to_dict(orient="records"),
        "20日平均收益率拖累前三因子": result_df.tail(3)[["因子名称", "20日平均收益率提升"]].to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_status(status_path, {"阶段": "已完成", "完成组数": total, "总组数": total})


if __name__ == "__main__":
    main()
