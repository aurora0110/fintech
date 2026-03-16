from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_structure_tags import add_all_structure_labels, load_one_csv  # type: ignore


DATA_DIR = ROOT / "data/20260313/normal"
RESULT_DIR = ROOT / "results/counterparty_tag_probe_20260314"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")


def in_scope(date_value: pd.Timestamp) -> bool:
    return not (EXCLUDE_START <= date_value <= EXCLUDE_END)


def _future_metrics(df: pd.DataFrame, idx: int) -> dict:
    event_close = float(df.at[idx, "close"])
    event_long = float(df.at[idx, "long_line"])
    future5 = df.iloc[idx + 1 : idx + 6]
    future10 = df.iloc[idx + 1 : idx + 11]
    future15 = df.iloc[idx + 1 : idx + 16]

    def close_ret(sub: pd.DataFrame) -> float:
        if sub.empty or event_close <= 0:
            return np.nan
        return float(sub.iloc[-1]["close"] / event_close - 1.0)

    def max_ret(sub: pd.DataFrame) -> float:
        if sub.empty or event_close <= 0:
            return np.nan
        return float(sub["high"].max() / event_close - 1.0)

    reclaim2 = bool((future5.iloc[:2]["close"] > future5.iloc[:2]["long_line"]).any()) if len(future5) else False
    reclaim15 = bool((future15["close"] > future15["long_line"]).any()) if len(future15) else False
    above_event_low15 = bool((future15["close"] >= float(df.at[idx, "low"])).all()) if len(future15) else False

    return {
        "ret5_close": close_ret(future5),
        "ret10_close": close_ret(future10),
        "max5": max_ret(future5),
        "max10": max_ret(future10),
        "reclaim_long_within2d": reclaim2,
        "reclaim_long_within15d": reclaim15,
        "all_close_above_event_low_15d": above_event_low15,
        "event_vs_long": float(event_close / event_long - 1.0) if event_long > 0 else np.nan,
    }


def evaluate_manual_cases() -> pd.DataFrame:
    judged_path = ROOT / "results/counterparty_ambiguous_fast_20260314/manual_judged_so_far.csv"
    judged = pd.read_csv(judged_path, dtype=str)
    rows = []
    for code, sub in judged.groupby("code"):
        path = DATA_DIR / f"{code}.txt"
        df = load_one_csv(str(path))
        if df is None:
            continue
        x = add_all_structure_labels(df)
        x = x[x["date"].map(in_scope)].reset_index(drop=True)
        date_to_idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(x["date"])}
        for _, row in sub.iterrows():
            date_str = str(row["date"])[:10]
            idx = date_to_idx.get(date_str)
            if idx is None:
                rows.append({"code": code, "date": date_str, "manual_label": row["label"], "status": "date_missing"})
                continue
            rows.append(
                {
                    "code": code,
                    "date": date_str,
                    "manual_label": row["label"],
                    "pred_counterparty": bool(x.at[idx, "counterparty_confirmed"]),
                    "pred_exempt": bool(x.at[idx, "counterparty_exempt"]),
                    "pred_point_sell": bool(x.at[idx, "point_any"]),
                    "pred_zone_end": bool(x.at[idx, "zone_end"]),
                    "pred_zone_any": bool(x.at[idx, "zone_any"]),
                    "counterparty_quick_reclaim": bool(x.at[idx, "counterparty_quick_reclaim"]),
                    "counterparty_sideways_confirm": bool(x.at[idx, "counterparty_sideways_confirm"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "manual_case_eval.csv", index=False)
    return out


def evaluate_universe() -> pd.DataFrame:
    rows = []
    files = sorted(DATA_DIR.glob("*.txt"))
    for path in files:
        df = load_one_csv(str(path))
        if df is None:
            continue
        x = add_all_structure_labels(df)
        x = x[x["date"].map(in_scope)].reset_index(drop=True)
        if x.empty:
            continue

        breakdown_base = (
            (x["trend_line"] >= x["long_line"])
            & ((x["close"] < x["long_line"]) | (x["close"] < x["trend_line"]) | x["keyk_break_close"] | x["keyk_break_low"])
        )
        categories = {
            "counterparty_confirmed": x["counterparty_confirmed"].fillna(False),
            "counterparty_exempt": x["counterparty_exempt"].fillna(False),
            "point_sell": x["point_any"].fillna(False),
            "zone_end": x["zone_end"].fillna(False),
            "other_breakdown": breakdown_base.fillna(False)
            & ~x["counterparty_confirmed"].fillna(False)
            & ~x["counterparty_exempt"].fillna(False)
            & ~x["point_any"].fillna(False)
            & ~x["zone_end"].fillna(False),
        }

        for label_name, mask in categories.items():
            indices = np.where(mask.to_numpy(dtype=bool))[0]
            for idx in indices:
                metrics = _future_metrics(x, int(idx))
                rows.append(
                    {
                        "code": path.stem,
                        "date": x.at[idx, "date"].strftime("%Y-%m-%d"),
                        "label": label_name,
                        **metrics,
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "universe_event_eval.csv", index=False)
    return out


def summarize_manual(manual_df: pd.DataFrame) -> pd.DataFrame:
    if "status" in manual_df.columns:
        good = manual_df[manual_df["status"].isna() | (manual_df["status"] == "")]
    else:
        good = manual_df.copy()
    rows = []
    for label in ["击穿对手盘", "出货", "没意义"]:
        sub = good[good["manual_label"] == label]
        if sub.empty:
            continue
        rows.append(
            {
                "manual_label": label,
                "count": int(len(sub)),
                "pred_counterparty_rate": float(sub["pred_counterparty"].mean()),
                "pred_exempt_rate": float(sub["pred_exempt"].mean()),
                "pred_point_sell_rate": float(sub["pred_point_sell"].mean()),
                "pred_zone_end_rate": float(sub["pred_zone_end"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "manual_summary.csv", index=False)
    return out


def summarize_universe(universe_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in universe_df.groupby("label"):
        rows.append(
            {
                "label": label,
                "count": int(len(sub)),
                "avg_ret5_close": float(sub["ret5_close"].mean()),
                "avg_ret10_close": float(sub["ret10_close"].mean()),
                "avg_max5": float(sub["max5"].mean()),
                "avg_max10": float(sub["max10"].mean()),
                "reclaim_long_within2d_rate": float(sub["reclaim_long_within2d"].mean()),
                "reclaim_long_within15d_rate": float(sub["reclaim_long_within15d"].mean()),
                "all_close_above_event_low_15d_rate": float(sub["all_close_above_event_low_15d"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    out.to_csv(RESULT_DIR / "universe_summary.csv", index=False)
    return out


def main():
    manual_df = evaluate_manual_cases()
    manual_summary = summarize_manual(manual_df)
    universe_df = evaluate_universe()
    universe_summary = summarize_universe(universe_df)

    summary = {
        "manual_cases": int(len(manual_df)),
        "manual_counterparty_cases": int((manual_df["manual_label"] == "击穿对手盘").sum()),
        "manual_distribution_cases": int((manual_df["manual_label"] == "出货").sum()),
        "manual_meaningless_cases": int((manual_df["manual_label"] == "没意义").sum()),
        "universe_events": int(len(universe_df)),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(manual_summary.to_string(index=False))
    print(universe_summary.to_string(index=False))


if __name__ == "__main__":
    main()
