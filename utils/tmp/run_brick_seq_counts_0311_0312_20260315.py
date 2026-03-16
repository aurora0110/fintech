from __future__ import annotations

from multiprocessing import Pool, cpu_count, get_context
from pathlib import Path
import sys

import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brick_filter


RAW_PATH = Path(
    "/Users/lidongyang/Desktop/Qstrategy/results/brick_signals_20260311_0312_no_limits_20260315_xargs/raw.tsv"
)
DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260313/normal")
OUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/brick_seqpattern_parallel_0311_0312_20260315")


def worker(item: tuple[str, list[str], dict, dict]) -> list[dict]:
    code, dts, name_map, score_map = item
    fp = DATA_DIR / f"{code}.txt"
    df = brick_filter.load_one_csv(str(fp))
    if df is None or df.empty:
        return []
    x = brick_filter.add_features(df)
    idx_map = {pd.Timestamp(d).strftime("%Y-%m-%d"): i for i, d in enumerate(pd.to_datetime(x["date"]))}
    out = []
    for dt in dts:
        idx = idx_map.get(dt)
        if idx is None:
            continue
        row = x.iloc[idx]
        if bool(row["sequence_pattern_ok"]):
            out.append(
                {
                    "date": dt,
                    "code": code,
                    "name": name_map.get((code, dt), ""),
                    "sort_score_old": score_map.get((code, dt)),
                    "double_prev_volume": bool(row["double_prev_volume"]),
                    "two_green_one_red": bool(row["pattern_two_green_one_red"]),
                    "g_r_g_r": bool(row["pattern_one_green_one_red_one_green_one_red"]),
                    "two_green_two_red_small_prev_red": bool(row["pattern_two_green_two_red_small_prev_red"]),
                }
            )
    return out


def main() -> None:
    raw = pd.read_csv(
        RAW_PATH,
        sep="\t",
        header=None,
        names=[
            "date",
            "code",
            "name",
            "sort_score",
            "signal_vs_ma5",
            "vol_vs_prev",
            "double_prev_volume",
            "note",
            "close",
            "stop_loss",
        ],
    )
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[
        raw["date"].isin([pd.Timestamp("2026-03-11"), pd.Timestamp("2026-03-12")])
    ].drop_duplicates(["date", "code"])
    raw["date_str"] = raw["date"].dt.strftime("%Y-%m-%d")

    need = raw.groupby("code")["date_str"].apply(list).to_dict()
    name_map = {(r.code, r.date_str): r.name for r in raw.itertuples(index=False)}
    score_map = {(r.code, r.date_str): r.sort_score for r in raw.itertuples(index=False)}

    items = [(code, dts, name_map, score_map) for code, dts in need.items()]
    ctx = get_context("fork")
    with ctx.Pool(processes=min(8, cpu_count())) as pool:
        chunks = pool.map(worker, items)

    rows = [r for chunk in chunks for r in chunk]
    out = pd.DataFrame(rows).sort_values(
        ["date", "sort_score_old", "code"], ascending=[True, False, True]
    ).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "signals.csv", index=False, encoding="utf-8-sig")
    out.groupby("date").size().reset_index(name="count").to_csv(
        OUT_DIR / "daily_counts.csv", index=False, encoding="utf-8-sig"
    )
    print(out.groupby("date").size().reset_index(name="count").to_string(index=False))
    for dt, sub in out.groupby("date"):
        print(f"\n### {dt}")
        print(
            sub[
                [
                    "code",
                    "name",
                    "double_prev_volume",
                    "two_green_one_red",
                    "g_r_g_r",
                    "two_green_two_red_small_prev_red",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
