from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import stoploss


def _read_header_lines(file_path: Path) -> list[str]:
    for encoding in ("gbk", "gb2312", "utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                lines = f.readlines()
            break
        except Exception:
            lines = []
    else:
        lines = []
    return [line.rstrip("\n") for line in lines[:2]]


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: build_daily_snapshot_from_source_v1_20260407.py <source_dir> <target_dir> <target_date>")

    source_dir = Path(sys.argv[1])
    target_dir = Path(sys.argv[2])
    target_date = pd.Timestamp(sys.argv[3])
    target_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    for src in sorted(source_dir.glob("*.txt")):
        df, err = stoploss.load_data(str(src))
        if err or df is None or df.empty:
            skipped += 1
            continue
        sub = df[df["日期"] <= target_date].copy()
        if sub.empty:
            skipped += 1
            continue

        header_lines = _read_header_lines(src)
        out_path = target_dir / src.name
        with open(out_path, "w", encoding="gbk", errors="ignore") as f:
            if header_lines:
                f.write((header_lines[0] if len(header_lines) >= 1 else "") + "\n")
                f.write((header_lines[1] if len(header_lines) >= 2 else "日期\t开盘\t最高\t最低\t收盘\t成交量\t成交额") + "\n")
            else:
                f.write(f"{src.stem.split('#')[-1]} 日线 前复权\n")
                f.write("日期\t开盘\t最高\t最低\t收盘\t成交量\t成交额\n")

            for row in sub.itertuples(index=False):
                f.write(
                    f"{pd.Timestamp(row.日期).strftime('%Y/%m/%d')} "
                    f"{float(row.开盘):.2f} "
                    f"{float(row.最高):.2f} "
                    f"{float(row.最低):.2f} "
                    f"{float(row.收盘):.2f} "
                    f"{float(row.成交量):.0f} "
                    f"{float(row.成交额):.2f}\n"
                )
        kept += 1

    print(f"source_dir={source_dir}")
    print(f"target_dir={target_dir}")
    print(f"target_date={target_date.date()}")
    print(f"kept_files={kept}")
    print(f"skipped_files={skipped}")


if __name__ == "__main__":
    main()
