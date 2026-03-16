from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b3filter  # type: ignore


RESULT_DIR = ROOT / "results/b3_weekly_probe_20260315"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = ROOT / "data"


def find_latest_available_data_dir(root_dir: Path, today_str: str) -> tuple[str | None, Path | None]:
    candidates: list[tuple[str, Path]] = []
    for normal_dir in root_dir.glob("20*/normal"):
        if not normal_dir.is_dir():
            continue
        if any(normal_dir.glob("*.txt")):
            candidates.append((normal_dir.parent.name, normal_dir))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0])
    for date_str, normal_dir in reversed(candidates):
        if date_str <= today_str:
            return date_str, normal_dir
    return candidates[-1]


def main() -> None:
    today_str = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")
    scan_date, data_dir = find_latest_available_data_dir(DATA_ROOT, today_str)
    if data_dir is None:
        summary = {"scan_date": None, "hit_count": 0, "error": "未找到任何可用txt数据"}
        (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        return

    rows: list[dict[str, object]] = []
    file_paths = sorted(data_dir.glob("*.txt"))
    total = len(file_paths)
    print(f"扫描目录: {scan_date} 共 {total} 只", flush=True)

    for i, file_path in enumerate(file_paths, 1):
        result = b3filter.check(str(file_path), hold_list=[])
        if result[0] == 1:
            rows.append(
                {
                    "code": file_path.stem,
                    "stock": file_path.stem.split("#")[-1],
                    "stop_price": float(result[1]),
                    "close_price": float(result[2]),
                    "rr_hint": result[3],
                    "reason": result[4],
                }
            )
        if i % 500 == 0 or i == total:
            print({"probe_progress": i, "total": total, "hits": len(rows)}, flush=True)

    hits_df = pd.DataFrame(rows)
    hits_df.to_csv(RESULT_DIR / "hits.csv", index=False, encoding="utf-8-sig")
    summary = {
        "scan_date": scan_date,
        "data_dir": str(data_dir),
        "stock_count": total,
        "hit_count": int(len(hits_df)),
        "first_hit": hits_df.iloc[0].to_dict() if len(hits_df) > 0 else None,
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
