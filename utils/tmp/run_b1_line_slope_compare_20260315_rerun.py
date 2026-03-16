from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tmp import run_b1_line_slope_compare_20260315 as base  # type: ignore


base.RESULT_DIR = ROOT / "results/b1_line_slope_compare_20260315_rerun"
base.RESULT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    base.main()
