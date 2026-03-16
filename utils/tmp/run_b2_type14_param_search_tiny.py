from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tmp.run_b2_type14_param_search_fast import (  # type: ignore
    RESULT_DIR as FAST_RESULT_DIR,
    build_signals,
    load_all,
    pick_best,
    run_trade_level,
    summarize,
)

RESULT_DIR = ROOT / "results/b2_type14_param_search_tiny_20260313"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    all_data = load_all()
    type1_exit = "hold30_close"
    type4_exit = "hold20_close"

    type1_rows = []
    idx = 0
    total1 = 1 * 2 * 2 * 3 * 2
    for upper_ratio in [0.3, 0.5]:
        for j_max in [90.0, 100.0]:
            for near_ratio in [1.01, 1.02, 1.03]:
                for jrank in [0.10, 0.15]:
                    idx += 1
                    params = {
                        "ret1_min": 0.04,
                        "upper_shadow_body_ratio": upper_ratio,
                        "j_max": j_max,
                        "type1_near_ratio": near_ratio,
                        "type1_j_rank20_max": jrank,
                        "type4_touch_ratio": 1.01,
                    }
                    sig = build_signals(all_data, params, "type1")
                    trades = run_trade_level(all_data, sig, type1_exit)
                    row = summarize(trades)
                    row.update(params)
                    type1_rows.append(row)
                    print(f"type1 参数进度: {idx}/{total1}")

    type4_rows = []
    idx = 0
    total4 = 1 * 2 * 2 * 3
    for upper_ratio in [0.3, 0.5]:
        for j_max in [90.0, 100.0]:
            for touch_ratio in [1.00, 1.01, 1.02]:
                idx += 1
                params = {
                    "ret1_min": 0.04,
                    "upper_shadow_body_ratio": upper_ratio,
                    "j_max": j_max,
                    "type1_near_ratio": 1.02,
                    "type1_j_rank20_max": 0.10,
                    "type4_touch_ratio": touch_ratio,
                }
                sig = build_signals(all_data, params, "type4")
                trades = run_trade_level(all_data, sig, type4_exit)
                row = summarize(trades)
                row.update(params)
                type4_rows.append(row)
                print(f"type4 参数进度: {idx}/{total4}")

    import pandas as pd

    type1_df = pd.DataFrame(type1_rows)
    type4_df = pd.DataFrame(type4_rows)
    type1_df.to_csv(RESULT_DIR / "type1_param_search.csv", index=False)
    type4_df.to_csv(RESULT_DIR / "type4_param_search.csv", index=False)

    best1 = pick_best(type1_df, 800)
    best4 = pick_best(type4_df, 200)
    summary = {
        "based_on_exit": {"type1": type1_exit, "type4": type4_exit},
        "best_type1": best1.to_dict(),
        "best_type4": best4.to_dict(),
        "fast_result_dir_reference": str(FAST_RESULT_DIR),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
