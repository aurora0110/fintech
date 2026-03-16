from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import utils.tmp.run_b1_distribution_exit_ab_20260314 as slow  # type: ignore


RESULT_DIR = ROOT / "results/b1_distribution_exit_ab_fast_20260314"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    stock_data, all_dates_full = slow.b1bt.load_all_data(slow.DATA_DIR)
    for code in list(stock_data.keys()):
        stock_data[code] = slow.add_distribution_labels_upper(stock_data[code])

    all_dates = [d for d in all_dates_full if not (slow.EXCLUDE_START <= d <= slow.EXCLUDE_END)]
    daily_scores = slow.build_signals(stock_data)
    pending_buy = slow.generate_pending_buy_signals_filtered(daily_scores, all_dates_full)

    rows = []
    for max_hold_days in [2, 5, 10, 20, 30]:
        for use_distribution_exit in [False, True]:
            params = {
                **slow.BASE_PARAMS,
                "max_hold_days": max_hold_days,
            }
            name = f"B1_J20Q10_hold{max_hold_days}_{'stop_plus_dist' if use_distribution_exit else 'stop_only'}"
            res = slow.run_backtest_custom(
                stock_data=stock_data,
                all_dates=all_dates,
                pending_buy_signals=pending_buy,
                regime_df=pd.DataFrame(),
                params=params,
                exp_name=name,
                use_distribution_exit=use_distribution_exit,
            )
            rows.append(res)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    summary = {
        "signal_days": int(len(daily_scores)),
        "buy_signal_count": int(sum(len(v) for v in pending_buy.values())),
        "comparison_rows": int(len(result_df)),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
