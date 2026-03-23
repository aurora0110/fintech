import sys
sys.path.insert(0, "/Users/lidongyang/Desktop/Qstrategy")

import time
from pathlib import Path
from utils import brick_filter, lgbm_p3_shallower_core10_daily_top9_filter

DATES_TO_TEST = ["20260315", "20260313", "20260310"]
DATA_ROOT = Path("/Users/lidongyang/Desktop/Qstrategy/data")


def run_single_date(date_str: str):
    normal_dir = DATA_ROOT / date_str / "normal"
    if not normal_dir.exists():
        print(f"\n{date_str}: 目录不存在")
        return

    file_paths = list(normal_dir.glob("*.txt"))
    if not file_paths:
        print(f"\n{date_str}: 无数据文件")
        return

    print(f"\n{'=' * 80}")
    print(f"日期: {date_str}")
    print(f"文件数: {len(file_paths)}")
    print("=" * 80)

    brick_results = []
    lgbm_results = []

    for fp in file_paths:
        file_name = fp.stem.split('#')[-1]

        try:
            brick_result = brick_filter.check(str(fp), None)
            if brick_result[0] == 1:
                brick_results.append([
                    file_name,
                    round(brick_result[1], 2),
                    round(brick_result[2], 2),
                    brick_result[3],
                ])
        except Exception as e:
            pass

        try:
            lgbm_result = lgbm_p3_shallower_core10_daily_top9_filter.check(str(fp), None)
            if lgbm_result[0] == 1:
                lgbm_results.append([
                    file_name,
                    round(lgbm_result[1], 2),
                    round(lgbm_result[2], 2),
                    round(lgbm_result[3], 6),
                ])
        except Exception as e:
            pass

    brick_results.sort(key=lambda x: x[0])
    lgbm_results.sort(key=lambda x: x[0])

    print(f"\nBRICK筛选: {len(brick_results)}条")
    if brick_results:
        print("-" * 60)
        for r in brick_results:
            print(f"  {r[0]} | 止损: {r[1]} | 收盘: {r[2]} | 得分: {r[3]}")

    print(f"\nLGBM_FINAL筛选: {len(lgbm_results)}条")
    if lgbm_results:
        print("-" * 60)
        for r in lgbm_results:
            print(f"  {r[0]} | 止损: {r[1]} | 收盘: {r[2]} | 得分: {r[3]}")

    brick_codes = set(r[0] for r in brick_results)
    lgbm_codes = set(r[0] for r in lgbm_results)

    print(f"\n对比:")
    print(f"  BRICK独有: {brick_codes - lgbm_codes}")
    print(f"  LGBM独有: {lgbm_codes - brick_codes}")
    print(f"  共同筛选: {brick_codes & lgbm_codes}")
    print(f"  是否完全相同: {brick_codes == lgbm_codes}")


def main():
    print("=" * 80)
    print("多日期对比测试: BRICK vs LGBM_FINAL")
    print("=" * 80)

    for date_str in DATES_TO_TEST:
        run_single_date(date_str)


if __name__ == "__main__":
    main()
