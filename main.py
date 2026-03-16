import time
from utils import b1filter, b3filter, pinfilter, brick_filter, holdprint, selectprint, stockDataValidator, stoploss, takeprofit
from utils.strategy_feature_cache import StrategyFeatureCache
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import json
from typing import Optional, Tuple
import os


DEFAULT_SCAN_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))
_WORKER_HOLD_LIST = None
_WORKER_HOLD_CODE_SET = None


def find_latest_available_data_dir(root_dir: str, today_str: str) -> Tuple[Optional[str], Optional[str]]:
    root = Path(root_dir)
    candidates = []
    for normal_dir in root.glob("20*/normal"):
        if not normal_dir.is_dir():
            continue
        txt_count = len(list(normal_dir.glob("*.txt")))
        if txt_count == 0:
            continue
        date_str = normal_dir.parent.name
        candidates.append((date_str, str(normal_dir)))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0])
    if today_str:
        for date_str, normal_dir in reversed(candidates):
            if date_str <= today_str:
                return date_str, normal_dir
    return candidates[-1]


def _init_scan_worker(hold_list, hold_code_set):
    global _WORKER_HOLD_LIST, _WORKER_HOLD_CODE_SET
    _WORKER_HOLD_LIST = hold_list
    _WORKER_HOLD_CODE_SET = set(hold_code_set)


def _scan_one_file(file_path_str: str):
    hold_list = _WORKER_HOLD_LIST or []
    hold_code_set = _WORKER_HOLD_CODE_SET or set()
    file_path = Path(file_path_str)
    file_name_no_suffix = file_path.stem
    file_name = file_name_no_suffix.split('#')[-1]
    feature_cache = StrategyFeatureCache(str(file_path))

    result = {
        "sell_list": [],
        "b1_list": [],
        "b3_list": [],
        "pin_list": [],
        "brick_list": [],
    }

    if file_name in hold_code_set:
        stoploss_result = stoploss.check(str(file_path), hold_list)
        takeprofit_result = takeprofit.check(str(file_path), hold_list)
        if stoploss_result[0] == 1:
            result["sell_list"].append([file_name, stoploss_result[1]])
        if takeprofit_result[0] == 1:
            result["sell_list"].append([file_name, takeprofit_result[1]])

    b1_result = b1filter.check(str(file_path), hold_list, feature_cache=feature_cache)
    if b1_result[0] == 1:
        result["b1_list"].append([
            file_name,
            str(b1_result[1].round(2)),
            str(b1_result[2].round(2)),
            str(b1_result[3]),
            str(b1_result[4]),
        ])

    b3_result = b3filter.check(str(file_path), hold_list, feature_cache=feature_cache)
    if b3_result[0] == 1:
        result["b3_list"].append([
            file_name,
            str(round(b3_result[1], 2)),
            str(round(b3_result[2], 2)),
            str(b3_result[3]),
            str(b3_result[4]),
        ])

    pin_result = pinfilter.check(str(file_path), feature_cache=feature_cache)
    if pin_result[0] == 1:
        recommendation = pin_result[2] if len(pin_result) >= 3 else ""
        note = pin_result[3] if len(pin_result) >= 4 else ""
        result["pin_list"].append([file_name, pin_result[1], recommendation, note])

    brick_result = brick_filter.check(str(file_path), hold_list, feature_cache=feature_cache)
    if brick_result[0] == 1:
        result["brick_list"].append([
            file_name,
            str(round(brick_result[1], 2)),
            str(round(brick_result[2], 2)),
            str(brick_result[3]),
            str(brick_result[4]),
            str(brick_result[5]),
            str(brick_result[6]),
            str(brick_result[7]),
            str(brick_result[8]),
        ])

    return result


def _extend_result_lists(target: dict, payload: dict) -> None:
    for key in target.keys():
        target[key].extend(payload.get(key, []))


if __name__ == '__main__':
    start_time = time.time()
    # 买入卖出信号
    b1_list = []
    b3_list = []
    pin_list = []
    brick_list = []
    sell_list = []

    # 获取今天日期
    today_str = datetime.today().strftime("%Y%m%d")
    print("今天的日期:", today_str)  # 输出示例：20260123

    # 打印持仓
    hold_path = "/Users/lidongyang/Desktop/Qstrategy/config/holding.yaml"
    hold_list = holdprint.show(hold_path)
    hold_code_set = {str(item[0]) for item in hold_list}

    # 校验处理数据
    data_dir_before = "/Users/lidongyang/Desktop/Qstrategy/data/" + today_str
    data_dir_after = "/Users/lidongyang/Desktop/Qstrategy/data/" + today_str + "/normal"
    if not Path(data_dir_before).exists():
        Path(data_dir_before).mkdir(parents=True, exist_ok=True)
    if not Path(data_dir_after).exists():
        Path(data_dir_after).mkdir(parents=True, exist_ok=True)

    today_raw_file_paths = list(Path(data_dir_before).glob('*.txt'))
    if today_raw_file_paths:
        stockDataValidator.main(data_dir_before)
        file_paths = list(Path(data_dir_after).glob('*.txt'))
        fallback_date_str = today_str
        if not file_paths:
            print(f"{today_str} 原始目录有数据，但校验后 normal 中没有通过文件")
            raise SystemExit(0)
    else:
        fallback_date_str, fallback_normal_dir = find_latest_available_data_dir(
            "/Users/lidongyang/Desktop/Qstrategy/data",
            today_str,
        )
        if fallback_normal_dir is None:
            print("未找到任何txt文件")
            stockDataValidator.main(data_dir_before)
            raise SystemExit(0)

        print(f"今日目录无原始数据，回退到最近有数据的交易日：{fallback_date_str}")
        data_dir_after = fallback_normal_dir
        data_dir_before = str(Path(fallback_normal_dir).parent)
        stockDataValidator.main(data_dir_before)
        file_paths = list(Path(data_dir_after).glob('*.txt'))

    print(f"实际扫描日期：{fallback_date_str}")
    #print("处理前的数据目录：", data_dir_before, "处理后的数据目录：", data_dir_before)
        
    result_map = {
        "sell_list": sell_list,
        "b1_list": b1_list,
        "b3_list": b3_list,
        "pin_list": pin_list,
        "brick_list": brick_list,
    }
    workers = int(os.environ.get("QSTRATEGY_MAIN_WORKERS", DEFAULT_SCAN_WORKERS))
    print(f"主流程并行进程数：{workers}")
    file_path_strs = [str(p) for p in file_paths]

    with ProcessPoolExecutor(
        max_workers=max(2, workers),
        initializer=_init_scan_worker,
        initargs=(hold_list, tuple(hold_code_set)),
    ) as executor:
        completed = 0
        total = len(file_path_strs)
        for payload in executor.map(_scan_one_file, file_path_strs, chunksize=16):
            completed += 1
            _extend_result_lists(result_map, payload)
            if completed % 200 == 0 or completed == total:
                print(f"主流程扫描进度: {completed}/{total}")

    sell_list.sort()
    b1_list.sort()
    b3_list.sort()
    pin_list.sort()
    brick_list.sort()
    print("💡持有列表：", sell_list, '\n', "💡b1列表：", b1_list, '\n', "💡b3列表：", b3_list, '\n', "💡单针列表：", pin_list, '\n', "💡brick列表：", brick_list, '\n')
    selectprint.show(sell_list, "持有")
    selectprint.show(b1_list, "B1")
    selectprint.show(b3_list, "B3")
    selectprint.show(pin_list, "单针")
    selectprint.show(brick_list, "BRICK")

    # 保存筛选结果到文件，供dashboard.py读取
    # 创建结果目录
    result_dir = "/Users/lidongyang/Desktop/Qstrategy/results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 保存结果到JSON文件
    result_file = os.path.join(result_dir, f"{today_str}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'b1_list': b1_list,
            'b3_list': b3_list,
            'sell_list': sell_list,
            'pin_list': pin_list,
            'brick_list': brick_list,
            'hold_list': hold_list
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n筛选结果已保存到：{result_file}")
    print("\n提示：运行 dashboard.py 可以查看这些结果的可视化展示")
    
    end_time = time.time()
    print("程序运行时间：", end_time - start_time, "秒")
