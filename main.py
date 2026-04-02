import time
import yaml
from utils import b1filter, b3filter, pinfilter, brick_filter, brickfilter_relaxed_fusion, brickfilter_case_rank_lgbm_top20, holdprint, selectprint, stockDataValidator, stoploss, takeprofit, lgbm_p3_shallower_core10_daily_top9_filter, b1filter_similar_filter
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
        "b1_similar_ml_candidates": [],
        "b3_list": [],
        "pin_list": [],
        "brick_list": [],
        "lgbm_final_list": [],
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
            str(round(float(b1_result[1]), 2)),
            str(round(float(b1_result[2]), 2)),
            str(b1_result[3]),
            str(b1_result[4]),
        ])

    try:
        b1_similar_ml_result = b1filter_similar_filter.check(
            str(file_path),
            hold_list,
            feature_cache=feature_cache,
        )
        if b1_similar_ml_result[0] == 1:
            result["b1_similar_ml_candidates"].append(
                {
                    "code": file_name,
                    "stop_loss_price": float(b1_similar_ml_result[1]),
                    "close_price": float(b1_similar_ml_result[2]),
                    "score": float(b1_similar_ml_result[3]),
                    "note": str(b1_similar_ml_result[4]),
                }
            )
    except Exception:
        pass

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
        ])

    lgbm_result = lgbm_p3_shallower_core10_daily_top9_filter.check(str(file_path), hold_list)
    if lgbm_result[0] == 1:
        result["lgbm_final_list"].append([
            file_name,
            str(round(lgbm_result[1], 2)),
            str(round(lgbm_result[2], 2)),
            str(lgbm_result[3]),
            str(lgbm_result[4]),
        ])

    return result


def _extend_result_lists(target: dict, payload: dict) -> None:
    for key in target.keys():
        target[key].extend(payload.get(key, []))


def _load_holding_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_holding_yaml(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _interactive_hold_management(hold_list: list, hold_path: str) -> list:
    print("\n" + "=" * 80)
    print("【持仓管理】")
    print("  增加持仓: add CODE STOP PROFIT TYPE NOTE | CODE2 STOP2 PROFIT2 TYPE2 NOTE2 ...")
    print("  删除持仓: del CODE1|CODE2|CODE3")
    print("  增加关注: watch CODE NOTE | CODE2 NOTE2")
    print("  删除关注: unwatch CODE1|CODE2")
    print("  直接回车跳过")
    print("=" * 80)

    raw = input("请输入操作（| 分隔多笔，q 退出）：\n").strip()
    if not raw:
        print("跳过持仓管理，继续扫描。")
        return hold_list

    holding = _load_holding_yaml(hold_path)
    hold_stocks = holding.get("hold_stocks", [])
    watch_stocks = holding.get("watch_stocks", [])
    today_date = datetime.today().strftime("%Y-%m-%d")

    for segment in raw.split("|"):
        segment = segment.strip()
        if not segment:
            continue
        parts = segment.split()
        cmd = parts[0].lower() if parts else ""

        try:
            if cmd == "add" and len(parts) >= 5:
                code, stop_p, profit_p, stype = parts[1], parts[2], parts[3], parts[4]
                note = " ".join(parts[5:]) if len(parts) > 5 else ""
                new_entry = {
                    "stock_code": code,
                    "stock_name": "",
                    "type": stype,
                    "stop_loss_price": float(stop_p),
                    "take_profit_price": float(profit_p),
                    "hold_days": 0,
                    "position_ratio": "0%",
                    "buy_date": [today_date],
                    "note": note,
                }
                existing = [i for i, s in enumerate(hold_stocks) if str(s.get("stock_code", "")) == code]
                if existing:
                    hold_stocks[existing[0]] = new_entry
                    print(f"  ✓ 已更新持仓：{code}")
                else:
                    hold_stocks.append(new_entry)
                    print(f"  ✓ 已添加持仓：{code}")

            elif cmd == "del":
                for code in parts[1:]:
                    code = code.strip()
                    before = len(hold_stocks)
                    hold_stocks = [s for s in hold_stocks if str(s.get("stock_code", "")) != code]
                    if len(hold_stocks) < before:
                        print(f"  ✓ 已删除持仓：{code}")
                    else:
                        print(f"  持仓中未找到：{code}")

            elif cmd == "watch" and len(parts) >= 2:
                code = parts[1]
                note = " ".join(parts[2:]) if len(parts) > 2 else ""
                new_watch = {
                    "stock_code": code,
                    "stock_name": "",
                    "type": "关注",
                    "stop_loss_price": 0,
                    "hold_days": 0,
                    "note": note,
                }
                existing = [i for i, s in enumerate(watch_stocks) if str(s.get("stock_code", "")) == code]
                if existing:
                    watch_stocks[existing[0]] = new_watch
                    print(f"  ✓ 已更新关注：{code}")
                else:
                    watch_stocks.append(new_watch)
                    print(f"  ✓ 已添加关注：{code}")

            elif cmd == "unwatch":
                for code in parts[1:]:
                    code = code.strip()
                    before = len(watch_stocks)
                    watch_stocks = [s for s in watch_stocks if str(s.get("stock_code", "")) != code]
                    if len(watch_stocks) < before:
                        print(f"  ✓ 已删除关注：{code}")
                    else:
                        print(f"  关注中未找到：{code}")

            else:
                print(f"  命令不识别，跳过：{segment}")

        except Exception as e:
            print(f"  操作失败：{e}，跳过：{segment}")

    holding["hold_stocks"] = hold_stocks
    holding["watch_stocks"] = watch_stocks
    _save_holding_yaml(hold_path, holding)
    print("  holding.yaml 已保存。")

    new_hold_list = [(s["stock_code"], s.get("stock_name", ""), s.get("type", ""),
                       s.get("stop_loss_price", 0), s.get("take_profit_price", 0),
                       s.get("hold_days", 0), s.get("position_ratio", "0%"),
                       s.get("buy_date", []), s.get("note", ""))
                      for s in hold_stocks]
    return new_hold_list


if __name__ == '__main__':
    start_time = time.time()
    b1_list = []
    b1_similar_ml_list = []
    b3_list = []
    pin_list = []
    brick_list = []
    brick_relaxed_fusion_list = []
    brick_case_rank_lgbm_top20_list = []
    lgbm_final_list = []
    sell_list = []

    today_str = datetime.today().strftime("%Y%m%d")
    print("今天的日期:", today_str)

    hold_path = "/Users/lidongyang/Desktop/Qstrategy/config/holding.yaml"
    hold_list = holdprint.show(hold_path)
    hold_list = _interactive_hold_management(hold_list, hold_path)
    hold_code_set = {str(item[0]) for item in hold_list}

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
    print("\n【B1 相似度冠军策略】")
    print(b1filter_similar_filter.strategy_description())
    print(b1filter_similar_filter.operation_suggestion())
    print("\n【BRICK relaxed_fusion 策略】")
    print(brickfilter_relaxed_fusion.strategy_description())
    print(brickfilter_relaxed_fusion.operation_suggestion())
    print("\n【BRICK case_rank_lgbm_top20 策略】")
    print(brickfilter_case_rank_lgbm_top20.strategy_description())
    print(brickfilter_case_rank_lgbm_top20.operation_suggestion())
        
    result_map = {
        "sell_list": sell_list,
        "b1_list": b1_list,
        "b1_similar_ml_candidates": [],
        "b3_list": b3_list,
        "pin_list": pin_list,
        "brick_list": brick_list,
        "brick_relaxed_fusion_list": brick_relaxed_fusion_list,
        "brick_case_rank_lgbm_top20_list": brick_case_rank_lgbm_top20_list,
        "lgbm_final_list": lgbm_final_list,
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
    b1_similar_ml_candidates = sorted(
        result_map["b1_similar_ml_candidates"],
        key=lambda item: (-float(item["score"]), str(item["code"])),
    )
    for item in b1_similar_ml_candidates[:3]:
        b1_similar_ml_list.append(
            [
                item["code"],
                str(round(float(item["stop_loss_price"]), 2)),
                str(round(float(item["close_price"]), 2)),
                str(round(float(item["score"]), 4)),
                item["note"],
            ]
        )
    b3_list.sort()
    pin_list.sort()
    brick_list.sort()
    print("开始执行 BRICK relaxed_fusion 全市场二次排序...")
    brick_relaxed_fusion_list = brickfilter_relaxed_fusion.scan_dir(
        data_dir_after,
        hold_list=hold_list,
        max_workers=workers,
    )
    print("开始执行 BRICK case_rank_lgbm_top20 全市场排序...")
    brick_case_rank_lgbm_top20_list = brickfilter_case_rank_lgbm_top20.scan_dir(
        data_dir_after,
        hold_list=hold_list,
        max_workers=workers,
    )
    lgbm_final_list.sort()
    print("💡持有列表：", sell_list, '\n', "💡b1列表：", b1_list, '\n', "💡B1_SIM_ML列表：", b1_similar_ml_list, '\n', "💡b3列表：", b3_list, '\n', "💡单针列表：", pin_list, '\n', "💡brick列表：", brick_list, '\n', "💡BRICK_RELAXED_FUSION列表：", brick_relaxed_fusion_list, '\n', "💡BRICK_CASE_RANK_LGBM_TOP20列表：", brick_case_rank_lgbm_top20_list, '\n', "💡LGBM_FINAL列表：", lgbm_final_list, '\n')
    selectprint.show(sell_list, "持有")
    selectprint.show(b1_list, "B1")
    selectprint.show(b1_similar_ml_list, "B1_SIM_ML")
    selectprint.show(b3_list, "B3")
    selectprint.show(pin_list, "单针")
    selectprint.show(brick_list, "BRICK")
    selectprint.show(brick_relaxed_fusion_list, "BRICK_RELAXED_FUSION")
    selectprint.show(brick_case_rank_lgbm_top20_list, "BRICK_CASE_RANK_LGBM_TOP20")
    selectprint.show(lgbm_final_list, "LGBM_FINAL")

    result_dir = "/Users/lidongyang/Desktop/Qstrategy/results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    result_file = os.path.join(result_dir, f"{today_str}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'b1_list': b1_list,
            'b1_similar_ml_list': b1_similar_ml_list,
            'b3_list': b3_list,
            'sell_list': sell_list,
            'pin_list': pin_list,
            'brick_list': brick_list,
            'brick_relaxed_fusion_list': brick_relaxed_fusion_list,
            'brick_case_rank_lgbm_top20_list': brick_case_rank_lgbm_top20_list,
            'lgbm_final_list': lgbm_final_list,
            'hold_list': hold_list
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n筛选结果已保存到：{result_file}")
    print("\n提示：运行 dashboard.py 可以查看这些结果的可视化展示")
    
    end_time = time.time()
    print("程序运行时间：", end_time - start_time, "秒")
