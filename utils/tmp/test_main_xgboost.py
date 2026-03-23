from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MODEL_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/ml")
MODEL_FILE = MODEL_DIR / "xgboost_brick_model.json"
THRESHOLD = 0.8

FEATURE_COLS = [
    'signal_ret', 'brick_red_len', 'brick_green_len_prev', 'rebound_ratio',
    'red_len_vs_1d', 'red_len_vs_3d', 'red_len_vs_5d', 'red_len_vs_10d',
    'green_len_prev_vs_1d', 'green_len_prev_vs_3d', 'green_len_prev_vs_5d', 'green_len_prev_vs_10d',
    'trend_spread', 'close_to_trend', 'close_to_long',
    'trend_slope_3', 'trend_slope_5', 'trend_slope_10',
    'ma10_slope_3', 'ma10_slope_5', 'ma10_slope_10',
    'ma20_slope_3', 'ma20_slope_5', 'ma20_slope_10',
    'signal_vs_ma5', 'ret1', 'ret5', 'ret10',
    'RSI14', 'MACD_DIF', 'MACD_DEA', 'MACD_hist',
    'KDJ_K', 'KDJ_D', 'KDJ_J',
    'body_ratio', 'close_location', 'upper_shadow_pct', 'lower_shadow_pct'
]

_model = None


def _load_model():
    global _model
    if not XGBOOST_AVAILABLE:
        return None
    if _model is not None:
        return _model
    try:
        if MODEL_FILE.exists():
            _model = xgb.Booster()
            _model.load_model(str(MODEL_FILE))
            return _model
        else:
            return None
    except Exception:
        return None


def main():
    today_str = datetime.now().strftime('%Y%m%d')
    data_dir = Path(f"/Users/lidongyang/Desktop/Qstrategy/data/{today_str}/normal")
    
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return
    
    file_paths = list(data_dir.glob('*.txt'))
    print(f"总文件数: {len(file_paths)}")
    
    brick_codes = ['002286', '600359', '603209', '688047']
    
    for code in brick_codes:
        matching = [fp for fp in file_paths if code in fp.name]
        if matching:
            fp = matching[0]
            print(f"\n找到文件: {fp.name}")
            
            from utils import xgboost_filter
            result = xgboost_filter.check(str(fp))
            print(f"  结果: {result}")
        else:
            print(f"\n{code}: 未找到文件")


if __name__ == "__main__":
    main()
