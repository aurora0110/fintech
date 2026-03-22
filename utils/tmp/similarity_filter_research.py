from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================
# 可选依赖
# =========================
SKLEARN_OK = True
try:
    from sklearn.cluster import KMeans
except Exception:
    SKLEARN_OK = False

LGBM_OK = True
try:
    import lightgbm as lgb
except Exception:
    LGBM_OK = False

PYWT_OK = True
try:
    import pywt
except Exception:
    PYWT_OK = False

SCIPY_OK = True
try:
    from scipy import stats
    from scipy.fft import fft
except Exception:
    SCIPY_OK = False

TQDM_OK = True
try:
    from tqdm import tqdm
except Exception:
    TQDM_OK = False

    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        return iterable


# =========================
# 配置
# =========================
DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data")

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"/Users/lidongyang/Desktop/Qstrategy/results/final_similarity_lab_relaxed_{RUN_TS}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_BARS = 160
HOLD_DAYS = 3
TARGET_RETURN = 0.035
STOP_LOSS = 0.99
EPS = 1e-12
MAX_WORKERS = 8

# 三段切分：research / validation / final_test
RESEARCH_RATIO = 0.60
VALIDATION_RATIO = 0.20
FINAL_TEST_RATIO = 0.20

# 为了降低同一股票密集重复信号
STOCK_SIGNAL_COOLDOWN_DAYS = 20

# 回测参数
INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
BASE_POSITION_PCT = 0.10

# 实验范围
SEQUENCE_LENS = [21, 30]
TEMPLATE_BUILDERS = ["recent_100", "sample_300", "cluster_100"]
REPRESENTATIONS = ["close_norm", "returns", "close_vol_concat"]
SCORERS = [
    "corr",
    "cosine",
    "euclidean",
    "dtw",
    "paa",
    "sax",
    "wavelet",
    "fft",
    "pipeline_corr_dtw",
    "supervised_similarity",
]
SELECT_MODES = [
    ("top_pct", 0.01),
    ("top_pct", 0.05),
    ("top_pct", 0.10),
    ("top_pct", 0.20),
    ("threshold", 0.80),
    ("threshold", 0.85),
    ("threshold", 0.90),
]
DAILY_TOPN_LIST = [5, 7, 8, 9, 10]

SUPERVISED_FEATURES = [
    "succ_max_corr", "succ_mean_corr",
    "succ_max_dtw", "succ_mean_dtw",
    "succ_max_euc", "succ_mean_euc",
    "fail_max_corr", "fail_mean_corr",
    "fail_max_dtw", "fail_mean_dtw",
    "fail_max_euc", "fail_mean_euc",
    "ret1", "ret5", "trend_spread", "close_to_long",
    "brick_red_len", "brick_green_len_prev", "signal_ret", "rebound_ratio",
    "RSI14", "MACD_hist", "KDJ_J", "body_ratio", "upper_shadow_pct", "lower_shadow_pct",
]


# =========================
# 工具函数
# =========================
def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > EPS)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    return series.ewm(alpha=m / n, adjust=False).mean()


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < EPS:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s


def minmax_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mx - mn) or abs(mx - mn) < EPS:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def compute_slope(series: pd.Series, window: int) -> pd.Series:
    def _slope(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) < window or np.any(np.isnan(arr)):
            return np.nan
        x = np.arange(window)
        slope, _ = np.polyfit(x, arr, 1)
        return slope
    return series.rolling(window).apply(_slope, raw=False)


def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


# =========================
# 数据读取
# =========================
def load_stock_data(file_path: str) -> Optional[pd.DataFrame]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) < MIN_BARS + 1:
            return None

        records = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 7:
                try:
                    records.append({
                        "date": parts[0],
                        "open": float(parts[1]),
                        "high": float(parts[2]),
                        "low": float(parts[3]),
                        "close": float(parts[4]),
                        "volume": float(parts[5]),
                    })
                except ValueError:
                    continue

        if not records:
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        df = df.sort_values("date").reset_index(drop=True)

        if len(df) < MIN_BARS:
            return None

        file_name = Path(file_path).stem
        code_match = re.search(r"(\d{6})", file_name)
        code = code_match.group(1) if code_match else file_name
        df["code"] = code
        return df
    except Exception:
        return None


# =========================
# 特征工程：保留 BRICK 结构，但删除量能/涨跌幅/趋势类前置条件
# =========================
def compute_relaxed_brick_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["ret1"] = x["close"].pct_change()
    x["ret5"] = x["close"].pct_change(5)
    x["ret10"] = x["close"].pct_change(10)
    x["ret20"] = x["close"].pct_change(20)
    x["signal_ret"] = safe_div(x["close"] - x["open"], x["open"], default=np.nan)

    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["ma10"] = x["close"].rolling(10).mean()
    x["ma20"] = x["close"].rolling(20).mean()
    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0

    x["ma10_slope_5"] = compute_slope(x["ma10"], 5)
    x["ma20_slope_5"] = compute_slope(x["ma20"], 5)

    x["trend_spread"] = safe_div(x["trend_line"] - x["long_line"], x["close"], default=np.nan)
    x["close_to_trend"] = safe_div(x["close"] - x["trend_line"], x["trend_line"], default=np.nan)
    x["close_to_long"] = safe_div(x["close"] - x["long_line"], x["long_line"], default=np.nan)

    delta = x["close"].diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    avg_up_14 = up.ewm(alpha=1 / 14, adjust=False).mean()
    avg_down_14 = down.ewm(alpha=1 / 14, adjust=False).mean()
    rs = safe_div(avg_up_14, avg_down_14.replace(0, np.nan))
    x["RSI14"] = 100 - 100 / (1 + rs)

    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    x["MACD_DIF"] = ema12 - ema26
    x["MACD_DEA"] = x["MACD_DIF"].ewm(span=9, adjust=False).mean()
    x["MACD_hist"] = 2 * (x["MACD_DIF"] - x["MACD_DEA"])

    llv9 = x["low"].rolling(9).min()
    hhv9 = x["high"].rolling(9).max()
    rsv = safe_div(x["close"] - llv9, (hhv9 - llv9).replace(0, np.nan)) * 100
    x["KDJ_K"] = pd.Series(rsv, index=x.index).ewm(alpha=1/3, adjust=False).mean()
    x["KDJ_D"] = x["KDJ_K"].ewm(alpha=1/3, adjust=False).mean()
    x["KDJ_J"] = 3 * x["KDJ_K"] - 2 * x["KDJ_D"]

    candle_range = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_ratio"] = safe_div((x["close"] - x["open"]).abs(), candle_range, default=np.nan)
    x["upper_shadow_pct"] = safe_div(x["high"] - x[["open", "close"]].max(axis=1), candle_range, default=np.nan)
    x["lower_shadow_pct"] = safe_div(x[["open", "close"]].min(axis=1) - x["low"], candle_range, default=np.nan)
    x["close_location"] = safe_div(x["close"] - x["low"], candle_range, default=np.nan)

    hhv4 = x["high"].rolling(4).max()
    llv4 = x["low"].rolling(4).min()
    den4 = (hhv4 - llv4).replace(0, np.nan)

    var1a = safe_div((hhv4 - x["close"]), den4) * 100 - 90
    var2a = tdx_sma(pd.Series(var1a, index=x.index), 4, 1) + 100
    var3a = safe_div((x["close"] - llv4), den4) * 100
    var4a = tdx_sma(pd.Series(var3a, index=x.index), 6, 1)
    var5a = tdx_sma(var4a, 6, 1) + 100
    var6a = var5a - var2a

    x["brick"] = np.where(var6a > 4, var6a - 4, 0.0)
    x["brick_prev"] = x["brick"].shift(1)
    x["brick_red_len"] = np.where(x["brick"] > x["brick_prev"], x["brick"] - x["brick_prev"], 0.0)
    x["brick_green_len"] = np.where(x["brick"] < x["brick_prev"], x["brick_prev"] - x["brick"], 0.0)
    x["brick_red"] = x["brick_red_len"] > 0
    x["brick_green"] = x["brick_green_len"] > 0
    x["prev_green_streak"] = pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(1)
    x["green_streak"] = pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index)
    x["rebound_ratio"] = safe_div(x["brick_red_len"], x["brick_green_len"].shift(1), default=np.nan)

    x["pattern_a_relaxed"] = (
        (x["prev_green_streak"] >= 3)
        & x["brick_red"]
    )
    x["pattern_b_relaxed"] = (
        (x["green_streak"].shift(3) >= 3)
        & x["brick_red"]
        & x["brick_green"].shift(1).eq(True)
        & x["brick_red"].shift(2).eq(True)
    )

    x["signal_relaxed"] = (
        x["pattern_a_relaxed"].fillna(False)
        | x["pattern_b_relaxed"].fillna(False)
    )

    return x


# =========================
# 标签与样本
# =========================
def label_trade(signal_low: float, next_open: float, future_highs: List[float], future_closes: List[float], future_dates: List[pd.Timestamp]) -> Dict[str, Any]:
    if not np.isfinite(next_open) or next_open <= 0:
        return {"result": "invalid", "label": np.nan, "ret": 0.0, "exit_date": None}

    stop_loss_price = signal_low * STOP_LOSS

    for i in range(len(future_highs)):
        if future_highs[i] >= next_open * (1 + TARGET_RETURN):
            return {"result": "success", "label": 1, "ret": TARGET_RETURN, "exit_date": future_dates[i]}
        if future_closes[i] <= stop_loss_price:
            return {"result": "failure", "label": 0, "ret": (stop_loss_price - next_open) / next_open, "exit_date": future_dates[i]}

    if len(future_closes) > 0:
        return {
            "result": "hold",
            "label": 0,
            "ret": (future_closes[-1] - next_open) / next_open,
            "exit_date": future_dates[-1]
        }

    return {"result": "invalid", "label": np.nan, "ret": 0.0, "exit_date": None}


def extract_sequence(window_df: pd.DataFrame, seq_len: int) -> Dict[str, np.ndarray]:
    w = window_df.tail(seq_len).copy()

    close = w["close"].values.astype(float)
    volume = w["volume"].values.astype(float)

    close_norm = minmax_1d(close)
    vol_norm = minmax_1d(volume)

    returns = safe_div(np.diff(close), close[:-1], default=0.0)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    returns = zscore_1d(returns)

    close_vol_concat = np.concatenate([close_norm, vol_norm], axis=0)

    return {
        "close_norm": close_norm,
        "vol_norm": vol_norm,
        "returns": returns,
        "close_vol_concat": close_vol_concat,
    }


def build_samples_for_one_stock(file_path: str) -> List[Dict[str, Any]]:
    df = load_stock_data(file_path)
    if df is None or df.empty:
        return []

    x = compute_relaxed_brick_features(df)
    code = str(x["code"].iloc[0])
    rows: List[Dict[str, Any]] = []

    max_seq_len = max(SEQUENCE_LENS)

    for idx in range(MIN_BARS, len(x) - HOLD_DAYS):
        row = x.iloc[idx]
        if not bool(row["signal_relaxed"]):
            continue
        if idx < max_seq_len:
            continue

        next_open = float(x.iloc[idx + 1]["open"])
        future_highs = x.iloc[idx + 1: idx + HOLD_DAYS + 1]["high"].tolist()
        future_closes = x.iloc[idx + 1: idx + HOLD_DAYS + 1]["close"].tolist()
        future_dates = x.iloc[idx + 1: idx + HOLD_DAYS + 1]["date"].tolist()

        label_info = label_trade(
            signal_low=float(row["low"]),
            next_open=next_open,
            future_highs=future_highs,
            future_closes=future_closes,
            future_dates=future_dates,
        )
        if label_info["result"] == "invalid":
            continue

        seq_map = {}
        for seq_len in SEQUENCE_LENS:
            seq_map[seq_len] = extract_sequence(x.iloc[idx - seq_len + 1: idx + 1], seq_len)

        rows.append({
            "code": code,
            "date": pd.Timestamp(row["date"]),
            "signal_idx": idx,
            "label": int(label_info["label"]),
            "result": label_info["result"],
            "ret": float(label_info["ret"]),
            "entry_date": pd.Timestamp(x.iloc[idx + 1]["date"]),
            "exit_date": pd.Timestamp(label_info["exit_date"]),
            "entry_price": next_open,

            "ret1": float(row["ret1"]) if pd.notna(row["ret1"]) else 0.0,
            "ret5": float(row["ret5"]) if pd.notna(row["ret5"]) else 0.0,
            "ret10": float(row["ret10"]) if pd.notna(row["ret10"]) else 0.0,
            "signal_ret": float(row["signal_ret"]) if pd.notna(row["signal_ret"]) else 0.0,
            "trend_spread": float(row["trend_spread"]) if pd.notna(row["trend_spread"]) else 0.0,
            "close_to_trend": float(row["close_to_trend"]) if pd.notna(row["close_to_trend"]) else 0.0,
            "close_to_long": float(row["close_to_long"]) if pd.notna(row["close_to_long"]) else 0.0,
            "ma10_slope_5": float(row["ma10_slope_5"]) if pd.notna(row["ma10_slope_5"]) else 0.0,
            "ma20_slope_5": float(row["ma20_slope_5"]) if pd.notna(row["ma20_slope_5"]) else 0.0,
            "brick_red_len": float(row["brick_red_len"]) if pd.notna(row["brick_red_len"]) else 0.0,
            "brick_green_len_prev": float(x["brick_green_len"].shift(1).iloc[idx]) if idx >= 1 and pd.notna(x["brick_green_len"].shift(1).iloc[idx]) else 0.0,
            "rebound_ratio": float(row["rebound_ratio"]) if pd.notna(row["rebound_ratio"]) else 0.0,
            "RSI14": float(row["RSI14"]) if pd.notna(row["RSI14"]) else 0.0,
            "MACD_DIF": float(row["MACD_DIF"]) if pd.notna(row["MACD_DIF"]) else 0.0,
            "MACD_DEA": float(row["MACD_DEA"]) if pd.notna(row["MACD_DEA"]) else 0.0,
            "MACD_hist": float(row["MACD_hist"]) if pd.notna(row["MACD_hist"]) else 0.0,
            "KDJ_K": float(row["KDJ_K"]) if pd.notna(row["KDJ_K"]) else 0.0,
            "KDJ_D": float(row["KDJ_D"]) if pd.notna(row["KDJ_D"]) else 0.0,
            "KDJ_J": float(row["KDJ_J"]) if pd.notna(row["KDJ_J"]) else 0.0,
            "body_ratio": float(row["body_ratio"]) if pd.notna(row["body_ratio"]) else 0.0,
            "upper_shadow_pct": float(row["upper_shadow_pct"]) if pd.notna(row["upper_shadow_pct"]) else 0.0,
            "lower_shadow_pct": float(row["lower_shadow_pct"]) if pd.notna(row["lower_shadow_pct"]) else 0.0,
            "close_location": float(row["close_location"]) if pd.notna(row["close_location"]) else 0.0,

            "pattern_a_relaxed": bool(row["pattern_a_relaxed"]),
            "pattern_b_relaxed": bool(row["pattern_b_relaxed"]),
            "seq_map": seq_map,
        })

    return rows


def apply_stock_signal_cooldown(records: List[Dict[str, Any]], cooldown_days: int = STOCK_SIGNAL_COOLDOWN_DAYS) -> List[Dict[str, Any]]:
    if not records:
        return records

    records_sorted = sorted(records, key=lambda r: (r["code"], r["date"]))
    keep = []
    last_by_code: Dict[str, pd.Timestamp] = {}

    for r in records_sorted:
        code = r["code"]
        dt = pd.Timestamp(r["date"])
        if code not in last_by_code or (dt - last_by_code[code]).days >= cooldown_days:
            keep.append(r)
            last_by_code[code] = dt
    return keep


def load_full_signal_dataset() -> List[Dict[str, Any]]:
    all_dirs = []
    for date_dir in sorted(DATA_DIR.glob("20*")):
        normal_dir = date_dir / "normal"
        if normal_dir.exists():
            all_dirs.append(normal_dir)

    if not all_dirs:
        raise ValueError("未找到数据目录")

    file_paths = []
    for normal_dir in all_dirs:
        file_paths.extend(sorted(normal_dir.glob("*.txt")))

    print(f"加载 {len(file_paths)} 个股票文件...")

    all_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        mapped = executor.map(build_samples_for_one_stock, [str(p) for p in file_paths], chunksize=20)
        for rows in tqdm(mapped, total=len(file_paths), desc="构建样本", ncols=100):
            all_rows.extend(rows)

    all_rows = apply_stock_signal_cooldown(all_rows, cooldown_days=STOCK_SIGNAL_COOLDOWN_DAYS)
    all_rows = sorted(all_rows, key=lambda r: r["date"])

    print(f"共找到 {len(all_rows)} 个 relaxed 信号")
    print(f"成功案例: {sum(r['result'] == 'success' for r in all_rows)}")
    print(f"失败案例: {sum(r['result'] == 'failure' for r in all_rows)}")
    print(f"持有到期: {sum(r['result'] == 'hold' for r in all_rows)}")
    return all_rows


# =========================
# 三段切分
# =========================
def split_three_way(records: List[Dict[str, Any]]):
    unique_dates = sorted(pd.to_datetime(pd.Series([r["date"] for r in records]).unique()))
    n = len(unique_dates)

    research_end = int(n * RESEARCH_RATIO)
    validation_end = int(n * (RESEARCH_RATIO + VALIDATION_RATIO))

    research_end = min(max(1, research_end), n - 2)
    validation_end = min(max(research_end + 1, validation_end), n - 1)

    research_end_date = unique_dates[research_end - 1]
    validation_start_date = unique_dates[research_end]
    validation_end_date = unique_dates[validation_end - 1]
    final_test_start_date = unique_dates[validation_end]

    research = [r for r in records if r["date"] <= research_end_date]
    validation = [r for r in records if validation_start_date <= r["date"] <= validation_end_date]
    final_test = [r for r in records if r["date"] >= final_test_start_date]

    meta = {
        "research_end_date": str(research_end_date.date()),
        "validation_start_date": str(validation_start_date.date()),
        "validation_end_date": str(validation_end_date.date()),
        "final_test_start_date": str(final_test_start_date.date()),
    }
    return research, validation, final_test, meta


# =========================
# 表示/模板
# =========================
def get_rep_vector(record: Dict[str, Any], seq_len: int, rep: str) -> np.ndarray:
    return np.asarray(record["seq_map"][seq_len][rep], dtype=float)


def build_templates(train_success: List[Dict[str, Any]], seq_len: int, rep: str, builder: str) -> List[np.ndarray]:
    if not train_success:
        return []

    if builder == "recent_100":
        arr = sorted(train_success, key=lambda r: r["date"])[-100:]
        return [get_rep_vector(r, seq_len, rep) for r in arr]

    if builder == "sample_300":
        arr = train_success
        n = min(300, len(arr))
        np.random.seed(42)
        idx = np.random.choice(len(arr), size=n, replace=False)
        return [get_rep_vector(arr[i], seq_len, rep) for i in idx]

    if builder == "cluster_100":
        if not SKLEARN_OK:
            arr = train_success[-100:]
            return [get_rep_vector(r, seq_len, rep) for r in arr]

        vecs = np.array([get_rep_vector(r, seq_len, rep) for r in train_success], dtype=float)
        n_clusters = min(100, len(vecs))
        if n_clusters <= 1:
            return [vecs[0]]

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(vecs)
        centers = km.cluster_centers_

        reps: List[np.ndarray] = []
        for cid in range(n_clusters):
            idxs = np.where(labels == cid)[0]
            if len(idxs) == 0:
                continue
            block = vecs[idxs]
            center = centers[cid]
            d = np.linalg.norm(block - center, axis=1)
            reps.append(block[np.argmin(d)])
        return reps

    raise ValueError(f"未知 builder: {builder}")


# =========================
# 距离/相似度
# =========================
def corr_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        return -1.0
    if np.std(a) < EPS or np.std(b) < EPS:
        return -1.0
    if SCIPY_OK:
        c, _ = stats.pearsonr(a, b)
        return float(c) if np.isfinite(c) else -1.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else -1.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < EPS:
        return -1.0
    return float(np.dot(a, b) / denom)


def euclidean_sim(a: np.ndarray, b: np.ndarray) -> float:
    dist = np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return float(1.0 / (1.0 + dist))


def simple_dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def dtw_sim(a: np.ndarray, b: np.ndarray) -> float:
    dist = simple_dtw_distance(a, b)
    norm = max(len(a), len(b))
    return float(1.0 / (1.0 + dist / max(norm, 1)))


def paa_transform(series: np.ndarray, segments: int) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    n = len(series)
    if segments <= 0:
        return series
    out = np.zeros(segments, dtype=float)
    for i in range(segments):
        start = int(i * n / segments)
        end = int((i + 1) * n / segments)
        if end <= start:
            end = min(start + 1, n)
        out[i] = np.mean(series[start:end])
    return out


def paa_sim(a: np.ndarray, b: np.ndarray, segments: int = 7) -> float:
    pa = paa_transform(a, segments)
    pb = paa_transform(b, segments)
    return euclidean_sim(pa, pb)


def sax_transform(series: np.ndarray, segments: int = 7, alphabet: int = 8) -> str:
    paa = paa_transform(zscore_1d(series), segments)
    if SCIPY_OK:
        bps = stats.norm.ppf(np.linspace(0, 1, alphabet + 1)[1:-1])
    else:
        bps = np.linspace(-1.5, 1.5, alphabet - 1)

    chars = []
    for val in paa:
        idx = 0
        for bp in bps:
            if val > bp:
                idx += 1
        chars.append(chr(ord("a") + idx))
    return "".join(chars)


def sax_sim(a: np.ndarray, b: np.ndarray, segments: int = 7, alphabet: int = 8) -> float:
    sa = sax_transform(a, segments=segments, alphabet=alphabet)
    sb = sax_transform(b, segments=segments, alphabet=alphabet)
    if len(sa) != len(sb):
        return 0.0
    dist = 0
    for ca, cb in zip(sa, sb):
        ia = ord(ca) - ord("a")
        ib = ord(cb) - ord("a")
        dist += abs(ia - ib)
    return float(1.0 / (1.0 + dist))


def wavelet_features(a: np.ndarray) -> np.ndarray:
    if not PYWT_OK:
        return np.array([], dtype=float)
    coeffs = pywt.wavedec(np.asarray(a, dtype=float), "db4", level=3)
    feats = []
    for c in coeffs:
        feats.extend([np.mean(c), np.std(c), np.max(c), np.min(c)])
    return np.asarray(feats, dtype=float)


def wavelet_sim(a: np.ndarray, b: np.ndarray) -> float:
    fa = wavelet_features(a)
    fb = wavelet_features(b)
    if len(fa) == 0 or len(fb) == 0 or len(fa) != len(fb):
        return 0.0
    return corr_sim(fa, fb)


def fft_features(a: np.ndarray, num_features: int = 10) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if SCIPY_OK:
        vals = fft(arr)
    else:
        vals = np.fft.fft(arr)
    mag = np.abs(vals[:len(arr)//2])
    if len(mag) == 0:
        return np.array([], dtype=float)
    idx = np.argsort(mag)[-min(num_features, len(mag)):]
    return mag[idx]


def fft_sim(a: np.ndarray, b: np.ndarray) -> float:
    fa = fft_features(a)
    fb = fft_features(b)
    if len(fa) == 0 or len(fb) == 0 or len(fa) != len(fb):
        return 0.0
    return corr_sim(fa, fb)


def topk_indices_by_score(scores: List[float], k: int) -> List[int]:
    if not scores:
        return []
    k = min(k, len(scores))
    arr = np.asarray(scores, dtype=float)
    idx = np.argpartition(-arr, k - 1)[:k]
    idx = idx[np.argsort(-arr[idx])]
    return idx.tolist()


# =========================
# 单方法打分
# =========================
def score_candidate_vs_templates(vec: np.ndarray, templates: List[np.ndarray], scorer: str) -> Dict[str, float]:
    if not templates:
        return {"score": -1.0, "aux1": 0.0, "aux2": 0.0}

    if scorer == "pipeline_corr_dtw":
        corrs = [corr_sim(vec, t) for t in templates]
        top_idx = topk_indices_by_score(corrs, k=min(10, len(corrs)))
        if not top_idx:
            return {"score": -1.0, "aux1": -1.0, "aux2": 0.0}
        top_corr = [corrs[i] for i in top_idx]
        top_dtw = [dtw_sim(vec, templates[i]) for i in top_idx]
        max_corr = max(top_corr) if top_corr else -1.0
        max_dtw = max(top_dtw) if top_dtw else -1.0
        score = 0.5 * max_corr + 0.5 * max_dtw
        return {"score": score, "aux1": max_corr, "aux2": max_dtw}

    sims = []
    for t in templates:
        if scorer == "corr":
            sims.append(corr_sim(vec, t))
        elif scorer == "cosine":
            sims.append(cosine_sim(vec, t))
        elif scorer == "euclidean":
            sims.append(euclidean_sim(vec, t))
        elif scorer == "dtw":
            sims.append(dtw_sim(vec, t))
        elif scorer == "paa":
            sims.append(paa_sim(vec, t))
        elif scorer == "sax":
            sims.append(sax_sim(vec, t))
        elif scorer == "wavelet":
            sims.append(wavelet_sim(vec, t))
        elif scorer == "fft":
            sims.append(fft_sim(vec, t))
        else:
            sims.append(0.0)

    score = max(sims) if sims else -1.0
    return {"score": score, "aux1": float(np.mean(sims)) if sims else 0.0, "aux2": float(np.std(sims)) if sims else 0.0}


# =========================
# supervised similarity
# =========================
def build_supervised_similarity_features(records: List[Dict[str, Any]], success_templates: List[np.ndarray], failure_templates: List[np.ndarray], seq_len: int, rep: str) -> pd.DataFrame:
    rows = []
    for r in records:
        vec = get_rep_vector(r, seq_len, rep)

        succ_corrs = [corr_sim(vec, t) for t in success_templates] if success_templates else [0.0]
        succ_dtws = [dtw_sim(vec, t) for t in success_templates] if success_templates else [0.0]
        succ_eucs = [euclidean_sim(vec, t) for t in success_templates] if success_templates else [0.0]

        fail_corrs = [corr_sim(vec, t) for t in failure_templates] if failure_templates else [0.0]
        fail_dtws = [dtw_sim(vec, t) for t in failure_templates] if failure_templates else [0.0]
        fail_eucs = [euclidean_sim(vec, t) for t in failure_templates] if failure_templates else [0.0]

        rows.append({
            "code": r["code"],
            "date": r["date"],
            "label": r["label"],
            "result": r["result"],
            "ret": r["ret"],
            "entry_date": r["entry_date"],
            "exit_date": r["exit_date"],
            "entry_price": r["entry_price"],

            "succ_max_corr": float(np.max(succ_corrs)),
            "succ_mean_corr": float(np.mean(succ_corrs)),
            "succ_max_dtw": float(np.max(succ_dtws)),
            "succ_mean_dtw": float(np.mean(succ_dtws)),
            "succ_max_euc": float(np.max(succ_eucs)),
            "succ_mean_euc": float(np.mean(succ_eucs)),

            "fail_max_corr": float(np.max(fail_corrs)),
            "fail_mean_corr": float(np.mean(fail_corrs)),
            "fail_max_dtw": float(np.max(fail_dtws)),
            "fail_mean_dtw": float(np.mean(fail_dtws)),
            "fail_max_euc": float(np.max(fail_eucs)),
            "fail_mean_euc": float(np.mean(fail_eucs)),

            "ret1": r["ret1"],
            "ret5": r["ret5"],
            "trend_spread": r["trend_spread"],
            "close_to_long": r["close_to_long"],
            "brick_red_len": r["brick_red_len"],
            "brick_green_len_prev": r["brick_green_len_prev"],
            "signal_ret": r["signal_ret"],
            "rebound_ratio": r["rebound_ratio"],
            "RSI14": r["RSI14"],
            "MACD_hist": r["MACD_hist"],
            "KDJ_J": r["KDJ_J"],
            "body_ratio": r["body_ratio"],
            "upper_shadow_pct": r["upper_shadow_pct"],
            "lower_shadow_pct": r["lower_shadow_pct"],
        })

    df = pd.DataFrame(rows)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def train_supervised_similarity_model(train_df: pd.DataFrame):
    if not LGBM_OK:
        return None
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        verbose=-1,
    )
    model.fit(train_df[SUPERVISED_FEATURES], train_df["label"])
    return model


# =========================
# 选择逻辑
# =========================
@dataclass(frozen=True)
class BaseConfig:
    builder: str
    seq_len: int
    rep: str
    scorer: str

    def key(self) -> str:
        return f"{self.builder}|len{self.seq_len}|{self.rep}|{self.scorer}"


@dataclass(frozen=True)
class StrategyConfig:
    builder: str
    seq_len: int
    rep: str
    scorer: str
    select_mode: str
    select_value: float
    daily_topn: int

    def name(self) -> str:
        return f"{self.builder}|len{self.seq_len}|{self.rep}|{self.scorer}|{self.select_mode}={self.select_value}|top{self.daily_topn}"

    def base_config(self) -> BaseConfig:
        return BaseConfig(self.builder, self.seq_len, self.rep, self.scorer)


def generate_strategy_pool() -> List[StrategyConfig]:
    pool: List[StrategyConfig] = []
    for builder in TEMPLATE_BUILDERS:
        for seq_len in SEQUENCE_LENS:
            for rep in REPRESENTATIONS:
                for scorer in SCORERS:
                    if scorer in {"wavelet", "fft"} and rep == "close_vol_concat":
                        continue
                    if scorer in {"sax", "paa"} and rep == "close_vol_concat":
                        continue

                    for mode, value in SELECT_MODES:
                        for topn in DAILY_TOPN_LIST:
                            pool.append(StrategyConfig(
                                builder=builder,
                                seq_len=seq_len,
                                rep=rep,
                                scorer=scorer,
                                select_mode=mode,
                                select_value=value,
                                daily_topn=topn,
                            ))
    return pool


def generate_base_config_list() -> List[BaseConfig]:
    seen = {}
    for cfg in generate_strategy_pool():
        seen[cfg.base_config().key()] = cfg.base_config()
    return list(seen.values())


def apply_selection_rules(df: pd.DataFrame, cfg: StrategyConfig, score_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    x = df.copy()
    x = x.sort_values(["date", score_col], ascending=[True, False]).reset_index(drop=True)

    if cfg.select_mode == "top_pct":
        x["date_rank"] = x.groupby("date").cumcount() + 1
        x["date_count"] = x.groupby("date")["date"].transform("count")
        x["date_keep_n"] = np.ceil(x["date_count"] * cfg.select_value).astype(int)
        x["date_keep_n"] = x["date_keep_n"].clip(lower=1)
        x = x[x["date_rank"] <= x["date_keep_n"]].copy()

    elif cfg.select_mode == "threshold":
        x = x[x[score_col] >= cfg.select_value].copy()

    else:
        raise ValueError(f"未知 select_mode: {cfg.select_mode}")

    if x.empty:
        return x

    x = x.sort_values(["entry_date", score_col], ascending=[True, False]).reset_index(drop=True)
    x = x.groupby("entry_date", group_keys=False).head(cfg.daily_topn).reset_index(drop=True)
    return x


# =========================
# 回测
# =========================
def prepare_daily_close_map(records_df: pd.DataFrame) -> Dict[Tuple[pd.Timestamp, str], float]:
    close_map: Dict[Tuple[pd.Timestamp, str], float] = {}
    for row in records_df.itertuples(index=False):
        entry_date = pd.Timestamp(row.entry_date)
        exit_date = pd.Timestamp(row.exit_date)
        code = row.code
        entry_price = float(row.entry_price)
        final_price = entry_price * (1 + float(row.ret))

        days = pd.date_range(entry_date, exit_date, freq="D")
        if len(days) <= 1:
            close_map[(entry_date, code)] = final_price
            continue

        last_idx = len(days) - 1
        for i, d in enumerate(days):
            frac = i / last_idx
            px = entry_price + (final_price - entry_price) * frac
            close_map[(d, code)] = px
    return close_map


def run_portfolio_backtest(signal_df: pd.DataFrame, strategy_name: str, score_col: str):
    if signal_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "strategy": strategy_name,
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_ret": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "ending_equity": INITIAL_CAPITAL,
        }

    df = signal_df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    mtm_map = prepare_daily_close_map(df)
    all_dates = sorted(pd.to_datetime(pd.Series(pd.concat([df["entry_date"], df["exit_date"]]).unique())))
    if not all_dates:
        return pd.DataFrame(), pd.DataFrame(), {
            "strategy": strategy_name,
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_ret": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "ending_equity": INITIAL_CAPITAL,
        }

    full_dates = pd.date_range(min(all_dates), max(all_dates), freq="D")
    entry_group = {d: g.copy() for d, g in df.groupby("entry_date")}

    cash = INITIAL_CAPITAL
    open_positions: List[Dict[str, Any]] = []
    closed_positions: List[Dict[str, Any]] = []
    equity_curve = []

    for current_date in full_dates:
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= current_date:
                cash += pos["shares"] * pos["exit_price"]
                closed_positions.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        todays = entry_group.get(current_date, None)
        if todays is not None and len(todays) > 0:
            todays = todays.sort_values(score_col, ascending=False)
            available_slots = max(0, MAX_POSITIONS - len(open_positions))
            if available_slots > 0:
                todays = todays.head(available_slots)
                for row in todays.itertuples(index=False):
                    alloc_cash = min(cash, INITIAL_CAPITAL * BASE_POSITION_PCT)
                    if alloc_cash <= 0 or row.entry_price <= 0:
                        continue
                    shares = alloc_cash / row.entry_price
                    cash -= alloc_cash
                    open_positions.append({
                        "code": row.code,
                        "entry_date": row.entry_date,
                        "exit_date": row.exit_date,
                        "entry_price": float(row.entry_price),
                        "exit_price": float(row.entry_price * (1 + row.ret)),
                        "shares": shares,
                        "entry_value": alloc_cash,
                        "ret": float(row.ret),
                        "score": float(getattr(row, score_col)),
                        "result": row.result,
                    })

        holding_value = 0.0
        for pos in open_positions:
            px = mtm_map.get((current_date, pos["code"]), pos["entry_price"])
            holding_value += pos["shares"] * px

        equity = cash + holding_value
        equity_curve.append({
            "date": current_date,
            "cash": cash,
            "holding_value": holding_value,
            "equity": equity,
            "open_positions": len(open_positions),
        })

    if open_positions:
        for pos in open_positions:
            cash += pos["shares"] * pos["exit_price"]
            closed_positions.append(pos)
        equity_curve.append({
            "date": full_dates[-1] + pd.Timedelta(days=1),
            "cash": cash,
            "holding_value": 0.0,
            "equity": cash,
            "open_positions": 0,
        })

    equity_df = pd.DataFrame(equity_curve).sort_values("date").reset_index(drop=True)
    trades_df = pd.DataFrame(closed_positions)

    if len(equity_df) > 0:
        equity_df["cummax"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = equity_df["equity"] / equity_df["cummax"] - 1.0
        max_dd = equity_df["drawdown"].min()
    else:
        max_dd = 0.0

    if len(trades_df) > 0:
        settled = trades_df[trades_df["result"].isin(["success", "failure"])]
        win_rate = float((settled["result"] == "success").mean()) if len(settled) > 0 else 0.0
        avg_trade_ret = float(trades_df["ret"].mean())
    else:
        win_rate = 0.0
        avg_trade_ret = 0.0

    ending_equity = float(equity_df["equity"].iloc[-1]) if len(equity_df) > 0 else INITIAL_CAPITAL
    total_return = ending_equity / INITIAL_CAPITAL - 1.0

    summary = {
        "strategy": strategy_name,
        "trades": int(len(trades_df)),
        "win_rate": win_rate,
        "avg_trade_ret": avg_trade_ret,
        "total_return": total_return,
        "max_drawdown": float(max_dd),
        "ending_equity": ending_equity,
    }
    return equity_df, trades_df, summary


# =========================
# 基础打分缓存
# =========================
class TemplateCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, int, str, str], List[np.ndarray]] = {}

    def get(self, records: List[Dict[str, Any]], label: str, builder: str, seq_len: int, rep: str) -> List[np.ndarray]:
        key = (label, builder, seq_len, rep)
        if key in self._cache:
            return self._cache[key]

        if label == "success":
            source = [r for r in records if r["label"] == 1]
        elif label == "failure":
            source = [r for r in records if r["label"] == 0]
        else:
            raise ValueError("label must be success or failure")

        templates = build_templates(source, seq_len, rep, builder)
        self._cache[key] = templates
        return templates


def build_scored_df_normal(stage_records: List[Dict[str, Any]], templates: List[np.ndarray], cfg: BaseConfig) -> pd.DataFrame:
    rows = []
    for r in stage_records:
        vec = get_rep_vector(r, cfg.seq_len, cfg.rep)
        scored = score_candidate_vs_templates(vec, templates, cfg.scorer)
        rows.append({
            "code": r["code"],
            "date": r["date"],
            "label": r["label"],
            "result": r["result"],
            "ret": r["ret"],
            "entry_date": r["entry_date"],
            "exit_date": r["exit_date"],
            "entry_price": r["entry_price"],
            "score": scored["score"],
            "aux1": scored["aux1"],
            "aux2": scored["aux2"],

            "ret1": r["ret1"],
            "ret5": r["ret5"],
            "ret10": r["ret10"],
            "signal_ret": r["signal_ret"],
            "trend_spread": r["trend_spread"],
            "close_to_trend": r["close_to_trend"],
            "close_to_long": r["close_to_long"],
            "ma10_slope_5": r["ma10_slope_5"],
            "ma20_slope_5": r["ma20_slope_5"],
            "brick_red_len": r["brick_red_len"],
            "brick_green_len_prev": r["brick_green_len_prev"],
            "rebound_ratio": r["rebound_ratio"],
            "RSI14": r["RSI14"],
            "MACD_hist": r["MACD_hist"],
            "KDJ_J": r["KDJ_J"],
            "body_ratio": r["body_ratio"],
            "upper_shadow_pct": r["upper_shadow_pct"],
            "lower_shadow_pct": r["lower_shadow_pct"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if cfg.scorer == "pipeline_corr_dtw":
        feature_mask = (
            (df["brick_red_len"] > 0)
            & (df["rebound_ratio"] > 0)
            & (df["close_to_long"] > -0.15)
            & (df["ret5"] > -0.20)
        )
        df = df[feature_mask].copy()

    return df.reset_index(drop=True)


def build_scored_df_supervised(
    train_records: List[Dict[str, Any]],
    stage_records: List[Dict[str, Any]],
    success_templates: List[np.ndarray],
    failure_templates: List[np.ndarray],
    cfg: BaseConfig,
) -> pd.DataFrame:
    train_sup_df = build_supervised_similarity_features(
        train_records, success_templates, failure_templates, cfg.seq_len, cfg.rep
    )
    stage_sup_df = build_supervised_similarity_features(
        stage_records, success_templates, failure_templates, cfg.seq_len, cfg.rep
    )

    model = train_supervised_similarity_model(train_sup_df)
    if model is None:
        stage_sup_df["score"] = stage_sup_df["succ_max_corr"]
    else:
        stage_sup_df["score"] = model.predict_proba(stage_sup_df[SUPERVISED_FEATURES])[:, 1]

    return stage_sup_df.reset_index(drop=True)


def build_base_scored_cache(
    train_records: List[Dict[str, Any]],
    stage_records: List[Dict[str, Any]],
    base_configs: List[BaseConfig],
    desc: str,
) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    template_cache = TemplateCache()

    iterator = tqdm(base_configs, total=len(base_configs), desc=desc, ncols=100)
    for base_cfg in iterator:
        key = base_cfg.key()

        if base_cfg.scorer == "supervised_similarity":
            success_templates = template_cache.get(train_records, "success", base_cfg.builder, base_cfg.seq_len, base_cfg.rep)
            failure_templates = template_cache.get(train_records, "failure", base_cfg.builder, base_cfg.seq_len, base_cfg.rep)
            scored_df = build_scored_df_supervised(
                train_records=train_records,
                stage_records=stage_records,
                success_templates=success_templates,
                failure_templates=failure_templates,
                cfg=base_cfg,
            )
        else:
            templates = template_cache.get(train_records, "success", base_cfg.builder, base_cfg.seq_len, base_cfg.rep)
            scored_df = build_scored_df_normal(
                stage_records=stage_records,
                templates=templates,
                cfg=base_cfg,
            )

        cache[key] = scored_df

    return cache


# =========================
# 主流程
# =========================
def main():
    print(f"结果目录: {OUTPUT_DIR}")

    all_records = load_full_signal_dataset()
    research_records, validation_records, final_test_records, meta = split_three_way(all_records)

    print("\n三段切分信息")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"research={len(research_records)}, validation={len(validation_records)}, final_test={len(final_test_records)}")

    # baseline
    baseline_df = pd.DataFrame([{
        "code": r["code"],
        "date": r["date"],
        "label": r["label"],
        "result": r["result"],
        "ret": r["ret"],
        "entry_date": r["entry_date"],
        "exit_date": r["exit_date"],
        "entry_price": r["entry_price"],
        "baseline_score": 0.0,
    } for r in final_test_records])

    _, _, baseline_summary = run_portfolio_backtest(baseline_df, "baseline_relaxed_prefilter", "baseline_score")

    strategy_pool = generate_strategy_pool()
    base_configs = generate_base_config_list()

    print(f"\n完整策略数: {len(strategy_pool)}")
    print(f"基础打分配置数: {len(base_configs)}")

    # 1) research -> validation 基础打分缓存
    validation_base_cache = build_base_scored_cache(
        train_records=research_records,
        stage_records=validation_records,
        base_configs=base_configs,
        desc="构建 validation 基础分数",
    )

    # 2) 在 validation 上遍历 select/topN
    validation_summaries = []
    validation_ranked = []

    for cfg in tqdm(strategy_pool, total=len(strategy_pool), desc="validation 参数评估", ncols=100):
        base_df = validation_base_cache[cfg.base_config().key()]
        selected = apply_selection_rules(base_df, cfg, "score")
        _, _, summary = run_portfolio_backtest(selected, cfg.name(), "score")
        validation_summaries.append(summary)
        validation_ranked.append((cfg, summary))

    validation_df = pd.DataFrame(validation_summaries).sort_values(
        ["total_return", "max_drawdown", "win_rate"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    validation_df.to_csv(OUTPUT_DIR / f"validation_summary_{RUN_TS}.csv", index=False, encoding="utf-8-sig")
    validation_df.head(20).to_csv(OUTPUT_DIR / f"validation_top20_{RUN_TS}.csv", index=False, encoding="utf-8-sig")

    # 3) 选 validation 冠军
    best_strategy_name = validation_df.iloc[0]["strategy"]
    best_cfg = None
    for cfg, summary in validation_ranked:
        if summary["strategy"] == best_strategy_name:
            best_cfg = cfg
            break

    if best_cfg is None:
        raise RuntimeError("未能找到 validation 最优配置")

    with open(OUTPUT_DIR / f"best_config_from_validation_{RUN_TS}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(best_cfg), f, ensure_ascii=False, indent=2)

    # 4) 用 research+validation 重新训练，只算 best_cfg 对应的 final_test 基础打分
    print("\n开始 final_test 最优配置重训练与评估...")
    best_base_cfg = best_cfg.base_config()
    final_base_cache = build_base_scored_cache(
        train_records=research_records + validation_records,
        stage_records=final_test_records,
        base_configs=[best_base_cfg],
        desc="构建 final_test 基础分数",
    )

    final_base_df = final_base_cache[best_base_cfg.key()]
    final_selected = apply_selection_rules(final_base_df, best_cfg, "score")
    final_equity_df, final_trades_df, final_summary = run_portfolio_backtest(final_selected, best_cfg.name(), "score")

    final_summary_rows = [baseline_summary, final_summary]
    final_test_df = pd.DataFrame(final_summary_rows).sort_values(
        ["total_return", "max_drawdown", "win_rate"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    final_test_df.to_csv(OUTPUT_DIR / f"final_test_summary_{RUN_TS}.csv", index=False, encoding="utf-8-sig")
    final_trades_df.to_csv(OUTPUT_DIR / f"final_test_best_trades_{RUN_TS}.csv", index=False, encoding="utf-8-sig")
    final_equity_df.to_csv(OUTPUT_DIR / f"final_test_best_equity_{RUN_TS}.csv", index=False, encoding="utf-8-sig")

    print("\n================ VALIDATION TOP 20 ================")
    print(validation_df.head(20).to_string(index=False))

    print("\n================ FINAL TEST ================")
    print(final_test_df.to_string(index=False))

    print("\n最优参数：")
    print(json.dumps(asdict(best_cfg), ensure_ascii=False, indent=2))

    print(f"\n结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()