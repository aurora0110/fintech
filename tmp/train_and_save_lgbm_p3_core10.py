from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import lightgbm as lgb


# ============================================================
# 路径配置
# ============================================================
DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data")
MODEL_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "lgbm_p3_shallower_core10_daily_top9.txt"
FEATURE_IMPORTANCE_FILE = MODEL_DIR / "lgbm_p3_shallower_core10_daily_top9_feature_importance.csv"
TRAIN_DATASET_FILE = MODEL_DIR / "lgbm_p3_shallower_core10_daily_top9_train_dataset.csv"

# ============================================================
# 参数配置
# ============================================================
MIN_BARS = 160
HOLD_DAYS = 3
TARGET_RETURN = 0.035
STOP_LOSS = 0.99
SEQUENCE_LEN = 21
STOCK_SIGNAL_COOLDOWN_DAYS = 20
EPS = 1e-12

# 最终定版主策略参数
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "random_state": 42,
}

CORE10_FEATURE_COLS = [
    "sim_euclidean",
    "sim_rank_today",
    "ret1",
    "ret5",
    "trend_spread",
    "ma20_slope_5",
    "close_to_long",
    "brick_green_len_prev",
    "brick_red_len",
    "signal_ret",
]


# ============================================================
# 工具函数
# ============================================================
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


def compute_slope(series: pd.Series, window: int) -> pd.Series:
    def _slope(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) < window or np.any(np.isnan(arr)):
            return np.nan
        x = np.arange(window)
        slope, _ = np.polyfit(x, arr, 1)
        return slope
    return series.rolling(window).apply(_slope, raw=False)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s


# ============================================================
# 读取单股票
# ============================================================
def load_one_txt(path: str) -> Optional[pd.DataFrame]:
    """
    读取你的 txt 股票数据
    默认格式类似：
    date open high low close volume amount
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
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
        df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

        if len(df) < MIN_BARS:
            return None

        file_name = Path(path).stem
        code_match = re.search(r"(\d{6})", file_name)
        code = code_match.group(1) if code_match else file_name
        df["code"] = code
        return df

    except Exception:
        return None


# ============================================================
# 特征工程
# ============================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)

    x["ret1"] = x["close"].pct_change()
    x["ret5"] = x["close"].pct_change(5)
    x["ret10"] = x["close"].pct_change(10)
    x["signal_ret"] = safe_div(x["close"] - x["open"], x["open"], default=np.nan)

    x["trend_line"] = x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    x["close_pullback_white"] = x["close"] < x["trend_line"] * 1.01
    x["close_above_white"] = x["close"] > x["trend_line"]

    x["ma20"] = x["close"].rolling(20).mean()
    x["ma20_slope_5"] = compute_slope(x["ma20"], 5)

    x["ma14"] = x["close"].rolling(14).mean()
    x["ma28"] = x["close"].rolling(28).mean()
    x["ma57"] = x["close"].rolling(57).mean()
    x["ma114"] = x["close"].rolling(114).mean()
    x["long_line"] = (x["ma14"] + x["ma28"] + x["ma57"] + x["ma114"]) / 4.0

    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = safe_div(x["volume"], x["vol_ma5_prev"])
    x["signal_vs_ma5_valid"] = x["signal_vs_ma5"].between(1, 2.2, inclusive="both")

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

    x["close_slope_10"] = (
        x["close"]
        .rolling(10)
        .apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if np.isfinite(s).all() else np.nan, raw=False)
    )
    x["not_sideways"] = np.abs(safe_div(x["close_slope_10"], x["close"].rolling(10).mean())) > 0.002

    x["up_leg_avg_vol"] = x["volume"].shift(4).rolling(3).mean()
    x["pullback_avg_vol"] = x["volume"].shift(1).rolling(3).mean()
    x["pullback_shrinking"] = x["pullback_avg_vol"] < x["up_leg_avg_vol"]

    x["pattern_a"] = (
        (x["prev_green_streak"] >= 3)
        & x["brick_red"]
        & x["close_pullback_white"].shift(1).fillna(False)
        & x["close_above_white"]
    )
    x["pattern_b"] = (
        (pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(3) >= 3)
        & x["brick_red"]
        & x["brick_green"].shift(1).fillna(False)
        & x["brick_red"].shift(2).fillna(False)
        & x["close_pullback_white"].shift(1).fillna(False)
        & x["close_above_white"]
    )

    x["rebound_ratio"] = safe_div(x["brick_red_len"], x["brick_green_len"].shift(1))
    x["signal_base"] = (
        (x["pattern_a"] | x["pattern_b"])
        & x["pullback_shrinking"].fillna(False)
        & x["signal_vs_ma5_valid"].fillna(False)
        & x["not_sideways"].fillna(False)
        & x["ret1"].notna()
    )

    x["trend_spread"] = safe_div(x["trend_line"] - x["long_line"], x["close"], default=np.nan)
    x["close_to_long"] = safe_div(x["close"] - x["long_line"], x["long_line"], default=np.nan)

    return x


# ============================================================
# 标签
# ============================================================
def label_from_future_path(signal_low: float, next_open: float, future_highs: List[float], future_closes: List[float]) -> Dict:
    if not np.isfinite(next_open) or next_open <= 0:
        return {"result": "invalid", "label": np.nan, "ret": 0.0}

    stop_loss_price = signal_low * STOP_LOSS

    for i in range(len(future_highs)):
        if future_highs[i] >= next_open * (1 + TARGET_RETURN):
            return {"result": "success", "label": 1, "ret": TARGET_RETURN}
        if future_closes[i] <= stop_loss_price:
            return {"result": "failure", "label": 0, "ret": (stop_loss_price - next_open) / next_open}

    if len(future_closes) > 0:
        return {"result": "hold", "label": 0, "ret": (future_closes[-1] - next_open) / next_open}

    return {"result": "invalid", "label": np.nan, "ret": 0.0}


# ============================================================
# 单股票样本构造
# ============================================================
def build_train_samples_for_one_stock(path: str) -> List[Dict]:
    df = load_one_txt(path)
    if df is None or df.empty:
        return []

    x = add_features(df)
    rows = []

    for idx in range(MIN_BARS, len(x) - HOLD_DAYS):
        row = x.iloc[idx]

        mask_a = bool(row["pattern_a"]) and float(row["rebound_ratio"]) >= 0.8 if pd.notna(row["rebound_ratio"]) else False
        mask_b = bool(row["pattern_b"]) and float(row["rebound_ratio"]) >= 1.0 if pd.notna(row["rebound_ratio"]) else False

        signal_ok = (
            bool(row["signal_base"])
            and pd.notna(row["ret1"])
            and float(row["ret1"]) <= 0.08
            and (mask_a or mask_b)
            and float(row["trend_line"]) > float(row["long_line"])
            and (-0.03 <= float(row["ret1"]) <= 0.11)
        )

        if not signal_ok:
            continue

        next_open = float(x.iloc[idx + 1]["open"]) if idx + 1 < len(x) else np.nan
        future_highs = x.iloc[idx + 1: idx + HOLD_DAYS + 1]["high"].tolist()
        future_closes = x.iloc[idx + 1: idx + HOLD_DAYS + 1]["close"].tolist()

        label_info = label_from_future_path(
            signal_low=float(row["low"]),
            next_open=next_open,
            future_highs=future_highs,
            future_closes=future_closes,
        )

        if label_info["result"] == "invalid":
            continue

        rows.append({
            "date": row["date"],
            "code": row["code"],
            "ret1": float(row["ret1"]) if pd.notna(row["ret1"]) else np.nan,
            "ret5": float(row["ret5"]) if pd.notna(row["ret5"]) else np.nan,
            "trend_spread": float(row["trend_spread"]) if pd.notna(row["trend_spread"]) else np.nan,
            "ma20_slope_5": float(row["ma20_slope_5"]) if pd.notna(row["ma20_slope_5"]) else np.nan,
            "close_to_long": float(row["close_to_long"]) if pd.notna(row["close_to_long"]) else np.nan,
            "brick_green_len_prev": float(x["brick_green_len"].shift(1).iloc[idx]) if idx >= 1 and pd.notna(x["brick_green_len"].shift(1).iloc[idx]) else np.nan,
            "brick_red_len": float(row["brick_red_len"]) if pd.notna(row["brick_red_len"]) else np.nan,
            "signal_ret": float(row["signal_ret"]) if pd.notna(row["signal_ret"]) else np.nan,
            "label": label_info["label"],
        })

    return rows


# ============================================================
# 全市场训练集
# ============================================================
def build_train_dataset(data_dir: Path) -> pd.DataFrame:
    all_dirs = []
    for date_dir in sorted(data_dir.glob("20*")):
        normal_dir = date_dir / "normal"
        if normal_dir.exists():
            all_dirs.append(normal_dir)

    if not all_dirs:
        raise ValueError("未找到数据目录")

    # 用全部历史 normal 文件训练
    file_paths = []
    for normal_dir in all_dirs:
        file_paths.extend(sorted(normal_dir.glob("*.txt")))

    print(f"共找到 {len(file_paths)} 个股票文件，开始构造训练样本...")

    all_rows = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for i, rows in enumerate(executor.map(build_train_samples_for_one_stock, [str(p) for p in file_paths], chunksize=20), 1):
            all_rows.extend(rows)
            if i % 500 == 0 or i == len(file_paths):
                print(f"训练样本进度: {i}/{len(file_paths)}")

    if not all_rows:
        raise ValueError("未构造出任何训练样本")

    df = pd.DataFrame(all_rows).sort_values(["date", "code"]).reset_index(drop=True)

    # 这里构造训练阶段的 sim_euclidean proxy 和 sim_rank_today
    z_cols = ["ret1", "ret5", "trend_spread", "ma20_slope_5", "close_to_long", "brick_green_len_prev", "brick_red_len", "signal_ret"]
    for c in z_cols:
        df[f"{c}_z"] = df.groupby("date")[c].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) and np.isfinite(s.std(ddof=0)) and s.std(ddof=0) > 1e-12 else 0.0
        )

    df["sim_euclidean"] = (
        0.20 * df["ret1_z"]
        + 0.20 * df["ret5_z"]
        + 0.18 * df["trend_spread_z"]
        + 0.12 * df["ma20_slope_5_z"]
        + 0.10 * df["close_to_long_z"]
        + 0.08 * df["brick_green_len_prev_z"]
        + 0.07 * df["brick_red_len_z"]
        + 0.05 * df["signal_ret_z"]
    )

    df["sim_euclidean"] = df.groupby("date")["sim_euclidean"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if (s.max() - s.min()) > 1e-12 else 0.5
    )

    df = df.sort_values(["date", "sim_euclidean", "code"], ascending=[True, False, True]).reset_index(drop=True)
    df["sim_rank_today"] = df.groupby("date").cumcount() + 1

    # 清理
    keep_cols = ["date", "code"] + CORE10_FEATURE_COLS + ["label"]
    df = df[keep_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


# ============================================================
# 训练并保存模型
# ============================================================
def train_and_save_model(train_df: pd.DataFrame):
    X = train_df[CORE10_FEATURE_COLS].copy()
    y = train_df["label"].copy()

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X, y)

    # 保存 Booster 模型文件，供 lgb.Booster(model_file=...) 直接加载
    booster = model.booster_
    booster.save_model(str(MODEL_FILE))

    fi_df = pd.DataFrame({
        "feature": CORE10_FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False, encoding="utf-8-sig")

    print(f"模型已保存: {MODEL_FILE}")
    print(f"特征重要性已保存: {FEATURE_IMPORTANCE_FILE}")


# ============================================================
# 主程序
# ============================================================
def main():
    train_df = build_train_dataset(DATA_DIR)
    train_df.to_csv(TRAIN_DATASET_FILE, index=False, encoding="utf-8-sig")
    print(f"训练集已保存: {TRAIN_DATASET_FILE}")
    print(f"训练样本数: {len(train_df)}")
    print(f"正样本数: {(train_df['label'] == 1).sum()}")
    print(f"负样本数: {(train_df['label'] == 0).sum()}")

    train_and_save_model(train_df)


if __name__ == "__main__":
    main()