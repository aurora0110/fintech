import re
import warnings
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# 基础配置
# ============================================================
DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/similarity_filter_daily_top7_10")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLD_DAYS = 3
TARGET_RETURN = 0.035
STOP_LOSS = 0.99
SEQUENCE_LEN = 21
MIN_BARS = 160

FINAL_HOLDOUT_RATIO = 0.20
STOCK_SIGNAL_COOLDOWN_DAYS = 20

INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
BASE_POSITION_PCT = 0.10

DAILY_TOPN_LIST = [7, 8, 9, 10]

SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# 工具函数
# ============================================================
def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > 1e-12)
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
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s


def normalize_minmax_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def date_str(x):
    return str(pd.Timestamp(x).date())


# ============================================================
# 数据读取
# ============================================================
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
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if len(df) < MIN_BARS:
            return None

        return df
    except Exception:
        return None


# ============================================================
# 信号逻辑
# ============================================================
def compute_brick_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["trend_line"] = df["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    df["ma14"] = df["close"].rolling(14).mean()
    df["ma28"] = df["close"].rolling(28).mean()
    df["ma57"] = df["close"].rolling(57).mean()
    df["ma114"] = df["close"].rolling(114).mean()
    df["long_line"] = (df["ma14"] + df["ma28"] + df["ma57"] + df["ma114"]) / 4.0

    hhv4 = df["high"].rolling(4).max()
    llv4 = df["low"].rolling(4).min()
    den4 = (hhv4 - llv4).replace(0, np.nan)

    var1a = safe_div((hhv4 - df["close"]), den4) * 100 - 90
    var2a = tdx_sma(pd.Series(var1a, index=df.index), 4, 1) + 100
    var3a = safe_div((df["close"] - llv4), den4) * 100
    var4a = tdx_sma(pd.Series(var3a, index=df.index), 6, 1)
    var5a = tdx_sma(var4a, 6, 1) + 100
    var6a = var5a - var2a

    df["brick"] = np.where(var6a > 4, var6a - 4, 0.0)
    df["brick_prev"] = df["brick"].shift(1)
    df["brick_red_len"] = np.where(df["brick"] > df["brick_prev"], df["brick"] - df["brick_prev"], 0.0)
    df["brick_green_len"] = np.where(df["brick"] < df["brick_prev"], df["brick_prev"] - df["brick"], 0.0)
    df["brick_red"] = df["brick_red_len"] > 0
    df["brick_green"] = df["brick_green_len"] > 0
    df["prev_green_streak"] = pd.Series(calc_green_streak(df["brick_green"].to_numpy()), index=df.index).shift(1)

    df["vol_ma5_prev"] = df["volume"].shift(1).rolling(5).mean()
    df["signal_vs_ma5"] = safe_div(df["volume"], df["vol_ma5_prev"])
    df["signal_vs_ma5_valid"] = df["signal_vs_ma5"].between(1, 2.2, inclusive="both")

    df["close_slope_10"] = df["close"].rolling(10).apply(
        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if np.isfinite(s).all() else np.nan,
        raw=False
    )
    df["not_sideways"] = np.abs(safe_div(df["close_slope_10"], df["close"].rolling(10).mean())) > 0.002

    df["up_leg_avg_vol"] = df["volume"].shift(4).rolling(3).mean()
    df["pullback_avg_vol"] = df["volume"].shift(1).rolling(3).mean()
    df["pullback_shrinking"] = df["pullback_avg_vol"] < df["up_leg_avg_vol"]

    df["close_pullback_white"] = df["close"] < df["trend_line"] * 1.01
    df["close_above_white"] = df["close"] > df["trend_line"]

    return df


def detect_pattern_a(df: pd.DataFrame) -> pd.Series:
    return (
        (df["prev_green_streak"] >= 3)
        & df["brick_red"]
        & df["close_pullback_white"].shift(1).fillna(False)
        & df["close_above_white"]
    )


def detect_pattern_b(df: pd.DataFrame) -> pd.Series:
    green_streak = pd.Series(calc_green_streak(df["brick_green"].to_numpy()), index=df.index)
    return (
        (green_streak.shift(3) >= 3)
        & df["brick_red"]
        & df["brick_green"].shift(1).fillna(False)
        & df["brick_red"].shift(2).fillna(False)
        & df["close_pullback_white"].shift(1).fillna(False)
        & df["close_above_white"]
    )


def extract_sequence_variants(df: pd.DataFrame, signal_idx: int) -> Optional[Dict]:
    start_idx = signal_idx - SEQUENCE_LEN + 1
    if start_idx < MIN_BARS or signal_idx >= len(df):
        return None

    window = df.iloc[start_idx: signal_idx + 1]
    if len(window) != SEQUENCE_LEN:
        return None

    close = window["close"].values.astype(float)
    close_norm = normalize_minmax_1d(close)

    returns = np.diff(close) / np.where(np.abs(close[:-1]) < 1e-12, np.nan, close[:-1])
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    returns_z = zscore_1d(returns)

    return {
        "close_norm": close_norm,
        "returns": returns_z,
    }


def apply_stock_signal_cooldown(signals: List[Dict], cooldown_days: int = 20) -> List[Dict]:
    if not signals:
        return []

    signals_sorted = sorted(signals, key=lambda x: (x["code"], x["date"]))
    result = []
    last_kept_date_by_code = {}

    for s in signals_sorted:
        code = s["code"]
        d = pd.Timestamp(s["date"])
        if code not in last_kept_date_by_code or (d - last_kept_date_by_code[code]).days >= cooldown_days:
            result.append(s)
            last_kept_date_by_code[code] = d

    return result


def process_stock_for_patterns(file_path: str) -> List[Dict]:
    df = load_stock_data(file_path)
    if df is None:
        return []

    df = compute_brick_features(df)

    pattern_a = detect_pattern_a(df)
    pattern_b = detect_pattern_b(df)
    signal = (
        (pattern_a | pattern_b)
        & df["pullback_shrinking"].fillna(False)
        & df["signal_vs_ma5_valid"].fillna(False)
        & df["not_sideways"].fillna(False)
    )

    results = []
    for idx in range(MIN_BARS, len(df) - HOLD_DAYS):
        if signal.iloc[idx]:
            seq = extract_sequence_variants(df, idx)
            if seq is None:
                continue

            file_name = Path(file_path).stem
            code_match = re.search(r"(\d{6})", file_name)
            code = code_match.group(1) if code_match else file_name

            results.append({
                "code": code,
                "date": df.iloc[idx]["date"],
                "signal_idx": idx,
                "sequence": seq,
                "signal_low": float(df.iloc[idx]["low"]),
                "next_open": float(df.iloc[idx + 1]["open"]) if idx + 1 < len(df) else np.nan,
                "future_highs": df.iloc[idx + 1: idx + HOLD_DAYS + 1]["high"].tolist(),
                "future_closes": df.iloc[idx + 1: idx + HOLD_DAYS + 1]["close"].tolist(),
                "future_dates": df.iloc[idx + 1: idx + HOLD_DAYS + 1]["date"].tolist(),
            })

    return apply_stock_signal_cooldown(results, cooldown_days=STOCK_SIGNAL_COOLDOWN_DAYS)


def load_all_patterns() -> List[Dict]:
    all_dates = []
    for date_dir in sorted(DATA_DIR.glob("20*")):
        normal_dir = date_dir / "normal"
        if normal_dir.exists():
            all_dates.append(date_dir.name)

    if not all_dates:
        print("未找到数据目录")
        return []

    date_str_ = all_dates[-1]
    normal_dir = DATA_DIR / date_str_ / "normal"
    file_paths = list(normal_dir.glob("*.txt"))
    print(f"加载 {len(file_paths)} 个股票文件...")

    all_patterns = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for patterns in executor.map(process_stock_for_patterns, [str(fp) for fp in file_paths], chunksize=20):
            all_patterns.extend(patterns)

    return apply_stock_signal_cooldown(
        sorted(all_patterns, key=lambda x: x["date"]),
        cooldown_days=STOCK_SIGNAL_COOLDOWN_DAYS
    )


# ============================================================
# 切分
# ============================================================
def split_research_and_holdout(all_patterns: List[Dict], holdout_ratio: float = 0.2):
    unique_dates = sorted(pd.to_datetime(pd.Series([p["date"] for p in all_patterns]).unique()))
    holdout_start_idx = int(len(unique_dates) * (1 - holdout_ratio))
    holdout_start_idx = min(max(1, holdout_start_idx), len(unique_dates) - 1)
    holdout_start_date = unique_dates[holdout_start_idx]

    research_patterns = [p for p in all_patterns if p["date"] < holdout_start_date]
    holdout_patterns = [p for p in all_patterns if p["date"] >= holdout_start_date]
    return research_patterns, holdout_patterns, holdout_start_date


# ============================================================
# 标签
# ============================================================
def label_from_future_path(signal: Dict) -> Dict:
    next_open = signal["next_open"]
    if not np.isfinite(next_open) or next_open <= 0:
        return {"result": "invalid", "ret": 0.0, "exit_date": None}

    stop_loss_price = signal["signal_low"] * STOP_LOSS
    future_highs = signal["future_highs"]
    future_closes = signal["future_closes"]
    future_dates = signal["future_dates"]

    for i in range(len(future_highs)):
        if future_highs[i] >= next_open * (1 + TARGET_RETURN):
            return {"result": "success", "ret": TARGET_RETURN, "exit_date": future_dates[i]}
        if future_closes[i] <= stop_loss_price:
            return {"result": "failure", "ret": (stop_loss_price - next_open) / next_open, "exit_date": future_dates[i]}

    if len(future_closes) > 0:
        return {
            "result": "hold",
            "ret": (future_closes[-1] - next_open) / next_open,
            "exit_date": future_dates[-1]
        }

    return {"result": "invalid", "ret": 0.0, "exit_date": None}


# ============================================================
# 模板与相似度
# ============================================================
def build_templates(train_success: List[Dict]) -> List[Dict]:
    if not SKLEARN_AVAILABLE:
        raise ImportError("需要 sklearn.cluster.KMeans")

    n_clusters = min(100, len(train_success))
    X = np.array([x["sequence"]["close_norm"] for x in train_success])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    reps = []
    for cid in range(n_clusters):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        block = X[idxs]
        center = centers[cid]
        d = np.linalg.norm(block - center, axis=1)
        best_idx = idxs[np.argmin(d)]
        reps.append(train_success[best_idx])
    return reps


def prepare_matrix(items: List[Dict], variant: str) -> np.ndarray:
    return np.array([np.asarray(x["sequence"][variant], dtype=float) for x in items], dtype=float)


def max_euclidean_scores(candidate_mat: np.ndarray, template_mat: np.ndarray, batch_size: int = 256) -> np.ndarray:
    T = template_mat.astype(float)
    T_sq = np.sum(T * T, axis=1)
    scores = np.full(candidate_mat.shape[0], -1.0, dtype=float)

    for start in range(0, candidate_mat.shape[0], batch_size):
        end = min(start + batch_size, candidate_mat.shape[0])
        C = candidate_mat[start:end].astype(float)
        C_sq = np.sum(C * C, axis=1, keepdims=True)
        dist2 = np.maximum(C_sq + T_sq.reshape(1, -1) - 2 * (C @ T.T), 0.0)
        dist = np.sqrt(dist2)
        sim = 1.0 / (1.0 + dist)
        scores[start:end] = np.max(sim, axis=1)

    return scores


def score_candidates(candidates: List[Dict], templates: List[Dict]) -> List[Dict]:
    cand_mat = prepare_matrix(candidates, "returns")
    tpl_mat = prepare_matrix(templates, "returns")
    scores = max_euclidean_scores(cand_mat, tpl_mat)

    result = []
    for x, s in zip(candidates, scores):
        label = label_from_future_path(x)
        if label["result"] == "invalid":
            continue
        result.append({
            "code": x["code"],
            "signal_date": x["date"],
            "entry_date": x["future_dates"][0] if len(x["future_dates"]) > 0 else None,
            "entry_price": x["next_open"],
            "exit_date": label["exit_date"],
            "ret": label["ret"],
            "score": float(s),
            "result": label["result"],
        })
    return result


# ============================================================
# 回测
# ============================================================
def run_portfolio_backtest(trades: List[Dict], strategy_name: str, daily_topn: Optional[int] = None):
    if not trades:
        return pd.DataFrame(), pd.DataFrame(), {
            "strategy": strategy_name,
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_ret": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "ending_equity": INITIAL_CAPITAL,
        }

    df = pd.DataFrame(trades).copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values(["entry_date", "score"], ascending=[True, False]).reset_index(drop=True)

    if daily_topn is not None:
        tmp = df.sort_values(["entry_date", "score"], ascending=[True, False])
        keep_idx = tmp.groupby("entry_date").head(daily_topn).index
        df = df.loc[keep_idx].sort_values(["entry_date", "score"], ascending=[True, False]).reset_index(drop=True)

    all_dates = sorted(pd.to_datetime(pd.Series(pd.concat([df["entry_date"], df["exit_date"]]).unique())))

    cash = INITIAL_CAPITAL
    open_positions = []
    closed_positions = []
    equity_curve = []

    for current_date in all_dates:
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= current_date:
                cash += pos["exit_value"]
                closed_positions.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        todays = df[df["entry_date"] == current_date].copy()
        if len(todays) > 0:
            todays = todays.sort_values("score", ascending=False)
            available_slots = max(0, MAX_POSITIONS - len(open_positions))

            if available_slots > 0:
                todays = todays.head(available_slots)

                for _, row in todays.iterrows():
                    position_value = min(cash, INITIAL_CAPITAL * BASE_POSITION_PCT)
                    if position_value <= 0:
                        break

                    exit_value = position_value * (1 + row["ret"])
                    cash -= position_value

                    open_positions.append({
                        "code": row["code"],
                        "entry_date": row["entry_date"],
                        "exit_date": row["exit_date"],
                        "entry_value": position_value,
                        "exit_value": exit_value,
                        "ret": row["ret"],
                        "score": row["score"],
                        "result": row["result"],
                    })

        open_cost = sum(p["entry_value"] for p in open_positions)
        equity = cash + open_cost

        equity_curve.append({
            "date": current_date,
            "cash": cash,
            "open_cost": open_cost,
            "equity": equity,
            "open_positions": len(open_positions),
        })

    if open_positions:
        for pos in open_positions:
            cash += pos["exit_value"]
            closed_positions.append(pos)
        equity_curve.append({
            "date": all_dates[-1] + pd.Timedelta(days=1),
            "cash": cash,
            "open_cost": 0.0,
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
        win_rate = (settled["result"] == "success").mean() if len(settled) > 0 else 0.0
        avg_trade_ret = trades_df["ret"].mean()
    else:
        win_rate = 0.0
        avg_trade_ret = 0.0

    ending_equity = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else INITIAL_CAPITAL
    total_return = ending_equity / INITIAL_CAPITAL - 1.0

    summary = {
        "strategy": strategy_name,
        "trades": len(trades_df),
        "win_rate": win_rate,
        "avg_trade_ret": avg_trade_ret,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "ending_equity": ending_equity,
    }

    return equity_df, trades_df, summary


# ============================================================
# 绘图
# ============================================================
def plot_equity_curves(equity_map: Dict[str, pd.DataFrame], output_png: Path):
    plt.figure(figsize=(14, 8))
    for name, df in equity_map.items():
        if len(df) == 0:
            continue
        plt.plot(df["date"], df["equity"], label=name)
    plt.title("Euclidean Daily Top7-10 Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def plot_drawdown_curves(equity_map: Dict[str, pd.DataFrame], output_png: Path):
    plt.figure(figsize=(14, 8))
    for name, df in equity_map.items():
        if len(df) == 0 or "drawdown" not in df.columns:
            continue
        plt.plot(df["date"], df["drawdown"], label=name)
    plt.title("Euclidean Daily Top7-10 Drawdown Curves")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 100)
    print("Euclidean Daily Top7~10 精细回测")
    print("=" * 100)

    all_patterns = load_all_patterns()
    if not all_patterns:
        return

    research_patterns, holdout_patterns, holdout_start_date = split_research_and_holdout(
        all_patterns, holdout_ratio=FINAL_HOLDOUT_RATIO
    )

    print(f"研究集信号数: {len(research_patterns)}")
    print(f"Holdout 信号数: {len(holdout_patterns)}")
    print(f"Holdout 开始日期: {date_str(holdout_start_date)}")

    train_success = [p for p in research_patterns if label_from_future_path(p)["result"] == "success"]
    np.random.seed(42)
    templates = build_templates(train_success)

    print("生成 euclidean 打分信号...")
    euclidean_trades = score_candidates(holdout_patterns, templates)

    baseline_trades = []
    for x in holdout_patterns:
        label = label_from_future_path(x)
        if label["result"] == "invalid":
            continue
        baseline_trades.append({
            "code": x["code"],
            "signal_date": x["date"],
            "entry_date": x["future_dates"][0] if len(x["future_dates"]) > 0 else None,
            "entry_price": x["next_open"],
            "exit_date": label["exit_date"],
            "ret": label["ret"],
            "score": 0.0,
            "result": label["result"],
        })

    equity_map = {}
    summary_rows = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    equity_df, trades_df, summary = run_portfolio_backtest(baseline_trades, "baseline", daily_topn=None)
    equity_map["baseline"] = equity_df
    summary_rows.append(summary)

    baseline_trades_path = OUTPUT_DIR / f"trades_baseline_{ts}.csv"
    baseline_equity_path = OUTPUT_DIR / f"equity_baseline_{ts}.csv"
    if len(trades_df) > 0:
        trades_df.to_csv(baseline_trades_path, index=False, encoding="utf-8-sig")
    if len(equity_df) > 0:
        equity_df.to_csv(baseline_equity_path, index=False, encoding="utf-8-sig")

    for n in DAILY_TOPN_LIST:
        name = f"euclidean_daily_top{n}"
        print(f"回测 {name} ...")
        equity_df, trades_df, summary = run_portfolio_backtest(euclidean_trades, name, daily_topn=n)
        equity_map[name] = equity_df
        summary_rows.append(summary)

        trades_path = OUTPUT_DIR / f"trades_{name}_{ts}.csv"
        equity_path = OUTPUT_DIR / f"equity_{name}_{ts}.csv"
        if len(trades_df) > 0:
            trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        if len(equity_df) > 0:
            equity_df.to_csv(equity_path, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows).sort_values("total_return", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("绩效汇总")
    print("=" * 100)
    print(summary_df)

    summary_path = OUTPUT_DIR / f"summary_{ts}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_equity_curves(equity_map, OUTPUT_DIR / f"equity_curves_{ts}.png")
    plot_drawdown_curves(equity_map, OUTPUT_DIR / f"drawdown_curves_{ts}.png")

    print(f"\n结果已保存到: {OUTPUT_DIR}")
    print(f"summary_file = {summary_path}")


if __name__ == "__main__":
    main()