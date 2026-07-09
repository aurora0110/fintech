from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils.common.brick import calc_brick_values, calc_green_streak, calc_pre_red_green_structure, calc_true_streak
from utils.common.indicators import calc_macd, calc_ma, calc_zhixing_long_line, calc_zhixing_trend_line, safe_div
from utils.common.io import extract_stock_code_from_filename, read_stock_daily


FINAL_CANDIDATE_NAME = "BRICK_FINAL_CANDIDATE_V1"
FINAL_BRICK_PARAMS = (4, 6, 6)


class IndexSnapshotError(RuntimeError):
    """Raised when the local index snapshot cannot support final candidates."""


def _bool_text(value: object) -> str:
    return "是" if bool(value) else "否"


def _pct_rank_within_date(df: pd.DataFrame, value_col: str, rank_col: str) -> None:
    df[rank_col] = df.groupby("signal_date")[value_col].rank(method="average", pct=True)


def _limit_up_pct(code: str) -> float:
    code = str(code)
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("8", "9", "4")):
        return 0.30
    return 0.10


def _next_open_tradeability(x: pd.DataFrame, signal_idx: int, code: str) -> dict[str, object]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(x):
        return {
            "entry_date": pd.NaT,
            "entry_price": np.nan,
            "is_tradeable_next_open": False,
            "skip_reason": "missing_next_open",
        }
    row = x.iloc[entry_idx]
    open_price = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    prev_close = float(x.at[signal_idx, "close"])
    if not np.isfinite(open_price) or open_price <= 0:
        return {
            "entry_date": row["date"],
            "entry_price": np.nan,
            "is_tradeable_next_open": False,
            "skip_reason": "invalid_next_open",
        }
    one_price = np.isclose(open_price, high) and np.isclose(open_price, low) and np.isclose(open_price, close)
    if np.isfinite(prev_close) and prev_close > 0:
        limit_up_threshold = prev_close * (1 + _limit_up_pct(code) * 0.995)
        if one_price and open_price >= limit_up_threshold:
            return {
                "entry_date": row["date"],
                "entry_price": open_price,
                "is_tradeable_next_open": False,
                "skip_reason": "limit_up_one_price_no_entry",
            }
    return {
        "entry_date": row["date"],
        "entry_price": open_price,
        "is_tradeable_next_open": True,
        "skip_reason": "",
    }


def build_latest_base_minimal_signal_from_file(file_path: str | Path, stock_name: str = "") -> dict[str, object] | None:
    """Return latest-day BASE_MINIMAL_POOL_V1 signal for one stock, or None.

    This deliberately does not call ``utils.brick_filter.add_features`` so the
    BRICK parameter is explicit and aligned with the final validation candidate.
    """
    path = Path(file_path)
    code = extract_stock_code_from_filename(path)
    df = read_stock_daily(path, min_bars=160, allow_zero_volume=True, amount_default=0.0)
    if df is None or df.empty:
        return None
    x = df.copy().reset_index(drop=True)
    n, m1, m2 = FINAL_BRICK_PARAMS
    brick = calc_brick_values(x, n=n, m1=m1, m2=m2)
    for col in brick.columns:
        x[col] = brick[col]
    brick_red_flag = x["brick_red"].astype(bool)
    brick_green_flag = x["brick_green"].astype(bool)
    x["trend_line"] = calc_zhixing_trend_line(x["close"])
    x["long_line"] = calc_zhixing_long_line(x["close"])
    x["vol_ma5_prev"] = x["volume"].shift(1).rolling(5).mean()
    x["signal_vs_ma5"] = pd.Series(safe_div(x["volume"], x["vol_ma5_prev"]), index=x.index)
    x["prev_green_streak"] = pd.Series(calc_green_streak(x["brick_green"].to_numpy()), index=x.index).shift(1)
    x["red_streak"] = pd.Series(calc_true_streak(x["brick_red"].to_numpy()), index=x.index)
    green_before_red, red_before_pullback = calc_pre_red_green_structure(
        x["brick_red"].to_numpy(),
        x["brick_green"].to_numpy(),
    )
    x["green_before_current_red"] = pd.Series(green_before_red, index=x.index)
    x["red_before_pullback"] = pd.Series(red_before_pullback, index=x.index)
    x["rebound_ratio"] = pd.Series(safe_div(x["brick_red_len"], x["brick_green_len"].shift(1)), index=x.index)
    x["brick_strict_reversal_066"] = (
        brick_green_flag.shift(1, fill_value=False)
        & brick_red_flag
        & x["brick_red_len"].gt(x["brick_green_len"].shift(1) * 0.66).fillna(False)
        & x["brick_green_len"].shift(1).gt(0).fillna(False)
    )
    x["minimal_green_to_red"] = (
        brick_red_flag
        & x["red_streak"].eq(1).fillna(False)
        & x["green_before_current_red"].ge(1).fillna(False)
        & x["close"].pct_change().notna()
    )
    latest_idx = int(len(x) - 1)
    if latest_idx < 1 or not bool(x.at[latest_idx, "minimal_green_to_red"]):
        return None
    signal_date = pd.Timestamp(x.at[latest_idx, "date"])
    tradeability = _next_open_tradeability(x, latest_idx, code)
    close_to_trend = float(safe_div(x.at[latest_idx, "close"] - x.at[latest_idx, "trend_line"], x.at[latest_idx, "trend_line"]))
    brick_power_raw = float(x.at[latest_idx, "rebound_ratio"]) if pd.notna(x.at[latest_idx, "rebound_ratio"]) else 0.0
    return {
        "strategy_name": FINAL_CANDIDATE_NAME,
        "stock_code": code,
        "stock_name": stock_name or code,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "signal_type": "GREEN_TO_RED_REBOUND_066" if bool(x.at[latest_idx, "brick_strict_reversal_066"]) else "GREEN_TO_RED",
        "candidate_pool": "BASE_MINIMAL_POOL_V1",
        "brick_param_name": "466",
        "brick_n": n,
        "brick_m1": m1,
        "brick_m2": m2,
        "brick_value": float(x.at[latest_idx, "brick"]),
        "prev_brick_value": float(x.at[latest_idx, "brick_prev"]) if pd.notna(x.at[latest_idx, "brick_prev"]) else np.nan,
        "brick_red_len": float(x.at[latest_idx, "brick_red_len"]),
        "brick_green_len_prev": float(x["brick_green_len"].shift(1).iat[latest_idx])
        if pd.notna(x["brick_green_len"].shift(1).iat[latest_idx])
        else np.nan,
        "rebound_ratio": brick_power_raw,
        "brick_power_raw": max(0.0, min(brick_power_raw, 8.0)) if np.isfinite(brick_power_raw) else 0.0,
        "close": float(x.at[latest_idx, "close"]),
        "trend_line": float(x.at[latest_idx, "trend_line"]),
        "long_line": float(x.at[latest_idx, "long_line"]),
        "trend_dev": close_to_trend,
        "close_to_trend": close_to_trend,
        "signal_vs_ma5": float(x.at[latest_idx, "signal_vs_ma5"]) if pd.notna(x.at[latest_idx, "signal_vs_ma5"]) else np.nan,
        "volume": float(x.at[latest_idx, "volume"]),
        "amount": float(x.at[latest_idx, "amount"]) if "amount" in x.columns and pd.notna(x.at[latest_idx, "amount"]) else np.nan,
        "signal_low": float(x.at[latest_idx, "low"]),
        **tradeability,
    }


def load_index_state_for_date(signal_date: str | pd.Timestamp, root: str | Path) -> tuple[dict[str, object], str]:
    root_path = Path(root)
    target = pd.Timestamp(signal_date)
    path = root_path / "data" / "index" / "sh000001_daily_latest.csv"
    try:
        source = str(path.relative_to(root_path))
    except ValueError:
        source = str(path)
    required_text = target.strftime("%Y-%m-%d")
    if not path.exists():
        raise IndexSnapshotError(
            "missing local index snapshot for BRICK_FINAL_CANDIDATE_V1; "
            f"snapshot_path={source}; snapshot_max_date=N/A; required_signal_date={required_text}; "
            "please update local index snapshot before generating formal candidates"
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    rename = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close"}
    df = df.rename(columns=rename)
    required_cols = {"date", "close"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise IndexSnapshotError(
            "local index snapshot missing required columns for BRICK_FINAL_CANDIDATE_V1; "
            f"snapshot_path={source}; snapshot_max_date=N/A; required_signal_date={required_text}; "
            f"missing_columns={missing_cols}; please update local index snapshot before generating formal candidates"
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    if df.empty:
        raise IndexSnapshotError(
            "local index snapshot has no valid rows for BRICK_FINAL_CANDIDATE_V1; "
            f"snapshot_path={source}; snapshot_max_date=N/A; required_signal_date={required_text}; "
            "please update local index snapshot before generating formal candidates"
        )
    snapshot_max_date = pd.Timestamp(df["date"].max())
    snapshot_max_text = snapshot_max_date.strftime("%Y-%m-%d")
    if snapshot_max_date < target:
        raise IndexSnapshotError(
            "local index snapshot is not up to date for BRICK_FINAL_CANDIDATE_V1; "
            f"snapshot_path={source}; snapshot_max_date={snapshot_max_text}; required_signal_date={required_text}; "
            "please update local index snapshot before generating formal candidates"
        )
    macd = calc_macd(df["close"])
    df["index_macd_dif"] = macd["dif"]
    df["index_macd_dea"] = macd["dea"]
    df["index_macd_hist"] = macd["macd"]
    df["index_ma20"] = calc_ma(df["close"], 20)
    hit = df[df["date"].eq(target)]
    if hit.empty:
        raise IndexSnapshotError(
            "local index snapshot does not contain required signal date for BRICK_FINAL_CANDIDATE_V1; "
            f"snapshot_path={source}; snapshot_max_date={snapshot_max_text}; required_signal_date={required_text}; "
            "please update local index snapshot before generating formal candidates"
        )
    row = hit.iloc[-1]
    dif = float(row["index_macd_dif"])
    dea = float(row["index_macd_dea"])
    hist = float(row["index_macd_hist"])
    prev_rows = df[df["date"].lt(target)].tail(1)
    prev_dif = float(prev_rows["index_macd_dif"].iloc[-1]) if not prev_rows.empty else np.nan
    dif_up = np.isfinite(prev_dif) and dif > prev_dif
    return {
        "index_state_missing": False,
        "index_state_date": target.strftime("%Y-%m-%d"),
        "index_macd_dif": dif,
        "index_macd_dea": dea,
        "index_macd_hist": hist,
        "index_hist_red": hist > 0,
        "index_dif_up": bool(dif_up),
        "index_red_hist_or_repair": (hist > 0) or bool(dif_up),
    }, source


def finalize_brick_candidate_rows(
    raw_rows: Iterable[dict[str, object]],
    *,
    root: str | Path,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(list(raw_rows))
    if df.empty:
        return df, df
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    latest_date = df["signal_date"].max()
    df = df[df["signal_date"].eq(latest_date)].copy()
    for col in ["brick_power_raw", "close_to_trend", "signal_vs_ma5", "trend_dev"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["line_near_5pct"] = (1.0 - (df["close_to_trend"] - 0.05).abs() / 0.04).clip(0.0, 1.0)
    df["volume_105"] = (1.0 - (df["signal_vs_ma5"] - 1.05).abs() / 0.45).clip(0.0, 1.0)
    for value_col, rank_col in [
        ("brick_power_raw", "sort_brick_rank"),
        ("line_near_5pct", "sort_line_near_5_rank"),
        ("volume_105", "sort_volume_105_rank"),
    ]:
        _pct_rank_within_date(df, value_col, rank_col)
    df["w_l50_b20_v30_score"] = (
        0.50 * df["sort_line_near_5_rank"]
        + 0.20 * df["sort_brick_rank"]
        + 0.30 * df["sort_volume_105_rank"]
    )
    index_state, index_source = load_index_state_for_date(latest_date, root)
    for key, value in index_state.items():
        df[key] = value
    df["index_source"] = index_source
    df["pass_index_repair"] = df["index_red_hist_or_repair"].fillna(False).astype(bool)
    df["pass_trend_dev"] = df["trend_dev"].le(0.08).fillna(False)
    df["final_pass"] = (
        df["is_tradeable_next_open"].fillna(False).astype(bool)
        & df["pass_index_repair"]
        & df["pass_trend_dev"]
    )
    reasons = []
    for row in df.itertuples(index=False):
        blocked: list[str] = []
        if not bool(getattr(row, "is_tradeable_next_open", False)):
            blocked.append(str(getattr(row, "skip_reason", "")) or "not_tradeable_next_open")
        if not bool(getattr(row, "pass_index_repair", False)):
            blocked.append("index_not_red_hist_or_repair")
        if not bool(getattr(row, "pass_trend_dev", False)):
            blocked.append("trend_dev_gt_8pct")
        reasons.append(";".join(blocked))
    df["blocked_reason"] = reasons
    ranked = df.sort_values(
        ["signal_date", "w_l50_b20_v30_score", "stock_code"],
        ascending=[True, False, True],
    ).copy()
    ranked["raw_rank"] = ranked.groupby("signal_date").cumcount() + 1
    final_df = ranked[ranked["final_pass"]].copy()
    final_df["final_rank"] = final_df.groupby("signal_date").cumcount() + 1
    final_df = final_df[final_df["final_rank"].le(top_n)].copy()
    for frame in (ranked, final_df):
        if "signal_date" in frame.columns:
            frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.strftime("%Y-%m-%d")
        if "entry_date" in frame.columns:
            frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return ranked, final_df


def render_final_candidate_text(final_df: pd.DataFrame, raw_df: pd.DataFrame, trade_date: str) -> str:
    lines = [
        "=" * 18 + f" BRICK_FINAL_CANDIDATE_V1 Top5 {trade_date} " + "=" * 18,
        "口径：466 + BASE_MINIMAL_POOL_V1 + W_L50_B20_V30 Top5 + 指数红柱/修复 + trend_dev<=8%。",
        "买卖验证口径：T日信号，T+1开盘买，T+4收盘卖；本文件只保存候选池，不代表已完成交易。",
        f"原始最小池信号数：{len(raw_df)}；最终候选数：{len(final_df)}。",
    ]
    if final_df.empty:
        lines.append("暂无通过最终过滤的候选。")
    for row in final_df.sort_values("final_rank").itertuples(index=False):
        lines.append(
            f"{int(row.final_rank):02d} | {row.stock_code} {row.stock_name} | "
            f"score={row.w_l50_b20_v30_score:.4f} | "
            f"brick_power={row.brick_power_raw:.4f} | "
            f"trend_dev={row.trend_dev:.4f} | "
            f"volume_ratio_5={row.signal_vs_ma5:.4f} | "
            f"指数修复={_bool_text(row.index_red_hist_or_repair)} | "
            f"收盘={row.close:.2f}"
        )
    lines.append("=" * 64)
    return "\n".join(lines)
