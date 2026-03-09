from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from core.data_loader import load_price_directory
from core.market_rules import is_limit_down, is_limit_up
from core.metrics import compute_metrics
from utils.multi_factor_research.factor_calculator import (
    FACTOR_COLUMNS,
    FACTOR_NAME_MAP,
    PENALTY_COLUMNS,
    build_repair_v2_candidate_mask,
    build_repair_v3_candidate_mask,
    build_trend_rebuilt_candidate_mask,
    build_trend_start_candidate_mask,
    count_trend_rebuilt_confirmation_hits,
    count_trend_rebuilt_support_hits,
    prepare_factor_frame,
)
from utils.multi_factor_research.weight_optimizer import PENALTY_WEIGHTS

持仓高位钝化扣分 = 0.08
中文因子到字段名 = {v: k for k, v in FACTOR_NAME_MAP.items()}


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 1_000_000.0
    max_positions: int = 10
    commission_rate: float = 0.0003
    slippage_rate: float = 0.001
    stamp_duty_rate: float = 0.001
    min_lot: int = 100
    top_quantile: float = 0.30
    min_score: float | None = None
    replacement_threshold: float = 0.03
    min_hold_days_for_replace: int = 5
    max_daily_replacements: int = 1
    buy_mode: str = "staged"
    exit_profile: dict | None = None
    use_trend_start_pool: bool = False
    trend_pool_mode: str = "start_v1"
    min_confirmation_hits: int = 1
    min_support_hits: int = 1
    rebuilt_min_confirmation_hits: int = 1
    rebuilt_min_support_hits: int = 2
    rebuilt_require_close_above_trend: bool = True
    rebuilt_require_close_above_bull_bear: bool = True
    stock_stop_cooldown_count: int = 0
    stock_stop_cooldown_days: int = 0
    stock_stop_skip_next_buys: int = 0
    reentry_mode: str = "bull_bear_only"
    recovery_min_confirmation_hits: int = 1
    recovery_min_support_hits: int = 2
    recovery_require_close_above_trend: bool = True
    recovery_require_close_above_bull_bear: bool = True
    sideways_mode: str = "关闭"
    sideways_filter_threshold: float = 0.55
    sideways_score_penalty_scale: float = 0.0


def _parse_weight_spec(spec: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for part in spec.split(";"):
        text = part.strip()
        if not text:
            continue
        name, value = text.split("=")
        因子名 = name.strip()
        字段名 = 中文因子到字段名.get(因子名, 因子名)
        weights[字段名] = float(value.strip().replace("分", ""))
    return weights


def _parse_scorecard_payload(payload: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    add_weights = _parse_weight_spec(payload.get("加分组合字符串", ""))
    penalty_weights = _parse_weight_spec(payload.get("扣分组合字符串", ""))
    return add_weights, penalty_weights


def _load_refined_weights(refined_root: Path) -> dict[str, dict]:
    mapping = {
        "fixed_take_profit": refined_root / "固定涨幅止盈_40pct" / "summary.json",
        "fixed_days": refined_root / "固定持有_30天" / "summary.json",
        "tiered": refined_root / "分批顺序止盈" / "summary.json",
        "j100_full_exit": refined_root / "分批顺序止盈" / "summary.json",
    }
    result: dict[str, dict] = {}
    for key, path in mapping.items():
        payload = json.loads(path.read_text())
        result[key] = payload["best_weighted_combos"]["best_by_score"]
    return result


def _load_formal_scorecard(scorecard_root: Path) -> tuple[Dict[str, float], Dict[str, float]]:
    payload = json.loads((scorecard_root / "正式评分卡.json").read_text())
    return _parse_scorecard_payload(payload)


def _prepare_stock_frames(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    prepared: Dict[str, pd.DataFrame] = {}
    for code, df in stock_data.items():
        raw = df.reset_index().copy()
        factor_df = prepare_factor_frame(raw, burst_window=20)
        factor_df["limit_pct"] = raw["limit_pct"].to_numpy()
        factor_df["is_suspended"] = raw["is_suspended"].to_numpy()
        factor_df = factor_df.set_index("date")
        prepared[code] = factor_df
    return prepared


def _calc_net_score_series(df: pd.DataFrame, add_weights: Dict[str, float], penalty_weights: Dict[str, float]) -> pd.Series:
    factor_score = sum(df[col].fillna(0.0) * weight for col, weight in add_weights.items())
    penalty_score = sum(df[col].fillna(0.0) * weight for col, weight in penalty_weights.items())
    return factor_score - penalty_score


横盘震荡扣分列 = [
    "flat_trend_slope_penalty",
    "line_entanglement_penalty",
    "box_oscillation_penalty",
    "sideways_without_confirmation_penalty",
    "amplitude_without_progress_penalty",
]


def _sideways_release_confirmed(row: pd.Series) -> bool:
    support_hits = (
        int(float(row.get("staged_volume_burst_factor", 0.0)) >= 0.2)
        + int(float(row.get("macd_dif_factor", 0.0)) >= 0.2)
        + int(float(row.get("rsi_bull_factor", 0.0)) >= 0.2)
        + int(float(row.get("daily_ma_bull_factor", 0.0)) >= 0.5)
        + int(float(row.get("key_k_support_factor", 0.0)) > 0.0)
    )
    prev_volume = float(row.get("prev_volume", row.get("volume", 0.0)))
    clear_release = (
        bool(row.get("is_bullish", False))
        and prev_volume > 0
        and float(row.get("volume", 0.0)) >= prev_volume * 1.8
        and float(row.get("return_pct", 0.0)) >= 0.03
    )
    return clear_release or support_hits >= 2


def _sideways_penalty_value(row: pd.Series) -> float:
    values = [float(row.get(col, 0.0)) for col in 横盘震荡扣分列]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_signal_map(
    prepared: Dict[str, pd.DataFrame],
    all_dates: List[pd.Timestamp],
    add_weights: Dict[str, float],
    penalty_weights: Dict[str, float],
    top_quantile: float,
    buy_mode: str,
    min_score: float | None,
    use_trend_start_pool: bool,
    trend_pool_mode: str,
    min_confirmation_hits: int,
    min_support_hits: int,
    rebuilt_min_confirmation_hits: int,
    rebuilt_min_support_hits: int,
    rebuilt_require_close_above_trend: bool,
    rebuilt_require_close_above_bull_bear: bool,
    sideways_mode: str,
    sideways_filter_threshold: float,
    sideways_score_penalty_scale: float,
) -> Dict[pd.Timestamp, List[dict]]:
    rows: List[dict] = []
    date_to_idx = {dt: idx for idx, dt in enumerate(all_dates)}
    for code, df in prepared.items():
        net_score = _calc_net_score_series(df, add_weights, penalty_weights)
        if use_trend_start_pool:
            if trend_pool_mode == "repair_v2":
                signal_mask = build_repair_v2_candidate_mask(
                    df,
                    j_threshold=-5.0,
                    min_core_hits=rebuilt_min_confirmation_hits,
                    min_structure_hits=rebuilt_min_support_hits,
                )
            elif trend_pool_mode == "repair_v3":
                signal_mask = build_repair_v3_candidate_mask(
                    df,
                    j_threshold=-5.0,
                    min_primary_hits=rebuilt_min_confirmation_hits,
                    min_support_hits=rebuilt_min_support_hits,
                )
            elif trend_pool_mode == "rebuilt_v1":
                signal_mask = build_trend_rebuilt_candidate_mask(
                    df,
                    j_threshold=(-5 if buy_mode == "strict_full" else 13),
                    min_confirmation_hits=rebuilt_min_confirmation_hits,
                    min_support_hits=rebuilt_min_support_hits,
                    require_close_above_trend=rebuilt_require_close_above_trend,
                    require_close_above_bull_bear=rebuilt_require_close_above_bull_bear,
                )
            else:
                signal_mask = build_trend_start_candidate_mask(
                    df,
                    j_threshold=(-5 if buy_mode == "strict_full" else 13),
                    require_trend_above=True,
                    min_confirmation_hits=min_confirmation_hits,
                    min_support_hits=min_support_hits,
                )
        else:
            if buy_mode == "strict_full":
                signal_mask = df["J"].lt(-5) & df["trend_line"].gt(df["bull_bear_line"])
            else:
                signal_mask = df["J"].lt(13) & df["trend_line"].gt(df["bull_bear_line"])
        for dt in df.index[signal_mask]:
            idx = date_to_idx.get(dt)
            if idx is None or idx + 1 >= len(all_dates):
                continue
            row = df.loc[dt]
            if float(row.get("bearish_max_volume_60_penalty", 0.0)) >= 0.5:
                continue
            sideways_penalty = _sideways_penalty_value(row)
            if sideways_mode in {"只过滤", "过滤+降分"}:
                if sideways_penalty >= sideways_filter_threshold and not _sideways_release_confirmed(row):
                    continue
            final_score = float(net_score.loc[dt])
            if sideways_mode in {"只降分", "过滤+降分"}:
                final_score -= sideways_penalty * sideways_score_penalty_scale
            rows.append(
                {
                    "signal_date": dt,
                    "exec_date": all_dates[idx + 1],
                    "code": code,
                    "score": final_score,
                }
            )
    if not rows:
        return {}
    signal_df = pd.DataFrame(rows).sort_values(["signal_date", "score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    cutoff = signal_df["score"].quantile(max(0.0, min(1.0, 1.0 - top_quantile)))
    signal_df = signal_df[signal_df["score"] >= cutoff].copy()
    if min_score is not None:
        signal_df = signal_df[signal_df["score"] >= float(min_score)].copy()
    signal_map: Dict[pd.Timestamp, List[dict]] = {}
    for row in signal_df.itertuples(index=False):
        signal_map.setdefault(row.exec_date, []).append({"code": row.code, "score": float(row.score), "signal_date": row.signal_date})
    for dt in signal_map:
        signal_map[dt].sort(key=lambda item: (-item["score"], item["code"]))
    return signal_map


def _sell_position(cash: float, shares: int, raw_price: float, config: PortfolioConfig) -> tuple[float, float]:
    exec_price = raw_price * (1.0 - config.slippage_rate)
    gross = shares * exec_price
    fee = gross * config.commission_rate
    tax = gross * config.stamp_duty_rate
    return cash + gross - fee - tax, exec_price


def _buy_shares(cash: float, entry_price: float, available_slots: int, config: PortfolioConfig) -> int:
    if available_slots <= 0 or entry_price <= 0:
        return 0
    allocation = cash / available_slots
    shares = int(allocation / entry_price / config.min_lot) * config.min_lot
    if shares <= 0:
        return 0
    total_cost = shares * entry_price * (1.0 + config.commission_rate)
    return shares if total_cost <= cash else 0


def _buy_shares_for_value(cash: float, entry_price: float, target_value: float, config: PortfolioConfig) -> int:
    if entry_price <= 0 or target_value <= 0:
        return 0
    value = min(cash, target_value)
    shares = int(value / entry_price / config.min_lot) * config.min_lot
    if shares <= 0:
        return 0
    total_cost = shares * entry_price * (1.0 + config.commission_rate)
    return shares if total_cost <= cash else 0


def _position_current_score(
    code: str,
    current_date: pd.Timestamp,
    prepared: Dict[str, pd.DataFrame],
    add_weights: Dict[str, float],
    penalty_weights: Dict[str, float],
    fallback: float,
    sideways_mode: str,
    sideways_score_penalty_scale: float,
) -> float:
    df = prepared[code]
    if current_date not in df.index:
        return fallback
    row = df.loc[current_date]
    reward = sum(float(row[col]) * weight for col, weight in add_weights.items())
    penalty = sum(float(row[col]) * weight for col, weight in penalty_weights.items())
    score = reward - penalty
    if sideways_mode in {"只降分", "过滤+降分"}:
        score -= _sideways_penalty_value(row) * sideways_score_penalty_scale
    return score


def _position_dynamic_penalty(position: dict, row: pd.Series) -> float:
    penalty = 0.0
    if float(row.get("J", 0.0)) > 80 and float(row.get("close", 0.0)) <= float(position["entry_price"]):
        penalty += 持仓高位钝化扣分
    return penalty


def _profile_value(config: PortfolioConfig, key: str, default):
    if not config.exit_profile:
        return default
    return config.exit_profile.get(key, default)


def _row_satisfies_structure_recovery(row: pd.Series, config: PortfolioConfig) -> bool:
    if float(row.get("bearish_max_volume_60_penalty", 0.0)) >= 0.5:
        return False
    if float(row.get("double_break_bull_bear_penalty", 0.0)) >= 0.5:
        return False
    if float(row.get("trend_line", 0.0)) <= float(row.get("bull_bear_line", 0.0)):
        return False
    if config.recovery_require_close_above_trend and float(row.get("close", 0.0)) < float(row.get("trend_line", 0.0)):
        return False
    if config.recovery_require_close_above_bull_bear and float(row.get("close", 0.0)) < float(row.get("bull_bear_line", 0.0)):
        return False
    return (
        count_trend_rebuilt_confirmation_hits(row) >= int(config.recovery_min_confirmation_hits)
        and count_trend_rebuilt_support_hits(row) >= int(config.recovery_min_support_hits)
    )


def _should_exit_on_weak_close(position: dict) -> bool:
    if not bool(position.get("启用三天连续弱收盘退出", False)):
        return False
    if int(position.get("连续弱收盘天数", 0)) < 3:
        return False
    if bool(position.get("三天弱收盘仅盈利保护阶段", False)):
        return bool(position.get("利润保护已启用", False))
    return True


def _should_use_half_stop(config: PortfolioConfig) -> bool:
    return bool(_profile_value(config, "启用半仓止损", False))


def _extended_structure_exit_days(position: dict, row: pd.Series, config: PortfolioConfig) -> int:
    base_days = int(_profile_value(config, "结构退出最长持有天数", 60))
    if not bool(_profile_value(config, "强票延长持有启用", False)):
        return base_days

    extra_days = int(_profile_value(config, "强票延长额外天数", 0))
    if extra_days <= 0:
        return base_days

    min_gain = float(_profile_value(config, "强票延长最低浮盈", 0.0))
    max_price = float(position.get("最高价", position["entry_price"]))
    if max_price < position["entry_price"] * (1.0 + min_gain):
        return base_days

    if bool(_profile_value(config, "强票延长要求利润保护", True)) and not bool(position.get("利润保护已启用", False)):
        return base_days

    if bool(_profile_value(config, "强票延长要求站上趋势线", True)) and float(row.get("close", 0.0)) < float(row.get("trend_line", 0.0)):
        return base_days

    if bool(_profile_value(config, "强票延长要求站上多空线", True)) and float(row.get("close", 0.0)) < float(row.get("bull_bear_line", 0.0)):
        return base_days

    return base_days + extra_days


def _is_trend_break_reason(reason: str) -> bool:
    return reason in {
        "多空线压制卖出",
        "趋势线破位卖出",
        "固定天数后多空线压制卖出",
        "固定天数后趋势线破位卖出",
        "固定涨幅后多空线压制卖出",
        "固定涨幅后趋势退出",
        "趋势主线重构_多空线压制卖出",
        "趋势主线重构_利润保护破位卖出",
        "趋势主线重构_最短持有后趋势破位卖出",
        "轻量分批趋势退出",
    }


def _is_half_stop_reason(reason: str) -> bool:
    return _is_stop_like_exit(reason)


def _sell_partial_position(
    cash: float,
    positions: Dict[str, dict],
    trades: List[dict],
    code: str,
    current_date: pd.Timestamp,
    raw_price: float,
    reason: str,
    share_ratio: float,
    config: PortfolioConfig,
) -> tuple[float, int, float]:
    pos = positions[code]
    target_shares = int((pos["shares"] * share_ratio) / config.min_lot) * config.min_lot
    shares_to_sell = min(pos["shares"], max(target_shares, config.min_lot))
    if shares_to_sell <= 0:
        return cash, 0, 0.0
    cash, exec_price = _sell_position(cash, shares_to_sell, raw_price, config)
    gross = shares_to_sell * exec_price
    fee = gross * config.commission_rate
    tax = gross * config.stamp_duty_rate
    pos["累计卖出到账"] += gross - fee - tax
    trades.append(
        {
            "position_id": pos["position_id"],
            "code": code,
            "entry_date": pos["entry_date"],
            "exit_date": current_date,
            "entry_price": pos["entry_price"],
            "exit_price": exec_price,
            "shares": shares_to_sell,
            "reason": reason,
        }
    )
    pos["shares"] -= shares_to_sell
    return cash, shares_to_sell, exec_price


def _schedule_half_stop(position: dict, current_date: pd.Timestamp, reason: str, trigger_low: float) -> None:
    position["半仓止损待执行"] = True
    position["半仓止损触发日"] = current_date
    position["半仓止损原因"] = reason
    position["半仓止损参考低点"] = float(trigger_low)


def _activate_half_stop_watch(position: dict, current_date: pd.Timestamp) -> None:
    position["半仓止损待执行"] = False
    position["半仓止损观察中"] = True
    position["半仓止损卖半仓日"] = current_date


def _schedule_half_stop_full_exit(position: dict, current_date: pd.Timestamp) -> None:
    position["半仓止损待次日清仓"] = True
    position["半仓止损清仓触发日"] = current_date


def _handle_exit_or_half_stop(
    cash: float,
    positions: Dict[str, dict],
    trades: List[dict],
    round_trips: List[dict],
    stock_risk_state: Dict[str, dict],
    code: str,
    current_date: pd.Timestamp,
    raw_price: float,
    reason: str,
    config: PortfolioConfig,
    trigger_low: float | None = None,
) -> tuple[float, bool]:
    pos = positions[code]
    if _should_use_half_stop(config) and _is_half_stop_reason(reason):
        if not pos.get("半仓止损待执行", False) and not pos.get("半仓止损观察中", False) and not pos.get("半仓止损待次日清仓", False):
            _schedule_half_stop(pos, current_date, reason, float(trigger_low if trigger_low is not None else raw_price))
            pos["半仓止损触发次数"] = int(pos.get("半仓止损触发次数", 0)) + 1
            return cash, False
    cash = _close_position(cash, positions, trades, code, current_date, raw_price, reason, config)
    _complete_round_trip(round_trips, pos, current_date, reason)
    _update_stock_cooldown(stock_risk_state, code, current_date, reason, config)
    return cash, True


def _is_high_volume_weak_bar(row: pd.Series) -> bool:
    volume = float(row.get("volume", 0.0))
    prev_volume = float(row.get("prev_volume", volume))
    volume_ma20 = float(row.get("volume_ma20", prev_volume))
    volume_expanded = (prev_volume > 0 and volume >= prev_volume * 1.5) or (volume_ma20 > 0 and volume >= volume_ma20 * 1.8)
    weak_bar = bool(row.get("is_bearish", False)) or float(row.get("return_pct", 0.0)) <= 0.005
    return volume_expanded and weak_bar


def _is_high_position(row: pd.Series, mode: str) -> bool:
    if mode == "ma_bias":
        ma20 = float(row.get("MA20", row.get("trend_line", 0.0)))
        if ma20 <= 0:
            return False
        ma_bias = (float(row.get("high", 0.0)) - ma20) / ma20
        threshold = max(float(row.get("avg_amplitude_20", 0.0)) * 1.5, 0.08)
        return ma_bias >= threshold
    trend_line = float(row.get("trend_line", 0.0))
    if trend_line <= 0:
        return False
    trend_bias = (float(row.get("high", 0.0)) - trend_line) / trend_line
    threshold = max(float(row.get("trend_bias_q70", 0.0)), 0.06)
    return trend_bias >= threshold


def _should_exit_high_volume(label: str, position: dict, row: pd.Series, config: PortfolioConfig) -> bool:
    if label != "high_volume_exit":
        return False
    if not bool(position.get("利润保护已启用", False)) and float(row.get("close", 0.0)) <= float(position["entry_price"]) * 1.03:
        return False
    mode = str(_profile_value(config, "高位定义", "trend_bias"))
    if not _is_high_position(row, mode):
        return False
    return _is_high_volume_weak_bar(row)


def _apply_trend_risk_controls(
    cash: float,
    positions: Dict[str, dict],
    trades: List[dict],
    round_trips: List[dict],
    stock_risk_state: Dict[str, dict],
    code: str,
    current_date: pd.Timestamp,
    row: pd.Series,
    pos: dict,
    hold_bars: int,
    config: PortfolioConfig,
) -> tuple[float, bool]:
    启动失败天数 = int(_profile_value(config, "启动失败观察天数", 0))
    启动失败最小涨幅 = float(_profile_value(config, "启动失败最小涨幅", 0.0))
    保本触发涨幅 = float(_profile_value(config, "保本触发涨幅", 0.0))
    保本止损系数 = float(_profile_value(config, "保本止损系数", 1.0))
    利润保护模式 = str(_profile_value(config, "利润保护模式", "固定"))
    自适应倍数 = float(_profile_value(config, "利润保护自适应倍数", 1.5))

    pos["最高价"] = max(float(pos.get("最高价", pos["entry_price"])), float(row["high"]))
    trigger_gain = 保本触发涨幅
    if 利润保护模式 == "振幅":
        trigger_gain = max(trigger_gain, float(row.get("avg_amplitude_20", 0.0)) * 自适应倍数)
    elif 利润保护模式 == "ATR":
        trigger_gain = max(trigger_gain, float(row.get("ATR14", 0.0)) / max(pos["entry_price"], 1e-9) * 自适应倍数)
    elif 利润保护模式 == "趋势线偏离":
        trigger_gain = max(trigger_gain, float(row.get("trend_bias_q70", 0.0)) * 自适应倍数)
    elif 利润保护模式 == "历史启动特征":
        trigger_gain = max(trigger_gain, float(row.get("historical_startup_gain_q60", 0.0)) * 自适应倍数)

    if trigger_gain > 0 and pos["最高价"] >= pos["entry_price"] * (1.0 + trigger_gain):
        pos["stop_price"] = max(pos["stop_price"], pos["entry_price"] * 保本止损系数)
        pos["利润保护已启用"] = True

    if 启动失败天数 > 0 and hold_bars >= 启动失败天数:
        if pos["最高价"] < pos["entry_price"] * (1.0 + 启动失败最小涨幅):
            cash, closed = _handle_exit_or_half_stop(
                cash,
                positions,
                trades,
                round_trips,
                stock_risk_state,
                code,
                current_date,
                float(row["close"]),
                "启动失败卖出",
                config,
                trigger_low=float(row["low"]),
            )
            return cash, closed

    第二层启动失败天数 = int(_profile_value(config, "第二层启动失败观察天数", 0))
    第二层启动失败最小涨幅 = float(_profile_value(config, "第二层启动失败最小涨幅", 0.0))
    if 第二层启动失败天数 > 0 and hold_bars >= 第二层启动失败天数 and not bool(pos.get("利润保护已启用", False)):
        if pos["最高价"] < pos["entry_price"] * (1.0 + 第二层启动失败最小涨幅):
            cash, closed = _handle_exit_or_half_stop(
                cash,
                positions,
                trades,
                round_trips,
                stock_risk_state,
                code,
                current_date,
                float(row["close"]),
                "第二层启动失败卖出",
                config,
                trigger_low=float(row["low"]),
            )
            return cash, closed

    return cash, False


def _position_replace_eligible(label: str, position: dict, df: pd.DataFrame, idx_now: int, idx_entry: int, hold_bars: int) -> bool:
    if hold_bars < position.get("最短换仓持有天数", 5):
        return False
    if label in {"fixed_take_profit", "fixed_days"}:
        check_bars = min(5, hold_bars)
        early_window = df.iloc[idx_entry:idx_entry + check_bars]
        if early_window.empty:
            return False
        early_high = float(early_window["high"].max())
        if early_high >= position["entry_price"] * 1.02:
            return False
        return hold_bars >= 5
    if label == "tiered" and not position.get("step1", False):
        return hold_bars >= 5
    if label == "j100_full_exit":
        return hold_bars >= 5
    return True


def _can_add_tranche(row: pd.Series) -> bool:
    if bool(row.get("is_suspended", False)):
        return False
    if float(row.get("trend_line", 0.0)) <= float(row.get("bull_bear_line", 0.0)):
        return False
    if float(row.get("bearish_max_volume_60_penalty", 0.0)) >= 0.5:
        return False
    key_close = row.get("关键K收盘价")
    if pd.notna(key_close) and float(row["close"]) < float(key_close):
        return False
    return float(row["close"]) >= float(row["trend_line"])


def _desired_buy_batch(row: pd.Series, buy_mode: str) -> int:
    j_value = float(row.get("J", 999.0))
    if buy_mode == "strict_full":
        return 3 if j_value < -5 else 0
    if buy_mode == "full":
        return 3 if j_value < 13 else 0
    if j_value < -5:
        return 3
    if j_value < 0:
        return 2
    if j_value < 13:
        return 1
    return 0


def _batch_target_ratio(batch: int) -> float:
    if batch <= 0:
        return 0.0
    if batch == 1:
        return 0.3
    if batch == 2:
        return 0.6
    return 1.0


def _close_position(
    cash: float,
    positions: Dict[str, dict],
    trades: List[dict],
    code: str,
    current_date: pd.Timestamp,
    raw_price: float,
    reason: str,
    config: PortfolioConfig,
) -> float:
    pos = positions[code]
    cash, exec_price = _sell_position(cash, pos["shares"], raw_price, config)
    gross = pos["shares"] * exec_price
    fee = gross * config.commission_rate
    tax = gross * config.stamp_duty_rate
    pos["累计卖出到账"] += gross - fee - tax
    trades.append(
        {
            "position_id": pos["position_id"],
            "code": code,
            "entry_date": pos["entry_date"],
            "exit_date": current_date,
            "entry_price": pos["entry_price"],
            "exit_price": exec_price,
            "shares": pos["shares"],
            "reason": reason,
        }
    )
    del positions[code]
    return cash


def _complete_round_trip(round_trips: List[dict], position: dict, current_date: pd.Timestamp, reason: str) -> None:
    cost_total = float(position.get("累计买入成本", 0.0))
    proceeds_total = float(position.get("累计卖出到账", 0.0))
    if cost_total <= 0:
        return
    round_trips.append(
        {
            "持仓编号": position["position_id"],
            "股票代码": position["code"],
            "首次买入日期": position["entry_date"],
            "最终卖出日期": current_date,
            "完整持有收益率": proceeds_total / cost_total - 1.0,
            "持有天数": max(0, int((current_date - position["entry_date"]).days)),
            "最终卖出原因": reason,
        }
    )


def _is_stop_like_exit(reason: str) -> bool:
    return reason in {"止损卖出", "启动失败卖出", "滴滴止损", "两根跌破多空线止损", "半仓止损后破低清仓"}


def _update_stock_cooldown(
    stock_risk_state: Dict[str, dict],
    code: str,
    current_date: pd.Timestamp,
    reason: str,
    config: PortfolioConfig,
) -> None:
    state = stock_risk_state.setdefault(
        code,
        {
            "连续止损次数": 0,
            "冷却到期日": None,
            "待跳过买入次数": 0,
            "未重新站上多空线前禁买": False,
            "结构恢复前禁买": False,
        },
    )
    if reason == "两根跌破多空线止损":
        state["未重新站上多空线前禁买"] = True
    if config.reentry_mode == "structure_recovery" and (_is_stop_like_exit(reason) or _is_trend_break_reason(reason)):
        state["结构恢复前禁买"] = True
    if _is_stop_like_exit(reason):
        state["连续止损次数"] += 1
        if config.stock_stop_cooldown_count > 0 and state["连续止损次数"] >= config.stock_stop_cooldown_count:
            if config.stock_stop_cooldown_days > 0:
                state["冷却到期日"] = current_date + pd.Timedelta(days=config.stock_stop_cooldown_days)
            if config.stock_stop_skip_next_buys > 0:
                state["待跳过买入次数"] = int(state.get("待跳过买入次数", 0)) + int(config.stock_stop_skip_next_buys)
            state["连续止损次数"] = 0
        elif config.stock_stop_cooldown_count > 0 and config.stock_stop_cooldown_days > 0:
            if state["连续止损次数"] >= config.stock_stop_cooldown_count:
                state["冷却到期日"] = current_date + pd.Timedelta(days=config.stock_stop_cooldown_days)
                state["连续止损次数"] = 0
    else:
        state["连续止损次数"] = 0


def _is_in_stock_cooldown(stock_risk_state: Dict[str, dict], code: str, current_date: pd.Timestamp) -> bool:
    state = stock_risk_state.get(code)
    if not state:
        return False
    cooldown_until = state.get("冷却到期日")
    if cooldown_until is None:
        return False
    return pd.Timestamp(current_date) <= pd.Timestamp(cooldown_until)


def _is_bull_bear_reentry_blocked(stock_risk_state: Dict[str, dict], code: str, row: pd.Series) -> bool:
    state = stock_risk_state.get(code)
    if not state or not state.get("未重新站上多空线前禁买", False):
        return False
    if float(row.get("close", 0.0)) > float(row.get("bull_bear_line", 0.0)):
        state["未重新站上多空线前禁买"] = False
        return False
    return True


def _is_structure_reentry_blocked(stock_risk_state: Dict[str, dict], code: str, row: pd.Series, config: PortfolioConfig) -> bool:
    if config.reentry_mode != "structure_recovery":
        return False
    state = stock_risk_state.get(code)
    if not state or not state.get("结构恢复前禁买", False):
        return False
    if _row_satisfies_structure_recovery(row, config):
        state["结构恢复前禁买"] = False
        return False
    return True


def _consume_stock_skip_if_needed(stock_risk_state: Dict[str, dict], code: str) -> bool:
    state = stock_risk_state.get(code)
    if not state:
        return False
    remaining = int(state.get("待跳过买入次数", 0))
    if remaining <= 0:
        return False
    state["待跳过买入次数"] = remaining - 1
    return True


def _round_trip_stats(round_trips: List[dict]) -> dict:
    if not round_trips:
        return {
            "平均持有期间收益率": 0.0,
            "盈利轮次占比": 0.0,
            "盈利轮次平均收益率": 0.0,
            "亏损轮次平均收益率": 0.0,
            "轮次收益率中位数": 0.0,
            "轮次收益率十分位": 0.0,
            "轮次收益率九十分位": 0.0,
        }

    frame = pd.DataFrame(round_trips)
    returns = pd.to_numeric(frame["完整持有收益率"], errors="coerce").dropna()
    if returns.empty:
        return {
            "平均持有期间收益率": 0.0,
            "盈利轮次占比": 0.0,
            "盈利轮次平均收益率": 0.0,
            "亏损轮次平均收益率": 0.0,
            "轮次收益率中位数": 0.0,
            "轮次收益率十分位": 0.0,
            "轮次收益率九十分位": 0.0,
        }

    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    return {
        "平均持有期间收益率": float(returns.mean()),
        "盈利轮次占比": float((returns > 0).mean()),
        "盈利轮次平均收益率": float(wins.mean()) if not wins.empty else 0.0,
        "亏损轮次平均收益率": float(losses.mean()) if not losses.empty else 0.0,
        "轮次收益率中位数": float(returns.median()),
        "轮次收益率十分位": float(returns.quantile(0.1)),
        "轮次收益率九十分位": float(returns.quantile(0.9)),
    }


def _can_open_position(row: pd.Series) -> bool:
    if bool(row.get("is_suspended", False)):
        return False
    if float(row.get("bull_bear_line", 0.0)) > float(row.get("trend_line", 0.0)):
        return False
    prev_close = float(row.get("prev_close", row["close"]))
    return not is_limit_up(float(row["open"]), prev_close, float(row.get("limit_pct", 0.10)))


def _can_close_position(row: pd.Series) -> bool:
    if bool(row.get("is_suspended", False)):
        return False
    prev_close = float(row.get("prev_close", row["close"]))
    return not is_limit_down(float(row["open"]), prev_close, float(row.get("limit_pct", 0.10)))


def _run_model(
    label: str,
    prepared: Dict[str, pd.DataFrame],
    all_dates: List[pd.Timestamp],
    add_weights: Dict[str, float],
    penalty_weights: Dict[str, float],
    config: PortfolioConfig,
) -> dict:
    signal_map = _build_signal_map(
        prepared,
        all_dates,
        add_weights,
        penalty_weights,
        config.top_quantile,
        config.buy_mode,
        config.min_score,
        config.use_trend_start_pool,
        config.trend_pool_mode,
        config.min_confirmation_hits,
        config.min_support_hits,
        config.rebuilt_min_confirmation_hits,
        config.rebuilt_min_support_hits,
        config.rebuilt_require_close_above_trend,
        config.rebuilt_require_close_above_bull_bear,
        config.sideways_mode,
        config.sideways_filter_threshold,
        config.sideways_score_penalty_scale,
    )
    cash = float(config.initial_capital)
    positions: Dict[str, dict] = {}
    trades: List[dict] = []
    round_trips: List[dict] = []
    stock_risk_state: Dict[str, dict] = {}
    equity_points: List[float] = []
    equity_index: List[pd.Timestamp] = []
    next_position_id = 1
    penalty_threshold = float(sum(penalty_weights.values()) / max(len([v for v in penalty_weights.values() if v > 0]), 1))
    初始止损系数 = float(_profile_value(config, "初始止损系数", 0.90))

    for current_date in tqdm(all_dates, desc=f"{label}净值回测", unit="日"):
        opened_today: set[str] = set()
        daily_replacements = 0

        for code in list(positions.keys()):
            pos = positions[code]
            df = prepared[code]
            if current_date not in df.index or code in opened_today:
                continue
            row = df.loc[current_date]
            if bool(row.get("is_suspended", False)):
                continue
            if pos.get("半仓止损待执行", False) and pd.Timestamp(current_date) > pd.Timestamp(pos.get("半仓止损触发日")):
                cash, sold_shares, _ = _sell_partial_position(
                    cash,
                    positions,
                    trades,
                    code,
                    current_date,
                    float(row["open"]),
                    f"{pos.get('半仓止损原因', '止损卖出')}_次日卖半仓",
                    0.5,
                    config,
                )
                if sold_shares > 0 and code in positions and positions[code]["shares"] > 0:
                    _activate_half_stop_watch(positions[code], current_date)
                    positions[code]["半仓止损次日卖半仓次数"] = int(positions[code].get("半仓止损次日卖半仓次数", 0)) + 1
                else:
                    positions[code]["半仓止损待执行"] = False
                continue
            if pos.get("半仓止损待次日清仓", False) and pd.Timestamp(current_date) > pd.Timestamp(pos.get("半仓止损清仓触发日")):
                cash = _close_position(cash, positions, trades, code, current_date, float(row["open"]), "半仓止损后破低清仓", config)
                _complete_round_trip(round_trips, pos, current_date, "半仓止损后破低清仓")
                pos["半仓破低清仓次数"] = int(pos.get("半仓破低清仓次数", 0)) + 1
                _update_stock_cooldown(stock_risk_state, code, current_date, "半仓止损后破低清仓", config)
                continue
            idx_now = df.index.get_loc(current_date)
            idx_entry = df.index.get_loc(pos["entry_date"])
            hold_bars = idx_now - idx_entry + 1
            if idx_now > 0 and float(row["close"]) < float(df.iloc[idx_now - 1]["low"]):
                pos["连续弱收盘天数"] = int(pos.get("连续弱收盘天数", 0)) + 1
            else:
                pos["连续弱收盘天数"] = 0

            cash, closed = _apply_trend_risk_controls(
                cash,
                positions,
                trades,
                round_trips,
                stock_risk_state,
                code,
                current_date,
                row,
                pos,
                hold_bars,
                config,
            )
            if closed:
                _update_stock_cooldown(stock_risk_state, code, current_date, "启动失败卖出", config)
                continue

            if float(row.get("double_break_bull_bear_penalty", 0.0)) >= 0.5:
                cash, closed = _handle_exit_or_half_stop(
                    cash,
                    positions,
                    trades,
                    round_trips,
                    stock_risk_state,
                    code,
                    current_date,
                    float(row["close"]),
                    "两根跌破多空线止损",
                    config,
                    trigger_low=float(row["low"]),
                )
                if closed:
                    pass
                continue

            if label == "fixed_take_profit":
                止盈涨幅 = float(_profile_value(config, "止盈涨幅", 0.40))
                固定止盈模式 = str(_profile_value(config, "固定止盈模式", "全卖"))
                最长持有天数 = int(_profile_value(config, "最长持有天数", 30))
                if float(row["bull_bear_line"]) > float(row["trend_line"]):
                    if float(row["close"]) < float(row["bull_bear_line"]):
                        cash, _ = _handle_exit_or_half_stop(cash, positions, trades, round_trips, stock_risk_state, code, current_date, float(row["close"]), "多空线压制卖出", config)
                        continue
                    if float(row["close"]) < float(row["trend_line"]):
                        cash, _ = _handle_exit_or_half_stop(cash, positions, trades, round_trips, stock_risk_state, code, current_date, float(row["close"]), "趋势线破位卖出", config)
                        continue
                if float(row["low"]) <= pos["stop_price"]:
                    cash, _ = _handle_exit_or_half_stop(cash, positions, trades, round_trips, stock_risk_state, code, current_date, pos["stop_price"], "止损卖出", config, trigger_low=float(row["low"]))
                    continue
                if _should_exit_on_weak_close(pos):
                    cash, _ = _handle_exit_or_half_stop(cash, positions, trades, round_trips, stock_risk_state, code, current_date, float(row["close"]), "三天连续弱收盘卖出", config)
                    continue
                tp_price = pos["entry_price"] * (1.0 + 止盈涨幅)
                if 固定止盈模式 == "留仓趋势":
                    if not pos.get("固定止盈已执行", False) and float(row["high"]) >= tp_price:
                        sell_shares = int((pos["shares"] * 0.7) / config.min_lot) * config.min_lot
                        if sell_shares <= 0:
                            sell_shares = pos["shares"]
                        cash, exec_price_local = _sell_position(cash, sell_shares, tp_price, config)
                        gross = sell_shares * exec_price_local
                        fee = gross * config.commission_rate
                        tax = gross * config.stamp_duty_rate
                        pos["累计卖出到账"] += gross - fee - tax
                        trades.append({"position_id": pos["position_id"], "code": code, "entry_date": pos["entry_date"], "exit_date": current_date, "entry_price": pos["entry_price"], "exit_price": exec_price_local, "shares": sell_shares, "reason": "固定涨幅部分止盈"})
                        pos["shares"] -= sell_shares
                        pos["固定止盈已执行"] = True
                        pos["stop_price"] = max(pos["stop_price"], float(row["low"]) * 0.98, pos["entry_price"])
                        if pos["shares"] <= 0:
                            _complete_round_trip(round_trips, pos, current_date, "固定涨幅部分止盈完成")
                            del positions[code]
                        continue
                    if pos.get("固定止盈已执行", False):
                        if float(row["bull_bear_line"]) > float(row["trend_line"]) and float(row["close"]) < float(row["bull_bear_line"]):
                            cash, _ = _handle_exit_or_half_stop(cash, positions, trades, round_trips, stock_risk_state, code, current_date, float(row["close"]), "固定涨幅后多空线压制卖出", config)
                            continue
                    if float(row["close"]) < float(row["trend_line"]):
                        cash, _ = _handle_exit_or_half_stop(cash, positions, trades, round_trips, stock_risk_state, code, current_date, float(row["close"]), "固定涨幅后趋势退出", config)
                        continue
                else:
                    if float(row["high"]) >= tp_price:
                        cash = _close_position(cash, positions, trades, code, current_date, tp_price, "固定涨幅止盈", config)
                        _complete_round_trip(round_trips, pos, current_date, "固定涨幅止盈")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "固定涨幅止盈", config)
                        continue
                if hold_bars >= 最长持有天数:
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "持有到期卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "持有到期卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "持有到期卖出", config)
                    continue

            elif label == "fixed_days":
                固定持有天数 = int(_profile_value(config, "固定持有天数", 30))
                固定持有模式 = str(_profile_value(config, "固定持有模式", "到期全卖"))
                结构退出最长持有天数 = int(_profile_value(config, "结构退出最长持有天数", 60))
                强票延长后最长持有天数 = _extended_structure_exit_days(pos, row, config)
                if 固定持有模式 == "趋势主线重构":
                    if float(row["bull_bear_line"]) > float(row["trend_line"]) and float(row["close"]) < float(row["bull_bear_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势主线重构_多空线压制卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "趋势主线重构_多空线压制卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "趋势主线重构_多空线压制卖出", config)
                        continue
                    if float(row["low"]) <= pos["stop_price"]:
                        cash, _ = _handle_exit_or_half_stop(
                            cash,
                            positions,
                            trades,
                            round_trips,
                            stock_risk_state,
                            code,
                            current_date,
                            pos["stop_price"],
                            "止损卖出",
                            config,
                            trigger_low=float(row["low"]),
                        )
                        continue
                    if _should_exit_on_weak_close(pos):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势主线重构_三天连续弱收盘卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "趋势主线重构_三天连续弱收盘卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "趋势主线重构_三天连续弱收盘卖出", config)
                        continue
                    if bool(pos.get("利润保护已启用", False)) and float(row["close"]) < float(row["trend_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势主线重构_利润保护破位卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "趋势主线重构_利润保护破位卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "趋势主线重构_利润保护破位卖出", config)
                        continue
                    if hold_bars >= 固定持有天数 and float(row["close"]) < float(row["trend_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势主线重构_最短持有后趋势破位卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "趋势主线重构_最短持有后趋势破位卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "趋势主线重构_最短持有后趋势破位卖出", config)
                        continue
                    if hold_bars >= 强票延长后最长持有天数:
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势主线重构_最长持有卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "趋势主线重构_最长持有卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "趋势主线重构_最长持有卖出", config)
                        continue
                elif 固定持有模式 == "到期后结构退出" and hold_bars >= 固定持有天数:
                    if float(row["bull_bear_line"]) > float(row["trend_line"]) and float(row["close"]) < float(row["bull_bear_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "固定天数后多空线压制卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "固定天数后多空线压制卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "固定天数后多空线压制卖出", config)
                        continue
                    if float(row["close"]) < float(row["trend_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "固定天数后趋势线破位卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "固定天数后趋势线破位卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "固定天数后趋势线破位卖出", config)
                        continue
                else:
                    if float(row["bull_bear_line"]) > float(row["trend_line"]):
                        if float(row["close"]) < float(row["bull_bear_line"]):
                            cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "多空线压制卖出", config)
                            _complete_round_trip(round_trips, pos, current_date, "多空线压制卖出")
                            _update_stock_cooldown(stock_risk_state, code, current_date, "多空线压制卖出", config)
                            continue
                        if float(row["close"]) < float(row["trend_line"]):
                            cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势线破位卖出", config)
                            _complete_round_trip(round_trips, pos, current_date, "趋势线破位卖出")
                            _update_stock_cooldown(stock_risk_state, code, current_date, "趋势线破位卖出", config)
                            continue
                if float(row["low"]) <= pos["stop_price"]:
                    cash, _ = _handle_exit_or_half_stop(
                        cash,
                        positions,
                        trades,
                        round_trips,
                        stock_risk_state,
                        code,
                        current_date,
                        pos["stop_price"],
                        "止损卖出",
                        config,
                        trigger_low=float(row["low"]),
                    )
                    continue
                if _should_exit_on_weak_close(pos):
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "三天连续弱收盘卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "三天连续弱收盘卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "三天连续弱收盘卖出", config)
                    continue
                if 固定持有模式 == "到期全卖" and hold_bars >= 固定持有天数:
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "固定天数卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "固定天数卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "固定天数卖出", config)
                    continue
                if 固定持有模式 == "到期后结构退出" and hold_bars >= 强票延长后最长持有天数:
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "固定天数后最长持有卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "固定天数后最长持有卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "固定天数后最长持有卖出", config)
                    continue

            elif label == "tiered":
                分批止盈模式 = str(_profile_value(config, "分批止盈模式", "五段"))
                分批最长持有天数 = int(_profile_value(config, "最长持有天数", 60))
                if float(row["bull_bear_line"]) > float(row["trend_line"]):
                    if float(row["close"]) < float(row["bull_bear_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "多空线压制卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "多空线压制卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "多空线压制卖出", config)
                        continue
                    if float(row["close"]) < float(row["trend_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势线破位卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "趋势线破位卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "趋势线破位卖出", config)
                        continue
                if float(row["low"]) <= pos["stop_price"]:
                    cash, _ = _handle_exit_or_half_stop(
                        cash,
                        positions,
                        trades,
                        round_trips,
                        stock_risk_state,
                        code,
                        current_date,
                        pos["stop_price"],
                        "止损卖出",
                        config,
                        trigger_low=float(row["low"]),
                    )
                    continue
                if _should_exit_on_weak_close(pos):
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "三天连续弱收盘卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "三天连续弱收盘卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "三天连续弱收盘卖出", config)
                    continue

                base_chunk = int((pos["initial_shares"] * 0.2) / config.min_lot) * config.min_lot
                prev_low = float(df.iloc[idx_now - 1]["low"]) if idx_now > 0 else float(row["low"])
                trend_line = float(row["trend_line"])
                j_value = float(row["J"])

                def sell_chunk(raw_price: float, reason: str, sell_all: bool = False) -> bool:
                    nonlocal cash
                    shares_to_sell = pos["shares"] if sell_all else min(pos["shares"], max(base_chunk, config.min_lot))
                    if shares_to_sell <= 0:
                        return False
                    cash, exec_price_local = _sell_position(cash, shares_to_sell, raw_price, config)
                    gross = shares_to_sell * exec_price_local
                    fee = gross * config.commission_rate
                    tax = gross * config.stamp_duty_rate
                    pos["累计卖出到账"] += gross - fee - tax
                    trades.append({"position_id": pos["position_id"], "code": code, "entry_date": pos["entry_date"], "exit_date": current_date, "entry_price": pos["entry_price"], "exit_price": exec_price_local, "shares": shares_to_sell, "reason": reason})
                    pos["shares"] -= shares_to_sell
                    return True

                if 分批止盈模式 == "轻量":
                    if not pos["step1"] and (j_value > 100 or float(row["high"]) >= pos["entry_price"] * 1.08):
                        if sell_chunk(max(float(row["close"]), pos["entry_price"] * 1.08), "第一段止盈"):
                            pos["step1"] = True
                            pos["stop_price"] = max(pos["stop_price"], float(row["low"]) * 0.98, pos["entry_price"])
                    if code not in positions or pos["shares"] <= 0:
                        if code in positions and pos["shares"] <= 0:
                            del positions[code]
                        continue
                    if pos["step1"] and not pos["step2"] and float(row["high"]) >= pos["entry_price"] * 1.15:
                        if sell_chunk(pos["entry_price"] * 1.15, "第二段止盈"):
                            pos["step2"] = True
                            pos["stop_price"] = max(pos["stop_price"], float(row["low"]) * 0.98, trend_line * 0.99, pos["entry_price"])
                    if code not in positions or pos["shares"] <= 0:
                        if code in positions and pos["shares"] <= 0:
                            del positions[code]
                        continue
                    if pos["step2"] and float(row["close"]) < float(row["trend_line"]):
                        sell_chunk(float(row["close"]), "轻量分批趋势退出", sell_all=True)
                        _complete_round_trip(round_trips, pos, current_date, "轻量分批趋势退出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "轻量分批趋势退出", config)
                        del positions[code]
                        continue
                else:
                    if not pos["step1"] and j_value > 100:
                        if sell_chunk(float(row["close"]), "第一段止盈"):
                            pos["step1"] = True
                            pos["stop_price"] = max(pos["stop_price"], float(row["low"]) * 0.95)

                    if code not in positions or pos["shares"] <= 0:
                        if code in positions and pos["shares"] <= 0:
                            del positions[code]
                        continue

                    if not pos["step2"] and float(row["high"]) >= pos["entry_price"] * 1.08:
                        if sell_chunk(pos["entry_price"] * 1.08, "第二段止盈"):
                            pos["step2"] = True
                            pos["stop_price"] = max(pos["stop_price"], float(row["low"]) * 0.95)

                    if code not in positions or pos["shares"] <= 0:
                        if code in positions and pos["shares"] <= 0:
                            del positions[code]
                        continue

                    if pos["step1"] and pos["step2"] and not pos["step3"] and float(row["high"]) > trend_line * 1.15:
                        if sell_chunk(float(row["close"]), "第三段止盈"):
                            pos["step3"] = True
                            pos["stop_price"] = max(pos["stop_price"], float(row["low"]) * 0.95)
                    elif pos["step1"] and pos["step2"] and pos["step3"] and not pos["step4"] and float(row["high"]) > trend_line * 1.20:
                        if sell_chunk(float(row["close"]), "第四段止盈"):
                            pos["step4"] = True
                            pos["stop_price"] = max(pos["stop_price"], float(row["low"]) * 0.95)
                    elif pos["step1"] and pos["step2"] and pos["step4"] and not pos["step5"] and float(row["high"]) > trend_line * 1.25:
                        if sell_chunk(float(row["close"]), "第五段止盈", sell_all=True):
                            pos["step5"] = True

                if pos["shares"] <= 0:
                    _complete_round_trip(round_trips, pos, current_date, "分批止盈完成")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "分批止盈完成", config)
                    del positions[code]
                    continue

                if pos["step2"]:
                    pos["stop_price"] = max(pos["stop_price"], trend_line * 0.98)
                    pos["dd_count"] = pos["dd_count"] + 1 if float(row["close"]) < prev_low else 0
                    if pos["dd_count"] >= 3:
                        sell_chunk(float(row["close"]), "滴滴止损", sell_all=True)
                        _complete_round_trip(round_trips, pos, current_date, "滴滴止损")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "滴滴止损", config)
                        del positions[code]
                        continue

                if hold_bars >= 分批最长持有天数:
                    sell_chunk(float(row["close"]), f"持有{分批最长持有天数}天卖出", sell_all=True)
                    _complete_round_trip(round_trips, pos, current_date, f"持有{分批最长持有天数}天卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, f"持有{分批最长持有天数}天卖出", config)
                    del positions[code]
                    continue

            elif label == "j100_full_exit":
                if float(row["bull_bear_line"]) > float(row["trend_line"]):
                    if float(row["close"]) < float(row["bull_bear_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "多空线压制卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "多空线压制卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "多空线压制卖出", config)
                        continue
                    if float(row["close"]) < float(row["trend_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势线破位卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "趋势线破位卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "趋势线破位卖出", config)
                        continue
                if float(row["low"]) <= pos["stop_price"]:
                    cash, _ = _handle_exit_or_half_stop(
                        cash,
                        positions,
                        trades,
                        round_trips,
                        stock_risk_state,
                        code,
                        current_date,
                        pos["stop_price"],
                        "止损卖出",
                        config,
                        trigger_low=float(row["low"]),
                    )
                    continue
                if _should_exit_on_weak_close(pos):
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "三天连续弱收盘卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "三天连续弱收盘卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "三天连续弱收盘卖出", config)
                    continue
                if float(row["J"]) > 100:
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "J>100全仓卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "J>100全仓卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "J>100全仓卖出", config)
                    continue
                if hold_bars >= 60:
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "持有60天卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "持有60天卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "持有60天卖出", config)
                    continue

            elif label == "penalty_exit":
                if float(row["bull_bear_line"]) > float(row["trend_line"]):
                    if float(row["close"]) < float(row["bull_bear_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "多空线压制卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "多空线压制卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "多空线压制卖出", config)
                        continue
                    if float(row["close"]) < float(row["trend_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势线破位卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "趋势线破位卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "趋势线破位卖出", config)
                        continue
                if float(row["low"]) <= pos["stop_price"]:
                    cash, _ = _handle_exit_or_half_stop(
                        cash,
                        positions,
                        trades,
                        round_trips,
                        stock_risk_state,
                        code,
                        current_date,
                        pos["stop_price"],
                        "止损卖出",
                        config,
                        trigger_low=float(row["low"]),
                    )
                    continue
                if _should_exit_on_weak_close(pos):
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "三天连续弱收盘卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "三天连续弱收盘卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "三天连续弱收盘卖出", config)
                    continue
                current_penalty = sum(float(row.get(col, 0.0)) * weight for col, weight in penalty_weights.items())
                if current_penalty >= penalty_threshold:
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "扣分阈值卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "扣分阈值卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "扣分阈值卖出", config)
                    continue
                if hold_bars >= 60:
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "持有60天卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, "持有60天卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, "持有60天卖出", config)
                    continue

            elif label == "high_volume_exit":
                最长持有天数 = int(_profile_value(config, "最长持有天数", 60))
                if float(row["bull_bear_line"]) > float(row["trend_line"]):
                    if float(row["close"]) < float(row["bull_bear_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "多空线压制卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "多空线压制卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "多空线压制卖出", config)
                        continue
                    if float(row["close"]) < float(row["trend_line"]):
                        cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), "趋势线破位卖出", config)
                        _complete_round_trip(round_trips, pos, current_date, "趋势线破位卖出")
                        _update_stock_cooldown(stock_risk_state, code, current_date, "趋势线破位卖出", config)
                        continue
                if float(row["low"]) <= pos["stop_price"]:
                    cash, _ = _handle_exit_or_half_stop(
                        cash,
                        positions,
                        trades,
                        round_trips,
                        stock_risk_state,
                        code,
                        current_date,
                        pos["stop_price"],
                        "止损卖出",
                        config,
                        trigger_low=float(row["low"]),
                    )
                    continue
                if _should_exit_high_volume(label, pos, row, config):
                    mode = str(_profile_value(config, "高位定义", "trend_bias"))
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), f"高位放量卖出_{mode}", config)
                    _complete_round_trip(round_trips, pos, current_date, f"高位放量卖出_{mode}")
                    _update_stock_cooldown(stock_risk_state, code, current_date, f"高位放量卖出_{mode}", config)
                    continue
                if hold_bars >= 最长持有天数:
                    cash = _close_position(cash, positions, trades, code, current_date, float(row["close"]), f"持有{最长持有天数}天卖出", config)
                    _complete_round_trip(round_trips, pos, current_date, f"持有{最长持有天数}天卖出")
                    _update_stock_cooldown(stock_risk_state, code, current_date, f"持有{最长持有天数}天卖出", config)
                    continue

            if code not in positions:
                continue
            if positions[code].get("半仓止损待执行", False) or positions[code].get("半仓止损观察中", False) or positions[code].get("半仓止损待次日清仓", False):
                continue
            if pos.get("买入批次", 1) < 3 and _can_add_tranche(row):
                if _is_bull_bear_reentry_blocked(stock_risk_state, code, row):
                    continue
                if _is_structure_reentry_blocked(stock_risk_state, code, row, config):
                    continue
                if config.buy_mode == "staged":
                    desired_batch = pos["买入批次"] + 1
                else:
                    desired_batch = _desired_buy_batch(row, config.buy_mode)
                if desired_batch > pos["买入批次"]:
                    add_price = float(row["open"]) * (1.0 + config.slippage_rate)
                    add_target_value = pos["计划仓位金额"] * (_batch_target_ratio(desired_batch) - _batch_target_ratio(pos["买入批次"]))
                    add_shares = _buy_shares_for_value(cash, add_price, add_target_value, config)
                    if add_shares > 0:
                        gross = add_shares * add_price
                        fee = gross * config.commission_rate
                        cash -= gross + fee
                        total_shares = pos["shares"] + add_shares
                        pos["entry_price"] = (pos["entry_price"] * pos["shares"] + add_price * add_shares) / total_shares
                        pos["shares"] = total_shares
                        pos["initial_shares"] += add_shares
                        pos["买入批次"] = desired_batch
                        pos["累计买入成本"] += gross + fee

        candidates = signal_map.get(current_date, [])
        for signal in candidates:
            if _is_in_stock_cooldown(stock_risk_state, signal["code"], current_date):
                continue
            if _consume_stock_skip_if_needed(stock_risk_state, signal["code"]):
                continue
            if signal["code"] in positions:
                continue
            df = prepared[signal["code"]]
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            if _is_bull_bear_reentry_blocked(stock_risk_state, signal["code"], row):
                continue
            if _is_structure_reentry_blocked(stock_risk_state, signal["code"], row, config):
                continue
            if not _can_open_position(row):
                continue

            available_slots = max(config.max_positions - len(positions), 0)
            if available_slots <= 0 and positions and daily_replacements < config.max_daily_replacements:
                replace_candidates = []
                for replace_code, replace_pos in positions.items():
                    replace_df = prepared[replace_code]
                    if current_date not in replace_df.index:
                        continue
                    replace_row = replace_df.loc[current_date]
                    if not _can_close_position(replace_row):
                        continue
                    idx_now = replace_df.index.get_loc(current_date)
                    idx_entry = replace_df.index.get_loc(replace_pos["entry_date"])
                    hold_bars = idx_now - idx_entry + 1
                    if not _position_replace_eligible(label, replace_pos, replace_df, idx_now, idx_entry, hold_bars):
                        continue
                    replace_score = replace_pos.get("prior_close_score", replace_pos["score"])
                    replace_candidates.append((replace_score, replace_code, replace_row))

                if replace_candidates:
                    replace_score, replace_code, replace_row = min(replace_candidates, key=lambda item: (item[0], item[1]))
                    if signal["score"] > replace_score + config.replacement_threshold:
                        cash = _close_position(cash, positions, trades, replace_code, current_date, float(replace_row["open"]), "换仓卖出", config)
                        _complete_round_trip(round_trips, replace_pos, current_date, "换仓卖出")
                        available_slots = 1
                        daily_replacements += 1

            if available_slots <= 0:
                continue

            entry_price = float(row["open"]) * (1.0 + config.slippage_rate)
            planned_value = cash / available_slots if available_slots > 0 else 0.0
            if config.buy_mode == "staged":
                desired_batch = 1
                target_ratio = 0.3
            else:
                desired_batch = _desired_buy_batch(row, config.buy_mode)
                target_ratio = _batch_target_ratio(desired_batch)
                if target_ratio <= 0:
                    continue
            shares = _buy_shares_for_value(cash, entry_price, planned_value * target_ratio, config)
            if shares <= 0:
                continue
            gross = shares * entry_price
            fee = gross * config.commission_rate
            cash -= gross + fee
            positions[signal["code"]] = {
                "position_id": next_position_id,
                "code": signal["code"],
                "entry_date": current_date,
                "entry_price": entry_price,
                "shares": shares,
                "initial_shares": shares,
                "stop_price": float(row["low"]) * 初始止损系数,
                "score": signal["score"],
                "prior_close_score": signal["score"],
                "计划仓位金额": planned_value,
                "买入批次": desired_batch,
                "最短换仓持有天数": config.min_hold_days_for_replace,
                "step1": False,
                "step2": False,
                "step3": False,
                "step4": False,
                "step5": False,
                "dd_count": 0,
                "累计买入成本": gross + fee,
                "累计卖出到账": 0.0,
                "最高价": float(row["high"]),
                "连续弱收盘天数": 0,
                "启用三天连续弱收盘退出": bool(_profile_value(config, "启用三天连续弱收盘退出", False)),
                "利润保护已启用": False,
                "半仓止损待执行": False,
                "半仓止损观察中": False,
                "半仓止损待次日清仓": False,
                "半仓止损触发日": None,
                "半仓止损卖半仓日": None,
                "半仓止损清仓触发日": None,
                "半仓止损原因": "",
                "半仓止损参考低点": None,
            }
            next_position_id += 1
            opened_today.add(signal["code"])

        for code, pos in positions.items():
            df = prepared[code]
            if current_date in df.index:
                base_score = _position_current_score(
                    code,
                    current_date,
                    prepared,
                    add_weights,
                    penalty_weights,
                    pos.get("prior_close_score", pos["score"]),
                    config.sideways_mode,
                    config.sideways_score_penalty_scale,
                )
                pos["prior_close_score"] = base_score - _position_dynamic_penalty(pos, df.loc[current_date])

        equity = cash
        for code, pos in positions.items():
            df = prepared[code]
            if current_date in df.index:
                equity += pos["shares"] * float(df.loc[current_date, "close"])
            else:
                equity += pos["shares"] * pos["entry_price"]
        equity_index.append(current_date)
        equity_points.append(equity)

    equity_curve = pd.Series(equity_points, index=pd.DatetimeIndex(equity_index), dtype=float)
    metrics = compute_metrics(equity_curve)
    round_trip_stats = _round_trip_stats(round_trips)
    payload = {
        "label": label,
        "metrics": metrics,
        "trade_count": len(trades),
        "完整轮次交易数": len(round_trips),
        **round_trip_stats,
        "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else config.initial_capital,
        "equity_curve": equity_curve,
        "trades": trades,
        "round_trips": round_trips,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weighted-factor portfolio backtests under capital constraints")
    parser.add_argument("data_dir")
    parser.add_argument("--refined-root", default="results/multi_factor_research_v4_refined")
    parser.add_argument("--output-root", default="results/weighted_portfolio_backtest")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--replacement-threshold", type=float, default=0.03)
    parser.add_argument("--min-hold-days-for-replace", type=int, default=5)
    parser.add_argument("--max-daily-replacements", type=int, default=1)
    parser.add_argument("--buy-mode", choices=["staged", "full", "strict_full"], default="staged")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--labels", default="")
    parser.add_argument("--scorecard-root", default="")
    parser.add_argument("--stock-stop-cooldown-count", type=int, default=0)
    parser.add_argument("--stock-stop-cooldown-days", type=int, default=0)
    parser.add_argument("--stock-stop-skip-next-buys", type=int, default=0)
    parser.add_argument("--sideways-mode", choices=["关闭", "只过滤", "只降分", "过滤+降分"], default="关闭")
    parser.add_argument("--sideways-filter-threshold", type=float, default=0.55)
    parser.add_argument("--sideways-score-penalty-scale", type=float, default=0.0)
    parser.add_argument("--use-trend-start-pool", action="store_true")
    parser.add_argument("--trend-pool-mode", choices=["start_v1", "rebuilt_v1"], default="start_v1")
    parser.add_argument("--min-confirmation-hits", type=int, default=1)
    parser.add_argument("--min-support-hits", type=int, default=1)
    parser.add_argument("--rebuilt-min-confirmation-hits", type=int, default=1)
    parser.add_argument("--rebuilt-min-support-hits", type=int, default=2)
    parser.add_argument("--rebuilt-require-close-above-trend", action="store_true")
    parser.add_argument("--rebuilt-require-close-above-bull-bear", action="store_true")
    parser.add_argument("--reentry-mode", choices=["bull_bear_only", "structure_recovery"], default="bull_bear_only")
    parser.add_argument("--recovery-min-confirmation-hits", type=int, default=1)
    parser.add_argument("--recovery-min-support-hits", type=int, default=2)
    parser.add_argument("--recovery-require-close-above-trend", action="store_true")
    parser.add_argument("--recovery-require-close-above-bull-bear", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stock_data, all_dates = load_price_directory(args.data_dir)
    prepared = _prepare_stock_frames(stock_data)
    config = PortfolioConfig(
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        min_score=args.min_score,
        replacement_threshold=args.replacement_threshold,
        min_hold_days_for_replace=args.min_hold_days_for_replace,
        max_daily_replacements=args.max_daily_replacements,
        buy_mode=args.buy_mode,
        stock_stop_cooldown_count=args.stock_stop_cooldown_count,
        stock_stop_cooldown_days=args.stock_stop_cooldown_days,
        stock_stop_skip_next_buys=args.stock_stop_skip_next_buys,
        use_trend_start_pool=args.use_trend_start_pool,
        trend_pool_mode=args.trend_pool_mode,
        min_confirmation_hits=args.min_confirmation_hits,
        min_support_hits=args.min_support_hits,
        rebuilt_min_confirmation_hits=args.rebuilt_min_confirmation_hits,
        rebuilt_min_support_hits=args.rebuilt_min_support_hits,
        rebuilt_require_close_above_trend=args.rebuilt_require_close_above_trend,
        rebuilt_require_close_above_bull_bear=args.rebuilt_require_close_above_bull_bear,
        reentry_mode=args.reentry_mode,
        recovery_min_confirmation_hits=args.recovery_min_confirmation_hits,
        recovery_min_support_hits=args.recovery_min_support_hits,
        recovery_require_close_above_trend=args.recovery_require_close_above_trend,
        recovery_require_close_above_bull_bear=args.recovery_require_close_above_bull_bear,
        sideways_mode=args.sideways_mode,
        sideways_filter_threshold=args.sideways_filter_threshold,
        sideways_score_penalty_scale=args.sideways_score_penalty_scale,
    )

    if args.scorecard_root.strip():
        add_weights, penalty_weights = _load_formal_scorecard(Path(args.scorecard_root))
        models = {
            "fixed_take_profit": {"add": add_weights, "penalty": penalty_weights},
            "fixed_days": {"add": add_weights, "penalty": penalty_weights},
            "tiered": {"add": add_weights, "penalty": penalty_weights},
            "high_volume_exit": {"add": add_weights, "penalty": penalty_weights},
            "penalty_exit": {"add": add_weights, "penalty": penalty_weights},
            "j100_full_exit": {"add": add_weights, "penalty": penalty_weights},
        }
    else:
        weights_payload = _load_refined_weights(Path(args.refined_root))
        models = {
            "fixed_take_profit": {
                "add": _parse_weight_spec(weights_payload["fixed_take_profit"]["combo"]),
                "penalty": {k: PENALTY_WEIGHTS.get(k, 0.0) for k in PENALTY_COLUMNS},
            },
            "fixed_days": {
                "add": _parse_weight_spec(weights_payload["fixed_days"]["combo"]),
                "penalty": {k: PENALTY_WEIGHTS.get(k, 0.0) for k in PENALTY_COLUMNS},
            },
            "tiered": {
                "add": _parse_weight_spec(weights_payload["tiered"]["combo"]),
                "penalty": {k: PENALTY_WEIGHTS.get(k, 0.0) for k in PENALTY_COLUMNS},
            },
            "high_volume_exit": {
                "add": _parse_weight_spec(weights_payload["fixed_days"]["combo"]),
                "penalty": {k: PENALTY_WEIGHTS.get(k, 0.0) for k in PENALTY_COLUMNS},
            },
            "penalty_exit": {
                "add": _parse_weight_spec(weights_payload["tiered"]["combo"]),
                "penalty": {k: PENALTY_WEIGHTS.get(k, 0.0) for k in PENALTY_COLUMNS},
            },
            "j100_full_exit": {
                "add": _parse_weight_spec(weights_payload["j100_full_exit"]["combo"]),
                "penalty": {k: PENALTY_WEIGHTS.get(k, 0.0) for k in PENALTY_COLUMNS},
            },
        }
    if args.labels.strip():
        selected = {label.strip() for label in args.labels.split(",") if label.strip()}
        models = {label: weights for label, weights in models.items() if label in selected}

    summary: dict[str, dict] = {}
    for label, scorecard in tqdm(models.items(), desc="回测策略", unit="个"):
        result = _run_model(label, prepared, all_dates, scorecard["add"], scorecard["penalty"], config)
        summary[label] = {
            "加分权重": scorecard["add"],
            "扣分权重": scorecard["penalty"],
            "metrics": result["metrics"],
            "trade_count": result["trade_count"],
            "完整轮次交易数": result["完整轮次交易数"],
            "平均持有期间收益率": result["平均持有期间收益率"],
            "盈利轮次占比": result["盈利轮次占比"],
            "盈利轮次平均收益率": result["盈利轮次平均收益率"],
            "亏损轮次平均收益率": result["亏损轮次平均收益率"],
            "轮次收益率中位数": result["轮次收益率中位数"],
            "轮次收益率十分位": result["轮次收益率十分位"],
            "轮次收益率九十分位": result["轮次收益率九十分位"],
            "final_equity": result["final_equity"],
            "换仓分差阈值": config.replacement_threshold,
            "最低买入净分": config.min_score,
            "最短换仓持有天数": config.min_hold_days_for_replace,
            "每日最大换仓数": config.max_daily_replacements,
            "个股连续止损冷却次数": config.stock_stop_cooldown_count,
            "个股冷却天数": config.stock_stop_cooldown_days,
            "个股连续止损跳过下次买入次数": config.stock_stop_skip_next_buys,
            "趋势候选池启用": config.use_trend_start_pool,
            "趋势候选池模式": config.trend_pool_mode,
            "启动确认最少命中数": config.min_confirmation_hits,
            "支撑最少命中数": config.min_support_hits,
            "重构确认最少命中数": config.rebuilt_min_confirmation_hits,
            "重构支撑最少命中数": config.rebuilt_min_support_hits,
            "重构要求收盘站上趋势线": config.rebuilt_require_close_above_trend,
            "重构要求收盘站上多空线": config.rebuilt_require_close_above_bull_bear,
            "再入场模式": config.reentry_mode,
            "结构恢复确认最少命中数": config.recovery_min_confirmation_hits,
            "结构恢复支撑最少命中数": config.recovery_min_support_hits,
            "结构恢复要求收盘站上趋势线": config.recovery_require_close_above_trend,
            "结构恢复要求收盘站上多空线": config.recovery_require_close_above_bull_bear,
            "横盘处理模式": config.sideways_mode,
            "横盘过滤阈值": config.sideways_filter_threshold,
            "横盘降分系数": config.sideways_score_penalty_scale,
            "买入方式": (
                "J<-5一次性买入"
                if config.buy_mode == "strict_full"
                else ("一次性买入" if config.buy_mode == "full" else "分批买入")
            ),
        }
        run_dir = output_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary[label], ensure_ascii=False, indent=2), encoding="utf-8")
        result["equity_curve"].rename("equity").to_csv(run_dir / "equity_curve.csv", encoding="utf-8")
        with (run_dir / "trades.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["position_id", "code", "entry_date", "exit_date", "entry_price", "exit_price", "shares", "reason"])
            writer.writeheader()
            for trade in result["trades"]:
                writer.writerow(
                    {
                        "position_id": trade.get("position_id"),
                        "code": trade["code"],
                        "entry_date": pd.Timestamp(trade["entry_date"]).strftime("%Y-%m-%d"),
                        "exit_date": pd.Timestamp(trade["exit_date"]).strftime("%Y-%m-%d"),
                        "entry_price": trade["entry_price"],
                        "exit_price": trade["exit_price"],
                        "shares": trade["shares"],
                        "reason": trade["reason"],
                    }
                )
        with (run_dir / "完整轮次交易.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["持仓编号", "股票代码", "首次买入日期", "最终卖出日期", "完整持有收益率", "持有天数", "最终卖出原因"])
            writer.writeheader()
            for item in result["round_trips"]:
                writer.writerow(
                    {
                        "持仓编号": item["持仓编号"],
                        "股票代码": item["股票代码"],
                        "首次买入日期": pd.Timestamp(item["首次买入日期"]).strftime("%Y-%m-%d"),
                        "最终卖出日期": pd.Timestamp(item["最终卖出日期"]).strftime("%Y-%m-%d"),
                        "完整持有收益率": item["完整持有收益率"],
                        "持有天数": item["持有天数"],
                        "最终卖出原因": item["最终卖出原因"],
                    }
                )

    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
