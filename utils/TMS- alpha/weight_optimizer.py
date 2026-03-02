from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from backtester import backtest_topk
from config import RunConfig
from metrics import objective_value


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(v, 0.0) for v in w.values())
    if s == 0:
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: max(v, 0.0) / s for k, v in w.items()}


def _eval_weights(
    df: pd.DataFrame,
    cfg: RunConfig,
    weights: Dict[str, float],
    threshold_params: Dict[str, float],
) -> float:
    bt = backtest_topk(df, cfg, factor_weights=weights, threshold_params=threshold_params)
    return objective_value(bt.metrics, cfg.objective_name, cfg.objective_weights)


def grid_weight_search(
    df: pd.DataFrame,
    cfg: RunConfig,
    factors: List[str],
    threshold_params: Dict[str, float],
    step: float = 0.1,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if len(factors) > 5:
        raise ValueError("Grid search is expensive when factor count > 5; use random search.")
    values = np.arange(0.0, 1.0 + 1e-9, step)
    rows = []
    best_w = None
    best_obj = -1e18

    def dfs(idx: int, remain: float, cur: List[float]) -> None:
        nonlocal best_w, best_obj
        if idx == len(factors) - 1:
            w_last = round(remain, 10)
            if w_last < 0:
                return
            full = cur + [w_last]
            w = {f: v for f, v in zip(factors, full)}
            w = _normalize_weights(w)
            obj = _eval_weights(df, cfg, w, threshold_params)
            rows.append({"weights": w, "objective": obj})
            if obj > best_obj:
                best_obj = obj
                best_w = w
            return
        for v in values:
            if v > remain:
                break
            dfs(idx + 1, remain - v, cur + [float(v)])

    dfs(0, 1.0, [])
    res = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return best_w or {f: 1.0 / len(factors) for f in factors}, res


def random_weight_search(
    df: pd.DataFrame,
    cfg: RunConfig,
    factors: List[str],
    threshold_params: Dict[str, float],
    n_samples: int = 5000,
    seed: int = 42,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    best_w = None
    best_obj = -1e18
    for _ in range(n_samples):
        vec = rng.random(len(factors))
        vec = vec / vec.sum()
        w = {f: float(v) for f, v in zip(factors, vec)}
        obj = _eval_weights(df, cfg, w, threshold_params)
        rows.append({"weights": w, "objective": obj})
        if obj > best_obj:
            best_obj = obj
            best_w = w
    res = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return best_w or {f: 1.0 / len(factors) for f in factors}, res


def bayes_like_weight_search(
    df: pd.DataFrame,
    cfg: RunConfig,
    factors: List[str],
    threshold_params: Dict[str, float],
    init_samples: int = 500,
    local_rounds: int = 300,
    seed: int = 42,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    # Lightweight Bayesian-like search: random init + local perturbation around incumbent.
    rng = np.random.default_rng(seed)
    best_w, hist = random_weight_search(
        df, cfg, factors, threshold_params, n_samples=init_samples, seed=seed
    )
    rows = hist.to_dict("records")
    best_obj = float(hist.iloc[0]["objective"]) if not hist.empty else -1e18

    vec = np.array([best_w[f] for f in factors], dtype=float)
    temp = 0.2
    for _ in range(local_rounds):
        noise = rng.normal(0, temp, size=len(factors))
        cand = np.clip(vec + noise, 1e-8, None)
        cand = cand / cand.sum()
        w = {f: float(v) for f, v in zip(factors, cand)}
        obj = _eval_weights(df, cfg, w, threshold_params)
        rows.append({"weights": w, "objective": obj})
        if obj > best_obj:
            best_obj = obj
            best_w = w
            vec = cand
        temp *= 0.995
        temp = max(temp, 0.02)

    res = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return best_w, res

