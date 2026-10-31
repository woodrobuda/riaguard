"""Performance metrics and overfitting statistics for simulated return series.

References: Bailey & Lopez de Prado (2012) "The Sharpe ratio efficient frontier" (PSR);
Bailey & Lopez de Prado (2014) "The deflated Sharpe ratio" (DSR); Bailey, Borwein,
Lopez de Prado & Zhu (2015) "The probability of backtest overfitting" (PBO via CSCV).
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats as _st

EULER_GAMMA = 0.5772156649015329


def _r(x) -> np.ndarray:
    a = np.asarray(x, float); return a[~np.isnan(a)]


def sharpe(returns, periods_per_year=252, rf=0.0) -> float:
    r = _r(returns) - rf / periods_per_year
    s = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / s * np.sqrt(periods_per_year)) if s > 0 else float("nan")


def sortino(returns, periods_per_year=252, mar=0.0) -> float:
    r = _r(returns) - mar
    dd = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if len(r) else 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year)) if dd > 0 else float("nan")


def equity_curve(returns) -> np.ndarray:
    return np.cumprod(1.0 + np.nan_to_num(np.asarray(returns, float)))


def cagr(returns, periods_per_year=252) -> float:
    r = _r(returns)
    if not len(r): return float("nan")
    return float(np.prod(1 + r) ** (periods_per_year / len(r)) - 1)


def drawdown(returns) -> np.ndarray:
    eq = np.r_[1.0, equity_curve(returns)]
    return (eq / np.maximum.accumulate(eq) - 1.0)[1:]


def max_drawdown(returns) -> float:
    d = drawdown(returns); return float(d.min()) if len(d) else 0.0


def max_drawdown_duration(returns) -> int:
    """Longest run of bars spent below a previous equity high (peak to recovery)."""
    best = cur = 0
    for v in drawdown(returns):
        cur = cur + 1 if v < 0 else 0; best = max(best, cur)
    return int(best)


def calmar(returns, periods_per_year=252) -> float:
    mdd = max_drawdown(returns)
    return float(cagr(returns, periods_per_year) / abs(mdd)) if mdd < 0 else float("nan")


def hit_rate(returns, position=None) -> float:
    """Fraction of bars with positive return among bars where a position was held."""
    r = np.asarray(returns, float)
    mask = ~np.isnan(r)
    if position is not None: mask &= np.abs(np.asarray(position, float)) > 1e-12
    return float((r[mask] > 0).mean()) if mask.any() else float("nan")


def turnover(position, periods_per_year=252) -> float:
    """Annualised sum of |change in position| (1.0 = one full notional traded)."""
    p = np.nan_to_num(np.asarray(position, float))
    if not len(p): return float("nan")
    return float(np.abs(np.diff(np.r_[0.0, p])).sum() * periods_per_year / len(p))


def exposure(position) -> float:
    p = np.nan_to_num(np.asarray(position, float)); return float(np.abs(p).mean()) if len(p) else float("nan")


def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0) -> float:
    """PSR: P(true per-period SR > sr_benchmark) given sample length, skew and kurtosis.
    ``sr_benchmark`` is per-period (not annualised)."""
    r = _r(returns); n = len(r)
    if n < 3 or r.std(ddof=1) == 0: return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3 = _st.skew(r); g4 = _st.kurtosis(r, fisher=False)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(_st.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / den))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR] of n_trials unskilled strategies (per-period units), False Strategy Theorem."""
    if n_trials <= 1: return 0.0
    z = _st.norm.ppf
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials)
                                    + EULER_GAMMA * z(1 - 1 / (n_trials * np.e))))


def deflated_sharpe_ratio(returns, n_trials: int, var_trial_sr: float | None = None,
                          trial_sharpes=None) -> float:
    """DSR = PSR evaluated against the SR expected from the best of ``n_trials`` lucky
    strategies. Pass the per-period Sharpe ratios of all trials (``trial_sharpes``) or
    their variance. Returns a probability in [0, 1]; > 0.95 is the usual bar."""
    if var_trial_sr is None:
        ts = _r(trial_sharpes) if trial_sharpes is not None else np.array([])
        var_trial_sr = float(ts.var(ddof=1)) if len(ts) > 1 else 0.0
    return probabilistic_sharpe_ratio(returns, expected_max_sharpe(n_trials, var_trial_sr))


def pbo_cscv(trial_returns: pd.DataFrame | np.ndarray, n_splits: int = 10) -> dict:
    """Probability of Backtest Overfitting via combinatorially symmetric cross-validation.
    ``trial_returns``: T x N matrix of per-period returns, one column per configuration.
    For each way of choosing half the S blocks as in-sample, pick the best IS config and
    record the relative rank of that config out-of-sample. PBO = P(OOS rank below median)."""
    M = np.asarray(trial_returns, float)
    T, N = M.shape
    if N < 2: raise ValueError("need >= 2 trials")
    S = n_splits - (n_splits % 2)
    blocks = np.array_split(np.arange(T), S)
    def sr(x):
        s = x.std(axis=0, ddof=1); return np.where(s > 0, x.mean(axis=0) / np.where(s > 0, s, 1), -np.inf)
    logits = []
    for is_idx in combinations(range(S), S // 2):
        ins = np.concatenate([blocks[i] for i in is_idx])
        oos = np.concatenate([blocks[i] for i in range(S) if i not in is_idx])
        best = int(np.argmax(sr(M[ins])))
        sro = sr(M[oos])
        w = (np.sum(sro < sro[best]) + 0.5 * (np.sum(sro == sro[best]) - 1) + 1) / (N + 1)  # rank in (0,1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "n_combinations": len(logits),
            "median_logit": float(np.median(logits))}


def summary(returns, position=None, periods_per_year=252) -> dict:
    r = pd.Series(returns).astype(float)
    out = {"CAGR": cagr(r, periods_per_year), "vol": float(_r(r).std(ddof=1) * np.sqrt(periods_per_year)),
           "Sharpe": sharpe(r, periods_per_year), "Sortino": sortino(r, periods_per_year),
           "Calmar": calmar(r, periods_per_year), "maxDD": max_drawdown(r),
           "maxDD_bars": max_drawdown_duration(r), "hit_rate": hit_rate(r, position),
           "PSR0": probabilistic_sharpe_ratio(r)}
    if position is not None:
        out["turnover"] = turnover(position, periods_per_year); out["exposure"] = exposure(position)
    return out
