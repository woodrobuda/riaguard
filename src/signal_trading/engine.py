"""Strictly causal, vectorised backtest engine (simulated positions only).

Timing convention (no look-ahead):
  * a signal at bar t uses data up to and including the close of bar t;
  * the target position is reached ``delay`` bars later (default 1: trade at the close
    of bar t, earn the return of bar t+1 onwards);
  * costs are charged on the bar in which the position changes.

This module never talks to a broker. It computes hypothetical P&L for research."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from .costs import CostModel
from . import metrics as M
from .splits import walk_forward_splits

MODES = ("long_flat", "long_short", "long_only")


@dataclass(frozen=True)
class Sizing:
    method: str = "unit"            # "unit" (|pos| = max_leverage) or "vol_target"
    target_vol: float = 0.10        # annualised, for vol_target
    vol_span: int = 60              # EWM span of the realised-vol estimate
    max_leverage: float = 1.0
    periods_per_year: int = 252

    def scale(self, close: pd.Series) -> pd.Series:
        if self.method == "unit":
            return pd.Series(self.max_leverage, index=close.index)
        if self.method != "vol_target": raise ValueError(self.method)
        r = close.pct_change()
        vol = r.ewm(span=self.vol_span, adjust=False, min_periods=self.vol_span).std() * np.sqrt(self.periods_per_year)
        return (self.target_vol / vol).clip(upper=self.max_leverage).fillna(0.0)


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series          # net simulated returns per bar
    gross: pd.Series
    costs: pd.Series
    position: pd.Series         # position held over each bar (fraction of equity)
    meta: dict = field(default_factory=dict)

    @property
    def equity(self): return (1 + self.returns).cumprod()

    def stats(self, periods_per_year=252) -> dict:
        s = M.summary(self.returns, self.position, periods_per_year)
        s["cost_drag_bps_yr"] = float(self.costs.mean() * periods_per_year * 1e4)
        return {"strategy": self.name, **s}


def target_position(direction: pd.Series, mode="long_flat", scale: pd.Series | float = 1.0) -> pd.Series:
    if mode not in MODES: raise ValueError(f"mode must be one of {MODES}")
    d = direction.astype(float).fillna(0.0)
    if mode == "long_flat": d = d.clip(lower=0.0)
    elif mode == "long_only": d = pd.Series(1.0, index=d.index)
    return d * scale


def simulate(close: pd.Series, target: pd.Series, costs: CostModel = CostModel(), delay: int = 1,
             name: str = "strategy", meta: dict | None = None) -> BacktestResult:
    """Simulate holding ``target`` (decided at each bar's close) with ``delay`` bars of lag."""
    if delay < 1: raise ValueError("delay must be >= 1 (a signal cannot trade on its own bar's return)")
    close = close.astype(float)
    ret = close.pct_change().fillna(0.0)
    pos = target.reindex(close.index).astype(float).shift(delay).fillna(0.0)
    dpos = pos.diff().fillna(pos)
    vol = ret.ewm(span=20, adjust=False).std().shift(1)            # causal vol for vol-scaled slippage
    c = costs.trade_cost(dpos, vol) + costs.holding_cost(pos)
    gross = pos * ret
    return BacktestResult(name, gross - c, gross, c, pos, dict(meta or {}, delay=delay, costs=costs.to_dict()))


def run(close: pd.Series, signal, mode="long_flat", costs: CostModel = CostModel(),
        sizing: Sizing = Sizing(), delay: int = 1) -> BacktestResult:
    tgt = target_position(signal.direction(close), mode, sizing.scale(close))
    return simulate(close, tgt, costs, delay, signal.label,
                    {"signal": signal.to_dict(), "mode": mode, "sizing": asdict(sizing)})


def walk_forward(close: pd.Series, candidates, train=756, test=126, mode="long_flat",
                 costs: CostModel = CostModel(), sizing: Sizing = Sizing(), delay=1, gap=0,
                 expanding=False, name="walk_forward", select_metric=M.sharpe):
    """Walk-forward model selection. For each fold choose the candidate with the best
    ``select_metric`` on the training window (net of costs), hold its target positions in
    the following test window, and re-simulate the stitched target so that switching
    costs between folds are charged. Because every candidate signal is causal (verified
    by ``assert_causal`` in tests), computing it once on the full series gives the same
    values a live run would have produced at each bar."""
    scale = sizing.scale(close)
    targets = {c.label: target_position(c.direction(close), mode, scale) for c in candidates}
    nets = {k: simulate(close, t, costs, delay).returns for k, t in targets.items()}
    folds = walk_forward_splits(len(close), train, test, expanding=expanding, gap=gap)
    stitched = pd.Series(np.nan, index=close.index); chosen = []
    for tr, te in folds:
        scores = {k: select_metric(r.iloc[tr]) for k, r in nets.items()}
        best = max(scores, key=lambda k: -np.inf if np.isnan(scores[k]) else scores[k])
        # target decided at bar te[0]-delay onwards governs returns from te[0]
        lo = max(te[0] - delay, 0); hi = te[-1] - delay + 1
        stitched.iloc[lo:hi] = targets[best].iloc[lo:hi].to_numpy()
        chosen.append({"test_start": str(close.index[te[0]].date()), "test_end": str(close.index[te[-1]].date()),
                       "chosen": best, "train_score": float(scores[best])})
    first = folds[0][1][0] if folds else len(close)
    res = simulate(close, stitched.fillna(0.0), costs, delay, name,
                   {"mode": mode, "sizing": asdict(sizing), "folds": chosen, "n_candidates": len(candidates)})
    sl = slice(first, folds[-1][1][-1] + 1) if folds else slice(0, 0)
    return BacktestResult(name, res.returns.iloc[sl], res.gross.iloc[sl], res.costs.iloc[sl],
                          res.position.iloc[sl], res.meta), nets


def assert_causal(signal_fn, close: pd.Series, n_check=8, seed=0, perturb=True):
    """Look-ahead test. For random cut points t: (a) truncating the series after t and
    (b) replacing everything after t with random noise must leave outputs at <= t unchanged."""
    full = np.asarray(signal_fn(close), float); rng = np.random.default_rng(seed)
    for t in rng.integers(len(close) // 4, len(close) - 2, n_check):
        part = np.asarray(signal_fn(close.iloc[: t + 1]), float)
        if not np.allclose(full[: t + 1], part, equal_nan=True):
            raise AssertionError(f"output at/before t={t} changes when the future is removed")
        if perturb:
            c2 = close.copy()
            c2.iloc[t + 1:] = close.iloc[t] * np.exp(np.cumsum(rng.normal(0, 0.05, len(close) - t - 1)))
            alt = np.asarray(signal_fn(c2), float)
            if not np.allclose(full[: t + 1], alt[: t + 1], equal_nan=True):
                raise AssertionError(f"output at/before t={t} changes when the future is perturbed")
    return True
