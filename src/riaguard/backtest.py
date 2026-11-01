"""Walk-forward backtest with strict no-look-ahead.

Convention: signal s_t is computed from closes up to and including bar t; the
position held over bar t+1 is s_t (i.e. ``position = signal.shift(1)``).
Costs: ``cost_bps`` charged on |change in position| (per side)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Result:
    name: str; equity: pd.Series; returns: pd.Series; position: pd.Series

    def stats(self, periods_per_year=252):
        r = self.returns.dropna()
        if len(r) == 0: return {}
        ann = (1 + r).prod() ** (periods_per_year / len(r)) - 1
        vol = r.std() * np.sqrt(periods_per_year)
        dd = (self.equity / self.equity.cummax() - 1).min()
        return {"strategy": self.name, "CAGR": ann, "vol": vol,
                "Sharpe": (r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else np.nan,
                "maxDD": dd, "turnover/yr": self.position.diff().abs().sum() * periods_per_year / len(r),
                "exposure": self.position.mean()}


def run_signal(close: pd.Series, signal: pd.Series, name: str, cost_bps=5.0) -> Result:
    ret = close.pct_change()
    pos = signal.shift(1).fillna(0.0).clip(-1, 1)            # <-- the no-look-ahead step
    cost = pos.diff().abs().fillna(pos.abs()) * cost_bps / 1e4
    strat = pos * ret.fillna(0.0) - cost
    return Result(name, (1 + strat).cumprod(), strat, pos)


def assert_causal(signal_fn, close: pd.Series, n_check=5, seed=0):
    """Empirical look-ahead test: truncating the future must not change past signals."""
    full = signal_fn(close); rng = np.random.default_rng(seed)
    for t in rng.integers(len(close) // 4, len(close) - 1, n_check):
        part = signal_fn(close.iloc[: t + 1])
        a, b = full.iloc[: t + 1].to_numpy(), part.to_numpy()
        if not np.allclose(a, b, equal_nan=True):
            raise AssertionError(f"signal at/before t={t} changes when future data is removed")
    return True


def walk_forward(close: pd.Series, make_signal, param_grid, train=504, test=126, cost_bps=5.0, name="wf"):
    """Rolling walk-forward: in each window pick the param with best in-sample Sharpe
    (signals computed only on data up to the window end), then apply it out-of-sample
    to the next ``test`` bars. Returns concatenated OOS Result and chosen params."""
    oos = []; poss = []; chosen = []
    for start in range(0, len(close) - train - 1, test):
        tr_end = start + train; te_end = min(tr_end + test, len(close))
        best, best_s = None, -np.inf
        hist = close.iloc[: tr_end]
        for p in param_grid:
            r = run_signal(hist, make_signal(hist, p), "x", cost_bps).returns.iloc[start:]
            s = r.mean() / r.std() if r.std() > 0 else -np.inf
            if s > best_s: best, best_s = p, s
        upto = close.iloc[: te_end]                      # history incl. test bars, computed causally
        res = run_signal(upto, make_signal(upto, best), "x", cost_bps)
        oos.append(res.returns.iloc[tr_end:te_end]); poss.append(res.position.iloc[tr_end:te_end]); chosen.append((close.index[tr_end], best))
    r = pd.concat(oos); pos = pd.concat(poss)
    return Result(name, (1 + r).cumprod(), r, pos), chosen
