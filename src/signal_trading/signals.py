"""Causal directional signals with explainable internals.

Every signal maps a close series to a DataFrame with columns
``direction`` (-1, 0 or +1; 0 during warm-up), ``score`` (the quantity compared with
``threshold``) and the intermediate filter outputs. Row t uses closes <= t only.
DSP signals are built from the online filters in :mod:`signal_trading.streaming`;
benchmarks (buy-and-hold, SMA crossover, time-series momentum) use plain pandas
rolling windows, which are also causal."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from . import streaming as st


@dataclass(frozen=True)
class Signal:
    name: str
    params: tuple = ()
    threshold: float = 0.0
    kind: str = "dsp"                      # "dsp" or "benchmark"
    description: str = field(default="", compare=False)

    def compute(self, close: pd.Series) -> pd.DataFrame:
        return _REGISTRY[self.name](close, *self.params)

    def direction(self, close: pd.Series) -> pd.Series:
        return self.compute(close)["direction"]

    @property
    def label(self) -> str:
        return f"{self.name}{self.params}" if self.params else self.name

    def to_dict(self):
        return {"name": self.name, "params": list(self.params), "threshold": self.threshold, "kind": self.kind}


def _frame(close, score, warm, **feat):
    score = np.asarray(score, float)
    d = np.sign(np.nan_to_num(score)); d[: warm] = 0.0; d[np.isnan(score)] = 0.0
    return pd.DataFrame({"direction": d, "score": score, **{k: np.asarray(v, float) for k, v in feat.items()}},
                        index=close.index)


def ema_cross(close, fast=20, slow=100):
    x = close.to_numpy(float)
    a = st.run(st.EMA(span=fast), x); b = st.run(st.EMA(span=slow), x)
    return _frame(close, a / b - 1.0, slow, ema_fast=a, ema_slow=b)


def kalman_slope(close, q_level=1e-4, q_slope=1e-6, r_window=50):
    lx = np.log(close.to_numpy(float))
    # observation noise from the first r_window returns only (fixed once known: causal)
    r = float(np.var(np.diff(lx[: r_window + 1]))) if len(lx) > r_window else 1e-4
    states = st.run(st.KalmanLocalLinearTrend(q_level, q_slope, r), lx)
    return _frame(close, states[:, 1], max(r_window, 20), kalman_level=states[:, 0], kalman_slope=states[:, 1])


def butter_trend(close, cutoff=0.02, order=2):
    """Causal Butterworth low-pass on log price; score = 5-bar slope of the filtered series."""
    from scipy import signal as sg
    b, a = sg.butter(order, cutoff)
    lx = np.log(close.to_numpy(float))
    y = st.run(st.IIR(b, a), lx)
    slope = np.r_[np.full(5, np.nan), y[5:] - y[:-5]]
    return _frame(close, slope, int(2 / cutoff), lowpass=y)


def sma_crossover(close, fast=50, slow=200):
    f = close.rolling(fast).mean(); s = close.rolling(slow).mean()
    return _frame(close, (f / s - 1.0).to_numpy(), slow, sma_fast=f, sma_slow=s)


def tsmom(close, lookback=252, skip=0):
    """Time-series momentum (Moskowitz, Ooi & Pedersen 2012): sign of trailing return."""
    past = close.shift(skip)
    ret = past / past.shift(lookback) - 1.0
    return _frame(close, ret.to_numpy(), lookback + skip, trailing_return=ret)


def buy_and_hold(close):
    return _frame(close, np.ones(len(close)), 0)


_REGISTRY = {"ema_cross": ema_cross, "kalman_slope": kalman_slope, "butter_trend": butter_trend,
             "sma_crossover": sma_crossover, "tsmom": tsmom, "buy_and_hold": buy_and_hold}


def make(name, *params, kind=None) -> Signal:
    if name not in _REGISTRY: raise KeyError(name)
    k = kind or ("benchmark" if name in ("sma_crossover", "tsmom", "buy_and_hold") else "dsp")
    return Signal(name, tuple(params), 0.0, k)


BENCHMARKS = [make("buy_and_hold"), make("sma_crossover", 50, 200), make("tsmom", 252)]
DSP_GRIDS = {
    "ema_cross": [make("ema_cross", f, s) for f in (10, 20, 50) for s in (100, 150, 200)],
    "kalman_slope": [make("kalman_slope", ql, qs) for ql in (1e-5, 1e-4) for qs in (1e-7, 1e-6, 1e-5)],
    "butter_trend": [make("butter_trend", c) for c in (0.01, 0.02, 0.04, 0.08)],
}
