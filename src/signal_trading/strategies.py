"""Filter-based example signals. Each maps a close series to a target position in
[-1, 1] (here long/flat) computed causally with the streaming filters."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import streaming as st


def buy_and_hold(close, _=None): return pd.Series(1.0, index=close.index)


def ema_cross(close, p=(20, 100)):
    f, s = p; x = close.to_numpy()
    a = st.run(st.EMA(span=f), x); b = st.run(st.EMA(span=s), x)
    sig = (a > b).astype(float); sig[: s] = 0.0
    return pd.Series(sig, index=close.index)


def sma_trend(close, n=200):
    y = st.run(st.FIR.sma(n), close.to_numpy())
    return pd.Series(np.where(np.isnan(y), 0.0, (close.to_numpy() > y).astype(float)), index=close.index)


def kalman_slope(close, p=(1e-4, 1e-6)):
    ql, qs = p; lx = np.log(close.to_numpy())
    kf = st.KalmanLocalLinearTrend(ql, qs, r=float(np.var(np.diff(lx[:50]))) if len(lx) > 50 else 1e-4)
    states = st.run(kf, lx); sig = (states[:, 1] > 0).astype(float); sig[:20] = 0.0
    return pd.Series(sig, index=close.index)
