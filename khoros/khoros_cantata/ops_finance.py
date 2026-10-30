"""Causal financial time-series operators (FINANCE toolbox).

Each operator is written once as an online state machine with ``reset()`` and
``step(inputs) -> outputs`` (chunks along axis 0).  The batch implementation is
"reset, then step over the whole record", so batch and streaming results are
identical by construction and every operator is causal: output ``t`` uses
inputs ``<= t`` only.

* ``klogret``   log return r_t = ln(x_t / x_{t-1}) (r_0 = fill)
* ``kzscore``   trailing z-score (x_t - mean) / std over the last ``window`` samples
* ``khold``     sample-and-hold of the last non-zero event (turns +1/-1 crossings into a position)
* ``kkalman``   local-linear-trend Kalman filter: outputs level (o) and slope (o2)
* ``kkalmanpair`` dynamic hedge ratio y_t = alpha_t + beta_t x_t + e_t (random-walk coefficients);
  outputs the one-step *prior* spread e_t = y_t - (alpha_{t|t-1} + beta_{t|t-1} x_t) (o),
  beta (o2, posterior) and alpha (o3, posterior)
* ``kthreshold3`` three-level regime: 0 below lo, 1 between, 2 above hi
"""
from __future__ import annotations
import collections
import numpy as np


class _Online:
    outputs = ('o',)
    def reset(self): pass
    def step(self, inputs): raise NotImplementedError
    def batch(self, inputs):
        self.reset(); return self.step({k: np.asarray(v, float) for k, v in inputs.items()})


class LogRet(_Online):
    def __init__(self, fill=0.0): self.fill = float(fill); self.reset()
    def reset(self): self.last = None
    def step(self, inputs):
        x = np.asarray(inputs['i'], float); out = np.empty(x.shape)
        for t in range(x.shape[0]):
            out[t] = self.fill if self.last is None else np.log(x[t] / self.last); self.last = x[t]
        return {'o': out}


class ZScore(_Online):
    def __init__(self, window=20, min_periods=2):
        self.w = int(window); self.minp = int(min_periods)
        if self.w < 2: raise ValueError('kzscore: -window must be >= 2')
        self.reset()
    def reset(self): self.buf = collections.deque(maxlen=self.w)
    def step(self, inputs):
        x = np.asarray(inputs['i'], float); out = np.empty(x.shape)
        for t in range(x.shape[0]):
            self.buf.append(x[t]); b = np.asarray(self.buf); s = b.std()
            out[t] = 0.0 if len(b) < self.minp or s == 0 else (x[t] - b.mean()) / s
        return {'o': out}


class Hold(_Online):
    def __init__(self, initial=0.0): self.init = float(initial); self.reset()
    def reset(self): self.v = self.init
    def step(self, inputs):
        x = np.asarray(inputs['i'], float); out = np.empty(x.shape)
        for t in range(x.shape[0]):
            if x[t] != 0 and not np.isnan(x[t]): self.v = x[t]
            out[t] = self.v
        return {'o': out}


class KalmanTrend(_Online):
    """State [level, slope]; F=[[1,1],[0,1]], H=[1,0], Q=diag(q_level,q_slope), R=r."""
    outputs = ('o', 'o2')
    def __init__(self, q_level=1e-4, q_slope=1e-6, r=1e-2):
        self.Q = np.diag([float(q_level), float(q_slope)]); self.R = float(r)
        if min(q_level, q_slope) < 0 or r <= 0: raise ValueError('kkalman: q >= 0 and r > 0 required')
        self.reset()
    def reset(self): self.x = None; self.P = np.eye(2) * 1e3
    def step(self, inputs):
        z = np.asarray(inputs['i'], float); lv = np.empty(z.shape); sl = np.empty(z.shape)
        F = np.array([[1., 1.], [0., 1.]])
        for t in range(z.shape[0]):
            if self.x is None: self.x = np.array([z[t], 0.0])
            else:
                x = F @ self.x; P = F @ self.P @ F.T + self.Q
                S = P[0, 0] + self.R; K = P[:, 0] / S
                self.x = x + K * (z[t] - x[0]); self.P = P - np.outer(K, P[0, :])
            lv[t], sl[t] = self.x
        return {'o': lv, 'o2': sl}


class KalmanPair(_Online):
    """Random-walk regression y = alpha + beta x; delta = coefficient drift, r = observation noise."""
    outputs = ('o', 'o2', 'o3')
    def __init__(self, delta=1e-4, r=1e-3):
        if not 0 < delta < 1 or r <= 0: raise ValueError('kkalmanpair: 0 < delta < 1 and r > 0 required')
        self.Vw = delta / (1 - delta) * np.eye(2); self.R = float(r); self.reset()
    def reset(self): self.th = np.zeros(2); self.P = np.eye(2)
    def step(self, inputs):
        x = np.asarray(inputs['i1'], float); y = np.asarray(inputs['i2'], float)
        e = np.empty(x.shape); b = np.empty(x.shape); a = np.empty(x.shape)
        for t in range(x.shape[0]):
            h = np.array([1.0, x[t]]); P = self.P + self.Vw
            yhat = h @ self.th; e[t] = y[t] - yhat
            S = h @ P @ h + self.R; K = P @ h / S
            self.th = self.th + K * e[t]; self.P = P - np.outer(K, h) @ P
            a[t], b[t] = self.th
        return {'o': e, 'o2': b, 'o3': a}


class Threshold3(_Online):
    def __init__(self, lo=0.0, hi=1.0):
        if lo > hi: raise ValueError('kthreshold3: -lo must be <= -hi')
        self.lo, self.hi = float(lo), float(hi)
    def step(self, inputs):
        x = np.asarray(inputs['i'], float)
        return {'o': np.where(x < self.lo, 0.0, np.where(x > self.hi, 2.0, 1.0))}


FACTORIES = {
    'klogret': lambda p: LogRet(p.get('fill', 0.0)),
    'kzscore': lambda p: ZScore(p.get('window', 20), p.get('minp', 2)),
    'khold': lambda p: Hold(p.get('initial', 0.0)),
    'kkalman': lambda p: KalmanTrend(p.get('ql', 1e-4), p.get('qs', 1e-6), p.get('r', 1e-2)),
    'kkalmanpair': lambda p: KalmanPair(p.get('delta', 1e-4), p.get('r', 1e-3)),
    'kthreshold3': lambda p: Threshold3(p.get('lo', 0.0), p.get('hi', 1.0)),
}
INPUTS = {'kkalmanpair': ('i1', 'i2')}


def kalman_trend_gain(q_level=1e-4, q_slope=1e-6, r=1e-2, iters=2000):
    """Steady-state Kalman gain [K_level, K_slope] (for lag estimates)."""
    k = KalmanTrend(q_level, q_slope, r); k.step({'i': np.zeros(iters)})
    F = np.array([[1., 1.], [0., 1.]]); P = F @ k.P @ F.T + k.Q
    return P[:, 0] / (P[0, 0] + k.R)
