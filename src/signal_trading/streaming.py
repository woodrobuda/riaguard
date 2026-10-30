"""Online (causal, one-sample-at-a-time) filters. Each has ``update(x) -> y`` and
``reset()``; output at step t depends only on inputs <= t."""
from __future__ import annotations
from collections import deque
import numpy as np


class EMA:
    """y_t = a x_t + (1-a) y_{t-1}; a = 2/(span+1). First-order IIR low-pass.
    Approx. low-frequency group delay = (1-a)/a samples."""
    def __init__(self, span: float | None = None, alpha: float | None = None):
        if alpha is None:
            alpha = 2.0 / (span + 1.0)
        self.alpha = float(alpha); self.reset()
    def reset(self): self.y = None
    def update(self, x: float) -> float:
        self.y = x if self.y is None else self.alpha * x + (1 - self.alpha) * self.y
        return self.y
    @property
    def lag(self): return (1 - self.alpha) / self.alpha


class FIR:
    """Causal FIR filter y_t = sum_k b_k x_{t-k}. Returns NaN until the window is full."""
    def __init__(self, taps):
        self.b = np.asarray(taps, float); self.reset()
    def reset(self): self.buf = deque(maxlen=len(self.b))
    def update(self, x: float) -> float:
        self.buf.appendleft(float(x))
        if len(self.buf) < len(self.b):
            return float("nan")
        return float(np.dot(self.b, self.buf))
    @classmethod
    def sma(cls, n): return cls(np.ones(n) / n)          # group delay (n-1)/2
    @classmethod
    def wma(cls, n):
        w = np.arange(n, 0, -1, dtype=float); return cls(w / w.sum())


class IIR:
    """Generic causal IIR (b, a) in direct form; e.g. from scipy.signal.butter."""
    def __init__(self, b, a):
        a = np.asarray(a, float); self.b = np.asarray(b, float) / a[0]; self.a = a / a[0]; self.reset()
    def reset(self):
        self.x = deque([0.0] * len(self.b), maxlen=len(self.b))
        self.y = deque([0.0] * (len(self.a) - 1), maxlen=max(len(self.a) - 1, 1)); self._init = False
    def update(self, x):
        if not self._init:  # start at steady state to avoid a big transient
            self.x = deque([x] * len(self.b), maxlen=len(self.b))
            self.y = deque([x] * (len(self.a) - 1), maxlen=max(len(self.a) - 1, 1)); self._init = True
        self.x.appendleft(float(x))
        y = float(np.dot(self.b, self.x) - np.dot(self.a[1:], list(self.y)[:len(self.a) - 1]))
        self.y.appendleft(y); return y


class KalmanLocalLinearTrend:
    """2-state Kalman filter: level + slope (local linear trend), observation = price.
    q_level, q_slope: process noise variances; r: observation noise variance."""
    def __init__(self, q_level=1e-4, q_slope=1e-6, r=1e-2):
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]]); self.H = np.array([[1.0, 0.0]])
        self.Q = np.diag([q_level, q_slope]); self.R = r; self.reset()
    def reset(self): self.x = None; self.P = np.eye(2) * 1e3
    def update(self, z: float):
        if self.x is None:
            self.x = np.array([z, 0.0]); return self.x.copy()
        x = self.F @ self.x; P = self.F @ self.P @ self.F.T + self.Q
        S = float((self.H @ P @ self.H.T)[0, 0]) + self.R; K = (P @ self.H.T).ravel() / S
        self.x = x + K * (z - x[0]); self.P = (np.eye(2) - np.outer(K, self.H)) @ P
        return self.x.copy()  # [level, slope]


class KalmanHedgeRatio:
    """Dynamic regression y_t = beta_t x_t + alpha_t + e_t with random-walk (beta, alpha).
    Classic pairs-trading use (cf. Chan, 'Algorithmic Trading', ch.3). Returns
    (beta, alpha, innovation e_t, innovation variance S_t) using the *prior* state,
    so the innovation at t is a genuine out-of-sample forecast error."""
    def __init__(self, delta=1e-4, r=1e-3):
        self.Vw = delta / (1 - delta) * np.eye(2); self.R = r; self.reset()
    def reset(self): self.theta = np.zeros(2); self.P = np.eye(2)
    def update(self, x: float, y: float):
        h = np.array([x, 1.0]); P = self.P + self.Vw
        yhat = h @ self.theta; e = y - yhat; S = h @ P @ h + self.R
        K = P @ h / S; self.theta = self.theta + K * e; self.P = P - np.outer(K, h) @ P
        return float(self.theta[0]), float(self.theta[1]), float(e), float(S)


class RollingZScore:
    def __init__(self, n):
        self.buf = deque(maxlen=n)
    def update(self, x):
        self.buf.append(float(x))
        if len(self.buf) < self.buf.maxlen: return float("nan")
        a = np.asarray(self.buf); s = a.std()
        return float((x - a.mean()) / s) if s > 0 else 0.0


def run(filter_obj, xs):
    """Convenience: feed a whole sequence through an online filter (still causal)."""
    filter_obj.reset(); return np.array([filter_obj.update(x) for x in xs])
