"""Optional bridge to khoros-cantata-python (https://github.com/woodrobuda/khoros-cantata-python).

Registers trading operators into its ``NativeOperatorRegistry`` so they can be
wired in a Cantata-style ``.wksp`` workspace, and reuses its ``khoros_fft`` and
``conv1d_khoros`` numerics. Import fails gracefully if khoros_cantata is absent."""
from __future__ import annotations
import numpy as np

try:
    from khoros_cantata.workspace_runtime import (NativeOperatorRegistry, WorkspaceExecutor,
                                                  HistoricalWorkspaceParser)
    from khoros_cantata.phase21_numeric import khoros_fft, conv1d_khoros
    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False

from . import streaming


def _require():
    if not AVAILABLE:
        raise ImportError("pip install khoros-cantata-python (see README) to use the adapter")


def causal_sma_khoros(x, n):
    """Trailing SMA via Khoros klinearop semantics: with kernel origin w0=0,
    y[j] = sum_q k[q] x[j-q] is causal. First n-1 outputs are partial (set NaN)."""
    _require(); y = conv1d_khoros(x, np.ones(n) / n, w0=0); y[: n - 1] = np.nan; return y


def power_spectrum_khoros(x):
    """|kfft|^2 of a demeaned series (scale=2 => 1/N normalisation)."""
    _require(); x = np.asarray(x, float); x = x - x.mean()
    return np.abs(khoros_fft(x, [0], direction=0, scale=2)) ** 2


def register_operators(registry=None):
    """Register SIGTRADE:* operators. Ports: input 'i' (and 'i2' for pairs),
    output 'o'. Parameters come from the glyph command line, e.g. ``kema -span 20``."""
    _require(); r = registry or NativeOperatorRegistry()
    arr = lambda v: np.asarray(v, float)
    r.register("SIGTRADE", "kema", lambda p, i: {"o": streaming.run(streaming.EMA(span=float(p.get("span", 20))), arr(i["i"]))})
    r.register("SIGTRADE", "ksma", lambda p, i: {"o": causal_sma_khoros(arr(i["i"]), int(p.get("n", 20)))})
    r.register("SIGTRADE", "kkalman", lambda p, i: {"o": streaming.run(streaming.KalmanLocalLinearTrend(
        float(p.get("ql", 1e-4)), float(p.get("qs", 1e-6)), float(p.get("r", 1e-2))), arr(i["i"]))[:, 0]})
    r.register("SIGTRADE", "kpower", lambda p, i: {"o": power_spectrum_khoros(arr(i["i"]))})
    r.register("SIGTRADE", "klogret", lambda p, i: {"o": np.r_[np.nan, np.diff(np.log(arr(i["i"])))]})
    r.register("SIGTRADE", "kcross", lambda p, i: {"o": np.where(arr(i["i"]) > arr(i["i2"]), 1.0, 0.0)})
    return r


def run_workspace(path, registry):
    """Parse and execute a .wksp file; returns the kernel Workspace (inspect node outputs)."""
    _require(); return WorkspaceExecutor(registry).execute(HistoricalWorkspaceParser().parse(path))
