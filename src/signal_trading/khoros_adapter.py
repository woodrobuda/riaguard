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

try:  # causal streaming mode (khoros-cantata-python 'operators' branch and later)
    from khoros_cantata import streaming as kstream
    STREAMING_AVAILABLE = True
except ImportError:  # pragma: no cover
    kstream = None; STREAMING_AVAILABLE = False

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


# ------------------------------------------------------------------ streaming mode
def _require_streaming():
    _require()
    if not STREAMING_AVAILABLE:
        raise ImportError("this khoros_cantata has no streaming module (needs the 'operators' branch or later)")


if STREAMING_AVAILABLE:
    class _OnlineStream(kstream.StreamOp):
        """Adapts a signal_trading online filter (``reset()``/``update(x)``) to khoros StreamOp."""
        def __init__(self, make, pick=None, warmup=0):
            self.make, self.pick, self.warmup = make, pick, int(warmup); self.reset()
        def reset(self): self.f = self.make(); self.n = 0
        def step(self, inputs):
            x = np.asarray(inputs["i"], float); out = np.empty(len(x))
            for t, v in enumerate(x):
                y = self.f.update(float(v)); y = y if self.pick is None else np.asarray(y)[self.pick]
                self.n += 1; out[t] = np.nan if self.n < self.warmup else y
            return {"o": out}

    class _LogRet(kstream.StreamOp):
        def reset(self): self.last = None
        def __init__(self): self.reset()
        def step(self, inputs):
            x = np.log(np.asarray(inputs["i"], float)); prev = np.r_[np.nan if self.last is None else self.last, x[:-1]]
            self.last = x[-1] if len(x) else self.last; return {"o": x - prev}


def register_streamable_operators():
    """Declare the SIGTRADE operators to the khoros streaming executor (causal ones get a
    stateful update; ``kpower`` is declared non-causal)."""
    _require_streaming(); T = "SIGTRADE"
    kstream.register_streamable("kema", lambda p, op, c: _OnlineStream(lambda: streaming.EMA(span=float(p.get("span", 20)))), T)
    kstream.register_streamable("ksma", lambda p, op, c: _OnlineStream(lambda: streaming.FIR.sma(int(p.get("n", 20))), warmup=int(p.get("n", 20))), T)
    kstream.register_streamable("kkalman", lambda p, op, c: _OnlineStream(lambda: streaming.KalmanLocalLinearTrend(
        float(p.get("ql", 1e-4)), float(p.get("qs", 1e-6)), float(p.get("r", 1e-2))), pick=0), T)
    kstream.register_streamable("klogret", lambda p, op, c: _LogRet(), T)
    kstream.register_streamable("kcross", lambda p, op, c: kstream.Pointwise(op, p), T)
    kstream.NON_CAUSAL["SIGTRADE:kpower"] = "power spectrum of the demeaned whole record (FFT)"


def stream_executor(path, registry=None):
    """Build a causal StreamingExecutor for a workspace using STREAM:kinput/koutput glyphs."""
    _require_streaming(); register_streamable_operators()
    return kstream.StreamingExecutor(path, register_operators(registry))


def stream_workspace(path, feeds, registry=None, chunk=1):
    """Stream whole records through a workspace ``chunk`` samples at a time; returns koutput arrays."""
    return stream_executor(path, registry).run(feeds, chunk=chunk)
