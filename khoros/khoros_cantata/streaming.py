"""Causal streaming execution for Cantata workspaces.

Batch execution (``WorkspaceExecutor``) hands each glyph its *whole* input
record.  Streaming execution pushes samples (or chunks of samples, along axis 0)
through the same graph incrementally; every glyph keeps its own state between
pushes, so output sample ``t`` can depend only on input samples ``<= t``.

Operators opt in by declaring a stateful update interface (:class:`StreamOp`)
via :func:`register_streamable`.  Operators known to need the whole record are
declared in :data:`NON_CAUSAL` with a reason; option-dependent ones (``kiir
-zerophase 1``, ``kfir -mode same``, ``klinearop -w0 >0``, ``kconvert -norm 1``)
are classified per glyph.  A graph containing anything that is not streamable
is rejected *unless* that glyph is wrapped in ``STREAM:krolling`` (a trailing-
window adapter: at each step it re-runs the batch operator on the last
``-window`` samples and emits only the final output sample).

I/O glyphs: ``STREAM:kinput -name price`` (fed by ``push``) and
``STREAM:koutput -name signal`` (collected).  The same workspace also runs in
batch mode with :func:`batch_execute`, which is how stream == batch is tested.
"""
from __future__ import annotations
import collections
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from scipy import signal as _sig
from .workspace_runtime import (HistoricalWorkspace, HistoricalWorkspaceParser, NativeOperatorRegistry,
                                WorkspaceExecutor, parse_command)
from . import ops_signal as S


class NonCausalGraphError(ValueError):
    """Raised when a workspace contains glyphs that cannot run causally."""
    def __init__(self, problems):
        self.problems = problems
        super().__init__('workspace is not streamable:\n' + '\n'.join(f'  glyph {i} ({n}): {r}' for i, n, r in problems))


class LookAheadError(AssertionError):
    pass

# ====================================================================== operator interface
class StreamOp:
    """Stateful causal operator. ``step`` receives chunks (arrays whose axis 0 is time,
    all inputs the same length) and returns equally long output chunks."""
    def reset(self): pass
    def step(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: raise NotImplementedError


class Pointwise(StreamOp):
    """Stateless sample-wise operator: the batch implementation applied to each chunk."""
    def __init__(self, batch_op, params): self.op, self.p = batch_op, params
    def step(self, inputs): return {k: np.asarray(v) for k, v in self.op(self.p, inputs).items()}


class FIRStream(StreamOp):
    def __init__(self, taps): self.b = np.asarray(taps, float); self.reset()
    def reset(self): self.zi = None
    def step(self, inputs):
        x = np.asarray(inputs['i'], float) if 'i' in inputs else np.asarray(inputs['i1'], float)
        if self.zi is None: self.zi = np.zeros((len(self.b) - 1,) + x.shape[1:])
        if len(self.b) == 1: return {'o': x * self.b[0]}
        y, self.zi = _sig.lfilter(self.b, [1.0], x, axis=0, zi=self.zi); return {'o': y}


class IIRStream(StreamOp):
    def __init__(self, sos): self.sos = np.atleast_2d(sos); self.reset()
    def reset(self): self.zi = None
    def step(self, inputs):
        x = np.asarray(inputs['i'], float)
        if self.zi is None: self.zi = np.zeros((self.sos.shape[0], 2) + x.shape[1:])
        y, self.zi = _sig.sosfilt(self.sos, x, axis=0, zi=self.zi); return {'o': y}


class EMAStream(StreamOp):
    def __init__(self, span=None, alpha=None):
        self.a = float(alpha) if alpha is not None else 2.0 / (float(span or 20) + 1); self.reset()
    def reset(self): self.y = None
    def step(self, inputs):
        x = np.asarray(inputs['i'], float); out = np.empty_like(x)
        for t in range(x.shape[0]):
            self.y = x[t] if self.y is None else self.a * x[t] + (1 - self.a) * self.y; out[t] = self.y
        return {'o': out}


class RunStatsStream(StreamOp):
    def __init__(self, stat='mean', window=None):
        self.stat, self.window = stat, (None if window is None else int(window)); self.reset()
    def reset(self):
        self.n = 0; self.s = self.ss = 0.0; self.mn = self.mx = None
        self.buf = collections.deque(maxlen=self.window) if self.window else None
    def step(self, inputs):
        x = np.asarray(inputs['i'], float); out = np.empty_like(x)
        f = {'mean': np.mean, 'var': np.var, 'std': np.std, 'min': np.min, 'max': np.max, 'sum': np.sum}[self.stat]
        for t in range(x.shape[0]):
            v = x[t]
            if self.buf is not None:
                self.buf.append(v); out[t] = f(np.asarray(self.buf), axis=0); continue
            self.n += 1; self.s = self.s + v; self.ss = self.ss + v * v
            self.mn = v if self.mn is None else np.minimum(self.mn, v); self.mx = v if self.mx is None else np.maximum(self.mx, v)
            m = self.s / self.n; var = np.maximum(self.ss / self.n - m * m, 0)
            out[t] = {'mean': m, 'var': var, 'std': np.sqrt(var), 'min': self.mn, 'max': self.mx, 'sum': self.s}[self.stat]
        return {'o': out}


class DelayStream(StreamOp):
    def __init__(self, n=1, fill=0.0): self.n, self.fill = int(n), fill; self.reset()
    def reset(self): self.buf = None
    def step(self, inputs):
        x = np.asarray(inputs['i'], float)
        if self.n == 0: return {'o': x.copy()}
        if self.buf is None: self.buf = np.full((self.n,) + x.shape[1:], self.fill)
        z = np.concatenate([self.buf, x], 0); self.buf = z[-self.n:]; return {'o': z[:x.shape[0]]}


class DiffStream(StreamOp):
    def __init__(self, fill=0.0): self.fill = fill; self.reset()
    def reset(self): self.last = None
    def step(self, inputs):
        x = np.asarray(inputs['i'], float)
        if x.shape[0] == 0: return {'o': x.copy()}
        prev = np.concatenate([[self.last] if self.last is not None else [x[0]], x[:-1]], 0)
        out = x - prev
        if self.last is None: out[0] = self.fill
        self.last = x[-1]; return {'o': out}


class CrossStream(StreamOp):
    def __init__(self, const=0.0): self.c = const; self.reset()
    def reset(self): self.above = None
    def step(self, inputs):
        a = np.asarray(inputs['i1'], float); b = np.asarray(inputs.get('i2', self.c), float)
        above = (a - b) > 0; above = np.broadcast_to(above, a.shape); out = np.zeros(a.shape)
        for t in range(a.shape[0]):
            if self.above is not None:
                out[t] = 1.0 if above[t] and not self.above else -1.0 if (not above[t]) and self.above else 0.0
            self.above = bool(above[t])
        return {'o': out}


class RollingCausal(StreamOp):
    """Trailing-window adapter: makes any same-length batch operator causal by running it
    on the last ``window`` samples (fewer at start-up) and emitting the final sample.
    If the operator cannot run on a not-yet-full start-up window, that sample is NaN."""
    def __init__(self, batch_op, params, window):
        if int(window) < 1: raise ValueError('krolling: -window must be >= 1')
        self.op, self.p, self.w = batch_op, params, int(window); self.reset()
    def reset(self): self.bufs = {}
    def step(self, inputs):
        n = len(next(iter(inputs.values()))); outs = collections.defaultdict(list)
        for k, v in inputs.items(): self.bufs.setdefault(k, collections.deque(maxlen=self.w))
        for t in range(n):
            win = {}
            for k, v in inputs.items():
                self.bufs[k].append(np.asarray(v)[t]); win[k] = np.asarray(self.bufs[k])
            try:
                res = self.op(self.p, win)
            except Exception:
                # Start-up: some operators (e.g. filtfilt) cannot run on a short window yet.
                if len(next(iter(self.bufs.values()))) >= self.w: raise
                res = {'o': np.full(1, np.nan)}
            for k, v in res.items(): outs[k].append(np.asarray(v)[-1])
        return {k: np.asarray(v) for k, v in outs.items()}

# ====================================================================== declarations
#: name -> factory(params, batch_op, connected_inputs) -> StreamOp (may raise NonCausal(reason))
STREAMABLE: dict[str, Callable[..., StreamOp]] = {}

POINTWISE = ['karith1', 'karith2', 'kadd', 'ksub', 'kmul', 'kdiv', 'kdivinto', 'kpow', 'kmin', 'kmax', 'kabsdiff',
             'kexp', 'klog', 'klog10', 'ksin', 'kcos', 'ktan', 'katan', 'kceil', 'ksign', 'kneg', 'kabs', 'ksqr', 'ksqrt',
             'krecip', 'kcbrt', 'kcube', 'kfloor', 'ktruncate', 'kbitnot', 'kclip', 'kclipout', 'kbitwise', 'kcompare',
             'klogexp', 'kbessel', 'kblend', 'kgate', 'kinverse', 'kcmplx', 'kreal2cmplx', 'kcmplx2real', 'kmag',
             'kreplace', 'kmapdata', 'kunmask', 'kconvert', 'identity']

#: operators that need the whole record (or future samples); reason shown on rejection.
#: Keys are 'name' or 'TOOLBOX:name'.
NON_CAUSAL: dict[str, str] = {
    'kfft': 'FFT over the whole record', 'kspectrum': 'FFT over the whole record', 'kpsd': 'spectral estimate over the whole record',
    'khilbert': 'Hilbert transform (FFT, two-sided)', 'kdwt': 'block wavelet transform (wavelet denoise needs the whole block)',
    'kidwt': 'block inverse wavelet transform', 'kwiener': 'frequency-domain / centred-window Wiener filter',
    'kconvolve': 'full/centred convolution (use kfir or klinearop -w0 0)', 'kcorr': 'two-sided correlation',
    'kresample': 'resampling interpolates between neighbours on both sides', 'khisto': 'histogram of the whole record',
    'khistops': 'histogram equalisation uses the whole-record CDF', 'knormal': 'normalises by whole-record min/max/mean',
    'kstats': 'whole-record summary', 'kflip': 'time reversal', 'kextract': 'region selection by absolute index',
    'kpad': 'absolute-index padding', 'kappend': 'record concatenation', 'korient': 'axis reordering', 'imedian': 'centred median window',
    'igradient': 'centred differences', 'ilaplacian': 'centred differences', 'iedge': 'centred differences', 'irotate': '2-D resampling',
    'kgsin': 'block generator', 'kgnoise': 'block generator', 'kshot': 'noise uses whole-record min/max by default',
}


class NonCausal(Exception):
    pass


def register_streamable(name, factory, toolbox='*'):
    """Declare an operator streamable. ``factory(params, batch_op, connected) -> StreamOp``.
    ``toolbox`` scopes the declaration (e.g. 'SIGTRADE'); '*' applies to any toolbox."""
    STREAMABLE[name.lower() if toolbox == '*' else f'{toolbox.upper()}:{name.lower()}'] = factory


def _factory(name, toolbox='*'):
    return STREAMABLE.get(f'{toolbox.upper()}:{name}') or STREAMABLE.get(name)


for _n in POINTWISE:
    register_streamable(_n, lambda p, op, c: Pointwise(op, p))


def _conv_factory(p, op, c):
    if p.get('norm'): raise NonCausal('-norm rescales by whole-record min/max')
    return Pointwise(op, p)
register_streamable('kconvert', _conv_factory)


def _fir_factory(p, op, c):
    if 'i2' in c: raise NonCausal('streaming kfir needs static taps from options, not a connected kernel')
    if p.get('mode', 'causal') != 'causal': raise NonCausal(f"-mode {p.get('mode')} is centred/full (non-causal)")
    return FIRStream(S.kfir_design(p.get('taps', 31), p.get('cutoff', 0.25), p.get('ftype', 'lowpass'), p.get('window', 'hamming')))
register_streamable('kfir', _fir_factory)


def _linop_factory(p, op, c):
    if 'i2' in c: raise NonCausal('streaming klinearop needs the kernel from -n, not a connected kernel')
    if int(p.get('w0', 0)) != 0 or p.get('ext'): raise NonCausal('-w0 > 0 / -ext shifts output earlier (uses future samples)')
    n = int(p.get('n', 3)); return FIRStream(np.ones(n) / n)
register_streamable('klinearop', _linop_factory)


def _iir_factory(p, op, c):
    if p.get('zerophase'): raise NonCausal('-zerophase 1 runs filtfilt (forward-backward, non-causal)')
    return IIRStream(S.kiir_design(p.get('order', 4), p.get('cutoff', 0.25), p.get('ftype', 'lowpass'), p.get('family', 'butter'),
                                   p.get('rp', 1.0), p.get('rs', 40.0)))
register_streamable('kiir', _iir_factory)
register_streamable('kema', lambda p, op, c: EMAStream(p.get('span'), p.get('alpha')))
register_streamable('krunstats', lambda p, op, c: RunStatsStream(p.get('stat', 'mean'), p.get('window')))
register_streamable('kdelay', lambda p, op, c: DelayStream(p.get('n', 1), p.get('fill', 0.0)))
register_streamable('kdiff', lambda p, op, c: DiffStream(p.get('fill', 0.0)))
register_streamable('kcross', lambda p, op, c: CrossStream(p.get('real', 0)))


from . import ops_finance as _F
for _n, _fac in _F.FACTORIES.items():
    register_streamable(_n, lambda p, op, c, _fac=_fac: _fac(p))


def classify(name, params=None, connected=(), toolbox='*'):
    """Return ('streamable'|'non-causal'|'undeclared', reason)."""
    name = name.lower(); params = params or {}; f = _factory(name, toolbox)
    if f is not None:
        try: f(params, None, set(connected)); return 'streamable', ''
        except NonCausal as e: return 'non-causal', str(e)
    for key in (f'{toolbox.upper()}:{name}', name):
        if key in NON_CAUSAL: return 'non-causal', NON_CAUSAL[key]
    return 'undeclared', 'operator has not declared a causal streaming interface'

# ====================================================================== I/O and wrapper glyphs (batch side)
def streaming_registry(feeds=None, registry=None, sink=None):
    """Batch registry with STREAM:kinput/koutput/krolling so a streaming workspace also runs in batch."""
    r = registry or NativeOperatorRegistry(); feeds = feeds or {}
    r.register('STREAM', 'kinput', lambda p, i: {'o': np.asarray(feeds[p['name']])})
    def out(p, i):
        if sink is not None: sink[p['name']] = np.asarray(i['i'])
        return {'o': i['i']}
    r.register('STREAM', 'koutput', out)
    def rolling(p, i):
        inner = r.resolve(p.get('tb', '*'), p['op']); rc = RollingCausal(inner, _inner_params(p), p.get('window', 32))
        return rc.step({k: np.asarray(v) for k, v in i.items()})
    r.register('STREAM', 'krolling', rolling)
    return r


def _inner_params(p): return {k: v for k, v in p.items() if k not in ('op', 'window', 'tb')}


def batch_execute(hws, feeds, registry=None):
    """Run a streaming workspace on whole records with the ordinary batch executor."""
    if not isinstance(hws, HistoricalWorkspace): hws = HistoricalWorkspaceParser().parse(hws)
    sink = {}; WorkspaceExecutor(streaming_registry(feeds, registry, sink)).execute(hws); return sink

# ====================================================================== executor
@dataclass
class _SNode:
    ident: int; name: str; tb: str; params: dict; op: StreamOp | None
    inputs: dict[str, tuple[int, str]] = field(default_factory=dict)  # port -> (src id, src port)
    kind: str = 'op'


class StreamingExecutor:
    """Push samples through a workspace causally.

    >>> ex = StreamingExecutor('pipe.wk'); ex.push(price=101.2)  # -> {'signal': array([...])}
    """
    def __init__(self, workspace, registry: NativeOperatorRegistry | None = None):
        hws = workspace if isinstance(workspace, HistoricalWorkspace) else HistoricalWorkspaceParser().parse(workspace)
        self.registry = streaming_registry(registry=registry or NativeOperatorRegistry())
        conn = collections.defaultdict(dict)
        for c in hws.connections: conn[c.in_id][c.in_port] = (c.out_id, c.out_port)
        self.nodes: dict[int, _SNode] = {}; problems = []
        for g in hws.glyphs:
            p = parse_command(g.command)[1]; name = g.oname.lower(); connected = set(conn[g.ident])
            n = _SNode(g.ident, name, g.toolbox.upper(), p, None, dict(conn[g.ident]))
            if name in ('kinput', 'koutput'):
                if 'name' not in p: problems.append((g.ident, name, 'needs -name'))
                n.kind = name
            elif name == 'krolling':
                inner = self.registry.resolve(p.get('tb', '*'), str(p.get('op', '')))
                if inner is None: problems.append((g.ident, name, f"unknown wrapped operator {p.get('op')!r}"))
                else: n.op = RollingCausal(inner, _inner_params(p), p.get('window', 32))
            else:
                status, why = classify(name, p, connected, g.toolbox)
                if status != 'streamable':
                    problems.append((g.ident, name, f'{status}: {why}; wrap it as STREAM:krolling -op {name} -window N'))
                else:
                    n.op = _factory(name, g.toolbox)(p, self.registry.resolve(g.toolbox, name), connected)
            self.nodes[g.ident] = n
        if problems: raise NonCausalGraphError(problems)
        self.order = self._toposort()
        self.inputs = sorted(n.params['name'] for n in self.nodes.values() if n.kind == 'kinput')
        self.outputs = sorted(n.params['name'] for n in self.nodes.values() if n.kind == 'koutput')

    def _toposort(self):
        indeg = {i: len(n.inputs) for i, n in self.nodes.items()}; down = collections.defaultdict(list)
        for i, n in self.nodes.items():
            for src, _ in n.inputs.values(): down[src].append(i)
        q = collections.deque(sorted(i for i, d in indeg.items() if d == 0)); order = []
        while q:
            i = q.popleft(); order.append(i)
            for j in down[i]:
                indeg[j] -= 1
                if indeg[j] == 0: q.append(j)
        if len(order) != len(self.nodes): raise NonCausalGraphError([(0, 'graph', 'cycle without a delay is not streamable')])
        return order

    def reset(self):
        for n in self.nodes.values():
            if n.op: n.op.reset()

    def push(self, feeds: dict[str, Any] | None = None, **kw) -> dict[str, np.ndarray]:
        """Push one sample (scalar) or a chunk (array, axis 0 = time) per input name; returns
        the new output chunk for every ``koutput``."""
        feeds = {**(feeds or {}), **kw}
        missing = set(self.inputs) - set(feeds)
        if missing: raise KeyError(f'missing stream inputs: {sorted(missing)}')
        vals: dict[tuple[int, str], np.ndarray] = {}; result = {}
        for i in self.order:
            n = self.nodes[i]; ins = {port: vals[src] for port, src in n.inputs.items()}
            if n.kind == 'kinput': vals[(i, 'o')] = np.atleast_1d(np.asarray(feeds[n.params['name']], float)); continue
            if n.kind == 'koutput': vals[(i, 'o')] = result[n.params['name']] = ins['i']; continue
            for k, v in n.op.step(ins).items(): vals[(i, k)] = v
        return result

    def run(self, feeds: dict[str, Any], chunk: int = 1) -> dict[str, np.ndarray]:
        """Stream whole records through in chunks of ``chunk`` samples; returns concatenated outputs."""
        arrs = {k: np.asarray(v, float) for k, v in feeds.items()}; n = len(next(iter(arrs.values())))
        parts = collections.defaultdict(list)
        for s in range(0, n, int(chunk)):
            for k, v in self.push({k: a[s:s + chunk] for k, a in arrs.items()}).items(): parts[k].append(v)
        return {k: np.concatenate(v, 0) for k, v in parts.items()}

# ====================================================================== look-ahead verification
def find_lookahead(fn: Callable[[np.ndarray], Any], x, checkpoints=None, trials=3, seed=0, atol=1e-12):
    """Perturb samples after each checkpoint t and report outputs at <= t that changed.
    ``fn`` maps an input record to an output array (or dict of arrays). Returns a list of
    (t, output_key, first_changed_index) violations (empty = no look-ahead detected)."""
    x = np.asarray(x, float); g = np.random.default_rng(seed); n = x.shape[0]
    checkpoints = sorted(set(checkpoints if checkpoints is not None else np.linspace(0, n - 2, min(12, n - 1)).astype(int)))
    as_dict = lambda r: r if isinstance(r, dict) else {'o': r}
    base = as_dict(fn(x)); bad = []
    for t in checkpoints:
        for _ in range(int(trials)):
            x2 = x.copy(); scale = np.nanstd(x) or 1.0
            x2[t + 1:] = x2[t + 1:] + g.normal(0, 10 * scale, x2[t + 1:].shape)
            pert = as_dict(fn(x2))
            for k, y in base.items():
                a, b = np.asarray(y)[:t + 1], np.asarray(pert[k])[:t + 1]
                diff = ~np.isclose(a, b, atol=atol, rtol=0, equal_nan=True)
                if diff.any(): bad.append((int(t), k, int(np.argmax(diff.reshape(diff.shape[0], -1).any(1))))); break
    return bad


def assert_no_lookahead(fn, x, **kw):
    bad = find_lookahead(fn, x, **kw)
    if bad: raise LookAheadError(f'output depends on future input samples: {bad[:5]}')


def streamable_table():
    """{operator: (status, reason)} with default options, for documentation."""
    from .operators import CATALOG
    names = set(CATALOG) | set(POINTWISE) | set(STREAMABLE) | set(NON_CAUSAL)
    return {n: classify(n) for n in sorted(names)}
