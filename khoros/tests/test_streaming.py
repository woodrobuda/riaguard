import runpy
from pathlib import Path
import numpy as np
import pytest
from khoros_cantata import ops_signal as S
from khoros_cantata.streaming import (StreamingExecutor, NonCausalGraphError, batch_execute, find_lookahead,
                                      assert_no_lookahead, LookAheadError, classify, POINTWISE, STREAMABLE)
from khoros_cantata.workspace_runtime import NativeOperatorRegistry

ROOT = Path(__file__).resolve().parents[1]
X = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, 240)))
Y = np.random.default_rng(4).normal(size=240)


def wk(tmp_path, glyphs, conns, name='w.wk'):
    lines = ['#GlyphBegin'] + [f'Glyph:{tb}:{op}::{i}:0:0:: {cmd}' for i, (tb, op, cmd) in glyphs.items()] + ['#GlyphEnd', '#ConnectionBegin']
    lines += [f'NodeConnection:{a}:{ap}:{b}:{bp}:' for a, ap, b, bp in conns] + ['#ConnectionEnd']
    p = tmp_path / name; p.write_text('\n'.join(lines) + '\n'); return p


def unary_wk(tmp_path, tb, op, cmd):
    return wk(tmp_path, {1: ('STREAM', 'kinput', "-name 'x'"), 2: (tb, op, cmd), 3: ('STREAM', 'koutput', "-name 'y'")},
              [(1, 'o', 2, 'i'), (2, 'o', 3, 'i')])


CAUSAL_UNARY = [
    ('SIGNAL', 'kema', '-span 12'), ('SIGNAL', 'kema', '-alpha 0.3'), ('SIGNAL', 'kiir', '-order 4 -cutoff 0.1'),
    ('SIGNAL', 'kiir', "-order 3 -cutoff 0.2 -family 'cheby1'"), ('SIGNAL', 'kfir', '-taps 21 -cutoff 0.1'),
    ('DATAMANIP', 'klinearop', '-n 5 -w0 0'), ('SIGNAL', 'krunstats', "-stat 'mean'"), ('SIGNAL', 'krunstats', "-stat 'std'"),
    ('SIGNAL', 'krunstats', "-stat 'max' -window 15"), ('SIGNAL', 'krunstats', "-stat 'var' -window 7"),
    ('SIGNAL', 'kdelay', '-n 3'), ('SIGNAL', 'kdiff', ''), ('DATAMANIP', 'kclip', '-lc 99 -uc 101'),
    ('DATAMANIP', 'karith1', "-op 'log'"), ('DATAMANIP', 'kmul', '-real 2'), ('DATAMANIP', 'ksub', '-real 100'),
    ('DATAMANIP', 'kcompare', "-op 'gt' -real 100"), ('DATAMANIP', 'kneg', ''), ('DATAMANIP', 'kabs', ''),
]


@pytest.mark.parametrize('tb,op,cmd', CAUSAL_UNARY)
@pytest.mark.parametrize('chunk', [1, 7, 240])
def test_stream_equals_batch_unary(tmp_path, tb, op, cmd, chunk):
    p = unary_wk(tmp_path, tb, op, cmd)
    batch = batch_execute(p, {'x': X})['y']
    stream = StreamingExecutor(p).run({'x': X}, chunk=chunk)['y']
    assert stream.shape == batch.shape and np.allclose(stream, batch, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('tb,op,cmd', CAUSAL_UNARY)
def test_causal_ops_have_no_lookahead(tmp_path, tb, op, cmd):
    ex = StreamingExecutor(unary_wk(tmp_path, tb, op, cmd))
    def f(x): ex.reset(); return ex.run({'x': x}, chunk=16)['y']
    assert_no_lookahead(f, X)


def test_binary_ops_and_fanout_stream_equals_batch(tmp_path):
    p = wk(tmp_path, {1: ('STREAM', 'kinput', "-name 'a'"), 2: ('STREAM', 'kinput', "-name 'b'"),
                      3: ('DATAMANIP', 'kadd', ''), 4: ('DATAMANIP', 'kblend', '-alpha 0.25'), 5: ('DATAMANIP', 'kgate', '-value -1'),
                      6: ('SIGNAL', 'kcross', ''), 7: ('DATAMANIP', 'kmax', ''),
                      11: ('STREAM', 'koutput', "-name 'add'"), 12: ('STREAM', 'koutput', "-name 'blend'"),
                      13: ('STREAM', 'koutput', "-name 'gate'"), 14: ('STREAM', 'koutput', "-name 'cross'"), 15: ('STREAM', 'koutput', "-name 'max'")},
           [(1, 'o', 3, 'i1'), (2, 'o', 3, 'i2'), (1, 'o', 4, 'i1'), (2, 'o', 4, 'i2'), (1, 'o', 5, 'i1'), (2, 'o', 5, 'i2'),
            (2, 'o', 6, 'i1'), (3, 'o', 6, 'i2'), (3, 'o', 7, 'i1'), (2, 'o', 7, 'i2'),
            (3, 'o', 11, 'i'), (4, 'o', 12, 'i'), (5, 'o', 13, 'i'), (6, 'o', 14, 'i'), (7, 'o', 15, 'i')])
    feeds = {'a': Y, 'b': np.roll(Y, 5)}
    b = batch_execute(p, feeds); s = StreamingExecutor(p).run(feeds, chunk=3)
    assert set(b) == set(s) == {'add', 'blend', 'gate', 'cross', 'max'}
    for k in b: assert np.allclose(b[k], s[k]), k


def test_every_declared_streamable_operator_is_pointwise_or_tested():
    tested = {op for _, op, _ in CAUSAL_UNARY} | {'kadd', 'kblend', 'kgate', 'kcross', 'kmax'}
    stateful = set(STREAMABLE) - set(POINTWISE)
    from khoros_cantata.ops_finance import FACTORIES  # tested in test_ops_finance.py
    assert stateful <= tested | {'kconvert'} | set(FACTORIES)


@pytest.mark.parametrize('name', sorted(set(POINTWISE) - {'kreal2cmplx', 'kmapdata', 'kunmask', 'identity', 'kbitnot'}))
def test_pointwise_ops_stream_like_batch(name):
    reg = NativeOperatorRegistry(); op = reg.resolve('*', name)
    params = {'op': 'abs'} if name == 'karith1' else {'op': 'mag'} if name == 'kcmplx' else {'real': 2.0, 'op': 'add'} if name == 'karith2' else {'lc': -1, 'uc': 1} if name.startswith('kclip') else {}
    x = np.linspace(0.1, 0.9, 20)
    ins = {'i': x, 'i1': x, 'i2': x[::-1].copy()}
    whole = np.asarray(next(iter(op(params, ins).values())))
    parts = np.concatenate([np.atleast_1d(next(iter(op(params, {k: v[s:s + 3] for k, v in ins.items()}).values()))) for s in range(0, 20, 3)])
    assert np.allclose(whole, parts)


NON_CAUSAL_CASES = [('DATAMANIP', 'kfft', '-w'), ('SIGNAL', 'kiir', '-order 2 -cutoff 0.1 -zerophase 1'),
                    ('SIGNAL', 'kfir', "-taps 11 -mode 'same'"), ('SIGNAL', 'khilbert', ''), ('SIGNAL', 'kdwt', ''),
                    ('DATAMANIP', 'klinearop', '-n 5 -w0 2'), ('IMAGE', 'imedian', ''), ('DATAMANIP', 'knormal', ''),
                    ('DATAMANIP', 'khistops', ''), ('DATAMANIP', 'kresample', '-wsize 10'), ('SIGNAL', 'kwiener', ''),
                    ('DATAMANIP', 'kconvert', '-norm 1'), ('IMAGE', 'igauss_func', '')]


@pytest.mark.parametrize('tb,op,cmd', NON_CAUSAL_CASES)
def test_non_causal_graph_rejected(tmp_path, tb, op, cmd):
    with pytest.raises(NonCausalGraphError) as e:
        StreamingExecutor(unary_wk(tmp_path, tb, op, cmd))
    assert op in str(e.value) and 'krolling' in str(e.value)


def test_toolbox_scoped_declarations(tmp_path):
    from khoros_cantata.streaming import register_streamable, Pointwise, STREAMABLE, NON_CAUSAL
    register_streamable('kfoo', lambda p, op, c: Pointwise(op, p), toolbox='MYTB')
    try:
        assert classify('kfoo', toolbox='MYTB')[0] == 'streamable' and classify('kfoo')[0] == 'undeclared'
        NON_CAUSAL['MYTB:kbar'] = 'whole record'; assert classify('kbar', toolbox='mytb') == ('non-causal', 'whole record')
    finally:
        STREAMABLE.pop('MYTB:kfoo'); NON_CAUSAL.pop('MYTB:kbar', None)


def test_classification_reasons():
    assert classify('kiir', {'zerophase': 1})[0] == 'non-causal' and classify('kiir')[0] == 'streamable'
    assert classify('kfft')[0] == 'non-causal' and classify('mmul')[0] == 'undeclared'
    assert classify('kfir', {}, {'i2'})[0] == 'non-causal'


def test_lookahead_detector_catches_non_causal_batch_ops():
    for f in (lambda x: S.kiir(x, S.kiir_design(2, 0.1), zerophase=True), lambda x: S.khilbert(x),
              lambda x: S.kfir(x, np.ones(5) / 5, mode='same'), lambda x: (x - x.mean()) / x.std()):
        assert find_lookahead(f, X)
        with pytest.raises(LookAheadError): assert_no_lookahead(f, X)
    assert find_lookahead(lambda x: S.kema(x, 10), X) == []
    assert find_lookahead(lambda x: {'a': S.kdiff(x), 'b': np.roll(x, -1)}, X)[0][1] == 'b'


@pytest.mark.parametrize('inner', ["-op 'kiir' -order 2 -cutoff 0.1 -zerophase 1", "-op 'knormal' -method 'zscore'",
                                   "-op 'khilbert'", "-op 'kfir' -taps 9 -mode 'same'", "-op 'kwiener' -size 5"])
def test_rolling_wrapper_is_causal_and_matches_batch(tmp_path, inner):
    p = unary_wk(tmp_path, 'STREAM', 'krolling', inner + ' -window 32')
    ex = StreamingExecutor(p)
    def f(x): ex.reset(); return ex.run({'x': x}, chunk=10)['y']
    y = f(X)
    assert y.shape == X.shape and np.all(np.isfinite(y[31:]))
    assert_no_lookahead(f, X)
    assert np.allclose(batch_execute(p, {'x': X})['y'], y, equal_nan=True)


def test_rolling_wrapper_semantics():
    from khoros_cantata.streaming import RollingCausal
    op = NativeOperatorRegistry().resolve('*', 'knormal')
    rc = RollingCausal(op, {'method': 'minmax'}, 3)
    y = rc.step({'i': np.array([0., 10, 5, 20, 0])})['o']
    assert np.allclose(y, [0, 1, 0.5, 1, 0])  # each = last element of minmax over trailing 3
    with pytest.raises(ValueError): RollingCausal(op, {}, 0)


def test_push_api_and_errors(tmp_path):
    p = unary_wk(tmp_path, 'SIGNAL', 'kema', '-alpha 0.5')
    ex = StreamingExecutor(p); assert ex.inputs == ['x'] and ex.outputs == ['y']
    assert ex.push(x=2.0)['y'].tolist() == [2.0] and ex.push({'x': [4.0, 4.0]})['y'].tolist() == [3.0, 3.5]
    ex.reset(); assert ex.push(x=0.0)['y'].tolist() == [0.0]
    with pytest.raises(KeyError): ex.push()
    bad = wk(tmp_path, {1: ('STREAM', 'kinput', ''), 2: ('STREAM', 'krolling', "-op 'nope'")}, [(1, 'o', 2, 'i')], 'bad.wk')
    with pytest.raises(NonCausalGraphError, match='needs -name'): StreamingExecutor(bad)
    cyc = wk(tmp_path, {1: ('DATAMANIP', 'kadd', ''), 2: ('DATAMANIP', 'kadd', '')}, [(1, 'o', 2, 'i1'), (2, 'o', 1, 'i1')], 'cyc.wk')
    with pytest.raises(NonCausalGraphError, match='cycle'): StreamingExecutor(cyc)


def test_example_workspace_runs():
    sig = runpy.run_path(str(ROOT / 'examples' / 'streaming_ema_crossover.py'))['main']()
    assert len(sig) > 2 and {s[1] for s in sig} == {'BUY', 'SELL'}
    kinds = [s[1] for s in sig]; assert all(a != b for a, b in zip(kinds, kinds[1:]))  # alternates
