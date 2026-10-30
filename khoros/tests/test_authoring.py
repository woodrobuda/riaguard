import copy, json, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
from khoros_cantata import authoring as A
from khoros_cantata.workspace_runtime import HistoricalWorkspaceParser

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / 'templates').glob('*.yaml'))
BASE = {'name': 't', 'mode': 'stream', 'inputs': ['price'], 'outputs': {'y': 'b.o'},
        'nodes': [{'id': 'a', 'op': 'kema', 'params': {'span': 9}, 'rationale': 'smooth'},
                  {'id': 'b', 'op': 'kcross', 'rationale': 'events'}],
        'edges': ['price -> a.i', 'price -> b.i1', 'a -> b.i2']}


def spec(**changes):
    s = copy.deepcopy(BASE); s.update(changes); return s


def errors(s): return [str(i) for i in A.validate(s).errors]


def test_base_valid_and_lag():
    r = A.validate(BASE); assert r.ok and not r.warnings and r.output_lag == {'y': 4.0}


@pytest.mark.parametrize('mutate,needle', [
    (lambda s: s['nodes'][0].update(op='kemma'), 'unknown operator'),
    (lambda s: s['nodes'][0]['params'].update(spam=1), 'unknown parameter -spam'),
    (lambda s: s['nodes'][0]['params'].update(span=0), 'outside valid range'),
    (lambda s: s['nodes'][0]['params'].update(span='ten'), 'must be a number'),
    (lambda s: s['nodes'].append({'id': 'c', 'op': 'krunstats', 'params': {'stat': 'median'}}), "not one of"),
    (lambda s: s['edges'].append('a.o9 -> b.i1'), 'no output port'),
    (lambda s: s['edges'].append('a -> b.i7'), 'no input port'),
    (lambda s: s['edges'].append('price -> b.i2'), 'connected twice'),
    (lambda s: s['edges'].remove('price -> a.i'), 'required input i is not connected'),
    (lambda s: s['edges'].append('ghost -> b.i1'), 'unknown source'),
    (lambda s: s['outputs'].update(z='a.o5'), 'has no port'),
    (lambda s: s['nodes'].append({'id': 'a', 'op': 'kema'}), 'duplicate id'),
    (lambda s: s.update(mode='live'), 'mode must be'),
    (lambda s: s['nodes'].append({'id': 'f', 'op': 'kfft'}) or s['edges'].append('price -> f.i'), 'non-causal'),
    (lambda s: s['nodes'].append({'id': 'g', 'op': 'kiir', 'params': {'zerophase': 1}}) or s['edges'].append('price -> g.i'), 'filtfilt'),
    (lambda s: s['nodes'].append({'id': 'h', 'op': 'mmul'}) or s['edges'] + ['price -> h.i1'], 'undeclared'),
])
def test_validation_errors(mutate, needle):
    s = copy.deepcopy(BASE); mutate(s)
    if any(n['id'] == 'h' for n in s['nodes']): s['edges'] += ['price -> h.i1', 'price -> h.i2']
    errs = errors(s); assert any(needle in e for e in errs), errs


def test_cycle_detected():
    s = spec(nodes=[{'id': 'a', 'op': 'kadd'}, {'id': 'b', 'op': 'kadd'}], edges=['a -> b.i1', 'b -> a.i1'], outputs={'y': 'a.o'})
    assert any('cycle' in e for e in errors(s))


def test_batch_mode_allows_non_causal_and_rolling_in_stream():
    s = spec(mode='batch', nodes=[{'id': 'f', 'op': 'knormal', 'rationale': 'x'}], edges=['price -> f.i'], outputs={'y': 'f.o'})
    assert A.validate(s).ok
    s = spec(nodes=[{'id': 'f', 'op': 'krolling', 'params': {'op': 'knormal', 'window': 30, 'method': 'zscore'}, 'rationale': 'x'}],
             edges=['price -> f.i'], outputs={'y': 'f.o'})
    r = A.validate(s); assert r.ok and r.output_lag['y'] is None
    v = A.verify(s, {'price': 100 + np.cumsum(np.random.default_rng(0).normal(size=300))}); assert v['passed']
    bad = copy.deepcopy(s); bad['nodes'][0]['params']['spam'] = 1
    assert any('unknown parameter -spam for knormal' in e for e in errors(bad))  # inner options checked against the wrapped op
    bad['nodes'][0]['params'] = {'op': 'nope'}; assert any('krolling needs -op' in e for e in errors(bad))


def test_warnings_and_unbounded_lag():
    s = spec(nodes=[{'id': 'a', 'op': 'krunstats'}, {'id': 'b', 'op': 'kema'}], edges=['price -> a.i', 'price -> b.i'], outputs={'y': 'a.o'})
    r = A.validate(s); assert r.ok and r.output_lag['y'] == float('inf')
    assert {w.where for w in r.warnings} >= {'a', 'b'}  # missing rationale, b unused


def test_build_roundtrip_and_comments():
    text = A.build_wk(BASE)
    assert "#   why: smooth" in text and "Glyph:SIGNAL:kema::" in text and "-span 9" in text
    hws = A._parse_text(text)
    assert [g.oname for g in hws.glyphs] == ['kinput', 'kema', 'kcross', 'koutput'] and len(hws.connections) == 4
    with pytest.raises(ValueError, match='does not validate'): A.build_wk(spec(mode='x'))


def test_explain_contains_sections():
    md = A.explain(BASE, verification={'passed': True})
    for needle in ('```mermaid', '| a | SIGNAL:kema', '4.00 samples', '## Caveats', 'VALID', 'passed: True'): assert needle in md


def test_json_spec_file(tmp_path):
    p = tmp_path / 's.json'; p.write_text(json.dumps(BASE)); assert A.validate(p).ok


@pytest.mark.parametrize('path', TEMPLATES, ids=lambda p: p.stem)
def test_templates_validate_verify_and_artifacts_in_sync(path):
    pytest.importorskip('yaml')
    r = A.validate(path); assert r.ok, r.summary()
    v = A.verify(path); assert v['passed'], {k: v[k] for k in v if k != 'outputs'}
    assert v['stream_vs_batch_max_abs_diff'] == 0.0 and v['lookahead_violations'] == []
    assert path.with_suffix('.wk').read_text() == A.build_wk(path), 'regenerate: cli author build'
    assert path.with_suffix('.md').exists()


def test_trend_template_behaviour():
    pytest.importorskip('yaml')
    p = ROOT / 'templates' / 'trend_ema_vol.yaml'; r = A.validate(p)
    assert r.output_lag['position'] == 19.5 and r.output_lag['volatility'] == 10.0
    o = A.run_stream(p, A.synthetic_feeds(p))
    assert set(np.unique(o['position'])) <= {-1, 0, 1} and (o['position'] == 0).any() and (o['position'] != 0).any()
    calm = o['volatility'] < 0.015; assert np.all(o['position'][~calm] == 0)


def test_pair_template_tracks_beta():
    pytest.importorskip('yaml')
    p = ROOT / 'templates' / 'meanrev_kalman_pair.yaml'; f = A.synthetic_feeds(p); o = A.run_stream(p, f)
    true = 1.5 + 0.2 * np.sin(2 * np.pi * np.arange(2000) / 2000)
    assert np.median(np.abs(o['beta'][500:] - true[500:])) < 0.1
    assert set(np.unique(o['position'])) <= {-1, 0, 1}


def test_cli_author(tmp_path):
    p = tmp_path / 's.json'; p.write_text(json.dumps(BASE)); run = lambda *a: subprocess.run([sys.executable, '-m', 'khoros_cantata.cli', 'author', *a], capture_output=True, text=True)
    assert run('validate', str(p)).returncode == 0
    r = run('build', str(p), '-o', str(tmp_path / 'o.wk')); assert r.returncode == 0 and (tmp_path / 'o.wk').exists()
    r = run('explain', str(p)); assert r.returncode == 0 and '# Workspace' in r.stdout
    r = run('verify', str(p)); assert r.returncode == 0 and '"passed": true' in r.stdout
    bad = tmp_path / 'b.json'; bad.write_text(json.dumps(spec(mode='x'))); assert run('validate', str(bad)).returncode == 1
    assert run('validate', str(tmp_path / 'missing.yaml')).returncode == 2
