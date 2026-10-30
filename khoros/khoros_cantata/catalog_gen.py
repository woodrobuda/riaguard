"""Generate the machine-readable operator catalog (catalog.json) and docs/CATALOG.md.

    python -m khoros_cantata.catalog_gen          # rewrite both files
    python -m khoros_cantata.catalog_gen --check  # exit 1 if they are out of date

Sources: the live NativeOperatorRegistry (names, toolboxes, ports), parameter names and
defaults introspected from ``operators.py`` / ``ops_finance.py`` source, streamability from
``streaming.classify`` and hand annotations from ``catalog_annotations.py``.
"""
from __future__ import annotations
import ast, json, re, sys
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parent
CATALOG_JSON = PKG / 'catalog.json'
CATALOG_MD = PKG.parent / 'docs' / 'CATALOG.md'

_UNARY_DEFAULTS = ['kneg', 'kabs', 'ksqr', 'ksqrt', 'krecip', 'kcbrt', 'kcube', 'kfloor', 'ktruncate', 'kbitnot',
                   'kclip', 'kclipout', 'kmegaclip', 'identity']
_LOOP_BINARY = ['kadd', 'ksub', 'kmul', 'kdiv', 'kdivinto', 'kpow', 'kmin', 'kmax', 'kabsdiff']
_LOOP_UNARY = ['kexp', 'klog', 'klog10', 'ksin', 'kcos', 'ktan', 'katan', 'kceil', 'ksign']
_FIN_INPUT_DOC = {'kkalmanpair': {'i1': 'x (independent leg)', 'i2': 'y (dependent leg)'}}


def _lit(s):
    try: return ast.literal_eval(s)
    except Exception: return None


def _type_of(v):
    return {bool: 'integer', int: 'integer', float: 'number', str: 'string', list: 'array', tuple: 'array'}.get(type(v), 'number')


def _source_params():
    """name -> {param: default} from p.get(...) / p[...] usage in registration lines."""
    out = {}
    def scan(text):
        ps = {}
        for k, d in re.findall(r"p\.get\('(\w+)'(?:,\s*([^()]*?(?:\([^()]*\))?[^(),]*))?\)", text):
            ps.setdefault(k, _lit(d) if d else None)
        for k in re.findall(r"p\['(\w+)'\]", text): ps.setdefault(k, None)
        if '_const_or(b, p)' in text: ps.setdefault('real', 0)
        if "_const_or(b, p, 'value')" in text: ps.setdefault('value', 0)
        if '_off(p)' in text: ps.update({k: 0 for k in ('woff', 'hoff', 'doff', 'toff', 'eoff')})
        if '_sz(p' in text or '_shape(p)' in text: ps.update({k: None for k in ('wsize', 'hsize', 'dsize', 'tsize', 'esize')})
        if "for a, k in enumerate('whdte') if p.get(k)" in text: ps.update({k: False for k in 'whdte'})
        ps = {k: v for k, v in ps.items() if k != 'k'}
        return ps
    src = (PKG / 'operators.py').read_text()
    for line in src.splitlines():
        m = re.match(r"\s+r\(\w+, '(\w+)',(.*)$", line)
        if m: out[m.group(1)] = scan(m.group(2))
    loop_b = re.search(r"r\(DM, 'k' \+ op,(.*)", src).group(1); loop_u = re.search(r"r\(DM, 'k' \+ fn,(.*)", src).group(1)
    for n in _LOOP_BINARY: out[n] = scan(loop_b)
    for n in _LOOP_UNARY: out[n] = {}
    for n in ('kerode', 'kdilate', 'kopen', 'kclose'): pass
    mop = re.search(r"r\(IM, 'i' \+ mop,(.*)", src).group(1)
    for n in ('ierode', 'idilate', 'iopen', 'iclose'): out[n] = scan(mop)
    for seg in ('val', 'mask', 'map', 'loc', 'time'): out['kasc2' + seg] = {}
    fin = (PKG / 'ops_finance.py').read_text()
    for name, body in re.findall(r"'(\w+)': lambda p: (.*)", fin): out[name] = scan(body)
    out['kclip'] = {'lc': None, 'uc': None}; out['kclipout'] = {'lc': None, 'uc': None}
    out['kinput'] = {'name': None}; out['koutput'] = {'name': None}; out['krolling'] = {'op': None, 'window': 32, 'tb': '*'}
    return out


def build_catalog():
    from .workspace_runtime import NativeOperatorRegistry
    from .streaming import classify, streaming_registry
    from .catalog_annotations import DESC, DETAIL
    reg = streaming_registry(registry=NativeOperatorRegistry()); params = _source_params(); ops = []
    for (tb, name), op in sorted(reg._ops.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if tb == '*' and name != 'identity': continue
        ins = list(op.inputs) or (['i'] if name in _UNARY_DEFAULTS else ['i1', 'i2'] if name in _LOOP_BINARY else ['i'])
        if name == 'kinput': ins = []
        outs = list(op.outputs)
        det = DETAIL.get(name, {})
        ps = {}
        for k, d in params.get(name, {}).items():
            ps[k] = {'type': _type_of(d) if d is not None else 'number', 'default': d, 'meaning': ''}
        for k, v in det.get('params', {}).items(): ps[k] = {**ps.get(k, {}), **v}
        status, why = classify(name, {}, (), tb) if name not in ('kinput', 'koutput', 'krolling') else ('streamable', 'stream I/O / adapter')
        conditional = {'kiir': 'not with -zerophase 1', 'kfir': '-mode causal with taps from options (i2 unconnected)',
                       'klinearop': '-w0 0, no -ext, kernel from -n', 'kconvert': 'not with -norm 1'}.get(name)
        pointwise = status == 'streamable' and not det.get('lag') and name not in DETAIL
        binary_const = name in _LOOP_BINARY or name in ('karith2', 'kcompare', 'kbitwise', 'kcross')
        ops.append({
            'name': name, 'toolbox': tb, 'description': DESC.get(name, ''),
            'inputs': [{'port': p, 'meaning': _FIN_INPUT_DOC.get(name, {}).get(p, ''),
                        'optional': bool(p == 'i2' and binary_const)} for p in ins],
            'outputs': [{'port': p} for p in outs],
            'params': ps,
            'streamable': status == 'streamable', 'streaming_status': status, 'streaming_note': conditional or why,
            'lag': det.get('lag', '0' if pointwise else None), 'lag_formula': det.get('lag_formula', '0 (pointwise)' if pointwise else 'n/a'),
            'cost': det.get('cost', 'O(1) per sample' if pointwise else 'O(N) per record (see implementation)'),
            'finance_uses': det.get('finance', []), 'caveats': det.get('caveats', []),
        })
    return {'version': 1, 'generated_by': 'khoros_cantata.catalog_gen', 'operators': ops}


def render_markdown(cat):
    L = ['# Operator catalog', '', f'Generated by `python -m khoros_cantata.catalog_gen` from the registry and '
         '`catalog_annotations.py`; machine-readable form: `khoros_cantata/catalog.json`. '
         f'{len(cat["operators"])} operators. Lag is the approximate low-frequency group delay in samples.', '',
         '| operator | toolbox | streamable | lag | description |', '|---|---|---|---|---|']
    for o in cat['operators']:
        st = 'yes' if o['streamable'] else 'no'
        if o['streaming_note'] and o['streamable'] and o['name'] in ('kiir', 'kfir', 'klinearop', 'kconvert'): st += f" ({o['streaming_note']})"
        L.append(f"| [{o['name']}](#{o['name']}) | {o['toolbox']} | {st} | {o['lag_formula']} | {o['description']} |")
    L += ['', '## Details', '']
    for o in cat['operators']:
        L += [f"### {o['name']}", '', f"**{o['toolbox']}** — {o['description']}", '',
              f"- Inputs: {', '.join(i['port'] + (' (optional)' if i['optional'] else '') + (' — ' + i['meaning'] if i['meaning'] else '') for i in o['inputs']) or 'none (source)'}",
              f"- Outputs: {', '.join(x['port'] for x in o['outputs'])}",
              f"- Streaming: {o['streaming_status']}" + (f" — {o['streaming_note']}" if o['streaming_note'] else ''),
              f"- Lag: {o['lag_formula']}; cost: {o['cost']}"]
        if o['params']:
            L.append('- Parameters:')
            for k, v in o['params'].items():
                extra = ''.join([f", range {v['range']}" if v.get('range') else '', f", choices {v['choices']}" if v.get('choices') else ''])
                L.append(f"  - `-{k}` ({v.get('type')}, default `{v.get('default')}`{extra}){': ' + v['meaning'] if v.get('meaning') else ''}")
        if o['finance_uses']: L.append(f"- Finance uses: {'; '.join(o['finance_uses'])}")
        if o['caveats']: L.append(f"- Caveats: {'; '.join(o['caveats'])}")
        L.append('')
    return '\n'.join(L) + '\n'


def _dump(cat): return json.dumps(cat, indent=1, sort_keys=False, default=str) + '\n'


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cat = build_catalog(); js, md = _dump(cat), render_markdown(cat)
    if '--check' in argv:
        ok = CATALOG_JSON.exists() and CATALOG_JSON.read_text() == js and CATALOG_MD.exists() and CATALOG_MD.read_text() == md
        print('catalog up to date' if ok else 'catalog OUT OF DATE: run python -m khoros_cantata.catalog_gen'); return 0 if ok else 1
    CATALOG_JSON.write_text(js); CATALOG_MD.write_text(md); print(f'wrote {CATALOG_JSON} and {CATALOG_MD} ({len(cat["operators"])} operators)'); return 0


# ---------------------------------------------------------------- lag evaluation (used by authoring)
def _kalman_lag(p):
    from .ops_finance import kalman_trend_gain
    k = kalman_trend_gain(p.get('ql', 1e-4), p.get('qs', 1e-6), p.get('r', 1e-2))[0]
    return (1 - k) / k


def _iir_lag(p):
    from . import ops_signal as S
    from scipy import signal as sg
    if p.get('ftype', 'lowpass') != 'lowpass': return 0.0
    sos = S.kiir_design(p.get('order', 4), p.get('cutoff', 0.25), 'lowpass', p.get('family', 'butter'), p.get('rp', 1.0), p.get('rs', 40.0))
    b, a = sg.sos2tf(sos); w, gd = sg.group_delay((b, a), w=[1e-4]); return float(gd[0])


def eval_lag(expr, params):
    if expr is None: return None
    return eval(expr, {'__builtins__': {}, '_kalman_lag': _kalman_lag, '_iir_lag': _iir_lag}, {'p': params})


if __name__ == '__main__':
    raise SystemExit(main())
