"""Authoring layer: build, validate, write and explain Khoros workspaces from a declarative spec.

Spec (YAML or JSON)::

    name: trend_ema_vol
    mode: stream                 # stream (causal, StreamingExecutor) or batch
    description: EMA crossover with a volatility filter
    inputs: [price]              # STREAM:kinput glyphs
    outputs: {position: gated.o} # STREAM:koutput glyphs: name -> node.port
    nodes:
      - id: fast
        op: kema                 # toolbox optional when the name is unique in the catalog
        params: {span: 10}
        rationale: fast trend estimate
    edges:
      - price -> fast.i          # source: input name or node[.port] (port defaults to o)
      - fast -> cross.i1
    synthetic: {kind: gbm, n: 1500, seed: 1}   # optional: data for `verify`

API: ``load_spec``, ``validate`` -> Report, ``build_wk``, ``explain``, ``run_stream``, ``run_batch``,
``verify`` (stream == batch replay + look-ahead check). CLI: ``python -m khoros_cantata.cli author ...``.
"""
from __future__ import annotations
import collections, json, math, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from .catalog_gen import CATALOG_JSON, eval_lag
from . import streaming as _st

_ID = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def load_catalog(path=CATALOG_JSON):
    ops = json.loads(Path(path).read_text())['operators']; by = collections.defaultdict(list)
    for o in ops: by[o['name']].append(o)
    return by


CAT = load_catalog()


def load_spec(src) -> dict:
    if isinstance(src, dict): return src
    p = Path(src); text = p.read_text()
    if p.suffix in ('.yaml', '.yml'):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


@dataclass
class Issue:
    level: str; where: str; message: str
    def __str__(self): return f'{self.level.upper()} [{self.where}] {self.message}'


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    node_lag: dict[str, float | None] = field(default_factory=dict)
    output_lag: dict[str, float | None] = field(default_factory=dict)
    resolved: dict[str, dict] = field(default_factory=dict)  # node id -> catalog entry
    edges: list[tuple[str, str, str, str]] = field(default_factory=list)  # (src, sport, dst, dport); src may be input
    order: list[str] = field(default_factory=list)
    @property
    def errors(self): return [i for i in self.issues if i.level == 'error']
    @property
    def warnings(self): return [i for i in self.issues if i.level == 'warning']
    @property
    def ok(self): return not self.errors
    def summary(self):
        lines = [f"{'VALID' if self.ok else 'INVALID'}: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        lines += [f'  {i}' for i in self.issues]
        for k, v in self.output_lag.items(): lines.append(f'  lag[{k}] = {_fmt_lag(v)} samples')
        return '\n'.join(lines)


def _fmt_lag(v): return 'unknown' if v is None else ('unbounded' if v == math.inf else f'{v:.2f}')


def _parse_ref(ref, default_port='o'):
    ref = str(ref).strip()
    return tuple(ref.split('.', 1)) if '.' in ref else (ref, default_port)


def _check_param(nid, op, k, v, entry, rep):
    spec = entry['params'].get(k)
    if spec is None:
        rep.issues.append(Issue('error', nid, f"unknown parameter -{k} for {op} (known: {', '.join(entry['params']) or 'none'})")); return
    t = spec.get('type')
    if t in ('number', 'integer'):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            rep.issues.append(Issue('error', nid, f'-{k} must be a {t}, got {v!r}')); return
        if t == 'integer' and float(v) != int(v): rep.issues.append(Issue('error', nid, f'-{k} must be an integer, got {v!r}'))
        r = spec.get('range')
        if r and not (r[0] <= v <= r[1]): rep.issues.append(Issue('error', nid, f'-{k}={v} outside valid range {r}'))
    elif t == 'string' and not isinstance(v, str):
        rep.issues.append(Issue('error', nid, f'-{k} must be a string, got {v!r}'))
    ch = spec.get('choices')
    if ch and v not in ch: rep.issues.append(Issue('error', nid, f'-{k}={v!r} not one of {ch}'))


def _resolve(nid, node, rep):
    op = str(node.get('op', '')); cands = CAT.get(op, [])
    tb = node.get('toolbox')
    if tb: cands = [c for c in cands if c['toolbox'].upper() == tb.upper()]
    if not cands:
        near = [n for n in CAT if n.startswith(op[:3])][:6]
        rep.issues.append(Issue('error', nid, f"unknown operator {op!r}" + (f' (did you mean: {", ".join(near)})' if near else ''))); return None
    if len(cands) > 1:
        rep.issues.append(Issue('error', nid, f"operator {op!r} exists in toolboxes {[c['toolbox'] for c in cands]}; set toolbox")); return None
    return cands[0]


def validate(spec) -> Report:
    spec = load_spec(spec); rep = Report(); mode = spec.get('mode', 'stream')
    if mode not in ('stream', 'batch'): rep.issues.append(Issue('error', 'spec', f'mode must be stream or batch, got {mode!r}'))
    inputs = list(spec.get('inputs') or []); nodes = spec.get('nodes') or []; ids = []
    if not nodes: rep.issues.append(Issue('error', 'spec', 'no nodes'))
    for name in inputs:
        if not _ID.match(str(name)): rep.issues.append(Issue('error', 'inputs', f'bad input name {name!r}'))
    for n in nodes:
        nid = str(n.get('id', ''))
        if not _ID.match(nid): rep.issues.append(Issue('error', nid or '?', 'node id must be an identifier')); continue
        if nid in ids or nid in inputs: rep.issues.append(Issue('error', nid, 'duplicate id (nodes and inputs share a namespace)')); continue
        ids.append(nid); entry = _resolve(nid, n, rep)
        if entry is None: continue
        if entry['toolbox'] == 'STREAM' and entry['name'] in ('kinput', 'koutput'):
            rep.issues.append(Issue('error', nid, 'declare stream I/O with inputs/outputs, not as nodes')); continue
        rep.resolved[nid] = entry
        for k, v in (n.get('params') or {}).items():
            if entry['name'] == 'krolling' and k not in ('op', 'window', 'tb'): continue  # inner operator options
            _check_param(nid, entry['name'], k, v, entry, rep)
        if entry['name'] == 'krolling':
            inner = (n.get('params') or {}).get('op')
            if not inner or inner not in CAT: rep.issues.append(Issue('error', nid, f'krolling needs -op naming a catalog operator, got {inner!r}'))
            else:
                ie = CAT[inner][0]
                for k, v in (n.get('params') or {}).items():
                    if k not in ('op', 'window', 'tb'): _check_param(nid, inner, k, v, ie, rep)
        if not n.get('rationale'): rep.issues.append(Issue('warning', nid, 'no rationale given (explanations will be thin)'))
    # edges
    connected = collections.defaultdict(dict)
    for e in spec.get('edges') or []:
        m = re.match(r'^\s*(\S+)\s*->\s*(\S+)\s*$', str(e))
        if not m: rep.issues.append(Issue('error', 'edges', f'bad edge {e!r} (use "src[.port] -> dst.port")')); continue
        (s, sp), (d, dp) = _parse_ref(m.group(1)), _parse_ref(m.group(2), None)
        if s in inputs: sp = 'o'
        elif s not in rep.resolved:
            if s not in ids: rep.issues.append(Issue('error', 'edges', f'unknown source {s!r} in {e!r}'))
            continue
        elif sp not in [o['port'] for o in rep.resolved[s]['outputs']]:
            rep.issues.append(Issue('error', 'edges', f"{s} has no output port {sp!r} (has {[o['port'] for o in rep.resolved[s]['outputs']]})")); continue
        if d not in rep.resolved:
            if d not in ids: rep.issues.append(Issue('error', 'edges', f'unknown target {d!r} in {e!r}'))
            continue
        ports = [i['port'] for i in rep.resolved[d]['inputs']]
        if dp is None: dp = ports[0] if len(ports) == 1 else None
        if dp not in ports and rep.resolved[d]['name'] != 'krolling':
            rep.issues.append(Issue('error', 'edges', f'{d} ({rep.resolved[d]["name"]}) has no input port {dp!r} (has {ports})')); continue
        if dp in connected[d]: rep.issues.append(Issue('error', 'edges', f'{d}.{dp} connected twice')); continue
        connected[d][dp] = (s, sp); rep.edges.append((s, sp, d, dp))
    for nid, entry in rep.resolved.items():
        for i in entry['inputs']:
            if not i['optional'] and i['port'] not in connected[nid] and entry['name'] != 'krolling':
                rep.issues.append(Issue('error', nid, f"required input {i['port']} is not connected"))
        if entry['name'] == 'krolling' and not connected[nid]: rep.issues.append(Issue('error', nid, 'krolling has no input'))
    # outputs
    outs = spec.get('outputs') or {}
    if not outs: rep.issues.append(Issue('error', 'outputs', 'no outputs declared'))
    for oname, ref in outs.items():
        s, sp = _parse_ref(ref)
        if not _ID.match(str(oname)): rep.issues.append(Issue('error', 'outputs', f'bad output name {oname!r}'))
        if s in inputs: continue
        if s not in rep.resolved: rep.issues.append(Issue('error', 'outputs', f'{oname}: unknown node {s!r}')); continue
        if sp not in [o['port'] for o in rep.resolved[s]['outputs']]: rep.issues.append(Issue('error', 'outputs', f'{oname}: {s} has no port {sp!r}'))
    # cycles / order
    indeg = {n: 0 for n in rep.resolved}; down = collections.defaultdict(list)
    for s, _, d, _ in rep.edges:
        if s in rep.resolved: indeg[d] += 1; down[s].append(d)
    q = collections.deque(n for n, k in indeg.items() if k == 0)
    while q:
        n = q.popleft(); rep.order.append(n)
        for m in down[n]:
            indeg[m] -= 1
            if indeg[m] == 0: q.append(m)
    if len(rep.order) != len(rep.resolved):
        rep.issues.append(Issue('error', 'graph', f'cycle among {sorted(set(rep.resolved) - set(rep.order))}'))
    used = {s for s, *_ in rep.edges} | {_parse_ref(r)[0] for r in outs.values()}
    for nid in rep.resolved:
        if nid not in used: rep.issues.append(Issue('warning', nid, 'output is never used'))
    # causality
    if mode == 'stream':
        for n in nodes:
            nid = str(n.get('id')); entry = rep.resolved.get(nid)
            if not entry or entry['name'] == 'krolling': continue
            try:
                status, why = _st.classify(entry['name'], n.get('params') or {}, set(connected[nid]), entry['toolbox'])
            except Exception as exc:  # bad parameter values already reported above
                if not any(i.where == nid and i.level == 'error' for i in rep.issues):
                    rep.issues.append(Issue('error', nid, f'operator rejected parameters: {exc}'))
                continue
            if status != 'streamable':
                rep.issues.append(Issue('error', nid, f"{entry['name']} is {status} in a streaming graph: {why}. "
                                                      f"Wrap it: op: krolling, params: {{op: {entry['name']}, window: N}}"))
    # lag
    nodespec = {str(n.get('id')): n for n in nodes}
    for nid in rep.order:
        entry = rep.resolved[nid]; p = {k: v['default'] for k, v in entry['params'].items() if v.get('default') is not None}
        p.update(nodespec[nid].get('params') or {})
        if entry['name'] == 'krolling':
            own = None
        else:
            try: own = eval_lag(entry.get('lag'), p)
            except Exception: own = None
            if entry['name'] == 'krunstats' and not p.get('window'): own = math.inf
        ups = [0.0 if s in inputs else rep.node_lag.get(s) for s, _, d, _ in rep.edges if d == nid]
        if own is None or any(u is None for u in ups): rep.node_lag[nid] = None
        else: rep.node_lag[nid] = own + (max(ups) if ups else 0.0)
        if rep.node_lag[nid] is None and entry['name'] != 'krolling':
            rep.issues.append(Issue('info', nid, f"lag of {entry['name']} is not modelled"))
    for oname, ref in outs.items():
        s, _ = _parse_ref(ref); rep.output_lag[oname] = 0.0 if s in inputs else rep.node_lag.get(s)
    return rep


def _opt(v):
    if isinstance(v, bool): return '1' if v else '0'
    if isinstance(v, str): return "'" + v.replace("'", '') + "'"
    if isinstance(v, (list, tuple)): return "'" + repr(list(v)) + "'"
    return repr(v)


def build_wk(spec, report: Report | None = None) -> str:
    """Return .wk text. Raises ValueError if the spec does not validate."""
    spec = load_spec(spec); rep = report or validate(spec)
    if not rep.ok: raise ValueError('spec does not validate:\n' + rep.summary())
    ids = {}; L = ['# Khoros Visual Programming Workspace', f"# generated by khoros_cantata.authoring from spec '{spec.get('name', '')}'",
                    f"# mode: {spec.get('mode', 'stream')}"]
    for line in str(spec.get('description', '')).splitlines(): L.append('# ' + line)
    L += ['VariablesBegin:', 'VariablesEnd:', '', '#GlyphBegin']
    k = 1
    for name in spec.get('inputs') or []:
        ids[name] = k; L += [f"# Glyph '{name}': stream input", f"Glyph:STREAM:kinput::{k}:20:{40 * k}:: -name '{name}'"]; k += 1
    nodespec = {str(n['id']): n for n in spec['nodes']}
    for nid in rep.order:
        n = nodespec[nid]; e = rep.resolved[nid]; ids[nid] = k
        lag = rep.node_lag.get(nid)
        L.append(f"# Glyph '{nid}' ({e['toolbox']}:{e['name']}): {e['description']}")
        if n.get('rationale'): L += ['#   why: ' + ln for ln in str(n['rationale']).splitlines()]
        L.append(f"#   cumulative lag: {_fmt_lag(lag)} samples")
        opts = ''.join(f' -{p} {_opt(v)}' for p, v in (n.get('params') or {}).items())
        L.append(f"Glyph:{e['toolbox']}:{e['name']}::{k}:{120 + 100 * (rep.order.index(nid) % 6)}:{40 * k}::{opts}"); k += 1
    for oname in spec.get('outputs') or {}:
        ids['__out__' + oname] = k; L += [f"# Glyph '{oname}': stream output", f"Glyph:STREAM:koutput::{k}:800:{40 * k}:: -name '{oname}'"]; k += 1
    L += ['#GlyphEnd', '#ConnectionBegin']
    for s, sp, d, dp in rep.edges: L.append(f'NodeConnection:{ids[s]}:{sp}:{ids[d]}:{dp}:')
    for oname, ref in spec['outputs'].items():
        s, sp = _parse_ref(ref); L.append(f"NodeConnection:{ids[s]}:{'o' if s in (spec.get('inputs') or []) else sp}:{ids['__out__' + oname]}:i:")
    L += ['#ConnectionEnd']
    return '\n'.join(L) + '\n'


def mermaid(spec, rep):
    spec = load_spec(spec); L = ['flowchart LR']
    for name in spec.get('inputs') or []: L.append(f'  {name}(["{name}"])')
    nodespec = {str(n['id']): n for n in spec['nodes']}
    for nid in rep.order:
        e = rep.resolved[nid]; ps = ' '.join(f'{k}={v}' for k, v in (nodespec[nid].get('params') or {}).items())
        L.append(f'  {nid}["{nid}<br/>{e["name"]} {ps}"]')
    for oname in spec.get('outputs') or {}: L.append(f'  out_{oname}(["{oname}"])')
    for s, sp, d, dp in rep.edges: L.append(f'  {s} -- "{sp}→{dp}" --> {d}')
    for oname, ref in spec['outputs'].items(): L.append(f'  {_parse_ref(ref)[0]} --> out_{oname}')
    return '\n'.join(L)


def explain(spec, report: Report | None = None, verification: dict | None = None) -> str:
    spec = load_spec(spec); rep = report or validate(spec); nodespec = {str(n['id']): n for n in spec.get('nodes') or []}
    L = [f"# Workspace `{spec.get('name', 'unnamed')}`", '', str(spec.get('description', '')).strip(), '',
         f"**Mode:** {spec.get('mode', 'stream')} · **Inputs:** {', '.join(spec.get('inputs') or []) or 'none'} · "
         f"**Outputs:** {', '.join(spec.get('outputs') or {})}", '', '## Graph', '', '```mermaid', mermaid(spec, rep) if rep.resolved else '', '```', '',
         '## Nodes', '', '| node | operator | parameters | why | streamable | own lag | cumulative lag |', '|---|---|---|---|---|---|---|']
    for nid in rep.order:
        e = rep.resolved[nid]; n = nodespec[nid]; ps = ', '.join(f'-{k} {v}' for k, v in (n.get('params') or {}).items()) or '—'
        st = 'adapter (causal)' if e['name'] == 'krolling' else 'yes' if e['streamable'] else 'no'
        L.append(f"| {nid} | {e['toolbox']}:{e['name']} — {e['description']} | {ps} | {n.get('rationale', '')} | {st} | {e['lag_formula']} | {_fmt_lag(rep.node_lag.get(nid))} |")
    L += ['', '## Latency', '']
    for k, v in rep.output_lag.items(): L.append(f'- `{k}`: approximately **{_fmt_lag(v)} samples** behind the input (sum of group delays along the slowest path).')
    L += ['', 'Lag figures are low-frequency group-delay approximations; true delay depends on the signal spectrum.', '',
          '## Caveats', '']
    seen = set()
    for nid in rep.order:
        for c in rep.resolved[nid]['caveats']:
            if (rep.resolved[nid]['name'], c) not in seen: seen.add((rep.resolved[nid]['name'], c)); L.append(f"- {rep.resolved[nid]['name']}: {c}")
    if not seen: L.append('- none recorded')
    L += ['', '## Validation', '', '```', rep.summary(), '```']
    if verification:
        L += ['', '## Verification (synthetic replay)', '']
        for k, v in verification.items(): L.append(f'- {k}: {v}')
    return '\n'.join(L) + '\n'

# ---------------------------------------------------------------- execution & verification
def synthetic_feeds(spec, n=None, seed=None):
    spec = load_spec(spec); syn = spec.get('synthetic') or {}; n = int(n or syn.get('n', 1500)); seed = syn.get('seed', 0) if seed is None else seed
    g = np.random.default_rng(seed); kind = syn.get('kind', 'gbm'); names = list(spec.get('inputs') or [])
    if kind == 'gbm':
        regime = np.where((np.arange(n) // max(n // 5, 1)) % 2 == 0, 0.008, 0.025)
        drift = 0.0008 * np.sin(2 * np.pi * np.arange(n) / max(n / 3, 1))
        return {k: 100 * np.exp(np.cumsum(drift + g.normal(0, 1, n) * regime)) * (1 + 0.1 * j) for j, k in enumerate(names)}
    if kind == 'pair':
        x = 50 + np.cumsum(g.normal(0, 0.5, n)); spread = np.zeros(n)
        for t in range(1, n): spread[t] = 0.95 * spread[t - 1] + g.normal(0, 0.3)
        beta = 1.5 + 0.2 * np.sin(2 * np.pi * np.arange(n) / n)
        return {names[0]: x, names[1]: 2.0 + beta * x + spread}
    raise ValueError(f'unknown synthetic kind {kind!r}')


def run_stream(spec, feeds, chunk=1):
    return _st.StreamingExecutor(_parse_text(build_wk(spec))).run(feeds, chunk=chunk)


def run_batch(spec, feeds):
    return _st.batch_execute(_parse_text(build_wk(spec)), feeds)


def _parse_text(text):
    import tempfile, os
    from .workspace_runtime import HistoricalWorkspaceParser
    with tempfile.NamedTemporaryFile('w', suffix='.wk', delete=False) as f: f.write(text)
    try: return HistoricalWorkspaceParser().parse(f.name)
    finally: os.unlink(f.name)


def verify(spec, feeds=None, chunk=7, atol=1e-9):
    """Replay feeds in streaming mode and batch mode; check equality and look-ahead.
    Returns a dict of results; raises nothing (inspect ``passed``)."""
    spec = load_spec(spec); feeds = feeds or synthetic_feeds(spec); names = list(spec.get('inputs') or [])
    s = run_stream(spec, feeds, chunk=chunk); b = run_batch(spec, feeds)
    diffs = {k: float(np.nanmax(np.abs(np.asarray(s[k], float) - np.asarray(b[k], float)))) if np.isfinite(np.asarray(s[k], float)).any() else 0.0 for k in s}
    nan_mismatch = [k for k in s if not np.array_equal(np.isnan(np.asarray(s[k], float)), np.isnan(np.asarray(b[k], float)))]
    X = np.stack([np.asarray(feeds[k], float) for k in names], 1)
    ex = _st.StreamingExecutor(_parse_text(build_wk(spec)))
    def f(x):
        ex.reset(); return ex.run({k: np.abs(x[:, j]) + 1e-6 for j, k in enumerate(names)}, chunk=64)
    la = _st.find_lookahead(f, X, trials=2)
    rep = validate(spec)
    return {'samples': int(X.shape[0]), 'stream_vs_batch_max_abs_diff': max(diffs.values()) if diffs else 0.0,
            'nan_pattern_mismatch': nan_mismatch, 'lookahead_violations': la,
            'output_lag_samples': {k: _fmt_lag(v) for k, v in rep.output_lag.items()},
            'passed': (max(diffs.values()) if diffs else 0) <= atol and not nan_mismatch and not la, 'outputs': s}
