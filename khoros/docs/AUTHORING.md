# Authoring workspaces (for agents and humans)

The authoring layer turns a **declarative spec** into a validated, commented Khoros workspace and
an explanation report. It is the recommended way for an AI agent to create workspaces: the agent edits
the spec; the tooling checks it against the operator catalog and writes the `.wk`.

```bash
python -m khoros_cantata.cli author validate templates/trend_ema_vol.yaml
python -m khoros_cantata.cli author build    templates/trend_ema_vol.yaml -o trend.wk
python -m khoros_cantata.cli author explain  templates/trend_ema_vol.yaml --verify -o trend.md
python -m khoros_cantata.cli author verify   templates/trend_ema_vol.yaml   # stream==batch + look-ahead
```

## Spec format
See the docstring of `khoros_cantata/authoring.py` and the templates. Keys: `name`, `mode` (`stream` | `batch`),
`description`, `inputs` (list of stream names), `outputs` (`name: node.port`), `nodes` (`id`, `op`,
optional `toolbox`, `params`, `rationale`), `edges` (`"src[.port] -> dst.port"`; a source may be an input
name; port defaults to `o`), optional `synthetic` (`kind: gbm | pair`, `n`, `seed`) for verification.

## What `validate` checks
- operator exists (and is unambiguous) in `catalog.json`; suggestions for near names;
- parameters: unknown names, type, numeric range, allowed choices; `krolling` inner options are checked
  against the wrapped operator;
- ports: source output and target input exist; no input connected twice; required inputs connected
  (`i2` of binary arithmetic is optional and falls back to `-real`);
- graph: cycles; unused nodes (warning); missing rationale (warning);
- streaming mode: every node must be streamable with its parameters and connections (`streaming.classify`),
  otherwise an error suggesting the `krolling` adapter;
- **lag**: each node's low-frequency group delay from the catalog formula (e.g. EMA `(N-1)/2`, SMA `(n-1)/2`,
  FIR `(taps-1)/2`, IIR computed numerically, Kalman from the steady-state gain), accumulated along the
  slowest path to every output. Cumulative running statistics report `unbounded`; unmodelled operators `unknown`.

## Regenerating the catalog
After adding or changing an operator: add a line to `catalog_annotations.DESC` (and `DETAIL` for richer
metadata), then run `python -m khoros_cantata.catalog_gen`. `tests/test_catalog.py` fails if a registered
operator is missing or the files are stale.

## Limits
- The lag figure is an approximation at low frequency; it does not replace the look-ahead test.
- Bar construction (ticks → time/volume bars) is not a graph operator: it changes the sample rate,
  which the sample-synchronous streaming executor does not model. Build bars before `kinput`.
- Templates are illustrative signal pipelines on synthetic data, not validated trading strategies.
