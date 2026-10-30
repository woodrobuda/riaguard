# khoros/ — snapshot of khoros-cantata-python (authoring branch)

This folder is a **point-in-time snapshot**, not a live mirror or submodule.

| | |
|---|---|
| Source repository | [woodrobuda/khoros-cantata-python](https://github.com/woodrobuda/khoros-cantata-python) |
| Upstream base (`main`) | `a9ba65c5ff3dbec544aa34dd022830f67babbca1` |
| Branch `operators` (PR #1, not merged) | `c8e72ef90e425d649d425d6e1d3c78c602956095` |
| Branch `authoring` (built on `operators`, local only) | `5ac8fd8a3b9347b95b4e60fa09c8a5050281cfe7` (snapshot taken from this commit via `git archive`) |
| Snapshot date | 2026-10-30 |

The `authoring` branch has not been pushed to khoros-cantata-python, and PR #1 is still open.
This copy **will diverge** from upstream: changes made here are not synced back automatically,
and later upstream changes are not pulled in. If PR #1 and the authoring work are merged upstream,
prefer depending on the upstream package and delete this folder.
License: Apache-2.0 (see `LICENSE`, `LICENSE_NOTICE.txt`), separate from the root project's license.

## Contents
- `khoros_cantata/` — engine: workspace parser/executors, VIFF, operator library (signal, image, data,
  matrix, FINANCE), causal `streaming.py`, `catalog.json` + `catalog_gen.py`, `authoring.py`, `cli.py`
- `templates/` — financial workspace specs (`*.yaml`) with generated `.wk` and explanations (`*.md`)
- `docs/` — `AUTHORING.md`, `CATALOG.md`, `STREAMING.md` (plus the original project docs)
- `examples/` — `streaming_ema_crossover.{py,wk}`
- `OPERATORS_STATUS.md`, `CHANGELOG.md`, `tests/`

## Install and test (from inside this folder, in its own venv)
```bash
cd khoros
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[authoring]" pytest      # numpy, scipy, PyYAML
pytest -q                                 # khoros test suite (separate from the root suite)
```
Some tests compare with pandas and are skipped if pandas is absent (`pip install pandas` to run them).

## Author CLI
```bash
python -m khoros_cantata.cli author validate templates/trend_ema_vol.yaml
python -m khoros_cantata.cli author build    templates/trend_ema_vol.yaml -o /tmp/trend.wk
python -m khoros_cantata.cli author explain  templates/trend_ema_vol.yaml --verify
python -m khoros_cantata.cli author verify   templates/meanrev_kalman_pair.yaml
python -m khoros_cantata.catalog_gen --check   # catalog up to date with the registry
python examples/streaming_ema_crossover.py
```

Templates are educational signal-processing examples on synthetic data, not trading advice.
