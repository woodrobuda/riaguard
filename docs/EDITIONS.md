# Editions and architecture

> Architecture plan only. None of the editions below is built yet.

riaguard ships as **one shared core** with three editions, built in this order:

| # | edition | who | how it runs | status |
|---|---|---|---|---|
| 1 | **Desktop** | a single adviser or analyst | local app on one machine, local files | planned next |
| 2 | **Self-hosted Docker** | a small firm's team | `docker compose` on the firm's own server or cloud account | after Desktop |
| 3 | **Managed** | firms that do not want to run servers | hosted by the project, per-firm tenant | later |

All three are analysis and alerts only (see [SCOPE.md](SCOPE.md)). None places orders.

## The core (`riaguard` package, this repo)
Everything in `src/riaguard/` is the core and stays edition-agnostic:

- `datasources`, `quality`: data adapters (Polygon free tier first, Yahoo for demos, local files) and checks
- `streaming`, `spectral`, `bars`, `signals`, `strategies`: measures, regimes and signals
- `engine`, `backtest`, `costs`, `splits`, `metrics`: hypothetical research and evaluation
- `explain`: versioned, explainable JSONL alerts (inputs, threshold, config hash)
- `plugins`: the `riaguard.plugins` entry-point hook for optional extensions

Rules that keep it separable:
1. The core never imports an edition or a plug-in. Editions and plug-ins depend on the core,
   never the reverse.
2. The core has no UI, web server, scheduler, database driver or auth code.
3. The core reads secrets only from environment variables and never writes them to disk.

## What the editions add (later, as separate packages)
| layer | Desktop | Self-hosted Docker | Managed |
|---|---|---|---|
| package | `riaguard-desktop` | `riaguard-server` (+ image) | `riaguard-server` + hosting ops |
| UI | local app / browser on localhost | web UI | web UI |
| scheduling | in-process | worker container | managed workers |
| alert store | SQLite file | Postgres | Postgres per tenant |
| users | one | firm users (OIDC) | firm users (OIDC) |
| audit | hash-chained alert log | same + WORM export | same + WORM export |

## Shared configs and alert history (so firms can move up)
- **Config**: one versioned file format (`riaguard.toml` / YAML, schema-versioned) for
  universes, data sources, signal parameters and thresholds. Every edition reads the same file;
  the config hash in each alert is computed the same way everywhere.
- **Alert history**: one versioned alert schema (today's JSONL from `explain`, later JSON Schema).
  Desktop stores it in SQLite and can export JSONL; Docker and Managed import that JSONL, so a
  firm keeps its full history, config hashes and hash chain when it upgrades.
- **Migrations**: Desktop → Docker → Managed is an export/import of config + alert history,
  with a verification step that recomputes the hash chain.

## Plug-ins
Optional integrations live in their own repos and register through `riaguard.plugins`
(for example [riaguard-khoros](https://github.com/woodrobuda/riaguard-khoros), the Khoros/Cantata
dataflow bridge, private for now). Every edition loads the same plug-ins the same way.
