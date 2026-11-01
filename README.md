# riaguard

Educational Python toolkit that applies classic signal processing (FIR/IIR moving averages, Kalman
filters, FFT/spectral analysis, wavelets, the Hilbert transform) to stock price and tick streams. It
includes a strict no-look-ahead walk-forward backtester and an optional bridge to
[khoros-cantata-python](https://github.com/woodrobuda/khoros-cantata-python).

> **DISCLAIMER.** This project is for education and research only. It is **not** investment, financial
> or trading advice. Nothing here is a recommendation to buy or sell any security. Backtests are
> hypothetical, use simplified costs, ignore slippage, taxes, borrow and market impact, and are
> subject to overfitting. Past (simulated) performance does not predict future results. Use at your
> own risk.

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,yfinance]"     # core + tests + free Yahoo data
pip install -e ".[khoros]"           # optional: Khoros/Cantata dataflow bridge
pytest -q
```
Notes on the optional Khoros bridge:
- Upstream khoros-cantata-python v0.22.0 cannot be pip-installed as-is ("Multiple top-level packages
  discovered in a flat-layout"). Clone it and add `[tool.setuptools] packages = ["khoros_cantata"]` to
  its `pyproject.toml` first, then run `pip install -e <path>`.
- The bridge's tests skip automatically if the package is absent.
- Causal streaming: with a khoros_cantata that has `khoros_cantata.streaming` (its local `operators`
  branch), `khoros_adapter.stream_executor(path)` runs a workspace tick by tick (`ex.push(price=p)`),
  using `STREAM:kinput`/`STREAM:koutput` glyphs; SIGTRADE `kema`, `ksma`, `kkalman`, `klogret`, `kcross`
  are declared causal and `kpower` is rejected as non-causal. Those tests skip on older khoros_cantata.

## Phase 1: evidence (backtesting, metrics, data, explainable alerts)

> **Not investment advice.** riaguard is an analysis and alerting toolkit. It simulates
> positions for research and **never places orders or connects to a broker**. All performance
> figures are hypothetical, simulated results on historical data.

* `engine`: a strictly causal vectorised backtester with walk-forward selection, long-flat, long-short and long-only modes, unit or vol-target sizing, and a `CostModel` (commission, spread, slippage, borrow).
* `metrics`: Sharpe, Sortino, Calmar, drawdown and its duration, hit rate, turnover and exposure, plus the probabilistic and deflated Sharpe ratio and PBO (CSCV).
* `signals`: DSP signals (EMA cross, Kalman slope, causal Butterworth trend) and benchmarks (buy-and-hold, SMA 50/200, 12-month time-series momentum).
* `datasources` and `quality`: a pluggable `DataSource` interface, local CSV/Parquet files, bundled offline sample data, Yahoo and Stooq (no key), a Polygon skeleton (env var key), and data checks.
* `explain`: JSONL alerts with inputs, filter values, the threshold crossed and the config hash.

```bash
python scripts/phase1_research.py      # reproduces docs/reports/phase1_results.md offline
```
**Headline (honest) result:** after walk-forward selection and costs, the DSP strategies beat buy-and-hold on Sharpe ratio in only a few of the 40
symbol/mode/family cases. Most of their value is lower drawdown in long-flat mode, not higher returns. See
[docs/reports/phase1_results.md](docs/reports/phase1_results.md), [docs/BACKTESTING.md](docs/BACKTESTING.md) and [docs/DATA.md](docs/DATA.md).

## Layout
| module | what |
|---|---|
| `data` | `load_csv`, `fetch_yfinance` (falls back to synthetic data offline), `synthetic_prices`, `synthetic_ticks` |
| `bars` | `time_bars`, `tick_bars`, `volume_bars`, `dollar_bars` from irregular ticks |
| `streaming` | online `update(x)` filters: `EMA`, `FIR` (`.sma`, `.wma`), `IIR(b, a)`, `KalmanLocalLinearTrend`, `KalmanHedgeRatio`, `RollingZScore` |
| `spectral` | `periodogram`, `welch`, `dominant_period`, `butter_lowpass`, `zero_phase` (non-causal!), `group_delay`, `wavelet_denoise`, `wavelet_mra`, `hilbert_phase`, `rolling_causal` |
| `strategies` | `buy_and_hold`, `ema_cross`, `sma_trend`, `kalman_slope` (long/flat) |
| `backtest` | `run_signal` (position = signal shifted 1 bar, costs in bps), `assert_causal`, `walk_forward`, `Result.stats()` |
| `khoros_adapter` | registers `SIGTRADE:{kema,ksma,kkalman,kpower,klogret,kcross}` in Khoros `NativeOperatorRegistry`; reuses `khoros_fft` and `conv1d_khoros`; `run_workspace(.wksp)` |

## Quick start
```python
from riaguard import data, streaming as st, strategies as S, backtest as bt
df, src = data.fetch_yfinance("SPY", start="2015-01-01")   # or data.load_csv("my.csv")
ema = st.EMA(span=20)
for price in df["close"]:            # tick-by-tick / bar-by-bar
    y = ema.update(price)
bt.assert_causal(S.ema_cross, df["close"])
print(bt.run_signal(df["close"], S.ema_cross(df["close"]), "ema", cost_bps=5).stats())
```

## Examples (`examples/`, plots in `examples/output/`)
1. `01_filters_lag.py`: smoothing vs lag. Plots group delay of the SMA, EMA and Butterworth filters,
   and shows a causal filter next to a look-ahead filtfilt.
2. `02_bars.py`: builds time, tick, volume and dollar bars from irregular ticks.
3. `03_spectral_wavelet.py`: periodogram pitfalls (a random walk shows fake cycles), wavelet
   denoising and multiresolution, and Hilbert instantaneous period.
4. `04_backtest.py [TICKER]`: filter signals against buy-and-hold, plus walk-forward EMA parameter
   selection.
5. `05_pairs_kalman.py`: Kalman dynamic hedge ratio and innovation z-score on a synthetic pair.
6. `06_khoros_workspace.py`: runs `trend.wksp`, a Cantata-style dataflow graph, through
   khoros-cantata-python.

See [docs/RESEARCH.md](docs/RESEARCH.md) for the literature notes and pitfalls.

## Khoros workspace authoring snapshot (`khoros/`)

[`khoros/`](khoros/) is a snapshot of the `authoring` branch of
[khoros-cantata-python](https://github.com/woodrobuda/khoros-cantata-python) (built on its unmerged
`operators` branch / PR #1): a NumPy/SciPy Khoros/Cantata dataflow engine with a causal streaming executor,
an operator catalog, FINANCE operators (Kalman trend/pairs, z-score, log returns) and an authoring CLI that
validates, builds and explains workspaces. Provenance, install and test instructions:
[`khoros/README_SNAPSHOT.md`](khoros/README_SNAPSHOT.md). It is a separate project with its own test suite
(not collected by the root `pytest`) and its own Apache-2.0 license.

- Templates: [`khoros/templates/`](khoros/templates/) — trend (EMA crossover + volatility filter),
  Kalman pairs mean reversion, volatility regime (spec, generated `.wk`, explanation)
- Docs: [`AUTHORING.md`](khoros/docs/AUTHORING.md), [`CATALOG.md`](khoros/docs/CATALOG.md), [`STREAMING.md`](khoros/docs/STREAMING.md)

The disclaimer above applies here too: templates are educational, run on synthetic data, and are
not investment, financial or trading advice.

## License
MIT (placeholder; see LICENSE, owner to confirm).
