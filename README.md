# riaguard

**Risk overlays, regime detection and explainable alerts for small RIAs and broker-dealers.**
Open source (Apache-2.0), self-hostable, and **analysis and alerts only: it never places orders.**

> **Not investment advice.** riaguard describes market data and model states; it does not
> recommend any security to anyone. All performance figures here are **hypothetical** backtests.
> See [DISCLAIMER.md](DISCLAIMER.md) and [docs/SCOPE.md](docs/SCOPE.md) (drafts pending securities
> counsel review).

## Who it is for
- **Primary users: small registered investment advisers (RIAs) and small broker-dealers** that
  want auditable risk and supervisory tooling without an enterprise budget. The firm stays
  responsible for any advice and for its own compliance; riaguard supplies impersonal analysis.
- **Community: retail quants and researchers**, who use and extend the open-source core.

## What it does (in order of emphasis)
1. **Risk overlays and regime detection** (the lead, with the strongest evidence). Causal
   trend, volatility and drawdown regimes from DSP filters (EMA, Kalman, Butterworth, spectral,
   wavelets). In phase 1 the clearest result was *risk reduction*, not extra return: for example
   on SPY, 2013–2026, the walk-forward EMA long-flat overlay had a hypothetical max drawdown of
   -27.0% and volatility of 12.1%, against -33.7% and 16.7% for buy-and-hold, with a similar
   Sharpe ratio (0.90 vs 0.91) and a lower CAGR (10.6% vs 14.8%).
2. **Surveillance-style alerts.** Explainable JSONL alerts with inputs, filter values, the
   threshold crossed and a config hash, worded descriptively ([docs/ALERT_WORDING.md](docs/ALERT_WORDING.md)).
3. **Return-seeking strategies, presented as research.** Walk-forward backtests with costs,
   deflated Sharpe and PBO, published with their negative results: most DSP strategies did
   **not** beat buy-and-hold on Sharpe after costs.

**Data:** Polygon free tier first (`POLYGON_API_KEY`, rate-limited, end-of-day/delayed), Yahoo
for demos, bundled sample data for offline tests. IEX Cloud shut down in 2024. See [docs/DATA.md](docs/DATA.md).

**Editions:** one shared core, then Desktop, Self-hosted Docker and Managed editions that share
configs and alert history ([docs/EDITIONS.md](docs/EDITIONS.md)). Only the core exists today.

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,yfinance]"     # core + tests + Yahoo demo data
pytest -q
export POLYGON_API_KEY=...            # optional: Polygon free tier (never commit keys)
```

## Plug-ins
Optional extensions register through the `riaguard.plugins` entry-point group
(`riaguard.plugins.available()`, `load(name)`, `load_all()`). The Khoros/Cantata dataflow bridge
now lives in the separate plug-in repo
[woodrobuda/riaguard-khoros](https://github.com/woodrobuda/riaguard-khoros) (private for now while
the licensing of its Khoros snapshot is confirmed).

## Phase 1: evidence (backtesting, metrics, data, explainable alerts)

> **Not investment advice.** riaguard is an analysis and alerting toolkit. It simulates
> positions for research and **never places orders or connects to a broker**. All performance
> figures are hypothetical, simulated results on historical data.

* `engine`: a strictly causal vectorised backtester with walk-forward selection, long-flat, long-short and long-only modes, unit or vol-target sizing, and a `CostModel` (commission, spread, slippage, borrow).
* `metrics`: Sharpe, Sortino, Calmar, drawdown and its duration, hit rate, turnover and exposure, plus the probabilistic and deflated Sharpe ratio and PBO (CSCV).
* `signals`: DSP signals (EMA cross, Kalman slope, causal Butterworth trend) and benchmarks (buy-and-hold, SMA 50/200, 12-month time-series momentum).
* `datasources` and `quality`: a pluggable `DataSource` interface, local CSV/Parquet files, bundled offline sample data, Yahoo and Stooq (no key), Polygon (free-tier rate limits, env var key), and data checks.
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
| `plugins` | `riaguard.plugins` entry-point hook for optional plug-ins |

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

See [docs/RESEARCH.md](docs/RESEARCH.md) for the literature notes and pitfalls.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md). Every commit needs a DCO sign-off (`git commit -s`); CI checks it.

## License
Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Contributions need a DCO sign-off (`git commit -s`); see [CONTRIBUTING.md](CONTRIBUTING.md).
