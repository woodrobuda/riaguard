# Backtesting (phase 1)

> Research and analysis only. **Not investment advice.** The engine simulates positions; it never
> connects to a broker or places orders. Simulated results are hypothetical and do not predict future returns.

## Pieces

| module | what |
|---|---|
| `signals` | `Signal` objects. `compute(close)` returns a frame with `direction` (-1/0/+1), `score` (the value compared with `threshold`) and the filter internals. DSP signals: `ema_cross`, `kalman_slope`, `butter_trend` (causal IIR), all built on `streaming`. Benchmarks: `buy_and_hold`, `sma_crossover(50, 200)`, `tsmom(252)`. Grids are in `DSP_GRIDS`. |
| `engine` | `target_position` (modes `long_flat`, `long_short`, `long_only`), `Sizing` (`unit` or `vol_target`), `simulate`, `run`, `walk_forward` and `assert_causal` (truncate *and* perturb the future). |
| `costs` | `CostModel`: commission, half-spread, fixed slippage and volatility-scaled slippage in bps per side, and borrow in bps/yr on shorts. |
| `splits` | `walk_forward_splits` (rolling or expanding, optional gap/embargo) and `purged_kfold` (purge + embargo). |
| `metrics` | Sharpe, Sortino, Calmar, CAGR, max drawdown and its duration, hit rate, turnover, exposure, PSR, deflated Sharpe ratio, `expected_max_sharpe`, and PBO via CSCV. |
| `explain` | Explainable alerts as JSONL (see below). |

## Timing (why there is no look-ahead)

* The signal at bar *t* uses closes up to and including *t*.
* The position is reached `delay` bars later (`delay >= 1` is enforced). With `delay=1` you trade at the close of *t* and earn the return from bar *t+1* onwards.
* Costs are charged on the bar where the position changes: `|Δpos| × (commission + half_spread + slippage [+ k·σ]) / 1e4`, plus borrow on short positions.
* `assert_causal(fn, close)` checks that outputs at or before random cut points do not change when the future is either removed or replaced with noise. Every built-in signal is tested this way, and so are engine positions and walk-forward returns.

## Walk-forward selection

`engine.walk_forward(close, candidates, train=756, test=126, mode=..., costs=...)`:

1. Each candidate is computed once over the full series. That is valid only because every candidate is causal, which the tests verify.
2. For each fold, the candidate with the best **net-of-cost** Sharpe on the training window is chosen.
3. Its target positions are stitched into the test window, and the stitched series is re-simulated so that the cost of switching between folds is charged.
4. The output is the concatenated out-of-sample result, plus the per-fold choices in `result.meta["folds"]`.

Use `gap=` to embargo bars between training and test windows. For labels that span several bars, use `splits.purged_kfold`.

## Multiple testing

Trying many configurations inflates the best Sharpe ratio. Report:

* `deflated_sharpe_ratio(returns, n_trials, trial_sharpes=...)`: the probability that the true Sharpe is above the best Sharpe expected from `n_trials` unskilled strategies (Bailey & López de Prado 2014). Count **every** configuration, mode and symbol you tried.
* `pbo_cscv(T×N returns, n_splits)`: the probability of backtest overfitting (Bailey, Borwein, López de Prado & Zhu 2015).

A high DSR does not mean the strategy beats a benchmark; compare against `BENCHMARKS` over the same window.

## Explainable alerts

`explain.alerts_from_signal(symbol, signal, close, config)` emits one record per change of direction:
`ts, symbol, signal, from, to, crossed, threshold, score, features` (filter values), `inputs` (the last closes),
`params, config` and `config_hash` (SHA-256 of the canonical JSON config), `engine_version` and `disclaimer`.
`explain.write_jsonl(records, path)` appends them with a `logged_at` stamp. See `docs/reports/sample_alerts.jsonl`.

## Reproducing the phase-1 study

```bash
pip install -e ".[dev]"
python scripts/phase1_research.py          # offline, uses the bundled sample data (about 15 s)
```
It writes `docs/reports/phase1_results.md` and `.json`. Use `--source yahoo` to refresh from Yahoo, and `--per-side-bps 10` to stress costs.

## Known limitations

* Bars are daily and fills happen at the close. There is no intraday fill model, no market impact beyond the vol-scaled slippage term, no taxes, and no capacity constraints.
* The engine handles one instrument at a time. Portfolio and multi-asset netting is on the roadmap.
* The trading calendar is approximate (regular NYSE holidays only). See `docs/DATA.md`.
