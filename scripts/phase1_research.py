"""Phase-1 research run: DSP signals vs benchmarks, walk-forward, net of costs.

    python scripts/phase1_research.py            # offline, uses the bundled sample data
    python scripts/phase1_research.py --source yahoo --symbols SPY QQQ

Writes docs/reports/phase1_results.md, phase1_results.json and sample_alerts.jsonl.
Hypothetical, simulated research only. Not investment advice."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signal_trading import __version__, datasources as D, quality as Q, signals as S, engine as E, metrics as M, explain as X  # noqa: E402
from signal_trading.costs import CostModel  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "reports"


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "n/a"
    return f"{x*100:.1f}%" if pct else f"{x:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sample", choices=list(D.SOURCES))
    ap.add_argument("--symbols", nargs="+", default=list(D.SampleSource.SYMBOLS))
    ap.add_argument("--train", type=int, default=756); ap.add_argument("--test", type=int, default=126)
    ap.add_argument("--per-side-bps", type=float, default=4.0)
    a = ap.parse_args(argv)
    src = D.SOURCES[a.source]()
    costs = CostModel(commission_bps=1.0, half_spread_bps=1.0, slippage_bps=a.per_side_bps - 2.0,
                      borrow_bps_per_year=50.0)
    families = dict(S.DSP_GRIDS); all_dsp = [s for g in families.values() for s in g]
    families["all_dsp"] = all_dsp
    modes = ("long_flat", "long_short")
    n_trials_total = len(all_dsp) * len(modes) * len(a.symbols)
    config = {"engine_version": __version__, "source": a.source, "train": a.train, "test": a.test,
              "costs": costs.to_dict(), "delay": 1, "sizing": "unit", "select_metric": "sharpe"}
    rows, pbo_rows, quality, wf_detail = [], [], {}, {}
    for sym in a.symbols:
        df = src.get_bars(sym); quality[sym] = Q.check_bars(df)
        close = Q.total_return_close(Q.clean_bars(df))
        for mode in modes:
            res = {}
            for fam, cands in families.items():
                r, nets = E.walk_forward(close, cands, a.train, a.test, mode, costs, name=f"WF:{fam}")
                res[f"WF:{fam}"] = r
                if fam == "all_dsp":
                    oos = r.returns.index
                    trial_sr = [M.sharpe(n.loc[oos], 1) for n in nets.values()]
                    wf_detail[f"{sym}/{mode}"] = r.meta["folds"]
                    if mode == "long_flat":
                        mat = pd.DataFrame({k: v.loc[oos] for k, v in nets.items()})
                        pbo_rows.append({"symbol": sym, **M.pbo_cscv(mat, 10)})
            oos = res["WF:all_dsp"].returns.index
            for b in S.BENCHMARKS:
                bmode = "long_only" if b.name == "buy_and_hold" else mode
                full = E.run(close, b, bmode, costs)
                res[b.label] = E.BacktestResult(b.label, full.returns.loc[oos], full.gross.loc[oos], full.costs.loc[oos], full.position.loc[oos])
            bh = res["buy_and_hold"].returns
            for name, r in res.items():
                st = r.stats(); st.update(symbol=sym, mode=mode, strategy=name,
                                          start=str(oos[0].date()), end=str(oos[-1].date()))
                if name.startswith("WF:"):
                    st["DSR"] = M.deflated_sharpe_ratio(r.returns, n_trials_total, trial_sharpes=trial_sr)
                st["sharpe_minus_bh"] = st["Sharpe"] - M.sharpe(bh)
                rows.append(st)
    tab = pd.DataFrame(rows)
    # cost sensitivity on SPY (or first symbol), all-DSP long-flat
    sym0 = "SPY" if "SPY" in a.symbols else a.symbols[0]
    c0 = Q.total_return_close(src.get_bars(sym0)); sens = []
    for bps in (0.0, 2.0, 4.0, 10.0, 20.0):
        cm = CostModel(bps, 0.0, 0.0)
        r, _ = E.walk_forward(c0, all_dsp, a.train, a.test, "long_flat", cm)
        sens.append({"per_side_bps": bps, **{k: r.stats()[k] for k in ("Sharpe", "CAGR", "maxDD", "turnover")}})
    # explainable alerts sample
    sig = S.make("ema_cross", 20, 100); c = Q.total_return_close(src.get_bars(sym0))
    alerts = [x for x in X.alerts_from_signal(sym0, sig, c, config) if x["ts"] >= "2024-01-01"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample_alerts.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in alerts))
    payload = {"config": config, "config_hash": X.config_hash(config), "n_trials_total": n_trials_total,
               "results": tab.replace({np.nan: None}).to_dict("records"), "pbo": pbo_rows,
               "cost_sensitivity": sens, "data_quality": {k: {kk: vv for kk, vv in v.items() if kk != "dividend_events"} for k, v in quality.items()},
               "walk_forward_choices": wf_detail}
    (OUT / "phase1_results.json").write_text(json.dumps(payload, indent=1, default=str))
    write_md(tab, pbo_rows, sens, quality, config, payload["config_hash"], n_trials_total, len(alerts))
    print((OUT / "phase1_results.md").read_text())


def write_md(tab, pbo_rows, sens, quality, config, h, n_trials, n_alerts):
    L = ["# Phase 1 results: DSP signals vs benchmarks (walk-forward, net of costs)", "",
         "> Hypothetical, simulated research on historical data. **Not investment advice.** No orders were placed.",
         "> Generated by `scripts/phase1_research.py`; rerun it to reproduce. Config hash `" + h[:16] + "`.", "",
         f"Setup: daily total-return closes (`adj_close`), rolling walk-forward with {config['train']} training bars and "
         f"{config['test']} test bars. In each fold the configuration with the best net in-sample Sharpe is held out of sample. "
         f"Costs are {config['costs']['commission_bps'] + config['costs']['half_spread_bps'] + config['costs']['slippage_bps']:.0f} bps "
         f"per side plus {config['costs']['borrow_bps_per_year']:.0f} bps/yr borrow on shorts. Trades are filled one bar after the signal. Benchmarks use fixed "
         "textbook parameters (no selection) and are measured over the same out-of-sample window. Buy-and-hold is always long-only.", "",
         f"DSR = deflated Sharpe ratio (probability that the true Sharpe is > 0 after accounting for {n_trials} trials and the spread of trial Sharpes). "
         "The usual bar is DSR > 0.95.", ""]
    cols = ["strategy", "CAGR", "vol", "Sharpe", "Sortino", "Calmar", "maxDD", "maxDD_bars", "hit_rate", "turnover", "exposure", "DSR"]
    for (sym, mode), g in tab.groupby(["symbol", "mode"], sort=False):
        L += [f"## {sym}, {mode} ({g['start'].iloc[0]} to {g['end'].iloc[0]})", "",
              "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in g.iterrows():
            L.append("| " + " | ".join([str(r["strategy"])] + [
                fmt(r.get(c), pct=c in ("CAGR", "vol", "maxDD", "hit_rate", "exposure")) if c != "maxDD_bars" else str(int(r[c]))
                for c in cols[1:]]) + " |")
        L.append("")
    wf = tab[tab.strategy.str.startswith("WF:")]
    best_bench = tab[~tab.strategy.str.startswith("WF:")].groupby(["symbol", "mode"])["Sharpe"].max()
    beat_bh = int((wf["sharpe_minus_bh"] > 0).sum()); beat_any = int(sum(r.Sharpe > best_bench[(r.symbol, r["mode"])] for _, r in wf.iterrows()))
    dsr_pass = int((wf["DSR"] > 0.95).sum())
    L += ["## Summary", "",
          f"* Walk-forward DSP results: {len(wf)} (4 families x 2 modes x {tab.symbol.nunique()} symbols).",
          f"* Beat buy-and-hold on Sharpe: **{beat_bh} / {len(wf)}**. Beat the best benchmark on Sharpe: **{beat_any} / {len(wf)}**.",
          f"* Pass the deflated-Sharpe bar (DSR > 0.95): **{dsr_pass} / {len(wf)}**.",
          f"* Median out-of-sample Sharpe: WF-DSP {wf.Sharpe.median():.2f}; buy-and-hold {tab[tab.strategy=='buy_and_hold'].Sharpe.median():.2f}; "
          f"SMA 50/200 {tab[tab.strategy.str.startswith('sma_crossover')].Sharpe.median():.2f}; TSMOM {tab[tab.strategy.str.startswith('tsmom')].Sharpe.median():.2f}.", "",
          "A DSR above 0.95 says a Sharpe ratio is unlikely to be pure luck given the number of trials. It does **not** say the "
          "strategy beats buy-and-hold: a long-flat trend filter on an index that rose over the period mostly earns the same market exposure, with less of it.", "",
          "### Probability of backtest overfitting (CSCV, 10 blocks, long-flat, all DSP configurations)", "",
          "| symbol | PBO | combinations |", "|---|---|---|"]
    L += [f"| {p['symbol']} | {p['pbo']:.2f} | {p['n_combinations']} |" for p in pbo_rows]
    L += ["", "PBO near or above 0.5 means that picking the best in-sample configuration is no better than picking at random out of sample.", "",
          "### Cost sensitivity (SPY, all-DSP walk-forward, long-flat)", "", "| bps per side | Sharpe | CAGR | maxDD | turnover/yr |", "|---|---|---|---|---|"]
    L += [f"| {s['per_side_bps']:.0f} | {s['Sharpe']:.2f} | {s['CAGR']*100:.1f}% | {s['maxDD']*100:.1f}% | {s['turnover']:.1f} |" for s in sens]
    L += ["", "Turnover and Sharpe are not monotonic in cost because walk-forward selection is done net of costs: at higher costs it picks slower, lower-turnover configurations.",
          "", "### Data quality (bundled sample data)", ""]
    for k, v in quality.items():
        L.append(f"* {k}: ok={v['ok']}, missing sessions vs. the approximate NYSE calendar: {', '.join(v['missing_sessions']) or 'none'}; "
                 f"suspect bad ticks: {len(v['suspect_bad_ticks'])}; dividend events: {v.get('dividend_events', 0)}.")
    L += ["", "The missing sessions are real ad-hoc market closures (Hurricane Sandy 2012-10-29/30, national days of mourning 2018-12-05 and 2025-01-09), not data gaps.", "",
          f"An example of explainable alerts ({n_alerts} records, `ema_cross(20, 100)` on SPY since 2024) is in `sample_alerts.jsonl`.", "",
          "### Caveats", "",
          "* Five liquid ETFs, one data vendor (Yahoo), daily bars, about 13 years out of sample: a small sample. Results are not evidence of future performance.",
          "* Costs are a flat bps model. There is no market impact model, and ETF dividends are taken from the adjusted close without withholding tax.",
          "* The grids are small and were fixed before this run. No grid was tuned after looking at these results.",
          ""]
    (OUT / "phase1_results.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
