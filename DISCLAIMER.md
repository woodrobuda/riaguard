# Disclaimer

> **DRAFT — pending securities counsel review before first firm use.** This is not legal advice.

**riaguard is software for analysis and alerts. It is not investment, financial, legal, tax or
trading advice, and nothing it outputs is a recommendation or an offer to buy, sell or hold
any security.** It never places orders. See [docs/SCOPE.md](docs/SCOPE.md).

## Hypothetical and backtested performance
All performance figures in this repository (README, `docs/reports/`, examples) are
**hypothetical**. They come from simulated backtests on historical data, not from any account
or client portfolio, and no money was invested. Firms subject to the SEC Marketing Rule
(Rule 206(4)-1 under the Advisers Act) or FINRA communications rules should treat any of these
figures as hypothetical performance and should not use them in advertisements without their
own compliance review. In particular:

- **Hindsight and selection.** Backtests are designed after the fact. Parameters were chosen
  by walk-forward selection over many trials (the report states how many and deflates the
  Sharpe ratio for it), and the research choices themselves can still overfit.
- **Simplified costs.** Commissions, spreads, slippage and borrow are modelled with fixed
  assumptions. Taxes, market impact, fees, financing and capacity are ignored or simplified.
- **Data limits.** Free and demo data (Polygon free tier, Yahoo) can be delayed, adjusted,
  incomplete or wrong, and the bundled universe is small and not survivorship-free.
- **No prediction.** Past real or simulated results do not indicate future results. Real
  trading would differ, and losses can exceed what a backtest shows.
- **Negative results are reported.** In phase 1 the strategies mostly did **not** beat
  buy-and-hold on a risk-adjusted basis after costs; see
  [docs/reports/phase1_results.md](docs/reports/phase1_results.md). Any figure quoted alone,
  without that context, is misleading.
- **No specific client.** Results are not tailored to any person's objectives, finances or
  risk tolerance.

## No warranty
Provided "as is" under the Apache License 2.0, without warranties or conditions of any kind.
The users of the software are responsible for how they use it.
