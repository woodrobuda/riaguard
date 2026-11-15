# Alert wording guide

> **DRAFT — pending securities counsel review before first firm use.** This is not legal advice.

riaguard alerts **describe** what the data and models show. They never **prescribe** an
action. A reader should learn *what changed, by how much, against which rule, and when*,
and nothing about what they or a client "should" do.

## Rules
1. State observations and model states, not actions. No "buy", "sell", "hold", "exit",
   "add", "trim", "rotate into", "go long/short", "take profit".
2. No advice verbs or urgency: avoid "should", "recommend", "consider buying", "act now",
   "opportunity", "don't miss".
3. No predictions or promises: avoid "will rise", "guaranteed", "safe", "can't lose".
   Prefer "has", "is", "crossed", "since".
4. Always show the evidence: the inputs, the threshold or rule crossed, the timestamp and
   data source (including whether data is delayed or end-of-day), and the config hash.
5. Impersonal: name the instrument or the portfolio under review, never a client's account
   or situation.
6. Keep the standard footer: *"Research alert. Not investment advice. No order was placed."*
   (already emitted in every JSONL alert as `disclaimer`).
7. Hypothetical results are labelled as hypothetical every time they appear.

## Examples

Numbers below are illustrative, except the first and fifth rows, which come from the phase-1 sample alert and report.

| Prescriptive (do not use) | Descriptive (use) |
|---|---|
| "Sell SPY now: the trend has turned." | "SPY: 20-day EMA (574.03) crossed below the 100-day EMA (575.66) at the 2026-09-30 close (EOD data). Rule `ema_cross(20,100)`, config ef6e2207." |
| "Reduce equity exposure, a crash is coming." | "Portfolio realised volatility (20-day) is 24.1%, above the 20% regime threshold for the first time since 2025-04-11." |
| "Great buying opportunity in QQQ." | "QQQ drawdown from its 252-day high is -12.4%, past the -10% review threshold." |
| "Client Smith should exit TLT." | "TLT: Kalman trend slope has been negative for 15 sessions (threshold: 10)." |
| "This strategy will beat the market." | "Hypothetical backtest, 2013–2026, net of modelled costs: Sharpe 0.90 vs 0.91 for buy-and-hold. Not indicative of future results." |
| "Unusual trading, investigate the rep." | "Account-level turnover in the review sample is 3.2× its 90-day median (threshold 3×). Flagged for supervisory review." |

## Checklist for new alert types
- [ ] Message uses only descriptive verbs (crossed, is, has been, rose, fell, exceeded).
- [ ] Inputs, threshold and config hash are present.
- [ ] Data timeliness is stated (real-time, delayed or EOD).
- [ ] Disclaimer footer is present.
- [ ] Reviewed against this guide in the PR.

## CI enforcement
Rules 1–3 are enforced in CI by `scripts/check_alert_wording.py` (job `alert-wording` in
`.github/workflows/ci.yml`). It scans string literals in alert-generating code
(`src/riaguard/explain.py`; docstrings and comments are skipped) and the sample alerts in
`docs/reports/sample_alerts.jsonl`. Keep its wordlist in sync with this guide. Run it locally with
`python scripts/check_alert_wording.py`.
