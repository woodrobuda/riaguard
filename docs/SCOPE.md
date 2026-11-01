# Scope of riaguard

> **DRAFT — pending securities counsel review before first firm use.** This is not legal advice.

## What riaguard is
riaguard is open-source **analysis and alerting** software. It reads market data, computes
risk, regime and signal-processing measures, and records explainable, versioned alerts. Its
primary users are **small registered investment advisers (RIAs) and small broker-dealers**,
who run it themselves for internal risk, research and supervisory review. Retail quants use
and contribute to the open-source core.

## What riaguard is not
- **Not an execution system.** It never places, routes, modifies or cancels orders, and has
  no broker order APIs. Positions in backtests are simulated. Contributions that add order
  entry are out of scope and will be rejected.
- **Not investment advice.** It does not make recommendations about any security to any
  person, and the project does not provide advisory services.
- **Not personalised.** Output is **impersonal**: it describes market data and model
  states for an instrument or portfolio. It does not consider any client's objectives,
  financial situation or needs, and it does not produce trade recommendations for specific
  clients.
- **Not a compliance system of record** by itself. Surveillance-style alerts are inputs to
  a firm's own supervisory review, not a substitute for it.

## Who stays responsible
The firm using riaguard (the RIA or broker-dealer and its supervised persons) remains
solely responsible for:
- any investment advice or recommendation it gives to clients, and its suitability or
  best-interest obligations;
- its supervision, surveillance, books-and-records and marketing obligations;
- validating any model or signal before relying on it, and monitoring it afterwards;
- the data licences it uses (Polygon, Yahoo and others have their own terms).

riaguard's authors and contributors provide the software "as is" under Apache-2.0
(see `LICENSE`), with no warranty and no duty to any firm or client.

## Product areas, in order of emphasis
1. **Risk overlays and regime detection** (lead area, strongest evidence so far): volatility,
   drawdown and trend-regime states that a firm may use as context in its own risk process.
2. **Surveillance-style alerts**: descriptive, explainable alerts (what changed, by how much,
   against which threshold, under which config hash) for supervisory review.
3. **Return-seeking strategies, as research only**: hypothetical backtests published with
   their negative results. They are not offered as a strategy or a recommendation.

## Wording
All alerts and docs use descriptive, not prescriptive, wording. See
[ALERT_WORDING.md](ALERT_WORDING.md) and [../DISCLAIMER.md](../DISCLAIMER.md).
