# Data layer (phase 1)

> Research and analysis only. Not investment advice.

## Bar schema
Every `DataSource.get_bars(symbol, start, end, interval)` returns a frame with a tz-aware UTC `DatetimeIndex` named `date`. Daily bars are stamped at midnight UTC on the session date. Columns are `open, high, low, close, volume`, plus optional `adj_close, dividends, splits`.

## Sources (`riaguard.datasources`)

**Order of preference:** Polygon free tier first (real use), Yahoo for demos, bundled sample data for tests and offline runs. **IEX Cloud shut down in 2024** (its API was retired on 2024-08-31), so there is no IEX Cloud adapter; the separate IEX exchange's own historical DEEP/TOPS files are a possible later plug-in.

| source | key needed | notes |
|---|---|---|
| `LocalFileSource(root)` | no | reads `<root>/<SYMBOL>.parquet`, `.csv.gz` or `.csv`. Parquet needs `pip install -e ".[parquet]"`. |
| `SampleSource()` | no | Bundled daily bars for SPY, QQQ, IWM, TLT and GLD, 2010-01-04 to 2026-09-30 (about 100 KB each), downloaded from Yahoo Finance via yfinance on 2026-10-31. They exist so that the tests and the research script run offline. Yahoo's terms apply; this data is for research only. |
| `YahooSource()` | no | for demos only (unofficial API, Yahoo's terms apply). Needs the `yfinance` extra. `close` is split-adjusted; `adj_close` is also dividend-adjusted. |
| `StooqSource()` | no | free daily CSV from stooq.com. Stooq may serve a JavaScript challenge to scripted clients, in which case a `RuntimeError` is raised. |
| `PolygonSource()` | `POLYGON_API_KEY` env var | **Primary source.** `/v2/aggs` aggregates with `next_url` pagination, tuned for Polygon's **free tier**: a process-wide throttle of 5 requests/minute (`calls_per_minute=`, raise it on a paid plan) and backoff on HTTP 429. Free-tier data is end-of-day/delayed with limited history, so use it for daily research and next-day alerts, not real-time monitoring. The key is sent in a header, never in the URL. Its live test is skipped unless the variable is set. Never put keys in code or in config files. |

Write your own source by subclassing `DataSource` and returning the schema above through `datasources.normalize`.

## Quality checks (`riaguard.quality`)

* `check_bars(df)` returns a report and never edits data. It covers duplicate or non-monotonic timestamps, NaNs, non-positive prices, OHLC inconsistencies and negative volume. It also lists suspect bad ticks (a robust-z spike that reverts on the next bar), missing or unexpected sessions against an approximate NYSE calendar, split events and the number of dividends.
* `clean_bars(df, drop_bad_ticks=False)` de-duplicates, sorts, drops NaN or non-positive rows, and can drop bad ticks.
* `adjust_for_splits(df)` back-adjusts raw OHLCV using a `splits` ratio column. `total_return_close(df)` returns the dividend-adjusted close for return maths.
* `regular_session(ticks)` keeps 09:30–16:00 America/New_York (DST-aware) on weekdays.
* `expected_sessions(start, end)` uses the regular NYSE holidays. Ad-hoc closures are *not* included and show up as missing sessions. In the sample data these are 2012-10-29/30 (Hurricane Sandy), 2018-12-05 and 2025-01-09. Early closes are not modelled. For production, use an exchange calendar package.

## Not yet covered (roadmap phase 2)
Tick and quote (TAQ) files, consolidated vs. primary-exchange prints, trade condition codes, corporate-action feeds (symbol changes, spin-offs), survivorship-free universes, and the Databento, IEX exchange (DEEP/TOPS) and Refinitiv/Bloomberg plug-ins.
