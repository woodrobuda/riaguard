# Data layer (phase 1)

> Research and analysis only. Not investment advice.

## Bar schema
Every `DataSource.get_bars(symbol, start, end, interval)` returns a frame with a tz-aware UTC `DatetimeIndex` named `date`. Daily bars are stamped at midnight UTC on the session date. Columns are `open, high, low, close, volume`, plus optional `adj_close, dividends, splits`.

## Sources (`riaguard.datasources`)

| source | key needed | notes |
|---|---|---|
| `LocalFileSource(root)` | no | reads `<root>/<SYMBOL>.parquet`, `.csv.gz` or `.csv`. Parquet needs `pip install -e ".[parquet]"`. |
| `SampleSource()` | no | Bundled daily bars for SPY, QQQ, IWM, TLT and GLD, 2010-01-04 to 2026-09-30 (about 100 KB each), downloaded from Yahoo Finance via yfinance on 2026-10-31. They exist so that the tests and the research script run offline. Yahoo's terms apply; this data is for research only. |
| `YahooSource()` | no | needs the `yfinance` extra. `close` is split-adjusted; `adj_close` is also dividend-adjusted. |
| `StooqSource()` | no | free daily CSV from stooq.com. Stooq may serve a JavaScript challenge to scripted clients, in which case a `RuntimeError` is raised. |
| `PolygonSource()` | `POLYGON_API_KEY` env var | A skeleton for `/v2/aggs` aggregates only. Its live test is skipped unless the variable is set. Never put keys in code or in config files. |

Write your own source by subclassing `DataSource` and returning the schema above through `datasources.normalize`.

## Quality checks (`riaguard.quality`)

* `check_bars(df)` returns a report and never edits data. It covers duplicate or non-monotonic timestamps, NaNs, non-positive prices, OHLC inconsistencies and negative volume. It also lists suspect bad ticks (a robust-z spike that reverts on the next bar), missing or unexpected sessions against an approximate NYSE calendar, split events and the number of dividends.
* `clean_bars(df, drop_bad_ticks=False)` de-duplicates, sorts, drops NaN or non-positive rows, and can drop bad ticks.
* `adjust_for_splits(df)` back-adjusts raw OHLCV using a `splits` ratio column. `total_return_close(df)` returns the dividend-adjusted close for return maths.
* `regular_session(ticks)` keeps 09:30–16:00 America/New_York (DST-aware) on weekdays.
* `expected_sessions(start, end)` uses the regular NYSE holidays. Ad-hoc closures are *not* included and show up as missing sessions. In the sample data these are 2012-10-29/30 (Hurricane Sandy), 2018-12-05 and 2025-01-09. Early closes are not modelled. For production, use an exchange calendar package.

## Not yet covered (roadmap phase 2)
Tick and quote (TAQ) files, consolidated vs. primary-exchange prints, trade condition codes, corporate-action feeds (symbol changes, spin-offs), survivorship-free universes, and the Databento, IEX and Refinitiv/Bloomberg plug-ins.
