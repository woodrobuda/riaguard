"""Data ingestion: CSV loader, optional yfinance download, synthetic fallback."""
from __future__ import annotations
import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def load_csv(path, time_col: str | None = None) -> pd.DataFrame:
    """Load OHLCV bars or ticks from CSV. Columns are lower-cased; first datetime-like
    column (or ``time_col``) becomes a sorted DatetimeIndex."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tc = time_col or next((c for c in df.columns if c in ("date", "datetime", "timestamp", "time")), None)
    if tc:
        df[tc] = pd.to_datetime(df[tc], utc=True)
        df = df.set_index(tc).sort_index()
    return df


def synthetic_prices(n=1500, seed=0, mu=0.0003, sigma=0.012, cycle_period=60, cycle_amp=0.0,
                     start="2018-01-01") -> pd.DataFrame:
    """Daily OHLCV from geometric Brownian motion (+ optional deterministic cycle).
    With cycle_amp=0 there is no exploitable structure: a good null test."""
    rng = np.random.default_rng(seed)
    r = mu + sigma * rng.standard_normal(n)
    t = np.arange(n)
    logp = np.log(100) + np.cumsum(r) + cycle_amp * np.sin(2 * np.pi * t / cycle_period)
    close = np.exp(logp)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, sigma / 2, n)) * close
    idx = pd.bdate_range(start, periods=n, tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close,
                         "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)


def synthetic_ticks(n=20000, seed=0, start="2024-01-02 14:30", mean_gap_s=1.5, sigma=0.0002) -> pd.DataFrame:
    """Irregularly spaced trade ticks (exponential inter-arrival times)."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(mean_gap_s, n)
    ts = pd.Timestamp(start, tz="UTC") + pd.to_timedelta(np.cumsum(gaps), unit="s")
    price = 100 * np.exp(np.cumsum(sigma * rng.standard_normal(n)))
    size = rng.integers(1, 500, n).astype(float)
    return pd.DataFrame({"price": price, "size": size}, index=ts)


def fetch_yfinance(ticker="SPY", start="2015-01-01", end=None, fallback=True) -> tuple[pd.DataFrame, str]:
    """Download daily bars via yfinance. Returns (df, source). Falls back to synthetic
    data on any failure (offline, rate-limit, missing package) if ``fallback``."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise RuntimeError("empty download")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df[OHLCV].dropna(), f"yfinance:{ticker}"
    except Exception as e:  # noqa: BLE001
        if not fallback:
            raise
        return synthetic_prices(), f"synthetic (yfinance failed: {e!r})"
