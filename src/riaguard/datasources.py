"""Pluggable market-data sources.

Standard bar schema returned by every source: a tz-aware (UTC) ``DatetimeIndex`` named
``date`` and float columns ``open, high, low, close, volume``; optionally ``adj_close``,
``dividends``, ``splits``. Use :func:`riaguard.quality.check_bars` on the result.

Sources: ``LocalFileSource`` (CSV/CSV.GZ/Parquet), ``SampleSource`` (small cached daily
files shipped with the package, for offline tests), ``YahooSource`` (yfinance extra, no
key), ``StooqSource`` (free CSV, no key), ``PolygonSource`` (skeleton; key from the
``POLYGON_API_KEY`` environment variable, never from code or config files)."""
from __future__ import annotations
import io, json, os, urllib.parse, urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["adj_close", "dividends", "splits"]
SAMPLE_DIR = Path(__file__).parent / "sample_data"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"adjclose": "adj_close", "stock_splits": "splits", "vol": "volume"})
    if not isinstance(df.index, pd.DatetimeIndex):
        tc = next((c for c in ("date", "datetime", "timestamp", "time") if c in df.columns), None)
        if tc is None: raise ValueError("no time column")
        df = df.set_index(tc)
    df.index = pd.to_datetime(df.index, utc=True); df.index.name = "date"
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"missing columns {missing}")
    cols = BAR_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    return df[cols].astype(float).sort_index()


def _clip(df, start, end):
    if start is not None: df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None: df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


class DataSource(ABC):
    name = "abstract"

    @abstractmethod
    def get_bars(self, symbol: str, start=None, end=None, interval: str = "1d") -> pd.DataFrame: ...

    def provenance(self, symbol: str) -> dict:
        return {"source": self.name, "symbol": symbol}


class LocalFileSource(DataSource):
    """Reads ``<root>/<SYMBOL>.{parquet,csv,csv.gz}``."""
    name = "local"

    def __init__(self, root): self.root = Path(root)

    def path(self, symbol):
        for ext in (".parquet", ".csv.gz", ".csv"):
            p = self.root / f"{symbol}{ext}"
            if p.exists(): return p
        raise FileNotFoundError(f"no file for {symbol} in {self.root}")

    def get_bars(self, symbol, start=None, end=None, interval="1d"):
        p = self.path(symbol)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        return _clip(normalize(df), start, end)

    def provenance(self, symbol):
        return {"source": self.name, "symbol": symbol, "path": str(self.path(symbol))}


class SampleSource(LocalFileSource):
    """Daily bars for SPY, QQQ, IWM, TLT, GLD (2010-01-04 to 2026-09-30, downloaded from
    Yahoo Finance via yfinance on 2026-10-31; ``close`` is split-adjusted, ``adj_close``
    also dividend-adjusted). For offline tests and reproducible research only."""
    name = "sample"
    SYMBOLS = ("SPY", "QQQ", "IWM", "TLT", "GLD")

    def __init__(self): super().__init__(SAMPLE_DIR)


class YahooSource(DataSource):
    name = "yahoo"

    def get_bars(self, symbol, start=None, end=None, interval="1d"):
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("pip install 'riaguard[yfinance]'") from e
        df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False,
                         actions=True, progress=False)
        if df is None or df.empty: raise RuntimeError(f"yahoo returned no data for {symbol}")
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return normalize(df)


class StooqSource(DataSource):
    """Free daily CSV from stooq.com (no key). US symbols take a ``.us`` suffix, added
    automatically. Stooq prices are split- and dividend-adjusted. Note: stooq may serve a
    browser challenge page to scripted clients; that raises ``RuntimeError``."""
    name = "stooq"
    URL = "https://stooq.com/q/d/l/?s={s}&i=d"

    def get_bars(self, symbol, start=None, end=None, interval="1d"):
        if interval != "1d": raise ValueError("stooq source supports daily bars only")
        s = symbol.lower() if "." in symbol else f"{symbol.lower()}.us"
        req = urllib.request.Request(self.URL.format(s=urllib.parse.quote(s)), headers={"User-Agent": "riaguard"})
        text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
        if not text.lower().startswith("date"): raise RuntimeError("stooq did not return CSV (blocked or unknown symbol)")
        return _clip(normalize(pd.read_csv(io.StringIO(text))), start, end)


class PolygonSource(DataSource):
    """Polygon.io aggregates (``/v2/aggs``) sized for the **free tier**.

    * The key comes from ``POLYGON_API_KEY`` (or ``api_key=``); never put it in code or config.
    * Free-tier limits are respected by default: at most ``calls_per_minute=5`` requests
      (a process-wide throttle), and HTTP 429 responses are retried with backoff.
    * Free-tier data is end-of-day / delayed with limited history, so this source is for
      daily research and next-day alerts, not real-time monitoring. Paid plans can raise
      ``calls_per_minute``.
    * ``next_url`` pagination is followed. Trades/quotes and corporate-action endpoints are
      not implemented (roadmap phase 2).
    """
    name = "polygon"
    BASE = "https://api.polygon.io"
    ENV = "POLYGON_API_KEY"
    _last_call = [0.0]  # shared across instances: the limit is per key, not per object

    def __init__(self, api_key: str | None = None, calls_per_minute: float = 5, max_retries: int = 3,
                 _sleep=None, _clock=None, _open=None):
        import time
        self.api_key = api_key or os.environ.get(self.ENV)
        if not self.api_key: raise RuntimeError(f"set {self.ENV} to use PolygonSource")
        self.min_interval = 60.0 / calls_per_minute if calls_per_minute else 0.0
        self.max_retries = max_retries
        self._sleep, self._clock = _sleep or time.sleep, _clock or time.monotonic
        self._open = _open or (lambda req: urllib.request.urlopen(req, timeout=30).read())

    def _get(self, url):
        import urllib.error
        for attempt in range(self.max_retries + 1):
            wait = self._last_call[0] + self.min_interval - self._clock()
            if wait > 0: self._sleep(wait)
            self._last_call[0] = self._clock()
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
            try:
                return json.loads(self._open(req))
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == self.max_retries: raise
                self._sleep(max(self.min_interval, 1.0) * (attempt + 1))
        raise RuntimeError("unreachable")  # pragma: no cover

    def get_bars(self, symbol, start=None, end=None, interval="1d"):
        mult, span = {"1d": (1, "day"), "1h": (1, "hour"), "1m": (1, "minute")}[interval]
        start = pd.Timestamp(start or "2000-01-01").strftime("%Y-%m-%d")
        end = pd.Timestamp(end or pd.Timestamp.today()).strftime("%Y-%m-%d")
        url = (f"{self.BASE}/v2/aggs/ticker/{urllib.parse.quote(symbol)}/range/{mult}/{span}/{start}/{end}"
               f"?adjusted=true&sort=asc&limit=50000")
        rows, payload = [], {}
        while url:
            payload = self._get(url); rows += payload.get("results") or []
            url = payload.get("next_url")
        if not rows: raise RuntimeError(f"polygon returned no data for {symbol}: {payload.get('status')}")
        df = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "date"})
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        return normalize(df)


SOURCES = {"local": LocalFileSource, "sample": SampleSource, "yahoo": YahooSource,
           "stooq": StooqSource, "polygon": PolygonSource}
