"""Data-quality checks and corrections for bars and ticks.

``check_bars`` returns a report (it never silently edits data); ``clean_bars`` applies the
fixes you choose. Session times use America/New_York regular hours (09:30-16:00). The
trading calendar is approximated by weekdays minus the regular NYSE holidays;
use an exchange calendar (e.g. pandas_market_calendars) for production work."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday)

NY = "America/New_York"


class NYSEApproxCalendar(AbstractHolidayCalendar):
    """Regular NYSE full-day holidays. Ad-hoc closures (e.g. 2012-10-29/30 Hurricane
    Sandy, national days of mourning) are not included and will show as missing."""
    rules = [Holiday("NewYear", month=1, day=1, observance=nearest_workday), USMartinLutherKingJr,
             USPresidentsDay, GoodFriday, USMemorialDay,
             Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
             Holiday("July4", month=7, day=4, observance=nearest_workday), USLaborDay, USThanksgivingDay,
             Holiday("Christmas", month=12, day=25, observance=nearest_workday)]


def expected_sessions(start, end) -> pd.DatetimeIndex:
    s = pd.Timestamp(start).tz_localize(None).normalize(); e = pd.Timestamp(end).tz_localize(None).normalize()
    days = pd.bdate_range(s, e)
    hol = NYSEApproxCalendar().holidays(s, e)
    # NYSE does not observe New Year's Day on the preceding Friday
    return days[~days.isin(hol) | ((days.month == 12) & (days.day == 31))]


def bad_ticks(close: pd.Series, z: float = 12.0, window: int = 50) -> pd.Series:
    """Flag returns more than ``z`` robust (MAD) deviations from a trailing median, that
    also reverse on the next bar (spike-and-revert): the usual signature of a bad print."""
    r = np.log(close).diff()
    med = r.rolling(window, min_periods=10).median().shift(1)
    mad = (r - med).abs().rolling(window, min_periods=10).median().shift(1) * 1.4826
    score = (r - med) / mad.replace(0, np.nan)
    revert = (np.sign(r) != np.sign(r.shift(-1))) & (r.shift(-1).abs() > 0.8 * r.abs())
    return (score.abs() > z) & revert


def check_bars(df: pd.DataFrame, daily: bool = True) -> dict:
    idx = df.index; rep = {"rows": int(len(df))}
    rep["duplicate_timestamps"] = int(idx.duplicated().sum())
    rep["non_monotonic"] = bool(not idx.is_monotonic_increasing)
    rep["nan_rows"] = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    rep["non_positive_price"] = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    rep["ohlc_inconsistent"] = int(((df["high"] < df[["open", "close", "low"]].max(axis=1) - 1e-9)
                                    | (df["low"] > df[["open", "close", "high"]].min(axis=1) + 1e-9)).sum())
    rep["negative_volume"] = int((df["volume"] < 0).sum())
    bt = bad_ticks(df["close"]); rep["suspect_bad_ticks"] = [str(t.date()) for t in df.index[bt.fillna(False).to_numpy()]]
    if daily and len(df):
        # daily bars are stamped with the session date (midnight UTC): do not shift time zones
        local = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        exp = expected_sessions(local.min(), local.max())
        rep["missing_sessions"] = [str(d.date()) for d in exp.difference(local)]
        rep["unexpected_sessions"] = [str(d.date()) for d in local.difference(exp)]
    if "splits" in df: rep["split_events"] = {str(t.date()): float(v) for t, v in df["splits"].items() if v not in (0, 1)}
    if "dividends" in df: rep["dividend_events"] = int((df["dividends"] > 0).sum())
    rep["ok"] = not (rep["duplicate_timestamps"] or rep["non_monotonic"] or rep["nan_rows"]
                     or rep["non_positive_price"] or rep["ohlc_inconsistent"])
    return rep


def adjust_for_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust OHLC and volume for split events in a ``splits`` column (ratio, e.g. 4.0
    for 4-for-1; 0 means no event). Use only on raw, unadjusted prices."""
    if "splits" not in df: return df.copy()
    s = df["splits"].replace(0, 1.0).fillna(1.0)
    factor = s[::-1].cumprod()[::-1].shift(-1).fillna(1.0)   # product of future splits
    out = df.copy()
    for c in ("open", "high", "low", "close"): out[c] = df[c] / factor
    out["volume"] = df["volume"] * factor
    return out


def total_return_close(df: pd.DataFrame) -> pd.Series:
    """Dividend- and split-adjusted close for return calculations (``adj_close`` if present)."""
    return df["adj_close"] if "adj_close" in df else df["close"]


def clean_bars(df: pd.DataFrame, drop_bad_ticks: bool = False) -> pd.DataFrame:
    out = df[~df.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if drop_bad_ticks: out = out[~bad_ticks(out["close"]).fillna(False)]
    return out


def regular_session(ticks: pd.DataFrame, start="09:30", end="16:00") -> pd.DataFrame:
    """Keep intraday rows within regular US equity hours (America/New_York, DST-aware)."""
    local = ticks.index.tz_convert(NY) if ticks.index.tz is not None else ticks.index.tz_localize("UTC").tz_convert(NY)
    t = local.strftime("%H:%M")
    return ticks[(t >= start) & (t < end) & (local.day_of_week < 5)]
