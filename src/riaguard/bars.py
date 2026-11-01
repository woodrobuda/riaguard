"""Bar construction from irregular ticks: time, tick, volume and dollar bars."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _agg(ticks: pd.DataFrame, group: np.ndarray) -> pd.DataFrame:
    g = ticks.assign(_g=group, _dv=ticks["price"] * ticks["size"]).groupby("_g")
    out = pd.DataFrame({
        "time": g.apply(lambda d: d.index[-1], include_groups=False),
        "open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
        "close": g["price"].last(), "volume": g["size"].sum(), "dollar": g["_dv"].sum(),
        "ticks": g.size()})
    out["vwap"] = out["dollar"] / out["volume"]
    return out.set_index("time")


def time_bars(ticks: pd.DataFrame, freq="1min") -> pd.DataFrame:
    """Calendar bars; empty intervals are dropped (not forward-filled)."""
    r = ticks.resample(freq, label="right", closed="right")
    out = pd.DataFrame({"open": r["price"].first(), "high": r["price"].max(), "low": r["price"].min(),
                        "close": r["price"].last(), "volume": r["size"].sum(), "ticks": r["price"].count()})
    return out.dropna(subset=["close"])


def _threshold_groups(values: np.ndarray, threshold: float) -> np.ndarray:
    """Assign bar ids so that a bar closes on the tick where the running sum reaches threshold."""
    grp = np.empty(len(values), dtype=np.int64); acc = 0.0; b = 0
    for i, v in enumerate(values):
        grp[i] = b; acc += v
        if acc >= threshold:
            b += 1; acc = 0.0
    return grp


def tick_bars(ticks, n_ticks: int):
    return _agg(ticks, np.arange(len(ticks)) // n_ticks)


def volume_bars(ticks, threshold: float):
    return _agg(ticks, _threshold_groups(ticks["size"].to_numpy(float), threshold))


def dollar_bars(ticks, threshold: float):
    return _agg(ticks, _threshold_groups((ticks["price"] * ticks["size"]).to_numpy(float), threshold))
