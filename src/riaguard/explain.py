"""Explainable alerts: every change in a signal's direction becomes a JSON record with
its inputs, filter values, the threshold crossed and a hash of the exact config.

Alerts describe *analysis* only. They are not orders and are never sent to a broker."""
from __future__ import annotations
import hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from . import __version__

DISCLAIMER = "Research alert. Not investment advice. No order was placed."


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    if hasattr(v, "item"): v = v.item()
    return v


def alerts_from_signal(symbol: str, sig, close: pd.Series, config: dict | None = None,
                       frame: pd.DataFrame | None = None, lookback_inputs: int = 5) -> list[dict]:
    frame = sig.compute(close) if frame is None else frame
    cfg = {"signal": sig.to_dict(), **(config or {})}
    h = config_hash(cfg)
    d = frame["direction"].to_numpy(); out = []
    for i in range(1, len(d)):
        if d[i] == d[i - 1]: continue
        row = frame.iloc[i]
        out.append({
            "ts": close.index[i].isoformat(), "symbol": symbol, "signal": sig.label,
            "event": "direction_change", "from": int(d[i - 1]), "to": int(d[i]),
            "threshold": sig.threshold, "score": _clean(float(row["score"])),
            "crossed": "up" if d[i] > d[i - 1] else "down",
            "features": {k: _clean(float(row[k])) for k in frame.columns if k not in ("direction", "score")},
            "inputs": {"close_last": [_clean(float(x)) for x in close.iloc[max(0, i - lookback_inputs + 1): i + 1]]},
            "params": list(sig.params), "config_hash": h, "config": cfg,
            "engine_version": __version__, "disclaimer": DISCLAIMER,
        })
    return out


def write_jsonl(records, path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with p.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({**r, "logged_at": stamp}, sort_keys=True) + "\n")
    return p
