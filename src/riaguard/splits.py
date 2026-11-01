"""Time-series train/test splits that respect causality.

* ``walk_forward_splits``: rolling (or expanding) train window followed by a test window.
* ``purged_kfold``: k contiguous test folds; training indices within ``purge`` bars before
  a test fold (labels that overlap it) and ``embargo`` bars after it are removed
  (Lopez de Prado, *Advances in Financial ML*, ch. 7).
All functions return lists of (train_idx, test_idx) integer numpy arrays."""
from __future__ import annotations
import numpy as np


def walk_forward_splits(n: int, train: int, test: int, step: int | None = None,
                        expanding: bool = False, gap: int = 0):
    """``gap`` bars are dropped between train end and test start (embargo)."""
    if train <= 0 or test <= 0: raise ValueError("train and test must be > 0")
    step = step or test; out = []
    start = 0
    while start + train + gap < n:
        tr_lo = 0 if expanding else start
        tr = np.arange(tr_lo, start + train)
        te = np.arange(start + train + gap, min(start + train + gap + test, n))
        if len(te) == 0: break
        out.append((tr, te)); start += step
    return out


def purged_kfold(n: int, k: int = 5, purge: int = 0, embargo: int = 0):
    if k < 2: raise ValueError("k >= 2")
    folds = np.array_split(np.arange(n), k); out = []
    for te in folds:
        lo, hi = te[0], te[-1]
        idx = np.arange(n)
        keep = (idx < lo - purge) | (idx > hi + embargo)
        out.append((idx[keep], te))
    return out
