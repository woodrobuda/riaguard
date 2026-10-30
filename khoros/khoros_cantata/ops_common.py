"""Shared helpers for the native operator library.

Data model: an operator input may be a Python scalar, a sequence, a NumPy array,
or a :class:`~khoros_cantata.viff.ViffObject`.  For a ViffObject the operator works
on the ``value`` segment (5-D, index order w,h,d,t,e) and returns a *copy* of the
object with the new value data, so attributes and other segments (mask, map,
location, time) travel with the data the way they did through a Cantata pipeline.
"""
from __future__ import annotations
import copy
import numpy as np
from .viff import ViffObject


def value(x):
    """Return the numeric payload of an operator input as an ndarray."""
    if isinstance(x, ViffObject):
        return np.asarray(x.segments['value'].data)
    return np.asarray(x)


def rewrap(template, data):
    """Put ``data`` back into the container type of ``template``."""
    if isinstance(template, ViffObject):
        out = copy.deepcopy(template)
        seg = out.segment('value'); data = np.asarray(data)
        seg.data = data; seg.size = tuple(data.shape)
        return out
    return data


def as5d(a):
    """Promote an array to the VIFF 5-D (w,h,d,t,e) layout."""
    a = np.asarray(a)
    return a.reshape(a.shape + (1,) * (5 - a.ndim)) if a.ndim < 5 else a


def squeeze_like(a, ndim):
    return a.reshape(a.shape[:ndim]) if a.ndim > ndim and all(s == 1 for s in a.shape[ndim:]) else a


def rng(seed=None):
    return np.random.default_rng(None if seed in (None, -1) else int(seed))


def plane_apply(a, fn):
    """Apply a 2-D (w,h) function to every plane of an N-D array (axes 0,1)."""
    a = np.asarray(a)
    if a.ndim <= 2:
        return fn(np.atleast_2d(a) if a.ndim == 2 else a[:, None])[..., 0] if a.ndim == 1 else fn(a)
    rest = a.shape[2:]
    flat = a.reshape(a.shape[:2] + (-1,))
    outs = [fn(flat[:, :, k]) for k in range(flat.shape[2])]
    o = np.stack(outs, axis=-1)
    return o.reshape(o.shape[:2] + rest)
