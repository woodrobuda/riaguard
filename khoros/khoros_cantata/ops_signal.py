"""1-D (and N-D along an axis) signal operators: windows, FIR/IIR, resampling,
noise sources, histograms, Wiener filtering, correlation, convolution, spectra
and wavelets.

Semantics follow the Khoros 2.x DATAMANIP descriptions where they are known
(kgnoise, knoise, kshot, khisto, khistops, kresample, kwiener, klinearop,
kimpulse, kgconst); window/FIR/IIR/correlation/spectral/wavelet operators have
no entry in the 134-operator matrix and are provided as an extended SIGNAL
toolbox with Khoros-style names.  All operators act along the VIFF width axis
(axis 0) unless ``axis`` is given.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as _sig
from .ops_common import rng

# ---------------------------------------------------------------- windows
WINDOWS = ('rect', 'bartlett', 'hanning', 'hamming', 'blackman', 'kaiser', 'gaussian', 'tukey')


def kwindow(n, wtype='hamming', beta=8.6, std=None, alpha=0.5, sym=True):
    """Window generator.  ``sym=False`` gives the periodic (DFT-even) form."""
    n = int(n); wtype = str(wtype).lower()
    if n <= 0: raise ValueError('window length must be positive')
    if wtype in ('rect', 'rectangular', 'boxcar'): return np.ones(n)
    if wtype in ('hanning', 'hann'): return _sig.get_window('hann', n, fftbins=not sym)
    if wtype == 'kaiser': return _sig.get_window(('kaiser', float(beta)), n, fftbins=not sym)
    if wtype == 'gaussian': return _sig.get_window(('gaussian', float(std or n / 6)), n, fftbins=not sym)
    if wtype == 'tukey': return _sig.get_window(('tukey', float(alpha)), n, fftbins=not sym)
    if wtype in ('bartlett', 'hamming', 'blackman'): return _sig.get_window(wtype, n, fftbins=not sym)
    raise ValueError(f'unknown window {wtype!r}; one of {WINDOWS}')


def kapply_window(x, wtype='hamming', axis=0, **kw):
    x = np.asarray(x); w = kwindow(x.shape[axis], wtype, **kw)
    shp = [1] * x.ndim; shp[axis] = -1
    return x * w.reshape(shp)

# ---------------------------------------------------------------- FIR / IIR
def kfir_design(numtaps, cutoff, ftype='lowpass', window='hamming', fs=1.0):
    """Windowed-sinc FIR design. ``cutoff`` in units of ``fs`` (default: cycles/sample, Nyquist 0.5)."""
    ftype = ftype.lower(); pass_zero = {'lowpass': True, 'highpass': False, 'bandpass': False, 'bandstop': True}[ftype]
    if window in ('rect', 'rectangular'): window = 'boxcar'
    if window == 'hanning': window = 'hann'
    numtaps = int(numtaps)
    if not pass_zero or ftype == 'bandstop':
        if numtaps % 2 == 0: raise ValueError('highpass/bandstop FIR needs an odd number of taps')
    return _sig.firwin(numtaps, cutoff, window=window, pass_zero=pass_zero if ftype in ('lowpass', 'highpass') else ftype, fs=fs)


def kfir(x, taps, axis=0, mode='causal'):
    """FIR filter. mode 'causal' = lfilter (same length, delayed); 'same' = centered; 'full'."""
    x = np.asarray(x); taps = np.asarray(taps, float)
    if mode == 'causal': return _sig.lfilter(taps, [1.0], x, axis=axis)
    return np.apply_along_axis(lambda v: np.convolve(v, taps, mode), axis, x)


def kiir_design(order, cutoff, btype='lowpass', family='butter', rp=1.0, rs=40.0, fs=1.0):
    """IIR design, returned as second-order sections (numerically stable)."""
    family = family.lower()
    kw = dict(N=int(order), Wn=cutoff, btype=btype, output='sos', fs=fs)
    if family in ('butter', 'butterworth'): return _sig.butter(**kw)
    if family in ('cheby1', 'chebyshev1'): return _sig.cheby1(rp=rp, **kw)
    if family in ('cheby2', 'chebyshev2'): return _sig.cheby2(rs=rs, **kw)
    if family in ('ellip', 'elliptic'): return _sig.ellip(rp=rp, rs=rs, **kw)
    if family == 'bessel': return _sig.bessel(**kw)
    raise ValueError(f'unknown IIR family {family!r}')


def kiir(x, sos, axis=0, zerophase=False):
    x = np.asarray(x, float); sos = np.atleast_2d(np.asarray(sos, float))
    return _sig.sosfiltfilt(sos, x, axis=axis) if zerophase else _sig.sosfilt(sos, x, axis=axis)


def kfreqz(b_or_sos, a=None, n=512):
    """Frequency response (freq in cycles/sample 0..0.5, complex h)."""
    b = np.asarray(b_or_sos, float)
    if a is None and b.ndim == 2 and b.shape[1] == 6:
        w, h = _sig.sosfreqz(b, worN=n)
    else:
        w, h = _sig.freqz(b, [1.0] if a is None else a, worN=n)
    return w / (2 * np.pi), h

# ---------------------------------------------------------------- convolution / correlation
def kconvolve(x, k, mode='full'):
    """N-D linear convolution via FFT (vectorized replacement for loops)."""
    return _sig.fftconvolve(np.asarray(x), np.asarray(k), mode=mode)


def kcorrelate(x, y=None, mode='full', normalize=None):
    """Cross-correlation r[l] = sum_n x[n+l] conj(y[n]); auto when y is None.
    normalize: None, 'biased' (/N), 'unbiased' (/(N-|l|)), 'coeff' (r[0]=1 for auto)."""
    x = np.asarray(x).ravel(); y = x if y is None else np.asarray(y).ravel()
    r = _sig.correlate(x, y, mode=mode, method='fft' if x.size > 64 else 'direct')
    if normalize is None: return r
    lags = _sig.correlation_lags(x.size, y.size, mode=mode); n = max(x.size, y.size)
    if normalize == 'biased': return r / n
    if normalize == 'unbiased': return r / (n - np.abs(lags))
    if normalize == 'coeff': return r / np.sqrt(np.sum(np.abs(x) ** 2) * np.sum(np.abs(y) ** 2))
    raise ValueError(normalize)

# ---------------------------------------------------------------- spectra
def kpsd(x, fs=1.0, method='periodogram', window='hann', nperseg=None, scaling='density'):
    x = np.asarray(x, float).ravel()
    if method == 'periodogram': return _sig.periodogram(x, fs=fs, window=window, scaling=scaling)
    if method == 'welch': return _sig.welch(x, fs=fs, window=window, nperseg=nperseg or min(256, x.size), scaling=scaling)
    raise ValueError(method)


def kspectrum(x, axis=0, output='magnitude', db=False):
    f = np.fft.fft(np.asarray(x), axis=axis)
    out = {'magnitude': np.abs(f), 'power': np.abs(f) ** 2, 'phase': np.angle(f)}[output]
    return 20 * np.log10(np.maximum(out, 1e-300)) if db and output == 'magnitude' else 10 * np.log10(np.maximum(out, 1e-300)) if db and output == 'power' else out

# ---------------------------------------------------------------- resampling
def _resample_axis(x, n, axis, interp):
    m = x.shape[axis]
    if n == m: return x
    if interp == 'fft': return _sig.resample(x, n, axis=axis)
    # Khoros samples the input at positions j * (m/n) (zero-order = nearest-lower, linear = interpolate)
    pos = np.arange(n) * (m / n)
    if interp in ('zero', 'nearest', 0):
        idx = np.minimum(np.floor(pos + 1e-9).astype(int), m - 1); return np.take(x, idx, axis=axis)
    if interp in ('linear', 1):
        i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, m - 1); f = pos - i0
        shp = [1] * x.ndim; shp[axis] = -1; f = f.reshape(shp)
        return np.take(x, i0, axis=axis) * (1 - f) + np.take(x, i1, axis=axis) * f
    raise ValueError(f'unknown interpolation {interp!r}')


def kresample(x, sizes=None, factors=None, interp='linear'):
    """Resample along each axis to ``sizes`` (tuple; None = keep) or by ``factors``.
    interp: 'zero' (zero-order hold), 'linear', or 'fft' (band-limited)."""
    x = np.asarray(x)
    if sizes is None and factors is None: raise ValueError('give sizes or factors')
    if sizes is None:
        f = np.atleast_1d(factors); sizes = [int(round(x.shape[a] * (f[a] if a < f.size else 1))) for a in range(x.ndim)]
    sizes = list(np.atleast_1d(sizes)) + [None] * x.ndim
    for a in range(x.ndim):
        if sizes[a] is not None and int(sizes[a]) > 0: x = _resample_axis(x, int(sizes[a]), a, interp)
    return x

# ---------------------------------------------------------------- sources & noise
def kgconst(shape, value=0.0, dtype=float):
    return np.full(shape, value, dtype=complex if np.iscomplexobj(value) else dtype)


def kimpulse(shape, position=None, amplitude=1.0, spacing=None):
    """Impulse (or impulse train with ``spacing`` along axis 0)."""
    o = np.zeros(shape); shape = o.shape
    if spacing:
        sl = [0] * o.ndim; sl[0] = slice(int(position[0]) if position else 0, None, int(spacing)); o[tuple(sl)] = amplitude
    else:
        o[tuple(position) if position else (0,) * o.ndim] = amplitude
    return o


def kgnoise(shape, ntype='gaussian', mean=0.0, variance=1.0, low=0.0, high=1.0, seed=None):
    """Noise generator. ntype: gaussian, uniform, rayleigh, poisson, exponential.
    For rayleigh/exponential ``variance`` sets the distribution's variance."""
    g = rng(seed); ntype = ntype.lower()
    if ntype == 'gaussian': return g.normal(mean, np.sqrt(variance), shape)
    if ntype == 'uniform': return g.uniform(low, high, shape)
    if ntype == 'rayleigh': return g.rayleigh(np.sqrt(variance / (2 - np.pi / 2)), shape)
    if ntype == 'exponential': return g.exponential(np.sqrt(variance), shape)
    if ntype == 'poisson': return g.poisson(mean if mean > 0 else variance, shape).astype(float)
    raise ValueError(f'unknown noise type {ntype!r}')


def knoise(x, ntype='gaussian', mode='additive', seed=None, **kw):
    """Introduce noise into data: additive (x+n) or multiplicative (x*n).
    ntype='poisson' with mode='signal' draws Poisson counts with mean x."""
    x = np.asarray(x, float)
    if mode == 'signal': return rng(seed).poisson(np.maximum(x, 0)).astype(float)
    n = kgnoise(x.shape, ntype, seed=seed, **kw)
    if mode == 'additive': return x + n
    if mode == 'multiplicative': return x * n
    raise ValueError(mode)


def kshot(x, probability=0.05, low=None, high=None, seed=None):
    """Shot (salt-and-pepper/impulse) noise: each sample replaced with ``low`` or
    ``high`` (defaults: data min/max) with total probability ``probability``."""
    x = np.array(x, float); g = rng(seed)
    lo = x.min() if low is None else low; hi = x.max() if high is None else high
    r = g.random(x.shape); x[r < probability / 2] = lo; x[(r >= probability / 2) & (r < probability)] = hi
    return x

# ---------------------------------------------------------------- histograms
def khisto(x, bins=256, vmin=None, vmax=None, normalize=False):
    """Histogram; returns (counts, bin_edges). Values outside [vmin,vmax] are ignored."""
    x = np.asarray(x).ravel()
    rng_ = (x.min() if vmin is None else vmin, x.max() if vmax is None else vmax)
    h, e = np.histogram(x, bins=int(bins), range=rng_)
    return (h / max(h.sum(), 1) if normalize else h), e


def khistops(x, op='equalize', bins=256, vmin=None, vmax=None, target=None):
    """Histogram operations: 'equalize' (CDF mapping into [vmin,vmax]),
    'stretch' (linear contrast stretch to [vmin,vmax]), 'match' (to ``target`` data)."""
    x = np.asarray(x, float); lo, hi = x.min(), x.max()
    omin = lo if vmin is None else vmin; omax = hi if vmax is None else vmax
    if op == 'stretch':
        return np.full_like(x, omin) if hi == lo else (x - lo) / (hi - lo) * (omax - omin) + omin
    if op == 'equalize':
        h, e = np.histogram(x, bins=int(bins), range=(lo, hi)); cdf = np.cumsum(h) / x.size
        return np.interp(x, e[1:], cdf) * (omax - omin) + omin
    if op == 'match':
        t = np.sort(np.asarray(target, float).ravel())
        ranks = np.argsort(np.argsort(x.ravel(), kind='stable'), kind='stable') / max(x.size - 1, 1)
        return np.interp(ranks, np.linspace(0, 1, t.size), t).reshape(x.shape)
    raise ValueError(op)

# ---------------------------------------------------------------- Wiener
def kwiener(x, psf=None, nsr=0.01, axes=None, adaptive_size=None, noise=None):
    """Wiener filter.
    * With ``psf``: frequency-domain restoration W = H* / (|H|^2 + NSR), where NSR is
      a scalar or an array (noise PSD / signal PSD) on the FFT grid. ``psf`` origin at index 0.
    * Without ``psf``: adaptive local Wiener smoothing (window ``adaptive_size``)."""
    x = np.asarray(x, float)
    if psf is None:
        return _sig.wiener(x, mysize=adaptive_size or 3, noise=noise)
    axes = tuple(range(x.ndim)) if axes is None else tuple(axes)
    psf = np.asarray(psf, float); pad = np.zeros(x.shape); pad[tuple(slice(0, s) for s in psf.shape)] = psf
    H = np.fft.fftn(pad, axes=axes); X = np.fft.fftn(x, axes=axes)
    W = np.conj(H) / (np.abs(H) ** 2 + nsr)
    return np.real(np.fft.ifftn(W * X, axes=axes))

# ---------------------------------------------------------------- wavelets
_WAVELETS = {
    'haar': np.array([1, 1]) / np.sqrt(2),
    'db2': np.array([1 + np.sqrt(3), 3 + np.sqrt(3), 3 - np.sqrt(3), 1 - np.sqrt(3)]) / (4 * np.sqrt(2)),
}


def _qmf(h): return h[::-1] * (-1) ** np.arange(h.size)


def kdwt(x, wavelet='haar', levels=1):
    """Periodic orthogonal DWT; returns [cA_n, cD_n, ..., cD_1]. len(x) must be divisible by 2**levels."""
    h = _WAVELETS[wavelet]; g = _qmf(h); a = np.asarray(x, float).ravel(); out = []
    for _ in range(int(levels)):
        n = a.size
        if n % 2: raise ValueError('length not divisible by 2**levels')
        idx = (np.arange(0, n, 2)[:, None] + np.arange(h.size)[None, :]) % n
        out.insert(0, a[idx] @ g); a = a[idx] @ h
    return [a] + out


def kidwt(coeffs, wavelet='haar'):
    h = _WAVELETS[wavelet]; g = _qmf(h); a = np.asarray(coeffs[0], float)
    for d in coeffs[1:]:
        n = a.size * 2; y = np.zeros(n)
        idx = (np.arange(0, n, 2)[:, None] + np.arange(h.size)[None, :]) % n
        np.add.at(y, idx, a[:, None] * h[None, :] + np.asarray(d)[:, None] * g[None, :]); a = y
    return a

# ---------------------------------------------------------------- causal time-series primitives
def kema(x, span=None, alpha=None):
    """Exponential moving average along axis 0: y0 = x0, y_t = a x_t + (1-a) y_{t-1};
    a = alpha or 2/(span+1)."""
    a = float(alpha) if alpha is not None else 2.0 / (float(span or 20) + 1)
    x = np.asarray(x, float)
    if x.shape[0] == 0: return x.copy()
    zi = (1 - a) * x[0]
    y, _ = _sig.lfilter([a], [1, -(1 - a)], x[1:], axis=0, zi=np.expand_dims(zi, 0)) if x.shape[0] > 1 else (x[1:], None)
    return np.concatenate([x[:1], y], axis=0)


def krunstats(x, stat='mean', window=None):
    """Causal running statistic along axis 0 (mean, var, std, min, max, sum).
    window=None: cumulative from the start; window=n: trailing window of up to n samples."""
    x = np.asarray(x, float); n = x.shape[0]
    if window is None:
        c = np.arange(1, n + 1).reshape((-1,) + (1,) * (x.ndim - 1))
        if stat == 'sum': return np.cumsum(x, 0)
        if stat == 'mean': return np.cumsum(x, 0) / c
        if stat in ('var', 'std'):
            m = np.cumsum(x, 0) / c; v = np.maximum(np.cumsum(x * x, 0) / c - m * m, 0)
            return np.sqrt(v) if stat == 'std' else v
        if stat == 'min': return np.minimum.accumulate(x, 0)
        if stat == 'max': return np.maximum.accumulate(x, 0)
        raise ValueError(stat)
    w = int(window); out = np.empty_like(x)
    f = {'mean': np.mean, 'var': np.var, 'std': np.std, 'min': np.min, 'max': np.max, 'sum': np.sum}[stat]
    for t in range(n): out[t] = f(x[max(0, t - w + 1):t + 1], axis=0)
    return out


def kdelay(x, n=1, fill=0.0):
    x = np.asarray(x, float); n = int(n)
    if n == 0: return x.copy()
    return np.concatenate([np.full((min(n, x.shape[0]),) + x.shape[1:], fill), x[:max(x.shape[0] - n, 0)]], 0)


def kdiff(x, fill=0.0):
    """First difference y_t = x_t - x_{t-1}; y_0 = fill."""
    x = np.asarray(x, float); return np.concatenate([np.full((min(1, x.shape[0]),) + x.shape[1:], fill), np.diff(x, axis=0)], 0)


def kcross(a, b):
    """Crossover events: +1 where a crosses above b at t, -1 where it crosses below, else 0 (t=0 -> 0)."""
    d = np.sign(np.asarray(a, float) - np.asarray(b, float)); above = d > 0
    out = np.zeros(d.shape)
    if d.shape[0] > 1:
        out[1:] = np.where(above[1:] & ~above[:-1], 1.0, np.where(~above[1:] & above[:-1], -1.0, 0.0))
    return out


def khilbert(x, output='envelope'):
    """Analytic-signal (Hilbert) transform along axis 0: envelope, phase, or imag (Hilbert transform).
    Uses the whole record (FFT), so it is NOT causal."""
    z = _sig.hilbert(np.asarray(x, float), axis=0)
    return {'envelope': np.abs(z), 'phase': np.unwrap(np.angle(z), axis=0), 'imag': np.imag(z)}[output]
