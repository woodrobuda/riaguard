"""Batch spectral, wavelet and Hilbert tools (scipy / PyWavelets).

WARNING: these are *non-causal over the window they see*. Applying them to a
whole series and then trading on the result leaks the future. Use
``rolling_causal`` to evaluate them only on data available at each step."""
from __future__ import annotations
import numpy as np
from scipy import signal


def periodogram(x, detrend="linear", window="hann", fs=1.0):
    """Windowed periodogram of a (returns / detrended) series."""
    return signal.periodogram(np.asarray(x, float), fs=fs, window=window, detrend=detrend)


def welch(x, nperseg=256, fs=1.0):
    return signal.welch(np.asarray(x, float), fs=fs, nperseg=min(nperseg, len(x)), detrend="linear")


def dominant_period(x, min_period=4, max_period=None):
    """Period (samples) of the largest Welch peak in [min_period, max_period]."""
    f, p = welch(x, nperseg=min(len(x), 256))
    max_period = max_period or len(x) / 2
    m = (f > 0) & (1 / np.where(f > 0, f, np.inf) >= min_period) & (1 / np.where(f > 0, f, np.inf) <= max_period)
    if not m.any(): return float("nan")
    return float(1 / f[m][np.argmax(p[m])])


def butter_lowpass(cutoff_period, order=2):
    """Design (b, a) Butterworth low-pass passing cycles longer than cutoff_period samples."""
    return signal.butter(order, 2.0 / cutoff_period)  # Wn normalised to Nyquist


def zero_phase(x, b, a):
    """filtfilt: zero lag but NON-CAUSAL (uses future samples). Analysis only."""
    return signal.filtfilt(b, a, np.asarray(x, float))


def group_delay(b, a=1, n=512):
    w, gd = signal.group_delay((b, a), w=n); return w / (2 * np.pi), gd


def wavelet_denoise(x, wavelet="db4", level=None, mode="soft"):
    """Donoho-Johnstone universal-threshold wavelet shrinkage (non-causal on the window)."""
    import pywt
    x = np.asarray(x, float)
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs[1:] = [pywt.threshold(c, thr, mode=mode) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="periodization")[: len(x)]


def wavelet_mra(x, wavelet="db4", level=4):
    """Multiresolution: returns list of (name, component) whose sum reconstructs x."""
    import pywt
    x = np.asarray(x, float); coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    out = []
    for i in range(len(coeffs)):
        keep = [c if j == i else np.zeros_like(c) for j, c in enumerate(coeffs)]
        name = f"A{level}" if i == 0 else f"D{level - i + 1}"
        out.append((name, pywt.waverec(keep, wavelet, mode="periodization")[: len(x)]))
    return out


def hilbert_phase(x):
    """Instantaneous amplitude, phase, period from the analytic signal of a
    (band-passed, zero-mean) series. FFT-based => non-causal + end effects."""
    z = signal.hilbert(np.asarray(x, float) - np.mean(x))
    ph = np.unwrap(np.angle(z)); f = np.gradient(ph) / (2 * np.pi)
    with np.errstate(divide="ignore"):
        per = np.where(f > 0, 1 / f, np.nan)
    return np.abs(z), ph, per


def rolling_causal(x, fn, window, min_periods=None):
    """Evaluate fn(x[t-window+1 : t+1])[-1] at each t: uses only past data, so
    any non-causal batch tool becomes a causal (but laggier / end-effect-prone) one."""
    x = np.asarray(x, float); out = np.full(len(x), np.nan); mp = min_periods or window
    for t in range(mp - 1, len(x)):
        seg = x[max(0, t - window + 1): t + 1]; out[t] = np.asarray(fn(seg))[-1]
    return out
