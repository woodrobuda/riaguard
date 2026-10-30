"""2-D image operators (IMAGE toolbox: imedian, igradient, irotate, igeowarp,
igauss_func, itexture, ifilt_design, igamut) plus an extended set (scale,
morphology, thresholding).  Images are 2-D arrays indexed like VIFF value data
(axis 0 = width/x, axis 1 = height/y); N-D inputs are processed plane-by-plane
over axes (0, 1).
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from .ops_common import plane_apply

_MODES = {'zero': 'constant', 'reflect': 'reflect', 'wrap': 'wrap', 'nearest': 'nearest', 'mirror': 'mirror'}


def imedian(x, wsize=3, hsize=None, border='reflect'):
    hsize = wsize if hsize is None else hsize
    return plane_apply(np.asarray(x), lambda a: ndi.median_filter(a, size=(int(wsize), int(hsize)), mode=_MODES[border]))

_KERNELS = {
    'sobel': (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).T,),
    'prewitt': (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).T,),
}


def igradient(x, operator='sobel', border='nearest'):
    """Gradient: returns (magnitude, direction_radians, gx, gy).
    gx is d/dx along axis 0 (width), gy along axis 1 (height).
    operator: sobel, prewitt, roberts, difference (forward difference)."""
    a = np.asarray(x, float); m = _MODES[border]
    def one(p):
        if operator in _KERNELS:
            k = _KERNELS[operator][0].astype(float)  # k[i,j]: i along x
            gx = ndi.correlate(p, k, mode=m); gy = ndi.correlate(p, k.T, mode=m)
        elif operator == 'roberts':
            gx = ndi.correlate(p, np.array([[1., 0], [0, -1]]), mode=m)
            gy = ndi.correlate(p, np.array([[0., 1], [-1, 0]]), mode=m)
        elif operator == 'difference':
            gx = np.diff(p, axis=0, append=p[-1:, :]); gy = np.diff(p, axis=1, append=p[:, -1:])
        else: raise ValueError(operator)
        return np.stack([np.hypot(gx, gy), np.arctan2(gy, gx), gx, gy], -1)
    if a.ndim == 2: r = one(a); return r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    outs = [plane_apply(a, lambda p, k=k: one(p)[..., k]) for k in range(4)]
    return tuple(outs)


def iedge(x, operator='sobel', threshold=None):
    mag = igradient(x, operator)[0]
    return mag if threshold is None else (mag >= threshold).astype(np.uint8)


def ilaplacian(x, border='nearest'):
    return plane_apply(np.asarray(x, float), lambda a: ndi.laplace(a, mode=_MODES[border]))


def irotate(x, angle, interp=1, background=0.0, reshape=False):
    """Rotate counter-clockwise (viewed with x right, y down: as displayed) by ``angle`` degrees
    about the image centre. interp: 0 nearest, 1 bilinear, 3 cubic."""
    return plane_apply(np.asarray(x, float), lambda a: ndi.rotate(a, -angle, axes=(0, 1), reshape=reshape,
                                                                   order=int(interp), cval=background, mode='constant'))


def iscale(x, wfactor, hfactor=None, interp=1):
    hfactor = wfactor if hfactor is None else hfactor
    return plane_apply(np.asarray(x, float), lambda a: ndi.zoom(a, (wfactor, hfactor), order=int(interp)))


def itranslate(x, dx, dy, interp=1, background=0.0, wrap=False):
    return plane_apply(np.asarray(x, float), lambda a: ndi.shift(a, (dx, dy), order=int(interp),
                                                                  mode='grid-wrap' if wrap else 'constant', cval=background))


def igeowarp(x, xcoef, ycoef, interp=1, background=0.0):
    """Polynomial geometric warp (inverse mapping).  For each output pixel (u,v) the
    source is x_src = sum c_k * term_k, terms = [1, u, v, u*v, u^2, v^2] (len 3 = affine,
    len 4 = bilinear, len 6 = full 2nd order)."""
    a = np.asarray(x, float); xc = np.asarray(xcoef, float); yc = np.asarray(ycoef, float)
    def one(p):
        u, v = np.indices(p.shape, dtype=float)
        terms = [np.ones_like(u), u, v, u * v, u * u, v * v]
        xs = sum(c * t for c, t in zip(xc, terms)); ys = sum(c * t for c, t in zip(yc, terms))
        return ndi.map_coordinates(p, [xs, ys], order=int(interp), cval=background, mode='constant')
    return plane_apply(a, one)


def igauss_func(wsize, hsize, xcenter=None, ycenter=None, xvar=1.0, yvar=None, amplitude=1.0, normalize=False):
    """2-D Gaussian surface g(x,y) = A exp(-((x-xc)^2/(2 xvar) + (y-yc)^2/(2 yvar)))."""
    yvar = xvar if yvar is None else yvar
    xc = (wsize - 1) / 2 if xcenter is None else xcenter; yc = (hsize - 1) / 2 if ycenter is None else ycenter
    u, v = np.indices((int(wsize), int(hsize)), dtype=float)
    g = amplitude * np.exp(-((u - xc) ** 2 / (2 * xvar) + (v - yc) ** 2 / (2 * yvar)))
    return g / g.sum() if normalize else g


def itexture(x, wsize=5, feature='variance'):
    """Local texture features over a wsize x wsize window: mean, variance, energy
    (mean of squares), contrast (max-min), entropy (of a 16-level local histogram)."""
    a = np.asarray(x, float); s = int(wsize)
    def one(p):
        if feature == 'mean': return ndi.uniform_filter(p, s)
        if feature == 'variance':
            m = ndi.uniform_filter(p, s); return np.maximum(ndi.uniform_filter(p * p, s) - m * m, 0)
        if feature == 'energy': return ndi.uniform_filter(p * p, s)
        if feature == 'contrast': return ndi.maximum_filter(p, s) - ndi.minimum_filter(p, s)
        if feature == 'entropy':
            lo, hi = p.min(), p.max(); q = np.zeros(p.shape, int) if hi == lo else np.minimum(((p - lo) / (hi - lo) * 16).astype(int), 15)
            ent = np.zeros(p.shape)
            for lvl in range(16):
                pr = ndi.uniform_filter((q == lvl).astype(float), s)
                ent -= np.where(pr > 1e-12, pr * np.log2(np.maximum(pr, 1e-12)), 0)
            return ent
        raise ValueError(feature)
    return plane_apply(a, one)


def ifilt_design(wsize, hsize, ftype='lowpass', shape='butterworth', cutoff=0.25, cutoff2=None, order=2, centered=True):
    """Frequency-domain filter transfer function on a wsize x hsize grid.
    cutoff in cycles/sample (radial, 0..0.5*sqrt2). ftype lowpass|highpass|bandpass|bandstop;
    shape ideal|butterworth|gaussian. centered=True puts DC at (w//2, h//2) (matches kfft -center 1)."""
    fu = np.fft.fftfreq(int(wsize)); fv = np.fft.fftfreq(int(hsize))
    r = np.hypot(fu[:, None], fv[None, :])
    def lp(c):
        if shape == 'ideal': return (r <= c).astype(float)
        if shape == 'butterworth': return 1 / (1 + (r / c) ** (2 * order))
        if shape == 'gaussian': return np.exp(-(r ** 2) / (2 * c ** 2))
        raise ValueError(shape)
    if ftype == 'lowpass': H = lp(cutoff)
    elif ftype == 'highpass': H = 1 - lp(cutoff)
    elif ftype in ('bandpass', 'bandstop'):
        if cutoff2 is None: raise ValueError('band filters need cutoff2')
        H = lp(max(cutoff, cutoff2)) - lp(min(cutoff, cutoff2))
        if ftype == 'bandstop': H = 1 - H
    else: raise ValueError(ftype)
    return np.fft.fftshift(H) if centered else H


def ifilter_apply(x, H, centered=True):
    a = np.asarray(x, float)
    Hn = np.fft.ifftshift(H) if centered else H
    return plane_apply(a, lambda p: np.real(np.fft.ifft2(np.fft.fft2(p) * Hn)))

# ------------------------------------------------- morphology / threshold
def _se(size, shape='square'):
    s = int(size)
    if shape == 'square': return np.ones((s, s), bool)
    if shape in ('disk', 'circle'):
        r = (s - 1) / 2; u, v = np.indices((s, s)); return (u - r) ** 2 + (v - r) ** 2 <= r * r + 1e-9
    if shape == 'cross':
        m = np.zeros((s, s), bool); m[s // 2, :] = m[:, s // 2] = True; return m
    raise ValueError(shape)


def imorph(x, op='dilate', size=3, shape='square', footprint=None):
    """Grey-level (binary for bool input) morphology: erode, dilate, open, close, gradient, tophat, blacktophat."""
    fp = _se(size, shape) if footprint is None else np.asarray(footprint, bool)
    a = np.asarray(x); binary = a.dtype == bool
    f = {'erode': ndi.grey_erosion, 'dilate': ndi.grey_dilation, 'open': ndi.grey_opening, 'close': ndi.grey_closing,
         'gradient': ndi.morphological_gradient, 'tophat': ndi.white_tophat, 'blacktophat': ndi.black_tophat}[op]
    out = plane_apply(a.astype(np.uint8) if binary else a, lambda p: f(p, footprint=fp))
    return out.astype(bool) if binary else out


def otsu_threshold(x, bins=256):
    a = np.asarray(x, float).ravel(); h, e = np.histogram(a, bins=bins); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0; m0 = np.cumsum(h * c)
    mu0 = m0 / np.maximum(w0, 1); mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)])


def ithreshold(x, low=None, high=None, method=None, inside=1, outside=0):
    """Output ``inside`` where low <= x <= high, else ``outside``. method='otsu' picks low automatically."""
    a = np.asarray(x, float)
    if method == 'otsu': low = otsu_threshold(a); high = None
    m = np.ones(a.shape, bool)
    if low is not None: m &= a >= low
    if high is not None: m &= a <= high
    return np.where(m, inside, outside)


def ilabel(x):
    """Connected-component labelling (8-connected) of non-zero pixels: (labels, count)."""
    return ndi.label(np.asarray(x) != 0, structure=np.ones((3, 3)))

# ------------------------------------------------- colour
def kmeans(data, k, iters=50, seed=0):
    """Plain Lloyd k-means on rows of ``data`` (n, d). Returns (labels, centers)."""
    X = np.asarray(data, float); X = X.reshape(len(X), -1)
    g = np.random.default_rng(seed); C = X[g.choice(len(X), int(k), replace=False)].copy()
    for _ in range(int(iters)):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(len(C))])
        if np.allclose(newC, C): break
        C = newC
    lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
    return lab, C


def igamut(rgb, ncolors=16, seed=0):
    """Colour quantization: (index_image, colormap). ``rgb`` has colour on the last axis."""
    a = np.asarray(rgb, float); flat = a.reshape(-1, a.shape[-1])
    lab, C = kmeans(flat, min(int(ncolors), len(np.unique(flat, axis=0))), seed=seed)
    return lab.reshape(a.shape[:-1]), C
