"""DATAMANIP arithmetic, logical, complex and data-manipulation operators,
segment operators on ViffObjects, the MATRIX toolbox, and small ENVISION/
GEOMETRY colour/extent helpers.

Matrix convention: a 2-D ndarray is a matrix in NumPy (row, col) order.  A
ViffObject is converted with rows = height, cols = width (``value[:, :, 0, 0, 0].T``).
"""
from __future__ import annotations
import copy, io
import numpy as np
from scipy import special as _sp, linalg as _la
from .viff import ViffObject

# ---------------------------------------------------------------- arithmetic
UNARY = {
    'abs': np.abs, 'neg': np.negative, 'sqrt': np.sqrt, 'sqr': np.square, 'cube': lambda x: x ** 3, 'cbrt': np.cbrt,
    'recip': lambda x: np.reciprocal(np.asarray(x, dtype=np.result_type(x, float))), 'exp': np.exp, 'log': np.log,
    'log10': np.log10, 'log2': np.log2, 'exp10': lambda x: np.power(10.0, x), 'exp2': np.exp2,
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh, 'asinh': np.arcsinh, 'acosh': np.arccosh, 'atanh': np.arctanh,
    'floor': np.floor, 'ceil': np.ceil, 'round': np.round, 'trunc': np.trunc, 'sign': np.sign,
    'erf': _sp.erf, 'erfc': _sp.erfc, 'gamma': _sp.gamma, 'lgamma': _sp.gammaln,
}
BINARY = {
    'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'div': np.true_divide, 'divinto': lambda a, b: np.true_divide(b, a),
    'pow': np.power, 'min': np.minimum, 'max': np.maximum, 'mod': np.mod, 'atan2': np.arctan2, 'hypot': np.hypot,
    'absdiff': lambda a, b: np.abs(a - b), 'and': np.logical_and, 'or': np.logical_or, 'xor': np.logical_xor,
}
BITWISE = {'and': np.bitwise_and, 'or': np.bitwise_or, 'xor': np.bitwise_xor, 'lshift': np.left_shift,
           'rshift': np.right_shift, 'nand': lambda a, b: ~(a & b), 'nor': lambda a, b: ~(a | b)}
COMPARE = {'eq': np.equal, 'ne': np.not_equal, 'lt': np.less, 'le': np.less_equal, 'gt': np.greater, 'ge': np.greater_equal}


def karith1(x, op):
    if op not in UNARY: raise ValueError(f'karith1: unknown function {op!r}')
    return UNARY[op](np.asarray(x))


def karith2(a, b, op):
    if op not in BINARY: raise ValueError(f'karith2: unknown operation {op!r}')
    return BINARY[op](np.asarray(a), np.asarray(b))


def kbitwise(a, b=None, op='and'):
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.integer) and a.dtype != bool: a = a.astype(np.int64)
    if op == 'not': return np.invert(a)
    b = np.asarray(b); b = b if np.issubdtype(b.dtype, np.integer) or b.dtype == bool else b.astype(np.int64)
    return BITWISE[op](a, b)


def kcompare(a, b, op='eq', true=1, false=0, tolerance=0.0):
    a = np.asarray(a); b = np.asarray(b)
    if op == 'eq' and tolerance: m = np.abs(a - b) <= tolerance
    elif op == 'ne' and tolerance: m = np.abs(a - b) > tolerance
    else: m = COMPARE[op](a, b)
    return np.where(m, true, false)


def klogexp(x, op='log', base=None):
    x = np.asarray(x, float)
    if base is not None: return np.log(x) / np.log(base) if op == 'log' else np.power(float(base), x)
    return UNARY[op](x)


def kbessel(x, kind='j', order=0):
    """Bessel functions: j (1st kind), y (2nd kind), i (modified 1st), k (modified 2nd)."""
    f = {'j': _sp.jv, 'y': _sp.yv, 'i': _sp.iv, 'k': _sp.kv}[kind]
    return f(order, np.asarray(x, float))


def kblend(a, b, alpha=0.5):
    """o = alpha * a + (1 - alpha) * b (alpha may be an array)."""
    return alpha * np.asarray(a, float) + (1 - np.asarray(alpha, float)) * np.asarray(b, float)


def kgate(x, gate, value=0.0, invert=False):
    """Pass x where gate is non-zero (zero with invert), else ``value``."""
    g = np.asarray(gate) != 0
    return np.where(~g if invert else g, np.asarray(x), value)

# ---------------------------------------------------------------- complex
def kcmplx(x, op='conj'):
    x = np.asarray(x)
    return {'conj': np.conj, 'mag': np.abs, 'magsq': lambda z: np.abs(z) ** 2, 'phase': np.angle, 'real': np.real,
            'imag': np.imag, 'logmag': lambda z: np.log(np.abs(z) + 1)}[op](x)


def kreal2cmplx(re, im=None):
    re = np.asarray(re, float)
    return re + 1j * (np.zeros_like(re) if im is None else np.asarray(im, float))


def kcmplx2real(x, part='real'):
    return kcmplx(x, part)

# ---------------------------------------------------------------- type / value manipulation
def kconvert(x, dtype='float', scale=None, offset=0.0, normalize=False, clip=True):
    """Convert data type. normalize=True maps [min,max] onto the target integer range;
    clip=True saturates (Khoros-style) instead of wrapping."""
    x = np.asarray(x); dt = np.dtype({'bit': bool, 'byte': np.int8, 'ubyte': np.uint8, 'short': np.int16, 'ushort': np.uint16,
                                      'int': np.int32, 'uint': np.uint32, 'long': np.int64, 'float': np.float32,
                                      'double': np.float64, 'complex': np.complex64, 'dcomplex': np.complex128}.get(dtype, dtype))
    y = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64)
    if normalize and np.issubdtype(dt, np.integer):
        info = np.iinfo(dt); lo, hi = y.real.min(), y.real.max()
        y = (y.real - lo) / (hi - lo if hi > lo else 1) * (info.max - info.min) + info.min
    elif scale is not None: y = y * scale + offset
    if np.issubdtype(dt, np.integer):
        y = np.real(y); info = np.iinfo(dt); y = np.round(y)
        if clip: y = np.clip(y, info.min, info.max)
    elif dt == bool: return y != 0
    elif not np.issubdtype(dt, np.complexfloating): y = np.real(y)
    return y.astype(dt)


def knormal(x, method='minmax', vmin=0.0, vmax=1.0):
    x = np.asarray(x, float)
    if method == 'minmax':
        lo, hi = x.min(), x.max(); return np.full_like(x, vmin) if hi == lo else (x - lo) / (hi - lo) * (vmax - vmin) + vmin
    if method == 'zscore': s = x.std(); return (x - x.mean()) / (s if s else 1)
    if method == 'sum': return x / x.sum()
    if method == 'max': return x / np.abs(x).max()
    raise ValueError(method)


def kreplace(x, low, high=None, new=0.0, outside=False):
    """Replace values in [low, high] (or outside it) with ``new``."""
    x = np.asarray(x).copy(); high = low if high is None else high
    m = (x >= low) & (x <= high)
    x[~m if outside else m] = new; return x


def kmapdata(x, cmap, clip=True):
    """Look-up-table mapping: o[...] = map[int(x[...])]; map may be (n,) or (n, k)."""
    idx = np.asarray(x).astype(np.int64); m = np.asarray(cmap)
    if clip: idx = np.clip(idx, 0, len(m) - 1)
    return m[idx]


def kunmask(x, mask, value=0.0):
    """Replace samples whose mask is 0 with ``value`` (Khoros: mask 1 = valid)."""
    return np.where(np.asarray(mask) != 0, np.asarray(x), value)

# ---------------------------------------------------------------- size / region
def kextract(x, offset, size):
    x = np.asarray(x); off = list(offset) + [0] * (x.ndim - len(offset)); sz = list(size) + [None] * (x.ndim - len(size))
    return x[tuple(slice(o, None if s is None else o + s) for o, s in zip(off, sz))].copy()


def kinset(x, y, offset):
    """Insert ``y`` into a copy of ``x`` at ``offset``; parts falling outside are cropped."""
    x = np.array(x); y = np.asarray(y); off = list(offset) + [0] * (x.ndim - len(offset))
    if y.ndim < x.ndim: y = y.reshape(y.shape + (1,) * (x.ndim - y.ndim))
    dst = tuple(slice(o, min(o + s, n)) for o, s, n in zip(off, y.shape, x.shape))
    src = tuple(slice(0, d.stop - d.start) for d in dst)
    x[dst] = y[src]; return x


def kpad(x, size, offset=None, value=0.0):
    """Pad to ``size`` placing data at ``offset`` (Khoros kpad -wsize -hsize -woff -hoff -real)."""
    x = np.asarray(x); size = list(size) + list(x.shape[len(size):])
    out = np.full(size, value, dtype=np.result_type(x, type(value)))
    return kinset(out, x, offset or [0] * x.ndim)


def kflip(x, axes=(0,)):
    return np.flip(np.asarray(x), axis=tuple(axes))


def korient(x, order):
    """Reorder axes; ``order`` is a permutation such as 'hwdte' or (1,0,2,3,4)."""
    x = np.asarray(x)
    if isinstance(order, str): order = ['whdte'.index(c) for c in order]
    order = list(order); a = x.reshape(x.shape + (1,) * (len(order) - x.ndim)) if x.ndim < len(order) else x
    return np.transpose(a, order)


def kappend(a, b, axis=0):
    a = np.asarray(a); b = np.asarray(b)
    return np.concatenate([a, b], axis=axis)


def kaxis(shape, axis=0, start=0.0, step=1.0):
    """Ramp data whose value is the coordinate along ``axis``."""
    return start + step * np.indices(shape, dtype=float)[axis]


def kgenloc(shape, origin=None, spacing=None):
    """Explicit rectilinear location data: array shape + (ndim,) of coordinates."""
    shape = tuple(shape); origin = origin or [0.0] * len(shape); spacing = spacing or [1.0] * len(shape)
    g = np.indices(shape, dtype=float)
    return np.stack([origin[i] + spacing[i] * g[i] for i in range(len(shape))], -1)


def klocxform(loc, matrix, translate=None):
    """Affine transform of location data (last axis = coordinate)."""
    loc = np.asarray(loc, float); out = loc @ np.asarray(matrix, float).T
    return out + np.asarray(translate, float) if translate is not None else out


def ksampline(x, p0, p1, n=None, interp=1):
    """Sample a 2-D image along the line p0->p1 (points in (x, y) = (axis0, axis1))."""
    from scipy import ndimage as ndi
    a = np.asarray(x, float); p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    n = int(n or np.ceil(np.hypot(*(p1 - p0))) + 1); t = np.linspace(0, 1, n)
    pts = p0[:, None] + (p1 - p0)[:, None] * t
    return ndi.map_coordinates(a, pts, order=int(interp), mode='nearest')


def kslicer(x, axis=2, index=0):
    return np.take(np.asarray(x), int(index), axis=axis)

# ---------------------------------------------------------------- import / print
def kimportasc(text, shape=None, dtype=float, comment='#'):
    """Parse whitespace/comma separated ASCII numbers (replacement for np.fromstring)."""
    if isinstance(text, (bytes, bytearray)): text = text.decode()
    toks = []
    for line in str(text).splitlines():
        line = line.split(comment, 1)[0] if comment else line
        toks += line.replace(',', ' ').split()
    a = np.array(toks, dtype=dtype)
    return a.reshape(shape) if shape is not None else a


def kimportraw(raw, dtype='float32', shape=None, offset=0, endian='little'):
    dt = np.dtype(dtype).newbyteorder('<' if endian == 'little' else '>')
    a = np.frombuffer(bytes(raw), dtype=dt, offset=int(offset)).astype(np.dtype(dtype))
    return a.reshape(shape) if shape is not None else a


def kprdata(x, fmt='%.6g', per_line=None):
    a = np.asarray(x); buf = io.StringIO()
    flat = a.reshape(-1, a.shape[-1] if a.ndim > 1 else a.size) if per_line is None else a.reshape(-1, int(per_line))
    np.savetxt(buf, flat, fmt=fmt); return buf.getvalue()


def kprval(x, index, fmt='%.6g'):
    return fmt % np.asarray(x)[tuple(index)]

# ---------------------------------------------------------------- VIFF segment operators
def _obj(o): return o if isinstance(o, ViffObject) else ViffObject()


def kcpseg(dst, src, segment):
    out = copy.deepcopy(_obj(dst)); out.segments[segment] = copy.deepcopy(src.segments[segment]); return out


def krmseg(obj, segment):
    out = copy.deepcopy(obj); out.segments.pop(segment, None); return out


def kinsertseg(obj, segment, data):
    out = copy.deepcopy(_obj(obj)); s = out.segment(segment); s.data = np.asarray(data); s.size = s.data.shape; return out


def kcptoval(obj, segment):
    """Copy a segment (mask/map/location/time) into the value segment."""
    return kinsertseg(obj, 'value', obj.segments[segment].data)


def kcpfromval(obj, segment):
    return kinsertseg(obj, segment, obj.segments['value'].data)


def kasc2seg(obj, text, segment='value', shape=None):
    """kasc2val/kasc2mask/kasc2map/kasc2loc/kasc2time: ASCII -> named segment."""
    return kinsertseg(obj, segment, kimportasc(text, shape))


def ksegcmp(a, b, segment='value', tol=0.0):
    sa, sb = a.segments.get(segment), b.segments.get(segment)
    if sa is None or sb is None: return sa is sb
    da, db = np.asarray(sa.data), np.asarray(sb.data)
    return da.shape == db.shape and bool(np.all(np.abs(da.astype(complex) - db.astype(complex)) <= tol))


def ksetdattr(obj, name, val, segment=None):
    out = copy.deepcopy(_obj(obj)); (out.segment(segment).attributes if segment else out.attributes)[name] = val; return out


def kcomment(obj, comment):
    return ksetdattr(obj, 'comment', comment)

# ---------------------------------------------------------------- MATRIX toolbox
def as_matrix(x):
    if isinstance(x, ViffObject):
        v = np.asarray(x.segments['value'].data); v = v.reshape(v.shape + (1,) * (5 - v.ndim)) if v.ndim < 5 else v
        return v[:, :, 0, 0, 0].T
    return np.atleast_2d(np.asarray(x))


def mmul(a, b): return as_matrix(a) @ as_matrix(b)
def minvert(a, pseudo=False): m = as_matrix(a); return np.linalg.pinv(m) if pseudo else np.linalg.inv(m)
def mpow(a, n): return np.linalg.matrix_power(as_matrix(a), int(n))
def mextract_diag(a, k=0): return np.diagonal(as_matrix(a), k).copy()
def mgdiag(v, k=0): return np.diag(np.asarray(v).ravel(), k)
def mreplicate(a, reps): return np.tile(as_matrix(a), reps)
def mrcsum(a, axis='rows'): return as_matrix(a).sum(axis=1 if axis == 'rows' else 0)
def mcovar(a, rowvar=False, bias=False): return np.atleast_2d(np.cov(as_matrix(a), rowvar=rowvar, bias=bias))


def mexchg(a, i, j, what='rows'):
    m = as_matrix(a).copy()
    if what == 'rows': m[[i, j], :] = m[[j, i], :]
    else: m[:, [i, j]] = m[:, [j, i]]
    return m


def meigen(a, symmetric=None):
    m = as_matrix(a); symmetric = np.allclose(m, m.conj().T) if symmetric is None else symmetric
    return np.linalg.eigh(m) if symmetric else np.linalg.eig(m)


def msvd(a, full=False): return np.linalg.svd(as_matrix(a), full_matrices=full)
def mlud(a): return _la.lu(as_matrix(a))  # (P, L, U) with A = P @ L @ U


def mlse(a, b):
    """Least-squares solution x minimising ||A x - b||; returns (x, residuals, rank, sv)."""
    return np.linalg.lstsq(as_matrix(a), np.asarray(b, float), rcond=None)

# ---------------------------------------------------------------- ENVISION / GEOMETRY helpers
def kcolorspace(x, conversion='rgb2hsv'):
    """Colour model conversion; colour on the last axis, RGB in [0,1].
    rgb2hsv, hsv2rgb, rgb2yiq, yiq2rgb, rgb2grey."""
    a = np.asarray(x, float)
    YIQ = np.array([[0.299, 0.587, 0.114], [0.595716, -0.274453, -0.321263], [0.211456, -0.522591, 0.311135]])
    if conversion == 'rgb2yiq': return a @ YIQ.T
    if conversion == 'yiq2rgb': return a @ np.linalg.inv(YIQ).T
    if conversion == 'rgb2grey': return a @ YIQ[0]
    if conversion == 'rgb2hsv':
        r, g, b = a[..., 0], a[..., 1], a[..., 2]; mx = a.max(-1); mn = a.min(-1); d = mx - mn
        dd = np.where(d == 0, 1, d)
        h = np.select([d == 0, mx == r, mx == g], [0.0, ((g - b) / dd) % 6, (b - r) / dd + 2], (r - g) / dd + 4) / 6
        return np.stack([h, np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx)), mx], -1)
    if conversion == 'hsv2rgb':
        h, s_, v = a[..., 0] * 6, a[..., 1], a[..., 2]; i = np.floor(h).astype(int) % 6; f = h - np.floor(h)
        p_, q, t = v * (1 - s_), v * (1 - s_ * f), v * (1 - s_ * (1 - f))
        r = np.choose(i, [v, q, p_, p_, t, v]); g = np.choose(i, [t, v, v, q, p_, p_]); b = np.choose(i, [p_, p_, t, v, v, q])
        return np.stack([r, g, b], -1)
    raise ValueError(conversion)


def ggencmap(n=256, scheme='grey'):
    t = np.linspace(0, 1, int(n))
    if scheme == 'grey': return np.stack([t, t, t], -1)
    if scheme == 'hot': return np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], -1)
    if scheme == 'rainbow':
        h = (1 - t) * 2 / 3; i = np.floor(h * 6).astype(int) % 6; f = h * 6 - np.floor(h * 6)
        q = 1 - f; tab = [(1, f, 0), (q, 1, 0), (0, 1, f), (0, q, 1), (f, 0, 1), (1, 0, q)]
        out = np.zeros((t.size, 3))
        for k in range(6):
            sel = i == k; out[sel] = np.stack([np.broadcast_to(c, t.shape)[sel] for c in tab[k]], -1)
        return out
    raise ValueError(scheme)


def gcolorize(x, cmap=None, vmin=None, vmax=None):
    a = np.asarray(x, float); cmap = ggencmap() if cmap is None else np.asarray(cmap)
    lo = a.min() if vmin is None else vmin; hi = a.max() if vmax is None else vmax
    idx = np.clip(((a - lo) / ((hi - lo) or 1) * (len(cmap) - 1)).round().astype(int), 0, len(cmap) - 1)
    return cmap[idx]


def gextents(loc):
    """Bounding box of location data (last axis = coordinate): (mins, maxs)."""
    p = np.asarray(loc, float).reshape(-1, np.asarray(loc).shape[-1]); return p.min(0), p.max(0)


def ggridder2d(points, values, wsize, hsize, bounds=None, method='linear', fill=0.0):
    """Grid scattered 2-D samples onto a regular wsize x hsize grid (axis 0 = x)."""
    from scipy.interpolate import griddata
    p = np.asarray(points, float); v = np.asarray(values, float)
    x0, x1, y0, y1 = bounds or (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())
    gx, gy = np.meshgrid(np.linspace(x0, x1, int(wsize)), np.linspace(y0, y1, int(hsize)), indexing='ij')
    return griddata(p, v, (gx, gy), method=method, fill_value=fill)
