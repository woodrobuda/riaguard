"""Registry glue: exposes the native operator library to Cantata workspaces.

Port naming follows Khoros 2 glyph conventions observed in historical
workspaces: single input ``i``, dual inputs ``i1``/``i2``, output ``o``
(secondary outputs ``o2``, ``o3``...; kcmplx2real keeps its historical ``om``).
Command-line options (``-name value``) arrive in ``params`` already literal-eval'd.
ViffObject inputs are unwrapped to their value segment and re-wrapped on output.
Generator operators (no input) take sizes from ``-wsize -hsize -dsize -tsize -esize``.
"""
from __future__ import annotations
import numpy as np
from .ops_common import value, rewrap
from . import ops_signal as S, ops_image as I, ops_datamanip as D
from .phase21_numeric import khoros_fft, khoros_sine, stats as _stats, conv1d_khoros

# operator name -> (toolbox, kind) ; populated by register_all for status reporting
CATALOG: dict[str, str] = {}


def _shape(p):
    dims = [int(p.get(k, 1)) for k in ('wsize', 'hsize', 'dsize', 'tsize', 'esize')]
    while len(dims) > 1 and dims[-1] == 1: dims.pop()
    return tuple(dims)


def _unary(fn):
    def op(p, i):
        x = i['i']; return {'o': rewrap(x, fn(value(x), p))}
    return op


def _binary(fn):
    def op(p, i):
        a = i.get('i1', i.get('i')); b = i.get('i2')
        return {'o': rewrap(a, fn(value(a), None if b is None else value(b), p))}
    return op


def _source(fn):
    return lambda p, i: {'o': fn(p)}


def _const_or(b, p, key='real'):
    """Khoros dual-input operators take a constant (-real) when i2 is not connected."""
    return b if b is not None else p.get(key, 0)


def register_all(reg):
    def r(tb, name, fn, inputs=('i',), outputs=('o',)):
        reg.register(tb, name, fn, list(inputs), list(outputs)); CATALOG[name] = tb
    DM, SG, IM, MX, EV, GE = 'DATAMANIP', 'SIGNAL', 'IMAGE', 'MATRIX', 'ENVISION', 'GEOMETRY'
    # ---- arithmetic
    r(DM, 'karith1', _unary(lambda x, p: D.karith1(x, p.get('op', p.get('function', 'abs')))))
    r(DM, 'karith2', _binary(lambda a, b, p: D.karith2(a, _const_or(b, p), p.get('op', 'add'))), ('i1', 'i2'))
    for op in ('add', 'sub', 'mul', 'div', 'divinto', 'pow', 'min', 'max', 'absdiff'):
        r(DM, 'k' + op, _binary(lambda a, b, p, op=op: D.karith2(a, _const_or(b, p), op)), ('i1', 'i2'))
    for fn in ('exp', 'log', 'log10', 'sin', 'cos', 'tan', 'atan', 'floor', 'ceil', 'sign'):
        if reg.resolve(DM, 'k' + fn) is None: r(DM, 'k' + fn, _unary(lambda x, p, fn=fn: D.karith1(x, fn)))
    r(DM, 'kbitwise', _binary(lambda a, b, p: D.kbitwise(a, _const_or(b, p, 'value') if p.get('op', 'and') != 'not' else None, p.get('op', 'and'))), ('i1', 'i2'))
    r(DM, 'kcompare', _binary(lambda a, b, p: D.kcompare(a, _const_or(b, p), p.get('op', 'eq'), p.get('true', 1), p.get('false', 0), p.get('tol', 0.0))), ('i1', 'i2'))
    r(DM, 'klogexp', _unary(lambda x, p: D.klogexp(x, p.get('op', 'log'), p.get('base'))))
    r(DM, 'kbessel', _unary(lambda x, p: D.kbessel(x, p.get('kind', 'j'), p.get('order', 0))))
    r(DM, 'kblend', _binary(lambda a, b, p: D.kblend(a, b, p.get('alpha', 0.5))), ('i1', 'i2'))
    r(DM, 'kgate', _binary(lambda a, b, p: D.kgate(a, b, p.get('value', 0.0), bool(p.get('invert', 0)))), ('i1', 'i2'))
    r(DM, 'kinverse', _unary(lambda x, p: D.karith1(x, 'recip')))
    # ---- complex
    r(DM, 'kcmplx', _unary(lambda x, p: D.kcmplx(x, p.get('op', 'conj'))))
    r(DM, 'kreal2cmplx', _binary(lambda a, b, p: D.kreal2cmplx(a, b)), ('i1', 'i2'))
    def c2r(p, i):
        part = 'mag' if p.get('mag') or p.get('logmag1') else 'phase' if p.get('phase') else 'imag' if p.get('imag') else 'logmag' if p.get('logmag') else 'real'
        x = i['i']; return {'om': rewrap(x, D.kcmplx2real(value(x), part))}
    r(DM, 'kcmplx2real', c2r, outputs=('om',))
    r(DM, 'kmag', c2r, outputs=('om',))
    # ---- data manipulation
    r(DM, 'kconvert', _unary(lambda x, p: D.kconvert(x, p.get('type', 'float'), p.get('scale'), p.get('offset', 0.0), bool(p.get('norm', 0)))))
    r(DM, 'knormal', _unary(lambda x, p: D.knormal(x, p.get('method', 'minmax'), p.get('min', 0.0), p.get('max', 1.0))))
    r(DM, 'kreplace', _unary(lambda x, p: D.kreplace(x, p.get('lval', 0), p.get('uval'), p.get('new', 0.0), bool(p.get('outside', 0)))))
    r(DM, 'kmapdata', _binary(lambda a, b, p: D.kmapdata(a, b)), ('i1', 'i2'))
    r(DM, 'kunmask', _binary(lambda a, b, p: D.kunmask(a, b, p.get('value', 0.0))), ('i1', 'i2'))
    def _off(p): return [int(p.get(k, 0)) for k in ('woff', 'hoff', 'doff', 'toff', 'eoff')]
    def _sz(p, x): return [int(p.get(k, n)) for k, n in zip(('wsize', 'hsize', 'dsize', 'tsize', 'esize'), list(x.shape) + [1] * 5)][:max(x.ndim, 1)]
    r(DM, 'kextract', _unary(lambda x, p: D.kextract(x, _off(p)[:x.ndim], _sz(p, x))))
    r(DM, 'kinset', _binary(lambda a, b, p: D.kinset(a, b, _off(p)[:a.ndim])), ('i1', 'i2'))
    r(DM, 'kpad', _unary(lambda x, p: D.kpad(x, _sz(p, x), _off(p)[:x.ndim], p.get('real', 0.0))))
    r(DM, 'kflip', _unary(lambda x, p: D.kflip(x, [a for a, k in enumerate('whdte') if p.get(k) and a < x.ndim] or [0])))
    r(DM, 'korient', _unary(lambda x, p: D.korient(x, p.get('order', 'hwdte'))))
    r(DM, 'kappend', _binary(lambda a, b, p: D.kappend(a, b, 'whdte'.index(p.get('axis', 'w')))), ('i1', 'i2'))
    r(DM, 'ksampline', _unary(lambda x, p: D.ksampline(x, (p.get('x1', 0), p.get('y1', 0)), (p.get('x2', 0), p.get('y2', 0)), p.get('n'))))
    r(DM, 'kprdata', _unary(lambda x, p: D.kprdata(x, p.get('format', '%.6g'))))
    r(DM, 'kprval', _unary(lambda x, p: D.kprval(x, [int(p.get(k, 0)) for k in 'whdte'][:x.ndim], p.get('format', '%.6g'))))
    r(DM, 'kimportasc', lambda p, i: {'o': D.kimportasc(i['i'])})
    r(DM, 'kimportraw', lambda p, i: {'o': D.kimportraw(i['i'], p.get('type', 'float32'), None, p.get('offset', 0), p.get('endian', 'little'))})
    r(DM, 'kkmeans', lambda p, i: (lambda lab, C: {'o': lab, 'o2': C})(*I.kmeans(value(i['i']).reshape(len(value(i['i'])), -1), p.get('k', 2), p.get('iter', 50), p.get('seed', 0))), outputs=('o', 'o2'))
    r(DM, 'kstats', lambda p, i: {'o': _stats(value(i['i']))})
    r(DM, 'kaxis', _source(lambda p: D.kaxis(_shape(p), 'whdte'.index(p.get('axis', 'w')), p.get('start', 0.0), p.get('step', 1.0))), ())
    r(DM, 'kgenloc', _source(lambda p: D.kgenloc(_shape(p))), ())
    r(DM, 'klocxform', lambda p, i: {'o': D.klocxform(value(i['i']), p['matrix'], p.get('translate'))})
    # segment operators (ViffObject in, ViffObject out)
    r(DM, 'kcpseg', lambda p, i: {'o': D.kcpseg(i['i1'], i['i2'], p.get('segment', 'mask'))}, ('i1', 'i2'))
    r(DM, 'krmseg', lambda p, i: {'o': D.krmseg(i['i'], p.get('segment', 'mask'))})
    r(DM, 'kinsertseg', lambda p, i: {'o': D.kinsertseg(i['i1'], p.get('segment', 'mask'), value(i['i2']))}, ('i1', 'i2'))
    r(DM, 'kcptoval', lambda p, i: {'o': D.kcptoval(i['i'], p.get('segment', 'mask'))})
    r(DM, 'kcpfromval', lambda p, i: {'o': D.kcpfromval(i['i'], p.get('segment', 'mask'))})
    for seg in ('val', 'mask', 'map', 'loc', 'time'):
        full = {'val': 'value', 'loc': 'location'}.get(seg, seg)
        r(DM, 'kasc2' + seg, lambda p, i, full=full: {'o': D.kasc2seg(i.get('i2'), i['i'], full)}, ('i', 'i2'))
    r(DM, 'ksegcmp', lambda p, i: {'o': D.ksegcmp(i['i1'], i['i2'], p.get('segment', 'value'), p.get('tol', 0.0))}, ('i1', 'i2'))
    r(DM, 'ksetdattr', lambda p, i: {'o': D.ksetdattr(i['i'], p['attribute'], p.get('value'), p.get('segment'))})
    r(DM, 'kcomment', lambda p, i: {'o': D.kcomment(i['i'], p.get('comment', ''))})
    # ---- transforms & signal (DATAMANIP names from the matrix)
    r(DM, 'kfft', _unary(lambda x, p: khoros_fft(x, [a for a, k in enumerate('whdte') if p.get(k) and a < x.ndim] or [0], p.get('dir', 0), p.get('scale', 1), p.get('center', 0))))
    r(DM, 'kgsin', _source(lambda p: khoros_sine(_shape(p), [p.get('amp', 1.0)], [p.get(k, 0) for k in ('wnum', 'hnum', 'dnum', 'tnum', 'enum')], [p.get(k, 0.0) for k in ('wphase', 'hphase', 'dphase', 'tphase', 'ephase')])), ())
    r(DM, 'klinearop', _binary(lambda a, b, p: conv1d_khoros(a, b if b is not None else np.ones(int(p.get('n', 3))) / int(p.get('n', 3)), p.get('w0', 0), bool(p.get('ext', 0)))), ('i1', 'i2'))
    r(DM, 'kgconst', _source(lambda p: S.kgconst(_shape(p), p.get('real', 0.0) + 1j * p['imag'] if p.get('imag') else p.get('real', 0.0))), ())
    r(DM, 'kimpulse', _source(lambda p: S.kimpulse(_shape(p), [int(p.get(k, 0)) for k in ('woff', 'hoff', 'doff', 'toff', 'eoff')][:len(_shape(p))], p.get('amp', 1.0), p.get('spacing'))), ())
    r(DM, 'kgnoise', _source(lambda p: S.kgnoise(_shape(p), p.get('type', 'gaussian'), p.get('mean', 0.0), p.get('var', 1.0), p.get('min', 0.0), p.get('max', 1.0), p.get('seed'))), ())
    r(DM, 'knoise', _unary(lambda x, p: S.knoise(x, p.get('type', 'gaussian'), p.get('mode', 'additive'), p.get('seed'), mean=p.get('mean', 0.0), variance=p.get('var', 1.0))))
    r(DM, 'kshot', _unary(lambda x, p: S.kshot(x, p.get('prob', 0.05), p.get('min'), p.get('max'), p.get('seed'))))
    r(DM, 'khisto', lambda p, i: (lambda h, e: {'o': h, 'o2': e})(*S.khisto(value(i['i']), p.get('bins', 256), p.get('min'), p.get('max'), bool(p.get('norm', 0)))), outputs=('o', 'o2'))
    r(DM, 'khistops', _binary(lambda a, b, p: S.khistops(a, p.get('op', 'equalize'), p.get('bins', 256), p.get('min'), p.get('max'), b)), ('i1', 'i2'))
    r(DM, 'kresample', _unary(lambda x, p: S.kresample(x, sizes=[p.get(k) for k in ('wsize', 'hsize', 'dsize', 'tsize', 'esize')][:x.ndim], interp=p.get('interp', 'linear'))))
    r(DM, 'kwiener', _binary(lambda a, b, p: S.kwiener(a, b, p.get('nsr', 0.01), adaptive_size=p.get('size'), noise=p.get('noise'))), ('i1', 'i2'))
    # ---- extended SIGNAL toolbox
    r(SG, 'kwindow', _unary(lambda x, p: S.kapply_window(x, p.get('type', 'hamming'))))
    r(SG, 'kgwindow', _source(lambda p: S.kwindow(int(p.get('wsize', 64)), p.get('type', 'hamming'), p.get('beta', 8.6))), ())
    r(SG, 'kfir', _binary(lambda a, b, p: S.kfir(a, b if b is not None else S.kfir_design(p.get('taps', 31), p.get('cutoff', 0.25), p.get('ftype', 'lowpass'), p.get('window', 'hamming')), mode=p.get('mode', 'causal'))), ('i1', 'i2'))
    r(SG, 'kiir', _unary(lambda x, p: S.kiir(x, S.kiir_design(p.get('order', 4), p.get('cutoff', 0.25), p.get('ftype', 'lowpass'), p.get('family', 'butter'), p.get('rp', 1.0), p.get('rs', 40.0), fs=1.0), zerophase=bool(p.get('zerophase', 0)))))
    r(SG, 'kconvolve', _binary(lambda a, b, p: S.kconvolve(a, b, p.get('mode', 'full'))), ('i1', 'i2'))
    r(SG, 'kcorr', _binary(lambda a, b, p: S.kcorrelate(a, b, p.get('mode', 'full'), p.get('norm'))), ('i1', 'i2'))
    r(SG, 'kpsd', lambda p, i: (lambda f, P: {'o': P, 'o2': f})(*S.kpsd(value(i['i']), p.get('fs', 1.0), p.get('method', 'periodogram'))), outputs=('o', 'o2'))
    r(SG, 'kspectrum', _unary(lambda x, p: S.kspectrum(x, 0, p.get('output', 'magnitude'), bool(p.get('db', 0)))))
    r(SG, 'kema', _unary(lambda x, p: S.kema(x, p.get('span'), p.get('alpha'))))
    r(SG, 'krunstats', _unary(lambda x, p: S.krunstats(x, p.get('stat', 'mean'), p.get('window'))))
    r(SG, 'kdelay', _unary(lambda x, p: S.kdelay(x, p.get('n', 1), p.get('fill', 0.0))))
    r(SG, 'kdiff', _unary(lambda x, p: S.kdiff(x, p.get('fill', 0.0))))
    r(SG, 'kcross', _binary(lambda a, b, p: S.kcross(a, _const_or(b, p))), ('i1', 'i2'))
    r(SG, 'khilbert', _unary(lambda x, p: S.khilbert(x, p.get('output', 'envelope'))))
    r(SG, 'kdwt', lambda p, i: {'o': S.kdwt(value(i['i']), p.get('wavelet', 'haar'), p.get('levels', 1))})
    r(SG, 'kidwt', lambda p, i: {'o': S.kidwt(i['i'], p.get('wavelet', 'haar'))})
    # ---- FINANCE (causal online operators; batch == reset + step over the record)
    from . import ops_finance as F
    for name, fac in F.FACTORIES.items():
        ins = F.INPUTS.get(name, ('i',)); outs = fac({}).outputs
        def fop(p, i, fac=fac, ins=ins):
            return fac(p).batch({k: value(i[k]) for k in ins})
        r('FINANCE', name, fop, ins, outs)
    # ---- IMAGE
    r(IM, 'imedian', _unary(lambda x, p: I.imedian(x, p.get('wsize', 3), p.get('hsize'))))
    def grad(p, i):
        x = i['i']; m, d, gx, gy = I.igradient(value(x), p.get('op', 'sobel'))
        return {'o': rewrap(x, m), 'o2': rewrap(x, d), 'o3': rewrap(x, gx), 'o4': rewrap(x, gy)}
    r(IM, 'igradient', grad, outputs=('o', 'o2', 'o3', 'o4'))
    r(IM, 'iedge', _unary(lambda x, p: I.iedge(x, p.get('op', 'sobel'), p.get('thresh'))))
    r(IM, 'ilaplacian', _unary(lambda x, p: I.ilaplacian(x)))
    r(IM, 'irotate', _unary(lambda x, p: I.irotate(x, p.get('angle', 0.0), p.get('interp', 1), p.get('real', 0.0), bool(p.get('resize', 0)))))
    r(IM, 'iscale', _unary(lambda x, p: I.iscale(x, p.get('wfactor', 1.0), p.get('hfactor'), p.get('interp', 1))))
    r(IM, 'itranslate', _unary(lambda x, p: I.itranslate(x, p.get('woff', 0), p.get('hoff', 0), p.get('interp', 1), p.get('real', 0.0), bool(p.get('wrap', 0)))))
    r(IM, 'igeowarp', _unary(lambda x, p: I.igeowarp(x, p['xcoef'], p['ycoef'], p.get('interp', 1), p.get('real', 0.0))))
    r(IM, 'igauss_func', _source(lambda p: I.igauss_func(p.get('wsize', 64), p.get('hsize', 64), p.get('xcenter'), p.get('ycenter'), p.get('xvar', 1.0), p.get('yvar'), p.get('amp', 1.0))), ())
    r(IM, 'itexture', _unary(lambda x, p: I.itexture(x, p.get('wsize', 5), p.get('feature', 'variance'))))
    r(IM, 'ifilt_design', _source(lambda p: I.ifilt_design(p.get('wsize', 64), p.get('hsize', 64), p.get('ftype', 'lowpass'), p.get('shape', 'butterworth'), p.get('cutoff', 0.25), p.get('cutoff2'), p.get('order', 2), bool(p.get('center', 1)))), ())
    r(IM, 'imorph', _unary(lambda x, p: I.imorph(x, p.get('op', 'dilate'), p.get('size', 3), p.get('shape', 'square'))))
    for mop in ('erode', 'dilate', 'open', 'close'):
        r(IM, 'i' + mop, _unary(lambda x, p, mop=mop: I.imorph(x, mop, p.get('size', 3), p.get('shape', 'square'))))
    r(IM, 'ithresh', _unary(lambda x, p: I.ithreshold(x, p.get('lval'), p.get('uval'), p.get('method'), p.get('inside', 1), p.get('outside', 0))))
    r(IM, 'ilabel', _unary(lambda x, p: I.ilabel(x)[0]))
    r(IM, 'igamut', lambda p, i: (lambda idx, cm: {'o': idx, 'o2': cm})(*I.igamut(value(i['i']), p.get('ncolors', 16))), outputs=('o', 'o2'))
    # ---- MATRIX
    m1 = lambda f: (lambda p, i: {'o': f(i['i'], p)})
    r(MX, 'mmul', lambda p, i: {'o': D.mmul(i['i1'], i['i2'])}, ('i1', 'i2'))
    r(MX, 'minvert', m1(lambda x, p: D.minvert(x, bool(p.get('pseudo', 0)))))
    r(MX, 'mpow', m1(lambda x, p: D.mpow(x, p.get('power', 2))))
    r(MX, 'mextract_diag', m1(lambda x, p: D.mextract_diag(x, p.get('k', 0))))
    r(MX, 'mgdiag', m1(lambda x, p: D.mgdiag(value(x), p.get('k', 0))))
    r(MX, 'mreplicate', m1(lambda x, p: D.mreplicate(x, (p.get('hrep', 1), p.get('wrep', 1)))))
    r(MX, 'mrcsum', lambda p, i: {'o': D.mrcsum(i['i'], 'rows'), 'o2': D.mrcsum(i['i'], 'cols')}, outputs=('o', 'o2'))
    r(MX, 'mcovar', m1(lambda x, p: D.mcovar(x)))
    r(MX, 'mexchg', m1(lambda x, p: D.mexchg(x, p.get('i', 0), p.get('j', 1), p.get('what', 'rows'))))
    r(MX, 'meigen', lambda p, i: (lambda w, v: {'o': w, 'o2': v})(*D.meigen(i['i'])), outputs=('o', 'o2'))
    r(MX, 'msvd', lambda p, i: (lambda u, s, vt: {'o': u, 'o2': s, 'o3': vt})(*D.msvd(i['i'])), outputs=('o', 'o2', 'o3'))
    r(MX, 'mlud', lambda p, i: (lambda P, L, U: {'o': P, 'o2': L, 'o3': U})(*D.mlud(i['i'])), outputs=('o', 'o2', 'o3'))
    r(MX, 'mlse', lambda p, i: {'o': D.mlse(i['i1'], value(i['i2']))[0]}, ('i1', 'i2'))
    # ---- ENVISION / GEOMETRY helpers
    r(EV, 'kcolorspace', _unary(lambda x, p: D.kcolorspace(x, p.get('conv', 'rgb2hsv'))))
    r(GE, 'ggencmap', _source(lambda p: D.ggencmap(p.get('n', 256), p.get('scheme', 'grey'))), ())
    r(GE, 'gcolorize', _binary(lambda a, b, p: D.gcolorize(a, b, p.get('min'), p.get('max'))), ('i1', 'i2'))
    r(GE, 'gextents', lambda p, i: (lambda lo, hi: {'o': lo, 'o2': hi})(*D.gextents(value(i['i']))), outputs=('o', 'o2'))
    r(GE, 'ggridder2d', _binary(lambda a, b, p: D.ggridder2d(a, b, p.get('wsize', 32), p.get('hsize', 32), None, p.get('method', 'linear'))), ('i1', 'i2'))
    r(GE, 'kslicer', _unary(lambda x, p: D.kslicer(x, 'whdte'.index(p.get('axis', 'd')), p.get('index', 0))))
    # toolbox-agnostic lookup: let '*' resolve any operator by name too
    for (tb, name), opobj in list(reg._ops.items()):
        if tb != '*' and ('*', name) not in reg._ops: reg._ops[('*', name)] = opobj
