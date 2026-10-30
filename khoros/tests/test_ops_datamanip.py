import numpy as np
import pytest
from khoros_cantata import ops_datamanip as D
from khoros_cantata.viff import ViffObject


def test_arith1_all_functions_run():
    x = np.array([0.25, 0.5])
    for name in D.UNARY: D.karith1(x + (1.5 if name == 'acosh' else 0), name)
    assert np.allclose(D.karith1([4, 9], 'sqrt'), [2, 3]) and np.allclose(D.karith1([2], 'recip'), [.5])
    with pytest.raises(ValueError): D.karith1(x, 'bogus')


def test_arith2_bitwise_compare_logexp():
    assert np.allclose(D.karith2([1, 2], [3, 4], 'add'), [4, 6]) and np.allclose(D.karith2([2], [8], 'divinto'), [4])
    assert np.allclose(D.karith2([3, -1], 2, 'max'), [3, 2]) and np.allclose(D.karith2([5], [3], 'absdiff'), [2])
    assert list(D.kbitwise([12], [10], 'and')) == [8] and list(D.kbitwise([1], [3], 'lshift')) == [8]
    assert list(D.kbitwise([0], op='not')) == [-1] and list(D.kbitwise([5.0], [1], 'xor')) == [4]
    assert list(D.kcompare([1, 2, 3], 2, 'ge')) == [0, 1, 1]
    assert list(D.kcompare([1, 2.05], 2, 'eq', 255, 0, 0.1)) == [0, 255]
    assert np.allclose(D.klogexp([8], 'log', 2), [3]) and np.allclose(D.klogexp([3], 'exp', 2), [8])
    assert np.allclose(D.klogexp([100], 'log10'), [2])


def test_bessel_blend_gate():
    assert np.isclose(D.kbessel([0.0], 'j', 0)[0], 1) and np.isclose(D.kbessel([1.0], 'i', 0)[0], 1.2660658777520082)
    assert np.allclose(D.kblend([0, 10], [10, 0], 0.25), [7.5, 2.5])
    assert list(D.kgate([1, 2, 3], [1, 0, 1], -1)) == [1, -1, 3] and list(D.kgate([1, 2], [1, 0], 0, True)) == [0, 2]


def test_complex():
    z = np.array([3 + 4j])
    assert D.kcmplx(z, 'mag')[0] == 5 and D.kcmplx(z, 'conj')[0] == 3 - 4j and D.kcmplx(z, 'magsq')[0] == 25
    assert D.kreal2cmplx([1], [2])[0] == 1 + 2j and D.kcmplx2real(z, 'imag')[0] == 4
    assert np.isclose(D.kcmplx(np.array([1j]), 'phase')[0], np.pi / 2)


def test_convert_normal_replace_map_unmask():
    assert D.kconvert([300.4, -5], 'ubyte').tolist() == [255, 0]
    assert D.kconvert([0., 0.5, 1.], 'ubyte', normalize=True).tolist() == [0, 128, 255]
    assert D.kconvert([1, 2], 'float', scale=2, offset=1).tolist() == [3, 5] and D.kconvert([0, 2], 'bit').tolist() == [False, True]
    assert D.kconvert([1], 'dcomplex').dtype == np.complex128
    assert np.allclose(D.knormal([2, 4, 6]), [0, .5, 1]) and np.isclose(D.knormal([1., 2, 3], 'zscore').std(), 1)
    assert np.isclose(D.knormal([1., 3], 'sum').sum(), 1) and D.knormal([-4., 2], 'max').min() == -1
    assert D.kreplace([1, 5, 9], 4, 6, 0).tolist() == [1, 0, 9] and D.kreplace([1, 5, 9], 4, 6, 0, True).tolist() == [0, 5, 0]
    assert D.kmapdata([0, 2, 9], [10, 20, 30]).tolist() == [10, 30, 30]
    assert D.kunmask([1, 2, 3], [1, 0, 1], -9).tolist() == [1, -9, 3]


def test_region_ops():
    a = np.arange(20).reshape(4, 5)
    assert D.kextract(a, [1, 2], [2, 2]).tolist() == [[7, 8], [12, 13]]
    ins = D.kinset(np.zeros((3, 3)), np.ones((2, 2)), [2, 2]); assert ins.sum() == 1 and ins[2, 2] == 1
    p = D.kpad([1, 2], [5], [2], 9); assert p.tolist() == [9, 9, 1, 2, 9]
    assert D.kflip([[1, 2], [3, 4]], (1,)).tolist() == [[2, 1], [4, 3]]
    assert D.korient(np.zeros((2, 3)), 'hwdte').shape == (3, 2, 1, 1, 1)
    assert D.kappend([1], [2, 3]).tolist() == [1, 2, 3]
    assert D.kaxis((3, 2), 1, 10, 2).tolist() == [[10, 12]] * 3
    loc = D.kgenloc((2, 3), [1, 0], [2, 1]); assert loc.shape == (2, 3, 2) and loc[1, 2].tolist() == [3, 2]
    assert np.allclose(D.klocxform([[1., 0]], [[0, -1], [1, 0]], [1, 1]), [[1, 2]])
    img = np.indices((5, 5))[0].astype(float)
    assert np.allclose(D.ksampline(img, (0, 0), (4, 4), n=5), [0, 1, 2, 3, 4]) and len(D.ksampline(img, (0, 0), (4, 0))) == 5
    assert D.kslicer(np.arange(8).reshape(2, 2, 2), 2, 1).tolist() == [[1, 3], [5, 7]]


def test_import_print():
    assert D.kimportasc("1 2, 3 # c\n4\n").tolist() == [1, 2, 3, 4]
    assert D.kimportasc(b"1 2 3 4", shape=(2, 2)).shape == (2, 2)
    raw = np.array([1.5, -2], '>f4').tobytes()
    assert D.kimportraw(raw, 'float32', endian='big').tolist() == [1.5, -2]
    assert D.kimportraw(b'\x00' + np.array([7], '<i2').tobytes(), 'int16', offset=1).tolist() == [7]
    assert D.kprdata([[1, 2], [3, 4]]) == "1 2\n3 4\n" and D.kprval([[1, 2.5]], (0, 1)) == '2.5'


def test_viff_segment_ops():
    o = ViffObject(); o.segment('value').data = np.arange(4.0); o.segment('mask').data = np.array([1, 0, 1, 1])
    assert 'mask' not in D.krmseg(o, 'mask').segments and 'mask' in o.segments
    v = D.kcptoval(o, 'mask'); assert v.segments['value'].data.tolist() == [1, 0, 1, 1]
    m = D.kcpfromval(o, 'map'); assert m.segments['map'].data.tolist() == [0, 1, 2, 3]
    c = D.kcpseg(ViffObject(), o, 'mask'); assert c.segments['mask'].data.tolist() == [1, 0, 1, 1]
    t = D.kasc2seg(None, "5 6", 'time'); assert t.segments['time'].data.tolist() == [5, 6]
    assert D.ksegcmp(o, o) and not D.ksegcmp(o, v) and D.ksegcmp(o, ViffObject(), 'map') is True
    a = D.ksetdattr(o, 'units', 'mV', 'value'); assert a.segments['value'].attributes['units'] == 'mV'
    assert D.kcomment(o, 'hi').attributes['comment'] == 'hi' and 'comment' not in o.attributes
    assert D.kinsertseg(o, 'location', [[0, 1]]).segments['location'].size == (1, 2)


def test_matrix():
    A = np.array([[4., 1], [2, 3]])
    assert np.allclose(D.mmul(A, D.minvert(A)), np.eye(2)) and np.allclose(D.minvert(np.ones((2, 3)), True).shape, (3, 2))
    assert np.allclose(D.mpow(A, 3), A @ A @ A) and D.mextract_diag(A).tolist() == [4, 3]
    assert D.mgdiag([1, 2]).tolist() == [[1, 0], [0, 2]] and D.mreplicate(A, (1, 2)).shape == (2, 4)
    assert D.mrcsum(A).tolist() == [5, 5] and D.mrcsum(A, 'cols').tolist() == [6, 4]
    assert D.mexchg(A, 0, 1).tolist() == [[2, 3], [4, 1]] and D.mexchg(A, 0, 1, 'cols').tolist() == [[1, 4], [3, 2]]
    X = np.random.default_rng(0).normal(size=(50, 3)); assert np.allclose(D.mcovar(X), np.cov(X.T))
    w, v = D.meigen(A); assert np.allclose(A @ v, v * w)
    S = A @ A.T; w, v = D.meigen(S); assert np.allclose(S @ v, v * w) and np.all(np.isreal(w))
    u, s, vt = D.msvd(A); assert np.allclose(u * s @ vt, A)
    P, L, U = D.mlud(A); assert np.allclose(P @ L @ U, A)
    x = D.mlse(np.c_[np.ones(5), np.arange(5)], 2 + 3 * np.arange(5.0))[0]; assert np.allclose(x, [2, 3])
    o = ViffObject(); o.segment('value').data = np.array([[1., 2, 3], [4, 5, 6]]).reshape(2, 3, 1, 1, 1)
    assert D.as_matrix(o).tolist() == [[1, 4], [2, 5], [3, 6]]


def test_colour_geometry():
    rgb = np.random.default_rng(0).random((10, 3))
    assert np.allclose(D.kcolorspace(D.kcolorspace(rgb, 'rgb2hsv'), 'hsv2rgb'), rgb)
    assert np.allclose(D.kcolorspace([1, 0, 0], 'rgb2hsv'), [0, 1, 1]) and np.allclose(D.kcolorspace([0, 0, 1], 'rgb2hsv'), [2 / 3, 1, 1])
    assert np.allclose(D.kcolorspace(D.kcolorspace(rgb, 'rgb2yiq'), 'yiq2rgb'), rgb)
    assert np.isclose(D.kcolorspace([1, 1, 1], 'rgb2grey'), 1)
    for sch in ('grey', 'hot', 'rainbow'):
        c = D.ggencmap(16, sch); assert c.shape == (16, 3) and c.min() >= 0 and c.max() <= 1
    assert np.allclose(D.ggencmap(8, 'rainbow')[0], [0, 0, 1]) and np.allclose(D.ggencmap(8, 'rainbow')[-1], [1, 0, 0])
    assert np.allclose(D.gcolorize([0, 1]), [[0, 0, 0], [1, 1, 1]])
    lo, hi = D.gextents([[0, 5], [3, -1]]); assert lo.tolist() == [0, -1] and hi.tolist() == [3, 5]
