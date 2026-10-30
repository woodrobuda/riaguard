import numpy as np
import pytest
from khoros_cantata import ops_image as I


def test_median_removes_impulse_and_per_plane():
    a = np.zeros((7, 7)); a[3, 3] = 100
    assert I.imedian(a, 3).max() == 0
    v = np.zeros((7, 7, 2)); v[3, 3, 1] = 9; assert I.imedian(v).shape == (7, 7, 2) and I.imedian(v).max() == 0


@pytest.mark.parametrize('op', ['sobel', 'prewitt', 'roberts', 'difference'])
def test_gradient_ramp_direction(op):
    u, v = np.indices((9, 9), dtype=float); ramp = 2 * u  # increases along x (axis 0)
    mag, ang, gx, gy = I.igradient(ramp, op)
    c = (slice(2, -2), slice(2, -2))
    if op == 'roberts':  # diagonal differences: equal magnitude on both
        assert np.allclose(np.abs(gx[c]), 2) and np.allclose(np.abs(gy[c]), 2)
    else:
        assert np.allclose(gy[c], 0) and np.all(gx[c] > 0) and np.allclose(mag[c], gx[c]) and np.allclose(ang[c], 0)


def test_edge_and_laplacian():
    a = np.zeros((10, 10)); a[5:, :] = 1
    e = I.iedge(a, threshold=1); assert e[4:6, 3].all() and not e[1, 3]
    assert np.allclose(I.ilaplacian(np.indices((6, 6))[0].astype(float))[1:-1, 1:-1], 0)


def test_rotate_90_and_identity():
    a = np.zeros((5, 5)); a[4, 2] = 1  # +x from centre
    r = I.irotate(a, 90, interp=0)
    assert np.unravel_index(r.argmax(), r.shape) == (2, 0)  # moves to -y (up as displayed)
    assert np.allclose(I.irotate(a, 0), a) and np.allclose(I.irotate(I.irotate(a, 90, 0), -90, 0), a)


def test_scale_translate_geowarp():
    assert I.iscale(np.ones((4, 6)), 2, 0.5).shape == (8, 3)
    a = np.zeros((5, 5)); a[1, 1] = 1
    assert I.itranslate(a, 2, 1, interp=0)[3, 2] == 1
    assert I.itranslate(a, 4, 0, interp=0, wrap=True)[0, 1] == 1
    assert np.allclose(I.igeowarp(a, [0, 1, 0], [0, 0, 1]), a)  # identity
    assert I.igeowarp(a, [-1, 1, 0], [0, 0, 1], interp=0)[2, 1] == 1  # x_src = u - 1  -> shift +1 in x


def test_gauss_func():
    g = I.igauss_func(11, 9, xvar=4)
    assert g.shape == (11, 9) and g[5, 4] == 1 and np.isclose(g[7, 4], np.exp(-4 / 8))
    assert np.isclose(I.igauss_func(5, 5, normalize=True).sum(), 1)


@pytest.mark.parametrize('f', ['mean', 'variance', 'energy', 'contrast', 'entropy'])
def test_texture_flat_vs_noise(f):
    flat = np.full((20, 20), 3.0); noise = np.random.default_rng(0).random((20, 20))
    tf = I.itexture(flat, 5, f); tn = I.itexture(noise, 5, f)
    if f in ('mean', 'energy'): assert np.allclose(tf, 3 if f == 'mean' else 9)
    else: assert np.allclose(tf, 0) and tn.mean() > 0


def test_filt_design_and_apply():
    H = I.ifilt_design(32, 32, 'lowpass', 'ideal', 0.1)
    assert H[16, 16] == 1 and H[0, 0] == 0
    Hb = I.ifilt_design(40, 40, 'lowpass', 'butterworth', 0.1, centered=False)
    assert Hb[0, 0] == 1 and np.isclose(Hb[4, 0], 0.5)
    hp = I.ifilt_design(32, 32, 'highpass', 'gaussian', 0.1, centered=False); assert hp[0, 0] == 0
    bp = I.ifilt_design(32, 32, 'bandpass', 'ideal', 0.1, 0.3, centered=False); assert bp[0, 0] == 0 and bp[6, 0] == 1
    img = np.full((32, 32), 2.0) + np.cos(np.pi * np.arange(32))[:, None]
    assert np.allclose(I.ifilter_apply(img, H), 2.0)


def test_morphology_and_threshold():
    a = np.zeros((9, 9)); a[3:6, 3:6] = 1
    assert I.imorph(a, 'erode').sum() == 1 and I.imorph(a, 'dilate').sum() == 25
    b = a.copy(); b[0, 0] = 1
    assert I.imorph(b, 'open')[0, 0] == 0 and I.imorph(b, 'open').sum() == 9
    assert I.imorph(a.astype(bool), 'dilate', 3, 'cross').sum() == 21
    assert I.imorph(a, 'gradient').max() == 1
    t = I.ithreshold([1, 5, 9], 4, 6); assert list(t) == [0, 1, 0]
    two = np.r_[np.full(100, 10.), np.full(100, 200.)]
    assert 10 < I.otsu_threshold(two) < 200 and I.ithreshold(two, method='otsu').sum() == 100
    lab, n = I.ilabel(np.array([[1, 0, 1], [0, 0, 0], [1, 1, 0]])); assert n == 3


def test_kmeans_and_gamut():
    pts = np.r_[np.zeros((10, 2)), np.ones((10, 2)) * 5]
    lab, C = I.kmeans(pts, 2); assert len(set(lab[:10])) == 1 and lab[0] != lab[-1]
    rgb = np.zeros((4, 4, 3)); rgb[:2] = [1, 0, 0]; rgb[2:] = [0, 0, 1]
    idx, cm = I.igamut(rgb, 4); assert idx.shape == (4, 4) and len(cm) == 2 and np.allclose(cm[idx], rgb)
