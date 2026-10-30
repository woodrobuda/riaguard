import numpy as np
import pytest
from khoros_cantata import ops_signal as S


@pytest.mark.parametrize('w', ['rect', 'bartlett', 'hanning', 'hamming', 'blackman', 'kaiser', 'gaussian', 'tukey'])
def test_windows_symmetric_peak(w):
    v = S.kwindow(33, w)
    assert v.shape == (33,) and np.allclose(v, v[::-1]) and np.isclose(v.max(), v[16])


def test_hamming_values_and_apply():
    assert np.allclose(S.kwindow(5, 'hamming'), [0.08, 0.54, 1.0, 0.54, 0.08])
    x = np.ones((5, 2)); assert np.allclose(S.kapply_window(x, 'hamming')[:, 1], S.kwindow(5, 'hamming'))
    with pytest.raises(ValueError): S.kwindow(4, 'nope')


def test_fir_lowpass_attenuates_high_tone():
    n = np.arange(2000); lo = np.sin(2 * np.pi * 0.02 * n); hi = np.sin(2 * np.pi * 0.4 * n)
    taps = S.kfir_design(101, 0.1)
    assert np.isclose(taps.sum(), 1.0)
    y = S.kfir(lo + hi, taps, mode='same')[200:-200]
    assert np.max(np.abs(y - lo[200:-200])) < 0.02
    with pytest.raises(ValueError): S.kfir_design(100, 0.1, 'highpass')


def test_iir_butter_response_and_zerophase():
    sos = S.kiir_design(4, 0.1, 'lowpass')
    f, h = S.kfreqz(sos, n=4096)
    assert abs(abs(h[0]) - 1) < 1e-9 and abs(abs(h[np.argmin(abs(f - 0.1))]) - 2 ** -0.5) < 1e-2
    x = np.sin(2 * np.pi * 0.01 * np.arange(1000))
    assert np.max(np.abs(S.kiir(x, sos, zerophase=True)[100:-100] - x[100:-100])) < 0.01
    for fam in ('cheby1', 'cheby2', 'ellip', 'bessel'): assert S.kiir_design(3, 0.2, family=fam).shape[1] == 6


def test_convolve_and_correlate():
    assert np.allclose(S.kconvolve([1, 2, 3], [0, 1, 0.5]), np.convolve([1, 2, 3], [0, 1, 0.5]))
    x = np.random.default_rng(1).normal(size=200); y = np.roll(x, 7)
    r = S.kcorrelate(y, x); lags = np.arange(-199, 200)
    assert lags[np.argmax(r)] == 7
    ac = S.kcorrelate(x, normalize='coeff'); assert np.isclose(ac.max(), 1) and np.argmax(ac) == 199


def test_psd_peak_and_spectrum():
    n = np.arange(1024); x = np.sin(2 * np.pi * 0.125 * n)
    f, P = S.kpsd(x); assert np.isclose(f[np.argmax(P)], 0.125)
    f, P = S.kpsd(x, method='welch'); assert abs(f[np.argmax(P)] - 0.125) < 0.01
    assert np.argmax(S.kspectrum(x)[:512]) == 128


def test_resample():
    x = np.arange(4.0)
    assert np.allclose(S.kresample(x, [8], interp='linear'), [0, .5, 1, 1.5, 2, 2.5, 3, 3])
    assert np.allclose(S.kresample(x, [8], interp='zero'), [0, 0, 1, 1, 2, 2, 3, 3])
    assert np.allclose(S.kresample(np.arange(8.0), [4], interp='zero'), [0, 2, 4, 6])
    s = np.sin(2 * np.pi * np.arange(32) / 32); up = S.kresample(s, [64], interp='fft')
    assert np.allclose(up, np.sin(2 * np.pi * np.arange(64) / 64), atol=1e-12)
    assert S.kresample(np.zeros((4, 6)), factors=[2, 0.5]).shape == (8, 3)


def test_sources_and_noise():
    assert np.all(S.kgconst((3, 2), 4.5) == 4.5)
    imp = S.kimpulse((10,), [3]); assert imp.sum() == 1 and imp[3] == 1
    assert np.array_equal(np.nonzero(S.kimpulse((10,), [1], spacing=4))[0], [1, 5, 9])
    g = S.kgnoise((200000,), 'gaussian', mean=2, variance=4, seed=0)
    assert abs(g.mean() - 2) < 0.02 and abs(g.var() - 4) < 0.05
    for t in ('uniform', 'rayleigh', 'exponential', 'poisson'):
        assert S.kgnoise((1000,), t, mean=3, seed=1).shape == (1000,)
    assert abs(S.kgnoise((200000,), 'rayleigh', variance=2, seed=0).var() - 2) < 0.05
    assert np.array_equal(S.kgnoise((5,), seed=3), S.kgnoise((5,), seed=3))
    x = np.zeros(10000); assert abs(S.knoise(x, variance=1, seed=0).std() - 1) < 0.03
    assert np.allclose(S.knoise(np.ones(5), 'uniform', 'multiplicative', seed=0, low=2, high=2), 2)
    sh = S.kshot(np.full(10000, 5.0), 0.1, 0, 255, seed=0)
    assert abs(np.mean(sh != 5) - 0.1) < 0.02 and set(np.unique(sh)) == {0, 5, 255}


def test_histogram_and_ops():
    h, e = S.khisto([0, 1, 1, 2, 2, 2], bins=3)
    assert list(h) == [1, 2, 3] and np.allclose(e, [0, 2 / 3, 4 / 3, 2])
    assert np.isclose(S.khisto([1, 2], bins=2, normalize=True)[0].sum(), 1)
    assert np.allclose(S.khistops([2, 4, 6], 'stretch', vmin=0, vmax=1), [0, .5, 1])
    eq = S.khistops(np.random.default_rng(0).exponential(size=10000), 'equalize', vmin=0, vmax=1)
    assert abs(np.mean(eq < 0.5) - 0.5) < 0.03
    m = S.khistops(np.array([3., 1, 2]), 'match', target=[10, 20, 30]); assert np.allclose(m, [30, 10, 20])


def test_wiener_restores_blur():
    rng = np.random.default_rng(0); x = rng.random(256); psf = np.array([0.5, 0.3, 0.2])
    blurred = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(psf, 256)))
    rest = S.kwiener(blurred, psf, nsr=1e-6)
    assert np.max(np.abs(rest - x)) < 1e-3 < np.max(np.abs(blurred - x))
    noisy = np.ones(500) + rng.normal(0, .3, 500)
    assert S.kwiener(noisy, adaptive_size=9).std() < noisy.std()


@pytest.mark.parametrize('w', ['haar', 'db2'])
def test_dwt_roundtrip(w):
    x = np.random.default_rng(2).normal(size=64)
    c = S.kdwt(x, w, 3)
    assert [len(v) for v in c] == [8, 8, 16, 32]
    assert np.isclose(sum((v ** 2).sum() for v in c), (x ** 2).sum())
    assert np.allclose(S.kidwt(c, w), x)
    assert np.allclose(S.kdwt([1, 1, 1, 1], 'haar')[1], 0)


def test_causal_primitives():
    pd = pytest.importorskip('pandas')
    x = np.random.default_rng(0).normal(size=300).cumsum()
    assert np.allclose(S.kema(x, span=10), pd.Series(x).ewm(span=10, adjust=False).mean())
    assert np.allclose(S.krunstats(x, 'mean'), pd.Series(x).expanding().mean())
    assert np.allclose(S.krunstats(x, 'std'), pd.Series(x).expanding().std(ddof=0))
    assert np.allclose(S.krunstats(x, 'mean', 20)[19:], pd.Series(x).rolling(20).mean()[19:])
    assert np.allclose(S.krunstats(x, 'min'), np.minimum.accumulate(x)) and np.allclose(S.krunstats(x, 'sum', 3)[2], x[:3].sum())
    assert np.allclose(S.kdelay([1., 2, 3], 2, -1), [-1, -1, 1]) and np.allclose(S.kdiff([1., 3, 6]), [0, 2, 3])
    assert np.allclose(S.kcross([1., 3, 3, 0], [2., 2, 2, 2]), [0, 1, 0, -1])
