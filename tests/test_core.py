import numpy as np, pandas as pd, pytest
from scipy import signal
from riaguard import streaming as st, spectral as sp, bars, data, backtest as bt, strategies as S


def test_ema_matches_pandas():
    x = np.random.default_rng(0).normal(size=300).cumsum()
    ours = st.run(st.EMA(span=10), x)
    ref = pd.Series(x).ewm(span=10, adjust=False).mean().to_numpy()
    assert np.allclose(ours, ref)


def test_fir_matches_lfilter():
    x = np.random.default_rng(1).normal(size=200); taps = signal.firwin(15, 0.1)
    ours = st.run(st.FIR(taps), x); ref = signal.lfilter(taps, 1, x)
    assert np.allclose(ours[14:], ref[14:]) and np.isnan(ours[:14]).all()


def test_iir_matches_lfilter_with_steady_state_init():
    x = np.random.default_rng(2).normal(size=300).cumsum() + 50; b, a = signal.butter(2, 0.1)
    ref, _ = signal.lfilter(b, a, x, zi=signal.lfilter_zi(b, a) * x[0])
    assert np.allclose(st.run(st.IIR(b, a), x), ref, atol=1e-8)


def test_kalman_trend_recovers_slope():
    t = np.arange(500); z = 0.05 * t + np.random.default_rng(3).normal(0, 0.5, 500)
    s = st.run(st.KalmanLocalLinearTrend(1e-3, 1e-6, 0.25), z)
    assert abs(s[-1, 1] - 0.05) < 0.01


def test_kalman_hedge_ratio_tracks_beta():
    rng = np.random.default_rng(4); x = 50 + rng.normal(size=1000).cumsum()
    y = 1.5 * x + 3 + rng.normal(0, 0.5, 1000); k = st.KalmanHedgeRatio(delta=1e-5, r=0.25)
    out = [k.update(a, b) for a, b in zip(x, y)]
    assert abs(out[-1][0] - 1.5) < 0.05


def test_streaming_is_causal():
    x = np.random.default_rng(5).normal(size=200)
    for f in (st.EMA(span=5), st.FIR.sma(7), st.KalmanLocalLinearTrend()):
        full = st.run(f, x); part = st.run(f, x[:120])
        assert np.allclose(full[:120], part, equal_nan=True)


def test_dominant_period():
    t = np.arange(1024); x = np.sin(2 * np.pi * t / 32) + 0.3 * np.random.default_rng(6).normal(size=1024)
    assert abs(sp.dominant_period(x) - 32) < 3


def test_wavelet_denoise_reduces_error_and_mra_sums():
    t = np.arange(1024); clean = np.sin(2 * np.pi * t / 128); noisy = clean + 0.4 * np.random.default_rng(7).normal(size=1024)
    den = sp.wavelet_denoise(noisy)
    assert np.mean((den - clean) ** 2) < 0.5 * np.mean((noisy - clean) ** 2)
    assert np.allclose(sum(c for _, c in sp.wavelet_mra(noisy, level=4)), noisy, atol=1e-8)


def test_hilbert_period():
    t = np.arange(2000); _, _, per = sp.hilbert_phase(np.sin(2 * np.pi * t / 40))
    assert abs(np.nanmedian(per[200:-200]) - 40) < 1


def test_filtfilt_is_noncausal_and_rolling_fixes_it():
    x = np.random.default_rng(8).normal(size=300).cumsum(); b, a = sp.butter_lowpass(20)
    full = sp.zero_phase(x, b, a); part = sp.zero_phase(x[:200], b, a)
    assert not np.allclose(full[:200], part)  # look-ahead!
    fn = lambda s: sp.zero_phase(s, b, a)
    r1 = sp.rolling_causal(x, fn, 60); r2 = sp.rolling_causal(x[:200], fn, 60)
    assert np.allclose(r1[:200], r2, equal_nan=True)


def test_bars():
    tk = data.synthetic_ticks(5000)
    vb = bars.volume_bars(tk, 20000); db = bars.dollar_bars(tk, 2e6); tb = bars.tick_bars(tk, 100)
    assert (vb["volume"].iloc[:-1] >= 20000).all() and (db["dollar"].iloc[:-1] >= 2e6).all()
    assert tb["ticks"].iloc[:-1].eq(100).all() and vb["volume"].sum() == tk["size"].sum()
    tm = bars.time_bars(tk, "1min"); assert tm["ticks"].sum() == len(tk)


def test_backtest_no_lookahead_and_costs():
    px = data.synthetic_prices(600)["close"]
    for f in (S.ema_cross, S.sma_trend, S.kalman_slope):
        assert bt.assert_causal(lambda c: f(c), px)
    bh = bt.run_signal(px, S.buy_and_hold(px), "bh", cost_bps=0)
    assert np.isclose(bh.equity.iloc[-1], px.iloc[-1] / px.iloc[0])
    # a perfect-foresight signal is caught by assert_causal
    cheat = lambda c: (c.shift(-1) > c).astype(float)
    with pytest.raises(AssertionError):
        bt.assert_causal(cheat, px)


def test_walk_forward_runs():
    px = data.synthetic_prices(1200)["close"]
    res, chosen = bt.walk_forward(px, S.ema_cross, [(10, 50), (20, 100)], train=400, test=100)
    assert len(chosen) >= 6 and np.isfinite(res.equity.iloc[-1])


def test_csv_roundtrip(tmp_path):
    df = data.synthetic_prices(50); p = tmp_path / "x.csv"; df.rename_axis("Date").to_csv(p)
    back = data.load_csv(p); assert np.allclose(back["close"], df["close"])
