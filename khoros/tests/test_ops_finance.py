import numpy as np
import pytest
from khoros_cantata import ops_finance as F
from khoros_cantata.streaming import assert_no_lookahead, classify

X = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300)))


def test_logret_zscore_hold_threshold():
    assert np.allclose(F.LogRet().batch({'i': [1, np.e, np.e]})['o'], [0, 1, 0])
    z = F.ZScore(3).batch({'i': [1., 2, 3, 3]})['o']
    assert z[0] == 0 and np.isclose(z[2], (3 - 2) / np.std([1, 2, 3])) and np.isclose(z[3], (3 - 8 / 3) / np.std([2, 3, 3]))
    assert F.Hold().batch({'i': [0, 1, 0, 0, -1, 0]})['o'].tolist() == [0, 1, 1, 1, -1, -1]
    assert F.Threshold3(1, 2).batch({'i': [0, 1.5, 3]})['o'].tolist() == [0, 1, 2]
    with pytest.raises(ValueError): F.ZScore(1)
    with pytest.raises(ValueError): F.Threshold3(2, 1)


def test_kalman_trend_tracks_ramp_and_slope():
    t = np.arange(400.0); y = 5 + 0.5 * t + np.random.default_rng(1).normal(0, 0.1, 400)
    o = F.KalmanTrend(1e-4, 1e-6, 1e-2).batch({'i': y})
    assert abs(o['o2'][-1] - 0.5) < 0.01 and abs(o['o'][-1] - (5 + 0.5 * 399)) < 0.3
    K = F.kalman_trend_gain(); assert 0 < K[0] < 1 and K[1] > 0
    with pytest.raises(ValueError): F.KalmanTrend(r=0)


def test_kalman_pair_recovers_beta():
    g = np.random.default_rng(2); x = 50 + np.cumsum(g.normal(0, 1, 2000)); y = 3 + 1.7 * x + g.normal(0, 0.5, 2000)
    o = F.KalmanPair(1e-5, 0.25).batch({'i1': x, 'i2': y})
    assert abs(np.median(o['o2'][-500:]) - 1.7) < 0.05 and abs(np.std(o['o'][-500:]) - 0.5) < 0.15


@pytest.mark.parametrize('name', sorted(F.FACTORIES))
def test_finance_ops_causal_chunk_invariant_and_streamable(name):
    ins = F.INPUTS.get(name, ('i',)); data = {k: X * (1 + 0.1 * j) for j, k in enumerate(ins)}
    whole = F.FACTORIES[name]({}).batch(data)
    op = F.FACTORIES[name]({}); op.reset()
    parts = [op.step({k: v[s:s + 7] for k, v in data.items()}) for s in range(0, 300, 7)]
    for k in whole: assert np.allclose(whole[k], np.concatenate([p[k] for p in parts]))
    assert classify(name)[0] == 'streamable'
    def f(x):
        d = {k: x * (1 + 0.1 * j) for j, k in enumerate(ins)}; return F.FACTORIES[name]({}).batch(d)
    assert_no_lookahead(f, X)
