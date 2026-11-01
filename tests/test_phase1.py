"""Phase-1 tests: causality, cost math, metrics against known values, splits, data, alerts."""
import json, os
import numpy as np, pandas as pd, pytest
from scipy import stats
from riaguard import signals as S, engine as E, metrics as M, splits as SP, datasources as D, quality as Q, explain as X
from riaguard.costs import CostModel, ZERO


@pytest.fixture(scope="module")
def spy():
    return Q.total_return_close(D.SampleSource().get_bars("SPY")).iloc[-1500:]


# ---------- no look-ahead ----------
ALL_SIGNALS = [s for g in S.DSP_GRIDS.values() for s in g[:2]] + S.BENCHMARKS


@pytest.mark.parametrize("sig", ALL_SIGNALS, ids=lambda s: s.label)
def test_signals_are_causal_truncate_and_perturb(sig, spy):
    E.assert_causal(lambda c: sig.compute(c)[["direction", "score"]].to_numpy(), spy, n_check=4)


def test_assert_causal_catches_lookahead(spy):
    with pytest.raises(AssertionError):
        E.assert_causal(lambda c: np.sign(c.shift(-1) - c).fillna(0), spy)


def test_perturbing_future_does_not_change_past_positions_or_returns(spy):
    sig = S.make("ema_cross", 20, 100); t = 1000
    a = E.run(spy, sig, "long_short")
    c2 = spy.copy(); c2.iloc[t + 1:] *= np.linspace(1, 3, len(c2) - t - 1)
    b = E.run(c2, sig, "long_short")
    # position over bar t+1 is decided at t, so positions match through t+1 and returns through t
    assert np.allclose(a.position.iloc[: t + 2], b.position.iloc[: t + 2])
    assert np.allclose(a.returns.iloc[: t + 1], b.returns.iloc[: t + 1])


def test_walk_forward_oos_unaffected_by_future(spy):
    cands = S.DSP_GRIDS["ema_cross"][:3]
    a, _ = E.walk_forward(spy, cands, train=500, test=100)
    cut = a.returns.index[300]
    c2 = spy.copy(); c2[c2.index > cut] *= 1.5
    b, _ = E.walk_forward(spy.where(spy.index <= cut, c2), cands, train=500, test=100)
    assert np.allclose(a.returns.loc[:cut].iloc[:-1], b.returns.loc[:cut].iloc[:-1])


def test_delay_zero_rejected(spy):
    with pytest.raises(ValueError):
        E.simulate(spy, pd.Series(1.0, index=spy.index), delay=0)


# ---------- engine & costs ----------
def test_simulate_timing_and_cost_math():
    idx = pd.bdate_range("2024-01-01", periods=5, tz="UTC")
    close = pd.Series([100, 110, 99, 99, 108.9], index=idx)
    tgt = pd.Series([1, 1, 0, -1, -1], index=idx, dtype=float)
    cm = CostModel(commission_bps=5, half_spread_bps=3, slippage_bps=2)   # 10 bps per side
    r = E.simulate(close, tgt, cm)
    assert list(r.position) == [0, 1, 1, 0, -1]
    assert np.allclose(r.gross, [0, 0.10, -0.10, 0, -0.10])
    assert np.allclose(r.costs, [0, 0.001, 0, 0.001, 0.001])
    assert np.allclose(r.returns, r.gross - r.costs)


def test_borrow_and_vol_slippage():
    idx = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    cm = CostModel(0, 0, 0, borrow_bps_per_year=252)
    assert np.allclose(cm.holding_cost(pd.Series([-1.0, 1.0, -0.5], index=idx)), [1e-4, 0, 0.5e-4])
    cm2 = CostModel(0, 0, 0, vol_slippage_k=0.1)
    c = cm2.trade_cost(pd.Series([1.0, 0, 0], index=idx), pd.Series([0.01, 0.01, 0.01], index=idx))
    assert np.isclose(c.iloc[0], 1.0 * 0.1 * 100 / 1e4)


def test_modes_and_vol_target_sizing(spy):
    d = pd.Series([-1, 0, 1], dtype=float)
    assert list(E.target_position(d, "long_flat")) == [0, 0, 1]
    assert list(E.target_position(d, "long_short")) == [-1, 0, 1]
    with pytest.raises(ValueError): E.target_position(d, "yolo")
    sc = E.Sizing("vol_target", target_vol=0.10, max_leverage=2.0).scale(spy)
    assert (sc <= 2.0).all() and (sc >= 0).all()
    r = E.simulate(spy, sc, ZERO)
    realised = r.returns.iloc[200:].std() * np.sqrt(252)
    assert 0.07 < realised < 0.13


# ---------- metrics against known values ----------
def test_drawdown_known():
    r = [1.0, -0.5, 2.0, -0.1]       # equity 2, 1, 3, 2.7
    assert np.isclose(M.max_drawdown(r), -0.5)
    assert M.max_drawdown_duration(r) == 1
    assert M.max_drawdown_duration([0.1, -0.1, 0.01, 0.01, 0.5]) == 3


def test_sharpe_sortino_known():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    assert np.isclose(M.sharpe(r, 1), r.mean() / r.std(ddof=1))
    assert np.isclose(M.sortino(r, 1), r.mean() / np.sqrt(np.mean(np.minimum(r, 0) ** 2)))
    assert np.isclose(M.cagr([0.1, 0.1], 2), 0.21)
    assert np.isclose(M.calmar([0.1, -0.5, 0.1], 3), M.cagr([0.1, -0.5, 0.1], 3) / 0.5)


def test_hit_rate_turnover_exposure():
    assert M.hit_rate([0.1, -0.1, 0.2, 0.0], [1, 1, 1, 0]) == pytest.approx(2 / 3)
    assert M.turnover([1, 1, 0, -1], periods_per_year=4) == pytest.approx(3.0)
    assert M.exposure([1, 0, -1, 0.5]) == pytest.approx(0.625)


def test_psr_normal_case_matches_closed_form():
    rng = np.random.default_rng(0); r = rng.normal(0.001, 0.01, 2000)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    exp = stats.norm.cdf(sr * np.sqrt(len(r) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    assert np.isclose(M.probabilistic_sharpe_ratio(r), exp)


def test_expected_max_sharpe_and_dsr():
    assert M.expected_max_sharpe(1, 1.0) == 0.0
    # Bailey & Lopez de Prado (2014) example: N=100 trials, V[SR]=1 -> E[max SR] ~ 2.5
    assert 2.4 < M.expected_max_sharpe(100, 1.0) < 2.6
    rng = np.random.default_rng(1); r = rng.normal(0.0005, 0.01, 1000)
    assert M.deflated_sharpe_ratio(r, 1, 0.0) == pytest.approx(M.probabilistic_sharpe_ratio(r))
    assert M.deflated_sharpe_ratio(r, 1000, 0.002) < M.deflated_sharpe_ratio(r, 2, 0.002)


def test_pbo_noise_vs_skill():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, (2000, 20))
    assert 0.2 < M.pbo_cscv(noise, 10)["pbo"] < 0.8
    skill = noise.copy(); skill[:, 0] += 0.003
    assert M.pbo_cscv(skill, 10)["pbo"] < 0.05


# ---------- splits ----------
def test_walk_forward_splits():
    f = SP.walk_forward_splits(100, 50, 20)
    assert [(tr[0], tr[-1], te[0], te[-1]) for tr, te in f] == [(0, 49, 50, 69), (20, 69, 70, 89), (40, 89, 90, 99)]
    g = SP.walk_forward_splits(100, 50, 20, gap=5, expanding=True)
    assert all(tr[0] == 0 and te[0] == tr[-1] + 6 for tr, te in g)
    for tr, te in f + g: assert tr.max() < te.min()


def test_purged_kfold():
    folds = SP.purged_kfold(100, k=5, purge=3, embargo=2)
    assert len(folds) == 5
    tr, te = folds[2]
    assert te[0] == 40 and te[-1] == 59
    assert not set(range(37, 62)) & set(tr)
    assert {36, 62} <= set(tr)
    assert sorted(np.concatenate([t for _, t in folds])) == list(range(100))


# ---------- data ----------
def test_sample_source_schema():
    df = D.SampleSource().get_bars("QQQ", start="2020-01-01", end="2020-12-31")
    assert list(df.columns[:5]) == D.BAR_COLUMNS and str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing and 250 <= len(df) <= 254


def test_local_csv_and_parquet_roundtrip(tmp_path):
    df = D.SampleSource().get_bars("SPY").iloc[:50]
    df.reset_index().to_csv(tmp_path / "AAA.csv", index=False)
    pd.testing.assert_frame_equal(D.LocalFileSource(tmp_path).get_bars("AAA"), df, check_freq=False)
    try:
        df.to_parquet(tmp_path / "BBB.parquet")
    except ImportError:
        pytest.skip("no parquet engine")
    pd.testing.assert_frame_equal(D.LocalFileSource(tmp_path).get_bars("BBB"), df, check_freq=False)


def test_quality_flags_problems():
    df = D.SampleSource().get_bars("SPY").iloc[100:400].copy()
    rep = Q.check_bars(df); assert rep["ok"] and rep["suspect_bad_ticks"] == [] and rep["missing_sessions"] == []
    bad = df.copy()
    bad.iloc[150, bad.columns.get_loc("close")] *= 1.5           # spike-and-revert bad print
    bad = pd.concat([bad, bad.iloc[[10]]]).sort_index()            # duplicate timestamp
    bad = bad.drop(bad.index[200])                                 # gap
    rep = Q.check_bars(bad)
    assert not rep["ok"] and rep["duplicate_timestamps"] == 1 and len(rep["missing_sessions"]) == 1
    assert str(df.index[150].date()) in rep["suspect_bad_ticks"]
    assert rep["ohlc_inconsistent"] >= 1
    clean = Q.clean_bars(bad, drop_bad_ticks=True)
    assert not clean.index.duplicated().any()


def test_known_market_closures_and_holidays():
    s = Q.expected_sessions("2024-01-01", "2024-12-31")
    assert len(s) == 252
    for d in ("2024-01-01", "2024-03-29", "2024-06-19", "2024-07-04", "2024-12-25"):
        assert pd.Timestamp(d) not in s
    assert pd.Timestamp("2024-10-14") in s and pd.Timestamp("2024-11-11") in s


def test_split_adjustment():
    idx = pd.bdate_range("2024-01-01", periods=4, tz="UTC")
    df = pd.DataFrame({"open": [400, 404, 101, 102], "high": [401, 405, 102, 103], "low": [399, 403, 100, 101],
                       "close": [400, 404, 101, 102], "volume": [10, 10, 40, 40], "splits": [0, 0, 4, 0]}, index=idx)
    adj = Q.adjust_for_splits(df)
    assert np.allclose(adj["close"], [100, 101, 101, 102]) and np.allclose(adj["volume"], [40, 40, 40, 40])


def test_regular_session_dst():
    ts = pd.DatetimeIndex(["2024-03-08 14:29", "2024-03-08 14:30", "2024-03-11 13:30", "2024-03-11 20:00"], tz="UTC")
    out = Q.regular_session(pd.DataFrame({"price": [1, 2, 3, 4]}, index=ts))
    assert list(out["price"]) == [2, 3]


def test_polygon_requires_env_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(RuntimeError): D.PolygonSource()


@pytest.mark.skipif(not os.environ.get("POLYGON_API_KEY"), reason="POLYGON_API_KEY not set")
def test_polygon_live():
    df = D.PolygonSource().get_bars("SPY", "2024-01-02", "2024-01-31")
    assert Q.check_bars(df, daily=False)["ok"]


# ---------- explainability ----------
def test_alerts_jsonl(tmp_path, spy):
    sig = S.make("ema_cross", 20, 100)
    recs = X.alerts_from_signal("SPY", sig, spy, {"run": "t"})
    assert recs and all(r["crossed"] in ("up", "down") for r in recs)
    r0 = recs[0]
    assert {"ema_fast", "ema_slow"} <= set(r0["features"]) and r0["threshold"] == 0.0
    assert (r0["score"] > 0) == (r0["crossed"] == "up")
    assert r0["config_hash"] == X.config_hash({"signal": sig.to_dict(), "run": "t"})
    assert "Not investment advice" in r0["disclaimer"]
    p = X.write_jsonl(recs, tmp_path / "a.jsonl")
    lines = p.read_text().splitlines(); assert len(lines) == len(recs) and json.loads(lines[0])["symbol"] == "SPY"
    assert X.config_hash({"a": 1, "b": 2}) == X.config_hash({"b": 2, "a": 1}) != X.config_hash({"a": 1, "b": 3})


def test_research_script_smoke(tmp_path, monkeypatch):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("p1", pathlib.Path(__file__).parents[1] / "scripts" / "phase1_research.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path)
    mod.main(["--symbols", "SPY", "--train", "504", "--test", "252"])
    assert (tmp_path / "phase1_results.md").exists() and json.loads((tmp_path / "phase1_results.json").read_text())["results"]
