import numpy as np, pytest
ka = pytest.importorskip("signal_trading.khoros_adapter")
if not ka.AVAILABLE:
    pytest.skip("khoros_cantata not installed", allow_module_level=True)


def test_causal_sma_matches_pandas():
    import pandas as pd
    x = np.random.default_rng(0).normal(size=100).cumsum()
    assert np.allclose(ka.causal_sma_khoros(x, 10)[9:], pd.Series(x).rolling(10).mean()[9:])


def test_power_spectrum_peak():
    t = np.arange(256); p = ka.power_spectrum_khoros(np.sin(2 * np.pi * 8 * t / 256))
    assert np.argmax(p[:128]) == 8


def test_workspace_pipeline(tmp_path):
    r = ka.register_operators()
    px = 100 * np.exp(np.random.default_rng(1).normal(0, .01, 300).cumsum())
    r.register("USER", "src", lambda p, i: {"o": px})
    got = {}
    r.register("USER", "sink", lambda p, i: (got.update(sig=i["i"]), {})[1])
    (tmp_path / "w.wksp").write_text(
        "VariablesBegin:\nVariablesEnd:\n"
        "Glyph:USER:src:1:0:0::\nGlyph:SIGTRADE:kema:2:0:0::kema -span 10\n"
        "Glyph:SIGTRADE:ksma:3:0:0::ksma -n 50\nGlyph:SIGTRADE:kcross:4:0:0::\nGlyph:USER:sink:5:0:0::\n"
        "#ConnectionBegin\nNodeConnection:1:o:2:i:\nNodeConnection:1:o:3:i:\n"
        "NodeConnection:2:o:4:i:\nNodeConnection:3:o:4:i2:\nNodeConnection:4:o:5:i:\n")
    ka.run_workspace(tmp_path / "w.wksp", r)
    assert got["sig"].shape == (300,) and set(np.unique(got["sig"])) <= {0.0, 1.0}


STREAM_WK = (
    "#GlyphBegin\n"
    "Glyph:STREAM:kinput::1:0:0:: -name 'price'\n"
    "Glyph:SIGTRADE:kema::2:0:0:: -span 10\n"
    "Glyph:SIGTRADE:ksma::3:0:0:: -n 30\n"
    "Glyph:SIGTRADE:kcross::4:0:0::\n"
    "Glyph:SIGTRADE:kkalman::5:0:0::\n"
    "Glyph:SIGTRADE:klogret::6:0:0::\n"
    "Glyph:STREAM:koutput::7:0:0:: -name 'sig'\n"
    "Glyph:STREAM:koutput::8:0:0:: -name 'kal'\n"
    "Glyph:STREAM:koutput::9:0:0:: -name 'ret'\n"
    "Glyph:STREAM:koutput::10:0:0:: -name 'sma'\n"
    "#GlyphEnd\n#ConnectionBegin\n"
    "NodeConnection:1:o:2:i:\nNodeConnection:1:o:3:i:\nNodeConnection:2:o:4:i:\nNodeConnection:3:o:4:i2:\n"
    "NodeConnection:4:o:7:i:\nNodeConnection:1:o:5:i:\nNodeConnection:5:o:8:i:\nNodeConnection:1:o:6:i:\n"
    "NodeConnection:6:o:9:i:\nNodeConnection:3:o:10:i:\n#ConnectionEnd\n")


def _stream_or_skip():
    if not ka.STREAMING_AVAILABLE:
        pytest.skip("installed khoros_cantata has no streaming module")


def test_streaming_matches_batch_and_is_causal(tmp_path):
    _stream_or_skip()
    from khoros_cantata.streaming import batch_execute, assert_no_lookahead
    (tmp_path / "s.wk").write_text(STREAM_WK)
    px = 100 * np.exp(np.random.default_rng(2).normal(0, .01, 400).cumsum())
    ex = ka.stream_executor(tmp_path / "s.wk")
    live = {k: [] for k in ex.outputs}
    for p in px:
        for k, v in ex.push(price=p).items(): live[k].append(v[0])
    batch = batch_execute(tmp_path / "s.wk", {"price": px}, ka.register_operators())
    for k in ("sig", "kal", "ret", "sma"):
        assert np.allclose(np.asarray(live[k]), batch[k], equal_nan=True), k
    def f(x):
        ex.reset(); return ex.run({"price": x}, chunk=25)
    assert_no_lookahead(f, px)


def test_streaming_rejects_kpower(tmp_path):
    _stream_or_skip()
    from khoros_cantata.streaming import NonCausalGraphError
    (tmp_path / "p.wk").write_text("#GlyphBegin\nGlyph:STREAM:kinput::1:0:0:: -name 'x'\n"
                                   "Glyph:SIGTRADE:kpower::2:0:0::\n#GlyphEnd\nNodeConnection:1:o:2:i:\n")
    with pytest.raises(NonCausalGraphError, match="kpower"):
        ka.stream_executor(tmp_path / "p.wk")
