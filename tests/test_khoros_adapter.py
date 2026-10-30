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
