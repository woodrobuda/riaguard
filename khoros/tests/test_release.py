"""Small clean-install tests for the public GitHub package.

These tests do not require the private historical corpus.  That is important
because the public repository intentionally does not redistribute historical
Khoros material whose redistribution rights have not been established.
"""

import numpy as np


from khoros_cantata.kforms_compat import parse_uis
from khoros_cantata.phase21_numeric import (
    conv1d_khoros,
    khoros_fft,
    khoros_sine,
    stats,
)
from khoros_cantata.workspace_runtime import HistoricalWorkspaceParser
from tempfile import TemporaryDirectory


def test_fft_round_trip():
    """A forward FFT followed by the matching inverse returns the input."""
    x = np.arange(8, dtype=float)
    f = khoros_fft(x, [0], direction=0, scale=1)
    y = khoros_fft(f, [0], direction=1, scale=2)
    assert np.allclose(y, x, atol=1e-10)


def test_sine_generator():
    """The source-derived sine generator creates the expected wave."""
    got = khoros_sine((8,), [1.0], [1])
    expected = np.sin(2 * np.pi * np.arange(8) / 8)
    assert np.allclose(got, expected)


def test_convolution_kernel_origin():
    """The convolution helper honors the historical kernel-origin rule."""
    got = conv1d_khoros([1, 2, 3, 4], [1, 10], w0=0, extended=False)
    expected = np.array([1, 12, 23, 34], dtype=float)
    assert np.allclose(got, expected)


def test_stats():
    """Basic statistics match NumPy for a small known data set."""
    result = stats([1, 2, 3, 4])
    assert result["mean"] == 2.5
    assert result["minimum"] == 1.0
    assert result["maximum"] == 4.0
    assert result["points"] == 4


def test_uis_parser_smoke():
    """The UIS parser can read a minimal historical form description."""
    doc = parse_uis("-F 1 0 0 1 1 0 0 0 0 0 0 0\n-E")
    assert doc.records


def test_workspace_parser_smoke():
    """A small historical workspace can be restored without an old executable."""
    text = (
        "VariablesBegin:\n"
        "VariablesEnd:\n"
        "Glyph:DATAMANIP:kabs:1:10:20::\n"
        "#ConnectionBegin\n"
        "NodeConnection:1:out:2:in:\n"
    )
    with TemporaryDirectory() as td:
        path = __import__("pathlib").Path(td) / "sample.wksp"
        path.write_text(text)
        ws = HistoricalWorkspaceParser().parse(path)
    assert len(ws.glyphs) == 1
    assert len(ws.connections) == 1
