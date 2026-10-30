"""Every operator registered by khoros_cantata.operators is exercised through the
registry (the path a Cantata workspace uses), with Khoros-style ports/options."""
import numpy as np
import pytest
from khoros_cantata.workspace_runtime import NativeOperatorRegistry, HistoricalWorkspaceParser, WorkspaceExecutor
from khoros_cantata.operators import CATALOG
from khoros_cantata.viff import ViffObject

R = NativeOperatorRegistry()
x4 = np.array([1.0, 4.0, 9.0, 16.0]); img = np.random.default_rng(0).random((8, 8))
A = np.array([[4., 1], [2, 3]])


def V(data, **segs):
    o = ViffObject(); o.segment('value').data = np.asarray(data)
    for k, v in segs.items(): o.segment(k).data = np.asarray(v)
    return o


ok = lambda o: o['o'] is not None
CASES = {
    'karith1': ({'op': 'sqrt'}, {'i': x4}, lambda o: np.allclose(o['o'], [1, 2, 3, 4])),
    'karith2': ({'op': 'mul', 'real': 2}, {'i1': x4}, lambda o: np.allclose(o['o'], 2 * x4)),
    'kadd': ({}, {'i1': x4, 'i2': 1}, lambda o: np.allclose(o['o'], x4 + 1)),
    'ksub': ({'real': 1}, {'i1': x4}, lambda o: np.allclose(o['o'], x4 - 1)),
    'kmul': ({}, {'i1': x4, 'i2': x4}, lambda o: np.allclose(o['o'], x4 ** 2)),
    'kdiv': ({}, {'i1': x4, 'i2': 2}, lambda o: np.allclose(o['o'], x4 / 2)),
    'kdivinto': ({'real': 1}, {'i1': x4}, lambda o: np.allclose(o['o'], 1 / x4)),
    'kpow': ({'real': 2}, {'i1': x4}, lambda o: np.allclose(o['o'], x4 ** 2)),
    'kmin': ({'real': 5}, {'i1': x4}, lambda o: o['o'].max() == 5),
    'kmax': ({'real': 5}, {'i1': x4}, lambda o: o['o'].min() == 5),
    'kabsdiff': ({'real': 4}, {'i1': x4}, lambda o: np.allclose(o['o'], [3, 0, 5, 12])),
    'kexp': ({}, {'i': [0.0]}, lambda o: np.allclose(o['o'], 1)),
    'klog': ({}, {'i': [1.0]}, lambda o: np.allclose(o['o'], 0)),
    'klog10': ({}, {'i': [100.0]}, lambda o: np.allclose(o['o'], 2)),
    'ksin': ({}, {'i': [0.0]}, lambda o: np.allclose(o['o'], 0)),
    'kcos': ({}, {'i': [0.0]}, lambda o: np.allclose(o['o'], 1)),
    'ktan': ({}, {'i': [0.0]}, lambda o: np.allclose(o['o'], 0)),
    'katan': ({}, {'i': [1.0]}, lambda o: np.allclose(o['o'], np.pi / 4)),
    'kceil': ({}, {'i': [1.2]}, lambda o: np.allclose(o['o'], 2)),
    'ksign': ({}, {'i': [-3.0]}, lambda o: np.allclose(o['o'], -1)),
    'kbitwise': ({'op': 'or', 'value': 1}, {'i1': [4]}, lambda o: list(o['o']) == [5]),
    'kcompare': ({'op': 'gt', 'real': 5}, {'i1': x4}, lambda o: list(o['o']) == [0, 0, 1, 1]),
    'klogexp': ({'op': 'exp', 'base': 2}, {'i': [3.0]}, lambda o: np.allclose(o['o'], 8)),
    'kbessel': ({'kind': 'j', 'order': 1}, {'i': [0.0]}, lambda o: np.allclose(o['o'], 0)),
    'kblend': ({'alpha': 0.5}, {'i1': [0.0], 'i2': [2.0]}, lambda o: np.allclose(o['o'], 1)),
    'kgate': ({}, {'i1': x4, 'i2': [1, 0, 0, 1]}, lambda o: list(o['o']) == [1, 0, 0, 16]),
    'kinverse': ({}, {'i': x4}, lambda o: np.allclose(o['o'], 1 / x4)),
    'kcmplx': ({'op': 'conj'}, {'i': [1 + 1j]}, lambda o: o['o'][0] == 1 - 1j),
    'kreal2cmplx': ({}, {'i1': [1.0], 'i2': [2.0]}, lambda o: o['o'][0] == 1 + 2j),
    'kcmplx2real': ({'mag': True}, {'i': np.array([3 + 4j])}, lambda o: o['om'][0] == 5),
    'kmag': ({'mag': True}, {'i': np.array([3 + 4j])}, lambda o: o['om'][0] == 5),
    'kconvert': ({'type': 'ubyte'}, {'i': [300.0, 1.4]}, lambda o: o['o'].tolist() == [255, 1]),
    'knormal': ({}, {'i': x4}, lambda o: o['o'].min() == 0 and o['o'].max() == 1),
    'kreplace': ({'lval': 4, 'uval': 9, 'new': 0}, {'i': x4}, lambda o: o['o'].tolist() == [1, 0, 0, 16]),
    'kmapdata': ({}, {'i1': [0, 1], 'i2': [7, 8]}, lambda o: o['o'].tolist() == [7, 8]),
    'kunmask': ({'value': -1}, {'i1': x4, 'i2': [1, 1, 0, 1]}, lambda o: o['o'].tolist() == [1, 4, -1, 16]),
    'kextract': ({'woff': 1, 'wsize': 2}, {'i': x4}, lambda o: o['o'].tolist() == [4, 9]),
    'kinset': ({'woff': 1}, {'i1': np.zeros(4), 'i2': [5.0]}, lambda o: o['o'].tolist() == [0, 5, 0, 0]),
    'kpad': ({'wsize': 256, 'hsize': 256, 'woff': 0, 'hoff': 0, 'real': 0}, {'i': np.ones((21, 1))}, lambda o: o['o'].shape == (256, 256) and o['o'].sum() == 21),
    'kflip': ({'w': True}, {'i': x4}, lambda o: o['o'].tolist() == [16, 9, 4, 1]),
    'korient': ({'order': 'hw'}, {'i': np.zeros((2, 3))}, lambda o: o['o'].shape == (3, 2)),
    'kappend': ({'axis': 'w'}, {'i1': x4, 'i2': [0.0]}, lambda o: o['o'].size == 5),
    'ksampline': ({'x1': 0, 'y1': 0, 'x2': 7, 'y2': 0}, {'i': img}, lambda o: np.allclose(o['o'], img[:, 0])),
    'kprdata': ({}, {'i': [1, 2]}, lambda o: o['o'] == '1 2\n'),
    'kprval': ({'w': 1}, {'i': x4}, lambda o: o['o'] == '4'),
    'kimportasc': ({}, {'i': '1 2\n3'}, lambda o: o['o'].tolist() == [1, 2, 3]),
    'kimportraw': ({'type': 'int16'}, {'i': np.array([3], '<i2').tobytes()}, lambda o: o['o'].tolist() == [3]),
    'kkmeans': ({'k': 2}, {'i': np.r_[np.zeros(5), np.ones(5) * 9]}, lambda o: len(set(o['o'])) == 2 and o['o2'].shape == (2, 1)),
    'kstats': ({}, {'i': x4}, lambda o: o['o']['mean'] == 7.5),
    'kaxis': ({'wsize': 3, 'hsize': 2, 'axis': 'h'}, {}, lambda o: o['o'].tolist() == [[0, 1]] * 3),
    'kgenloc': ({'wsize': 2, 'hsize': 2}, {}, lambda o: o['o'].shape == (2, 2, 2)),
    'klocxform': ({'matrix': [[2, 0], [0, 2]]}, {'i': [[1.0, 1.0]]}, lambda o: o['o'].tolist() == [[2, 2]]),
    'kcpseg': ({'segment': 'mask'}, {'i1': V(x4), 'i2': V(x4, mask=[1, 0, 1, 1])}, lambda o: 'mask' in o['o'].segments),
    'krmseg': ({'segment': 'mask'}, {'i': V(x4, mask=[1, 1, 1, 1])}, lambda o: 'mask' not in o['o'].segments),
    'kinsertseg': ({'segment': 'time'}, {'i1': V(x4), 'i2': [0, 1, 2, 3]}, lambda o: 'time' in o['o'].segments),
    'kcptoval': ({'segment': 'mask'}, {'i': V(x4, mask=[0, 1, 0, 1])}, lambda o: o['o'].segments['value'].data.tolist() == [0, 1, 0, 1]),
    'kcpfromval': ({'segment': 'map'}, {'i': V(x4)}, lambda o: o['o'].segments['map'].data.tolist() == x4.tolist()),
    'kasc2val': ({}, {'i': '1 2'}, lambda o: o['o'].segments['value'].data.tolist() == [1, 2]),
    'kasc2mask': ({}, {'i': '1 0', 'i2': V([5.0, 6.0])}, lambda o: o['o'].segments['mask'].data.tolist() == [1, 0]),
    'kasc2map': ({}, {'i': '0 255'}, lambda o: 'map' in o['o'].segments),
    'kasc2loc': ({}, {'i': '0 1'}, lambda o: 'location' in o['o'].segments),
    'kasc2time': ({}, {'i': '0 1'}, lambda o: 'time' in o['o'].segments),
    'ksegcmp': ({}, {'i1': V(x4), 'i2': V(x4)}, lambda o: o['o'] is True),
    'ksetdattr': ({'attribute': 'units', 'value': 'V'}, {'i': V(x4)}, lambda o: o['o'].attributes['units'] == 'V'),
    'kcomment': ({'comment': 'c'}, {'i': V(x4)}, lambda o: o['o'].attributes['comment'] == 'c'),
    'kfft': ({'dir': 0, 'scale': 1, 'w': True}, {'i': np.ones(4)}, lambda o: np.allclose(o['o'], [4, 0, 0, 0])),
    'kgsin': ({'wsize': 8, 'wnum': 1}, {}, lambda o: np.allclose(o['o'], np.sin(2 * np.pi * np.arange(8) / 8))),
    'klinearop': ({'w0': 0}, {'i1': [1, 2, 3, 4], 'i2': [1, 10]}, lambda o: o['o'].tolist() == [1, 12, 23, 34]),
    'kgconst': ({'wsize': 3, 'real': 2.0}, {}, lambda o: o['o'].tolist() == [2, 2, 2]),
    'kimpulse': ({'wsize': 4, 'woff': 2}, {}, lambda o: o['o'].tolist() == [0, 0, 1, 0]),
    'kgnoise': ({'wsize': 50000, 'mean': 1, 'var': 4, 'seed': 0}, {}, lambda o: abs(o['o'].std() - 2) < 0.05),
    'knoise': ({'var': 1, 'seed': 0}, {'i': np.zeros(5000)}, lambda o: abs(o['o'].std() - 1) < 0.05),
    'kshot': ({'prob': 1.0, 'min': 0, 'max': 1, 'seed': 0}, {'i': np.full(100, .5)}, lambda o: set(o['o']) <= {0, 1}),
    'khisto': ({'bins': 4}, {'i': x4}, lambda o: o['o'].sum() == 4 and o['o2'].size == 5),
    'khistops': ({'op': 'stretch', 'min': 0, 'max': 1}, {'i1': x4}, lambda o: o['o'].max() == 1),
    'kresample': ({'wsize': 8, 'interp': 'zero'}, {'i': x4}, lambda o: o['o'].tolist() == [1, 1, 4, 4, 9, 9, 16, 16]),
    'kwiener': ({'nsr': 1e-9}, {'i1': x4, 'i2': [1.0]}, lambda o: np.allclose(o['o'], x4)),
    'kwindow': ({'type': 'rect'}, {'i': x4}, lambda o: np.allclose(o['o'], x4)),
    'kgwindow': ({'wsize': 5, 'type': 'hamming'}, {}, lambda o: np.allclose(o['o'], [.08, .54, 1, .54, .08])),
    'kfir': ({'mode': 'causal'}, {'i1': x4, 'i2': [0.0, 1.0]}, lambda o: np.allclose(o['o'], [0, 1, 4, 9])),
    'kiir': ({'order': 2, 'cutoff': 0.2}, {'i': np.ones(400)}, lambda o: abs(o['o'][-1] - 1) < 1e-6),
    'kconvolve': ({}, {'i1': [1.0, 1.0], 'i2': [1.0, 1.0]}, lambda o: np.allclose(o['o'], [1, 2, 1])),
    'kcorr': ({}, {'i1': [1.0, 2.0], 'i2': [1.0, 2.0]}, lambda o: np.allclose(o['o'], [2, 5, 2])),
    'kpsd': ({}, {'i': np.sin(np.pi / 2 * np.arange(64))}, lambda o: np.isclose(o['o2'][np.argmax(o['o'])], 0.25)),
    'kspectrum': ({}, {'i': np.ones(4)}, lambda o: np.allclose(o['o'], [4, 0, 0, 0])),
    'kema': ({'alpha': 0.5}, {'i': [0.0, 2.0, 2.0]}, lambda o: np.allclose(o['o'], [0, 1, 1.5])),
    'krunstats': ({'stat': 'max', 'window': 2}, {'i': [3.0, 1.0, 0.0]}, lambda o: np.allclose(o['o'], [3, 3, 1])),
    'kdelay': ({'n': 1}, {'i': [1.0, 2.0]}, lambda o: np.allclose(o['o'], [0, 1])),
    'kdiff': ({}, {'i': [1.0, 4.0]}, lambda o: np.allclose(o['o'], [0, 3])),
    'kcross': ({}, {'i1': [0.0, 2.0, 0.0], 'i2': [1.0, 1.0, 1.0]}, lambda o: np.allclose(o['o'], [0, 1, -1])),
    'khilbert': ({}, {'i': np.cos(2 * np.pi * np.arange(64) / 8)}, lambda o: np.allclose(o['o'], 1)),
    'klogret': ({}, {'i': [1.0, np.e]}, lambda o: np.allclose(o['o'], [0, 1])),
    'kzscore': ({'window': 3}, {'i': [1.0, 2.0, 3.0]}, lambda o: np.isclose(o['o'][2], 1 / np.std([1, 2, 3]))),
    'khold': ({}, {'i': [0.0, 1.0, 0.0]}, lambda o: o['o'].tolist() == [0, 1, 1]),
    'kkalman': ({}, {'i': np.arange(50.0)}, lambda o: abs(o['o2'][-1] - 1) < 0.05),
    'kkalmanpair': ({}, {'i1': np.arange(1.0, 50), 'i2': 2 * np.arange(1.0, 50)}, lambda o: abs(o['o2'][-1] - 2) < 0.1),
    'kthreshold3': ({'lo': 0, 'hi': 1}, {'i': [-1.0, 0.5, 2.0]}, lambda o: o['o'].tolist() == [0, 1, 2]),
    'kdwt': ({'levels': 1}, {'i': [1.0, 1.0]}, lambda o: np.allclose(o['o'][0], [np.sqrt(2)])),
    'kidwt': ({}, {'i': [np.array([np.sqrt(2)]), np.array([0.0])]}, lambda o: np.allclose(o['o'], [1, 1])),
    'imedian': ({'wsize': 3}, {'i': img}, lambda o: o['o'].shape == img.shape),
    'igradient': ({'op': 'sobel'}, {'i': img}, lambda o: all(k in o for k in ('o', 'o2', 'o3', 'o4'))),
    'iedge': ({'thresh': 0.5}, {'i': img}, lambda o: set(np.unique(o['o'])) <= {0, 1}),
    'ilaplacian': ({}, {'i': np.ones((5, 5))}, lambda o: np.allclose(o['o'], 0)),
    'irotate': ({'angle': 360}, {'i': img}, lambda o: np.allclose(o['o'], img, atol=1e-6)),
    'iscale': ({'wfactor': 2}, {'i': img}, lambda o: o['o'].shape == (16, 16)),
    'itranslate': ({'woff': 1, 'interp': 0}, {'i': img}, lambda o: np.allclose(o['o'][1:], img[:-1])),
    'igeowarp': ({'xcoef': [0, 1, 0], 'ycoef': [0, 0, 1]}, {'i': img}, lambda o: np.allclose(o['o'], img)),
    'igauss_func': ({'wsize': 5, 'hsize': 5}, {}, lambda o: o['o'][2, 2] == 1),
    'itexture': ({'feature': 'mean', 'wsize': 3}, {'i': np.ones((5, 5))}, lambda o: np.allclose(o['o'], 1)),
    'ifilt_design': ({'wsize': 16, 'hsize': 16, 'shape': 'ideal'}, {}, lambda o: o['o'][8, 8] == 1),
    'imorph': ({'op': 'dilate'}, {'i': np.eye(5)}, lambda o: o['o'].sum() > 5),
    'ierode': ({}, {'i': np.eye(5)}, lambda o: o['o'].sum() == 0),
    'idilate': ({}, {'i': np.eye(5)}, lambda o: o['o'].sum() > 5),
    'iopen': ({}, {'i': np.ones((5, 5))}, lambda o: o['o'].sum() == 25),
    'iclose': ({}, {'i': np.ones((5, 5))}, lambda o: o['o'].sum() == 25),
    'ithresh': ({'lval': 0.5}, {'i': img}, lambda o: o['o'].sum() == (img >= .5).sum()),
    'ilabel': ({}, {'i': np.eye(3)}, lambda o: o['o'].max() == 1),
    'igamut': ({'ncolors': 2}, {'i': np.random.default_rng(1).random((4, 4, 3))}, lambda o: o['o2'].shape == (2, 3)),
    'mmul': ({}, {'i1': A, 'i2': np.eye(2)}, lambda o: np.allclose(o['o'], A)),
    'minvert': ({}, {'i': A}, lambda o: np.allclose(o['o'] @ A, np.eye(2))),
    'mpow': ({'power': 2}, {'i': A}, lambda o: np.allclose(o['o'], A @ A)),
    'mextract_diag': ({}, {'i': A}, lambda o: o['o'].tolist() == [4, 3]),
    'mgdiag': ({}, {'i': [1, 2]}, lambda o: o['o'].tolist() == [[1, 0], [0, 2]]),
    'mreplicate': ({'wrep': 2}, {'i': A}, lambda o: o['o'].shape == (2, 4)),
    'mrcsum': ({}, {'i': A}, lambda o: o['o'].tolist() == [5, 5] and o['o2'].tolist() == [6, 4]),
    'mcovar': ({}, {'i': np.random.default_rng(0).random((10, 2))}, lambda o: o['o'].shape == (2, 2)),
    'mexchg': ({}, {'i': A}, lambda o: o['o'].tolist() == [[2, 3], [4, 1]]),
    'meigen': ({}, {'i': A}, lambda o: np.allclose(sorted(o['o'].real), [2, 5])),
    'msvd': ({}, {'i': A}, lambda o: np.allclose(o['o'] * o['o2'] @ o['o3'], A)),
    'mlud': ({}, {'i': A}, lambda o: np.allclose(o['o'] @ o['o2'] @ o['o3'], A)),
    'mlse': ({}, {'i1': np.eye(2), 'i2': [1.0, 2.0]}, lambda o: np.allclose(o['o'], [1, 2])),
    'kcolorspace': ({'conv': 'rgb2grey'}, {'i': np.ones((2, 3))}, lambda o: np.allclose(o['o'], 1)),
    'ggencmap': ({'n': 4}, {}, lambda o: o['o'].shape == (4, 3)),
    'gcolorize': ({}, {'i1': [0.0, 1.0]}, lambda o: o['o'].shape == (2, 3)),
    'ggridder2d': ({'wsize': 5, 'hsize': 5}, {'i1': [[0, 0], [1, 0], [0, 1], [1, 1]], 'i2': [0.0, 1, 0, 1]}, lambda o: np.allclose(o['o'][:, 2], np.linspace(0, 1, 5))),
    'gextents': ({}, {'i': [[0, 1], [2, -1]]}, lambda o: o['o'].tolist() == [0, -1]),
    'kslicer': ({'axis': 'd', 'index': 1}, {'i': np.arange(8).reshape(2, 2, 2)}, lambda o: o['o'].tolist() == [[1, 3], [5, 7]]),
}


def test_every_registered_operator_has_a_case():
    assert sorted(CATALOG) == sorted(CASES)


@pytest.mark.parametrize('name', sorted(CASES))
def test_operator_via_registry(name):
    params, inputs, check = CASES[name]
    op = R.resolve(CATALOG[name], name); assert op is not None
    assert R.resolve('*', name) is not None
    assert check(op(params, inputs))


def test_viff_object_passes_through_unary_op():
    o = V(x4.reshape(4, 1, 1, 1, 1)); o.attributes['comment'] = 'keep'
    out = R.resolve('DATAMANIP', 'karith1')({'op': 'sqrt'}, {'i': o})['o']
    assert isinstance(out, ViffObject) and out.attributes['comment'] == 'keep'
    assert np.allclose(out.segments['value'].data.ravel(), [1, 2, 3, 4]) and np.allclose(o.segments['value'].data.ravel(), x4)


def test_signal_workspace_end_to_end(tmp_path):
    """kgnoise -> kiir -> kresample -> khisto, written in the Khoros 2 workspace dialect."""
    w = tmp_path / 'sig.wk'
    w.write_text(
        "#GlyphBegin\n"
        "Glyph:DATAMANIP:kgnoise::1:0:0:: -wsize 4096 -mean 0 -var 1 -seed 3\n"
        "Glyph:SIGNAL:kiir::2:0:0:: -order 4 -cutoff 0.05 -family 'butter'\n"
        "Glyph:DATAMANIP:kresample::3:0:0:: -wsize 512 -interp 'zero'\n"
        "Glyph:DATAMANIP:khisto::4:0:0:: -bins 16\n"
        "Glyph:DATAMANIP:kclip::5:0:0:: -lc 0 -uc 1\n"
        "#GlyphEnd\n"
        "NodeConnection:1:o:2:i:\nNodeConnection:2:o:3:i:\nNodeConnection:3:o:4:i:\nNodeConnection:3:o:5:i:\n")
    ws = WorkspaceExecutor().execute(HistoricalWorkspaceParser().parse(w))
    filt = ws.nodes['2'].outputs['o'].data.value; res = ws.nodes['3'].outputs['o'].data.value
    assert filt.shape == (4096,) and filt.std() < 0.5 and res.shape == (512,)
    assert ws.nodes['4'].outputs['o'].data.value.sum() == 512
    clipped = ws.nodes['5'].outputs['o'].data.value
    assert clipped.min() >= 0 and clipped.max() <= 1 and (clipped == 0).any()


def test_image_workspace_end_to_end(tmp_path):
    w = tmp_path / 'img.wk'
    w.write_text(
        "#GlyphBegin\n"
        "Glyph:IMAGE:igauss_func::1:0:0:: -wsize 32 -hsize 32 -xvar 20\n"
        "Glyph:DATAMANIP:kshot::2:0:0:: -prob 0.1 -min 0 -max 1 -seed 1\n"
        "Glyph:IMAGE:imedian::3:0:0:: -wsize 3\n"
        "Glyph:IMAGE:irotate::4:0:0:: -angle 45\n"
        "Glyph:IMAGE:igradient::5:0:0:: -op 'sobel'\n"
        "#GlyphEnd\n"
        "NodeConnection:1:o:2:i:\nNodeConnection:2:o:3:i:\nNodeConnection:3:o:4:i:\nNodeConnection:4:o:5:i:\n")
    ws = WorkspaceExecutor().execute(HistoricalWorkspaceParser().parse(w))
    clean = ws.nodes['1'].outputs['o'].data.value; med = ws.nodes['3'].outputs['o'].data.value
    noisy = ws.nodes['2'].outputs['o'].data.value
    assert np.abs(med - clean).mean() < np.abs(noisy - clean).mean() / 3
    assert ws.nodes['5'].outputs['o'].data.value.shape == (32, 32)
