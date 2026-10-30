import json
import pytest
from khoros_cantata import catalog_gen
from khoros_cantata.catalog_gen import CATALOG_JSON, build_catalog, eval_lag
from khoros_cantata.streaming import classify, streaming_registry
from khoros_cantata.workspace_runtime import NativeOperatorRegistry

CAT = json.loads(CATALOG_JSON.read_text())
BY = {(o['toolbox'], o['name']): o for o in CAT['operators']}


def test_every_registered_operator_is_in_catalog():
    reg = streaming_registry(registry=NativeOperatorRegistry())
    missing = [f'{tb}:{n}' for tb, n in reg._ops if tb != '*' and (tb, n) not in BY]
    assert not missing, f'run python -m khoros_cantata.catalog_gen; missing: {missing}'


def test_catalog_files_up_to_date():
    assert catalog_gen.main(['--check']) == 0


@pytest.mark.parametrize('key', sorted(BY))
def test_entry_well_formed_and_consistent(key):
    o = BY[key]
    assert o['description'] and o['outputs'] and isinstance(o['params'], dict)
    for k, v in o['params'].items():
        assert v['type'] in ('number', 'integer', 'string', 'array')
        if v.get('range'): assert v['range'][0] <= v['range'][1]
        if v.get('choices') and v.get('default') is not None: assert v['default'] in v['choices']
    if o['name'] not in ('kinput', 'koutput', 'krolling'):
        assert o['streamable'] == (classify(o['name'], {}, (), o['toolbox'])[0] == 'streamable')
    if o['lag'] is not None:
        defaults = {k: v['default'] for k, v in o['params'].items() if v.get('default') is not None}
        lag = eval_lag(o['lag'], defaults)
        assert lag is None or lag >= 0


def test_known_lags():
    assert eval_lag(BY[('SIGNAL', 'kema')]['lag'], {'span': 21}) == 10
    assert eval_lag(BY[('DATAMANIP', 'klinearop')]['lag'], {'n': 5}) == 2
    assert eval_lag(BY[('SIGNAL', 'kfir')]['lag'], {'taps': 31}) == 15
    import numpy as np
    from khoros_cantata import ops_signal as S
    h = S.kiir(np.r_[1.0, np.zeros(4000)], S.kiir_design(2, 0.05))  # DC group delay = impulse-response centroid
    assert abs(eval_lag(BY[('SIGNAL', 'kiir')]['lag'], {'order': 2, 'cutoff': 0.05}) - (np.arange(h.size) @ h) / h.sum()) < 1e-3
    assert eval_lag(BY[('FINANCE', 'kkalman')]['lag'], {}) > 0
    assert BY[('SIGNAL', 'kfft')]['streamable'] is False if ('SIGNAL', 'kfft') in BY else not BY[('DATAMANIP', 'kfft')]['streamable']
