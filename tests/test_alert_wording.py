import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("caw", ROOT / "scripts" / "check_alert_wording.py")
caw = importlib.util.module_from_spec(spec); spec.loader.exec_module(caw)


def test_repo_alert_text_is_clean():
    assert caw.main(ROOT) == 0


def test_flags_prescriptive_words():
    for bad in ["Sell SPY now", "You should exit", "Great buying opportunity", "Act now",
                "This will rise", "guaranteed gain", "Take profit", "Go long QQQ"]:
        assert caw.find_banned(bad), bad


def test_no_false_positive_on_descriptive_or_substrings():
    for ok in ["EMA crossed below threshold", "buy_and_hold benchmark", "household", "address",
               "Research alert. Not investment advice. No order was placed."]:
        assert not caw.find_banned(ok), ok


def test_docstrings_skipped_literals_flagged():
    src = '"""Docstring may say you should not buy."""\nMSG = "Consider buying now"\n'
    out = caw.scan_python(src)
    assert len(out) == 1 and "consider buying" in out[0]


def test_jsonl_scan():
    assert caw.scan_jsonl('{"note": "sell now", "config_hash": "x"}\n')
    assert not caw.scan_jsonl('{"crossed": "down", "ts": "2026"}\n')
