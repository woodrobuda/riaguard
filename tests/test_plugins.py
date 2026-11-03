from importlib.metadata import EntryPoint
import types, pytest
from riaguard import plugins


def _fake_eps(monkeypatch, eps):
    monkeypatch.setattr(plugins, "entry_points", lambda group: [e for e in eps if e.group == group])


def test_no_plugins(monkeypatch):
    _fake_eps(monkeypatch, [])
    assert plugins.available() == []
    with pytest.raises(LookupError): plugins.load("khoros")


def test_load_all_registers(monkeypatch):
    mod = types.ModuleType("fake_rg_plugin")
    mod.plugin = types.SimpleNamespace(name="fake", register=lambda ctx: ctx.add_signal("fake_sig", lambda x: x))
    import sys; monkeypatch.setitem(sys.modules, "fake_rg_plugin", mod)
    _fake_eps(monkeypatch, [EntryPoint("fake", "fake_rg_plugin:plugin", plugins.GROUP)])
    assert plugins.available() == ["fake"] and plugins.load("fake").name == "fake"
    ctx, loaded = plugins.load_all(plugins.PluginContext())
    assert "fake" in loaded and ctx.signals["fake_sig"](3) == 3
    with pytest.raises(ValueError): ctx.add_signal("fake_sig", lambda x: x)
