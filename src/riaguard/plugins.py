"""Plug-in hook for optional riaguard extensions.

A plug-in is a separately installed distribution that advertises an entry point in the
``riaguard.plugins`` group, for example in its ``pyproject.toml``::

    [project.entry-points."riaguard.plugins"]
    khoros = "riaguard_khoros:plugin"

The entry point resolves to an object with a ``name`` attribute and, optionally, a
``register(ctx)`` callable. ``register`` receives a :class:`PluginContext` and may add data
sources or signals through it. Plug-ins must follow riaguard's scope: analysis and alerts
only, never order execution. The core never imports a plug-in by name.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Callable

GROUP = "riaguard.plugins"


@dataclass
class PluginContext:
    """What a plug-in may extend. ``datasources`` is the live :data:`riaguard.datasources.SOURCES` map."""
    datasources: dict = field(default_factory=dict)
    signals: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def add_datasource(self, key: str, cls) -> None:
        if key in self.datasources: raise ValueError(f"data source {key!r} already registered")
        self.datasources[key] = cls

    def add_signal(self, key: str, fn: Callable[..., Any]) -> None:
        if key in self.signals: raise ValueError(f"signal {key!r} already registered")
        self.signals[key] = fn


def available() -> list[str]:
    """Names of installed plug-ins (entry points in the ``riaguard.plugins`` group)."""
    return sorted(ep.name for ep in entry_points(group=GROUP))


def load(name: str):
    """Import and return one plug-in object by entry-point name."""
    for ep in entry_points(group=GROUP):
        if ep.name == name: return ep.load()
    raise LookupError(f"riaguard plug-in {name!r} is not installed (installed: {available()})")


def load_all(ctx: PluginContext | None = None) -> tuple[PluginContext, dict[str, Any]]:
    """Load every installed plug-in and call its ``register(ctx)`` if it has one."""
    if ctx is None:
        from .datasources import SOURCES
        ctx = PluginContext(datasources=SOURCES)
    loaded = {}
    for ep in entry_points(group=GROUP):
        obj = ep.load(); loaded[ep.name] = obj
        reg = getattr(obj, "register", None)
        if callable(reg): reg(ctx)
    return ctx, loaded
