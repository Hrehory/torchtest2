"""Built-in plugins and plugin lookup.

``check_model(..., plugin="lightning")`` resolves the name with :func:`get_plugin`: first the
built-in plugins below, then the entry point of that name in the ``torchtest2.plugins`` group of
installed packages (then the 2021 ``plugins`` group). Only the requested entry point is loaded, so
a broken unrelated plugin cannot break the lookup. None of the built-in plugins imports its
framework, so they add no dependencies.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Any

from torchtest2.plugin_base import Plugin, as_plugin
from torchtest2.plugins.detection import TorchvisionDetectionPlugin
from torchtest2.plugins.huggingface import HuggingFacePlugin
from torchtest2.plugins.lightning import LightningPlugin

ENTRY_POINT_GROUP = "torchtest2.plugins"
LEGACY_ENTRY_POINT_GROUP = "plugins"

BUILTIN_PLUGINS: dict[str, type[Plugin]] = {
    cls.name: cls for cls in (TorchvisionDetectionPlugin, HuggingFacePlugin, LightningPlugin)
}


def _load(entry_point: Any) -> Any:
    """Load an entry point; a plugin that fails to import is reported and left out."""
    try:
        return entry_point.load()
    except Exception as e:  # a broken third-party plugin must not break torchtest2
        warnings.warn(
            f"torchtest2 plugin {entry_point.name!r} could not be loaded ({type(e).__name__}: {e})",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def _load_group(group: str, name: str | None = None) -> dict[str, Any]:
    found = {}
    for ep in entry_points(group=group):
        if name is None or ep.name == name:
            loaded = _load(ep)
            if loaded is not None:
                found[ep.name] = loaded
    return found


def plugin_discovery(group: str | None = None) -> dict[str, Any]:
    """Plugin classes by name: the built-in ones and those that installed packages register under
    the ``torchtest2.plugins`` (and the 2021 ``plugins``) entry-point group; a built-in name is
    never overridden. With ``group``, only that entry-point group is loaded. Plugins that fail to
    import are left out with a ``RuntimeWarning``."""
    if group is not None:
        return _load_group(group)
    found: dict[str, Any] = {}
    for name in (LEGACY_ENTRY_POINT_GROUP, ENTRY_POINT_GROUP):  # the new group wins
        found.update(_load_group(name))
    found.update(BUILTIN_PLUGINS)
    return found


def get_plugin(plugin: str | type[Plugin] | Plugin | Any) -> Plugin:
    """Return a plugin instance for a name, a plugin class or an instance. 2021-style plugin
    objects (with ``split_batch``, ``train_inference``, ... but no base class) are wrapped.

    A name is looked up among the built-in plugins first; otherwise only the entry point of that
    name is loaded (other installed plugins are not imported)."""
    if isinstance(plugin, str):
        name = plugin
        if name in BUILTIN_PLUGINS:
            plugin = BUILTIN_PLUGINS[name]
        else:
            found = _load_group(ENTRY_POINT_GROUP, name) or _load_group(
                LEGACY_ENTRY_POINT_GROUP, name
            )
            if name not in found:
                available = sorted(
                    {*BUILTIN_PLUGINS}
                    | {
                        ep.name
                        for group in (ENTRY_POINT_GROUP, LEGACY_ENTRY_POINT_GROUP)
                        for ep in entry_points(group=group)
                    }
                )
                raise ValueError(f"unknown plugin {name!r}; available: {available}")
            plugin = found[name]
    if isinstance(plugin, type):
        plugin = plugin()
    return as_plugin(plugin)


def _first_optimizer(config: Any) -> Any:
    """The first optimizer in anything ``LightningModule.configure_optimizers`` may return."""
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get("optimizer")
    if isinstance(config, (list, tuple)):
        return _first_optimizer(config[0]) if config else None
    return config


__all__ = [
    "BUILTIN_PLUGINS",
    "ENTRY_POINT_GROUP",
    "HuggingFacePlugin",
    "LightningPlugin",
    "Plugin",
    "TorchvisionDetectionPlugin",
    "get_plugin",
    "plugin_discovery",
]
