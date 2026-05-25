"""
Regression test: every node class exported from ``glider.nodes.*`` must be
registered with the ``FlowEngine`` so the visual editor can instantiate it.

Half the node library was unreachable from the editor for a long time
because the ``register_hardware_nodes`` / ``register_interface_nodes`` /
``register_math_nodes`` / ``register_comparison_nodes`` /
``register_logic_control_nodes`` functions did not exist — the classes
were defined and exported in ``__all__`` but ``FlowEngine.create_node(
type_str, ...)`` returned ``None`` for every one of them. Dragging "Add"
or "DigitalWrite" onto the canvas silently did nothing.

This test catches that class of bug in CI: it asserts that every concrete
node class exported from each ``glider.nodes.*`` subpackage's ``__all__``
ends up registered with the engine after ``GliderCore._register_builtin_nodes``
runs. New nodes added to ``__all__`` without a matching ``register_node``
call will fail this test immediately.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterator

import pytest

from glider.core.flow_engine import FlowEngine
from glider.nodes.base_node import GliderNode

# Modules whose ``__all__`` entries are concrete node classes we expect to be
# registered. Audio/video have their own ``register_*_nodes`` functions; the
# top-level packages re-export those classes via ``__init__.py``.
_NODE_MODULES = [
    "glider.nodes.experiment_nodes",
    "glider.nodes.control_nodes",
    "glider.nodes.flow_function_nodes",
    "glider.nodes.hardware",
    "glider.nodes.interface",
    "glider.nodes.logic",
    "glider.nodes.vision.zone_nodes",
]


def _iter_exported_node_classes() -> Iterator[tuple[str, type[GliderNode]]]:
    """Yield (qualname, class) for every GliderNode subclass exported from a
    module above.

    Filters out abstract bases, NodeDefinition / PortDefinition dataclasses,
    helper register_* functions, and anything that isn't a class inheriting
    from GliderNode.
    """
    seen: set[type] = set()
    for mod_name in _NODE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:  # pragma: no cover — surface import errors loudly
            pytest.fail(f"Could not import {mod_name}: {e}")

        exported = getattr(mod, "__all__", None)
        if exported is None:
            continue

        for name in exported:
            obj = getattr(mod, name, None)
            if not inspect.isclass(obj):
                continue
            if not issubclass(obj, GliderNode):
                continue
            if obj is GliderNode:
                continue
            # Skip abstract base classes from glider.nodes.base_node that subpackages
            # may re-export.
            if obj in seen:
                continue
            seen.add(obj)
            yield f"{mod_name}.{name}", obj


@pytest.fixture
def engine_with_builtins():
    """A FlowEngine with every built-in node registered (mirrors what
    GliderCore._register_builtin_nodes does, without spinning up the rest of
    the core).
    """
    engine = FlowEngine()

    # Mirror the exact sequence in GliderCore._register_builtin_nodes so the
    # test catches a registration missing from the wiring, not from the
    # per-module function.
    from glider.nodes.control_nodes import register_control_nodes
    from glider.nodes.experiment_nodes import register_experiment_nodes
    from glider.nodes.flow_function_nodes import register_flow_function_nodes
    from glider.nodes.hardware import register_hardware_nodes
    from glider.nodes.interface import register_interface_nodes
    from glider.nodes.interface.audio_nodes import register_audio_nodes
    from glider.nodes.interface.video_nodes import register_video_nodes
    from glider.nodes.logic import (
        register_comparison_nodes,
        register_logic_control_nodes,
        register_math_nodes,
    )
    from glider.nodes.logic.flow_nodes import register_logic_nodes
    from glider.nodes.vision.zone_nodes import register_zone_nodes

    register_experiment_nodes(engine)
    register_control_nodes(engine)
    register_logic_nodes(engine)
    register_flow_function_nodes(engine)
    register_zone_nodes(engine)
    register_audio_nodes(engine)
    register_video_nodes(engine)
    register_interface_nodes(engine)
    register_hardware_nodes(engine)
    register_math_nodes(engine)
    register_comparison_nodes(engine)
    register_logic_control_nodes(engine)

    return engine


@pytest.mark.parametrize(
    "qualname,node_cls",
    list(_iter_exported_node_classes()),
    ids=lambda v: v if isinstance(v, str) else v.__name__,
)
def test_exported_node_is_registered(engine_with_builtins, qualname, node_cls):
    """Every GliderNode subclass exported from glider.nodes.* must be in
    the engine's registry under some name. If this test fails for a node
    class you added, add a ``flow_engine.register_node("YourName", YourCls)``
    call in the corresponding ``register_*_nodes`` function.
    """
    registry = engine_with_builtins._node_registry  # noqa: SLF001
    if node_cls not in registry.values():
        registered = sorted({type_str: c.__name__ for type_str, c in registry.items()}.items())
        pytest.fail(
            f"{qualname} ({node_cls.__name__}) is exported in __all__ but not "
            f"registered with FlowEngine. Add a register_node call.\n"
            f"Currently registered types: {registered}"
        )
