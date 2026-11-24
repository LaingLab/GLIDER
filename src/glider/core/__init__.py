"""
GLIDER Core - The headless orchestration engine.

The Core is responsible for experiment lifecycle management, hardware
driver management, and flow execution. It has no GUI dependencies and
can run as a standalone service.
"""

import importlib

__all__ = [
    "GliderCore",
    "ExperimentSession",
    "HardwareManager",
    "FlowEngine",
    "DataRecorder",
]

_HOMES = {
    "GliderCore": "glider_core",
    "ExperimentSession": "experiment_session",
    "HardwareManager": "hardware_manager",
    "FlowEngine": "flow_engine",
    "DataRecorder": "data_recorder",
}


def __getattr__(name: str):
    # Lazy: glider_core imports glider.vision, which imports glider.core
    # submodules. Eager imports here made that a cycle for anyone who touched
    # glider.vision before glider.core.
    if name in _HOMES:
        return getattr(importlib.import_module(f"glider.core.{_HOMES[name]}"), name)
    raise AttributeError(f"module 'glider.core' has no attribute {name!r}")
