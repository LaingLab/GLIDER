"""
GLIDER - General Laboratory Interface for Design, Experimentation, and Recording

A modular experimental orchestration platform for laboratory hardware control
through visual flow-based programming.
"""

from glider._version import __version__

__author__ = "LaingLab"

__all__ = ["GliderCore", "ExperimentSession", "__version__"]


def __getattr__(name: str):
    # Lazy, so importing any glider submodule does not drag in the core (and
    # OpenCV with it): the launch splash runs in its own process and has to be
    # up in a fraction of a second.
    if name == "GliderCore":
        from glider.core.glider_core import GliderCore

        return GliderCore
    if name == "ExperimentSession":
        from glider.core.experiment_session import ExperimentSession

        return ExperimentSession
    raise AttributeError(f"module 'glider' has no attribute {name!r}")
