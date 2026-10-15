"""Experiment presets: the type an experiment declares, and what it unlocks.

The open experiment's ``metadata.experiment_type`` is one value shared across
the app: the Experiment tab and Run-tab presets set it; Session Review and the
Batch Pose window read and edit it. Each preset lists the analyses it unlocks,
and every analysis's UI asks :func:`has_analysis` for its own key -- so adding a
protocol for a type (an Open Field analysis, say) is one entry in ``PRESETS``
plus that protocol's code, and nothing else needs the type's name. Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ANALYSIS_PUP_RETRIEVAL",
    "EXPERIMENT_TYPES",
    "OPEN_FIELD",
    "PRESETS",
    "PUP_RETRIEVAL",
    "ExperimentPreset",
    "has_analysis",
    "preset_for",
    "type_label",
]

OPEN_FIELD = "Open Field"
PUP_RETRIEVAL = "Pup Retrieval"

ANALYSIS_PUP_RETRIEVAL = "pup_retrieval"


@dataclass(frozen=True)
class ExperimentPreset:
    """An experiment type and the analyses Session Review offers under it."""

    name: str
    analyses: frozenset[str] = frozenset()


PRESETS: tuple[ExperimentPreset, ...] = (
    ExperimentPreset(""),
    ExperimentPreset(OPEN_FIELD),
    ExperimentPreset(PUP_RETRIEVAL, frozenset({ANALYSIS_PUP_RETRIEVAL})),
    *(
        ExperimentPreset(name)
        for name in (
            "Morris Water Maze",
            "Elevated Plus Maze",
            "Rotarod",
            "Novel Object Recognition",
            "Fear Conditioning",
            "Forced Swim Test",
            "Tail Suspension Test",
            "Y-Maze",
            "Barnes Maze",
            "Radial Arm Maze",
            "Social Interaction",
            "Light-Dark Box",
            "Other",
        )
    ),
)

EXPERIMENT_TYPES: tuple[str, ...] = tuple(p.name for p in PRESETS)


def preset_for(experiment_type: str | None) -> ExperimentPreset | None:
    """The preset an experiment type names, or None for a custom type.

    An exact (case-insensitive) name first; else the longest preset name inside
    the type, because Run-tab presets store their *name* as the type --
    "Pup Retrieval (P4, 5 min)" is a Pup Retrieval experiment.
    """
    text = (experiment_type or "").strip().casefold()
    for preset in PRESETS:
        if preset.name.casefold() == text:
            return preset
    named = [p for p in PRESETS if p.name and p.name.casefold() in text]
    return max(named, key=lambda p: len(p.name)) if named else None


def has_analysis(experiment_type: str | None, key: str) -> bool:
    """Whether *experiment_type*'s preset unlocks the analysis *key*."""
    preset = preset_for(experiment_type)
    return preset is not None and key in preset.analyses


def type_label(experiment_type: str) -> str:
    return experiment_type or "None"
