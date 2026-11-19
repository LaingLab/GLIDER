"""Plugin measures over a range: extra columns for the epoch table and its exports. Qt-free.

A plugin adds measures by declaring an entry point in the ``glider.range_metrics``
group whose target is a factory: called with no arguments, it returns one
:class:`RangeMetricProvider` (or a list of them). For example::

    [project.entry-points."glider.range_metrics"]
    lift3d = "glider_lift3d.metrics:provider"

Session Review asks each provider for every session and range it computes
(``AnalysisWindow.range_rows``), so the values land in the Epoch table, the
range-stats export and the time-bin export alike. A provider returns ``{}`` for a
session it has nothing for; its metrics are offered only when some session
answers. Keys should carry the plugin's name as a prefix, so two plugins never
collide with each other or with GLIDER's own columns, and must not end in
``_pct`` (read as seconds over the range; see :func:`glider.analysis.epochs.metric_value`).

A provider that raises is logged and dropped for the rest of the session; it
never breaks an export.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from glider.analysis.epochs import Metric

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "glider.range_metrics"


@runtime_checkable
class RangeMetricProvider(Protocol):
    name: str

    def metrics(self) -> list[Metric]:
        """Every measure this provider can give, in reading order."""
        ...

    def values(self, view: Any, start_frame: int, end_frame: int) -> dict[str, float]:
        """Measures over frames ``[start_frame, end_frame]`` (inclusive) of *view*.

        *view* is the session's :class:`~glider.analysis.behavior.session_view.SessionView`
        (``pose_path``, ``video_path``, ``fps``, ``px_per_mm``). ``{}`` when the
        session has nothing for this provider; NaN for a measure it cannot give
        over this range.
        """
        ...


def load_range_metric_providers(entry_points=None) -> list[RangeMetricProvider]:
    """Every provider the installed plugins offer, in entry-point name order."""
    if entry_points is None:
        from importlib.metadata import entry_points as _eps

        entry_points = _eps(group=ENTRY_POINT_GROUP)
    providers: list[RangeMetricProvider] = []
    for ep in sorted(entry_points, key=lambda e: e.name):
        try:
            made = ep.load()()
        except Exception:  # noqa: BLE001 - a broken plugin is skipped, not fatal
            logger.exception("range metric plugin %r failed to load", ep.name)
            continue
        for provider in made if isinstance(made, list | tuple) else [made]:
            if not isinstance(provider, RangeMetricProvider):
                logger.warning("range metric plugin %r gave %r", ep.name, provider)
                continue
            providers.append(provider)
    return providers


class RangeMetricHost:
    """Collects plugin columns for one session row, isolating provider failures."""

    def __init__(self, providers: list[RangeMetricProvider]):
        self.providers = list(providers)
        self._dead: set[int] = set()

    def columns(self, view: Any, start_frame: int, end_frame: int) -> dict:
        """Plugin values for a row, plus ``_extra_metrics`` naming them for the catalog."""
        row: dict = {}
        offered: list[Metric] = []
        for provider in self.providers:
            if id(provider) in self._dead:
                continue
            try:
                values = provider.values(view, start_frame, end_frame)
                if not values:
                    continue
                metrics = [m for m in provider.metrics() if m.key in values]
            except Exception:  # noqa: BLE001
                logger.exception("range metric provider %r failed; dropped", provider.name)
                self._dead.add(id(provider))
                continue
            row.update({m.key: values[m.key] for m in metrics})
            offered += metrics
        if offered:
            row["_extra_metrics"] = tuple(offered)
        return row
