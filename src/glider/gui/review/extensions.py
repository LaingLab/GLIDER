"""Plugin panels for Session Review.

A plugin adds a panel by declaring an entry point in the ``glider.review_panel``
group whose target is a factory: called with no arguments, it returns one
:class:`ReviewPanel` (or a list of them). For example, in the plugin's
``pyproject.toml``::

    [project.entry-points."glider.review_panel"]
    lift3d = "glider_lift3d.review:panels"

Session Review then

* places each ``"side"`` panel as a column beside the video and each
  ``"timeline"`` panel as a tab next to the ethogram, all hidden until the user
  ticks them in the window's View menu (the choice is remembered);
* while a panel is hidden, calls nothing on it; shown, it gets the current
  session and frame;
* calls :meth:`ReviewPanel.set_session` whenever the shown session changes
  (``None`` when nothing is shown), and :meth:`ReviewPanel.set_frame` on every
  frame change, including during playback, so ``set_frame`` must be cheap.

A panel that raises is logged and switched off for the rest of the window's
life; one plugin's bug must not take the review window down with it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "glider.review_panel"
PLACEMENTS = ("side", "timeline")


@dataclass
class ReviewContext:
    """What a panel is told about the shown session."""

    #: The loaded session (:class:`glider.analysis.behavior.session_view.SessionView`):
    #: poses, keypoint names, fps, video path, scale.
    view: Any
    #: The ethogram CSV or recording folder the session was opened from.
    source: Path
    #: The drawn arena (:class:`glider.vision.arena.ArenaCalibration`), or None.
    arena: Any
    #: First and last frame the timeline shows.
    frame_bounds: tuple[int, int]
    #: Move the whole window (video, timeline, panels) to a frame.
    seek: Callable[[int], None]


@runtime_checkable
class ReviewPanel(Protocol):
    title: str
    placement: str  # "side" or "timeline"

    def widget(self) -> Any: ...

    def set_session(self, context: ReviewContext | None) -> None: ...

    def set_frame(self, frame: int) -> None: ...


def load_review_panels(entry_points=None) -> list[ReviewPanel]:
    """Every panel the installed plugins offer, in entry-point name order.

    ``entry_points`` replaces discovery (tests); it is an iterable of objects
    with ``name`` and ``load()``.
    """
    if entry_points is None:
        from importlib.metadata import entry_points as _eps

        entry_points = _eps(group=ENTRY_POINT_GROUP)
    panels: list[ReviewPanel] = []
    for ep in sorted(entry_points, key=lambda e: e.name):
        try:
            made = ep.load()()
        except Exception:  # noqa: BLE001 - a broken plugin is skipped, not fatal
            logger.exception("review panel plugin %r failed to load", ep.name)
            continue
        for panel in made if isinstance(made, list | tuple) else [made]:
            if not isinstance(panel, ReviewPanel) or panel.placement not in PLACEMENTS:
                logger.warning("review panel plugin %r gave an unusable panel %r", ep.name, panel)
                continue
            panels.append(panel)
    return panels


class PanelHost:
    """Delivers session and frame changes to panels, isolating their failures."""

    def __init__(self, panels: list[ReviewPanel]):
        self.panels = list(panels)
        self._dead: set[int] = set()
        # Panels the user has hidden get nothing; shown again, they are brought
        # up to date with the last session and frame.
        self._inactive: set[int] = set()
        self._context: ReviewContext | None = None
        self._frame: int | None = None

    def set_active(self, panel: ReviewPanel, active: bool) -> None:
        if not active:
            self._inactive.add(id(panel))
            return
        if id(panel) not in self._inactive:
            return
        self._inactive.discard(id(panel))
        self._call(panel, "set_session", self._context)
        if self._frame is not None:
            self._call(panel, "set_frame", self._frame)

    def is_active(self, panel: ReviewPanel) -> bool:
        return id(panel) not in self._inactive and id(panel) not in self._dead

    def _call(self, panel: ReviewPanel, method: str, arg) -> None:
        if id(panel) in self._dead or id(panel) in self._inactive:
            return
        try:
            getattr(panel, method)(arg)
        except Exception:  # noqa: BLE001
            logger.exception("review panel %r failed in %s; switched off", panel.title, method)
            self._dead.add(id(panel))
            try:
                panel.widget().setEnabled(False)
            except Exception:  # noqa: BLE001
                pass

    def set_session(self, context: ReviewContext | None) -> None:
        self._context, self._frame = context, None
        for panel in self.panels:
            self._call(panel, "set_session", context)

    def set_frame(self, frame: int) -> None:
        self._frame = frame
        for panel in self.panels:
            self._call(panel, "set_frame", frame)
