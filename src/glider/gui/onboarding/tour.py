"""Interactive walkthrough controller + the golden-path step list.

:class:`Tour` walks a list of :class:`TourStep`, driving a :class:`TourOverlay`
to spotlight each step's target widget. v1 is passive: the user advances with
Back / Next / Skip (or Esc). Targets are resolved by string key against the
host window's ``tour_targets()`` registry, so this module stays decoupled from
the concrete widget wiring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PyQt6.QtCore import QObject, QSettings
from PyQt6.QtWidgets import QApplication, QDockWidget, QWidget

from glider.gui.onboarding.overlay import TourOverlay

logger = logging.getLogger(__name__)

# QSettings flag, namespaced alongside the existing first_run/* keys.
TOUR_COMPLETE_KEY = "first_run/tour_complete"


@dataclass(frozen=True)
class TourStep:
    """One walkthrough step.

    ``target_key`` names a widget in the host's ``tour_targets()`` registry, or
    ``None`` for a centered step with no spotlight (welcome / finish).
    """

    target_key: str | None
    title: str
    body: str


def golden_path_steps() -> list[TourStep]:
    """The v1 golden path: build a tiny experiment, then run it."""
    return [
        TourStep(
            None,
            "Welcome to GLIDER",
            "Let's build and run a tiny experiment together — about a minute. "
            "You can leave the tour anytime with Skip or Esc.",
        ),
        TourStep(
            "node_library",
            "1 · The Node Library",
            "Experiments are built from blocks. You drag these onto the canvas — "
            "every experiment starts with a “Start Experiment” node.",
        ),
        TourStep(
            "canvas",
            "2 · The Canvas",
            "This is your experiment flow. Drop nodes here and connect them "
            "left-to-right to define what happens, and in what order.",
        ),
        TourStep(
            "hardware",
            "3 · Connect Hardware",
            "Add your Arduino or Raspberry Pi here. Nodes like “Output” and "
            "“Input” drive the pins on these boards during a run.",
        ),
        TourStep(
            "run",
            "4 · Run It",
            "When your flow is ready, press Start to run the experiment. The "
            "status indicator turns green while it's running.",
        ),
        TourStep(
            None,
            "You're set!",
            "That's the core loop: build a flow, connect hardware, run it. "
            "Replay this anytime from Help → Replay Tutorial.",
        ),
    ]


def tour_complete(settings: QSettings | None = None) -> bool:
    """Return True once the walkthrough has been finished or skipped."""
    s = settings if settings is not None else QSettings()
    return bool(s.value(TOUR_COMPLETE_KEY, False, type=bool))


class Tour(QObject):
    """Drives a :class:`TourOverlay` through a list of :class:`TourStep`."""

    def __init__(
        self,
        host: QWidget,
        steps: list[TourStep] | None = None,
        settings: QSettings | None = None,
    ):
        super().__init__(host)
        self._host = host
        self._steps = steps if steps is not None else golden_path_steps()
        self._settings = settings if settings is not None else QSettings()
        self._index = 0
        self._overlay: TourOverlay | None = None

    def start(self) -> None:
        if not self._steps:
            return
        self._overlay = TourOverlay(self._host)
        self._overlay.next_requested.connect(self._next)
        self._overlay.back_requested.connect(self._back)
        self._overlay.skip_requested.connect(self._finish)
        self._index = 0
        self._show()

    # --- Step navigation ---

    def _resolve(self, key: str | None) -> QWidget | None:
        if key is None:
            return None
        targets = self._host.tour_targets()
        return targets.get(key)

    def _prepare(self, widget: QWidget | None) -> None:
        """Make the target visible before we spotlight it (raise tabbed docks)."""
        if isinstance(widget, QDockWidget):
            widget.show()
            widget.raise_()
            QApplication.processEvents()

    def _show(self) -> None:
        if self._overlay is None:
            return
        step = self._steps[self._index]
        widget = self._resolve(step.target_key)
        self._prepare(widget)
        widget = self._resolve(step.target_key)  # geometry may settle after raise
        self._overlay.show_step(widget, step.title, step.body, self._index, len(self._steps))

    def _next(self) -> None:
        if self._index >= len(self._steps) - 1:
            self._finish()
            return
        self._index += 1
        self._show()

    def _back(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._show()

    def _finish(self) -> None:
        self._settings.setValue(TOUR_COMPLETE_KEY, True)
        if self._overlay is not None:
            self._overlay.hide()
            self._overlay.deleteLater()
            self._overlay = None


def start_tour(host: QWidget, *, steps: list[TourStep] | None = None) -> Tour:
    """Construct and start a tour, keeping a reference alive on the host."""
    tour = Tour(host, steps=steps)
    # Hold a reference so the QObject/overlay aren't garbage-collected mid-tour.
    host._active_tour = tour  # type: ignore[attr-defined]
    tour.start()
    return tour
