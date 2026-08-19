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

from PyQt6.QtCore import QObject, QSettings, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QStackedWidget,
    QTabWidget,
    QWidget,
)

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
            "dock_tabs",
            "2 · Switch panels",
            "Panels share space as tabs. Click a tab along the bottom to switch "
            "between Node Library, Hardware, and Device Control.",
        ),
        TourStep(
            "hardware",
            "3 · Connect Hardware",
            "Add your Arduino or Raspberry Pi here. Nodes like “Output” and "
            "“Input” drive the pins on these boards during a run.",
        ),
        TourStep(
            "canvas",
            "4 · The Canvas",
            "This is your experiment flow. Drop nodes here and connect them "
            "left-to-right to define what happens, and in what order.",
        ),
        TourStep(
            "properties",
            "5 · Node Properties",
            "Select a node on the canvas and its settings appear here — timings, "
            "pins, values. This is where you configure each step.",
        ),
        TourStep(
            "camera",
            "6 · Camera & Vision",
            "Live camera preview, recording, and computer-vision overlays live "
            "here — track your subject and score behavior while an experiment runs.",
        ),
        TourStep(
            "run",
            "7 · Run It",
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


#: QSettings flag for the Behavior Analysis walkthrough. Separate from the
#: main tour's: someone who has built an experiment has not necessarily
#: scored behaviour, and finishing one should not silence the other.
BEHAVIOR_TOUR_COMPLETE_KEY = "first_run/behavior_tour_complete"


def behavior_steps() -> list[TourStep]:
    """The Behavior Analysis pipeline, tab by tab.

    Ordered as the work is actually done — label, fit, check, score. The
    order is the lesson: the tabs sit left to right in that sequence, and
    scoring a cohort with a model nobody has reviewed is the mistake this
    walkthrough exists to prevent.
    """
    return [
        TourStep(
            None,
            "Scoring behavior",
            "Four stages, left to right: label some clips, fit a model, check "
            "what it learned, then score your videos. About a minute to walk "
            "through — Skip or Esc leaves anytime.",
        ),
        TourStep(
            "tab_annotate",
            "1 · Annotate",
            "Pick a folder of videos and their pose CSVs, and GLIDER proposes "
            "short clips spread across the behavior it sees. You label each "
            "one; that is the training data.",
        ),
        TourStep(
            "annotate_resume",
            "Picking up again",
            "Labelling a few hundred clips takes more than one sitting. Resume "
            "reopens the same queue with your finished clips marked done — and "
            "'Render more' inside the annotator adds fresh ones on top.",
        ),
        TourStep(
            "tab_train",
            "2 · Train",
            "Add the same sessions here and fit a classifier. Window length, "
            "class weight and the feature families all live on this tab — they "
            "move the score far more than the model settings do.",
        ),
        TourStep(
            "train_cv",
            "Measure it honestly",
            "Cross-validate splits whole animals across folds, so every score "
            "comes from a recording the model never saw. Tick 'and fit a "
            "model' to get the estimate and the bundle in one pass.",
        ),
        TourStep(
            "tab_review",
            "3 · Review",
            "Every run is saved and opens here: the headline score with its "
            "spread, per-class F1 worst first, and the confusion matrix — "
            "which pairs of behaviors the model actually mixes up.",
        ),
        TourStep(
            "tab_apply",
            "4 · Apply",
            "Point a trained model at recorded video to write the ethogram. "
            "Review the run first: a model scored on one lucky split can look "
            "far better than it is.",
        ),
        TourStep(
            None,
            "That's the loop",
            "Label, fit, review, score — and back to labelling wherever the "
            "confusion matrix says the model is guessing. Replay this anytime "
            "from the Tutorial button.",
        ),
    ]


#: QSettings flag for the Batch Pose Tracking walkthrough.
POSE_BATCH_TOUR_COMPLETE_KEY = "first_run/pose_batch_tour_complete"


def pose_batch_steps() -> list[TourStep]:
    """Batch pose inference, in the order the form has to be filled in.

    Two of these steps exist because the mistake they prevent is silent.
    Bodypart order is baked into every CSV the batch writes, and a wrong
    order does not fail — it mislabels every downstream analysis. Calibration
    is what turns pixels into centimetres, and without it a speed threshold
    means nothing.
    """
    return [
        TourStep(
            None,
            "Tracking a folder of video",
            "This runs a trained pose model over whole directories and writes a "
            "DLC-format CSV beside each video. Fill the left column top to "
            "bottom, then start the run on the right. Skip or Esc leaves anytime.",
        ),
        TourStep(
            "model",
            "1 · The model",
            "Point this at your trained YOLO-pose .pt file. GLIDER reads the "
            "keypoints straight out of the weights, so the bodypart list below "
            "usually fills itself in.",
        ),
        TourStep(
            "bodyparts",
            "Check the bodypart order",
            "These names are written into every CSV this batch produces, in this "
            "order. Nothing downstream can detect a wrong one — it just labels "
            "the tail as the nose forever. 'Edit…' lays them out on a figure.",
        ),
        TourStep(
            "videos",
            "2 · The videos",
            "Drop folders or files here, or use the buttons. 'Include "
            "subdirectories' is on, so pointing at one cohort folder usually "
            "picks up everything under it.",
        ),
        TourStep(
            "calibration",
            "3 · Scale, per video",
            "Every video needs a pixels-to-centimetres scale before the batch "
            "will run. Videos shot on the same rig at the same camera height can "
            "share one — calibrate a video, then 'Copy to Selected'.",
        ),
        TourStep(
            "filter",
            "4 · Smoothing (optional)",
            "Drops low-confidence points, bridges short gaps and median-filters "
            "the rest. The smoothed result becomes the main CSV and the raw "
            "inference is kept beside it as *_raw.csv, so this is reversible.",
        ),
        TourStep(
            "run",
            "5 · Run it",
            "Progress and a log appear on this rail. Overwrite is off by "
            "default, so if a batch is interrupted you can start it again and it "
            "picks up where it stopped instead of redoing finished videos.",
        ),
        TourStep(
            None,
            "Then what?",
            "The CSVs land beside the videos, which is where Behavior Analysis "
            "and Session Review both look for them. Replay this anytime from the "
            "Tutorial button.",
        ),
    ]


#: QSettings flag for the Session Review walkthrough.
SESSION_REVIEW_TOUR_COMPLETE_KEY = "first_run/session_review_tour_complete"


def session_review_steps() -> list[TourStep]:
    """Reading a scored session — and the cohort it belongs to.

    The window's central interaction is dragging a time window on the ethogram,
    and nothing on screen announces it. That step is the reason this
    walkthrough exists.
    """
    return [
        TourStep(
            None,
            "Reading a scored session",
            "Load an ethogram, pick a stretch of time, and read what happened in "
            "it — for one animal or the whole cohort at once. Skip or Esc leaves "
            "anytime.",
        ),
        TourStep(
            "open",
            "1 · Load a session",
            "Opens one scored ethogram along with its poses and video, if they " "sit beside it.",
        ),
        TourStep(
            "open_folder",
            "Or the whole cohort",
            "Loads every ethogram beneath a folder. Worth preferring: the "
            "question is usually what thirty animals did between minutes two and "
            "seven, and answering it a file at a time lets the window drift.",
        ),
        TourStep(
            "canvas",
            "2 · The frame",
            "The tracked keypoints over the video frame. Use it to confirm the "
            "tracking is sound before trusting any number below it.",
        ),
        TourStep(
            "ethogram",
            "3 · Select a window",
            "Every scored frame as a colored band. Click to scrub — and "
            "shift-drag (or right-drag) to select a stretch of time. That "
            "selection is what every number in the panel below is computed over.",
        ),
        TourStep(
            "cohort_table",
            "4 · One row per animal",
            "The same window applied to every loaded session. Freeze and Dart "
            "are the cut-offs each session was actually scored with, in cm/s — "
            "this is the only place the number a methods section quotes exists.",
        ),
        TourStep(
            "zones",
            "Zones (optional)",
            "Load or draw regions of interest and the Zones tab reports time "
            "in zone, entries and latency — for the selected window, across "
            "every loaded session.",
        ),
        TourStep(
            "export",
            "5 · Export",
            "Writes the per-session numbers for the selected window to CSV, so "
            "the table on screen and the table in your analysis are the same "
            "table.",
        ),
        TourStep(
            None,
            "That's the window",
            "Load a cohort, drag a stretch of time, read it or export it. Replay "
            "this anytime from the Tutorial button.",
        ),
    ]


def tour_complete(
    settings: QSettings | None = None,
    key: str = TOUR_COMPLETE_KEY,
) -> bool:
    """Return True once the named walkthrough has been finished or skipped."""
    s = settings if settings is not None else QSettings()
    return bool(s.value(key, False, type=bool))


class Tour(QObject):
    """Drives a :class:`TourOverlay` through a list of :class:`TourStep`."""

    #: Emitted once the walkthrough has resolved -- finished, skipped or
    #: dismissed with Esc, which are all the same thing to anything waiting for
    #: the overlay to go away. Setting the completion flag is not enough on its
    #: own: a follow-on step (the first-run Lab Setup form) has no other way to
    #: learn that the scrim has lifted, and showing itself any earlier would
    #: cover the very widget the spotlight is pointing at.
    finished = pyqtSignal()

    def __init__(
        self,
        host: QWidget,
        steps: list[TourStep] | None = None,
        settings: QSettings | None = None,
        complete_key: str = TOUR_COMPLETE_KEY,
    ):
        super().__init__(host)
        self._host = host
        self._steps = steps if steps is not None else golden_path_steps()
        self._settings = settings if settings is not None else QSettings()
        # Which walkthrough this is, for the "seen it" flag. Each tool keeps its
        # own: finishing the behaviour walkthrough must not mark the main tour
        # done and rob someone of the walkthrough they have never seen.
        self._complete_key = complete_key
        self._index = 0
        self._overlay: TourOverlay | None = None

    def start(self) -> None:
        if not self._steps:
            return
        # One walkthrough per window. A tool window can offer its tour on first
        # open and still have a Tutorial button; pressing it mid-tour would
        # otherwise stack a second overlay on the first, each dimming the other.
        running = getattr(self._host, "_active_tour", None)
        if running is not None and running is not self:
            running._finish()
        self._host._active_tour = self  # type: ignore[attr-defined]

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
        """Make the target visible before we spotlight it.

        Raises tabbed docks, and brings forward any :class:`QTabWidget` page
        the target sits on. Without the tab half, a step pointing at something
        on a tab that is not showing spotlights a widget nobody can see —
        which is most of a tabbed window's tour.
        """
        if isinstance(widget, QDockWidget):
            widget.show()
            widget.raise_()
        self._raise_tabs(widget)
        if widget is not None:
            QApplication.processEvents()

    @staticmethod
    def _raise_tabs(widget: QWidget | None) -> None:
        """Select the tab page holding ``widget``, for every tab widget above it.

        Walks the whole ancestor chain rather than stopping at the first tab
        widget: tabs nest, and a target two levels in needs both switched or
        it stays hidden behind the outer one.
        """
        node = widget
        while node is not None:
            parent = node.parentWidget()
            if isinstance(parent, QStackedWidget):
                tabs = parent.parentWidget()
                if isinstance(tabs, QTabWidget) and tabs.currentWidget() is not node:
                    tabs.setCurrentWidget(node)
            node = parent

    def _show(self) -> None:
        if self._overlay is None:
            return
        step = self._steps[self._index]
        widget = self._resolve(step.target_key)
        self._prepare(widget)
        if self._overlay is None:
            # _prepare pumps the event loop to let the raised tab lay out, and
            # anything delivered in that window can end this tour — another
            # walkthrough starting, or the host closing. Both leave us holding
            # a dismissed overlay.
            return
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
        self._settings.setValue(self._complete_key, True)
        if getattr(self._host, "_active_tour", None) is self:
            self._host._active_tour = None  # type: ignore[attr-defined]
        if self._overlay is not None:
            self._overlay.hide()
            self._overlay.deleteLater()
            self._overlay = None
        self.finished.emit()


def start_tour(
    host: QWidget,
    *,
    steps: list[TourStep] | None = None,
    complete_key: str = TOUR_COMPLETE_KEY,
) -> Tour:
    """Construct and start a tour, keeping a reference alive on the host."""
    # Parented to the host, and start() registers it as the host's active tour,
    # so neither the QObject nor its overlay is collected mid-tour.
    tour = Tour(host, steps=steps, complete_key=complete_key)
    tour.start()
    return tour


def offer_tour_once(
    host: QWidget,
    steps: list[TourStep],
    complete_key: str,
    *,
    settings: QSettings | None = None,
) -> Tour | None:
    """Run ``steps`` the first time ``host`` is shown, and never again.

    The tool windows never pass through the first-launch welcome — they are
    opened on demand, months apart, by someone who has already used the rest of
    the app. So each one offers its own walkthrough the first time it opens.

    The flag is set here rather than when the tour finishes: a walkthrough that
    comes back every time you open the window until you sit through it is a
    nag, and the Tutorial button makes it recoverable either way.
    """
    s = settings if settings is not None else QSettings()
    if tour_complete(s, key=complete_key):
        return None
    s.setValue(complete_key, True)
    # Parented to the host, so Qt keeps it alive; it registers itself as the
    # host's active tour once it actually starts, not before.
    tour = Tour(host, steps=steps, complete_key=complete_key)
    # Deferred: on the first show the layout has not settled, and spotlighting
    # a widget before it has its final geometry outlines the wrong rectangle.
    QTimer.singleShot(0, tour.start)
    return tour
