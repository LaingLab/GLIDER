"""Session Review: a Resolve-style editor for one recording or a whole cohort.

Sessions down the left (the media pool), the video in the middle, what the
selected range contains on the right, and the timeline full width below:
behaviour and hardware as tracks under one playhead. Selecting a range *is* the
analysis -- the inspector fills as soon as the drag ends.

Everything computed lives in Qt-free modules
(:mod:`glider.analysis.behavior.session_view`, :mod:`glider.analysis.timeline`,
:mod:`glider.analysis.cohort`), and the widgets in :mod:`glider.gui.review` only
draw. This module wires them together.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPoint, QSettings, Qt, QTimer
from PyQt6.QtGui import QBrush, QIcon, QKeySequence, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from glider.analysis import markers as mk
from glider.analysis.behavior.session_view import SessionView, SessionViewError
from glider.analysis.cohort import discover_sessions, recording_candidates, session_id_for
from glider.analysis.epochs import (
    CURRENT,
    Epoch,
    metric_catalog,
    threshold_text,
    tidy_frame,
    wide_frame,
)
from glider.analysis.timeline import (
    build_timeline,
    describe_summary,
    describe_value,
    hardware_in_range,
    is_binary,
    lane_role,
)
from glider.gui.review.epoch_table import EpochSession, EpochTable
from glider.gui.review.inspector import Inspector, MarkerRow
from glider.gui.review.marker_editor import COHORT_NEEDS_A_FOLDER, MarkerEditor
from glider.gui.review.pool import PoolEntry, SessionPool, ethogram_strip
from glider.gui.review.timeline import (
    TOP_H,
    TimelinePanel,
    behavior_order,
    behavior_qcolor,
    lane_colour,
    swatch,
)
from glider.gui.review.viewer import KeypointCanvas, Transport
from glider.gui.review.viewport import format_seconds, format_timecode
from glider.gui.styles import colors
from glider.gui.widgets.pastel_glyphs import lucide_icon
from glider.gui.widgets.tool_ui import (
    StatusPill,
    apply_tool_theme,
    caption,
    data_font,
    set_button_role,
    set_text_role,
)

logger = logging.getLogger(__name__)


def _settings() -> QSettings:
    """Where the window keeps its layout. Tests point this at a temp file."""
    return QSettings()


def _short_path(path: Path, keep: int = 3) -> str:
    """The tail of a path, enough to recognise a session without filling the row.

    Apply-run outputs are nested several folders deep under a share, and the
    last few components (cohort / session / file) are what tells one animal
    from another. The full path stays on the tooltip.
    """
    parts = Path(path).parts
    return str(path) if len(parts) <= keep else "…/" + "/".join(parts[-keep:])


def _measure_item(text: str) -> QTableWidgetItem:
    """A measured value: monospace, right-aligned.

    These columns are read downward to compare animals, and centred
    proportional digits put every decimal point in a different place, so the
    column could not be scanned at all. See the type-roles note in
    :mod:`glider.gui.widgets.tool_ui`.
    """
    item = QTableWidgetItem(text)
    item.setFont(data_font())
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def _behavior_item(name: str, order: list[str]) -> QTableWidgetItem:
    """A behaviour name carrying the colour of its own band on the timeline.

    The colour comes from the same :func:`behavior_qcolor` the ethogram bar
    and the annotated video overlay use, so one behaviour is one colour in the
    timeline, in this table, and in the exported MP4. Without the chip the
    table and the bar directly above it share no visual link at all, and
    finding which stripe a row refers to means reading the labels one by one.
    """
    item = QTableWidgetItem(name)
    chip = QPixmap(10, 10)
    chip.fill(Qt.GlobalColor.transparent)
    painter = QPainter(chip)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(behavior_qcolor(name, order)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, 10, 10, 3, 3)
    finally:
        painter.end()
    item.setIcon(QIcon(chip))
    return item


def _recording_or_none(folder: Path):
    """The GLIDER recording in *folder*, or None if there isn't one.

    ``Session.load`` returns an empty session for any readable directory at
    all, so "did it load" answers nothing — "did discovery find an artifact"
    is the question, and an empty frame is how a missing one arrives.
    """
    from glider.analysis import Session

    session = Session.load(folder)
    found = any(not frame.empty for frame in (session.tracking, session.data, session.events))
    return session if found else None


def _dress_table(table: QTableWidget) -> None:
    """Make a results table read as a table rather than a grid in a box.

    The columns used to keep their default 100px width whatever the window
    size, leaving a wide empty strip to the right of the last one and making
    the numbers look like debug output. The first column carries the name and
    takes the slack; the numeric columns size to their content.
    """
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    header.setHighlightSections(False)
    # The stretched name column is left-aligned content, so its heading is too.
    heading = table.horizontalHeaderItem(0)
    if heading is not None:
        heading.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.verticalHeader().setDefaultSectionSize(28)


# Everything a lab's files can make loading raise. GLIDER installs no
# excepthook, so one of these escaping a menu action would end the process --
# and a running experiment with it. (SessionViewError is a ValueError; it is
# named for the reader.)
_UNREADABLE = (SessionViewError, OSError, ValueError, KeyError)


def _select_all_text() -> str:
    """⌘A on macOS, Ctrl+A elsewhere."""
    return QKeySequence(QKeySequence.StandardKey.SelectAll).toString(
        QKeySequence.SequenceFormat.NativeText
    )


_EDIT_KEYS = {
    Qt.Key.Key_I,
    Qt.Key.Key_O,
    Qt.Key.Key_X,
    Qt.Key.Key_Z,
    Qt.Key.Key_Escape,
    Qt.Key.Key_J,
    Qt.Key.Key_K,
    Qt.Key.Key_L,
}


class AnalysisWindow(QMainWindow):
    """Scrub a session, select a range, and read what is in it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session Review")
        self.resize(1440, 900)
        self._view: SessionView | None = None
        # The shown session: an ethogram CSV, or a recording folder.
        self._ethogram_csv: Path | None = None
        self._frame = 0
        self._rate = 1
        self._heatmap_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._cohort: list[tuple[Path, SessionView]] = []
        self._ids: list[str] = []
        self._groups: list[str] = []
        self._timelines: list = []
        self._recordings_of: list = []
        self._shown = -1
        # Rows per range, keyed on the range and the zones (see range_rows).
        self._cohort_cache: dict[tuple, list[dict]] = {}
        # The current range in seconds: what the user selected, kept as is
        # while other animals are shown (see _show_session).
        self._current_span: tuple[float, float] | None = None
        self._switching = False  # a selection set by a session switch, not the user
        # Recording folder -> its loaded Session (None: nothing GLIDER wrote
        # there). A recording's CSVs are megabytes parsed on the GUI thread.
        self._recordings: dict[Path, object] = {}
        self._zones = None
        self._tour = None
        self._cohort_root: Path | None = None
        # Marker files by the folder they sit in, kept for the cohort's
        # lifetime so markers typed while a write was failing survive a
        # session switch. The cohort file exists only for a folder opened as
        # a cohort.
        self._marker_stores: dict[Path, mk.MarkerStore] = {}
        self._cohort_store: mk.MarkerStore | None = None
        self._editor: MarkerEditor | None = None

        central = QWidget()
        central.setObjectName("ToolPage")
        page = QVBoxLayout(central)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)
        page.addWidget(self._build_top_bar())

        self._pool = SessionPool()
        # The tree must not steal J/K/L/arrow-key frame stepping when it has
        # focus after a click -- those keys are the whole point of the window.
        self._pool.tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._pool.session_picked.connect(self._on_session_picked)
        self._canvas = KeypointCanvas()
        self._transport = Transport()
        viewer = QFrame()
        viewer.setObjectName("ReviewPanel")
        viewer_column = QVBoxLayout(viewer)
        viewer_column.setContentsMargins(0, 0, 0, 0)
        viewer_column.setSpacing(0)
        viewer_column.addWidget(self._canvas, 1)
        viewer_column.addWidget(self._transport)
        self._inspector = Inspector()

        self._top_split = QSplitter(Qt.Orientation.Horizontal)
        for widget in (self._pool, viewer, self._inspector):
            self._top_split.addWidget(widget)
        self._top_split.setCollapsible(1, False)
        self._top_split.setStretchFactor(1, 1)
        self._top_split.setSizes([250, 860, 330])

        self._timeline_panel = TimelinePanel()
        self._split = QSplitter(Qt.Orientation.Vertical)
        self._split.addWidget(self._top_split)
        self._split.addWidget(self._timeline_panel)
        self._split.setCollapsible(0, False)
        self._split.setCollapsible(1, False)
        self._split.setSizes([520, 380])
        page.addWidget(self._split, 1)
        self.setCentralWidget(central)
        self._build_status_bar()

        # The names below are what the rest of this module -- and its tests --
        # reach for. They point into the new panels.
        self._bar = self._timeline_panel.view
        self._tables = self._timeline_panel
        self._bouts = self._inspector.bouts
        self._zone_table = self._inspector.zones
        self._summary = self._inspector.range_text
        self._session_text = self._inspector.summary
        self._export_btn = self._inspector.export_btn
        self._trail_s = self._inspector.trail_s
        transport = self._transport
        self._play, self._clock, self._bout_label = transport.play, transport.clock, transport.bout
        self._video_on = transport.video_on
        self._heatmap_on = transport.heatmap_on
        self._trail_on = transport.trail_on

        self._bar.scrubbed.connect(self._set_frame)
        self._bar.selection_changed.connect(self._on_selection)
        self._bar.selection_cleared.connect(self._on_selection_cleared)
        self._bar.context_menu_requested.connect(self._show_timeline_menu)
        self._bar.hidden_changed.connect(self._remember_hidden)
        self._wire_transport()
        self._build_epoch_table()
        self._build_bout_stepper()
        self._build_fixes()
        self._export_btn.clicked.connect(self._export_window)
        self._trail_s.valueChanged.connect(self._apply_trail)
        self._bar.marker_clicked.connect(self._go_to_marker)
        self._bar.marker_activated.connect(self._edit_marker)
        self._bar.marker_changed.connect(self._marker_dragged)
        self._inspector.marker_picked.connect(self._go_to_marker)
        self._inspector.save_marker_btn.clicked.connect(self._add_range_marker)
        self._inspector.export_markers_btn.clicked.connect(self._export_markers)
        self._timeline_panel.marker_btn.clicked.connect(lambda: self._add_point_marker())
        self._timeline_panel.range_marker_btn.clicked.connect(self._add_range_marker)

        self._restore_layout()
        # Opened with parent=None, so nothing hands it the app theme.
        apply_tool_theme(self)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    # ------------------------------------------------------------------
    # construction

    def _menu_button(self, text: str, items, *, role: str | None = None):
        button = QPushButton(text)
        # One size and one arrow for every top-bar menu (tools.qss). Left to
        # the role styles, Open was taller than its neighbours and each arrow
        # sat in its button's bottom-right corner.
        button.setObjectName("ReviewMenu")
        if role is not None:
            set_button_role(button, role)
        menu = QMenu(button)
        actions = [menu.addAction(label, slot) for label, slot in items]
        button.setMenu(menu)
        return button, actions

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("ReviewTopBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 6, 10, 6)
        row.setSpacing(10)
        title = QLabel("Session Review")
        title.setObjectName("ReviewTitle")
        row.addWidget(title)
        self._path_label = QLabel("No session loaded")
        set_text_role(self._path_label, "muted")
        row.addWidget(self._path_label)
        self._session_state = StatusPill("None loaded")
        self._hw_pill = StatusPill("")
        self._hw_pill.setVisible(False)
        self._scale_pill = StatusPill("")
        self._scale_pill.setVisible(False)
        for pill in (self._session_state, self._hw_pill, self._scale_pill):
            row.addWidget(pill)
        row.addStretch(1)
        self._open_btn, _ = self._menu_button(
            "Open",
            [
                ("Open ethogram…", self._open),
                ("Open recording folder…", self._open_recording),
                ("Open folder as cohort…", self._open_folder),
            ],
            role="primary",
        )
        self._zones_btn, _ = self._menu_button(
            "Zones",
            [
                ("Draw zones…", self._draw_zones),
                ("Load zones…", self._load_zones),
                ("Clear zones", self._clear_zones),
            ],
        )
        self._export_menu_btn, export_actions = self._menu_button(
            "Export",
            [
                ("Range stats (CSV)…", self._export_window),
                ("Epoch table, tidy (CSV)…", lambda: self._export_epochs("tidy")),
                ("Epoch table, wide (CSV)…", lambda: self._export_epochs("wide")),
                ("Markers (CSV)…", self._export_markers),
                ("Heatmap (PNG + CSV)…", self._export_heatmap),
            ],
        )
        self._export_window_action = export_actions[0]
        self._export_heatmap_btn = export_actions[-1]
        self._export_window_action.setEnabled(False)
        self._export_heatmap_btn.setEnabled(False)
        self._tour_btn, _ = self._menu_button(
            "?", [("Tutorial", self.start_tour), ("Keyboard shortcuts", self._show_shortcuts)]
        )
        set_button_role(self._tour_btn, "ghost")
        self._tour_btn.setObjectName("ReviewHelp")
        for button in (self._open_btn, self._zones_btn, self._export_menu_btn, self._tour_btn):
            row.addWidget(button)
        return bar

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        bar.setObjectName("ReviewStatus")
        bar.setSizeGripEnabled(False)
        self._status = QLabel("")
        self._status.setFont(data_font(9))
        set_text_role(self._status, "muted")
        self._status_path = QLabel("")
        self._status_path.setFont(data_font(9))
        set_text_role(self._status_path, "muted")
        bar.addWidget(self._status, 1)
        bar.addPermanentWidget(self._status_path)

    def _wire_transport(self) -> None:
        t = self._transport
        t.play.clicked.connect(self._toggle_play)
        t.back.clicked.connect(lambda: self._step_frames(-1))
        t.forward.clicked.connect(lambda: self._step_frames(+1))
        t.to_start.clicked.connect(lambda: self._jump(start=True))
        t.to_end.clicked.connect(lambda: self._jump(start=False))
        t.video_on.setEnabled(False)
        t.video_on.setToolTip(
            "Draw the session's video behind the keypoints. Enabled when a "
            "video for this session can be found."
        )
        t.video_on.toggled.connect(self._canvas.set_show_video)
        t.poses_on.toggled.connect(self._canvas.set_show_poses)
        t.trail_on.toggled.connect(self._apply_trail)
        t.heatmap_on.toggled.connect(self._apply_heatmap)
        t.zones_on.toggled.connect(self._canvas.set_show_zones)
        t.hud_on.toggled.connect(self._canvas.set_show_hud)

    def _build_epoch_table(self) -> None:
        # One row per animal, a column group per epoch. The cohort is the unit
        # of analysis, and this is the table that gets exported.
        self._epoch = EpochTable()
        self._epoch.session_picked.connect(self._pool.select)
        self._epoch.session_opened.connect(self._open_from_epochs)
        self._epoch.export_requested.connect(self._export_epochs)
        for table in (self._bouts, self._zone_table):
            _dress_table(table)
        self._timeline_panel.addTab(self._epoch, "Epoch table")
        self._timeline_panel.currentChanged.connect(self._on_tab_changed)
        self._inspector.epoch_btn.clicked.connect(
            lambda: self._tables.setCurrentWidget(self._epoch)
        )

    def _build_bout_stepper(self) -> None:
        # Jumping between bouts, which is what reviewing an ethogram actually
        # consists of. (Controls unchanged from the old transport strip.)
        slot = self._timeline_panel.bout_slot
        slot.addWidget(caption("Bouts"))
        self._bout_filter = QComboBox()
        self._bout_filter.setToolTip(
            "Which bouts the [ and ] keys step between. 'Any change' stops at "
            "every boundary; a behaviour stops only at the starts of that one."
        )
        self._bout_filter.setMinimumWidth(140)
        slot.addWidget(self._bout_filter)
        self._prev_bout = QPushButton()
        self._prev_bout.setIcon(lucide_icon("chevron-left", colors.TEXT_SECONDARY, 16))
        self._prev_bout.setToolTip("Previous bout  ( [ )")
        self._prev_bout.setMaximumWidth(34)
        set_button_role(self._prev_bout, "icon")
        self._prev_bout.clicked.connect(lambda: self._step_bout(-1))
        self._next_bout = QPushButton()
        self._next_bout.setIcon(lucide_icon("chevron-right", colors.TEXT_SECONDARY, 16))
        self._next_bout.setToolTip("Next bout  ( ] )")
        self._next_bout.setMaximumWidth(34)
        set_button_role(self._next_bout, "icon")
        self._next_bout.clicked.connect(lambda: self._step_bout(+1))
        slot.addWidget(self._prev_bout)
        slot.addWidget(self._next_bout)

    def _build_fixes(self) -> None:
        self._pick_poses = QPushButton("Choose pose CSV…")
        self._pick_poses.clicked.connect(self._choose_pose_csv)
        self._pick_poses.setVisible(False)
        self._fix_resolution = QPushButton("Set arena size from video…")
        self._fix_resolution.clicked.connect(self._resolution_from_video)
        self._fix_resolution.setVisible(False)
        self._inspector.fixes.addWidget(self._pick_poses)
        self._inspector.fixes.addWidget(self._fix_resolution)
        self._inspector.fixes.addStretch(1)

    def _restore_layout(self) -> None:
        settings = _settings()
        for key, splitter in (("review/split", self._split), ("review/top", self._top_split)):
            state = settings.value(key)
            if state is not None:
                splitter.restoreState(state)

    def _save_layout(self) -> None:
        settings = _settings()
        settings.setValue("review/split", self._split.saveState())
        settings.setValue("review/top", self._top_split.saveState())

    # ------------------------------------------------------------------
    # walkthrough

    def tour_targets(self) -> dict[str, QWidget | None]:
        """Widgets the Session Review walkthrough spotlights, by step key."""
        return {
            "open": self._open_btn,
            "open_folder": self._open_btn,
            "canvas": self._canvas,
            "ethogram": self._bar,
            "epoch_table": self._epoch,
            "zones": self._zones_btn,
            "export": self._export_menu_btn,
        }

    def start_tour(self) -> None:
        """Run the Session Review walkthrough (also the Tutorial button)."""
        from glider.gui.onboarding.tour import (
            SESSION_REVIEW_TOUR_COMPLETE_KEY,
            Tour,
            session_review_steps,
        )

        # Held on the instance: the overlay is parented to this window, and a
        # Tour that goes out of scope takes its overlay with it mid-step.
        self._tour = Tour(
            self,
            steps=session_review_steps(),
            complete_key=SESSION_REVIEW_TOUR_COMPLETE_KEY,
        )
        self._tour.start()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Offer the walkthrough the first time this window is opened."""
        super().showEvent(event)
        from glider.gui.onboarding.tour import (
            SESSION_REVIEW_TOUR_COMPLETE_KEY,
            offer_tour_once,
            session_review_steps,
        )

        offer_tour_once(self, session_review_steps(), SESSION_REVIEW_TOUR_COMPLETE_KEY)

    def _refresh_bout_filter(self) -> None:
        """Repopulate the picker for the shown session, keeping the choice."""
        previous = self._bout_filter.currentData()
        self._bout_filter.blockSignals(True)
        self._bout_filter.clear()
        self._bout_filter.addItem("Any change", None)
        for name in self._bar.behavior_order():
            self._bout_filter.addItem(name, name)
        index = self._bout_filter.findData(previous)
        # A behaviour the new session does not contain falls back to "any"
        # rather than silently stepping over nothing.
        self._bout_filter.setCurrentIndex(max(0, index))
        self._bout_filter.blockSignals(False)

    def _step_bout(self, direction: int) -> None:
        """Move the playhead to the next/previous bout start."""
        if self._view is None:
            return
        starts = self._view.bout_starts(self._bout_filter.currentData())
        if starts.size == 0:
            return
        first, last = self._bar.frame_bounds()
        starts = starts[(starts >= first) & (starts <= last)]
        if starts.size == 0:
            return
        if direction > 0:
            later = starts[starts > self._frame]
            target = int(later[0]) if later.size else int(starts[-1])
        else:
            earlier = starts[starts < self._frame]
            target = int(earlier[-1]) if earlier.size else int(starts[0])
        self._set_frame(target)

    def _describe_bout(self) -> str:
        """The bout under the playhead, as text."""
        bout = self._view.bout_at(self._frame) if self._view else None
        if bout is None:
            return "—"
        start, end, label = bout
        fps = self._view.fps or 1.0
        return (
            f"{label or 'unscored'} · {(end - start + 1) / fps:.2f} s · "
            f"{(self._frame - start) / fps:.2f} s in"
        )

    # ------------------------------------------------------------------
    # loading

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ethogram", "", "Ethogram CSV (*.csv);;All Files (*)"
        )
        if path:
            self.load(Path(path))

    def _open_recording(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Recording folder")
        if folder:
            self.open_path(Path(folder))

    def open_path(self, path: Path) -> None:
        """Open an ethogram CSV, or a recording folder."""
        path = Path(path)
        # Reopening must read the disk again: a recording may still be growing.
        self._recordings.clear()
        if path.is_file():
            self.load(path)
            return
        session = self._recording(path)
        problem = None
        if session is not None:
            try:
                view = SessionView.from_recording(session)
            except _UNREADABLE as e:
                # CV off writes no tracking CSV, and such a folder is the source
                # of an offline apply run: its one ethogram is what to open.
                problem = e
            else:
                self._cohort_root = None
                # A different session: the range (and its heatmap) belonged to
                # the one that just left, exactly as a single ethogram does.
                self._bar.clear_selection()
                self._set_cohort([(path, view)])
                return
        ethograms = sorted(path.rglob("ethogram_raw.csv"))
        if len(ethograms) == 1:
            self.load(ethograms[0])
            return
        found = "no ethogram_raw.csv" if not ethograms else f"{len(ethograms)} ethograms"
        more = "  Use Open folder as cohort to load them all." if ethograms else ""
        recording = (
            "holds no GLIDER recording (no tracking, events or data CSV)"
            if problem is None
            else f"holds a GLIDER recording that could not be opened ({problem})"
        )
        QMessageBox.critical(self, "Open session", f"{path.name} {recording} and {found}.{more}")

    def load(self, ethogram_csv: Path, *, pose_csv: Path | None = None) -> None:
        """Load a single session, replacing whatever was open."""
        self._recordings.clear()
        try:
            view = SessionView.load(ethogram_csv, pose_csv=pose_csv)
        except _UNREADABLE as e:
            QMessageBox.critical(self, "Open session", str(e))
            return
        # A different recording: the range (and its heatmap) belonged to the
        # one that just left. Only switching within a cohort keeps it.
        self._bar.clear_selection()
        self._cohort_root = None
        self._set_cohort([(Path(ethogram_csv), view)])

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder of sessions")
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, root: Path) -> None:
        """Every ethogram and recording beneath ``root``, as one grouped cohort."""
        root = Path(root)
        self._recordings.clear()
        try:
            sources, warning = discover_sessions(root)
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Open cohort", f"Could not read {root}: {e}")
            return
        if not sources:
            QMessageBox.warning(
                self,
                "Open cohort",
                f"No sessions found under {root}.\n\nLooked for ethogram_raw.csv files "
                "and folders of GLIDER recording CSVs.",
            )
            return
        loaded, ids, groups, failed = [], [], [], []
        for source in sources:
            try:
                if source.path.is_dir():
                    session = self._recording(source.path)
                    if session is None:
                        raise SessionViewError("not a readable recording")
                    view = SessionView.from_recording(session)
                else:
                    view = SessionView.load(source.path)
            except _UNREADABLE as e:  # one bad session must not lose the rest
                failed.append(f"{source.session_id}: {e}")
                continue
            loaded.append((source.path, view))
            ids.append(source.session_id)
            groups.append(source.group)
        if not loaded:
            QMessageBox.critical(self, "Open cohort", "\n".join(failed) or "nothing loaded")
            return
        self._cohort_root = root
        self._set_cohort(loaded, ids=ids, groups=groups)
        problems = ([warning] if warning else []) + (
            [f"{len(failed)} could not be read:\n" + "\n".join(failed[:8])] if failed else []
        )
        if problems:
            QMessageBox.warning(self, "Open cohort", "\n\n".join(problems))

    def load_many(self, ethograms: list[Path]) -> None:
        """Load a cohort. The first becomes the shown session."""
        self._recordings.clear()
        loaded: list[tuple[Path, SessionView]] = []
        failed: list[str] = []
        for path in ethograms:
            try:
                loaded.append((path, SessionView.load(path)))
            except _UNREADABLE as e:  # one bad file must not lose the rest
                failed.append(f"{path.parent.name}: {e}")
        if not loaded:
            QMessageBox.critical(self, "Open cohort", "\n".join(failed) or "nothing loaded")
            return
        self._cohort_root = None
        self._set_cohort(loaded)
        if failed:
            QMessageBox.warning(
                self,
                "Open cohort",
                f"Loaded {len(loaded)}; {len(failed)} could not be read:\n\n"
                + "\n".join(failed[:8]),
            )

    def _set_cohort(self, loaded, *, ids=None, groups=None) -> None:
        """Make ``loaded`` the cohort and show its first session.

        Timelines are built up front: the sessions panel badges which sessions
        have hardware, and a recording is parsed once per cohort either way.
        """
        # Everything is built before anything is assigned, so a failure part
        # way leaves the cohort on screen whole rather than half replaced.
        loaded = list(loaded)
        ids = list(ids) if ids is not None else [session_id_for(p) for p, _ in loaded]
        groups = list(groups) if groups is not None else [""] * len(loaded)
        built = [self._timeline_for(path, view) for path, view in loaded]
        self._cohort, self._ids, self._groups = loaded, ids, groups
        self._invalidate_cohort_cache()
        self._marker_stores = {}
        self._cohort_store = (
            None
            if self._cohort_root is None
            else self._open_store(self._cohort_root / mk.COHORT_FILE, t0=None)
        )
        self._timelines = [timeline for timeline, _ in built]
        self._recordings_of = [recording for _, recording in built]
        order = behavior_order(label for _, view in self._cohort for label in view.labels)
        self._pool.set_entries(
            [
                PoolEntry(
                    name=sid,
                    group=group,
                    duration_s=view.duration_s,
                    has_video=view.video_path is not None,
                    has_poses=view.pose_path is not None,
                    has_hardware=bool(timeline.lanes),
                    strip=ethogram_strip(view.labels, order),
                )
                for (_path, view), sid, group, timeline in zip(
                    self._cohort, self._ids, self._groups, self._timelines, strict=True
                )
            ]
        )
        self._pool.blockSignals(True)
        self._pool.select(0)
        self._pool.blockSignals(False)
        self._show_session(0)

    def _recording(self, folder: Path):
        """The recording in ``folder`` (cached), or None."""
        folder = Path(folder).resolve()
        if folder not in self._recordings:
            try:
                self._recordings[folder] = _recording_or_none(folder)
            except (OSError, ValueError, KeyError):
                logger.debug("no usable recording in %s", folder, exc_info=True)
                self._recordings[folder] = None
        return self._recordings[folder]

    def _timeline_for(self, path: Path, view: SessionView):
        """``(timeline, recording)`` for a session.

        A recording folder is its own recording, and its behaviour lanes come
        from its tracking (passing ``view`` too would draw them twice). An
        ethogram looks for the recording it was scored from, nearest first.
        """
        path = Path(path)
        if path.is_dir():
            session = self._recording(path)
            if session is not None:
                try:
                    return build_timeline(session, None), session
                except (OSError, ValueError, KeyError):
                    logger.debug("unusable events in %s", path, exc_info=True)
            return build_timeline(None, view), None
        for folder in recording_candidates(path, view.video_path):
            session = self._recording(folder)
            if session is None:
                continue
            try:
                return build_timeline(session, view), session
            except (OSError, ValueError, KeyError):
                logger.debug("unusable events in %s", folder, exc_info=True)
                self._recordings[folder] = None
        return build_timeline(None, view), None

    def _on_session_picked(self, index: int) -> None:
        if 0 <= index < len(self._cohort):
            self._show_session(index)

    def _show_session(self, index: int) -> None:
        """Put one of the loaded sessions on screen, keeping the selected range.

        The range is the question being asked; switching which animal answers
        it must not silently reset it. It is kept in seconds on each session's
        own zero -- the time rule the epoch table uses -- and never re-derived
        from the animal on screen, so the inspector and the table's row for
        this animal describe the same stretch, and flicking between animals
        never moves it.
        """
        path, view = self._cohort[index]
        span = self._current_span
        self._shown = index
        self._adopt(path, view, self._timelines[index])
        self._epoch.set_shown(index)
        if span is None:
            return
        bounds = self._scored_bounds(view)
        frames = (
            None
            if bounds is None
            else mk.frames_in(self._timelines[index], view.fps, *span, bounds)
        )
        self._switching = True
        try:
            if frames is not None:
                self._bar.set_selection(*frames)
            else:
                self._on_selection_cleared()
        finally:
            self._switching = False

    def _adopt(self, path: Path, view: SessionView, timeline) -> None:
        """Show an already-loaded session."""
        self._view = view
        self._ethogram_csv = Path(path)
        self._path_label.setText(_short_path(Path(path)))
        self._path_label.setToolTip(str(path))
        self._bar.set_session(view, timeline)
        self._canvas.set_hud_names(self._bar.behavior_order())
        self._bar.set_hidden(_settings().value(self._hidden_key(), [], type=list) or [])
        self._canvas.set_view(view)
        # The overlay belongs to the session that just left.
        self._canvas.set_heatmap(None)
        self._refresh_heatmap_export_state()
        self._refresh_bout_filter()
        self._apply_trail()
        self._inspector.clear_range()
        self._set_frame(self._bar.frame_bounds()[0])
        # A recording has no pose sidecar to repair -- its poses are the
        # tracked centroid, and its resolution comes from the rig, not a CSV.
        is_recording = Path(path).is_dir()
        self._pick_poses.setVisible(view.xy is None and not is_recording)
        self._fix_resolution.setVisible(
            view.xy is not None and view.resolution is None and not is_recording
        )

        has_video = view.video_path is not None
        self._video_on.setEnabled(has_video)
        self._canvas.set_show_video(has_video and self._video_on.isChecked())

        # A recording's one keypoint is the tracked centre, not a pose.
        poses = "centroid" if view.keypoint_names == ["centroid"] else "poses"
        if view.xy is None:
            self._session_state.set_state("warn", "No poses")
        elif not has_video:
            self._session_state.set_state("ok", f"{poses.capitalize()} only")
        elif not view.video_is_aligned:
            self._session_state.set_state("warn", "Video may not align")
        else:
            self._session_state.set_state("ok", f"Video + {poses}")
        lanes = timeline.lanes if timeline is not None else []
        self._hw_pill.setVisible(bool(lanes))
        self._hw_pill.set_state("ok", f"Hardware · {len(lanes)} lanes")
        self._scale_pill.setVisible(True)
        if view.px_per_mm:
            self._scale_pill.set_state("idle", f"{view.px_per_mm:.2f} px/mm")
        else:
            self._scale_pill.set_state("warn", "Uncalibrated")

        # Body of the old summary text, unchanged, but set on the Session tab:
        found = f"  Poses: {view.pose_path.name}." if view.pose_path else "  No poses found."
        if is_recording:
            found = (
                "  Live recording: labels are the rig's behavioral_state; "
                "position is the tracked centroid." + found
            )
        if has_video:
            found += f"  Video: {view.video_path.name}."
            if not view.video_is_aligned:
                found += (
                    f"  ⚠ It has {view.video_frames:,} frames against the session's "
                    f"{int(view.frames[-1]) - view.first_video_frame + 1:,}, "
                    "so frames may not line up."
                )
        self._session_text.setText(
            f"{view.n_rows:,} scored rows at {view.fps:.2f} fps "
            f"({view.duration_s / 60:.1f} min)."
            + found
            + ("" if view.px_per_mm else "  No calibration found: distances unavailable.")
        )
        self._update_status()
        self._publish_markers()

    def _update_status(self) -> None:
        view = self._view
        if view is None:
            self._status.setText("")
            return
        parts = [f"{view.fps:.2f} fps", f"{view.n_rows:,} rows"]
        if view.pose_path is not None:
            parts.append(f"poses {view.pose_path.name}")
        if view.video_path is not None:
            parts.append(f"video {view.video_path.name}")
        timeline = self._bar.timeline()
        if timeline is not None and timeline.lanes:
            parts.append(f"{len(timeline.lanes)} hardware lanes")
        shown = 0 <= self._shown < len(self._recordings_of)
        recording = self._recordings_of[self._shown] if shown else None
        if recording is not None:
            parts.append(f"events {len(recording.events):,} rows")
            parts.append(f"tracking {len(recording.tracking):,} rows")
        markers = self._shown_markers()
        if markers:
            parts.append(f"{len(markers)} marker{'' if len(markers) == 1 else 's'}")
        self._status.setText("   ·   ".join(parts))
        self._status_path.setText(_short_path(self._ethogram_csv, keep=4))
        self._status_path.setToolTip(str(self._ethogram_csv))

    def _hidden_key(self) -> str:
        """Per session on disk: ids repeat across cohorts (sessions/m01/)."""
        if not 0 <= self._shown < len(self._cohort):
            return "review/hidden/none"
        path = str(Path(self._cohort[self._shown][0]).resolve())
        return f"review/hidden/{hashlib.sha1(path.encode('utf-8')).hexdigest()[:16]}"

    def _remember_hidden(self, keys: list) -> None:
        if self._cohort:
            _settings().setValue(self._hidden_key(), list(keys))

    # ------------------------------------------------------------------
    # playback

    def keyPressEvent(self, event):  # noqa: N802 - Qt override
        """Frame-accurate scrubbing from the keyboard.

        Dragging the ethogram is fast but coarse — on a 45,000-frame session
        one pixel is tens of frames, so a bout boundary cannot be found with
        the mouse at all. Left/Right step exactly one frame; shift steps ten
        and ctrl a second, for covering ground without losing precision.

        ``[`` and ``]`` jump straight to the previous/next bout, which is the
        movement review actually consists of: the frames worth stopping on are
        the boundaries, and stepping to them beats hunting for them.
        """
        if self._view is None:
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._select_all()
            event.accept()
            return

        shortcut = event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        )
        if event.key() in _EDIT_KEYS and not shortcut:
            # Held J/L would otherwise double the rate at the key-repeat rate.
            if not (event.isAutoRepeat() and event.key() in (Qt.Key.Key_J, Qt.Key.Key_L)):
                self._edit_key(event.key(), event.modifiers())
            event.accept()
            return

        if event.key() in (Qt.Key.Key_M, Qt.Key.Key_Up, Qt.Key.Key_Down) and not shortcut:
            if not event.isAutoRepeat() or event.key() != Qt.Key.Key_M:
                if event.key() == Qt.Key.Key_Up:
                    self._step_marker(-1)
                elif event.key() == Qt.Key.Key_Down:
                    self._step_marker(1)
                elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self._add_range_marker()
                else:
                    self._add_point_marker()
            event.accept()
            return

        key = event.key()
        if key not in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Home,
            Qt.Key.Key_End,
            Qt.Key.Key_Space,
            Qt.Key.Key_BracketLeft,
            Qt.Key.Key_BracketRight,
        ):
            super().keyPressEvent(event)
            return

        if key == Qt.Key.Key_Space:
            self._toggle_play()
            event.accept()
            return

        if key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
            self._step_bout(1 if key == Qt.Key.Key_BracketRight else -1)
            event.accept()
            return

        first, last = self._bar.frame_bounds()
        if key == Qt.Key.Key_Home:
            target = first
        elif key == Qt.Key.Key_End:
            target = last
        else:
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                step = max(1, int(round(self._view.fps)))
            elif modifiers & Qt.KeyboardModifier.ShiftModifier:
                step = 10
            else:
                step = 1
            if key == Qt.Key.Key_Left:
                step = -step
            target = self._frame + step

        # Stepping past either end holds there rather than wrapping: a scrub
        # that jumps from the last frame to the first reads as a glitch.
        self._stop()
        self._set_frame(max(first, min(last, target)))
        event.accept()

    def _stop(self) -> None:
        self._timer.stop()
        self._rate = 1
        self._transport.set_playing(False)
        self._transport.rate.setText("1×")

    def _edit_key(self, key, modifiers) -> None:
        if key == Qt.Key.Key_I:
            self._set_in(self._frame)
        elif key == Qt.Key.Key_O:
            self._set_out(self._frame)
        elif key == Qt.Key.Key_X:
            self._select_bout(self._frame)
        elif key == Qt.Key.Key_Z:
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                self._bar.fit()
            else:
                self._bar.zoom_to_selection()
        elif key == Qt.Key.Key_Escape:
            self._bar.clear_selection()
        elif key == Qt.Key.Key_K:
            self._stop()
        else:
            self._shuttle(-1 if key == Qt.Key.Key_J else 1)

    def _set_in(self, frame: int) -> None:
        _first, last = self._bar.frame_bounds()
        selection = self._bar.selection()
        out = selection[1] if selection is not None and selection[1] >= frame else last
        self._bar.set_selection(frame, out)

    def _set_out(self, frame: int) -> None:
        first, _last = self._bar.frame_bounds()
        selection = self._bar.selection()
        start = selection[0] if selection is not None and selection[0] <= frame else first
        self._bar.set_selection(start, frame)

    def _select_bout(self, frame: int) -> None:
        bout = self._view.bout_at(frame) if self._view else None
        if bout is not None:
            self._bar.set_selection(bout[0], bout[1])

    def _shuttle(self, direction: int) -> None:
        """J and L: play backward or forward, doubling on each repeat, up to 8x."""
        if self._view is None:
            return
        if self._timer.isActive() and (self._rate > 0) == (direction > 0):
            self._rate = max(-8, min(8, self._rate * 2))
        else:
            self._rate = direction
        self._transport.rate.setText(f"{self._rate}×")
        if not self._timer.isActive():
            self._timer.start(int(1000 / max(1.0, self._view.fps)))
            self._transport.set_playing(True)

    def _step_frames(self, step: int) -> None:
        if self._view is None:
            return
        first, last = self._bar.frame_bounds()
        self._stop()
        self._set_frame(max(first, min(last, self._frame + step)))

    def _jump(self, *, start: bool) -> None:
        if self._view is None:
            return
        first, last = self._bar.frame_bounds()
        self._stop()
        self._set_frame(first if start else last)

    def _range_title(self) -> str:
        selection = self._bar.selection()
        if selection is None or self._view is None:
            return "No range"
        start, end = selection
        t_in = self._bar.seconds_of_axis(self._bar.axis_of_frame(start))
        t_out = self._bar.seconds_of_axis(self._bar.axis_of_frame(end + 1))
        return f"Range · {format_seconds(t_in)} → {format_seconds(t_out)} · {t_out - t_in:.1f} s"

    def _timeline_menu(self, frame: int) -> QMenu:
        """What right-clicking the lanes offers. Built separately so tests can read it."""
        has_range = self._bar.selection() is not None
        menu = QMenu(self)
        menu.addAction(self._range_title()).setEnabled(False)
        menu.addSeparator()
        menu.addAction(f"Select whole session\t{_select_all_text()}", self._select_all).setEnabled(
            self._view is not None
        )
        menu.addAction("Set In here\tI", lambda: self._set_in(frame))
        menu.addAction("Set Out here\tO", lambda: self._set_out(frame))
        menu.addAction("Select bout under cursor\tX", lambda: self._select_bout(frame))
        writable = self._marker_problem() is None
        menu.addSeparator()
        menu.addAction("Save range as marker…\t⇧M", self._add_range_marker).setEnabled(
            has_range and writable
        )
        menu.addAction("Add marker here\tM", lambda: self._add_point_marker(frame)).setEnabled(
            writable
        )
        menu.addSeparator()
        for text, slot in (
            ("Zoom to range\tZ", self._bar.zoom_to_selection),
            ("Loop range", self._loop_range),
            ("Export range stats…", self._export_window),
            ("Copy range timecode", self._copy_range),
        ):
            menu.addAction(text, slot).setEnabled(has_range)
        menu.addSeparator()
        menu.addAction("Clear range\tEsc", self._bar.clear_selection).setEnabled(has_range)
        return menu

    def _show_timeline_menu(self, global_pos, frame: int) -> None:
        menu = self._timeline_menu(frame)
        menu.exec(global_pos)
        menu.deleteLater()  # parented to the window: one would pile up per right-click

    def _loop_range(self) -> None:
        selection = self._bar.selection()
        if selection is None:
            return
        self._timeline_panel.loop.setChecked(True)
        self._set_frame(selection[0])
        if not self._timer.isActive():
            self._toggle_play()

    def _copy_range(self) -> None:
        selection = self._bar.selection()
        if selection is None or self._view is None:
            return
        fps = self._view.fps or 30.0
        t_in = self._bar.seconds_of_axis(self._bar.axis_of_frame(selection[0]))
        t_out = self._bar.seconds_of_axis(self._bar.axis_of_frame(selection[1] + 1))
        QApplication.clipboard().setText(
            f"{format_timecode(t_in, fps)} - {format_timecode(t_out, fps)}"
        )

    def _show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            "Keyboard shortcuts",
            "Space  play / pause\n"
            "J  K  L  shuttle back / stop / forward (repeat for 2×, 4×, 8×)\n"
            "← →  one frame   ⇧ ten   ⌘/Ctrl one second\n"
            "Home  End  session start / end\n"
            "[  ]  previous / next bout\n"
            "I  O  set In / Out at the playhead\n"
            "X  select the bout under the playhead\n"
            f"{_select_all_text()}  select the whole session\n"
            "Z  zoom to range    ⇧Z  fit the session\n"
            "Esc  clear the range\n"
            "M  point marker at the playhead    ⇧M  range marker from In/Out\n"
            "↑  ↓  previous / next marker; double-click a marker to edit it\n"
            "⌘/Ctrl-scroll or pinch  zoom    scroll  pan\n"
            "Drag the lanes to select; drag the ruler to scrub.",
        )

    def _set_frame(self, frame: int) -> None:
        self._frame = int(frame)
        self._bar.set_frame(self._frame, follow=self._timer.isActive())
        self._canvas.set_frame(self._frame)
        self._canvas.set_hud(self._hud_behavior(), self._hud_chips())
        self._bout_label.setText(self._describe_bout())
        if self._view and self._view.fps:
            seconds = self._bar.seconds_of_axis(self._bar.axis_of_frame(self._frame))
            self._clock.setText(format_timecode(seconds, self._view.fps))
            first, last = self._bar.frame_bounds()
            self._transport.position.setText(f"{self._frame - first:,} / {last - first:,}")

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._stop()
        elif self._view is not None:
            # Wall-clock, not frame-locked: a smooth approximate rate reads
            # better in review than a stuttering exact one.
            self._rate = 1
            self._timer.start(int(1000 / max(1.0, self._view.fps)))
            self._transport.set_playing(True)

    def _advance(self) -> None:
        if self._view is None:
            return
        lo, hi = self._bar.frame_bounds()
        selection = self._bar.selection()
        looping = self._timeline_panel.loop.isChecked() and selection is not None
        if looping:
            lo, hi = selection
        target = self._frame + self._rate
        if lo <= target <= hi:
            self._set_frame(target)
        elif looping:
            self._set_frame(lo if self._rate > 0 else hi)
        else:
            # At 8x the last step overshoots; land on the bound, not short of it.
            self._set_frame(max(lo, min(hi, target)))
            self._stop()

    def _hud_behavior(self):
        bout = self._view.bout_at(self._frame) if self._view else None
        if bout is None or not bout[2]:
            return None
        start, _end, label = bout
        seconds_in = (self._frame - start) / (self._view.fps or 30.0)
        return f"{label}  {seconds_in:.2f} s in", behavior_qcolor(label, self._bar.behavior_order())

    def _hud_chips(self) -> list:
        """Every output the rig is driving on this frame. Inputs are not a HUD's job."""
        timeline = self._bar.timeline()
        if timeline is None or not self._bar.uses_ms():
            return []
        ms = self._bar.axis_of_frame(self._frame)
        chips = []
        for lane in timeline.lanes:
            if lane_role(lane) == "input":
                continue
            text, active = describe_value(lane, ms)
            if active:
                label = lane.label if is_binary(lane) else f"{lane.label} {text}"
                chips.append((label, lane_colour(lane)))
        return chips

    def _choose_pose_csv(self) -> None:
        """Point the session at its poses when discovery could not."""
        if self._ethogram_csv is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pose CSV for this session",
            str(self._ethogram_csv.parent),
            "Pose CSV (*.csv);;All Files (*)",
        )
        if path:
            self.load(self._ethogram_csv, pose_csv=Path(path))

    def _resolution_from_video(self) -> None:
        """Recover the arena size from the source video and keep it."""
        if self._view is None or self._view.pose_path is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Video this session was tracked from",
            str(self._view.source.parent if self._view.source else ""),
            "Video (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not path:
            return
        from glider.vision.pose.dlc import backfill_resolution
        from glider.vision.video_source import video_resolution

        resolution = video_resolution(path)
        if resolution is None:
            QMessageBox.warning(
                self, "Arena size", f"Could not read a frame size from {Path(path).name}."
            )
            return
        if not backfill_resolution(self._view.pose_path, resolution):
            QMessageBox.warning(
                self,
                "Arena size",
                "Read the video, but could not update the pose sidecar "
                f"({self._view.pose_path.name}). Is it writable?",
            )
            return
        self.load(self._view.source)

    def _refresh_heatmap_export_state(
        self, grid_tuple: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> None:
        """Record the grid the overlay was drawn from, and gate the export.

        Enabled only when the canvas actually drew: set_heatmap refuses a grid
        whose peak is <= 0, so the checkbox being on is not enough.
        """
        self._heatmap_grid = grid_tuple
        self._export_heatmap_btn.setEnabled(grid_tuple is not None and self._canvas.has_heatmap())

    def _apply_heatmap(self, *_args) -> None:
        """Bin the selected window, or clear the overlay."""
        selection = self._bar.selection()
        if not self._heatmap_on.isChecked() or self._view is None or selection is None:
            self._canvas.set_heatmap(None)
            self._refresh_heatmap_export_state()
            return
        from glider.analysis.behavior import spatial

        try:
            grid, x_edges, y_edges = spatial.occupancy_grid(
                self._view, bins=60, start_frame=selection[0], end_frame=selection[1]
            )
        except spatial.SpatialError as e:
            logger.info("no heatmap for this session: %s", e)
            self._canvas.set_heatmap(None)
            self._refresh_heatmap_export_state()
            return
        self._canvas.set_heatmap(grid)
        self._refresh_heatmap_export_state((grid, x_edges, y_edges))

    def _apply_trail(self, *_args) -> None:
        self._canvas.set_trail(self._trail_s.value(), self._trail_on.isChecked())

    def _select_all(self) -> None:
        if self._view is not None and self._view.n_rows:
            self._bar.set_selection(int(self._view.frames[0]), int(self._view.frames[-1]))

    # ------------------------------------------------------------------
    # markers

    def _open_store(self, path: Path, *, t0: str | None) -> mk.MarkerStore:
        store = mk.MarkerStore(path, t0=t0)
        if store.error is not None:
            # Said once, when the file is first met; the store stays read-only.
            QMessageBox.warning(
                self,
                "Markers",
                f"{store.error}\n\nIts markers cannot be edited, and the file "
                "will not be overwritten.",
            )
        return store

    def _session_folder(self, index: int) -> Path:
        path = Path(self._cohort[index][0])
        return path if path.is_dir() else path.parent

    def _session_store(self, index: int | None = None) -> mk.MarkerStore | None:
        """The shown (or given) session's marker file, opened once per cohort."""
        index = self._shown if index is None else index
        if not 0 <= index < len(self._cohort):
            return None
        folder = self._session_folder(index).resolve()
        store = self._marker_stores.get(folder)
        if store is None:
            timeline = self._timeline_at(index)
            store = self._open_store(folder / mk.SESSION_FILE, t0=mk.t0_of(timeline))
            self._marker_stores[folder] = store
            if store.t0_changed:
                self.statusBar().showMessage(
                    f"{mk.SESSION_FILE} was saved when this session's time zero was "
                    "different, so its markers may be shifted.",
                    12000,
                )
        return store

    def _shown_markers(self) -> list:
        """This session's markers and the cohort's, as the timeline draws them."""
        store = self._session_store()
        own = list(store.markers) if store is not None else []
        shared = list(self._cohort_store.markers) if self._cohort_store is not None else []
        return own + shared

    def _store_for(self, scope: str) -> mk.MarkerStore | None:
        return self._cohort_store if scope == "cohort" else self._session_store()

    def _marker_problem(self) -> str | None:
        """Why this session's markers cannot be added to, or None."""
        store = self._session_store()
        if store is None:
            return "Load a session to add markers."
        if store.error is not None:
            return f"{store.path.name} could not be read, so its markers are read-only."
        return None

    def _publish_markers(self) -> None:
        """Hand the markers to everything that shows them."""
        markers = self._shown_markers()
        self._bar.set_markers(markers)
        rows = [
            MarkerRow(
                id=m.id,
                kind=m.kind,
                name=m.name,
                colour=swatch(m.color),
                scope=m.scope,
                time=(
                    f"{format_seconds(m.start_s)} – {format_seconds(m.end_s)}"
                    if m.is_range
                    else format_seconds(m.start_s)
                ),
                duration=f"{m.duration_s:.1f} s" if m.is_range else "",
                note=m.note,
            )
            for m in sorted(markers, key=lambda m: (not m.is_range, m.start_s))
        ]
        self._inspector.set_markers(rows)
        problem = self._marker_problem()
        self._inspector.set_marker_note(problem or "")
        panel = self._timeline_panel
        for button, tip in (
            (panel.marker_btn, "Add a point marker at the playhead  (M)"),
            (panel.range_marker_btn, "Save the In/Out range as a range marker  (⇧M)"),
        ):
            button.setEnabled(problem is None)
            button.setToolTip(problem or tip)
        self._inspector.save_marker_btn.setEnabled(
            problem is None and self._bar.selection() is not None
        )
        self._refresh_inside()
        self._update_status()
        self._fill_epochs()

    def _refresh_inside(self) -> None:
        """The range markers the selection falls inside, on the range card."""
        if self._bar.selection() is None or self._current_span is None:
            self._inspector.set_inside([])
            return
        start_s, end_s = self._current_span
        eps = 1e-6
        self._inspector.set_inside(
            [
                m.name or "Range"
                for m in self._shown_markers()
                if m.is_range and m.start_s - eps <= start_s and end_s <= m.end_s + eps
            ]
        )

    def _add_point_marker(self, frame: int | None = None) -> None:
        """M: a point marker at the playhead (or where the menu opened), then its editor."""
        problem = self._marker_problem()
        if problem is not None or self._view is None:
            self.statusBar().showMessage(problem or "", 8000)
            return
        marker = mk.Marker(
            "point", self._bar.seconds_of_frame(self._frame if frame is None else frame)
        )
        if self._save_marker(marker):
            self._edit_marker(marker.id)

    def _add_range_marker(self) -> None:
        """⇧M: the In/Out range as a range marker, then its editor."""
        problem = self._marker_problem()
        selection = self._bar.selection()
        if problem is not None or selection is None:
            self.statusBar().showMessage(
                problem or "Set In and Out first (I and O), or drag across the lanes.", 8000
            )
            return
        marker = mk.Marker("range", *(self._current_span or self._span_seconds(*selection)))
        if self._save_marker(marker):
            self._edit_marker(marker.id)

    def _save_marker(self, marker, *, previous_scope: str | None = None) -> bool:
        """Keep ``marker`` in its scope's file, moving it if its scope changed.

        A failed write keeps the markers in memory -- nothing typed is lost --
        and says which file and why.
        """
        store = self._store_for(marker.scope)
        if store is None or not store.writable:
            self.statusBar().showMessage("That marker file cannot be written.", 8000)
            return False
        touched = [store]
        if previous_scope is not None and previous_scope != marker.scope:
            old = self._store_for(previous_scope)
            if old is not None and old.writable:
                old.remove(marker.id)
                touched.append(old)
        store.put(marker)
        for each in touched:
            try:
                each.save()
            except (OSError, mk.MarkerFileError) as e:
                QMessageBox.critical(
                    self,
                    "Markers",
                    f"Could not write {each.path}: {e}\n\n"
                    "The markers are kept in this window until it closes.",
                )
        self._publish_markers()
        return True

    def _find_marker(self, marker_id: str):
        return next((m for m in self._shown_markers() if m.id == marker_id), None)

    def _edit_marker(self, marker_id: str, at: QPoint | None = None) -> None:
        """Open the editor on a marker, anchored under it."""
        marker = self._find_marker(marker_id)
        if marker is None:
            return
        store = self._store_for(marker.scope)
        if store is None or not store.writable:
            self.statusBar().showMessage(
                f"{marker.name or 'This marker'} is in a file that cannot be written.", 8000
            )
            return
        cohort = self._cohort_store
        if cohort is None:
            cohort_problem = COHORT_NEEDS_A_FOLDER
        elif not cohort.writable:
            cohort_problem = f"{cohort.path.name} could not be read, so it cannot be added to."
        else:
            cohort_problem = None
        editor = MarkerEditor(
            marker,
            cohort_problem=cohort_problem,
            in_out=self._current_span if self._bar.selection() is not None else None,
            parent=self,
        )
        editor.done.connect(
            lambda edited, scope=marker.scope: self._save_marker(edited, previous_scope=scope)
        )
        editor.deleted.connect(self._delete_marker)
        self._editor = editor
        if at is None:
            x = self._bar.x_of_axis(self._bar.axis_of_seconds(marker.start_s))
            at = self._bar.mapToGlobal(QPoint(int(x), TOP_H))
        editor.popup(at)

    def _delete_marker(self, marker_id: str) -> None:
        marker = self._find_marker(marker_id)
        store = None if marker is None else self._store_for(marker.scope)
        if store is None or not store.writable:
            return
        store.remove(marker_id)
        try:
            store.save()
        except (OSError, mk.MarkerFileError) as e:
            QMessageBox.critical(self, "Markers", f"Could not write {store.path}: {e}")
        self._publish_markers()

    def _marker_dragged(self, marker_id: str, start_s: float, end_s) -> None:
        """Keep where a dragged marker landed, or put it back if it cannot be kept."""
        marker = self._find_marker(marker_id)
        if marker is None:
            return
        if not self._save_marker(replace(marker, start_s=start_s, end_s=end_s)):
            self._publish_markers()

    def _go_to_marker(self, marker_id: str) -> None:
        """Seek to a marker; a range is selected too, so its numbers fill."""
        marker = self._find_marker(marker_id)
        if marker is None or self._view is None:
            return
        timeline, fps = self._bar.timeline(), self._bar.fps()
        first, last = self._bar.frame_bounds()
        self._stop()
        self._set_frame(max(first, min(last, mk.frame_at(timeline, fps, marker.start_s))))
        if marker.is_range:
            bounds = self._scored_bounds(self._view)
            frames = (
                None
                if bounds is None
                else mk.frames_in(timeline, fps, marker.start_s, marker.end_s, bounds)
            )
            if frames is not None:
                self._bar.set_selection(*frames)

    def _step_marker(self, direction: int) -> None:
        """Up / down: the playhead to the previous / next marker start."""
        if self._view is None:
            return
        timeline, fps = self._bar.timeline(), self._bar.fps()
        starts = sorted({mk.frame_at(timeline, fps, m.start_s) for m in self._shown_markers()})
        ahead = [f for f in starts if (f > self._frame if direction > 0 else f < self._frame)]
        if not ahead:
            return
        first, last = self._bar.frame_bounds()
        self._stop()
        self._set_frame(max(first, min(last, ahead[0] if direction > 0 else ahead[-1])))

    def _export_markers(self) -> None:
        """Every loaded session's markers and the cohort's, as one CSV."""
        if not self._cohort:
            return
        import pandas as pd

        def record(m, session: str) -> dict:
            return {
                "session": session,
                "scope": m.scope,
                "kind": m.kind,
                "name": m.name,
                "color": m.color,
                "start_s": m.start_s,
                "end_s": m.end_s,
                "duration_s": m.duration_s if m.is_range else None,
                "note": m.note,
                "id": m.id,
            }

        records = [
            record(m, "") for m in (self._cohort_store.markers if self._cohort_store else [])
        ]
        for i, sid in enumerate(self._ids):
            store = self._session_store(i)
            records += [record(m, sid) for m in (store.markers if store else [])]
        default = (self._cohort_root or self._session_folder(self._shown)) / "markers.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export markers", str(default), "CSV Files (*.csv)"
        )
        if not path:
            return
        columns = [
            "session",
            "scope",
            "kind",
            "name",
            "color",
            "start_s",
            "end_s",
            "duration_s",
            "note",
            "id",
        ]
        try:
            pd.DataFrame.from_records(records, columns=columns).to_csv(path, index=False)
        except OSError as e:
            QMessageBox.critical(self, "Export markers", f"Could not write {path}: {e}")
            return
        self.statusBar().showMessage(f"Wrote {path}", 10000)

    # ------------------------------------------------------------------
    # the selected range

    def _on_selection(self, start: int, end: int) -> None:
        if self._view is None:
            return
        if not self._switching:
            self._current_span = self._span_seconds(start, end)
        stats = self._view.segment_stats(start, end)
        self._fill_bouts(stats)
        self._fill_movement(stats)
        self._summary.setText(self._describe(stats))
        self._fill_epochs()
        self._fill_zones(start, end)
        self._fill_hardware(start, end)
        self._apply_heatmap()
        self._export_btn.setEnabled(True)
        self._export_window_action.setEnabled(True)
        self._inspector.save_marker_btn.setEnabled(self._marker_problem() is None)
        self._refresh_inside()

    def _on_tab_changed(self, _index: int) -> None:
        """Fill the Epoch table when it is shown; hidden, a drag never pays for it."""
        self._fill_epochs()

    def _on_selection_cleared(self) -> None:
        if not self._switching:
            self._current_span = None
        self._inspector.clear_range()
        self._epoch.clear()
        self._tables.setTabText(self._tables.indexOf(self._epoch), "Epoch table")
        self._fill_epochs()  # the cohort's ranges stay, less the current one
        self._export_btn.setEnabled(False)
        self._export_window_action.setEnabled(False)
        self._apply_heatmap()

    def _fill_movement(self, stats) -> None:
        fps = self._view.fps or 30.0
        t_in = self._bar.seconds_of_axis(self._bar.axis_of_frame(stats.start_frame))
        t_out = self._bar.seconds_of_axis(self._bar.axis_of_frame(stats.end_frame + 1))
        frames = stats.end_frame - stats.start_frame + 1
        self._inspector.set_range(
            f"IN  {format_timecode(t_in, fps)}    OUT  {format_timecode(t_out, fps)}",
            f"{stats.duration_s:.2f} s · {frames:,} frames",
        )
        if stats.distance_cm is None:
            self._inspector.set_kpis("—", "—", "—")
        else:
            self._inspector.set_kpis(
                f"{stats.distance_cm:.1f}",
                f"{stats.mean_speed_cm_s:.2f}",
                f"{stats.peak_speed_cm_s:.2f}",
            )

    def _fill_hardware(self, start: int, end: int) -> None:
        timeline = self._bar.timeline()
        if timeline is None or not timeline.lanes or not self._bar.uses_ms():
            self._inspector.set_hardware([])
            return
        by_key = {lane.key: lane for lane in timeline.lanes}
        summaries = hardware_in_range(
            timeline.lanes, self._bar.axis_of_frame(start), self._bar.axis_of_frame(end + 1)
        )
        self._inspector.set_hardware(
            [(s.label, describe_summary(s), lane_colour(by_key[s.key])) for s in summaries]
        )

    def _draw_zones(self) -> None:
        """Open the zone editor on the frame currently on screen.

        The same editor the live rig uses, seeded with a still instead of a
        camera — zones drawn against the arena the animal was actually in
        beat zones drawn from memory against a blank canvas.
        """
        frame = self._canvas.current_frame()
        if frame is None:
            QMessageBox.warning(
                self,
                "Draw zones",
                "Zones are drawn on a video frame, and no video was found for "
                "this session.\n\nUse “Load zones…” to bring in a "
                "configuration drawn elsewhere.",
            )
            return

        from glider.gui.dialogs.zone_dialog import ZoneDialog
        from glider.vision.zones import ZoneConfiguration

        config = self._zones if self._zones is not None else ZoneConfiguration()
        dialog = ZoneDialog(None, config, parent=self, frame=frame)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            self._zones = dialog.get_zone_configuration()
        finally:
            dialog.deleteLater()
        self._adopt_zones()

    def _adopt_zones(self) -> None:
        """Show the current zones and re-report the selected range."""
        names = [z.name for z in getattr(self._zones, "zones", [])]
        self._canvas.set_zones(self._zones)
        self._invalidate_cohort_cache()
        self.statusBar().showMessage(f"Zones: {', '.join(names) or '(none defined)'}", 8000)
        selection = self._bar.selection()
        if selection is not None:
            self._on_selection(*selection)
        self._fill_epochs()

    def _clear_zones(self) -> None:
        self._zones = None
        self._adopt_zones()

    def _load_zones(self) -> None:
        """Load a zone configuration and re-report the current window."""
        from glider.analysis.behavior.spatial import SpatialError, load_zones

        path, _ = QFileDialog.getOpenFileName(
            self, "Zone configuration", "", "Zone files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            self._zones = load_zones(path)
        except SpatialError as e:
            QMessageBox.critical(self, "Load zones", str(e))
            return
        self._adopt_zones()

    def zone_rows(self, start: int, end: int, view=None):
        """Per-zone occupancy for a window, or an empty frame without zones."""
        import pandas as pd

        from glider.analysis.behavior.spatial import SpatialError, zone_occupancy

        target = view if view is not None else self._view
        if self._zones is None or target is None:
            return pd.DataFrame()
        try:
            return zone_occupancy(target, self._zones, start_frame=start, end_frame=end)
        except SpatialError as e:
            logger.info("no zone occupancy for this session: %s", e)
            return pd.DataFrame()

    def _fill_zones(self, start: int, end: int) -> None:
        rows = self.zone_rows(start, end)
        self._zone_table.setRowCount(len(rows))
        for r, (_, row) in enumerate(rows.iterrows()):
            latency = row["latency_s"]
            values = [
                str(row["zone"]),
                f"{row['total_s']:.2f}",
                f"{100 * row['fraction']:.1f}%",
                str(int(row["n_entries"])),
                f"{row['mean_bout_s']:.2f}",
                "never" if latency != latency else f"{latency:.2f}",
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value) if c == 0 else _measure_item(value)
                self._zone_table.setItem(r, c, item)
        # The inspector's tables are Fixed-height and must track their rows.
        self._zone_table.setMaximumHeight(self._zone_table.sizeHint().height())
        self._inspector.set_zone_count(len(rows))

    def _invalidate_cohort_cache(self) -> None:
        self._cohort_cache = {}

    def _timeline_at(self, index: int):
        return self._timelines[index] if 0 <= index < len(self._timelines) else None

    @staticmethod
    def _scored_bounds(view) -> tuple[int, int] | None:
        """The frames a session has rows for: what any range is clipped to."""
        if view is None or not view.n_rows:
            return None
        return int(view.frames[0]), int(view.frames[-1])

    def _span_seconds(self, start: int, end: int) -> tuple[float, float]:
        """The shown session's frames ``[start, end]`` as ``[start_s, end_s)``.

        Read from the timeline widget's own session, so it is right even in
        the middle of a switch, before the window's indices have moved on.
        """
        timeline, fps = self._bar.timeline(), self._bar.fps()
        return mk.seconds_at(timeline, fps, start), mk.seconds_at(timeline, fps, end + 1)

    def cohort_rows(self, start: int, end: int) -> list[dict]:
        """The shown session's frames ``[start, end]``, as a range, per session."""
        if self._view is None:
            return []
        return self.range_rows(*self._span_seconds(start, end))

    def range_rows(self, start_s: float, end_s: float) -> list[dict]:
        """``[start_s, end_s)`` on each session's own zero, one row per session.

        The time rule (:mod:`glider.analysis.markers`): seconds from flow start
        where a session has one, from its first video frame otherwise, turned
        into *that* session's frames. "Minutes two to seven" then means the
        same stretch of each animal's protocol whatever its frame rate or
        however long its rig ran before flow start. A range past a session's
        end is clipped to it; one wholly outside gives a row that says so.

        Cached on the range and the zones: switching which session is shown
        changes nothing here, and recomputing thirty sessions -- each a pass
        over 45,000 frames -- to redraw a table that did not change made
        flicking between animals feel like the app had hung.
        """
        key = (round(float(start_s), 6), round(float(end_s), 6), id(self._zones), len(self._cohort))
        cached = self._cohort_cache.get(key)
        if cached is not None:
            return cached
        rows = [self._session_row(i, start_s, end_s) for i in range(len(self._cohort))]
        # An animal that never showed a behaviour in the range showed it for
        # 0 s: a measurement, not a gap.
        states = sorted({s for row in rows for s in row.get("_states", ())})
        for row in rows:
            if not row.get("outside"):
                for state in states:
                    row.setdefault(f"{state}_s", 0.0)
        # ponytail: cleared wholesale past 64 ranges; an LRU if dragging
        # with the epoch table open ever shows up in a profile.
        if len(self._cohort_cache) >= 64:
            self._cohort_cache = {}
        self._cohort_cache[key] = rows
        return rows

    def _session_row(self, index: int, start_s: float, end_s: float) -> dict:
        """One session's numbers over ``[start_s, end_s)`` on its own zero."""
        _path, view = self._cohort[index]
        timeline = self._timeline_at(index)
        base = {
            "session": self._ids[index],
            "group": self._groups[index] if index < len(self._groups) else "",
            "t0": mk.t0_of(timeline),
        }
        bounds = self._scored_bounds(view)
        frames = (
            None if bounds is None else mk.frames_in(timeline, view.fps, start_s, end_s, bounds)
        )
        if frames is None:
            return {**base, "outside": True, "start_s": start_s, "end_s": end_s}
        start, end = frames
        stats = view.segment_stats(start, end)
        scored = [lab for lab in view.labels if lab]
        top = ""
        if not stats.bouts.empty:
            top = str(stats.bouts.iloc[0]["state"])
        # Freezing and darting are ordinary states of `bouts` now, so they are
        # read from there rather than from a parallel table.
        by_state = stats.bouts.set_index("state") if not stats.bouts.empty else None

        def total(state, table=by_state):
            if table is None or state not in table.index:
                return 0.0
            return float(table.loc[state, "total_s"])

        return {
            **base,
            "start_frame": start,
            "end_frame": end,
            "start_s": mk.seconds_at(timeline, view.fps, start),
            "end_s": mk.seconds_at(timeline, view.fps, end + 1),
            "scored_rows": len(scored),
            "duration_s": stats.duration_s,
            "distance_cm": stats.distance_cm,
            "mean_cm_s": stats.mean_speed_cm_s,
            "peak_cm_s": stats.peak_speed_cm_s,
            "freezing_s": total("freezing"),
            "darting_s": total("darting"),
            # Exported alongside the durations they explain: a table of
            # freezing seconds is not interpretable without the line that was
            # drawn to produce it.
            "freeze_threshold_cm_s": view.applied_freeze_cm_s,
            "dart_threshold_cm_s": view.applied_dart_cm_s,
            "freeze_threshold_px_frame": view.applied_freeze_px,
            "dart_threshold_px_frame": view.applied_dart_px,
            "duration_min": stats.duration_s / 60.0,
            # What this window alone would give, as the Selected-window panel
            # reports it -- distinct from the applied thresholds above, which
            # are what actually produced the labels. The unit varies per
            # session (cm/s with a pixel scale, px/frame without), so it
            # travels in its own column rather than being baked into these
            # names.
            "window_freeze_threshold": stats.freeze_threshold,
            "window_dart_threshold": stats.dart_threshold,
            "window_threshold_unit": stats.threshold_unit,
            "top_behavior": top,
            "_states": tuple(str(s) for s in stats.bouts["state"] if s),
            **{
                f"{state}_s": float(seconds)
                for state, seconds in zip(stats.bouts["state"], stats.bouts["total_s"], strict=True)
            },
            **self._zone_columns(start, end, view),
            **self._hardware_columns(timeline, view.fps, start, end),
        }

    @staticmethod
    def _hardware_columns(timeline, fps: float, start: int, end: int) -> dict:
        """Seconds on per device, flattened onto a session row; 0 for an idle one."""
        if not mk.uses_ms(timeline) or not timeline.lanes:
            return {}
        start_ms = mk.seconds_at(timeline, fps, start) * 1000.0
        end_ms = mk.seconds_at(timeline, fps, end + 1) * 1000.0
        on = {s.key: s.on_ms for s in hardware_in_range(timeline.lanes, start_ms, end_ms)}
        names = [lane.label.replace(" ", "_") for lane in timeline.lanes]
        columns = {
            f"hw_{name}_on_s": on.get(lane.key, 0.0) / 1000.0
            for name, lane in zip(names, timeline.lanes, strict=True)
        }
        return {**columns, "_devices": tuple(names)}

    def _zone_columns(self, start: int, end: int, view) -> dict:
        """Time, fraction and entries per zone, flattened onto a session row.

        Flat columns rather than a nested table because this is what gets
        exported and pasted into a statistics package: one row per animal,
        one column per measure.
        """
        zones = self.zone_rows(start, end, view)
        if zones.empty:
            return {}
        out: dict[str, float] = {}
        for _, row in zones.iterrows():
            zone = str(row["zone"]).replace(" ", "_")
            out[f"zone_{zone}_s"] = float(row["total_s"])
            out[f"zone_{zone}_frac"] = float(row["fraction"])
            out[f"zone_{zone}_entries"] = int(row["n_entries"])
            out[f"zone_{zone}_latency_s"] = float(row["latency_s"])
        out["_zones"] = tuple(str(z).replace(" ", "_") for z in zones["zone"])
        return out

    @staticmethod
    def _threshold_text(row: dict, side: str) -> str:
        """A cut-off in cm/s, falling back to px/frame (see :func:`epochs.threshold_text`)."""
        return threshold_text(row, side)

    def _export_window(self) -> None:
        """Write the per-session window table to CSV."""
        selection = self._bar.selection()
        if selection is None or not self._cohort:
            return
        default = self._cohort[0][0].parent.parent / "window_summary.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export window summary", str(default), "CSV Files (*.csv)"
        )
        if not path:
            return
        import pandas as pd

        frame = pd.DataFrame(
            self.range_rows(*(self._current_span or self._span_seconds(*selection)))
        )
        # Tuples for the window's own use, and the outside flag (an outside
        # session's empty cells already say it), are not measurements.
        frame = frame[[c for c in frame.columns if not str(c).startswith("_") and c != "outside"]]
        try:
            frame.to_csv(path, index=False)
        except OSError as e:
            QMessageBox.critical(self, "Export window", f"Could not write {path}: {e}")
            return
        self.statusBar().showMessage(f"Wrote {path}", 10000)

    def _epochs(self) -> list[Epoch]:
        """The cohort's range markers in time order, then the current range.

        Current range is added whenever `_current_span` is set, not only when
        the shown animal has a selection: a switch to an animal the range
        doesn't reach clears only that animal's *selection* (see
        `_show_session`), and gating on the selection would make the whole
        column vanish while viewing that animal, instead of showing it as an
        outside row.
        """
        store = self._cohort_store
        ranges = sorted(
            (m for m in (store.markers if store is not None else []) if m.is_range),
            key=lambda m: (m.start_s, m.end_s),
        )
        # Seeded with "Current range" so a user's own marker of that name is
        # the one renumbered -- the actual Current range epoch below always
        # keeps the literal name, to agree with tidy_frame's own uniquing.
        seen: Counter = Counter({"Current range": 1})
        epochs = []
        for m in ranges:
            base = m.name or "Range"
            seen[base] += 1
            name = base if seen[base] == 1 else f"{base} ({seen[base]})"
            epochs.append(Epoch(m.id, name, m.start_s, m.end_s, m.color))
        if self._current_span is not None:
            epochs.append(Epoch(CURRENT, "Current range", *self._current_span))
        return epochs

    def _epoch_sessions(self) -> list[EpochSession]:
        return [
            EpochSession(i, sid, group)
            for i, (sid, group) in enumerate(zip(self._ids, self._groups, strict=True))
        ]

    def _fill_epochs(self) -> None:
        """Compute and show the Epoch table -- only while it is on screen."""
        if not self._cohort or self._tables.currentWidget() is not self._epoch:
            return
        epochs = self._epochs()
        rows = {e.key: self.range_rows(e.start_s, e.end_s) for e in epochs}
        catalog = metric_catalog(r for block in rows.values() for r in block)
        self._epoch.set_data(epochs, self._epoch_sessions(), rows, catalog)
        self._epoch.set_shown(self._shown)
        count = self._epoch.session_count()
        self._tables.setTabText(
            self._tables.indexOf(self._epoch), f"Epoch table ({count})" if count else "Epoch table"
        )

    def _open_from_epochs(self, index: int) -> None:
        """Double-click: that animal, on the timeline, to see why a number looks odd."""
        self._pool.select(index)
        self._tables.setCurrentIndex(0)

    def _export_epochs(self, shape: str) -> None:
        """The epoch table as on screen -- its epochs and metrics -- tidy or wide."""
        if not self._cohort:
            return
        epochs = self._epoch.visible(self._epochs())
        rows = {e.key: self.range_rows(e.start_s, e.end_s) for e in epochs}
        catalog = metric_catalog(r for block in rows.values() for r in block)
        metrics = self._epoch.metric_keys(catalog)
        if not epochs or not metrics:
            QMessageBox.information(
                self,
                "Export epoch table",
                "Nothing to export yet: save a range marker for the whole cohort, "
                "or select a range.",
            )
            return
        tidy = tidy_frame(epochs, rows, metrics)
        frame = tidy if shape == "tidy" else wide_frame(tidy, metrics)
        root = self._cohort_root or self._session_folder(max(0, self._shown))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export epoch table",
            str(root / f"epoch_table_{shape}.csv"),
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            frame.to_csv(path, index=False)
        except OSError as e:
            QMessageBox.critical(self, "Export epoch table", f"Could not write {path}: {e}")
            return
        self.statusBar().showMessage(f"Wrote {path}", 10000)

    def _export_heatmap(self) -> None:
        """Write the heatmap on screen as a PNG figure plus a CSV of its grid."""
        selection = self._bar.selection()
        if self._heatmap_grid is None or selection is None or self._ethogram_csv is None:
            return
        start, end = selection
        # Named from the session on screen, not the cohort root: _export_window
        # uses self._cohort[0][0], which is the *first* session and the wrong
        # answer whenever a later one is displayed.
        session_dir = (
            self._ethogram_csv if self._ethogram_csv.is_dir() else self._ethogram_csv.parent
        )
        default = session_dir / f"{session_dir.name}_heatmap_{start}-{end}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export heatmap", str(default), "PNG Files (*.png)"
        )
        if not path:
            return

        from glider.analysis.behavior.spatial import write_occupancy_export

        grid, x_edges, y_edges = self._heatmap_grid
        try:
            png_path, csv_path = write_occupancy_export(
                grid,
                x_edges,
                y_edges,
                Path(path),
                title=f"{session_dir.name}  frames {start}-{end}",
            )
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Export heatmap", f"Could not write {path}: {e}")
            return
        wrote = str(csv_path) if png_path is None else f"{png_path} and {csv_path.name}"
        note = "" if png_path is not None else " (no figure: matplotlib is not installed)"
        self.statusBar().showMessage(f"Wrote {wrote}{note}", 10000)

    def _describe(self, stats) -> str:
        span = (
            f"{stats.start_frame}–{stats.end_frame}  "
            f"({stats.duration_s / 60:.2f} min, {stats.duration_s:.1f} s)"
        )
        if stats.distance_cm is None:
            movement = "distance unavailable (no calibration for this session)"
        else:
            movement = (
                f"{stats.distance_cm:.1f} cm travelled, "
                f"mean {stats.mean_speed_cm_s:.2f} cm/s, peak {stats.peak_speed_cm_s:.2f} cm/s"
            )
        if stats.freeze_threshold is None:
            thresholds = "thresholds unavailable (no poses)"
        else:
            thresholds = (
                f"this window alone would give freeze {stats.freeze_threshold:.3f} / "
                f"dart {stats.dart_threshold:.3f} {stats.threshold_unit} "
                "— shown for comparison; the loaded labels are unchanged"
            )
        return f"Frames {span}\n{movement}\n{thresholds}"

    def _fill_bouts(self, stats) -> None:
        rows = stats.bouts
        self._bouts.setRowCount(len(rows))
        # The bar's own pooled order, not one re-derived from the ethogram
        # alone -- so a row's chip really is that row's stripe. With a
        # tracking lane in the timeline the two orders differ, and the table
        # and the bar directly above it then disagreed on every colour.
        order = self._bar.behavior_order()
        for r, (_, row) in enumerate(rows.iterrows()):
            values = [
                str(row["state"] or "(unscored)"),
                str(int(row["n_bouts"])),
                f"{row['total_s']:.2f}",
                f"{100 * row['fraction']:.1f}%",
                f"{row['mean_s']:.2f}",
                f"{row['median_s']:.2f}",
            ]
            for c, value in enumerate(values):
                item = _behavior_item(value, order) if c == 0 else _measure_item(value)
                self._bouts.setItem(r, c, item)
        # The inspector's tables are Fixed-height and must track their rows.
        self._bouts.setMaximumHeight(self._bouts.sizeHint().height())

    def closeEvent(self, event):  # noqa: N802 - Qt override
        self._save_layout()
        self._timer.stop()
        self._canvas._close_reader()
        super().closeEvent(event)


__all__ = [
    "AnalysisWindow",
    "KeypointCanvas",
    "behavior_order",
    "behavior_qcolor",
]
