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

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QBrush, QIcon, QPainter, QPixmap
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

from glider.analysis.behavior.session_view import SessionView, SessionViewError
from glider.analysis.cohort import discover_sessions, recording_candidates, session_id_for
from glider.analysis.timeline import (
    build_timeline,
    describe_summary,
    describe_value,
    hardware_in_range,
    is_binary,
    lane_role,
)
from glider.gui.review.inspector import Inspector
from glider.gui.review.pool import PoolEntry, SessionPool, ethogram_strip
from glider.gui.review.timeline import TimelinePanel, behavior_order, behavior_qcolor, lane_colour
from glider.gui.review.viewer import KeypointCanvas, Transport
from glider.gui.review.viewport import format_seconds, format_timecode
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
        self._cohort_cache: tuple[tuple, list[dict]] | None = None
        # Recording folder -> its loaded Session (None: nothing GLIDER wrote
        # there). A recording's CSVs are megabytes parsed on the GUI thread.
        self._recordings: dict[Path, object] = {}
        self._zones = None
        self._tour = None
        self._cohort_root: Path | None = None

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
        self._build_cohort_table()
        self._build_bout_stepper()
        self._build_fixes()
        self._export_btn.clicked.connect(self._export_window)
        self._trail_s.valueChanged.connect(self._apply_trail)

        self._restore_layout()
        # Opened with parent=None, so nothing hands it the app theme.
        apply_tool_theme(self)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    # ------------------------------------------------------------------
    # construction

    def _menu_button(self, text: str, items, *, role: str | None = None):
        button = QPushButton(text)
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
        self._export_menu_btn, (self._export_window_action, self._export_heatmap_btn) = (
            self._menu_button(
                "Export",
                [
                    ("Range stats (CSV)…", self._export_window),
                    ("Heatmap (PNG + CSV)…", self._export_heatmap),
                ],
            )
        )
        self._export_window_action.setEnabled(False)
        self._export_heatmap_btn.setEnabled(False)
        self._tour_btn, _ = self._menu_button(
            "?", [("Tutorial", self.start_tour), ("Keyboard shortcuts", self._show_shortcuts)]
        )
        set_button_role(self._tour_btn, "ghost")
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

    def _build_cohort_table(self) -> None:
        # Per-session rows for the same range. The cohort is the unit of
        # analysis, and a per-animal breakdown is what gets exported.
        self._cohort_table = QTableWidget(0, 9)
        self._cohort_table.setHorizontalHeaderLabels(
            [
                "Session",
                "Scored",
                "Distance (cm)",
                "Mean (cm/s)",
                "Freezing (s)",
                "Darting (s)",
                # The cut-offs that produced those two columns, per session, in
                # the unit they were chosen in: the only place the number a
                # methods section has to quote actually exists.
                "Freeze < (cm/s)",
                "Dart > (cm/s)",
                "Top behavior",
            ]
        )
        self._cohort_table.setToolTip(
            "Freeze/Dart are the thresholds this session was scored with, read "
            "from its run.json — not thresholds recomputed from the selection."
        )
        self._cohort_table.verticalHeader().setVisible(False)
        for table in (self._cohort_table, self._bouts, self._zone_table):
            _dress_table(table)
        self._timeline_panel.addTab(self._cohort_table, "Cohort")

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
        self._prev_bout = QPushButton("◀")
        self._prev_bout.setToolTip("Previous bout  ( [ )")
        self._prev_bout.setMaximumWidth(34)
        set_button_role(self._prev_bout, "icon")
        self._prev_bout.clicked.connect(lambda: self._step_bout(-1))
        self._next_bout = QPushButton("▶")
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
            "cohort_table": self._cohort_table,
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
        if path.is_file():
            self.load(path)
            return
        session = self._recording(path)
        if session is not None:
            try:
                view = SessionView.from_recording(session)
            except SessionViewError as e:
                QMessageBox.critical(self, "Open session", str(e))
                return
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
        QMessageBox.critical(
            self,
            "Open session",
            f"{path.name} holds no GLIDER recording (no tracking, events or data CSV) "
            f"and {found}.{more}",
        )

    def load(self, ethogram_csv: Path, *, pose_csv: Path | None = None) -> None:
        """Load a single session, replacing whatever was open."""
        try:
            view = SessionView.load(ethogram_csv, pose_csv=pose_csv)
        except SessionViewError as e:
            QMessageBox.critical(self, "Open session", str(e))
            return
        # A different recording: the range (and its heatmap) belonged to the
        # one that just left. Only switching within a cohort keeps it.
        self._bar.clear_selection()
        self._set_cohort([(Path(ethogram_csv), view)])

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder of sessions")
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, root: Path) -> None:
        """Every ethogram and recording beneath ``root``, as one grouped cohort."""
        root = Path(root)
        sources, warning = discover_sessions(root)
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
            except SessionViewError as e:  # one bad session must not lose the rest
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
        loaded: list[tuple[Path, SessionView]] = []
        failed: list[str] = []
        for path in ethograms:
            try:
                loaded.append((path, SessionView.load(path)))
            except SessionViewError as e:  # one bad file must not lose the rest
                failed.append(f"{path.parent.name}: {e}")
        if not loaded:
            QMessageBox.critical(self, "Open cohort", "\n".join(failed) or "nothing loaded")
            return
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
        self._cohort = list(loaded)
        self._ids = list(ids) if ids is not None else [session_id_for(p) for p, _ in loaded]
        self._groups = list(groups) if groups is not None else [""] * len(loaded)
        self._invalidate_cohort_cache()
        built = [self._timeline_for(path, view) for path, view in self._cohort]
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
        """Put one of the loaded sessions on screen, keeping the selected range."""
        path, view = self._cohort[index]
        selection = self._bar.selection()
        self._shown = index
        self._adopt(path, view, self._timelines[index])
        if selection is not None:
            # The range is the question being asked; switching which animal
            # answers it must not silently reset it.
            self._bar.set_selection(*selection)

    def _adopt(self, path: Path, view: SessionView, timeline) -> None:
        """Show an already-loaded session."""
        self._view = view
        self._ethogram_csv = Path(path)
        self._path_label.setText(_short_path(Path(path)))
        self._path_label.setToolTip(str(path))
        self._bar.set_session(view, timeline)
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

        if view.xy is None:
            self._session_state.set_state("warn", "No poses")
        elif not has_video:
            self._session_state.set_state("ok", "Poses only")
        elif not view.video_is_aligned:
            self._session_state.set_state("warn", "Video may not align")
        else:
            self._session_state.set_state("ok", "Video + poses")
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
                    f"{int(view.frames[-1]) + 1:,}, so frames may not line up."
                )
        self._session_text.setText(
            f"{view.n_rows:,} scored rows at {view.fps:.2f} fps "
            f"({view.duration_s / 60:.1f} min)."
            + found
            + ("" if view.px_per_mm else "  No calibration found: distances unavailable.")
        )
        self._update_status()

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
        self._status.setText("   ·   ".join(parts))
        self._status_path.setText(_short_path(self._ethogram_csv, keep=4))
        self._status_path.setToolTip(str(self._ethogram_csv))

    def _hidden_key(self) -> str:
        sid = self._ids[self._shown] if 0 <= self._shown < len(self._ids) else "session"
        return f"review/hidden/{sid}"

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

        if event.key() in _EDIT_KEYS:
            self._edit_key(event.key(), event.modifiers())
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
        self._play.setText("▶  Play")
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
            self._play.setText("❚❚  Pause")

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
        menu.addAction("Select whole session\t⌘A", self._select_all).setEnabled(
            self._view is not None
        )
        menu.addAction("Set In here\tI", lambda: self._set_in(frame))
        menu.addAction("Set Out here\tO", lambda: self._set_out(frame))
        menu.addAction("Select bout under cursor\tX", lambda: self._select_bout(frame))
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
        self._timeline_menu(frame).exec(global_pos)

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
            "⌘/Ctrl-A  select the whole session\n"
            "Z  zoom to range    ⇧Z  fit the session\n"
            "Esc  clear the range\n"
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
            self._play.setText("❚❚  Pause")

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
    # the selected range

    def _on_selection(self, start: int, end: int) -> None:
        if self._view is None:
            return
        stats = self._view.segment_stats(start, end)
        self._fill_bouts(stats)
        self._fill_movement(stats)
        self._summary.setText(self._describe(stats))
        self._fill_cohort(start, end)
        self._fill_zones(start, end)
        self._fill_hardware(start, end)
        self._apply_heatmap()
        self._export_btn.setEnabled(True)
        self._export_window_action.setEnabled(True)

    def _on_selection_cleared(self) -> None:
        self._inspector.clear_range()
        self._cohort_table.setRowCount(0)
        self._tables.setTabText(self._tables.indexOf(self._cohort_table), "Cohort")
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
        self._cohort_cache = None

    def cohort_rows(self, start: int, end: int) -> list[dict]:
        """The selected window, per loaded session.

        The same frame window is applied to every session rather than a
        per-session fraction: "minutes two to seven" has to mean the same
        stretch in each animal or the comparison is not one.

        Cached on the window and the zones, because that is all it depends
        on. Switching which session is *shown* changes nothing here, and
        recomputing thirty sessions — each a pass over 45,000 frames — to
        redraw a table that did not change made flicking between animals feel
        like the app had hung.
        """
        key = (start, end, id(self._zones), len(self._cohort))
        if self._cohort_cache is not None and self._cohort_cache[0] == key:
            return self._cohort_cache[1]

        rows = []
        for sid, (_path, view) in zip(self._ids, self._cohort, strict=True):
            stats = view.segment_stats(start, end)
            scored = [lab for lab in view.labels if lab]
            top = ""
            if not stats.bouts.empty:
                top = str(stats.bouts.iloc[0]["state"])
            # Freezing and darting are ordinary states of `bouts` now, so
            # they are read from there rather than from a parallel table.
            by_state = stats.bouts.set_index("state") if not stats.bouts.empty else None

            def total(state, table=by_state):
                if table is None or state not in table.index:
                    return 0.0
                return float(table.loc[state, "total_s"])

            rows.append(
                {
                    "session": sid,
                    "scored_rows": len(scored),
                    "duration_s": stats.duration_s,
                    "distance_cm": stats.distance_cm,
                    "mean_cm_s": stats.mean_speed_cm_s,
                    "peak_cm_s": stats.peak_speed_cm_s,
                    "freezing_s": total("freezing"),
                    "darting_s": total("darting"),
                    # Exported alongside the durations they explain: a table of
                    # freezing seconds is not interpretable without the line
                    # that was drawn to produce it.
                    "freeze_threshold_cm_s": view.applied_freeze_cm_s,
                    "dart_threshold_cm_s": view.applied_dart_cm_s,
                    "freeze_threshold_px_frame": view.applied_freeze_px,
                    "dart_threshold_px_frame": view.applied_dart_px,
                    "duration_min": stats.duration_s / 60.0,
                    # What this window alone would give, as the Selected-window
                    # panel reports it -- distinct from the applied thresholds
                    # above, which are what actually produced the labels. The
                    # unit varies per session (cm/s with a pixel scale,
                    # px/frame without), so it travels in its own column rather
                    # than being baked into these names.
                    "window_freeze_threshold": stats.freeze_threshold,
                    "window_dart_threshold": stats.dart_threshold,
                    "window_threshold_unit": stats.threshold_unit,
                    "top_behavior": top,
                    **{
                        f"{state}_s": float(total)
                        for state, total in zip(
                            stats.bouts["state"], stats.bouts["total_s"], strict=True
                        )
                    },
                    **self._zone_columns(start, end, view),
                }
            )
        self._cohort_cache = (key, rows)
        return rows

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
        return out

    @staticmethod
    def _threshold_text(row: dict, side: str) -> str:
        """A cut-off in cm/s, falling back to px/frame, or an em dash.

        An uncalibrated run has real thresholds in pixels; showing nothing
        would claim it had none, and showing a converted number would invent
        the scale it never had.
        """
        real = row.get(f"{side}_threshold_cm_s")
        if real is not None:
            return f"{real:.2f}"
        pixels = row.get(f"{side}_threshold_px_frame")
        return "—" if pixels is None else f"{pixels:.3f} px/f"

    def _fill_cohort(self, start: int, end: int) -> None:
        rows = self.cohort_rows(start, end)
        self._cohort_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                row["session"],
                f"{row['scored_rows']:,}",
                "—" if row["distance_cm"] is None else f"{row['distance_cm']:.1f}",
                "—" if row["mean_cm_s"] is None else f"{row['mean_cm_s']:.2f}",
                f"{row['freezing_s']:.2f}",
                f"{row['darting_s']:.2f}",
                self._threshold_text(row, "freeze"),
                self._threshold_text(row, "dart"),
                row["top_behavior"] or "—",
            ]
            for c, value in enumerate(values):
                # Column 0 is the session name and the last is a behaviour
                # name; everything between them is a measured quantity. No
                # colour chip here: a cohort row's colours come from ITS own
                # session's label set, so a chip would only sometimes match
                # the bar -- decoration wearing the costume of information.
                item = (
                    QTableWidgetItem(value) if c in (0, len(values) - 1) else _measure_item(value)
                )
                self._cohort_table.setItem(r, c, item)
        self._tables.setTabText(1, f"Cohort ({len(rows)})")

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

        start, end = selection
        frame = pd.DataFrame(self.cohort_rows(start, end))
        frame.insert(1, "start_frame", start)
        frame.insert(2, "end_frame", end)
        try:
            frame.to_csv(path, index=False)
        except OSError as e:
            QMessageBox.critical(self, "Export window", f"Could not write {path}: {e}")
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
