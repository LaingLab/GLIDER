"""
AnalysisPanel — in-app surface for inspecting a finished recording.

Loads a recording directory via ``glider.analysis.Session`` and renders
the standard ethology suite (ethogram, trajectory, occupancy, velocity,
zones, bouts) as static matplotlib plots in a tabbed view. A summary
card on the left shows experiment metadata + per-state breakdown so the
operator can confirm "did this run record cleanly" before opening
notebooks for paper-quality analysis.

This panel is *post*-run only — it doesn't touch the live capture
pipeline. Loading is cheap (CSV parsing only); plotting is also cheap
for the recording sizes we expect (minutes-long, single-subject).
Larger recordings (hours / multi-subject) may want notebook workflows
instead; that's by design.

Video sync is a Phase 4 feature; this version is static plots only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from glider.analysis import (
    Session,
    compute_bouts,
    plot_bouts,
    plot_ethogram,
    plot_occupancy_heatmap,
    plot_trajectory,
    plot_velocity,
    plot_zone_dwell,
)
from glider.gui.panels.analysis.plot_widgets import MplCanvas

logger = logging.getLogger(__name__)


class AnalysisPanel(QWidget):
    """Tabbed post-analysis panel for a single recording directory."""

    # Fires after a recording successfully loads — useful for hosts that
    # want to react (status bar updates, window title changes, etc.).
    recording_loaded = pyqtSignal(Path)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._session: Session | None = None
        self._object_id: int = 0
        self._build_ui()
        self._sync_enabled_states()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def session(self) -> Session | None:
        return self._session

    def load_recording(self, directory: Path) -> bool:
        """Load a recording directory. Returns True on success.

        Errors are surfaced as a QMessageBox warning rather than raised;
        the panel stays in its previous (or empty) state on failure.
        """
        try:
            session = Session.load(directory)
        except Exception as e:
            logger.exception("AnalysisPanel: failed to load recording from %s", directory)
            QMessageBox.warning(
                self,
                "Load failed",
                f"Could not load recording from:\n{directory}\n\n{type(e).__name__}: {e}",
            )
            return False

        self._session = session
        self._path_label.setText(str(directory))
        self._path_label.setToolTip(str(directory))
        self._populate_object_combo()
        self._update_summary()
        self._redraw_all_plots()
        self._sync_enabled_states()
        self.recording_loaded.emit(directory)
        return True

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        layout.addLayout(self._build_toolbar())

        # Summary card | tabbed plots, split horizontally so the user
        # can resize either side. QSplitter is the right primitive here
        # rather than fixed widths — a wide screen wants more plot area.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_summary_card())
        splitter.addWidget(self._build_plot_tabs())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 800])
        layout.addWidget(splitter)

    def _build_toolbar(self) -> QHBoxLayout:
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._open_btn = QPushButton("Open Recording…")
        self._open_btn.setToolTip("Choose a directory containing GLIDER CSV outputs")
        self._open_btn.clicked.connect(self._on_open)
        toolbar.addWidget(self._open_btn)

        self._path_label = QLabel("(no recording loaded)")
        self._path_label.setStyleSheet("color: #888;")
        self._path_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(self._path_label, stretch=1)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip("Re-read the current recording from disk")
        self._refresh_btn.clicked.connect(self._on_refresh)
        toolbar.addWidget(self._refresh_btn)

        return toolbar

    def _build_summary_card(self) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setMinimumWidth(200)
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        title = QLabel("<b>Summary</b>")
        title.setStyleSheet("font-size: 13px;")
        v.addWidget(title)

        # Key/value pairs — kept as a dict so update_summary just fills
        # the right label without having to know the layout.
        self._summary_labels: dict[str, QLabel] = {}
        for key in ("Experiment", "Duration", "Frames", "FPS", "Objects"):
            row = QHBoxLayout()
            row.setSpacing(4)
            k_label = QLabel(f"{key}:")
            k_label.setStyleSheet("color: #888;")
            row.addWidget(k_label)
            v_label = QLabel("—")
            v_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            v_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(v_label)
            v.addLayout(row)
            self._summary_labels[key] = v_label

        # Per-state breakdown (multi-line label so it can grow).
        states_title = QLabel("<b>States:</b>")
        states_title.setContentsMargins(0, 6, 0, 0)
        v.addWidget(states_title)
        self._states_label = QLabel("—")
        self._states_label.setStyleSheet("font-family: monospace;")
        self._states_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        v.addWidget(self._states_label)

        v.addStretch()

        obj_title = QLabel("<b>Object:</b>")
        v.addWidget(obj_title)
        self._object_combo = QComboBox()
        self._object_combo.setToolTip("Tracked object whose plots are shown")
        self._object_combo.currentIndexChanged.connect(self._on_object_changed)
        v.addWidget(self._object_combo)

        return card

    def _build_plot_tabs(self) -> QWidget:
        self._plot_tabs = QTabWidget()
        # Each tab owns its own canvas so switching tabs doesn't need a
        # re-render — they all paint once on load.
        self._ethogram_canvas = MplCanvas()
        self._trajectory_canvas = MplCanvas()
        self._occupancy_canvas = MplCanvas()
        self._velocity_canvas = MplCanvas()
        self._zones_canvas = MplCanvas()
        self._bouts_canvas = MplCanvas()

        self._plot_tabs.addTab(self._ethogram_canvas, "Ethogram")
        self._plot_tabs.addTab(self._trajectory_canvas, "Trajectory")
        self._plot_tabs.addTab(self._occupancy_canvas, "Occupancy")
        self._plot_tabs.addTab(self._velocity_canvas, "Velocity")
        self._plot_tabs.addTab(self._zones_canvas, "Zones")
        self._plot_tabs.addTab(self._bouts_canvas, "Bouts")

        return self._plot_tabs

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_open(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Open recording directory",
            "",  # start at last used / cwd
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if not directory:
            return
        self.load_recording(Path(directory))

    def _on_refresh(self) -> None:
        if self._session is None:
            return
        self.load_recording(self._session.directory)

    def _on_object_changed(self) -> None:
        data = self._object_combo.currentData()
        if data is not None:
            self._object_id = int(data)
            self._update_summary()
            self._redraw_all_plots()

    # ------------------------------------------------------------------
    # State sync helpers
    # ------------------------------------------------------------------

    def _sync_enabled_states(self) -> None:
        has_session = self._session is not None
        self._refresh_btn.setEnabled(has_session)
        self._object_combo.setEnabled(has_session)

    def _populate_object_combo(self) -> None:
        """Build the dropdown of tracked object IDs from the loaded session."""
        self._object_combo.blockSignals(True)
        self._object_combo.clear()
        if self._session is None or self._session.tracking.empty:
            self._object_combo.addItem("(none)", userData=None)
        else:
            ids = sorted(self._session.tracking["object_id"].unique())
            for oid in ids:
                self._object_combo.addItem(f"obj {int(oid)}", userData=int(oid))
            self._object_id = int(ids[0])
        self._object_combo.blockSignals(False)

    def _update_summary(self) -> None:
        if self._session is None:
            for label in self._summary_labels.values():
                label.setText("—")
            self._states_label.setText("—")
            return

        s = self._session
        self._summary_labels["Experiment"].setText(s.metadata.get("Experiment", "—"))

        duration = s.flow_duration_s
        self._summary_labels["Duration"].setText(
            f"{duration:.2f} s" if duration is not None else "—"
        )
        self._summary_labels["Frames"].setText(str(len(s.tracking)) if not s.tracking.empty else "0")
        fps = s.frame_rate
        self._summary_labels["FPS"].setText(f"{fps:.1f}" if fps is not None else "—")

        if not s.tracking.empty:
            self._summary_labels["Objects"].setText(str(s.tracking["object_id"].nunique()))
        else:
            self._summary_labels["Objects"].setText("0")

        # State breakdown — percent of post-flow time spent in each state
        # for the selected object. Empty if no tracking or all-NaN states.
        if not s.tracking.empty:
            intervals = s.ethogram(object_id=self._object_id)
            if not intervals.empty:
                durations = intervals.groupby("state")["duration_ms"].sum()
                total = durations.sum()
                if total > 0:
                    lines = [
                        f"  {state}: {100 * dur / total:.0f}%"
                        for state, dur in durations.sort_values(ascending=False).items()
                    ]
                    self._states_label.setText("\n".join(lines))
                else:
                    self._states_label.setText("(no state data)")
            else:
                self._states_label.setText("(no behavioral states)")
        else:
            self._states_label.setText("—")

    # ------------------------------------------------------------------
    # Plot rendering
    # ------------------------------------------------------------------

    def _redraw_all_plots(self) -> None:
        """Render every tab against the current session + object. Each
        canvas owns its own figure so this is straight-line code rather
        than juggling shared state."""
        if self._session is None:
            for canvas in (
                self._ethogram_canvas,
                self._trajectory_canvas,
                self._occupancy_canvas,
                self._velocity_canvas,
                self._zones_canvas,
                self._bouts_canvas,
            ):
                canvas.clear()
            return

        s = self._session
        oid = self._object_id

        # Ethogram
        ax = self._ethogram_canvas.fresh_axes()
        plot_ethogram(s.ethogram(object_id=oid), ax=ax)
        self._ethogram_canvas.redraw()

        # Trajectory
        ax = self._trajectory_canvas.fresh_axes()
        plot_trajectory(s.trajectory(object_id=oid), ax=ax)
        self._trajectory_canvas.redraw()

        # Occupancy heatmap
        ax = self._occupancy_canvas.fresh_axes()
        heatmap, x_edges, y_edges = s.occupancy(object_id=oid, bins=40)
        plot_occupancy_heatmap(heatmap, x_edges, y_edges, ax=ax)
        self._occupancy_canvas.redraw()

        # Velocity
        ax = self._velocity_canvas.fresh_axes()
        plot_velocity(s.velocity(object_id=oid), ax=ax)
        self._velocity_canvas.redraw()

        # Zone dwell
        ax = self._zones_canvas.fresh_axes()
        plot_zone_dwell(s.zone_dwell(object_id=oid), ax=ax)
        self._zones_canvas.redraw()

        # Bouts — pick the dominant state automatically; later we can
        # add a state selector if users want per-state histograms.
        ax = self._bouts_canvas.fresh_axes()
        intervals = s.ethogram(object_id=oid)
        if not intervals.empty:
            durations = intervals.groupby("state")["duration_ms"].sum()
            dominant = str(durations.idxmax())
            bouts = compute_bouts(intervals, state=dominant)
            plot_bouts(bouts, ax=ax, title=f"Bout durations — {dominant}")
        else:
            plot_bouts(pd.Series([], dtype="float64"), ax=ax)
        self._bouts_canvas.redraw()
