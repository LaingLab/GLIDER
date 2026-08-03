"""Scrub an analysed session, select a window, and ask what is in it.

Built around two things the outputs already contain but nothing yet showed.

The ethogram *is* the timeline. Rendering it as a coloured bar makes the
session's structure legible at a glance and doubles as the scrubber, so
picking a window and seeing what is in it are the same gesture.

The poses stand in for the video. Annotated video is expensive and usually not
kept; the pose CSV is small and almost always survives, so a session can be
replayed as moving keypoints long after the pixels are gone.

Everything computed here lives in
:mod:`glider.analysis.behavior.session_view`, which is Qt-free and tested
without a display. This module only draws.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from glider.analysis.behavior.session_view import SessionView, SessionViewError
from glider.gui.styles import colors

logger = logging.getLogger(__name__)

_BAR_HEIGHT = 46
_TRAIL_DEFAULT_S = 5.0


def behavior_qcolor(name: str) -> QColor:
    """The colour the annotated video would have drawn this behaviour in.

    Shared with the overlay so a bout looks the same wherever it is shown;
    blank (unscored) frames read as background rather than a colour.
    """
    if not name:
        return QColor(colors.BORDER)
    from glider.analysis.behavior.classify.overlay import color_for_behavior

    b, g, r = color_for_behavior(name)
    return QColor(r, g, b)


class EthogramBar(QWidget):
    """The ethogram as a timeline: bands to read, and the scrubber to drag.

    Clicking or dragging with the left button scrubs; dragging with shift (or
    the right button) selects a window. Selection and playhead are separate so
    a chosen window survives scrubbing around inside it.
    """

    scrubbed = pyqtSignal(int)  # frame
    selection_changed = pyqtSignal(int, int)  # start, end frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(_BAR_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._view: SessionView | None = None
        self._frame = 0
        self._selection: tuple[int, int] | None = None
        self._drag_anchor: int | None = None

    def set_view(self, view: SessionView | None) -> None:
        self._view = view
        self._frame = 0
        self._selection = None
        self.update()

    def set_frame(self, frame: int) -> None:
        self._frame = int(frame)
        self.update()

    def selection(self) -> tuple[int, int] | None:
        return self._selection

    def set_selection(self, start: int, end: int) -> None:
        self._selection = (int(min(start, end)), int(max(start, end)))
        self.update()
        self.selection_changed.emit(*self._selection)

    # ------------------------------------------------------------------

    def _total_frames(self) -> int:
        if self._view is None or self._view.n_rows == 0:
            return 0
        return int(self._view.frames[-1]) + 1

    def _frame_at(self, x: float) -> int:
        total = self._total_frames()
        if total == 0 or self.width() <= 0:
            return 0
        return max(0, min(total - 1, int(x / self.width() * total)))

    def _x_of(self, frame: int) -> float:
        total = self._total_frames()
        return 0.0 if total == 0 else frame / total * self.width()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(colors.BASE))
        total = self._total_frames()
        if self._view is None or total == 0:
            painter.setPen(QPen(QColor(colors.TEXT_MUTED)))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a session to see its ethogram"
            )
            return

        # Freezing/darting is a second, independent scoring of the same
        # frames, so it gets its own lane rather than overwriting the posture
        # it co-occurs with — a frame can be both grooming and freezing.
        speed = self._view.speed_labels if self._view.has_speed_axis else []
        posture_height = self.height() * (0.68 if speed else 1.0)

        self._paint_lane(painter, self._view.labels, 0.0, posture_height)
        if speed:
            self._paint_lane(painter, speed, posture_height, self.height() - posture_height)

        if self._selection is not None:
            start, end = self._selection
            x0, x1 = self._x_of(start), self._x_of(end + 1)
            painter.fillRect(
                QRectF(x0, 0, max(1.0, x1 - x0), self.height()),
                QBrush(colors.qcolor_with_alpha(QColor(colors.ACCENT), 0.28)),
            )
            painter.setPen(QPen(QColor(colors.ACCENT), 2))
            painter.drawLine(QPointF(x0, 0), QPointF(x0, self.height()))
            painter.drawLine(QPointF(x1, 0), QPointF(x1, self.height()))

        painter.setPen(QPen(QColor(colors.TEXT_PRIMARY), 2))
        x = self._x_of(self._frame)
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))

    def _paint_lane(self, painter, labels, top: float, height: float) -> None:
        """One run-length band per label run, across the full width."""
        if height <= 0 or not labels:
            return
        frames = self._view.frames
        run_start, run_label = 0, labels[0]
        for i in range(1, len(labels) + 1):
            if i < len(labels) and labels[i] == run_label:
                continue
            x0 = self._x_of(int(frames[run_start]))
            x1 = self._x_of(int(frames[i - 1]) + 1)
            painter.fillRect(QRectF(x0, top, max(1.0, x1 - x0), height), behavior_qcolor(run_label))
            if i < len(labels):
                run_start, run_label = i, labels[i]

    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if self._view is None:
            return
        frame = self._frame_at(event.position().x())
        selecting = (
            event.button() == Qt.MouseButton.RightButton
            or event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        )
        if selecting:
            self._drag_anchor = frame
            self.set_selection(frame, frame)
        else:
            self._drag_anchor = None
            self.scrubbed.emit(frame)

    def mouseMoveEvent(self, event):
        if self._view is None:
            return
        frame = self._frame_at(event.position().x())
        if self._drag_anchor is not None:
            self.set_selection(self._drag_anchor, frame)
        elif event.buttons() & Qt.MouseButton.LeftButton:
            self.scrubbed.emit(frame)

    def mouseReleaseEvent(self, _event):
        self._drag_anchor = None


class KeypointCanvas(QWidget):
    """The animal drawn from its poses, with a trailing centroid track.

    Coordinates are pixels in the source video, so the canvas needs that
    video's resolution to place them. Without it the arena's true extent is
    unknown, and stretching the points to fit their own range would silently
    redraw the enclosure as whatever the animal happened to visit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view: SessionView | None = None
        self._frame = 0
        self._trail_s = _TRAIL_DEFAULT_S
        self._show_trail = True
        self._show_video = True
        self._zones = None
        self._heatmap = None
        self._reader = None  # VideoFileSource, opened lazily
        self._cached: tuple[int, QImage] | None = None

    def set_view(self, view: SessionView | None) -> None:
        self._close_reader()
        self._view = view
        self._frame = 0
        self.update()

    # ------------------------------------------------------------------
    # video
    # ------------------------------------------------------------------

    def set_show_video(self, enabled: bool) -> None:
        self._show_video = bool(enabled)
        self.update()

    def set_zones(self, zones) -> None:
        """Outline a zone configuration over the arena."""
        self._zones = zones
        self.update()

    def set_heatmap(self, grid) -> None:
        """Show (or clear, with None) an occupancy histogram over the arena.

        ``grid`` is the ``(nx, ny)`` array ``compute_occupancy`` returns, in
        the same pixel space the arena is drawn in.
        """
        self._heatmap = None
        if grid is None or not getattr(grid, "size", 0) or not np.isfinite(grid).any():
            self.update()
            return
        peak = float(grid.max())
        if peak <= 0:
            self.update()
            return
        # Normalised to its own peak, so a short window is still readable;
        # this is a picture of where time went, not an absolute count.
        normalised = np.clip(grid / peak, 0.0, 1.0)
        nx, ny = normalised.shape
        rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        accent = QColor(colors.ACCENT)
        rgba[..., 0] = accent.red()
        rgba[..., 1] = accent.green()
        rgba[..., 2] = accent.blue()
        # Transposed because histogram2d's first axis is x and an image's is y.
        alpha = (np.sqrt(normalised.T) * 210).astype(np.uint8)
        alpha[normalised.T <= 0] = 0  # never-visited cells stay clear
        rgba[..., 3] = alpha
        self._heatmap = QImage(rgba.tobytes(), nx, ny, 4 * nx, QImage.Format.Format_RGBA8888).copy()
        self.update()

    def has_video(self) -> bool:
        return self._view is not None and self._view.video_path is not None

    def _close_reader(self) -> None:
        if self._reader is not None:
            self._reader.release()
            self._reader = None
        self._cached = None

    def _frame_image(self, index: int) -> QImage | None:
        """The video frame for *index*, as a QImage, or None.

        Decoded on demand and cached by index: a repaint from resizing or a
        selection change must not cost another decode, and scrubbing one frame
        at a time is a sequential read rather than a seek.
        """
        if not self._show_video or self._view is None or self._view.video_path is None:
            return None
        if self._cached is not None and self._cached[0] == index:
            return self._cached[1]
        if self._reader is None:
            from glider.vision.video_source import VideoFileSource

            reader = VideoFileSource()
            if not reader.load(self._view.video_path):
                logger.info("could not open %s for playback", self._view.video_path)
                self._view.video_path = None  # stop retrying every repaint
                return None
            self._reader = reader
        frame = self._reader.read_frame(index)
        if frame is None:
            return None
        # cv2 gives BGR; copy because the QImage must own its buffer once the
        # numpy array goes out of scope.
        height, width = frame.shape[:2]
        image = QImage(frame.data, width, height, 3 * width, QImage.Format.Format_BGR888).copy()
        self._cached = (index, image)
        return image

    def set_frame(self, frame: int) -> None:
        self._frame = int(frame)
        self.update()

    def set_trail(self, seconds: float, enabled: bool) -> None:
        self._trail_s, self._show_trail = float(seconds), bool(enabled)
        self.update()

    def _transform(self):
        """Scale and offset mapping video pixels onto the widget, or None."""
        if self._view is None or not self._view.resolution:
            return None
        width, height = self._view.resolution
        scale = min(self.width() / width, self.height() / height)
        return scale, (self.width() - width * scale) / 2, (self.height() - height * scale) / 2

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(colors.CANVAS))

        transform = self._transform()
        if (
            self._view is None
            or transform is None
            or (self._view.xy is None and not self.has_video())
        ):
            painter.setPen(QPen(QColor(colors.TEXT_MUTED)))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._why_blank(),
            )
            return

        scale, dx, dy = transform
        width, height = self._view.resolution
        arena = QRectF(dx, dy, width * scale, height * scale)

        image = self._frame_image(self._frame)
        if image is not None:
            painter.drawImage(arena, image)
        if self._heatmap is not None:
            painter.drawImage(arena, self._heatmap)
        painter.setPen(QPen(QColor(colors.BORDER), 1))
        painter.drawRect(arena)
        self._paint_zones(painter, arena)

        def to_widget(point):
            return QPointF(point[0] * scale + dx, point[1] * scale + dy)

        if self._show_trail and self._view.xy is not None:
            trail = self._view.trail(self._frame, self._trail_s)
            if trail is not None and len(trail) > 1:
                # Fade the tail so recent travel reads as the leading edge.
                for i in range(1, len(trail)):
                    alpha = 0.15 + 0.65 * (i / len(trail))
                    painter.setPen(QPen(colors.qcolor_with_alpha(QColor(colors.ACCENT), alpha), 2))
                    painter.drawLine(to_widget(trail[i - 1]), to_widget(trail[i]))

        if self._view.xy is not None and 0 <= self._frame < len(self._view.xy):
            points = self._view.xy[self._frame]
            names = self._view.keypoint_names
            for i, point in enumerate(points):
                if not np.isfinite(point).all():
                    continue
                painter.setBrush(QBrush(behavior_qcolor(names[i] if i < len(names) else str(i))))
                painter.setPen(QPen(QColor(colors.CANVAS), 1))
                painter.drawEllipse(to_widget(point), 5, 5)

        label = self._view.label_at(self._frame)
        painter.setPen(QPen(QColor(colors.TEXT_PRIMARY)))
        painter.drawText(
            QRectF(8, 6, self.width() - 16, 20),
            Qt.AlignmentFlag.AlignLeft,
            f"frame {self._frame}   {label or '(unscored)'}",
        )

    def _paint_zones(self, painter, arena: QRectF) -> None:
        """Outline each zone in the colour the live overlay draws it in.

        Zone geometry is normalised, so it maps onto whatever rectangle the
        arena occupies on screen without knowing the resolution.
        """
        if self._zones is None:
            return
        for zone in getattr(self._zones, "zones", []):
            if not zone.vertices:
                continue
            b, g, r = zone.color
            painter.setPen(QPen(QColor(r, g, b), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            points = [
                QPointF(arena.x() + vx * arena.width(), arena.y() + vy * arena.height())
                for vx, vy in zone.vertices
            ]
            name = str(getattr(zone.shape, "value", zone.shape)).lower()
            if name == "rectangle" and len(points) >= 2:
                painter.drawRect(QRectF(points[0], points[1]).normalized())
            elif name == "circle" and len(points) >= 2:
                radius = (
                    (points[1].x() - points[0].x()) ** 2 + (points[1].y() - points[0].y()) ** 2
                ) ** 0.5
                painter.drawEllipse(points[0], radius, radius)
            elif len(points) >= 3:
                painter.drawPolygon(*points)

    def _why_blank(self) -> str:
        if self._view is None:
            return "Load a session"
        if self._view.xy is None:
            return (
                "No pose CSV could be found for this session.\n\n"
                "Looked at the path recorded in run.json, then beside the "
                "ethogram, then for a CSV named after this session in the "
                "folders above.\n\n"
                "Use “Choose pose CSV…” above to point at it."
            )
        return (
            "This session's pose sidecar records no resolution, so the arena "
            "cannot be sized.\n\n"
            "Use “Set arena size from video…” above to read it from the "
            "source video — it is stored, so this is a one-off."
        )


class AnalysisWindow(QMainWindow):
    """Scrub a session, select a window, and read what is in it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Behavior Analysis — session review")
        self.resize(1020, 720)
        self._view: SessionView | None = None
        self._ethogram_csv: Path | None = None
        self._frame = 0
        # Every loaded session, in the order they were found. The canvas shows
        # one of them; the window applies to all of them.
        self._cohort: list[tuple[Path, SessionView]] = []

        central = QWidget()
        layout = QVBoxLayout(central)

        top = QHBoxLayout()
        self._path_label = QLabel("No session loaded")
        self._path_label.setWordWrap(True)
        open_btn = QPushButton("Open ethogram…")
        open_btn.clicked.connect(self._open)
        # A cohort is the unit of analysis, not a session: the question is
        # almost always "what did these thirty animals do between minutes two
        # and seven", and answering it one file at a time invites the window
        # to drift between them.
        self._zones = None
        zones_btn = QPushButton("Load zones…")
        zones_btn.setToolTip(
            "A zone configuration from the zone editor. Time in zone, entries "
            "and latency are then reported for the selected window, for every "
            "loaded session."
        )
        zones_btn.clicked.connect(self._load_zones)

        open_folder_btn = QPushButton("Open cohort folder…")
        open_folder_btn.setToolTip(
            "Load every ethogram beneath a folder. The selected window then "
            "applies to all of them at once."
        )
        open_folder_btn.clicked.connect(self._open_folder)

        self._sessions = QComboBox()
        self._sessions.setMinimumWidth(180)
        self._sessions.setToolTip("Which loaded session the canvas and timeline show")
        self._sessions.currentIndexChanged.connect(self._on_session_picked)
        self._sessions.setVisible(False)
        # Runs made before the sidecar carried a resolution can still be
        # viewed: the video knows the number, so offer to read it from there
        # rather than make the operator re-run hours of inference.
        # Poses are looked for automatically, but a cohort can be laid out in
        # ways no search should guess at. This is the escape hatch.
        self._pick_poses = QPushButton("Choose pose CSV…")
        self._pick_poses.clicked.connect(self._choose_pose_csv)
        self._pick_poses.setVisible(False)
        top.addWidget(self._pick_poses)

        self._fix_resolution = QPushButton("Set arena size from video…")
        self._fix_resolution.clicked.connect(self._resolution_from_video)
        self._fix_resolution.setVisible(False)
        top.addWidget(self._path_label, 1)
        top.addWidget(self._sessions)
        top.addWidget(self._fix_resolution)
        top.addWidget(open_btn)
        top.addWidget(open_folder_btn)
        top.addWidget(zones_btn)
        layout.addLayout(top)

        self._canvas = KeypointCanvas()
        layout.addWidget(self._canvas, 1)

        self._bar = EthogramBar()
        self._bar.scrubbed.connect(self._set_frame)
        self._bar.selection_changed.connect(self._on_selection)
        layout.addWidget(self._bar)

        layout.addLayout(self._build_controls())

        self._summary = QLabel("Shift-drag (or right-drag) the ethogram to select a window.")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._bouts = QTableWidget(0, 6)
        self._bouts.setHorizontalHeaderLabels(
            ["Behavior", "Bouts", "Total (s)", "Fraction", "Mean (s)", "Median (s)"]
        )
        self._bouts.verticalHeader().setVisible(False)

        # Per-session rows for the same window. The cohort is the unit of
        # analysis, and a per-animal breakdown is what gets exported — so it
        # sits beside the shown session rather than replacing it.
        self._cohort_table = QTableWidget(0, 7)
        self._cohort_table.setHorizontalHeaderLabels(
            [
                "Session",
                "Scored",
                "Distance (cm)",
                "Mean (cm/s)",
                "Freezing (s)",
                "Darting (s)",
                "Top behavior",
            ]
        )
        self._cohort_table.verticalHeader().setVisible(False)

        self._zone_table = QTableWidget(0, 6)
        self._zone_table.setHorizontalHeaderLabels(
            ["Zone", "Time (s)", "Fraction", "Entries", "Mean bout (s)", "Latency (s)"]
        )
        self._zone_table.verticalHeader().setVisible(False)

        self._tables = QTabWidget()
        self._tables.addTab(self._bouts, "This session")
        self._tables.addTab(self._cohort_table, "Cohort")
        self._tables.addTab(self._zone_table, "Zones")
        layout.addWidget(self._tables, 1)

        export_row = QHBoxLayout()
        export_row.addStretch(1)
        self._export_btn = QPushButton("Export window…")
        self._export_btn.setToolTip(
            "Write the selected window's per-session numbers to a CSV, so the "
            "table on screen and the one in the analysis are the same table."
        )
        self._export_btn.clicked.connect(self._export_window)
        self._export_btn.setEnabled(False)
        export_row.addWidget(self._export_btn)
        layout.addLayout(export_row)

        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._play = QPushButton("Play")
        self._play.clicked.connect(self._toggle_play)
        row.addWidget(self._play)

        self._video_on = QCheckBox("Video")
        self._video_on.setChecked(True)
        self._video_on.setEnabled(False)
        self._video_on.setToolTip(
            "Draw the session's video behind the keypoints. Enabled when a "
            "video for this session can be found."
        )
        self._video_on.toggled.connect(self._canvas.set_show_video)
        row.addWidget(self._video_on)

        self._heatmap_on = QCheckBox("Heatmap")
        self._heatmap_on.setToolTip(
            "Where the animal spent the selected window, binned over the arena."
        )
        self._heatmap_on.toggled.connect(self._apply_heatmap)
        row.addWidget(self._heatmap_on)

        self._trail_on = QCheckBox("Centroid trail")
        self._trail_on.setChecked(True)
        self._trail_on.toggled.connect(self._apply_trail)
        row.addWidget(self._trail_on)

        self._trail_s = QDoubleSpinBox()
        self._trail_s.setRange(0.5, 60.0)
        self._trail_s.setValue(_TRAIL_DEFAULT_S)
        self._trail_s.setSuffix(" s")
        self._trail_s.valueChanged.connect(self._apply_trail)
        row.addWidget(self._trail_s)

        select_all = QPushButton("Select whole session")
        select_all.clicked.connect(self._select_all)
        row.addWidget(select_all)
        row.addStretch(1)

        self._clock = QLabel("—")
        row.addWidget(self._clock)
        return row

    # ------------------------------------------------------------------

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ethogram", "", "Ethogram CSV (*.csv);;All Files (*)"
        )
        if path:
            self.load(Path(path))

    def load(self, ethogram_csv: Path, *, pose_csv: Path | None = None) -> None:
        """Load a single session, replacing whatever was open."""
        try:
            view = SessionView.load(ethogram_csv, pose_csv=pose_csv)
        except SessionViewError as e:
            QMessageBox.critical(self, "Open session", str(e))
            return
        self._cohort = [(Path(ethogram_csv), view)]
        self._sessions.blockSignals(True)
        self._sessions.clear()
        self._sessions.addItem(Path(ethogram_csv).parent.name)
        self._sessions.blockSignals(False)
        self._sessions.setVisible(False)
        self._adopt(Path(ethogram_csv), view)

    def _adopt(self, ethogram_csv: Path, view: SessionView) -> None:
        """Show an already-loaded session."""
        self._view = view
        self._ethogram_csv = Path(ethogram_csv)
        self._path_label.setText(str(ethogram_csv))
        self._bar.set_view(view)
        self._canvas.set_view(view)
        self._apply_trail()
        self._set_frame(0)
        self._pick_poses.setVisible(view.xy is None)
        self._fix_resolution.setVisible(view.xy is not None and view.resolution is None)

        has_video = view.video_path is not None
        self._video_on.setEnabled(has_video)
        self._canvas.set_show_video(has_video and self._video_on.isChecked())

        found = f"  Poses: {view.pose_path.name}." if view.pose_path else "  No poses found."
        if has_video:
            found += f"  Video: {view.video_path.name}."
            if not view.video_is_aligned:
                # The annotated video is written from a queue that drops
                # frames, so a short file means every later frame is offset by
                # an unknown amount. Say so rather than scrub it confidently.
                found += (
                    f"  ⚠ It has {view.video_frames:,} frames against the session's "
                    f"{int(view.frames[-1]) + 1:,}, so frames may not line up."
                )
        self._summary.setText(
            f"{view.n_rows:,} scored rows at {view.fps:.2f} fps "
            f"({view.duration_s / 60:.1f} min)."
            + found
            + ("" if view.px_per_mm else "  No calibration found: distances unavailable.")
        )

    def keyPressEvent(self, event):
        """Frame-accurate scrubbing from the keyboard.

        Dragging the ethogram is fast but coarse — on a 45,000-frame session
        one pixel is tens of frames, so a bout boundary cannot be found with
        the mouse at all. Left/Right step exactly one frame; shift steps ten
        and ctrl a second, for covering ground without losing precision.
        """
        if self._view is None:
            super().keyPressEvent(event)
            return

        key = event.key()
        if key not in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Home,
            Qt.Key.Key_End,
            Qt.Key.Key_Space,
        ):
            super().keyPressEvent(event)
            return

        if key == Qt.Key.Key_Space:
            self._toggle_play()
            event.accept()
            return

        last = int(self._view.frames[-1]) if self._view.n_rows else 0
        if key == Qt.Key.Key_Home:
            target = int(self._view.frames[0]) if self._view.n_rows else 0
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
        self._timer.stop()
        self._play.setText("Play")
        self._set_frame(max(0, min(last, target)))
        event.accept()

    def _open_folder(self) -> None:
        """Load every ethogram beneath a folder as one cohort."""
        folder = QFileDialog.getExistingDirectory(self, "Folder of apply-run outputs")
        if not folder:
            return
        found = sorted(Path(folder).rglob("ethogram_raw.csv"))
        if not found:
            QMessageBox.warning(
                self,
                "Open cohort",
                f"No ethogram_raw.csv found under {folder}.\n\n"
                "Pick the output folder an apply run wrote to — each session "
                "lives in its own subfolder there.",
            )
            return
        self.load_many(found)

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

        self._cohort = loaded
        self._sessions.blockSignals(True)
        self._sessions.clear()
        self._sessions.addItems([p.parent.name for p, _ in loaded])
        self._sessions.blockSignals(False)
        self._sessions.setVisible(len(loaded) > 1)
        self._show_session(0)
        if failed:
            QMessageBox.warning(
                self,
                "Open cohort",
                f"Loaded {len(loaded)}; {len(failed)} could not be read:\n\n"
                + "\n".join(failed[:8]),
            )

    def _on_session_picked(self, index: int) -> None:
        if 0 <= index < len(self._cohort):
            self._show_session(index)

    def _show_session(self, index: int) -> None:
        """Put one of the loaded sessions on screen, keeping the window."""
        path, view = self._cohort[index]
        selection = self._bar.selection()
        self._adopt(path, view)
        if selection is not None:
            # The window is the question being asked; switching which animal
            # answers it must not silently reset it.
            self._bar.set_selection(*selection)

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

    def _set_frame(self, frame: int) -> None:
        self._frame = int(frame)
        self._bar.set_frame(self._frame)
        self._canvas.set_frame(self._frame)
        if self._view and self._view.fps:
            seconds = self._frame / self._view.fps
            self._clock.setText(f"{int(seconds) // 60:d}:{seconds % 60:05.2f}")

    def _apply_heatmap(self, *_args) -> None:
        """Bin the selected window, or clear the overlay."""
        selection = self._bar.selection()
        if not self._heatmap_on.isChecked() or self._view is None or selection is None:
            self._canvas.set_heatmap(None)
            return
        from glider.analysis.behavior.spatial import SpatialError, occupancy_grid

        try:
            grid, _x, _y = occupancy_grid(
                self._view, bins=60, start_frame=selection[0], end_frame=selection[1]
            )
        except SpatialError as e:
            logger.info("no heatmap for this session: %s", e)
            self._canvas.set_heatmap(None)
            return
        self._canvas.set_heatmap(grid)

    def _apply_trail(self, *_args) -> None:
        self._canvas.set_trail(self._trail_s.value(), self._trail_on.isChecked())

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._play.setText("Play")
        elif self._view is not None:
            # Playback is deliberately wall-clock, not frame-locked: this is a
            # review tool, and a smooth approximate rate reads better than a
            # stuttering exact one.
            self._timer.start(int(1000 / max(1.0, self._view.fps)))
            self._play.setText("Pause")

    def _advance(self) -> None:
        if self._view is None:
            return
        total = int(self._view.frames[-1]) + 1 if self._view.n_rows else 0
        if self._frame + 1 >= total:
            self._timer.stop()
            self._play.setText("Play")
            return
        self._set_frame(self._frame + 1)

    def _select_all(self) -> None:
        if self._view is not None and self._view.n_rows:
            self._bar.set_selection(int(self._view.frames[0]), int(self._view.frames[-1]))

    def _on_selection(self, start: int, end: int) -> None:
        if self._view is None:
            return
        stats = self._view.segment_stats(start, end)
        self._fill_bouts(stats)
        self._summary.setText(self._describe(stats))
        self._fill_cohort(start, end)
        self._fill_zones(start, end)
        self._apply_heatmap()
        self._export_btn.setEnabled(True)

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
        names = [z.name for z in self._zones.zones]
        self._summary.setText(
            f"{self._summary.text()}\nZones: {', '.join(names) or '(none defined)'}"
        )
        self._canvas.set_zones(self._zones)
        selection = self._bar.selection()
        if selection is not None:
            self._on_selection(*selection)

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
                item = QTableWidgetItem(value)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._zone_table.setItem(r, c, item)
        self._tables.setTabText(2, f"Zones ({len(rows)})" if len(rows) else "Zones")

    def cohort_rows(self, start: int, end: int) -> list[dict]:
        """The selected window, per loaded session.

        The same frame window is applied to every session rather than a
        per-session fraction: "minutes two to seven" has to mean the same
        stretch in each animal or the comparison is not one.
        """
        rows = []
        for path, view in self._cohort:
            stats = view.segment_stats(start, end)
            scored = [lab for lab in view.labels if lab]
            top = ""
            if not stats.bouts.empty:
                top = str(stats.bouts.iloc[0]["state"])
            speed = stats.speed_bouts.set_index("state") if not stats.speed_bouts.empty else None
            rows.append(
                {
                    "session": path.parent.name,
                    "scored_rows": len(scored),
                    "duration_s": stats.duration_s,
                    "distance_cm": stats.distance_cm,
                    "mean_cm_s": stats.mean_speed_cm_s,
                    "peak_cm_s": stats.peak_speed_cm_s,
                    "freezing_s": (
                        float(speed.loc["freezing", "total_s"])
                        if speed is not None and "freezing" in speed.index
                        else 0.0
                    ),
                    "darting_s": (
                        float(speed.loc["darting", "total_s"])
                        if speed is not None and "darting" in speed.index
                        else 0.0
                    ),
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
                row["top_behavior"] or "—",
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
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
        self._summary.setText(f"{self._summary.text()}\nWrote {path}")

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
        # Posture and speed are independent scorings of the same frames, so
        # they are stacked with a divider rather than summed into one list:
        # their fractions each run to 1 over the window, and adding them
        # would double-count it.
        import pandas as pd

        rows = stats.bouts
        if not stats.speed_bouts.empty:
            divider = pd.DataFrame([{**dict.fromkeys(rows.columns), "state": "— speed axis —"}])
            rows = pd.concat([rows, divider, stats.speed_bouts], ignore_index=True)
        self._bouts.setRowCount(len(rows))
        for r, (_, row) in enumerate(rows.iterrows()):
            if pd.isna(row["n_bouts"]):  # the divider carries no numbers
                values = [str(row["state"]), "", "", "", "", ""]
            else:
                values = [
                    str(row["state"] or "(unscored)"),
                    str(int(row["n_bouts"])),
                    f"{row['total_s']:.2f}",
                    f"{100 * row['fraction']:.1f}%",
                    f"{row['mean_s']:.2f}",
                    f"{row['median_s']:.2f}",
                ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._bouts.setItem(r, c, item)

    def closeEvent(self, event):
        self._timer.stop()
        self._canvas._close_reader()
        super().closeEvent(event)


__all__ = ["AnalysisWindow", "EthogramBar", "KeypointCanvas", "behavior_qcolor"]
