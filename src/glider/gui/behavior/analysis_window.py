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
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
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

        # One rect per run of identical labels, so a long recording paints in
        # a few dozen fills rather than tens of thousands.
        frames, labels = self._view.frames, self._view.labels
        run_start, run_label = 0, labels[0]
        for i in range(1, len(labels) + 1):
            if i < len(labels) and labels[i] == run_label:
                continue
            x0 = self._x_of(int(frames[run_start]))
            x1 = self._x_of(int(frames[i - 1]) + 1)
            painter.fillRect(
                QRectF(x0, 0, max(1.0, x1 - x0), self.height()), behavior_qcolor(run_label)
            )
            if i < len(labels):
                run_start, run_label = i, labels[i]

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

    def set_view(self, view: SessionView | None) -> None:
        self._view = view
        self._frame = 0
        self.update()

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
        if self._view is None or self._view.xy is None or transform is None:
            painter.setPen(QPen(QColor(colors.TEXT_MUTED)))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._why_blank(),
            )
            return

        scale, dx, dy = transform
        width, height = self._view.resolution
        painter.setPen(QPen(QColor(colors.BORDER), 1))
        painter.drawRect(QRectF(dx, dy, width * scale, height * scale))

        def to_widget(point):
            return QPointF(point[0] * scale + dx, point[1] * scale + dy)

        if self._show_trail:
            trail = self._view.trail(self._frame, self._trail_s)
            if trail is not None and len(trail) > 1:
                # Fade the tail so recent travel reads as the leading edge.
                for i in range(1, len(trail)):
                    alpha = 0.15 + 0.65 * (i / len(trail))
                    painter.setPen(QPen(colors.qcolor_with_alpha(QColor(colors.ACCENT), alpha), 2))
                    painter.drawLine(to_widget(trail[i - 1]), to_widget(trail[i]))

        if 0 <= self._frame < len(self._view.xy):
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

    def _why_blank(self) -> str:
        if self._view is None:
            return "Load a session"
        if self._view.xy is None:
            return "No pose CSV was found beside this ethogram, so there is nothing to draw."
        return (
            "This session's pose sidecar records no resolution, so the arena "
            "cannot be sized. Re-run tracking, or add it to the .meta.json."
        )


class AnalysisWindow(QMainWindow):
    """Scrub a session, select a window, and read what is in it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Behavior Analysis — session review")
        self.resize(1020, 720)
        self._view: SessionView | None = None
        self._frame = 0

        central = QWidget()
        layout = QVBoxLayout(central)

        top = QHBoxLayout()
        self._path_label = QLabel("No session loaded")
        self._path_label.setWordWrap(True)
        open_btn = QPushButton("Open ethogram…")
        open_btn.clicked.connect(self._open)
        # Runs made before the sidecar carried a resolution can still be
        # viewed: the video knows the number, so offer to read it from there
        # rather than make the operator re-run hours of inference.
        self._fix_resolution = QPushButton("Set arena size from video…")
        self._fix_resolution.clicked.connect(self._resolution_from_video)
        self._fix_resolution.setVisible(False)
        top.addWidget(self._path_label, 1)
        top.addWidget(self._fix_resolution)
        top.addWidget(open_btn)
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
        layout.addWidget(self._bouts, 1)

        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._play = QPushButton("Play")
        self._play.clicked.connect(self._toggle_play)
        row.addWidget(self._play)

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

    def load(self, ethogram_csv: Path) -> None:
        """Load a session and everything sitting beside it."""
        try:
            view = SessionView.load(ethogram_csv)
        except SessionViewError as e:
            QMessageBox.critical(self, "Open session", str(e))
            return
        self._view = view
        self._path_label.setText(str(ethogram_csv))
        self._bar.set_view(view)
        self._canvas.set_view(view)
        self._apply_trail()
        self._set_frame(0)
        self._fix_resolution.setVisible(view.xy is not None and view.resolution is None)
        self._summary.setText(
            f"{view.n_rows:,} scored rows at {view.fps:.2f} fps "
            f"({view.duration_s / 60:.1f} min)."
            + ("" if view.px_per_mm else "  No calibration found: distances unavailable.")
        )

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
                item = QTableWidgetItem(value)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._bouts.setItem(r, c, item)

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)


__all__ = ["AnalysisWindow", "EthogramBar", "KeypointCanvas", "behavior_qcolor"]
