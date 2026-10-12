"""The desktop run page: a camera-first cockpit.

    +-------------------------------------+-----------+
    | header (title, pill, field tags)    | run box   |
    +-------------------------------------+-----------+
    |                                     | functions |
    |  camera                 fps · drops | + devices |
    |                              ● REC  |           |
    +-------------------------------------+-----------+

Panels are built by MainWindow. This view places them and owns the camera
health overlay (fps, dropped frames, REC), which MainWindow feeds once a second.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

_RAIL_WIDTH = 340


class DashboardView(QWidget):
    def __init__(self, camera, header, run_control, device_controls, parent=None):
        super().__init__(parent)
        self._panels = (camera, header, run_control, device_controls)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(8)

        self._stage = QWidget()
        stage = QVBoxLayout(self._stage)
        stage.setContentsMargins(0, 0, 0, 0)
        stage.setSpacing(8)
        stage.addWidget(header)
        self._camera_frame = self._card(camera)
        stage.addWidget(self._camera_frame, 1)
        row.addWidget(self._stage, 1)

        self._rail = QWidget()
        self._rail.setFixedWidth(_RAIL_WIDTH)
        rail = QVBoxLayout(self._rail)
        rail.setContentsMargins(0, 0, 0, 0)
        rail.setSpacing(8)
        rail.addWidget(self._card(run_control))
        device_controls.setMinimumHeight(0)
        rail.addWidget(self._card(device_controls), 1)
        row.addWidget(self._rail)

        self._overlay = QWidget(self._camera_frame)
        self._overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        ov = QHBoxLayout(self._overlay)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(6)
        self._fps = QLabel("-- fps")
        self._dropped = QLabel("0 dropped")
        self._rec = QLabel("● REC")
        self._rec.setProperty("recording", True)
        self._rec.hide()
        for label in (self._fps, self._dropped, self._rec):
            label.setProperty("cameraBadge", True)
            ov.addWidget(label)
        self._dropped.setProperty("warn", False)
        self._camera_frame.installEventFilter(self)

    @staticmethod
    def _card(widget: QWidget) -> QFrame:
        frame = QFrame()
        frame.setProperty("runCard", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        return frame

    def eventFilter(self, obj, event):  # noqa: N802 (Qt override)
        if obj is self._camera_frame and event.type() == QEvent.Type.Resize:
            self._place_overlay()
        return super().eventFilter(obj, event)

    def _place_overlay(self) -> None:
        self._overlay.adjustSize()
        hint = self._overlay.sizeHint()
        frame = self._camera_frame.rect()
        self._overlay.setGeometry(
            frame.right() - hint.width() - 10,
            frame.bottom() - hint.height() - 10,
            hint.width(),
            hint.height(),
        )
        self._overlay.raise_()

    # --- public API ---

    def stage(self) -> QWidget:
        return self._stage

    def rail(self) -> QWidget:
        return self._rail

    def set_health(self, fps: float | None, dropped: int, recording: bool) -> None:
        self._fps.setText("-- fps" if fps is None else f"{fps:.1f} fps")
        self._dropped.setText(f"{dropped} dropped")
        self._dropped.setProperty("warn", dropped > 0)
        self._dropped.style().unpolish(self._dropped)
        self._dropped.style().polish(self._dropped)
        self._rec.setVisible(recording)
        self._place_overlay()

    def update_state(self, state_name: str, recording: bool = False) -> None:
        # ``recording`` kept for MainWindow's call signature; REC comes from set_health.
        for widget in self._panels:
            if hasattr(widget, "update_state"):
                widget.update_state(state_name)
