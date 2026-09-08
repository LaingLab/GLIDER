"""Per-camera exposure and image controls, with a live look at what they cost.

Eight cameras on one rig had eight different exposures and nothing said so. One
ran auto-exposure out to its one-second cap and stalled; another clipped 64% of
its frame to pure white, which no bitrate recovers because the information is
gone in the sensor. Both were only found afterwards, from the files.

So this panel is not really about editing settings - the settings dialog could
do that. It is about the readout beside them: the share of the frame currently
clipped, while the operator can still turn a dial. A number that appears after
the session is a post-mortem; the same number during it is a fix.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import colors
from glider.gui.widgets.tool_ui import caption, hint, set_button_role
from glider.vision.camera_manager import CameraSettings

__all__ = ["CameraPropertiesPanel", "exposure_verdict", "frame_exposure"]

#: Above this share of the frame at full white, detail is being destroyed
#: before it reaches the encoder. Chosen from a real failure: the unusable
#: recording sat at 64%, the ones that scored fine at under 5%.
CLIPPED_BAD = 20.0

#: Worth mentioning, not worth alarming about.
CLIPPED_WARN = 5.0

#: A frame this flat has too little contrast to separate an animal from the
#: floor, whatever the exposure reads.
FLAT_STDDEV = 15.0

#: How long after the last keystroke the change is pushed to the camera.
#: Applying on every tick of a spin box would hammer the device while someone
#: holds an arrow key down.
_APPLY_DEBOUNCE_MS = 250


def frame_exposure(frame: np.ndarray) -> tuple[float, float, float]:
    """``(mean, stddev, percent clipped white)`` for one frame.

    Subsampled rather than exact: this runs on every displayed frame for the
    selected camera, and a tenth of the pixels answers the question just as
    well as all of them.
    """
    if frame is None or frame.size == 0:
        return (0.0, 0.0, 0.0)
    grey = frame[..., 0] if frame.ndim == 3 else frame
    sample = grey[::3, ::3]
    return (
        float(sample.mean()),
        float(sample.std()),
        float(100.0 * (sample >= 250).mean()),
    )


def exposure_verdict(mean: float, stddev: float, clipped: float) -> tuple[str, str]:
    """``(message, colour)`` describing what the exposure is costing.

    Ordered by what actually ruins a recording. Clipping comes first because it
    is unrecoverable; darkness is merely bad.
    """
    if clipped >= CLIPPED_BAD:
        return (f"{clipped:.0f}% of the frame is pure white - detail is being lost", colors.ERROR)
    if clipped >= CLIPPED_WARN:
        return (f"{clipped:.0f}% clipped - bright, worth easing back", colors.WARNING)
    if stddev < FLAT_STDDEV:
        return ("very low contrast - an animal will not separate from the floor", colors.WARNING)
    if mean < 40:
        return ("very dark - exposure will run long and the frame rate with it", colors.WARNING)
    return (f"looks good ({clipped:.1f}% clipped)", colors.STATE_OK)


class CameraPropertiesPanel(QWidget):
    """Exposure and image controls for whichever camera is selected."""

    #: (camera_id, settings) once the operator has stopped typing.
    settings_changed = pyqtSignal(str, object)
    #: The operator asked to rename this camera.
    rename_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera_id: str | None = None
        self._settings: CameraSettings | None = None
        # Set while loading a camera's values into the widgets, so that filling
        # the form does not read as the operator editing it and push the values
        # straight back at the camera.
        self._loading = False

        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(_APPLY_DEBOUNCE_MS)
        self._apply_timer.timeout.connect(self._emit_settings)

        self._build_ui()
        self.set_camera(None, None, "")

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self._name_label = QLabel("No camera selected")
        self._name_label.setStyleSheet("font-weight: bold;")
        header.addWidget(self._name_label)
        header.addStretch(1)
        self._rename_button = QPushButton("Rename")
        self._rename_button.setToolTip("Give this camera a name that says which arena it is")
        set_button_role(self._rename_button, "ghost")
        self._rename_button.clicked.connect(self._on_rename)
        header.addWidget(self._rename_button)
        layout.addLayout(header)

        form = QFormLayout()
        form.setSpacing(6)

        self._auto_exposure = QCheckBox("Auto exposure")
        self._auto_exposure.setToolTip(
            "Auto-exposure in a dark arena runs the exposure time out to a full "
            "second, and the frame rate follows it. Set it manually for a rig."
        )
        self._auto_exposure.toggled.connect(self._on_auto_toggled)
        form.addRow("", self._auto_exposure)

        self._exposure = self._spin(-13, 0, "Lower is darker. Driver units, not seconds.")
        form.addRow(caption("Exposure"), self._exposure)

        self._brightness = self._spin(0, 255, "")
        form.addRow(caption("Brightness"), self._brightness)

        self._contrast = self._spin(0, 255, "")
        form.addRow(caption("Contrast"), self._contrast)
        layout.addLayout(form)

        self._readout = QLabel("")
        self._readout.setWordWrap(True)
        layout.addWidget(self._readout)

        layout.addWidget(
            hint(
                "Readings come from the live frame. Clipped pixels are gone before "
                "the video is encoded, so no bitrate recovers them."
            )
        )
        layout.addStretch(1)

    def _spin(self, low: int, high: int, tip: str) -> QSpinBox:
        box = QSpinBox()
        box.setRange(low, high)
        if tip:
            box.setToolTip(tip)
        box.valueChanged.connect(self._on_edited)
        return box

    # ------------------------------------------------------------------

    def set_camera(self, camera_id: str | None, settings, display_name: str = "") -> None:
        """Show *camera_id*'s settings, or clear the panel when None."""
        self._apply_timer.stop()
        self._camera_id = camera_id
        self._settings = settings
        enabled = camera_id is not None and settings is not None
        for widget in (self._auto_exposure, self._exposure, self._brightness, self._contrast):
            widget.setEnabled(enabled)
        self._rename_button.setEnabled(camera_id is not None)

        if not enabled:
            self._name_label.setText("No camera selected")
            self._readout.setText("")
            return

        self._name_label.setText(display_name or camera_id)
        self._loading = True
        try:
            self._auto_exposure.setChecked(bool(settings.auto_exposure))
            self._exposure.setValue(int(settings.exposure))
            self._brightness.setValue(int(settings.brightness))
            self._contrast.setValue(int(settings.contrast))
        finally:
            self._loading = False
        self._exposure.setEnabled(not settings.auto_exposure)

    def update_readout(self, frame: np.ndarray) -> None:
        """Refresh the exposure readout from the selected camera's latest frame."""
        if self._camera_id is None:
            return
        mean, stddev, clipped = frame_exposure(frame)
        message, colour = exposure_verdict(mean, stddev, clipped)
        self._readout.setText(f"mean {mean:.0f}   contrast {stddev:.0f}   {message}")
        self._readout.setStyleSheet(f"color: {colour};")

    # ------------------------------------------------------------------

    def _on_rename(self) -> None:
        if self._camera_id is not None:
            self.rename_requested.emit(self._camera_id)

    def _on_auto_toggled(self, checked: bool) -> None:
        self._exposure.setEnabled(not checked)
        self._on_edited()

    def _on_edited(self, *_args) -> None:
        if self._loading or self._camera_id is None:
            return
        self._apply_timer.start()

    def _emit_settings(self) -> None:
        """Hand the camera a copy, never the object the manager is holding.

        Mutating the manager's own settings in place would change what every
        other camera sharing that object records at, which is how a rig ends up
        recording at one camera's exposure.
        """
        if self._camera_id is None or self._settings is None:
            return
        import dataclasses

        updated = dataclasses.replace(
            self._settings,
            auto_exposure=self._auto_exposure.isChecked(),
            exposure=self._exposure.value(),
            brightness=self._brightness.value(),
            contrast=self._contrast.value(),
        )
        self._settings = updated
        self.settings_changed.emit(self._camera_id, updated)
