"""CalibrationDialog drives from any FrameProvider, camera or video."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from glider.gui.dialogs.calibration_dialog import CalibrationDialog
from glider.vision.calibration import CameraCalibration


class _StillProvider:
    """Not seekable — stands in for the live camera."""

    def __init__(self, connected=True, w=640, h=480):
        self.is_connected = connected
        self._frame = np.zeros((h, w, 3), dtype=np.uint8)

    def get_frame(self):
        return (self._frame, 0.0) if self.is_connected else None


class _ScrubProvider(_StillProvider):
    """Seekable — stands in for a video file."""

    frame_count = 12

    def __init__(self, **kw):
        super().__init__(**kw)
        self.position = 0

    def seek(self, frame_index):
        self.position = max(0, min(int(frame_index), self.frame_count - 1))


class _GhostProvider(_StillProvider):
    """Reports disconnected but would happily hand back a frame if asked.

    `_StillProvider.get_frame()` returns None when disconnected, so it can't
    distinguish a guarded `_capture_frame` (checks `is_connected` first, never
    calls `get_frame`) from an unguarded one (calls `get_frame`, gets None,
    bails the same way). This provider makes the two paths diverge: a bug
    that dropped the guard would call `get_frame` and get a real frame back.
    """

    def __init__(self, **kw):
        super().__init__(connected=False, **kw)
        self.get_frame_calls = 0

    def get_frame(self):
        self.get_frame_calls += 1
        return (self._frame, 0.0)


class _DistinctFrameScrubProvider(_ScrubProvider):
    """Seekable provider whose frame content encodes the current position.

    Lets a test tell "the preview redrew with whatever frame was already
    there" apart from "the preview actually received a new frame for this
    position" — a plain zeros frame looks the same before and after a seek.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_frame_calls = 0

    def get_frame(self):
        self.get_frame_calls += 1
        if not self.is_connected:
            return None
        h, w = self._frame.shape[:2]
        frame = np.full((h, w, 3), self.position, dtype=np.uint8)
        return (frame, 0.0)


def _dialog(qtbot, provider, **kw):
    dialog = CalibrationDialog(frame_provider=provider, calibration=CameraCalibration(), **kw)
    qtbot.addWidget(dialog)
    return dialog


class TestProviderWiring:
    def test_captures_the_initial_frame_from_any_provider(self, qtbot):
        dialog = _dialog(qtbot, _StillProvider(w=800, h=600))
        assert dialog._calibration.calibration_width == 800
        assert dialog._calibration.calibration_height == 600

    def test_disconnected_provider_leaves_resolution_unset(self, qtbot):
        dialog = _dialog(qtbot, _StillProvider(connected=False))
        assert dialog._calibration.calibration_width == 0


class TestScrubber:
    def test_absent_for_a_still_provider(self, qtbot):
        assert _dialog(qtbot, _StillProvider())._scrubber is None

    def test_present_for_a_seekable_provider(self, qtbot):
        dialog = _dialog(qtbot, _ScrubProvider())
        assert dialog._scrubber is not None
        assert dialog._scrubber.maximum() == _ScrubProvider.frame_count - 1

    def test_moving_it_seeks_the_provider(self, qtbot):
        provider = _ScrubProvider()
        dialog = _dialog(qtbot, provider)
        dialog._scrubber.setValue(7)
        assert provider.position == 7


class TestFileButtons:
    def test_shown_by_default(self, qtbot):
        assert _dialog(qtbot, _StillProvider())._save_btn is not None

    def test_hidden_when_suppressed(self, qtbot):
        # Next to a batch master file, a per-file save is a trap.
        dialog = _dialog(qtbot, _StillProvider(), show_file_buttons=False)
        assert dialog._save_btn is None
        assert dialog._load_btn is None


class TestCaptureButton:
    def test_clicking_it_with_a_dead_source_still_warns(self, qtbot, monkeypatch):
        # clicked() emits `checked`, always False for a non-checkable button.
        # Connected straight to _capture_frame it would land in `announce` and
        # silently swallow this warning.
        dialog = _dialog(qtbot, _StillProvider(connected=False))
        warned = {}
        monkeypatch.setattr(
            "glider.gui.dialogs.calibration_dialog.QMessageBox.warning",
            lambda *a, **k: warned.setdefault("shown", True),
        )
        dialog._capture_btn.click()
        assert warned.get("shown") is True

    def test_construction_is_silent_for_a_dead_source(self, qtbot, monkeypatch):
        warned = {}
        monkeypatch.setattr(
            "glider.gui.dialogs.calibration_dialog.QMessageBox.warning",
            lambda *a, **k: warned.setdefault("shown", True),
        )
        _dialog(qtbot, _StillProvider(connected=False))
        assert warned == {}


class TestZeroLengthLine:
    def test_a_zero_length_line_is_refused(self, qtbot, monkeypatch):
        dialog = _dialog(qtbot, _StillProvider())
        warned = {}
        monkeypatch.setattr(
            "glider.gui.dialogs.calibration_dialog.QMessageBox.warning",
            lambda *a, **k: warned.setdefault("shown", True),
        )
        dialog._on_line_defined((100, 100), (100, 100))
        dialog._add_pending_line()
        assert warned.get("shown") is True
        assert dialog._calibration.lines == []

    def test_a_real_line_is_accepted(self, qtbot):
        dialog = _dialog(qtbot, _StillProvider())
        dialog._on_line_defined((0, 240), (640, 240))
        dialog._length_spin.setValue(100.0)
        dialog._add_pending_line()
        assert len(dialog._calibration.lines) == 1


class TestCaptureFrameGuard:
    def test_disconnected_provider_is_never_asked_for_a_frame(self, qtbot):
        # Pins the `is_connected` guard in `_capture_frame` itself: a
        # `_StillProvider` can't tell a guarded miss from an unguarded one
        # since its `get_frame()` also returns None while disconnected. This
        # provider would return a real frame if `get_frame` were ever called.
        provider = _GhostProvider()
        dialog = _dialog(qtbot, provider)
        assert dialog._calibration.calibration_width == 0
        assert provider.get_frame_calls == 0

    def test_explicit_capture_on_disconnected_provider_skips_get_frame(self, qtbot):
        provider = _GhostProvider()
        dialog = _dialog(qtbot, provider)
        assert dialog._capture_frame(announce=False) is False
        assert provider.get_frame_calls == 0


class TestFrameLabel:
    def test_shows_initial_position(self, qtbot):
        dialog = _dialog(qtbot, _ScrubProvider())
        assert dialog._frame_label.text() == "1 / 12"

    def test_updates_after_scrubbing(self, qtbot):
        dialog = _dialog(qtbot, _ScrubProvider())
        dialog._scrubber.setValue(7)
        assert dialog._frame_label.text() == "8 / 12"

    def test_does_not_advance_past_a_failed_seek(self, qtbot):
        # A failed read (announce=False) is silent; the label must not lie
        # about which frame the preview is actually showing.
        provider = _ScrubProvider()
        dialog = _dialog(qtbot, provider)
        provider.is_connected = False
        dialog._scrubber.setValue(7)
        assert dialog._frame_label.text() == "1 / 12"


class TestCaptureButtonLabel:
    def test_relabeled_for_a_seekable_provider(self, qtbot):
        dialog = _dialog(qtbot, _ScrubProvider())
        assert dialog._capture_btn.text() == "Use This Frame"

    def test_stays_capture_frame_for_a_still_provider(self, qtbot):
        dialog = _dialog(qtbot, _StillProvider())
        assert dialog._capture_btn.text() == "Capture Frame"


class TestClearButtonWithoutFileButtons:
    def test_exists_and_works_when_file_buttons_are_hidden(self, qtbot, monkeypatch):
        # The trap `show_file_buttons=False` exists to avoid: losing the
        # clear button entirely instead of just moving it to its own row.
        dialog = _dialog(qtbot, _StillProvider(), show_file_buttons=False)
        assert dialog._clear_btn is not None

        dialog._on_line_defined((0, 240), (640, 240))
        dialog._length_spin.setValue(100.0)
        dialog._add_pending_line()
        assert len(dialog._calibration.lines) == 1

        monkeypatch.setattr(
            "glider.gui.dialogs.calibration_dialog.QMessageBox.question",
            lambda *a, **k: QMessageBox.StandardButton.Yes,
        )
        dialog._clear_btn.click()
        assert dialog._calibration.lines == []


class TestScrubRecapture:
    def test_scrubbing_recaptures_a_new_frame(self, qtbot):
        # Distinguishes "seeks the provider" from "actually re-pulls and
        # redraws a frame for the new position" — a regression could seek
        # without recapturing and this would still look fine on a still frame.
        provider = _DistinctFrameScrubProvider()
        dialog = _dialog(qtbot, provider)
        calls_before = provider.get_frame_calls

        dialog._scrubber.setValue(7)

        assert provider.get_frame_calls == calls_before + 1
        assert (dialog._preview._frame == 7).all()

    def test_calibration_lines_survive_a_scrub(self, qtbot):
        # Issue 3 removed a redundant `set_calibration` call from `_on_scrub`.
        # The preview must still be drawing over the *current* calibration
        # (same object, still holding its lines) after scrubbing, not a stale
        # or emptied one.
        dialog = _dialog(qtbot, _ScrubProvider())
        dialog._on_line_defined((0, 240), (640, 240))
        dialog._length_spin.setValue(100.0)
        dialog._add_pending_line()
        assert len(dialog._calibration.lines) == 1

        dialog._scrubber.setValue(7)

        assert dialog._preview._calibration is dialog._calibration
        assert len(dialog._preview._calibration.lines) == 1
