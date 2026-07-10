"""Tests for CameraPreviewWidget pose-skeleton + behavior-label overlays.

Chosen approach: the drawing is factored into a small ``_draw_overlays(frame)``
helper on the widget that mutates/returns the BGR numpy frame in place. We test
that helper directly on numpy frames (clean, pixel-region assertions), and we
also exercise the public ``update_frame`` path to confirm the overlay draw
actually runs end-to-end (the resulting QPixmap differs when overlays are set).
"""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from glider.gui.panels.camera_panel import CameraPreviewWidget


def _solid_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """A solid mid-gray BGR frame."""
    return np.full((h, w, 3), 100, dtype=np.uint8)


def _make_widget(qtbot) -> CameraPreviewWidget:
    w = CameraPreviewWidget()
    qtbot.addWidget(w)
    w.resize(320, 240)
    w.show()
    return w


def test_draw_overlays_noop_without_state(qtbot):
    w = _make_widget(qtbot)
    frame = _solid_frame()
    out = w._draw_overlays(frame.copy())
    # No pose / label set → frame unchanged.
    assert np.array_equal(out, frame)


def test_draw_overlays_applies_skeleton_and_badge(qtbot):
    w = _make_widget(qtbot)
    base = _solid_frame()

    w.set_pose_overlay(np.array([[10, 10], [30, 30], [50, 20]], dtype=float))
    w.set_behavior_label("Grooming")
    w.set_behavior_vocab(["Grooming", "Rearing"])

    out = w._draw_overlays(base.copy())
    # The draw path ran: some pixels differ from the solid frame.
    assert np.count_nonzero(np.any(out != base, axis=2)) > 0

    # Skeleton region (around the keypoints) changed.
    skel_region_changed = np.any(out[5:55, 5:55] != base[5:55, 5:55])
    assert skel_region_changed

    # Badge is drawn near the top-left (default x=16, y=16).
    badge_region_changed = np.any(out[16:60, 16:120] != base[16:60, 16:120])
    assert badge_region_changed


def test_set_pose_overlay_none_clears(qtbot):
    w = _make_widget(qtbot)
    base = _solid_frame()

    w.set_pose_overlay(np.array([[10, 10], [30, 30], [50, 20]], dtype=float))
    changed = w._draw_overlays(base.copy())
    assert not np.array_equal(changed, base)

    w.set_pose_overlay(None)
    cleared = w._draw_overlays(base.copy())
    assert np.array_equal(cleared, base)


def test_update_frame_runs_overlay_path(qtbot):
    """End-to-end: update_frame renders differently once overlays are set."""
    w = _make_widget(qtbot)

    w.update_frame(_solid_frame())
    before = w.pixmap().toImage()

    w.set_pose_overlay(np.array([[10, 10], [30, 30], [50, 20]], dtype=float))
    w.set_behavior_label("Grooming")
    w.set_behavior_vocab(["Grooming", "Rearing"])
    w.update_frame(_solid_frame())
    after = w.pixmap().toImage()

    # The rendered pixmaps must differ once overlays are drawn.
    def _bytes(img):
        img = img.convertToFormat(img.format())
        ptr = img.bits()
        ptr.setsize(img.sizeInBytes())
        return bytes(ptr)

    assert _bytes(before) != _bytes(after)
