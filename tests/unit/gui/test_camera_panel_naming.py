"""Naming animals by click at run start (identity "Name at start")."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from glider.gui.panels.camera_panel import CameraPanel, CameraPreviewWidget, FrameData
from glider.vision.camera_manager import CameraManager
from glider.vision.cv_processor import CVProcessor, CVSettings, Detection, DetectionBackend

pytestmark = pytest.mark.usefixtures("qtbot")


def _det(x: int, y: int = 10) -> Detection:
    return Detection(class_id=0, class_name="mouse", confidence=0.9, bbox=(x, y, 10, 10))


def _frame() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


def _cv(mode="continuity", dets=None, backend=DetectionBackend.YOLO_V8):
    cv = CVProcessor(
        CVSettings(
            backend=backend,
            identity_mode=mode,
            animals=[
                {"name": "Mouse A", "class_name": None},
                {"name": "Mouse B", "class_name": None},
            ],
        )
    )
    cv._initialized = True
    cv._detect_yolo = MagicMock(side_effect=lambda *_a: list(dets or [_det(0), _det(100)]))
    return cv


def _live_panel(qtbot, cv):
    panel = CameraPanel(CameraManager(), cv)
    qtbot.addWidget(panel)
    panel._cv_enabled_cb.setChecked(True)
    panel._preview_active = True
    panel._preview.to_frame = lambda x, y: (float(x), float(y))  # identity mapping
    return panel


def _feed(panel):
    frame = _frame()
    dets, tracked, motion = panel._cv_processor.process_frame(frame, 0.0)
    panel._process_frame_on_main_thread(
        FrameData(frame=frame, timestamp=0.0), dets, tracked, motion
    )


def _click(panel, x, y):
    panel._preview.frame_clicked.emit(x, y)


def test_preview_maps_widget_clicks_to_frame_pixels(qtbot):
    preview = CameraPreviewWidget()
    qtbot.addWidget(preview)
    preview.resize(400, 400)
    preview.update_frame(_frame())  # 200x100 shown at 400x200, 100 px bars above/below
    assert preview.to_frame(10, 130) == pytest.approx((5, 15))
    assert preview.to_frame(10, 50) is None  # on the letterbox bar


def test_live_frames_prompt_for_each_name_in_turn(qtbot):
    panel = _live_panel(qtbot, _cv())
    changes = []
    panel.naming_changed.connect(lambda: changes.append(True))
    _feed(panel)
    assert not panel.naming_complete
    assert panel._naming_label.text() == "Click Mouse A"
    assert not panel._naming_bar.isHidden()
    assert panel._naming_done_btn.isHidden()

    _click(panel, 5, 15)
    assert panel._naming_label.text() == "Click Mouse B"
    _click(panel, 105, 15)
    assert not panel._naming_done_btn.isHidden()
    assert panel._naming_done_btn.text() == "Done (other animals unnamed)"
    assert not panel.naming_complete

    panel._naming_done_btn.click()
    assert panel.naming_complete
    assert panel._cv_processor.identity_ready
    assert panel._naming_bar.isHidden()
    assert changes


def test_clicks_after_the_names_are_anonymous(qtbot):
    cv = _cv(dets=[_det(0), _det(100), _det(150)])
    panel = _live_panel(qtbot, cv)
    _feed(panel)
    seeded = []
    real = cv.seed_identity
    cv.seed_identity = lambda p, a: (seeded.append((dict(p), list(a))), real(p, a))[1]
    _click(panel, 5, 15)
    _click(panel, 105, 15)
    _click(panel, 155, 15)
    panel._naming_done_btn.click()
    assert seeded == [({"Mouse A": (5, 15), "Mouse B": (105, 15)}, [(155, 15)])]
    assert panel.naming_complete


def test_a_name_that_found_no_animal_is_asked_again(qtbot):
    panel = _live_panel(qtbot, _cv(dets=[_det(0)]))
    _feed(panel)
    _click(panel, 5, 15)
    _click(panel, 105, 15)
    panel._naming_done_btn.click()
    assert not panel.naming_complete
    assert panel._naming_label.text().endswith("Click Mouse B")
    assert panel._naming_done_btn.isHidden()


def test_appearance_mode_never_asks(qtbot):
    panel = _live_panel(qtbot, _cv(mode="appearance"))
    _feed(panel)
    assert panel._naming_bar.isHidden()
    assert panel.naming_complete


def test_live_behavior_under_identity_needs_only_the_behavior_model(qtbot, tmp_path):
    """The identity path takes keypoint names from CV settings and runs no
    separate pose net, so asking for both blocked Start for nothing."""
    panel = _live_panel(qtbot, _cv(mode="appearance"))
    started = []
    panel._start_animal_behavior = lambda: started.append(True)
    panel._behavior_pkl = tmp_path / "m.pkl"
    panel._update_live_controls_enabled()
    assert panel._live_behavior_btn.isEnabled()
    panel._toggle_live_behavior()
    assert started == [True]


def test_single_animal_live_behavior_still_needs_pose_and_names(qtbot, tmp_path):
    panel = _live_panel(qtbot, _cv(mode="off"))
    panel._behavior_pkl = tmp_path / "m.pkl"
    panel._update_live_controls_enabled()
    assert not panel._live_behavior_btn.isEnabled()
