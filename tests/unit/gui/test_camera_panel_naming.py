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


def test_a_click_on_empty_floor_asks_again(qtbot):
    panel = _live_panel(qtbot, _cv())
    _feed(panel)
    _click(panel, 50, 80)  # nowhere near either animal
    assert panel._naming_label.text() == "No animal there \u2014 click on Mouse A"
    assert panel._placements == {}
    _click(panel, 6, 14)  # inside Mouse A's box
    assert panel._naming_label.text() == "Click Mouse B"


def test_one_animal_cannot_take_two_names(qtbot):
    panel = _live_panel(qtbot, _cv())
    _feed(panel)
    _click(panel, 5, 15)
    _click(panel, 6, 16)  # Mouse A again
    assert panel._naming_label.text() == "No animal there \u2014 click on Mouse B"


def test_animals_that_cross_before_done_keep_their_clicked_names(qtbot):
    """Each click binds when it is made and follows its animal frame by frame,
    so the names do not swap when the animals trade places before Done."""

    def at(ax, bx):
        return [_det(ax, 10), _det(bx, 50)]  # Mouse A on the top row, B below

    cv = _cv()
    panel = _live_panel(qtbot, cv)
    cv._detect_yolo.side_effect = lambda *_a: at(0, 100)
    _feed(panel)
    _click(panel, 5, 15)  # Mouse A
    _click(panel, 105, 55)  # Mouse B
    for step in range(1, 6):
        cv._detect_yolo.side_effect = lambda *_a, s=step: at(20 * s, 100 - 20 * s)
        _feed(panel)
    panel._naming_done_btn.click()
    assert panel.naming_complete
    _, tracked, _ = cv.process_frame(_frame(), 1.0)
    names = {t.animal: t.centroid for t in tracked}
    assert names == {"Mouse A": (105, 15), "Mouse B": (5, 55)}


def test_tentative_names_are_drawn_while_naming(qtbot):
    panel = _live_panel(qtbot, _cv())
    _feed(panel)
    _click(panel, 5, 15)
    assert panel._preview._naming_marks == [("Mouse A", (5, 15))]
    _click(panel, 105, 15)
    panel._naming_done_btn.click()
    assert panel._preview._naming_marks == []


def test_a_finished_run_names_the_animals_again(qtbot):
    from types import SimpleNamespace

    from glider.core.experiment_session import FlowConfig, NodeConfig
    from glider.gui.runner.readiness import compute_readiness

    cv = _cv()
    panel = _live_panel(qtbot, cv)
    _feed(panel)
    _click(panel, 5, 15)
    _click(panel, 105, 15)
    panel._naming_done_btn.click()
    panel.on_experiment_state("RUNNING")
    panel.on_experiment_state("PAUSED")
    assert cv.identity_ready  # a pause is the same trial
    panel.on_experiment_state("RUNNING")
    panel.on_experiment_state("STOPPING")
    assert not cv.identity_ready
    core = SimpleNamespace(
        session=SimpleNamespace(
            flow=FlowConfig(nodes=[NodeConfig(id="0", node_type="StartExperiment")]),
            metadata=SimpleNamespace(name=""),
        ),
        hardware_manager=SimpleNamespace(
            is_any_board_connected=lambda: True, connected_board_description=lambda: ""
        ),
        cv_processor=cv,
    )
    assert compute_readiness(core).identity_message == "Name the animals on the camera view"
    _feed(panel)
    assert panel._naming_label.text() == "Click Mouse A"


def test_name_at_start_turns_computer_vision_on(qtbot):
    panel = CameraPanel(CameraManager(), _cv())
    qtbot.addWidget(panel)
    assert panel._cv_enabled_cb.isChecked()


def test_other_modes_leave_computer_vision_alone(qtbot):
    panel = CameraPanel(CameraManager(), _cv(mode="appearance"))
    qtbot.addWidget(panel)
    assert not panel._cv_enabled_cb.isChecked()


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
