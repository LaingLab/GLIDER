"""Camera-panel wiring for the pose-tracking backend.

Choosing a SLEAP/DeepLabCut model in the panel now also points *tracking* at
it, not only the live behaviour classifier that already consumed it. See
docs/superpowers/specs/2026-08-25-pose-tracking-backend-design.md, section 6.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

NAMES = ["snout", "left_ear", "right_ear"]


def _make_panel(qtbot):
    from glider.gui.panels.camera_panel import CameraPanel
    from glider.vision.camera_manager import CameraManager
    from glider.vision.cv_processor import CVProcessor

    panel = CameraPanel(CameraManager(), CVProcessor())
    qtbot.addWidget(panel)
    panel.show()
    return panel


@pytest.fixture
def dlc_folder(tmp_path, monkeypatch):
    root = tmp_path / "exp_dlc"
    root.mkdir()
    (root / "glider_pose.json").write_text(
        json.dumps(
            {
                "kind": "dlc",
                "onnx": "model.onnx",
                "keypoint_names": NAMES,
                "output_stride": 8.0,
                "locref_stdev": 7.2831,
            }
        )
    )
    (root / "model.onnx").write_bytes(b"stub")

    # _apply_pose_model now really tries to load this model (POSE_MODEL
    # backend); ``model.onnx`` is a stub, not a parseable ONNX graph, so stand
    # in for onnxruntime the way the pose backend's own tests do.
    from glider.vision.pose import backend as pose_backend_mod

    monkeypatch.setattr(pose_backend_mod, "_make_session", lambda spec: object())
    return root


def test_selecting_a_pose_model_points_tracking_at_the_resolved_onnx(qtbot, dlc_folder):
    from glider.vision.cv_processor import DetectionBackend

    panel = _make_panel(qtbot)
    panel._apply_pose_model(dlc_folder)

    settings = panel._cv_processor.settings
    assert settings.backend == DetectionBackend.POSE_MODEL
    # The resolved model.onnx -- not the folder _pose_model_path keeps for the
    # (separate) live behaviour classifier.
    assert settings.model_path == str(dlc_folder / "model.onnx")
    assert settings.keypoint_names == NAMES


def test_selecting_a_pose_model_calls_update_settings(qtbot, dlc_folder, monkeypatch):
    panel = _make_panel(qtbot)

    calls = []
    monkeypatch.setattr(panel._cv_processor, "update_settings", calls.append)

    panel._apply_pose_model(dlc_folder)

    assert len(calls) == 1


def test_selecting_a_yolo_pt_does_not_switch_the_backend(qtbot, tmp_path, monkeypatch):
    """load_pose_backend needs caller-supplied names for YOLO; the panel does
    not prompt for them here, so a .pt selection leaves CV settings alone."""
    from glider.vision.cv_processor import DetectionBackend

    panel = _make_panel(qtbot)
    calls = []
    monkeypatch.setattr(panel._cv_processor, "update_settings", calls.append)

    pt = tmp_path / "best.pt"
    pt.write_bytes(b"stub")
    panel._apply_pose_model(pt)

    assert calls == []
    assert panel._cv_processor.settings.backend == DetectionBackend.BACKGROUND_SUBTRACTION
    assert panel._cv_processor.settings.model_path is None


def test_settings_dialog_can_still_override_the_panel_pick(qtbot, dlc_folder):
    """The panel picker is a convenience, not a lock -- Settings can override it."""
    from glider.vision.cv_processor import DetectionBackend

    panel = _make_panel(qtbot)
    panel._apply_pose_model(dlc_folder)
    assert panel._cv_processor.settings.backend == DetectionBackend.POSE_MODEL

    overridden = panel._cv_processor.settings.copy()
    overridden.backend = DetectionBackend.BACKGROUND_SUBTRACTION
    panel._cv_processor.update_settings(overridden)

    assert panel._cv_processor.settings.backend == DetectionBackend.BACKGROUND_SUBTRACTION


def test_picking_a_second_pose_model_reloads_with_its_own_names(qtbot, tmp_path, monkeypatch):
    """A later pick's names must win, not linger from the first model."""
    from glider.vision.pose import backend as pose_backend_mod

    monkeypatch.setattr(pose_backend_mod, "_make_session", lambda spec: object())

    first = tmp_path / "first_dlc"
    first.mkdir()
    (first / "glider_pose.json").write_text(
        json.dumps({"kind": "dlc", "onnx": "model.onnx", "keypoint_names": ["a", "b"]})
    )
    (first / "model.onnx").write_bytes(b"stub")

    second = tmp_path / "second_sleap"
    second.mkdir()
    (second / "glider_pose.json").write_text(
        json.dumps({"kind": "sleap", "onnx": "model.onnx", "keypoint_names": ["x", "y", "z"]})
    )
    (second / "model.onnx").write_bytes(b"stub")

    panel = _make_panel(qtbot)
    panel._apply_pose_model(Path(first))
    assert panel._cv_processor.settings.keypoint_names == ["a", "b"]

    panel._apply_pose_model(Path(second))
    assert panel._cv_processor.settings.keypoint_names == ["x", "y", "z"]
    assert panel._cv_processor.settings.model_path == str(second / "model.onnx")
