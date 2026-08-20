"""Dragging models and videos onto the camera panel.

Drops are synthesised as real ``QDropEvent``s over ``QMimeData`` local-file
URLs, so these exercise the same path a mouse drag takes — not just the pure
``classify_drop`` helper, which has its own tests in test_model_drop.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


def _mime(paths):
    from PyQt6.QtCore import QMimeData, QUrl

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def _drop(panel, paths):
    """Send a real QDropEvent carrying *paths* to the panel.

    ``mime`` is bound to a local on purpose: QDropEvent does not take ownership
    of its QMimeData, so passing the constructor a temporary lets Python free it
    immediately and the event is left pointing at freed memory — which crashes
    the interpreter with an access violation rather than failing a test.
    """
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QDropEvent

    mime = _mime(paths)
    event = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    panel.dropEvent(event)
    return event


def _drag_enter(panel, paths):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtGui import QDragEnterEvent

    mime = _mime(paths)  # see _drop: must outlive the event
    event = QDragEnterEvent(
        QPoint(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    panel.dragEnterEvent(event)
    return event


@pytest.fixture
def dlc_folder(tmp_path):
    root = tmp_path / "exp6_dlc"
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
    return root


def test_panel_accepts_drops(qtbot):
    panel = _make_panel(qtbot)
    assert panel.acceptDrops() is True


def test_dropping_a_dlc_folder_fills_and_locks_the_names(qtbot, dlc_folder):
    panel = _make_panel(qtbot)
    _drop(panel, [dlc_folder])

    assert panel._pose_model_path == dlc_folder
    assert panel._kp_names_edit.text() == ", ".join(NAMES)
    # Locked: the model's training order is authoritative, and it is precisely
    # what an operator cannot reliably retype.
    assert panel._kp_names_edit.isReadOnly() is True
    assert "DeepLabCut" in panel._pose_model_label.text()
    assert "3 kp" in panel._pose_model_label.text()


def test_dropping_a_pt_leaves_the_names_editable(qtbot, tmp_path):
    panel = _make_panel(qtbot)
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"stub")

    _drop(panel, [pt])

    assert panel._pose_model_path == pt
    assert panel._kp_names_edit.isReadOnly() is False
    assert "YOLO" in panel._pose_model_label.text()


def test_a_pt_after_a_dlc_folder_restores_editability(qtbot, tmp_path, dlc_folder):
    panel = _make_panel(qtbot)
    _drop(panel, [dlc_folder])
    assert panel._kp_names_edit.isReadOnly() is True

    pt = tmp_path / "best.pt"
    pt.write_bytes(b"stub")
    _drop(panel, [pt])

    assert panel._kp_names_edit.isReadOnly() is False


def test_dropping_a_behavior_model(qtbot, tmp_path):
    panel = _make_panel(qtbot)
    pkl = tmp_path / "behaviour.pkl"
    pkl.write_bytes(b"stub")

    _drop(panel, [pkl])

    assert panel._behavior_pkl == pkl
    assert "behaviour.pkl" in panel._behavior_model_label.text()


def test_dropping_a_video_loads_it(qtbot, tmp_path, monkeypatch):
    panel = _make_panel(qtbot)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"stub")

    loaded = {}
    monkeypatch.setattr(panel._video_source, "load", lambda p: loaded.setdefault("path", p) or True)
    monkeypatch.setattr(type(panel._video_source), "frame_count", property(lambda self: 10))
    monkeypatch.setattr(panel, "_on_seek", lambda n: None)

    _drop(panel, [clip])

    assert loaded["path"] == str(clip)


def test_one_drop_fills_several_slots(qtbot, tmp_path, dlc_folder):
    panel = _make_panel(qtbot)
    pkl = tmp_path / "behaviour.pkl"
    pkl.write_bytes(b"stub")

    _drop(panel, [dlc_folder, pkl])

    assert panel._pose_model_path == dlc_folder
    assert panel._behavior_pkl == pkl


def test_second_file_of_a_kind_is_ignored(qtbot, tmp_path):
    panel = _make_panel(qtbot)
    first = tmp_path / "first.pkl"
    second = tmp_path / "second.pkl"
    first.write_bytes(b"x")
    second.write_bytes(b"x")

    _drop(panel, [first, second])

    assert panel._behavior_pkl == first


def test_unknown_file_changes_nothing(qtbot, tmp_path):
    panel = _make_panel(qtbot)
    junk = tmp_path / "notes.txt"
    junk.write_text("hi")

    _drop(panel, [junk])

    assert panel._pose_model_path is None
    assert panel._behavior_pkl is None


def test_drop_while_running_changes_no_state(qtbot, tmp_path, dlc_folder):
    panel = _make_panel(qtbot)
    panel._behavior_running = True

    _drop(panel, [dlc_folder])

    assert panel._pose_model_path is None
    assert panel._kp_names_edit.text() == ""


def test_a_folder_without_a_sidecar_warns_and_keeps_state(qtbot, tmp_path, monkeypatch):
    panel = _make_panel(qtbot)
    root = tmp_path / "bare"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"stub")

    warned: list = []
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a) or None)
    )

    _drop(panel, [root])

    assert warned, "the operator must be told the sidecar is missing"
    assert "export_pose_onnx" in warned[0][2]
    assert panel._pose_model_path is None


def test_drag_enter_accepts_a_known_kind(qtbot, dlc_folder):
    panel = _make_panel(qtbot)
    event = _drag_enter(panel, [dlc_folder])
    assert event.isAccepted() is True
    assert panel.property("dropActive") == "true"


def test_drag_enter_ignores_unknown_kinds(qtbot, tmp_path):
    panel = _make_panel(qtbot)
    junk = tmp_path / "notes.txt"
    junk.write_text("hi")
    event = _drag_enter(panel, [junk])
    assert event.isAccepted() is False


def test_drag_leave_clears_the_highlight(qtbot, dlc_folder):
    from PyQt6.QtGui import QDragLeaveEvent

    panel = _make_panel(qtbot)
    _drag_enter(panel, [dlc_folder])
    panel.dragLeaveEvent(QDragLeaveEvent())
    assert panel.property("dropActive") == "false"


def test_picker_and_drop_share_one_path(qtbot, dlc_folder):
    """The Browse… dialog and a drag must land in exactly the same state."""
    dropped = _make_panel(qtbot)
    _drop(dropped, [dlc_folder])

    picked = _make_panel(qtbot)
    picked._apply_pose_model(Path(dlc_folder))

    assert picked._pose_model_path == dropped._pose_model_path
    assert picked._kp_names_edit.text() == dropped._kp_names_edit.text()
    assert picked._pose_model_label.text() == dropped._pose_model_label.text()


def test_np_import_is_used():
    # Keeps the numpy import honest if this file grows frame-level tests.
    assert np.zeros(1).shape == (1,)
