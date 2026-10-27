"""Animal Annotator: undo and redo (Ctrl+Z / Cmd+Z)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QKeySequence

from glider.gui.animal_annotator.history import LABELS, Edit, History
from glider.gui.animal_annotator.window import AnimalAnnotatorWindow
from glider.vision.training.labels import MISSING
from glider.vision.training.project import StudioProject
from glider.vision.training.templates import builtin_templates

MOUSE = builtin_templates()[0].names


# ----------------------------------------------------------------------
# history
# ----------------------------------------------------------------------


def test_history_undo_redo_and_new_edit_clears_redo():
    h = History()
    assert not h.can_undo and h.undo() is None
    a, b = Edit(0, LABELS, [], [1]), Edit(0, LABELS, [1], [2])
    h.push(a)
    h.push(b)
    assert h.undo() is b and h.can_redo
    assert h.redo() is b
    assert h.undo() is b
    h.push(Edit(1, LABELS, [], [3]))
    assert not h.can_redo


def test_history_is_bounded():
    h = History(limit=3)
    for i in range(5):
        h.push(Edit(i, LABELS, i, i + 1))
    assert [h.undo().frame for _ in range(3)] == [4, 3, 2]
    assert h.undo() is None


# ----------------------------------------------------------------------
# window
# ----------------------------------------------------------------------


def _clip(path, frames=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot open an MJPG writer")
    for i in range(frames):
        frame = np.full((48, 64, 3), 40, dtype=np.uint8)
        cv2.circle(frame, ((i * 3) % 60 + 2, 24), 4, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("animal_annotator/label_mode", "sequential")
    w = AnimalAnnotatorWindow(settings=settings)
    qtbot.addWidget(w)
    w.resize(1100, 700)
    w.show()
    qtbot.waitExposed(w)
    project = StudioProject.create(tmp_path / "proj", "U", MOUSE)
    w.open_project(project.folder)
    w._pending_videos = [_clip(tmp_path / "a.avi")]
    w._refresh_videos()
    w._per_video.setValue(3)
    w._method.setCurrentIndex(w._method.findData("uniform"))
    w._extract()
    qtbot.waitUntil(lambda: w._thread is None, timeout=20000)
    w._label_frame(0)
    return w


def _placed(window, frame_index=None):
    project = window._project
    frame = project.frames[window._frame_index if frame_index is None else frame_index]
    instances = project.read_instances(frame)
    return sum(v != MISSING for inst in instances for _x, _y, v in inst.keypoints)


def test_undo_removes_the_last_point_and_redo_puts_it_back(window):
    assert not window._undo_btn.isEnabled()
    window._canvas.place(5, 5)
    window._canvas.place(10, 10)
    assert _placed(window) == 2 and window._undo_btn.isEnabled()
    window._undo()
    assert _placed(window) == 1
    assert sum(window._canvas.placed()) == 1  # the canvas follows the file
    window._undo()
    assert _placed(window) == 0
    assert not window._project.is_labelled(window._project.frames[0])
    window._redo()
    window._redo()
    assert _placed(window) == 2 and not window._redo_btn.isEnabled()


def test_undo_goes_back_to_the_frame_the_edit_was_on(window):
    window._canvas.place(5, 5)
    window._step_frame(1)
    assert window._frame_index == 1
    window._undo()
    assert window._frame_index == 0
    assert not window._project.is_labelled(window._project.frames[0])


def test_skip_can_be_undone(window):
    window._skip_btn.setChecked(True)
    assert window._project.frames[0].skipped
    window._undo()
    assert not window._project.frames[0].skipped
    assert not window._skip_btn.isChecked()


def test_a_new_edit_after_undo_drops_the_redo(window):
    window._canvas.place(5, 5)
    window._undo()
    window._canvas.place(20, 20)
    assert not window._redo_btn.isEnabled()


def test_ctrl_z_undoes(qtbot, window):
    window._canvas.place(5, 5)
    window._canvas.setFocus()
    key = QKeySequence(QKeySequence.StandardKey.Undo)[0]
    qtbot.keyClick(window._canvas, key.key(), key.keyboardModifiers())
    assert _placed(window) == 0


def test_shift_ctrl_z_redoes(qtbot, window):
    window._canvas.place(5, 5)
    window._undo()
    qtbot.keyClick(
        window._canvas,
        Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert _placed(window) == 1


def test_undo_keeps_the_zoom(window):
    canvas = window._canvas
    canvas.place(5, 5)
    canvas._fitted = False
    canvas.scale(2.0, 2.0)
    zoom = canvas.transform().m11()
    window._undo()
    assert canvas.transform().m11() == pytest.approx(zoom)


def test_opening_another_project_clears_the_history(window, tmp_path):
    window._canvas.place(5, 5)
    other = StudioProject.create(tmp_path / "other", "O", MOUSE)
    window.open_project(other.folder)
    assert not window._history.can_undo
