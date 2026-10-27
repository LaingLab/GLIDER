"""Animal Annotator window and labelling canvas (#224, phase 1)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, QSettings, Qt

from glider.analysis.behavior.keypoint_schema import KeypointSchema
from glider.gui.animal_annotator.canvas import LabelCanvas
from glider.gui.animal_annotator.window import AnimalAnnotatorWindow
from glider.vision.training.labels import HIDDEN, MISSING, VISIBLE, PoseInstance
from glider.vision.training.project import StudioProject

MOUSE = KeypointSchema.default_mouse().names
W, H = 200, 100


@pytest.fixture
def canvas(qtbot):
    c = LabelCanvas(MOUSE)
    qtbot.addWidget(c)
    c.resize(800, 500)
    c.show()
    qtbot.waitExposed(c)
    c.set_frame(np.zeros((H, W, 3), dtype=np.uint8), [])
    return c


def _view(canvas: LabelCanvas, x: float, y: float) -> QPoint:
    return canvas.mapFromScene(QPointF(x, y))


def _click(qtbot, canvas, x, y, button=Qt.MouseButton.LeftButton):
    qtbot.mouseClick(canvas.viewport(), button, pos=_view(canvas, x, y))


# ----------------------------------------------------------------------
# canvas
# ----------------------------------------------------------------------


def test_clicks_place_keypoints_in_schema_order(qtbot, canvas):
    assert canvas.current_keypoint == 0
    _click(qtbot, canvas, 50, 50)
    _click(qtbot, canvas, 60, 50)
    assert canvas.current_keypoint == 2
    (inst,) = canvas.instances()
    assert inst.keypoints[0][:2] == pytest.approx((50 / W, 50 / H), abs=0.02)
    assert inst.keypoints[1][2] == VISIBLE
    assert inst.keypoints[2][2] == MISSING


def test_placing_every_point_ends_the_sequence(qtbot, canvas):
    for i in range(len(MOUSE)):
        _click(qtbot, canvas, 20 + i * 20, 50)
    assert canvas.current_keypoint == -1
    _click(qtbot, canvas, 190, 90)  # nothing left to place: ignored
    assert len(canvas.instances()) == 1


def test_dragging_a_point_moves_it(qtbot, canvas):
    _click(qtbot, canvas, 50, 50)
    vp = canvas.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, pos=_view(canvas, 50, 50))
    qtbot.mouseMove(vp, _view(canvas, 120, 70))
    with qtbot.waitSignal(canvas.changed):
        qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=_view(canvas, 120, 70))
    (inst,) = canvas.instances()
    assert inst.keypoints[0][:2] == pytest.approx((120 / W, 70 / H), abs=0.02)
    assert inst.keypoints[1][2] == MISSING  # the drag placed nothing new


def test_right_click_toggles_hidden(qtbot, canvas):
    _click(qtbot, canvas, 50, 50)
    _click(qtbot, canvas, 50, 50, Qt.MouseButton.RightButton)
    assert canvas.instances()[0].keypoints[0][2] == HIDDEN
    _click(qtbot, canvas, 50, 50, Qt.MouseButton.RightButton)
    assert canvas.instances()[0].keypoints[0][2] == VISIBLE


def test_remove_point_then_click_replaces_it(qtbot, canvas):
    _click(qtbot, canvas, 50, 50)
    _click(qtbot, canvas, 60, 50)
    canvas.remove_selected_point()  # the last pressed: keypoint 1
    assert canvas.current_keypoint == 1
    _click(qtbot, canvas, 150, 20)
    assert canvas.instances()[0].keypoints[1][:2] == pytest.approx((0.75, 0.2), abs=0.02)


def test_choosing_a_keypoint_lets_a_click_move_it(qtbot, canvas):
    _click(qtbot, canvas, 50, 50)
    canvas.set_current_keypoint(0)
    _click(qtbot, canvas, 150, 80)
    assert canvas.instances()[0].keypoints[0][:2] == pytest.approx((0.75, 0.8), abs=0.02)


def test_several_animals(qtbot, canvas):
    _click(qtbot, canvas, 30, 30)
    canvas.new_animal()
    assert canvas.animal_count == 2 and canvas.current_animal == 1
    _click(qtbot, canvas, 170, 70)
    assert len(canvas.instances()) == 2
    canvas.new_animal()
    canvas.new_animal()  # an empty animal is not stacked on another empty one
    assert canvas.animal_count == 3
    canvas.delete_animal()
    assert canvas.animal_count == 2


def test_pressing_another_animals_point_selects_it(qtbot, canvas):
    _click(qtbot, canvas, 30, 30)
    canvas.new_animal()
    _click(qtbot, canvas, 170, 70)
    _click(qtbot, canvas, 30, 30)
    assert canvas.current_animal == 0


def test_existing_labels_are_shown(qtbot, canvas):
    inst = PoseInstance([(0.5, 0.5, VISIBLE)] + [(0.0, 0.0, MISSING)] * 6)
    canvas.set_frame(np.zeros((H, W, 3), dtype=np.uint8), [inst])
    assert canvas.placed()[0] and not canvas.placed()[1]
    assert canvas.current_keypoint == 1
    assert canvas.instances()[0].keypoints[0][:2] == pytest.approx((0.5, 0.5))


def test_clicks_off_the_image_place_nothing(qtbot, canvas):
    assert not canvas.place(-5, 10)
    assert not canvas.place(W + 5, 10)
    assert canvas.instances() == []


# ----------------------------------------------------------------------
# window
# ----------------------------------------------------------------------


def _clip(path, frames=40):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot open an MJPG writer")
    for i in range(frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.circle(frame, ((i * 3) % 60 + 2, 24), 4, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = AnimalAnnotatorWindow(settings=settings)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    return w


def _extract(qtbot, window, videos, per_video=4):
    window._pending_videos = list(videos)
    window._refresh_videos()
    window._per_video.setValue(per_video)
    window._method.setCurrentIndex(window._method.findData("uniform"))
    window._extract()
    qtbot.waitUntil(lambda: window._thread is None, timeout=20000)


def test_window_starts_without_a_project(window):
    assert window._project is None
    assert not window._extract_btn.isEnabled()
    assert not window._tabs.isTabEnabled(1)


def test_extract_label_and_export(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    assert window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi"), _clip(tmp_path / "b.avi")])
    assert len(window._project.frames) == 8
    assert window._video_list.count() == 2

    window._tabs.setCurrentIndex(1)
    assert window._frame_index == 0
    canvas = window._canvas
    for _ in range(len(window._project.frames)):
        canvas.place(10, 10)
        canvas.place(20, 20)
        window._step_frame(1)
    counts = window._project.counts()
    assert counts["labelled"] == 8

    window._skip_btn.setChecked(True)  # on the last frame
    assert window._project.counts()["skipped"] == 1
    assert StudioProject.load(project.folder).frames[-1].skipped

    from PyQt6.QtWidgets import QMessageBox

    shown = []
    qtbot_patch = pytest.MonkeyPatch()
    qtbot_patch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    try:
        window._export()
    finally:
        qtbot_patch.undo()
    assert (project.folder / "data.yaml").exists()
    listed = (project.folder / "train.txt").read_text().splitlines()
    listed += (project.folder / "val.txt").read_text().splitlines()
    assert len(listed) == 7  # eight labelled, one skipped
    assert "validation frame(s)" in shown[0]
    assert "one video" not in shown[0]  # two videos: split by video, no warning


def test_labels_are_saved_as_you_go(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=2)
    window._tabs.setCurrentIndex(1)
    window._canvas.place(5, 5)
    frame = window._project.frames[window._frame_index]
    saved = StudioProject.load(project.folder).read_instances(frame)
    assert saved and saved[0].keypoints[0][2] == VISIBLE


def test_reopens_the_last_project(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    project = StudioProject.create(tmp_path / "proj", "Again", MOUSE)
    first = AnimalAnnotatorWindow(settings=settings)
    qtbot.addWidget(first)
    first.open_project(project.folder)
    second = AnimalAnnotatorWindow(settings=settings)
    qtbot.addWidget(second)
    assert second._project is not None and second._project.name == "Again"


def test_removing_a_video_drops_its_frames(qtbot, window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi"), _clip(tmp_path / "b.avi")], per_video=2)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window._video_list.item(0).setSelected(True)
    window._remove_videos()
    assert len(window._project.frames) == 2
    assert len(window._project.videos) == 1


def test_analyze_card_and_menu_open_the_studio(main_window_factory, qtbot):
    main = main_window_factory()
    titles = [card.title for card in main._analyze_page.cards()]
    assert "Animal Annotator" in titles
    main._analyze_page.tool_chosen.emit("annotator")
    studio = main._animal_annotator_window
    assert studio is not None
    qtbot.addWidget(studio)
    main._open_animal_annotator()
    assert main._animal_annotator_window is studio
