"""Animal Annotator window and labelling canvas (#224, phase 1)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, QSettings, Qt
from PyQt6.QtWidgets import QApplication

from glider.analysis.behavior.keypoint_schema import KeypointSchema
from glider.gui.animal_annotator.canvas import LabelCanvas
from glider.gui.animal_annotator.history import SKIP, Edit
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
    window.extract(list(videos), per_video, "uniform")
    qtbot.waitUntil(lambda: window._thread is None, timeout=20000)


def test_window_starts_without_a_project(window):
    assert window._project is None
    assert window._stack.currentIndex() == 0
    assert not window._grid.add_btn.isEnabled()
    assert not window._grid.export_btn.isEnabled()


def test_extract_label_and_export(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    assert window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi"), _clip(tmp_path / "b.avi")])
    assert len(window._project.frames) == 8
    assert len(window._project.videos) == 2
    assert f"Pose · {len(MOUSE)} keypoints" in window._grid.summary.text()

    window.show_editor(0)
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
    listed = []
    for name in ("train.txt", "val.txt", "test.txt"):
        if (project.folder / name).exists():
            listed += (project.folder / name).read_text().splitlines()
    assert len(listed) == 7  # eight labelled, one skipped
    assert "validation frame(s)" in shown[0]
    assert "has frames in train and in val/test" not in shown[0]  # whole videos held out
    assert "one video" not in shown[0]


def test_labels_are_saved_as_you_go(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=2)
    window.show_editor(0)
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
    window._grid.remove_video_requested.emit(next(iter(window._project.videos)))
    assert len(window._project.frames) == 2
    assert len(window._project.videos) == 1


def _select(window, rows):
    from PyQt6.QtCore import QItemSelectionModel

    gallery = window._grid.gallery
    for row in rows:
        gallery.selectionModel().select(
            gallery.gallery_model.index(row), QItemSelectionModel.SelectionFlag.Select
        )


def test_move_selection_to_test(qtbot, window, tmp_path):
    from glider.gui.animal_annotator.gallery import GalleryModel

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=4)
    grid = window._grid
    assert grid.selection_bar.isHidden()
    _select(window, [0, 1])
    assert not grid.selection_bar.isHidden()
    rows = grid.gallery.selected_rows()
    assert rows == [0, 1]
    grid.move_requested.emit(rows, "test")
    saved = StudioProject.load(project.folder).frames
    assert saved[0].split == saved[1].split == "test"
    model = grid.gallery.gallery_model
    assert model.data(model.index(0), GalleryModel.SplitRole) == "test"
    tests = sum(f.split == "test" for f in window._project.frames)
    assert grid.split_chips["test"].text().endswith(str(tests))


def test_bulk_skip(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=4)
    window._grid.skip_requested.emit([1, 2], True)
    saved = StudioProject.load(project.folder).frames
    assert saved[1].skipped and saved[2].skipped and not saved[0].skipped
    assert window._grid.status_chips["skipped"].text().endswith("2")
    window._grid.skip_requested.emit([1], False)
    assert not StudioProject.load(project.folder).frames[1].skipped


def test_tiles_do_not_open_while_extracting(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=2)
    window.extract([_clip(tmp_path / "b.avi")], 4, "uniform")
    window._grid.frame_activated.emit(0)
    assert window._stack.currentIndex() == 0
    assert not window._grid.add_btn.isEnabled()
    qtbot.waitUntil(lambda: window._thread is None, timeout=20000)
    assert window._grid.add_btn.isEnabled()
    window._grid.frame_activated.emit(0)
    assert window._stack.currentIndex() == 1


def test_leak_notice(qtbot, window, tmp_path):
    from glider.vision.training.splits import leak_notice

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=4)
    grid = window._grid
    grid.move_requested.emit([0, 1], "train")
    grid.move_requested.emit([2, 3], "val")
    assert grid.notice.isHidden()  # one video: nothing to act on, so nothing said
    _extract(qtbot, window, [_clip(tmp_path / "b.avi")], per_video=2)
    assert not grid.notice.isHidden()
    assert grid.notice.text() == leak_notice(window._project)
    assert grid.notice.text().startswith("a.avi has frames in train and in val/test")
    grid.move_requested.emit([0, 1, 2, 3], "train")
    assert grid.notice.isHidden()


def _png(path, value=128):
    import cv2
    import numpy as np

    image = np.full((48, 64, 3), value, np.uint8)
    cv2.imwrite(str(path), image)
    return path


def test_images_are_added_alongside_videos(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    shots = tmp_path / "shots"
    shots.mkdir()
    images = [_png(shots / "cage1.png", 40), _png(shots / "cage2.jpg", 200)]
    window.extract([_clip(tmp_path / "a.avi")], 2, "uniform", images=images)
    qtbot.waitUntil(lambda: window._thread is None, timeout=20000)
    frames = window._project.frames
    assert sorted(f.source for f in frames if f.source) == ["cage1.png", "cage2.jpg"]
    assert all(window._project.image_path(f).exists() for f in frames)
    key = next(f.video for f in frames if f.source)
    assert window._project.is_image_folder(key)
    # adding the same images again adds nothing
    window.extract([], 2, "uniform", images=images)
    qtbot.waitUntil(lambda: window._thread is None, timeout=20000)
    assert len(window._project.frames) == len(frames)
    reloaded = StudioProject.load(project.folder)
    assert {f.source for f in reloaded.frames} == {"", "cage1.png", "cage2.jpg"}


def test_search_filters_the_grid(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    shots = tmp_path / "shots"
    shots.mkdir()
    window.extract([_clip(tmp_path / "a.avi")], 2, "uniform", images=[_png(shots / "rearing.png")])
    qtbot.waitUntil(lambda: window._thread is None, timeout=20000)
    grid = window._grid
    grid.search.setText("rearing")
    rows = grid.gallery.visible_rows()
    assert [window._project.frames[r].source for r in rows] == ["rearing.png"]
    assert not grid.all_chip.isChecked()
    grid.search.setText("a.avi")
    assert len(grid.gallery.visible_rows()) == 2
    grid.all_chip.click()
    assert grid.search.text() == ""
    assert len(grid.gallery.visible_rows()) == 3


def test_rail_tools_are_named(window):
    names = [b.text() for b in window._mode_buttons.values()]
    assert names == ["Points", "Skeleton"]
    assert [b.text() for b in window._tool_buttons.values()] == ["Polygon", "Auto"]


def test_train_dialog_needs_annotations_and_builds_args(qtbot, tmp_path):
    from glider.gui.animal_annotator.train_dialog import TrainDialog, base_model

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    dialog = TrainDialog(project)
    qtbot.addWidget(dialog)
    assert not dialog._start_btn.isEnabled()
    assert base_model(project, "n") == "yolo11n-pose.pt"
    args = dialog.train_args(tmp_path / "data.yaml", "run1")
    assert args["model"] == "yolo11n-pose.pt"
    assert args["epochs"] == 100 and args["imgsz"] == 640
    assert args["project"] == str(project.folder / "runs")
    assert "device" not in args


def test_open_old_project_migrates_splits(qtbot, window, tmp_path):
    from glider.vision.training.project import FrameRecord

    project = StudioProject.create(tmp_path / "proj", "Old", MOUSE)
    project.videos["v0"] = str(tmp_path / "v0.avi")
    for i in range(3):
        project.frames.append(FrameRecord(f"v0_f{i:06d}", "v0", i))
    project.save()
    assert '"split"' not in (project.folder / "studio.json").read_text()
    assert window.open_project(project.folder)
    assert all(f.split for f in window._project.frames)
    assert all(f.split for f in StudioProject.load(project.folder).frames)


def test_returning_to_grid_reapplies_filter(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=3)
    grid = window._grid
    grid.status_chips["todo"].click()
    assert grid.gallery.visible_rows() == [0, 1, 2]
    window.show_editor(1)
    window._canvas.place(5, 5)
    window.show_grid()
    assert window._stack.currentIndex() == 0
    assert grid.gallery.visible_rows() == [0, 2]
    grid.status_chips["todo"].click()  # clicking the active chip clears it
    assert grid.gallery.visible_rows() == [0, 1, 2]


def test_escape_returns_to_grid_when_nothing_to_cancel(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=2)
    window.show_editor(0)
    window._cancel()
    assert window._stack.currentIndex() == 0


def test_split_combo_moves_the_frame(qtbot, window, tmp_path):
    from glider.gui.animal_annotator.gallery import GalleryModel

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=2)
    window.show_editor(1)
    combo = window._split_combo
    combo.setCurrentIndex(combo.findData("test"))
    assert StudioProject.load(project.folder).frames[1].split == "test"
    assert window._split_badge.text() == "TEST"
    model = window._grid.gallery.gallery_model
    assert model.data(model.index(1), GalleryModel.SplitRole) == "test"
    window.show_editor(0)  # showing another frame never rewrites a split
    assert window._split_combo.currentData() == window._project.frames[0].split
    assert StudioProject.load(project.folder).frames[1].split == "test"


def test_stepping_follows_the_grid_filter(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=4)
    window.show_editor(1)
    window._canvas.place(5, 5)  # frame 1 annotated
    window.show_grid()
    window._grid.status_chips["todo"].click()
    assert window._grid.gallery.visible_rows() == [0, 2, 3]
    window.show_editor(0)
    assert "1 of 3" in window._frame_title.text()
    assert window._filmstrip.visible_rows() == [0, 2, 3]
    window._step_frame(1)
    assert window._frame_index == 2
    assert "2 of 3" in window._frame_title.text()


def test_stepping_from_a_frame_the_filter_now_hides(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=4)
    window._grid.status_chips["todo"].click()
    window.show_editor(1)
    window._canvas.place(5, 5)  # frame 1 no longer matches the filter
    window._grid.gallery.apply_filter()  # even once the grid re-filters
    assert 1 not in window._grid.gallery.visible_rows()
    window._step_frame(1)
    assert window._frame_index == 2
    window.show_editor(1)
    window._grid.gallery.apply_filter()
    window._step_frame(-1)
    assert window._frame_index == 0


def test_question_mark_lists_shortcuts(window):
    from PyQt6.QtWidgets import QLabel

    dialog = window._shortcuts_dialog()
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Undo" in text and "skip" in text


def test_filmstrip_click_opens_that_frame(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=3)
    window.show_editor(0)
    window._filmstrip.frame_activated.emit(2)
    assert window._frame_index == 2
    assert window._filmstrip.currentIndex().row() == 2


def test_add_images_dialog_lists_project_videos_unticked(qtbot, tmp_path):
    from glider.gui.animal_annotator.add_images import AddImagesDialog

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    project.videos["a"] = str(tmp_path / "a.avi")
    dialog = AddImagesDialog(project)
    qtbot.addWidget(dialog)
    assert dialog.videos == [] and dialog.per_video == 20 and dialog.method == "diverse"
    dialog.add_paths([str(tmp_path / "b.avi"), str(tmp_path / "a.avi")])
    assert dialog.videos == [(tmp_path / "b.avi").resolve()]
    dialog._list.item(0).setCheckState(Qt.CheckState.Checked)
    assert dialog.videos == [tmp_path / "a.avi", (tmp_path / "b.avi").resolve()]
    dialog.add_paths([str(tmp_path / "c.PNG")])
    assert dialog.images == [(tmp_path / "c.PNG").resolve()]
    assert len(dialog.videos) == 2


def test_add_images_dialog_hides_image_folders(qtbot, tmp_path):
    from glider.gui.animal_annotator.add_images import AddImagesDialog

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    project.videos["a"] = str(tmp_path / "a.avi")
    project.videos["shots"] = str(tmp_path / "shots")
    dialog = AddImagesDialog(project)
    qtbot.addWidget(dialog)
    assert dialog._list.count() == 1
    assert not dialog._ok.isEnabled()
    dialog.add_paths([str(tmp_path / "x.jpg")])
    assert dialog._ok.isEnabled() and not dialog._per_video.isEnabled()


def test_tools_card_and_menu_open_the_studio(main_window_factory, qtbot):
    main = main_window_factory()
    titles = [card.title for card in main._tools_page.cards()]
    assert "Animal Annotator" in titles
    main._tools_page.tool_chosen.emit("annotator")
    studio = main._animal_annotator_window
    assert studio is not None
    qtbot.addWidget(studio)
    main._open_animal_annotator()
    assert main._animal_annotator_window is studio


def test_several_videos_in_one_extract_do_not_leak(qtbot, window, tmp_path):
    from glider.vision.training.splits import leaky_videos

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / n) for n in ("a.avi", "b.avi", "c.avi")], 6)
    assert leaky_videos(window._project) == []
    assert window._grid.notice.isHidden()


def test_videos_that_yield_no_frames_are_not_left_registered(qtbot, window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video")
    _extract(qtbot, window, [_clip(tmp_path / "a.avi"), bad], 3)
    assert list(window._project.videos) == ["a"]
    assert list(StudioProject.load(project.folder).videos) == ["a"]


def test_file_actions_are_disabled_while_extracting(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    window.extract([_clip(tmp_path / "a.avi")], 4, "uniform")
    assert not any(a.isEnabled() for a in window._file_actions)
    assert window.open_project(project.folder) is False
    qtbot.waitUntil(lambda: window._thread is None, timeout=20000)
    assert all(a.isEnabled() for a in window._file_actions)


def test_split_combo_ignores_the_wheel(qtbot, window, tmp_path):
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QWheelEvent

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], 3)
    window.show_editor(0)
    before = window._project.frames[0].split
    combo = window._split_combo
    pos = QPointF(5, 5)
    event = QWheelEvent(
        pos,
        combo.mapToGlobal(pos),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(combo, event)
    assert window._project.frames[0].split == before
    assert combo.currentData() == before


def test_stale_sam_status_leaves_the_hint_alone(qtbot, window, tmp_path):
    window._sam_request_id = 5
    window._hint.setText("kept")
    window._on_sam_status(4, "Loading…")
    assert window._hint.text() == "kept"
    window._on_sam_status(5, "Segmenting…")
    assert window._hint.text() == "Segmenting…"


def test_removing_a_video_clears_undo(qtbot, window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi"), _clip(tmp_path / "b.avi")], 2)
    window.show_editor(2)
    window._history.push(Edit(2, SKIP, False, True))
    window._refresh_undo_buttons()
    assert window._undo_btn.isEnabled()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window._grid.remove_video_requested.emit("a")
    assert not window._history.can_undo
    assert not window._undo_btn.isEnabled()
