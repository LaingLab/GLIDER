"""Animal Annotator instance segmentation: the canvas tools and the window's Auto flow."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, QRectF, QSettings, Qt

from glider.gui.animal_annotator.seg_canvas import AUTO, POLYGON, SegmentCanvas
from glider.gui.animal_annotator.window import AnimalAnnotatorWindow, NewProjectDialog
from glider.vision.training.labels import SegInstance
from glider.vision.training.project import TASK_SEGMENT, FrameRecord, StudioProject

W, H = 200, 100
SQUARE = [(0.1, 0.1), (0.4, 0.1), (0.4, 0.5), (0.1, 0.5)]


@pytest.fixture
def canvas(qtbot):
    c = SegmentCanvas(["dam", "pup"])
    qtbot.addWidget(c)
    c.resize(800, 500)
    c.show()
    qtbot.waitExposed(c)
    c.set_frame(np.zeros((H, W, 3), dtype=np.uint8), [])
    return c


def _view(canvas, x: float, y: float) -> QPoint:
    return canvas.mapFromScene(QPointF(x, y))


def _click(qtbot, canvas, x, y, button=Qt.MouseButton.LeftButton, modifier=None):
    kwargs = {"pos": _view(canvas, x, y)}
    if modifier is not None:
        kwargs["modifier"] = modifier
    qtbot.mouseClick(canvas.viewport(), button, **kwargs)


def _drag(qtbot, canvas, start, end):
    vp = canvas.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, pos=_view(canvas, *start))
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    move = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(_view(canvas, *end)),
        QPointF(vp.mapToGlobal(_view(canvas, *end))),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(vp, move)
    qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=_view(canvas, *end))


# ----------------------------------------------------------------------
# Polygon tool
# ----------------------------------------------------------------------


def test_clicking_round_and_back_to_the_start_closes_an_outline(qtbot, canvas):
    changes = []
    canvas.changed.connect(lambda: changes.append(1))
    for x, y in ((20, 20), (80, 20), (80, 60), (20, 60)):
        _click(qtbot, canvas, x, y)
    assert canvas.drafting and canvas.draft_length == 4
    assert canvas.instances() == []  # an unfinished outline is not a label
    _click(qtbot, canvas, 20, 20)  # the first vertex closes it
    assert not canvas.drafting
    (inst,) = canvas.instances()
    assert len(inst.polygon) == 4 and inst.class_id == 0
    assert inst.polygon[1] == pytest.approx((80 / W, 20 / H), abs=0.02)
    assert changes == [1]


def test_enter_closes_and_escape_abandons(qtbot, canvas):
    for x, y in ((20, 20), (80, 20)):
        _click(qtbot, canvas, x, y)
    assert not canvas.commit()  # two vertices can't enclose anything
    _click(qtbot, canvas, 50, 70)
    assert canvas.commit()
    assert canvas.shape_count == 1
    _click(qtbot, canvas, 150, 20)
    canvas.cancel()
    assert not canvas.drafting and canvas.shape_count == 1


def test_backspace_takes_back_the_last_vertex(qtbot, canvas):
    for x, y in ((20, 20), (80, 20), (80, 60)):
        _click(qtbot, canvas, x, y)
    canvas.remove()
    assert canvas.draft_length == 2


def test_dragging_a_vertex_moves_it(qtbot, canvas):
    canvas.set_instances([SegInstance(SQUARE)])
    canvas.select_shape(0)
    _drag(qtbot, canvas, (0.4 * W, 0.1 * H), (0.6 * W, 0.2 * H))
    (inst,) = canvas.instances()
    assert inst.polygon[1] == pytest.approx((0.6, 0.2), abs=0.02)


def test_shift_click_on_an_edge_adds_a_vertex_and_right_click_removes_one(qtbot, canvas):
    canvas.set_instances([SegInstance(SQUARE)])
    canvas.select_shape(0)
    _click(qtbot, canvas, 0.25 * W, 0.1 * H, modifier=Qt.KeyboardModifier.ShiftModifier)
    assert canvas.shape_vertex_count(0) == 5
    _click(qtbot, canvas, 0.25 * W, 0.1 * H, button=Qt.MouseButton.RightButton)
    assert canvas.shape_vertex_count(0) == 4
    # an outline keeps at least three
    _click(qtbot, canvas, 0.1 * W, 0.1 * H, button=Qt.MouseButton.RightButton)
    _click(qtbot, canvas, 0.4 * W, 0.1 * H, button=Qt.MouseButton.RightButton)
    assert canvas.shape_vertex_count(0) == 3


def test_clicking_inside_another_outline_selects_it(qtbot, canvas):
    other = [(0.6, 0.1), (0.9, 0.1), (0.9, 0.5), (0.6, 0.5)]
    canvas.set_instances([SegInstance(SQUARE, 0), SegInstance(other, 1)])
    canvas.select_shape(0)
    _click(qtbot, canvas, 0.75 * W, 0.3 * H)
    assert canvas.current_shape == 1 and canvas.current_class == 1
    assert not canvas.drafting


def test_choosing_a_class_reclasses_the_selected_outline(qtbot, canvas):
    canvas.set_instances([SegInstance(SQUARE, 0)])
    canvas.select_shape(0)
    canvas.set_class(1)
    assert canvas.instances()[0].class_id == 1


def test_choosing_a_class_after_finishing_an_outline_leaves_it_alone(qtbot, canvas):
    """Outline a dam, pick "pup" for the next animal: the dam stays a dam."""
    for x, y in ((20, 20), (80, 20), (80, 60)):
        canvas.add_vertex(x, y)
    canvas.commit()
    canvas.set_class(1)
    assert canvas.instances()[0].class_id == 0
    assert canvas.current_class == 1
    canvas.select_shape(0)  # picked on purpose: now it is re-classed
    canvas.set_class(1)
    assert canvas.instances()[0].class_id == 1


def test_delete_with_nothing_else_pending_removes_the_outline(qtbot, canvas):
    canvas.set_instances([SegInstance(SQUARE, 0)])
    canvas.select_shape(0)
    canvas.remove()
    assert canvas.shape_count == 0


# ----------------------------------------------------------------------
# Auto tool (the canvas half: prompts in, proposal out)
# ----------------------------------------------------------------------


def test_auto_clicks_become_prompts(qtbot, canvas):
    canvas.set_tool(AUTO)
    asked = []
    canvas.prompt_changed.connect(lambda: asked.append(canvas.prompts()))
    _click(qtbot, canvas, 50, 50)
    _click(qtbot, canvas, 120, 50, button=Qt.MouseButton.RightButton)
    points, labels, box = asked[-1]
    assert labels == [1, 0] and box is None
    assert points[0] == pytest.approx((50, 50), abs=1.5)


def test_auto_drag_is_a_box_prompt(qtbot, canvas):
    canvas.set_tool(AUTO)
    _drag(qtbot, canvas, (20, 20), (90, 70))
    points, _labels, box = canvas.prompts()
    assert points == []
    assert box == pytest.approx((20, 20, 90, 70), abs=1.5)


def test_accepting_a_proposal_makes_an_outline_and_clears_the_prompts(qtbot, canvas):
    canvas.set_tool(AUTO)
    canvas.set_class(1)
    canvas.add_prompt(50, 50)
    canvas.set_proposal([(20, 20), (80, 20), (80, 60), (20, 60)])
    assert canvas.has_proposal
    assert canvas.commit()
    (inst,) = canvas.instances()
    assert inst.class_id == 1 and len(inst.polygon) == 4
    assert not canvas.has_prompts and not canvas.has_proposal


def test_switching_tools_drops_pending_work(qtbot, canvas):
    canvas.set_tool(AUTO)
    canvas.set_prompt_box(QRectF(10, 10, 50, 50))
    canvas.set_tool(POLYGON)
    assert not canvas.has_prompts


# ----------------------------------------------------------------------
# window
# ----------------------------------------------------------------------


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = AnimalAnnotatorWindow(settings=settings)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    yield w
    w.close()


def _seg_project(tmp_path, frames: int = 2) -> StudioProject:
    project = StudioProject.create(
        tmp_path / "seg", "Pups", [], task=TASK_SEGMENT, class_names=["dam", "pup"]
    )
    for i in range(frames):
        frame = FrameRecord(f"v{i % 2}_f{i:06d}", f"v{i % 2}", i)
        image = np.zeros((H, W, 3), np.uint8)
        cv2.rectangle(image, (40, 30), (120, 70), (255, 255, 255), -1)
        cv2.imwrite(str(project.image_path(frame)), image)
        project.frames.append(frame)
    project.save()
    return project


class _FakeSegmenter:
    """SAM stand-in: the mask is the white rectangle in every test frame."""

    instances = []

    def __init__(self, model):
        self.model = model
        self.image_key = None
        self._image = None
        _FakeSegmenter.instances.append(self)

    def load(self):
        pass

    def set_image(self, key, bgr):
        self.image_key, self._image = key, bgr

    def segment(self, points, labels, box):
        return self._image[:, :, 0] > 127


def test_new_project_dialog_offers_segmentation(qtbot, tmp_path):
    dialog = NewProjectDialog(tmp_path)
    qtbot.addWidget(dialog)
    dialog.set_task(TASK_SEGMENT)
    dialog._classes.setText("dam, pup ,  ")
    assert dialog.task == TASK_SEGMENT
    assert dialog.class_names == ["dam", "pup"]
    assert not dialog._form.isRowVisible(dialog._skeleton_row)
    dialog.set_task("pose")
    assert dialog.class_names == ["animal"]
    assert dialog._form.isRowVisible(dialog._skeleton_row)


def test_segment_project_shows_the_segmentation_controls(window, tmp_path):
    project = _seg_project(tmp_path)
    assert window.open_project(project.folder)
    window._tabs.setCurrentIndex(1)
    assert isinstance(window._canvas, SegmentCanvas)
    assert not window._seg_panel.isHidden() and window._pose_panel.isHidden()
    assert window._skeleton_btn.isHidden()
    assert window._class_list.count() == 2


def test_polygon_outline_is_saved_and_undone(qtbot, window, tmp_path):
    project = _seg_project(tmp_path)
    window.open_project(project.folder)
    window._tabs.setCurrentIndex(1)
    window._set_tool(POLYGON)
    canvas = window._canvas
    for x, y in ((20, 20), (80, 20), (80, 60)):
        canvas.add_vertex(x, y)
    window._commit()
    frame = window._project.frames[window._frame_index]
    (saved,) = StudioProject.load(project.folder).read_instances(frame)
    assert len(saved.polygon) == 3
    assert window._shape_list.count() == 1
    window._undo()
    assert not project.is_labelled(frame)
    assert canvas.shape_count == 0


def test_auto_tool_runs_sam_and_accepts_its_outline(qtbot, window, tmp_path, monkeypatch):
    import glider.vision.training.sam as sam

    _FakeSegmenter.instances = []
    monkeypatch.setattr(sam, "SamSegmenter", _FakeSegmenter)
    project = _seg_project(tmp_path)
    window.open_project(project.folder)
    window._tabs.setCurrentIndex(1)
    window._set_tool(AUTO)
    assert not window._auto_bar.isHidden()
    canvas = window._canvas
    window._class_list.setCurrentRow(1)
    window._on_class_clicked(window._class_list.item(1))

    canvas.add_prompt(80, 50)
    qtbot.waitUntil(lambda: canvas.has_proposal, timeout=10000)
    xs = [p[0] for p in canvas._proposal]
    assert min(xs) == pytest.approx(40, abs=2) and max(xs) == pytest.approx(120, abs=2)
    assert window._accept_btn.isEnabled()

    window._commit()
    frame = window._project.frames[window._frame_index]
    (saved,) = StudioProject.load(project.folder).read_instances(frame)
    assert saved.class_id == 1
    assert not canvas.has_prompts

    # The frame was encoded once; a second click on it reuses that.
    canvas.add_prompt(60, 40)
    qtbot.waitUntil(lambda: canvas.has_proposal, timeout=10000)
    assert len(_FakeSegmenter.instances) == 1


def test_a_sam_answer_for_another_frame_is_ignored(qtbot, window, tmp_path):
    project = _seg_project(tmp_path)
    window.open_project(project.folder)
    window._tabs.setCurrentIndex(1)
    window._set_tool(AUTO)
    stale = window._sam_request_id
    window._step_frame(1)  # moving on invalidates anything in flight
    window._on_sam_result(stale, [(1, 1), (5, 1), (5, 5)], "")
    assert not window._canvas.has_proposal


def test_sam_errors_are_reported_not_raised(qtbot, window, tmp_path):
    project = _seg_project(tmp_path)
    window.open_project(project.folder)
    window._tabs.setCurrentIndex(1)
    window._set_tool(AUTO)
    window._on_sam_result(window._sam_request_id, None, "no network")
    assert "no network" in window._sam_status.text()


def test_pose_projects_are_unchanged(window, tmp_path):
    from glider.analysis.behavior.keypoint_schema import KeypointSchema
    from glider.gui.animal_annotator.canvas import LabelCanvas

    project = StudioProject.create(tmp_path / "pose", "P", KeypointSchema.default_mouse().names)
    window.open_project(project.folder)
    assert isinstance(window._canvas, LabelCanvas)
    assert window._seg_panel.isHidden() and not window._pose_panel.isHidden()
    assert window._auto_bar.isHidden()
