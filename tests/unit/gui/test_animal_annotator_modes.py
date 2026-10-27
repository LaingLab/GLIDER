"""Animal Annotator: tool order, the image gallery, and Skeleton labelling mode."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPointF, QSettings, Qt
from PyQt6.QtWidgets import QMenu

from glider.gui.animal_annotator.canvas import SKELETON, LabelCanvas
from glider.gui.animal_annotator.window import AnimalAnnotatorWindow
from glider.vision.training.labels import MISSING, VISIBLE, PoseInstance
from glider.vision.training.project import StudioProject
from glider.vision.training.templates import builtin_templates

MOUSE_TEMPLATE = builtin_templates()[0]
MOUSE = MOUSE_TEMPLATE.names


def _clip(path, frames=40):
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
    w = AnimalAnnotatorWindow(settings=settings)
    qtbot.addWidget(w)
    w.resize(1200, 800)
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


# ----------------------------------------------------------------------
# Animal Annotator first
# ----------------------------------------------------------------------


def test_animal_annotator_is_the_first_tool(main_window_factory):
    main = main_window_factory()
    assert main._analyze_page.cards()[0].title == "Animal Annotator"
    tools = next(m for m in main.findChildren(QMenu) if m.title().replace("&", "") == "Tools")
    assert tools.actions()[0].text().replace("&", "").startswith("Animal Annotator")


# ----------------------------------------------------------------------
# gallery
# ----------------------------------------------------------------------


def test_gallery_shows_every_frame_four_across(qtbot, window, tmp_path):
    from glider.gui.animal_annotator.gallery import COLUMNS

    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    gallery = window._gallery
    assert gallery.gallery_model.rowCount() == 0
    assert "No images" in gallery.empty_text
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=6)
    assert gallery.gallery_model.rowCount() == 6
    assert COLUMNS == 4
    _extract(qtbot, window, [_clip(tmp_path / "b.avi")], per_video=20)  # enough to scroll
    gallery.doItemsLayout()
    rows = [gallery.visualRect(gallery.gallery_model.index(i)).top() for i in range(5)]
    assert rows[0] == rows[3] < rows[4]  # four on the first row, the fifth wraps


def test_no_project_shows_a_blank_gallery(window):
    assert window._gallery.gallery_model.rowCount() == 0
    assert "No project" in window._gallery.empty_text


def test_gallery_thumbnail_draws_labelled_keypoints(tmp_path):
    from glider.gui.animal_annotator.gallery import render_thumbnail
    from glider.vision.training.frames import extract_frames

    project = StudioProject.create(
        tmp_path / "proj", "T", MOUSE, colours=[(255, 0, 255)] * len(MOUSE)
    )
    (frame,) = extract_frames(project, _clip(tmp_path / "a.avi"), 1, "uniform")
    bare = render_thumbnail(project, frame, width=128).toImage()
    project.write_instances(frame, [PoseInstance([(0.5, 0.5, VISIBLE)] * len(MOUSE))])
    drawn = render_thumbnail(project, frame, width=128).toImage()
    centre = (64, drawn.height() // 2)
    assert drawn.pixelColor(*centre).name() != bare.pixelColor(*centre).name()
    assert drawn.pixelColor(*centre).red() > 200 and drawn.pixelColor(*centre).blue() > 200


def test_double_clicking_a_tile_opens_it_for_labelling(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=4)
    window._gallery.frame_activated.emit(2)
    assert window._tabs.currentIndex() == 1
    assert window._frame_index == 2


def test_a_label_edit_refreshes_its_tile(qtbot, window, tmp_path):
    project = StudioProject.create(tmp_path / "proj", "Test", MOUSE)
    window.open_project(project.folder)
    _extract(qtbot, window, [_clip(tmp_path / "a.avi")], per_video=2)
    model = window._gallery.gallery_model
    model.data(model.index(0), Qt.ItemDataRole.DecorationRole)  # cache it
    window._set_mode("sequential")
    window._label_frame(0)
    with qtbot.waitSignal(model.dataChanged):
        window._canvas.place(5, 5)
    assert model.data(model.index(0), model.StatusRole) == "labelled"


# ----------------------------------------------------------------------
# skeleton mode
# ----------------------------------------------------------------------

IMG_W, IMG_H = 800, 400


@pytest.fixture
def canvas(qtbot):
    c = LabelCanvas(MOUSE, MOUSE_TEMPLATE.colours)
    c.set_layout([(k.x, k.y) for k in MOUSE_TEMPLATE.keypoints])
    c.set_mode(SKELETON)
    qtbot.addWidget(c)
    c.resize(900, 500)
    c.show()
    qtbot.waitExposed(c)
    c.set_frame(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8), [])
    return c


def _v(c, x, y):
    return c.mapFromScene(QPointF(x, y))


def _click(qtbot, c, x, y):
    qtbot.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, pos=_v(c, x, y))


def _drag(qtbot, c, start: QPointF, end: QPointF):
    vp = c.viewport()
    qtbot.mousePress(vp, Qt.MouseButton.LeftButton, pos=c.mapFromScene(start))
    qtbot.mouseMove(vp, c.mapFromScene(end))
    qtbot.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=c.mapFromScene(end))


def _points(c, animal=0):
    return [(x, y) for x, y, v in c._animals[animal] if v != MISSING]


def _span(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


def test_one_click_drops_the_whole_skeleton(qtbot, canvas):
    _click(qtbot, canvas, 400, 200)
    pts = _points(canvas)
    assert len(pts) == len(MOUSE)
    xs = [p[0] for p in pts]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(400, abs=3)
    nose, tail = pts[MOUSE.index("nose")], pts[MOUSE.index("tail_base")]
    assert nose[1] < tail[1]  # in the template's shape: nose above tail
    assert len(canvas.instances()) == 1


def test_clicking_elsewhere_adds_another_animal_at_the_same_size(qtbot, canvas):
    _click(qtbot, canvas, 200, 200)
    canvas.scale_current(1.5)
    first = max(_span(_points(canvas, 0)))
    _click(qtbot, canvas, 650, 200)
    assert canvas.animal_count == 2 and canvas.current_animal == 1
    assert max(_span(_points(canvas, 1))) == pytest.approx(first, rel=0.03)


def test_dragging_a_box_sizes_the_skeleton(qtbot, canvas):
    _drag(qtbot, canvas, QPointF(100, 100), QPointF(300, 300))
    w, h = _span(_points(canvas))
    assert w == pytest.approx(200, abs=6) and h == pytest.approx(200, abs=6)


def test_dragging_inside_the_box_pans_the_skeleton(qtbot, canvas):
    _click(qtbot, canvas, 400, 200)
    before = _points(canvas)
    box = canvas.animal_box(0)
    # Inside the box, away from keypoints and from the corner (scale) handles.
    start = QPointF(box.left() + 4, box.center().y())
    assert canvas.handle_at(canvas.mapFromScene(start)) is None
    assert canvas.point_at(canvas.mapFromScene(start)) is None
    _drag(qtbot, canvas, start, QPointF(start.x() + 60, start.y() + 30))
    for (x0, y0), (x1, y1) in zip(before, _points(canvas), strict=True):
        assert (x1 - x0, y1 - y0) == pytest.approx((60, 30), abs=4)


def test_the_top_handle_tilts_the_skeleton(qtbot, canvas):
    _click(qtbot, canvas, 400, 200)
    before = _points(canvas)
    centre = canvas._centre(0)
    handle = canvas.handles()["rotate"]
    radius = math.hypot(handle.x() - centre.x(), handle.y() - centre.y())
    # A quarter turn clockwise: from straight above the centre to its right.
    _drag(qtbot, canvas, handle, QPointF(centre.x() + radius, centre.y()))
    after = _points(canvas)
    assert before[MOUSE.index("nose")][1] < centre.y()
    assert after[MOUSE.index("nose")][0] > centre.x() + 5  # the nose now points right
    assert _span(after)[0] == pytest.approx(_span(before)[1], abs=6)


def test_a_corner_handle_scales_the_skeleton(qtbot, canvas):
    _click(qtbot, canvas, 400, 200)
    before = max(_span(_points(canvas)))
    centre = canvas._centre(0)
    corner = canvas.handles()["br"]
    out = QPointF(
        centre.x() + (corner.x() - centre.x()) * 1.4,
        centre.y() + (corner.y() - centre.y()) * 1.4,
    )
    _drag(qtbot, canvas, corner, out)
    assert max(_span(_points(canvas))) == pytest.approx(before * 1.4, rel=0.05)


def test_single_keypoints_still_drag_in_skeleton_mode(qtbot, canvas):
    _click(qtbot, canvas, 400, 200)
    nose, tail = MOUSE.index("nose"), MOUSE.index("tail_base")
    x, y = _points(canvas)[nose]
    tail_before = _points(canvas)[tail]
    _drag(qtbot, canvas, QPointF(x, y), QPointF(x + 25, y - 10))
    assert _points(canvas)[nose] == pytest.approx((x + 25, y - 10), abs=2)
    assert _points(canvas)[tail] == pytest.approx(tail_before)


def test_keyboard_tilt_keeps_the_shape(canvas):
    canvas.place_skeleton(400, 200)
    before = _points(canvas)
    canvas.rotate_current(90)
    after = _points(canvas)

    def d(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    assert d(after[0], after[2]) == pytest.approx(d(before[0], before[2]), abs=1)
    assert after != before


def test_mode_is_remembered(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    project = StudioProject.create(tmp_path / "proj", "M", MOUSE)
    first = AnimalAnnotatorWindow(settings=settings)
    qtbot.addWidget(first)
    first.open_project(project.folder)
    first._set_mode("sequential")
    first._toggle_mode()
    assert first._canvas.mode == SKELETON
    second = AnimalAnnotatorWindow(settings=settings)
    qtbot.addWidget(second)
    assert second._mode == SKELETON and second._canvas.mode == SKELETON


def test_new_projects_store_the_template_layout(tmp_path):
    layout = [(k.x, k.y) for k in MOUSE_TEMPLATE.keypoints]
    p = StudioProject.create(tmp_path / "p", "M", MOUSE, layout=layout)
    assert StudioProject.load(p.folder).keypoint_layout[2] == pytest.approx(layout[2])


def test_no_connection_lines_are_drawn(qtbot, canvas):
    from PyQt6.QtWidgets import QGraphicsLineItem

    from glider.gui.animal_annotator.canvas import SEQUENTIAL

    canvas.set_mode(SEQUENTIAL)  # Skeleton mode's tilt-handle stem is a line too
    canvas.place_skeleton(400, 200)
    canvas.set_mode(SEQUENTIAL)
    assert not [i for i in canvas.scene().items() if isinstance(i, QGraphicsLineItem)]
