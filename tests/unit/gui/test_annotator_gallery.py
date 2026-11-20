"""Image grid: tile data, filtering, click vs. select, reflow, filmstrip."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt

from glider.gui.animal_annotator.gallery import GalleryModel, GalleryView
from glider.vision.training.frames import extract_frames
from glider.vision.training.labels import VISIBLE, PoseInstance
from glider.vision.training.project import StudioProject

NAMES = ["nose", "tail"]
CTRL = Qt.KeyboardModifier.ControlModifier


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
def project(tmp_path):
    p = StudioProject.create(tmp_path / "proj", "T", NAMES)
    extract_frames(p, _clip(tmp_path / "a.avi"), 6, "uniform")
    extract_frames(p, _clip(tmp_path / "b.avi"), 6, "uniform")
    return p


@pytest.fixture
def gallery(qtbot, project):
    g = GalleryView()
    qtbot.addWidget(g)
    g.resize(900, 600)
    g.set_project(project)
    g.show()
    qtbot.waitExposed(g)
    return g


def _label(project, gallery, row):
    project.write_instances(project.frames[row], [PoseInstance([(0.5, 0.5, VISIBLE)] * 2)])
    gallery.gallery_model.invalidate(row)


def _centre(gallery, row):
    return gallery.visualRect(gallery.gallery_model.index(row)).center()


def test_tile_shows_split_and_status(project, gallery):
    model = gallery.gallery_model
    frame = project.frames[0]
    assert model.data(model.index(0), GalleryModel.SplitRole) == frame.split
    assert model.data(model.index(0), GalleryModel.AnimalCountRole) == 0
    _label(project, gallery, 0)
    assert model.data(model.index(0), GalleryModel.AnimalCountRole) == 1


def test_filter_by_status_and_split(project, gallery):
    _label(project, gallery, 0)
    _label(project, gallery, 1)
    gallery.set_filter("labelled", None, None)
    assert gallery.visible_rows() == [0, 1]
    split = project.frames[0].split
    expect = [i for i, f in enumerate(project.frames) if f.split == split]
    gallery.set_filter(None, split, None)
    assert gallery.visible_rows() == expect
    gallery.set_filter("labelled", split, None)
    assert gallery.visible_rows() == [i for i in (0, 1) if i in expect]
    video = project.frames[-1].video
    gallery.set_filter(None, None, video)
    assert gallery.visible_rows() == [i for i, f in enumerate(project.frames) if f.video == video]
    gallery.set_filter(None, None, None)
    assert gallery.visible_rows() == list(range(len(project.frames)))


def test_filter_survives_reload(project, gallery):
    gallery.set_filter(None, None, project.frames[0].video)
    before = gallery.visible_rows()
    gallery.gallery_model.reload()
    assert gallery.visible_rows() == before


def test_plain_click_opens_modifier_click_selects(qtbot, gallery):
    got = []
    gallery.frame_activated.connect(got.append)
    qtbot.mouseClick(gallery.viewport(), Qt.MouseButton.LeftButton, pos=_centre(gallery, 2))
    assert got == [2]
    got.clear()
    qtbot.mouseClick(gallery.viewport(), Qt.MouseButton.LeftButton, CTRL, _centre(gallery, 1))
    qtbot.mouseClick(gallery.viewport(), Qt.MouseButton.LeftButton, CTRL, _centre(gallery, 3))
    assert got == []
    assert {1, 3} <= set(gallery.selected_rows())


def test_grid_reflows_to_width(qtbot, gallery):
    def cols():
        gallery.doItemsLayout()
        top = gallery.visualRect(gallery.gallery_model.index(0)).top()
        return sum(
            1
            for i in range(gallery.gallery_model.rowCount())
            if gallery.visualRect(gallery.gallery_model.index(i)).top() == top
        )

    widths = {}
    for w in (500, 1400):
        gallery.resize(w, 600)
        qtbot.wait(20)
        widths[w] = cols()
        assert gallery.gridSize().width() >= 180
        assert not gallery.horizontalScrollBar().isVisible()
    assert widths[500] < widths[1400]


def test_strip_is_one_row_and_models_share(qtbot, gallery):
    strip = GalleryView(strip=True)
    qtbot.addWidget(strip)
    strip.share_model(gallery)
    strip.resize(500, 130)
    strip.show()
    assert strip.model() is gallery.model()
    strip.doItemsLayout()
    tops = {
        strip.visualRect(strip.gallery_model.index(i)).top()
        for i in range(strip.gallery_model.rowCount())
    }
    assert len(tops) == 1
    assert strip.gridSize().width() <= 110
