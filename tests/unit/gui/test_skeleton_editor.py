"""Skeleton templates and the template editor (#224)."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QMessageBox

from glider.gui.animal_annotator.skeleton_editor import (
    SkeletonCanvas,
    SkeletonEditorDialog,
    TemplatePicker,
)
from glider.vision.training import templates as tmod
from glider.vision.training.labels import VISIBLE, PoseInstance
from glider.vision.training.project import StudioProject, StudioProjectError
from glider.vision.training.templates import (
    POSE_PALETTE,
    SkeletonTemplate,
    TemplateError,
    builtin_templates,
    list_custom_templates,
    save_custom_template,
)


@pytest.fixture(autouse=True)
def templates_dir(tmp_path, monkeypatch):
    folder = tmp_path / "templates"
    monkeypatch.setattr(tmod, "custom_templates_dir", lambda: folder)
    return folder


# ----------------------------------------------------------------------
# templates
# ----------------------------------------------------------------------


def test_builtins_are_valid_and_mouse_comes_first():
    found = builtin_templates()
    assert [t.name for t in found] == ["Mouse", "Person", "Hand", "Quadruped", "Box"]
    for t in found:
        assert t.problem() is None, t.name
        assert all(0 <= k.x <= 1 and 0 <= k.y <= 1 for k in t.keypoints), t.name
    counts = {t.name: len(t.keypoints) for t in found}
    assert counts == {"Mouse": 7, "Person": 17, "Hand": 21, "Quadruped": 17, "Box": 4}


def test_default_colours_follow_the_ultralytics_palette():
    t = SkeletonTemplate("x")
    for i in range(3):
        t.add_keypoint(0.1 * i, 0.5)
    assert t.colours == list(POSE_PALETTE[:3])
    assert t.names == ["point_0", "point_1", "point_2"]


def test_remove_keypoint_moves_the_rest_up():
    t = SkeletonTemplate("x")
    for i in range(3):
        t.add_keypoint(0.1 * i, 0.5)
    t.remove_keypoint(1)
    assert t.names == ["point_0", "point_2"]


def test_old_template_files_with_connections_still_load(tmp_path):
    import json

    data = builtin_templates()[4].to_dict()
    data["skeleton"] = [[0, 1], [1, 2]]
    (tmp_path / "old.json").write_text(json.dumps(data))
    assert len(SkeletonTemplate.load(tmp_path / "old.json").keypoints) == 4
    assert "skeleton" not in builtin_templates()[4].to_dict()


def test_custom_templates_save_list_rename_delete(templates_dir):
    t = builtin_templates()[0].copy("Rat")
    save_custom_template(t)
    assert [c.name for c in list_custom_templates()] == ["Rat"]
    with pytest.raises(TemplateError):
        save_custom_template(t.copy("Rat"))  # clash
    with pytest.raises(TemplateError):
        save_custom_template(t.copy("Mouse"))  # built-in name
    renamed = t.copy("Rat side")
    save_custom_template(renamed, replacing="Rat")
    assert [c.name for c in list_custom_templates()] == ["Rat side"]
    tmod.delete_custom_template("Rat side")
    assert list_custom_templates() == []


def test_template_round_trips(tmp_path):
    t = builtin_templates()[2]
    t.save(tmp_path / "h.json")
    back = SkeletonTemplate.load(tmp_path / "h.json")
    assert back.names == t.names and back.colours == t.colours


def test_unreadable_custom_template_is_skipped(templates_dir):
    templates_dir.mkdir(parents=True)
    (templates_dir / "bad.json").write_text("{not json")
    assert list_custom_templates() == []


# ----------------------------------------------------------------------
# project
# ----------------------------------------------------------------------


def test_project_keeps_template_colours(tmp_path):
    hand = builtin_templates()[2]
    p = StudioProject.create(
        tmp_path / "p", "Hands", hand.names, colours=hand.colours, template_name="Hand"
    )
    again = StudioProject.load(p.folder)
    assert again.keypoint_colours == hand.colours
    assert again.template_name == "Hand"


def test_keypoint_count_is_frozen_once_labelled(tmp_path):
    from glider.vision.training.project import FrameRecord

    mouse = builtin_templates()[0]
    p = StudioProject.create(tmp_path / "p", "M", mouse.names)
    p.apply_template(mouse.names[:-1], [])  # nothing labelled yet: fine
    assert len(p.keypoint_names) == 6
    p.frames.append(FrameRecord("v_f000000", "v", 0))
    p.write_instances(p.frames[0], [PoseInstance([(0.5, 0.5, VISIBLE)] * 6)])
    with pytest.raises(StudioProjectError):
        p.apply_template(mouse.names, [])
    p.apply_template([f"k{i}" for i in range(6)], [(1, 2, 3)] * 6)  # rename: fine
    assert StudioProject.load(p.folder).keypoint_names[0] == "k0"


# ----------------------------------------------------------------------
# editor canvas
# ----------------------------------------------------------------------


@pytest.fixture
def canvas(qtbot):
    t = SkeletonTemplate("x")
    c = SkeletonCanvas(t)
    qtbot.addWidget(c)
    c.resize(400, 400)
    c.show()
    qtbot.waitExposed(c)
    return c


def _at(c: SkeletonCanvas, x, y):
    return c.to_widget(x, y).toPoint()


def test_clicks_place_keypoints(qtbot, canvas):
    for x in (0.2, 0.5, 0.8):
        qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, x, 0.5))
    t = canvas._template
    assert len(t.keypoints) == 3
    assert canvas.selected == 2
    assert t.keypoints[1].x == pytest.approx(0.5, abs=0.01)


def test_clicking_a_keypoint_toggles_selection(qtbot, canvas):
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.2, 0.5))
    qtbot.keyClick(canvas, Qt.Key.Key_Escape)
    assert canvas.selected == -1
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.2, 0.5))
    assert canvas.selected == 0 and len(canvas._template.keypoints) == 1
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.2, 0.5))
    assert canvas.selected == -1


def test_drag_moves_a_keypoint(qtbot, canvas):
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.2, 0.5))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.8, 0.5))
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.2, 0.5))
    qtbot.mouseMove(canvas, _at(canvas, 0.3, 0.2))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.3, 0.2))
    k = canvas._template.keypoints[0]
    assert (k.x, k.y) == pytest.approx((0.3, 0.2), abs=0.01)
    assert len(canvas._template.keypoints) == 2


def test_delete_removes_the_selected_keypoint(qtbot, canvas):
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.2, 0.5))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.8, 0.5))
    qtbot.keyClick(canvas, Qt.Key.Key_Delete)
    assert len(canvas._template.keypoints) == 1


def test_locked_canvas_cannot_add_or_delete(qtbot, canvas):
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.2, 0.5))
    canvas.locked = True
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=_at(canvas, 0.8, 0.5))
    canvas.select(0)
    qtbot.keyClick(canvas, Qt.Key.Key_Delete)
    assert len(canvas._template.keypoints) == 1


def test_hit_testing_maps_back():
    c = SkeletonCanvas(builtin_templates()[0])
    c.resize(300, 300)
    nose = c._template.keypoints[2]
    assert c.keypoint_at(c.to_widget(nose.x, nose.y)) == 2
    assert c.keypoint_at(QPointF(-50, -50)) == -1


# ----------------------------------------------------------------------
# dialog and picker
# ----------------------------------------------------------------------


def test_dialog_rows_follow_the_canvas(qtbot):
    dialog = SkeletonEditorDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    c = dialog.canvas
    qtbot.mouseClick(c, Qt.MouseButton.LeftButton, pos=_at(c, 0.3, 0.3))
    qtbot.mouseClick(c, Qt.MouseButton.LeftButton, pos=_at(c, 0.6, 0.6))
    assert len(dialog._rows) == 2
    assert dialog._summary.text() == "2 keypoints"
    dialog._rows[0].name.setText("nose")
    dialog._rows[0].name.textEdited.emit("nose")
    assert dialog.template().names[0] == "nose"


def test_dialog_start_from_copies_a_template(qtbot):
    dialog = SkeletonEditorDialog()
    qtbot.addWidget(dialog)
    dialog.start_from(builtin_templates()[1])
    t = dialog.template()
    assert len(t.keypoints) == 17
    assert t.name == "Person (copy)"
    assert len(dialog._rows) == 17


def test_dialog_save_reports_problems_and_stays_open(qtbot, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))
    dialog = SkeletonEditorDialog(on_save=lambda t: "nope")
    qtbot.addWidget(dialog)
    dialog._name.setText("T")
    dialog._on_save_clicked()  # no keypoints yet
    dialog.start_from(builtin_templates()[4])
    dialog._on_save_clicked()  # on_save refuses
    assert len(warnings) == 2 and "Nope." in warnings[1]
    assert dialog.result() == 0


def test_picker_lists_custom_templates(qtbot, templates_dir):
    save_custom_template(builtin_templates()[4].copy("Cage corners"))
    picker = TemplatePicker()
    qtbot.addWidget(picker)
    chosen = []
    picker.template_chosen.connect(chosen.append)
    picker._populate()
    headings = [
        w.text()
        for a in picker._menu.actions()
        if (w := getattr(a, "defaultWidget", lambda: None)()) is not None and hasattr(w, "text")
    ]
    headings = [h.lower() for h in headings]
    assert "default templates" in headings and "custom templates" in headings
    picker.choose(list_custom_templates()[0])
    assert chosen[-1].name == "Cage corners"
    assert picker.text().startswith("Cage corners")


def test_new_project_dialog_uses_the_picked_template(qtbot, tmp_path):
    from glider.gui.animal_annotator.window import NewProjectDialog

    dialog = NewProjectDialog(tmp_path)
    qtbot.addWidget(dialog)
    assert dialog.template.name == "Mouse"
    dialog._picker.choose(builtin_templates()[3])
    assert len(dialog.keypoint_names) == 17
