"""The zone editor: dragging and nudging zones, and placing them at a set size (#223)."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPointF, QSettings, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent

from glider.gui.dialogs.zone_dialog import ZoneDialog
from glider.vision.arena import ArenaCalibration
from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "zones.ini"), QSettings.Format.IniFormat)


def _editor(qtbot, settings, config=None, calibration=None):
    config = config if config is not None else ZoneConfiguration()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dialog = ZoneDialog(
        None, config, frame=frame, calibration=calibration, settings=settings
    ).embed()
    qtbot.addWidget(dialog)
    preview = dialog._preview
    # One widget pixel per frame pixel, so test coordinates are frame pixels.
    preview.setMinimumSize(640, 480)
    preview.resize(640, 480)
    preview._update_display()
    assert preview._image_rect == (0, 0, 640, 480)
    return dialog, preview


def _mouse(widget, kind, x, y):
    button = Qt.MouseButton.LeftButton
    buttons = Qt.MouseButton.NoButton if kind == "release" else button
    event_type = {
        "press": QMouseEvent.Type.MouseButtonPress,
        "move": QMouseEvent.Type.MouseMove,
        "release": QMouseEvent.Type.MouseButtonRelease,
    }[kind]
    pos = QPointF(x, y)
    event = QMouseEvent(
        event_type,
        pos,
        widget.mapToGlobal(pos),
        Qt.MouseButton.NoButton if kind == "move" else button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )
    {
        "press": widget.mousePressEvent,
        "move": widget.mouseMoveEvent,
        "release": widget.mouseReleaseEvent,
    }[kind](event)


def _key(widget, key, shift=False):
    mods = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    widget.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, key, mods))


def _config(*zones):
    config = ZoneConfiguration()
    for z in zones:
        config.add_zone(z)
    return config


class TestMoving:
    @pytest.mark.parametrize(
        "shape, vertices",
        [
            (ZoneShape.RECTANGLE, [(0.25, 0.25), (0.5, 0.5)]),
            (ZoneShape.CIRCLE, [(0.375, 0.375), (0.45, 0.375)]),
            (ZoneShape.POLYGON, [(0.25, 0.25), (0.5, 0.25), (0.375, 0.5)]),
        ],
    )
    def test_dragging_a_zone_moves_it_and_reports_the_change(
        self, qtbot, settings, shape, vertices
    ):
        zone = Zone(id="a", name="A", shape=shape, vertices=list(vertices))
        dialog, preview = _editor(qtbot, settings, _config(zone))
        changes = []
        dialog.zones_changed.connect(lambda: changes.append(1))

        # Grab inside the zone, carry it 64 px right and 48 px down.
        _mouse(preview, "press", 220, 200)
        _mouse(preview, "move", 250, 220)
        _mouse(preview, "move", 284, 248)
        _mouse(preview, "release", 284, 248)

        for (x0, y0), (x1, y1) in zip(vertices, zone.vertices, strict=True):
            assert x1 == pytest.approx(x0 + 0.1)
            assert y1 == pytest.approx(y0 + 0.1)
        assert changes == [1]  # once per drag, on release
        assert dialog.get_zone_configuration().zones[0] is zone
        # The table follows the canvas selection.
        assert dialog._table.selectionModel().selectedRows()[0].row() == 0

    def test_a_click_without_a_drag_selects_but_does_not_move(self, qtbot, settings):
        zone = Zone("a", "A", ZoneShape.RECTANGLE, [(0.25, 0.25), (0.5, 0.5)])
        dialog, preview = _editor(qtbot, settings, _config(zone))
        changes = []
        dialog.zones_changed.connect(lambda: changes.append(1))
        _mouse(preview, "press", 220, 200)
        _mouse(preview, "release", 220, 200)
        assert preview.selected_zone() is zone
        assert zone.vertices == [(0.25, 0.25), (0.5, 0.5)]
        assert changes == []

    def test_clicking_empty_canvas_clears_the_selection(self, qtbot, settings):
        zone = Zone("a", "A", ZoneShape.RECTANGLE, [(0.25, 0.25), (0.5, 0.5)])
        dialog, preview = _editor(qtbot, settings, _config(zone))
        _mouse(preview, "press", 220, 200)
        _mouse(preview, "release", 220, 200)
        _mouse(preview, "press", 600, 450)
        _mouse(preview, "release", 600, 450)
        assert preview.selected_zone() is None
        assert dialog._table.selectionModel().selectedRows() == []

    def test_arrow_keys_nudge_the_selected_zone(self, qtbot, settings):
        zone = Zone("a", "A", ZoneShape.RECTANGLE, [(0.25, 0.25), (0.5, 0.5)])
        dialog, preview = _editor(qtbot, settings, _config(zone))
        changes = []
        dialog.zones_changed.connect(lambda: changes.append(1))
        dialog._table.selectRow(0)  # selecting in the table selects on the canvas
        assert preview.selected_zone() is zone

        _key(preview, Qt.Key.Key_Right)
        _key(preview, Qt.Key.Key_Down, shift=True)
        assert zone.vertices[0] == pytest.approx((0.25 + 1 / 640, 0.25 + 10 / 480))
        assert len(changes) == 2

    def test_a_zone_cannot_be_dragged_off_the_frame(self, qtbot, settings):
        zone = Zone("a", "A", ZoneShape.RECTANGLE, [(0.25, 0.25), (0.5, 0.5)])
        dialog, preview = _editor(qtbot, settings, _config(zone))
        _mouse(preview, "press", 220, 200)
        _mouse(preview, "move", 2000, 200)
        _mouse(preview, "release", 2000, 200)
        assert zone.vertices[1][0] == pytest.approx(1.0)


class TestFixedSize:
    def test_one_click_places_a_rectangle_of_the_set_pixel_size(self, qtbot, settings):
        dialog, preview = _editor(qtbot, settings)
        assert dialog._unit_label.text() == "px"
        dialog._fixed_size_check.setChecked(True)
        dialog._width_spin.setValue(64)
        dialog._height_spin.setValue(96)
        dialog._draw_btn.setChecked(True)

        _mouse(preview, "press", 320, 240)

        pending = dialog._pending_zone
        assert pending is not None and pending.shape == ZoneShape.RECTANGLE
        (x1, y1), (x2, y2) = pending.vertices
        assert ((x2 - x1) * 640, (y2 - y1) * 480) == pytest.approx((64, 96))
        assert ((x1 + x2) / 2, (y1 + y2) / 2) == pytest.approx((0.5, 0.5))
        dialog._add_pending_zone()
        assert dialog.get_zone_configuration().zones == [pending]

    def test_a_circle_takes_a_diameter(self, qtbot, settings):
        dialog, preview = _editor(qtbot, settings)
        dialog._circle_radio.setChecked(True)
        dialog._fixed_size_check.setChecked(True)
        dialog._width_spin.setValue(80)
        assert dialog._width_label.text() == "Diameter:"
        assert dialog._height_spin.isHidden()
        dialog._draw_btn.setChecked(True)

        _mouse(preview, "press", 100, 100)

        (cx, cy), (rx, ry) = dialog._pending_zone.vertices
        assert dialog._pending_zone.shape == ZoneShape.CIRCLE
        assert (cx * 640, cy * 480) == pytest.approx((100, 100))
        assert ((rx - cx) * 640, ry - cy) == pytest.approx((40, 0))

    def test_without_fixed_size_drawing_still_takes_two_clicks(self, qtbot, settings):
        dialog, preview = _editor(qtbot, settings)
        dialog._draw_btn.setChecked(True)
        _mouse(preview, "press", 100, 100)
        assert dialog._pending_zone is None
        _mouse(preview, "press", 200, 150)
        assert dialog._pending_zone is not None

    def test_polygons_have_no_fixed_size(self, qtbot, settings):
        dialog, _preview = _editor(qtbot, settings)
        dialog._fixed_size_check.setChecked(True)
        dialog._polygon_radio.setChecked(True)
        assert not dialog._fixed_size_check.isEnabled()
        assert dialog._preview._stamp is None

    def test_with_an_arena_the_size_is_in_cm_on_the_floor(self, qtbot, settings):
        arena = ArenaCalibration(
            corners=[(0, 0), (1, 0), (1, 1), (0, 1)],
            width_cm=40.0,
            height_cm=30.0,
            frame_size=(640, 480),
        )
        dialog, preview = _editor(qtbot, settings, calibration=arena)
        assert dialog._unit_label.text() == "cm"
        dialog._fixed_size_check.setChecked(True)
        dialog._width_spin.setValue(10)
        dialog._height_spin.setValue(5)
        dialog._draw_btn.setChecked(True)

        _mouse(preview, "press", 320, 240)

        zone = dialog._pending_zone
        assert zone.shape == ZoneShape.POLYGON
        floor = arena.to_arena_cm(zone.vertices)
        xs, ys = [p[0] for p in floor], [p[1] for p in floor]
        assert max(xs) - min(xs) == pytest.approx(10)
        assert max(ys) - min(ys) == pytest.approx(5)

    def test_a_scalar_calibration_turns_cm_into_pixels(self, qtbot, settings):
        class Scalar:
            is_calibrated = True
            pixels_per_mm = 2.0  # 20 px/cm at 320 px wide, so 40 px/cm at 640
            calibration_width = 320

        dialog, preview = _editor(qtbot, settings, calibration=Scalar())
        assert dialog._unit_label.text() == "cm"
        dialog._fixed_size_check.setChecked(True)
        dialog._width_spin.setValue(2)
        dialog._height_spin.setValue(1)
        dialog._draw_btn.setChecked(True)
        _mouse(preview, "press", 320, 240)
        (x1, y1), (x2, y2) = dialog._pending_zone.vertices
        assert ((x2 - x1) * 640, (y2 - y1) * 480) == pytest.approx((80, 40))

    def test_sizes_are_remembered_per_unit(self, qtbot, settings):
        dialog, _preview = _editor(qtbot, settings)
        dialog._fixed_size_check.setChecked(True)
        dialog._width_spin.setValue(123)
        dialog._height_spin.setValue(45)

        again, _ = _editor(qtbot, settings)
        assert again._fixed_size_check.isChecked()
        assert (again._width_spin.value(), again._height_spin.value()) == (123, 45)

        # Centimetres are a different quantity: they keep their own default.
        arena = ArenaCalibration(corners=[(0, 0), (1, 0), (1, 1), (0, 1)])
        again.set_calibration(arena)
        assert again._unit_label.text() == "cm"
        assert again._width_spin.value() == 10.0
