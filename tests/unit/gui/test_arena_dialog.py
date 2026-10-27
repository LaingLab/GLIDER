"""Arena dialog: corner picking, including corners outside the frame."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPoint

from glider.gui.dialogs.arena_dialog import ArenaCanvas, ArenaDialog

SQUARE = [(0.2, 0.1), (0.8, 0.1), (0.8, 0.9), (0.2, 0.9)]


@pytest.fixture
def frame():
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def canvas(qtbot, frame):
    widget = ArenaCanvas()
    qtbot.addWidget(widget)
    widget.resize(800, 600)
    widget.set_frame(frame)
    return widget


class TestCoordinates:
    def test_canvas_and_normalized_coords_round_trip(self, canvas):
        for point in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.25, 0.75)]:
            back = canvas._to_norm(canvas._to_canvas(point))
            assert back == pytest.approx(point, abs=0.01)

    def test_frame_is_inset_leaving_a_clickable_margin(self, canvas):
        rect = canvas._frame_rect()
        assert rect.x() > 0 and rect.y() > 0
        assert rect.right() < canvas.width()
        assert rect.bottom() < canvas.height()

    def test_a_click_in_the_margin_gives_coords_outside_the_frame(self, canvas):
        # The whole reason for the margin: several videos in this cohort have
        # the far arena corners above the top edge of the picture.
        above = canvas._to_norm(QPoint(canvas.width() // 2, 2))
        assert above[1] < 0.0

    def test_frame_keeps_its_aspect_ratio(self, canvas):
        rect = canvas._frame_rect()
        assert rect.width() / rect.height() == pytest.approx(640 / 480, abs=0.02)


class TestPicking:
    def _click(self, canvas, norm):
        canvas.mousePressEvent(_press(canvas._to_canvas(norm)))

    def test_four_clicks_complete_an_arena(self, canvas):
        for corner in SQUARE:
            self._click(canvas, corner)
        assert canvas.is_complete
        assert canvas.calibration() is not None

    def test_a_fifth_click_is_ignored(self, canvas):
        for corner in [*SQUARE, (0.5, 0.5)]:
            self._click(canvas, corner)
        assert len(canvas.corners) == 4

    def test_clicking_an_existing_corner_starts_a_drag(self, canvas):
        for corner in SQUARE:
            self._click(canvas, corner)
        self._click(canvas, SQUARE[0])
        assert canvas._dragging == 0

    def test_dragging_moves_that_corner_only(self, canvas):
        for corner in SQUARE:
            self._click(canvas, corner)
        self._click(canvas, SQUARE[0])
        canvas.mouseMoveEvent(_press(canvas._to_canvas((0.3, 0.2))))
        assert canvas.corners[0] == pytest.approx((0.3, 0.2), abs=0.01)
        assert canvas.corners[1:] == pytest.approx(SQUARE[1:], abs=1e-9)

    def test_undo_removes_the_last_corner(self, canvas):
        for corner in SQUARE:
            self._click(canvas, corner)
        canvas.undo()
        assert len(canvas.corners) == 3
        assert canvas.calibration() is None

    def test_clear_removes_everything(self, canvas):
        for corner in SQUARE:
            self._click(canvas, corner)
        canvas.clear()
        assert canvas.corners == []

    def test_corners_outside_the_frame_are_not_clamped(self, canvas):
        clipped = [(0.2, -0.15), (0.8, -0.15), (0.85, 0.9), (0.15, 0.9)]
        canvas.set_corners(clipped)
        assert canvas.corners == pytest.approx(clipped, abs=1e-9)
        assert canvas.calibration().is_clipped is True

    def test_calibration_carries_the_frame_size(self, canvas):
        canvas.set_corners(SQUARE)
        assert canvas.calibration().frame_size == (640, 480)


class TestDialog:
    def test_ok_is_disabled_until_four_corners_are_placed(self, qtbot, frame):
        from PyQt6.QtWidgets import QDialogButtonBox

        dialog = ArenaDialog(frame)
        qtbot.addWidget(dialog)
        ok = dialog.box.button(QDialogButtonBox.StandardButton.Ok)
        assert not ok.isEnabled()
        dialog.canvas.set_corners(SQUARE)
        assert ok.isEnabled()

    def test_status_reports_the_scale(self, qtbot, frame):
        dialog = ArenaDialog(frame)
        qtbot.addWidget(dialog)
        dialog.canvas.set_corners(SQUARE)
        assert "px/cm" in dialog.status.text()

    def test_a_degenerate_quad_disables_ok(self, qtbot, frame):
        from PyQt6.QtWidgets import QDialogButtonBox

        dialog = ArenaDialog(frame)
        qtbot.addWidget(dialog)
        dialog.canvas.set_corners([(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)])
        assert not dialog.box.button(QDialogButtonBox.StandardButton.Ok).isEnabled()

    def test_zone_size_is_reported(self, qtbot, frame):
        dialog = ArenaDialog(frame)
        qtbot.addWidget(dialog)
        dialog.zone_spin.setValue(12.5)
        assert dialog.zone_size_cm() == 12.5


def _press(pos: QPoint):
    """A left-button press at *pos*, without needing a real event loop."""
    from PyQt6.QtCore import QPointF
    from PyQt6.QtCore import Qt as _Qt
    from PyQt6.QtGui import QMouseEvent

    return QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(pos),
        _Qt.MouseButton.LeftButton,
        _Qt.MouseButton.LeftButton,
        _Qt.KeyboardModifier.NoModifier,
    )


class TestRoundArena:
    """Issue #212: a round apparatus is outlined as an ellipse, not four corners."""

    def test_three_clicks_make_a_circle_with_its_diameter(self, qtbot, frame):
        dialog = ArenaDialog(frame)
        qtbot.addWidget(dialog)
        dialog.shape_combo.setCurrentIndex(dialog.shape_combo.findData("ellipse"))
        dialog.arena_spin.setValue(40.0)

        dialog.canvas.set_corners([(0.5, 0.5), (0.8, 0.5), (0.5, 0.9)])

        arena = dialog.calibration()
        assert arena.shape == "ellipse"
        assert (arena.width_cm, arena.height_cm) == (40.0, 40.0)
        assert dialog.height_spin.isHidden() or not dialog.height_spin.isVisibleTo(dialog)

    def test_reopening_restores_shape_and_size(self, qtbot, frame):
        from glider.vision.arena import ArenaCalibration

        dialog = ArenaDialog(frame)
        qtbot.addWidget(dialog)
        dialog.set_arena(
            ArenaCalibration(corners=SQUARE, width_cm=40.0, height_cm=25.0, frame_size=(640, 480))
        )

        assert dialog.calibration().height_cm == 25.0  # the old "Side" box lost this


ARM_ENDS = [(0.5, 0.1), (0.8, 0.5), (0.5, 0.9), (0.2, 0.5)]


class TestPlusMaze:
    """Issue #217: an elevated plus maze is outlined by the ends of its four arms."""

    def _plus_dialog(self, qtbot, frame):
        dialog = ArenaDialog(frame)
        qtbot.addWidget(dialog)
        dialog.shape_combo.setCurrentIndex(dialog.shape_combo.findData("plus"))
        return dialog

    def test_four_arm_ends_make_a_plus_maze(self, qtbot, frame):
        from PyQt6.QtWidgets import QDialogButtonBox

        dialog = self._plus_dialog(qtbot, frame)
        dialog.arm_spin.setValue(6.0)
        dialog.canvas.set_corners(ARM_ENDS)

        arena = dialog.calibration()
        assert arena.shape == "plus"
        assert arena.arm_width_cm == 6.0
        assert dialog.box.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
        assert np.array(arena.plus_handles()) == pytest.approx(np.array(ARM_ENDS))

    def test_switching_to_plus_starts_at_a_maze_size(self, qtbot, frame):
        dialog = self._plus_dialog(qtbot, frame)
        assert dialog.arena_spin.value() == 75.0
        assert dialog.arm_spin.isVisibleTo(dialog)
        assert not dialog.zone_spin.isVisibleTo(dialog)  # the centre is the crossing

    def test_arm_width_is_hidden_for_other_shapes(self, qtbot, frame):
        dialog = ArenaDialog(frame)
        qtbot.addWidget(dialog)
        assert not dialog.arm_spin.isVisibleTo(dialog)

    def test_arms_wider_than_the_maze_disable_ok(self, qtbot, frame):
        from PyQt6.QtWidgets import QDialogButtonBox

        dialog = self._plus_dialog(qtbot, frame)
        dialog.arena_spin.setValue(20.0)
        dialog.height_spin.setValue(20.0)
        dialog.arm_spin.setValue(25.0)
        dialog.canvas.set_corners(ARM_ENDS)

        assert not dialog.box.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
        assert "narrower" in dialog.status.text()

    def test_a_crossed_click_order_disables_ok_without_crashing_the_paint(self, qtbot, frame):
        from PyQt6.QtWidgets import QDialogButtonBox

        dialog = self._plus_dialog(qtbot, frame)
        dialog.canvas.set_corners([ARM_ENDS[0], ARM_ENDS[2], ARM_ENDS[1], ARM_ENDS[3]])
        dialog.canvas.repaint()

        assert not dialog.box.button(QDialogButtonBox.StandardButton.Ok).isEnabled()

    def test_reopening_restores_arm_ends_and_width(self, qtbot, frame):
        from glider.vision.arena import ArenaCalibration

        saved = ArenaCalibration.from_plus(
            ARM_ENDS, width_cm=60.0, arm_width_cm=8.0, frame_size=(640, 480)
        )
        dialog = ArenaDialog(frame)
        qtbot.addWidget(dialog)
        dialog.set_arena(saved)

        again = dialog.calibration()
        assert again.shape == "plus"
        assert (again.width_cm, again.arm_width_cm) == (60.0, 8.0)
        assert np.array(dialog.canvas.corners) == pytest.approx(np.array(ARM_ENDS))

    def test_the_canvas_paints_a_finished_maze(self, canvas):
        canvas.set_shape("plus")
        canvas.set_arena_size(75.0, 75.0)
        canvas.set_corners(ARM_ENDS)
        canvas.repaint()  # must not raise
        assert canvas.calibration().shape == "plus"
