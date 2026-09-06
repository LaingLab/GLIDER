"""The preview grid has to keep its shape past nine cameras.

A 16-camera rig is the point of this work, and the old layout topped out at a
fixed 3x3: cameras 10-16 were created and recorded but never placed in the
grid, so they vanished from the UI while still writing files.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.gui.widgets.multi_camera_preview import MultiCameraPreviewWidget, grid_columns


class TestGridColumns:
    def test_small_counts_keep_their_familiar_shapes(self):
        # These were hand-picked before and people are used to them; a bare
        # ceil(sqrt) would turn 2 cameras into a 2x1 that reads as a column.
        assert grid_columns(1) == 1
        assert grid_columns(2) == 2
        assert grid_columns(3) == 2
        assert grid_columns(4) == 2
        assert grid_columns(5) == 3
        assert grid_columns(6) == 3

    def test_it_keeps_growing_past_nine(self):
        assert grid_columns(9) == 3
        assert grid_columns(10) == 4
        assert grid_columns(12) == 4
        assert grid_columns(16) == 4
        assert grid_columns(17) == 5

    def test_the_grid_is_never_taller_than_it_is_wide(self):
        # A tall thin stack wastes a wide monitor and shrinks every tile.
        for n in range(1, 33):
            cols = grid_columns(n)
            rows = -(-n // cols)
            assert rows <= cols + 1, f"{n} cameras -> {cols}x{rows}"

    def test_zero_cameras_does_not_divide_by_zero(self):
        assert grid_columns(0) >= 1


@pytest.fixture
def preview(qtbot):
    widget = MultiCameraPreviewWidget()
    qtbot.addWidget(widget)
    return widget


class TestPlacement:
    def test_sixteen_cameras_all_get_a_tile(self, preview):
        for i in range(16):
            preview.add_camera(f"cam{i}", is_primary=(i == 0))
        assert preview.camera_count == 16
        assert len(preview._tiles) == 16

    def test_every_tile_is_actually_in_the_layout(self, preview):
        # The old bug: tiles existed but were never added past the ninth slot.
        for i in range(16):
            preview.add_camera(f"cam{i}")
        placed = {
            preview._grid_layout.itemAt(i).widget() for i in range(preview._grid_layout.count())
        }
        assert placed == set(preview._tiles.values())

    def test_tiles_occupy_distinct_cells(self, preview):
        for i in range(16):
            preview.add_camera(f"cam{i}")
        cells = set()
        for i in range(preview._grid_layout.count()):
            row, col, _, _ = preview._grid_layout.getItemPosition(i)
            cells.add((row, col))
        assert len(cells) == 16

    def test_removing_a_camera_reflows_the_rest(self, preview):
        for i in range(16):
            preview.add_camera(f"cam{i}")
        preview.remove_camera("cam0")
        assert preview.camera_count == 15
        placed = {
            preview._grid_layout.itemAt(i).widget() for i in range(preview._grid_layout.count())
        }
        assert placed == set(preview._tiles.values())

    def test_frames_reach_the_right_tile_at_sixteen(self, preview):
        for i in range(16):
            preview.add_camera(f"cam{i}")
        frame = np.full((48, 64, 3), 128, np.uint8)
        preview.update_frame("cam15", frame)  # the one the old grid dropped
        assert preview._tiles["cam15"]._preview.pixmap() is not None
