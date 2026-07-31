"""The keypoint schema and its editor.

Order is load-bearing and invisible in a comma-separated field, so it gets a
figure, a validator, and a file you can reload.
"""

from __future__ import annotations

import json

import pytest

from glider.analysis.behavior.keypoint_schema import (
    SCHEMA_VERSION,
    Keypoint,
    KeypointSchema,
    KeypointSchemaError,
)

TRAINED_ORDER = [
    "left_ear",
    "right_ear",
    "nose",
    "body_center",
    "left_hip",
    "right_hip",
    "tail_base",
]


class TestSchema:
    def test_default_matches_the_trained_order(self):
        assert KeypointSchema.default_mouse().names == TRAINED_ORDER

    def test_default_points_sit_inside_the_figure(self):
        for kp in KeypointSchema.default_mouse().keypoints:
            assert 0.0 <= kp.x <= 1.0 and 0.0 <= kp.y <= 1.0

    def test_nose_is_ahead_of_tail_on_the_figure(self):
        # y runs nose-to-tail; a reader uses this to sanity-check the layout.
        by_name = {k.name: k for k in KeypointSchema.default_mouse().keypoints}
        assert by_name["nose"].y < by_name["body_center"].y < by_name["tail_base"].y

    def test_a_good_schema_has_no_problem(self):
        assert KeypointSchema.default_mouse().problem() is None

    def test_duplicate_names_are_rejected(self):
        s = KeypointSchema([Keypoint("nose", 0.5, 0.1), Keypoint("nose", 0.5, 0.9)])
        assert "used twice" in s.problem()

    def test_blank_names_are_rejected(self):
        s = KeypointSchema([Keypoint("nose", 0.5, 0.1), Keypoint("  ", 0.5, 0.9)])
        assert "no name" in s.problem()

    def test_an_empty_schema_is_rejected(self):
        assert KeypointSchema().problem() is not None


class TestReordering:
    def test_moving_down_swaps_with_the_next(self):
        s = KeypointSchema.default_mouse()
        assert s.move(0, +1) == 1
        assert s.names[:2] == ["right_ear", "left_ear"]

    def test_moving_up_swaps_with_the_previous(self):
        s = KeypointSchema.default_mouse()
        assert s.move(2, -1) == 1
        assert s.names[:3] == ["left_ear", "nose", "right_ear"]

    def test_moving_past_the_ends_is_clamped_not_wrapped(self):
        s = KeypointSchema.default_mouse()
        before = list(s.names)
        assert s.move(0, -1) == 0
        assert s.move(len(before) - 1, +1) == len(before) - 1
        assert s.names == before

    def test_an_out_of_range_index_is_ignored(self):
        s = KeypointSchema.default_mouse()
        before = list(s.names)
        assert s.move(99, -1) == 99
        assert s.names == before


class TestPersistence:
    def test_round_trip_preserves_order_and_positions(self, tmp_path):
        original = KeypointSchema.default_mouse()
        original.move(0, +2)
        path = tmp_path / "keypoints.json"
        original.save(path)

        loaded = KeypointSchema.load(path)
        assert loaded.names == original.names
        assert loaded.keypoints[0].x == pytest.approx(original.keypoints[0].x)

    def test_written_shape_is_the_documented_schema(self, tmp_path):
        path = tmp_path / "k.json"
        KeypointSchema.default_mouse().save(path)
        data = json.loads(path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert [k["name"] for k in data["keypoints"]] == TRAINED_ORDER

    def test_unknown_version_is_refused(self, tmp_path):
        path = tmp_path / "k.json"
        path.write_text(json.dumps({"schema_version": 99, "keypoints": []}))
        with pytest.raises(KeypointSchemaError, match="schema_version"):
            KeypointSchema.load(path)

    def test_malformed_file_is_refused(self, tmp_path):
        path = tmp_path / "k.json"
        path.write_text("{not json")
        with pytest.raises(KeypointSchemaError):
            KeypointSchema.load(path)

    def test_a_malformed_entry_is_refused(self, tmp_path):
        path = tmp_path / "k.json"
        path.write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "keypoints": [{"name": "a"}]})
        )
        with pytest.raises(KeypointSchemaError, match="malformed"):
            KeypointSchema.load(path)


class TestEditorDialog:
    def _editor(self, qtbot, schema=None):
        from glider.gui.behavior.keypoint_editor import KeypointEditorDialog

        dialog = KeypointEditorDialog(schema)
        qtbot.addWidget(dialog)
        return dialog

    def test_opens_on_the_default_schema(self, qtbot):
        assert self._editor(qtbot).names() == TRAINED_ORDER

    def test_the_list_shows_the_index_of_every_point(self, qtbot):
        editor = self._editor(qtbot)
        texts = [editor._list.item(i).text() for i in range(editor._list.count())]
        assert texts[0].startswith("0:")
        assert "tail_base" in texts[-1]

    def test_reordering_updates_the_names_it_returns(self, qtbot):
        editor = self._editor(qtbot)
        editor._list.setCurrentRow(0)
        editor._move(+1)
        assert editor.names()[:2] == ["right_ear", "left_ear"]

    def test_dragging_a_point_stores_a_normalised_position(self, qtbot):
        from PyQt6.QtCore import QPointF

        from glider.gui.behavior import keypoint_editor as ke

        editor = self._editor(qtbot)
        editor.point_moved(0, QPointF(ke._FIG * 0.25, ke._FIG * 0.75))
        assert editor.schema().keypoints[0].x == pytest.approx(0.25)
        assert editor.schema().keypoints[0].y == pytest.approx(0.75)

    def test_a_drag_outside_the_figure_is_clamped(self, qtbot):
        from PyQt6.QtCore import QPointF

        from glider.gui.behavior import keypoint_editor as ke

        editor = self._editor(qtbot)
        editor.point_moved(0, QPointF(-500.0, ke._FIG * 5))
        kp = editor.schema().keypoints[0]
        assert kp.x == pytest.approx(0.0) and kp.y == pytest.approx(1.0)

    def test_confirmation_is_blocked_while_the_schema_is_invalid(self, qtbot):
        schema = KeypointSchema([Keypoint("nose", 0.5, 0.1), Keypoint("nose", 0.5, 0.9)])
        editor = self._editor(qtbot, schema)
        assert editor._ok.isEnabled() is False
        assert "used twice" in editor._problem.text()

    def test_removing_a_point_shortens_the_schema(self, qtbot):
        editor = self._editor(qtbot)
        editor._list.setCurrentRow(0)
        editor._remove()
        assert editor.names() == TRAINED_ORDER[1:]

    def test_the_silhouette_is_drawn_not_loaded(self, qtbot):
        # No third-party asset: the figure must come from code alone.
        from glider.gui.behavior.keypoint_editor import mouse_silhouette

        path = mouse_silhouette()
        assert not path.isEmpty()
        rect = path.boundingRect()
        assert rect.width() > 0 and rect.height() > 0
