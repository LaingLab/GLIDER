"""CVProcessor routes live tracking through the identity assigner."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from glider.vision.cv_processor import CVProcessor, CVSettings, Detection, DetectionBackend
from glider.vision.zones import ZoneTracker


def _det(x: int, y: int = 10) -> Detection:
    return Detection(class_id=0, class_name="mouse", confidence=0.9, bbox=(x, y, 10, 10))


def _frame() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


def _proc(mode="appearance", backend=DetectionBackend.YOLO_BYTETRACK, dets=None, **kw):
    s = CVSettings(
        backend=backend,
        identity_mode=mode,
        animals=[{"name": "A", "class_name": None}, {"name": "B", "class_name": None}],
        **kw,
    )
    proc = CVProcessor(s)
    proc._initialized = True
    proc._active_backend = backend
    proc._detect_yolo = MagicMock(side_effect=lambda *_a, **_k: list(dets or [_det(0), _det(100)]))
    proc._detect = MagicMock(side_effect=AssertionError("ByteTrack path must not run"))
    return proc


def test_identity_names_animals_with_stable_ids():
    proc = _proc()
    _, tracked, _ = proc.process_frame(_frame(), 0.0)
    assert sorted((t.animal, t.track_id) for t in tracked) == [("A", 0), ("B", 1)]


def test_frame_skipping_is_ignored_while_identity_is_active():
    proc = _proc(process_every_n_frames=3)
    for i in range(4):
        proc.process_frame(_frame(), float(i))
        assert proc.last_result_cached is False
    assert proc._detect_yolo.call_count == 4


def test_continuity_before_seed_gives_unnamed_extras():
    proc = _proc(mode="continuity")
    _, tracked, _ = proc.process_frame(_frame(), 0.0)
    assert proc.identity_ready is False
    assert tracked and all(t.animal is None for t in tracked)
    assert proc.seed_identity({"A": (5, 15), "B": (105, 15)}, []) == []
    assert proc.identity_ready is True
    _, tracked, _ = proc.process_frame(_frame(), 1.0)
    assert sorted(t.animal for t in tracked) == ["A", "B"]


def test_pose_backend_reports_why_and_falls_back():
    proc = _proc(backend=DetectionBackend.POSE_MODEL)
    assert "YOLO" in proc.identity_unavailable_reason
    assert not proc.identity_active
    proc._detect = MagicMock(return_value=[_det(0)])
    _, tracked, _ = proc.process_frame(_frame(), 0.0)
    proc._detect.assert_called_once()
    assert all(t.animal is None for t in tracked)


def test_off_has_no_reason_and_needs_no_every_frame():
    proc = _proc(mode="off")
    assert proc.identity_unavailable_reason is None
    assert not proc.identity_active
    assert not proc.needs_every_frame


def test_changing_animals_or_mode_resets_the_assigner():
    proc = _proc(mode="continuity")
    proc.process_frame(_frame(), 0.0)
    proc.seed_identity({"A": (5, 15), "B": (105, 15)}, [])
    assert proc.identity_ready
    s = proc.settings.copy()
    s.animals = [{"name": "A", "class_name": None}]
    proc.update_settings(s)
    assert not proc.identity_ready


def test_zone_tracker_receives_the_hidden_animal_at_its_frozen_spot():
    proc = _proc()
    proc._zone_tracker = MagicMock(spec=ZoneTracker)
    proc._zone_tracker.update.return_value = {}
    proc.process_frame(_frame(), 0.0)
    proc._detect_yolo.side_effect = lambda *_a, **_k: [_det(0)]
    proc.process_frame(_frame(), 1.0)
    tracked = proc._zone_tracker.update.call_args.args[0]
    hidden = [t for t in tracked if not t.visible]
    assert len(hidden) == 1 and hidden[0].centroid == (105, 15)


def _seeded():
    proc = _proc(mode="continuity")
    proc.process_frame(_frame(), 0.0)
    proc.seed_identity({"A": (5, 15), "B": (105, 15)}, [])
    assert proc.identity_ready
    return proc


@pytest.mark.parametrize(
    "field, value",
    [
        ("identity_separation_px", 80.0),
        ("identity_relabel_s", 1.0),
        ("identity_max_travel_px", 20.0),
    ],
)
def test_changing_an_identity_tuning_rebuilds_the_assigner(field, value):
    proc = _seeded()
    s = proc.settings.copy()
    setattr(s, field, value)
    proc.update_settings(s)
    assert not proc.identity_ready


def test_animals_edited_in_place_still_rebuild_the_assigner():
    proc = _seeded()
    proc.settings.animals[0]["name"] = "Z"  # the live list, mutated in place
    proc.update_settings(proc.settings)
    assert not proc.identity_ready


def test_an_unrelated_change_keeps_the_seed():
    proc = _seeded()
    s = proc.settings.copy()
    s.confidence_threshold = 0.3
    proc.update_settings(s)
    assert proc.identity_ready


def test_class_names_come_from_the_loaded_yolo_model():
    proc = _proc()
    assert proc.class_names == []
    proc._yolo_model = MagicMock(names={0: "dam", 1: "pup"})
    assert proc.class_names == ["dam", "pup"]


def test_overlay_names_each_animal_and_dims_a_hidden_one():
    from glider.vision.cv_processor import TrackedObject

    proc = CVProcessor(CVSettings(show_labels=False, show_trails=False))
    seen = TrackedObject(0, "m", (10, 10, 40, 40), 0.9, (30, 30), animal="Dam")
    hidden = TrackedObject(1, "m", (110, 10, 40, 40), 0.9, (130, 30), animal="Pup", visible=False)
    out = proc.draw_overlays(_frame(), [], [seen, hidden])
    color = np.array(proc.settings.overlay_color)
    # A name is written at each centroid.
    assert out[22:34, 30:60].any() and out[22:34, 130:160].any()
    # The hidden animal is drawn, but never in the live overlay colour.
    assert (out[10:52, 108:152] == color).all(axis=2).sum() == 0
    assert (out[10:52, 8:52] == color).all(axis=2).sum() > 0


def test_clear_identity_drops_only_the_seed():
    proc = _seeded()
    proc._trail_history[7] = [(1, 1)]
    proc.clear_identity()
    assert not proc.identity_ready
    assert proc._trail_history[7] == [(1, 1)]


def test_clear_identity_resets_the_animal_classifiers():
    proc = _seeded()
    clf = MagicMock()
    proc.set_animal_classifiers(clf)
    proc.clear_identity()
    clf.reset.assert_called_once_with()


def test_never_seen_animal_is_not_in_any_zone():
    from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape

    s = CVSettings(
        backend=DetectionBackend.YOLO_BYTETRACK,
        identity_mode="appearance",
        animals=[{"name": "Dam", "class_name": "dam"}, {"name": "Pup", "class_name": "pup"}],
    )
    proc = CVProcessor(s)
    proc._initialized = True
    proc._active_backend = DetectionBackend.YOLO_BYTETRACK
    pup = Detection(class_id=1, class_name="pup", confidence=0.9, bbox=(150, 80, 10, 10))
    proc._detect_yolo = MagicMock(return_value=[pup])
    zt = ZoneTracker()
    zone = Zone(id="z", name="corner", shape=ZoneShape.RECTANGLE, vertices=[(0, 0), (0.2, 0.2)])
    zt.set_zone_configuration(ZoneConfiguration(zones=[zone]))
    proc._zone_tracker = zt
    proc.process_frame(_frame(), 0.0)
    state = zt.get_zone_state("z")
    assert not state.occupied and not state.entered and state.object_count == 0
