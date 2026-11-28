"""ZoneState carries which named animals are in a zone and which are visible."""

from glider.vision.cv_processor import TrackedObject
from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape, ZoneTracker


def _obj(tid, animal, visible=True, xy=(50, 50)):
    return TrackedObject(
        track_id=tid,
        class_name="mouse",
        bbox=(xy[0] - 5, xy[1] - 5, 10, 10),
        confidence=0.9,
        centroid=xy,
        animal=animal,
        visible=visible,
        age=1,  # observed at least once (a never-seen slot is in no zone)
    )


def test_hidden_named_animal_stays_in_zone_but_not_visible():
    zone = Zone(id="z", name="Z", shape=ZoneShape.RECTANGLE, vertices=[(0.0, 0.0), (1.0, 1.0)])
    tracker = ZoneTracker()
    tracker.set_zone_configuration(
        ZoneConfiguration(zones=[zone], config_width=100, config_height=100)
    )

    states = tracker.update([_obj(1, "A"), _obj(2, "B", visible=False), _obj(3, None)], 100, 100)

    assert states["z"].animals == {"A", "B"}
    assert states["z"].visible_animals == frozenset({"A"})
