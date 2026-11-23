"""Live multi-animal identity assigner (pure numpy, no camera)."""

import numpy as np

from glider.vision.cv_processor import Detection
from glider.vision.identity_live import FLAG_ORDER, IdentityAssigner


def det(x, y, cls="mouse", w=40, h=20, kps=None):
    d = Detection(class_id=0, class_name=cls, confidence=0.9, bbox=(x - w // 2, y - h // 2, w, h))
    if kps is not None:
        d._keypoints = np.asarray(kps, dtype=float)
    return d


def by_name(out):
    return {o.animal: o for o in out if o.animal is not None}


def rows(out):
    return [(o.animal, o.track_id, o.visible, o.identity_flag, o.centroid) for o in out]


def test_flag_order():
    assert FLAG_ORDER == ("close", "stitched", "gap", "relabeled")


def test_continuity_seed_and_follow():
    a = IdentityAssigner(["A", "B"], mode="continuity")
    assert a.seed([det(100, 100), det(300, 100)], {"A": (98, 102), "B": (305, 99)}) == []
    assert a.ready
    out = a.update([det(310, 100), det(105, 100)])  # order shuffled
    assert rows(out) == [("A", 0, True, "", (105, 100)), ("B", 1, True, "", (310, 100))]


def test_unseeded_continuity_returns_unnamed_extras():
    a = IdentityAssigner(["A", "B"], mode="continuity")
    assert not a.ready
    out = a.update([det(100, 100), det(300, 100)])
    assert rows(out) == [(None, 2, True, "", (100, 100)), (None, 3, True, "", (300, 100))]


def test_crossing_keeps_names():
    # A moves right, B left, pass at x=200 with 20 px vertical gap
    a = IdentityAssigner(["A", "B"], mode="continuity")
    a.seed([det(100, 90), det(300, 110)], {"A": (100, 90), "B": (300, 110)})
    for t in range(1, 21):
        dets = [det(100 + 10 * t, 90), det(300 - 10 * t, 110)]
        out = a.update(dets[::-1] if t % 2 else dets)
    names = by_name(out)
    assert names["A"].centroid == (300, 90) and names["B"].centroid == (100, 110)
    assert names["A"].centroid[0] > names["B"].centroid[0]


def test_huddle_then_separate():
    a = IdentityAssigner(["A", "B"], mode="continuity")
    a.seed([det(100, 100), det(300, 100)], {"A": (100, 100), "B": (300, 100)})
    for t in range(1, 16):  # approach: A to 175, B to 225
        a.update([det(100 + 5 * t, 100), det(300 - 5 * t, 100)])
    for _ in range(20):  # one merged blob, slightly left of centre
        out = a.update([det(195, 100, w=80)])
        visible = [o for o in out if o.animal and o.visible]
        hidden = [o for o in out if o.animal and not o.visible]
        assert len(visible) == 1 and "close" in visible[0].identity_flag.split("+")
        assert len(hidden) == 1 and hidden[0].identity_flag == "gap"
    assert by_name(out)["A"].visible and by_name(out)["B"].centroid == (225, 100)
    for t in range(1, 6):  # separate
        out = a.update([det(195 + 10 * t, 100), det(195 - 10 * t, 100)])
    names = by_name(out)
    assert names["A"].centroid == (145, 100) and names["B"].centroid == (245, 100)
    assert names["A"].visible and names["B"].visible


def test_target_hidden_never_jumps():
    a = IdentityAssigner(["A"], mode="continuity")
    a.seed([det(100, 100), det(400, 100)], {"A": (100, 100)})
    for _ in range(60):  # the reach grows, but (400,100) was already there
        out = a.update([det(400, 100)])
        assert rows(out) == [("A", 0, False, "gap", (100, 100)), (None, 1, True, "", (400, 100))]


def test_hidden_reacquires_newly_appeared_detection_in_grown_reach():
    a = IdentityAssigner(["A"], mode="continuity")
    a.seed([det(100, 100)], {"A": (100, 100)})
    for _ in range(10):
        a.update([])
    # 250 px away, reach is 40 * 11 = 440, and nothing was near it last frame
    assert rows(a.update([det(350, 100)])) == [("A", 0, True, "stitched", (350, 100))]


def test_hidden_never_takes_animal_walking_up_from_afar():
    a = IdentityAssigner(["A"], mode="continuity")
    a.seed([det(100, 100), det(500, 100)], {"A": (100, 100)})
    # walks 25 px/frame from 500 to 145 (just beyond the one-frame reach), then waits
    for x in [*range(500, 144, -25), 145, 145, 145]:
        out = a.update([det(x, 100)])
        assert rows(out) == [("A", 0, False, "gap", (100, 100)), (None, 1, True, "", (x, 100))]


def test_reacquire_within_reach_is_stitched():
    a = IdentityAssigner(["A"], mode="continuity")
    a.seed([det(100, 100)], {"A": (100, 100)})
    for _ in range(3):
        assert rows(a.update([])) == [("A", 0, False, "gap", (100, 100))]
    # 60 px away, limit is 40 * (3 + 1) = 160
    assert rows(a.update([det(160, 100)])) == [("A", 0, True, "stitched", (160, 100))]
    assert rows(a.update([det(165, 100)])) == [("A", 0, True, "", (165, 100))]


def test_travel_limit_grows_with_frames_hidden():
    a = IdentityAssigner(["A"], mode="continuity")
    a.seed([det(100, 100)], {"A": (100, 100)})
    a.update([])  # hidden 1 frame
    out = a.update([det(200, 100)])  # 100 px > 40 * 2
    assert rows(out) == [("A", 0, False, "gap", (100, 100)), (None, 1, True, "", (200, 100))]
    a.update([])
    out = a.update([det(200, 100)])  # 100 px <= 40 * 4, newly appeared
    assert rows(out) == [("A", 0, True, "stitched", (200, 100))]


def test_one_frame_label_flicker_is_outvoted():
    a = IdentityAssigner(["Dam", "Pup"], mode="appearance", class_names=["dam", "pup"])
    assert a.ready
    a.update([det(100, 100, "dam"), det(130, 100, "pup")])
    out = a.update([det(100, 100, "pup"), det(130, 100, "pup")])  # dam mislabelled once
    assert rows(out) == [
        ("Dam", 0, True, "close", (100, 100)),
        ("Pup", 1, True, "close", (130, 100)),
    ]
    out = a.update([det(100, 100, "dam"), det(130, 100, "pup")])
    assert [o.identity_flag for o in out] == ["close", "close"]
    assert by_name(out)["Dam"].centroid == (100, 100)


def test_sustained_mismatch_relabels():
    a = IdentityAssigner(
        ["Dam", "Pup"], mode="appearance", class_names=["dam", "pup"], relabel_frames=5
    )
    a.update([det(100, 100, "dam"), det(300, 100, "pup")])
    dets = [det(100, 100, "pup"), det(300, 100, "pup"), det(500, 300, "dam")]
    for _ in range(5):
        out = a.update(dets)
        assert rows(out)[0] == ("Dam", 0, True, "", (100, 100))
    out = a.update(dets)  # sixth frame: beyond the travel limit, but relabelled
    assert rows(out) == [
        ("Dam", 0, True, "relabeled", (500, 300)),
        ("Pup", 1, True, "", (300, 100)),
        (None, 2, True, "", (100, 100)),
    ]
    out = a.update(dets)
    assert rows(out)[0] == ("Dam", 0, True, "", (500, 300))


def test_extras_get_ids_from_n():
    a = IdentityAssigner(["A"], mode="continuity")
    a.seed([det(100, 100), det(300, 100), det(500, 100)], {"A": (100, 100)})
    out = a.update([det(500, 100), det(100, 100), det(300, 100)])
    assert rows(out) == [
        ("A", 0, True, "", (100, 100)),
        (None, 1, True, "", (500, 100)),
        (None, 2, True, "", (300, 100)),
    ]


def test_anonymous_slots_block_lookalike_steal():
    # A and one anonymous click; A hidden; anonymous animal walks onto A's last spot
    a = IdentityAssigner(["A"], mode="continuity")
    assert a.seed([det(100, 100), det(300, 100)], {"A": (100, 100)}, [(300, 100)]) == []
    for x in [*range(300, 99, -25), 100, 100]:
        out = a.update([det(x, 100)])
        assert rows(out)[0][:3] == ("A", 0, False) and out[0].centroid == (100, 100)
        assert out[1].animal is None and out[1].track_id == 1 and out[1].centroid == (x, 100)
        assert len(out) == 2


def test_hidden_anonymous_slot_reported():
    a = IdentityAssigner(["A"], mode="continuity")
    a.seed([det(100, 100), det(300, 100)], {"A": (100, 100)}, [(300, 100)])
    out = a.update([det(100, 100)])
    assert rows(out) == [("A", 0, True, "", (100, 100)), (None, 1, False, "gap", (300, 100))]


def test_flags_join_in_fixed_order():
    a = IdentityAssigner(["A", "B"], mode="continuity")
    a.seed([det(100, 100), det(150, 100)], {"A": (100, 100), "B": (150, 100)})
    out = a.update([det(150, 100)])
    assert rows(out) == [("A", 0, False, "gap", (100, 100)), ("B", 1, True, "close", (150, 100))]
    out = a.update([det(100, 100), det(150, 100)])
    assert rows(out)[0] == ("A", 0, True, "close+stitched", (100, 100))


def test_keypoints_drive_cost():
    # Same centroids, but keypoints say the animals swapped heading; low-conf points ignored.
    a = IdentityAssigner(["A", "B"], mode="continuity", max_travel_px=200)
    kp_a = [[90, 100, 0.9], [110, 100, 0.9]]
    kp_b = [[150, 100, 0.9], [170, 100, 0.9]]
    a.seed([det(100, 100, kps=kp_a), det(160, 100, kps=kp_b)], {"A": (100, 100), "B": (160, 100)})
    # Detection near A's centroid carries B's keypoint layout and vice versa.
    out = a.update(
        [
            det(130, 100, kps=[[150, 100, 0.9], [170, 100, 0.9]]),
            det(131, 100, kps=[[90, 100, 0.9], [110, 100, 0.1]]),
        ]
    )
    assert by_name(out)["A"].centroid == (131, 100) and by_name(out)["B"].centroid == (130, 100)


def test_seed_reports_unbound_names():
    a = IdentityAssigner(["A", "B"], mode="continuity")
    assert a.seed([det(100, 100)], {"A": (100, 100), "B": (500, 500)}) == ["B"] and not a.ready


def test_reset_clears_seed():
    a = IdentityAssigner(["A"], mode="continuity")
    a.seed([det(100, 100)], {"A": (100, 100)})
    a.reset()
    assert not a.ready
    assert rows(a.update([det(100, 100)])) == [(None, 1, True, "", (100, 100))]


def test_hidden_slot_is_frozen_not_extrapolated():
    a = IdentityAssigner(["A"], mode="continuity")
    a.seed([det(100, 100)], {"A": (100, 100)})
    for x in (130, 160, 190):  # moving right 30 px/frame
        a.update([det(x, 100)])
    a.update([])
    a.update([])
    # Extrapolating would predict 220 and pick 230; frozen at 190 picks 190.
    out = a.update([det(230, 100), det(190, 100)])
    assert rows(out)[0] == ("A", 0, True, "close+stitched", (190, 100))


def test_appearance_first_frame_matches_by_class():
    a = IdentityAssigner(["Dam", "Pup"], mode="appearance", class_names=["dam", "pup"])
    out = a.update([det(100, 100, "pup"), det(300, 100, "dam")])
    assert rows(out) == [("Dam", 0, True, "", (300, 100)), ("Pup", 1, True, "", (100, 100))]
