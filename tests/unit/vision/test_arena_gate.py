"""Arena gating: rejecting detections that left the floor."""

from __future__ import annotations

import numpy as np
import pytest

from glider.vision.arena import ArenaCalibration
from glider.vision.arena_gate import ArenaGateSettings, gate_pose_csv, gate_to_arena
from glider.vision.pose.dlc import meta_path, read_pose_meta

# The fronto-parallel square from test_arena.py: a real 400x400 px square in a
# 640x480 frame, which is close to what these rigs produce (30 cm at 13.3 px/cm).
_L, _R, _T, _B = 120 / 640, 520 / 640, 40 / 480, 440 / 480
SQUARE = [(_L, _T), (_R, _T), (_R, _B), (_L, _B)]

# A steeply oblique view whose vanishing line crosses the frame at y ~ 90.
# Its homography carries a NEGATIVE scale: w is -1.42 at the arena centre, so
# the whole interior sits on the negative-w side. That is the geometry behind
# `test_a_steeply_oblique_arena_does_not_gate_its_own_interior`.
HORIZON_CORNERS = [(0.40, 0.35), (0.60, 0.35), (0.99, 0.98), (0.01, 0.98)]
#: Pixel centre of the oblique arena, for padding frames that must stay inside
#: it. (320, 240) is the centre of the SQUARE arena and lands outside this one.
OBLIQUE_CENTRE_PX = [320.0, 219.3]


def _pose(xy, confidence=None, names=("a", "b", "c", "d")):
    from glider.vision.pose import PoseData

    xy = np.asarray(xy, dtype=float)
    if confidence is None:
        confidence = np.where(np.isfinite(xy[:, :, 0]), 0.9, 0.0)
    return PoseData(xy=xy, confidence=confidence, keypoint_names=list(names), fps=30.0)


def _arena(corners=SQUARE, **kw):
    kw.setdefault("width_cm", 30.0)
    kw.setdefault("height_cm", 30.0)
    kw.setdefault("frame_size", (640, 480))
    return ArenaCalibration(corners=corners, **kw)


def _one_frame(*points, pad=(320.0, 240.0)):
    """One frame of four keypoints, padded to length with an in-arena point.

    The pad matters: it must be inside whichever arena the test uses, or the
    quorum blanks the frame and per-keypoint assertions read 0 for the wrong
    reason. (320, 240) is the centre of SQUARE; use OBLIQUE_CENTRE_PX for
    HORIZON_CORNERS.
    """
    pts = list(points) + [list(pad)] * (4 - len(points))
    return np.array([pts], dtype=float)


def _write_track(
    tmp_path,
    *,
    name="t1_d1DLC_exp-7.csv",
    resolution=(640, 480),
    fps=30.0,
    gate_block=None,
    outside=False,
    pad=(320.0, 240.0),
):
    """A real pose CSV plus its sidecar, so gate_pose_csv has something to read.

    ``pad`` moves the in-arena keypoints, which is how a second call stands in
    for a *re-run of inference*: same video, same output path, different
    coordinates.
    """
    from glider.vision.pose.dlc import to_dlc_csv

    xy = _one_frame([-900.0, -900.0], pad=pad) if outside else _one_frame(pad=pad)
    pose = _pose(np.repeat(xy, 10, axis=0))
    pose.fps = fps
    pose.metadata["resolution"] = list(resolution)
    if gate_block:
        pose.metadata["arena_gate"] = gate_block
    csv = tmp_path / name
    to_dlc_csv(pose, csv)
    return csv


class TestGeometry:
    def test_a_centred_keypoint_is_inside(self):
        out, report = gate_to_arena(_pose(_one_frame()), _arena())
        assert report.keypoints_masked == 0
        assert np.isfinite(out.xy).all()

    def test_a_keypoint_just_outside_survives_the_default_margin(self):
        """Default margin is 7.5 cm on a 30 cm arena; 13.33 px/cm. 5 cm past
        the left wall is x = 120 - 66.7, which is a rear, not a glitch."""
        _, report = gate_to_arena(_pose(_one_frame([53.3, 240.0])), _arena())
        assert report.keypoints_masked == 0

    def test_a_keypoint_well_outside_is_masked(self):
        """10 cm out exceeds the 7.5 cm margin. Only ONE keypoint moves, so the
        frame survives the quorum and the mask is per-keypoint -- putting all
        four outside would blank the frame and report keypoints_masked == 0."""
        _, report = gate_to_arena(_pose(_one_frame([120 - 133.3, 240.0])), _arena())
        assert report.keypoints_masked == 1
        assert report.frames_blanked == 0

    def test_a_masked_stray_loses_its_confidence_too(self):
        """Spec: both stages spell rejection the same way. A NaN position with
        a 0.9 likelihood is a contradictory row a DLC reader will trust."""
        out, _ = gate_to_arena(_pose(_one_frame([120 - 133.3, 240.0])), _arena())
        assert np.isnan(out.xy[0, 0]).all()
        assert out.confidence[0, 0] == 0.0

    def test_a_point_near_the_vanishing_line_is_masked(self):
        """As w approaches 0 the projected centimetres blow up. inf compares
        correctly against the margin, so the rectangle test catches this on its
        own -- no separate horizon guard is needed or wanted (see below)."""
        pose = _pose(_one_frame([320.0, 95.0], pad=OBLIQUE_CENTRE_PX))
        _, report = gate_to_arena(pose, _arena(corners=HORIZON_CORNERS))
        assert report.keypoints_masked == 1

    def test_a_steeply_oblique_arena_does_not_gate_its_own_interior(self):
        """Regression guard against a tempting bug. A homography is defined up
        to scale, so `w` can be negative across the WHOLE arena -- it is -1.42
        at the centre of this one. Rejecting points on `w <= 0` would therefore
        blank every frame of every video on a rig like this."""
        pose = _pose(_one_frame(pad=OBLIQUE_CENTRE_PX))
        _, report = gate_to_arena(pose, _arena(corners=HORIZON_CORNERS))
        assert report.keypoints_masked == 0
        assert report.frames_blanked == 0


class TestResolution:
    def test_the_pose_resolution_wins_over_the_arena(self):
        """The arena records where the corners were clicked; the pose records
        what the video was tracked at. Using the wrong one skews every point."""
        pose = _pose(_one_frame())
        pose.xy[:] = [640.0, 480.0]
        pose.metadata["resolution"] = [1280, 960]
        _, report = gate_to_arena(pose, _arena())
        assert report.keypoints_masked == 0

    def test_an_explicit_resolution_beats_the_arena(self):
        pose = _pose(np.full((1, 4, 2), [640.0, 480.0]))
        _, report = gate_to_arena(pose, _arena(), resolution=(1280, 960))
        assert report.keypoints_masked == 0

    def test_it_refuses_rather_than_guessing(self):
        """from_dlc_csv populates no metadata, so a CSV-loaded track has no
        resolution of its own. Falling through silently would gate the wrong
        region."""
        with pytest.raises(ValueError, match="resolution"):
            gate_to_arena(_pose(_one_frame()), _arena(frame_size=(0, 0)))


class TestReport:
    def test_it_names_the_keypoints_it_masked(self):
        _, report = gate_to_arena(_pose(_one_frame([-900.0, -900.0])), _arena())
        assert report.masked_by_keypoint["a"] == 1
        assert report.masked_by_keypoint["b"] == 0

    def test_it_records_the_settings_and_the_arena(self):
        settings = ArenaGateSettings(margin_cm=2.0)
        _, report = gate_to_arena(_pose(_one_frame()), _arena(), settings=settings)
        assert report.settings == settings
        assert len(report.arena_corners) == 4

    def test_an_explicit_margin_overrides_the_default(self):
        """5 cm out survives the 7.5 cm default but not a 2 cm margin."""
        pose = _pose(_one_frame([53.3, 240.0]))
        _, report = gate_to_arena(pose, _arena(), settings=ArenaGateSettings(margin_cm=2.0))
        assert report.keypoints_masked == 1

    def test_an_empty_pose_returns_a_zeroed_report(self):
        _, report = gate_to_arena(_pose(np.zeros((0, 4, 2))), _arena())
        assert report.frames_total == 0
        assert report.blanked_fraction == 0.0

    def test_a_degenerate_arena_propagates(self):
        """Spec: callers catch and skip, mirroring _score_zones. The gate does
        not silently pass the pose through."""
        from glider.vision.arena import DegenerateArenaError

        with pytest.raises(DegenerateArenaError):
            gate_to_arena(_pose(_one_frame()), _arena(corners=[(0.5, 0.5)] * 4))


class TestDetected:
    def test_ultralytics_zero_padding_is_not_detected(self):
        """Raw YOLO pads unlocalized keypoints with (0,0) at confidence 0.
        Counting those as detected would mask them as out-of-arena and, once
        the quorum lands, blank any frame under half localized."""
        xy = _one_frame([0.0, 0.0])
        conf = np.array([[0.0, 0.9, 0.9, 0.9]])
        _, report = gate_to_arena(_pose(xy, conf), _arena())
        assert report.keypoints_masked == 0

    def test_a_raw_track_and_its_masked_equivalent_gate_identically(self):
        """Same settings must mean the same thing on the inference path (raw,
        zero-padded) and the post-hoc path (an already NaN-masked CSV)."""
        conf = np.array([[0.0, 0.9, 0.9, 0.9]])
        raw = _one_frame([0.0, 0.0])
        masked = raw.copy()
        masked[0, 0] = np.nan

        _, from_raw = gate_to_arena(_pose(raw, conf), _arena())
        _, from_masked = gate_to_arena(_pose(masked, conf), _arena())
        assert from_raw.keypoints_masked == from_masked.keypoints_masked
        assert from_raw.frames_considered == from_masked.frames_considered

    def test_a_uniform_confidence_track_warns_once(self, caplog):
        """A model with no keypoint confidences gets np.ones (core.py:428), so
        (0,0) pads read as detected. Say so rather than gating silently."""
        pose = _pose(_one_frame(), confidence=np.ones((1, 4)))
        with caplog.at_level("WARNING"):
            gate_to_arena(pose, _arena())
        assert caplog.text.count("uniformly 1.0") == 1


class TestQuorum:
    def test_an_occluded_but_in_arena_frame_survives(self):
        """3 of 4 localized, all inside. Legitimate occlusion, not a glitch."""
        xy = np.full((1, 4, 2), np.nan)
        xy[0, :3] = [320.0, 240.0]
        _, report = gate_to_arena(_pose(xy), _arena())
        assert report.frames_blanked == 0

    def test_a_relocated_skeleton_is_blanked_whole(self):
        """3 of 4 detected keypoints outside: below min_inside_fraction=0.5."""
        xy = _one_frame([-900.0, -900.0], [-900.0, -900.0], [-900.0, -900.0])
        out, report = gate_to_arena(_pose(xy), _arena())
        assert report.frames_blanked == 1
        assert np.isnan(out.xy[0]).all()
        assert (out.confidence[0] == 0).all()

    def test_keypoints_in_a_blanked_frame_are_not_counted_as_strays(self):
        """They were discarded by the frame verdict, not by their position;
        counting both would double-report the same rejection."""
        xy = _one_frame([-900.0, -900.0], [-900.0, -900.0], [-900.0, -900.0])
        _, report = gate_to_arena(_pose(xy), _arena())
        assert report.keypoints_masked == 0

    def test_a_zero_detection_frame_is_excluded_not_blanked(self):
        xy = np.full((2, 4, 2), np.nan)
        xy[0] = [320.0, 240.0]
        _, report = gate_to_arena(_pose(xy), _arena())
        assert report.frames_total == 2
        assert report.frames_considered == 1
        assert report.frames_blanked == 0
        assert report.blanked_fraction == 0.0

    def test_blanked_fraction_is_zero_when_nothing_was_considered(self):
        _, report = gate_to_arena(_pose(np.full((5, 4, 2), np.nan)), _arena())
        assert report.frames_considered == 0
        assert report.blanked_fraction == 0.0


class TestDetectedFraction:
    def test_it_is_off_by_default(self):
        """A partial-but-in-arena frame survives. Default-on would blank every
        legitimately occluded frame in every cohort."""
        xy = np.full((1, 4, 2), np.nan)
        xy[0, :1] = [320.0, 240.0]
        _, report = gate_to_arena(_pose(xy), _arena())
        assert report.frames_blanked == 0

    def test_at_one_it_reproduces_the_prototype(self):
        """min_detected_fraction=1.0 blanks any incomplete frame, which is what
        reject_partial_frames(min_keypoints=None) did."""
        xy = np.full((1, 4, 2), np.nan)
        xy[0, :3] = [320.0, 240.0]
        _, report = gate_to_arena(
            _pose(xy), _arena(), settings=ArenaGateSettings(min_detected_fraction=1.0)
        )
        assert report.frames_blanked == 1

    def test_a_complete_in_arena_frame_survives_at_one(self):
        _, report = gate_to_arena(
            _pose(_one_frame()),
            _arena(),
            settings=ArenaGateSettings(min_detected_fraction=1.0),
        )
        assert report.frames_blanked == 0


class TestInsideFraction:
    def test_it_scores_a_fully_in_arena_detection_as_one(self):
        from glider.vision.arena_gate import inside_fraction

        xy = np.full((4, 2), [320.0, 240.0])
        conf = np.full(4, 0.9)
        assert inside_fraction(_arena(), xy, conf, (640, 480)) == 1.0

    def test_padding_does_not_drag_the_score_down(self):
        """The same (0,0) trap, at candidate-selection time: a good detection
        with one pad must not score 3/4 and lose to a bench-floor blob."""
        from glider.vision.arena_gate import inside_fraction

        xy = np.array([[0.0, 0.0], [320.0, 240.0], [320.0, 240.0], [320.0, 240.0]])
        conf = np.array([0.0, 0.9, 0.9, 0.9])
        assert inside_fraction(_arena(), xy, conf, (640, 480)) == 1.0

    def test_a_detection_with_nothing_detected_scores_zero(self):
        from glider.vision.arena_gate import inside_fraction

        assert inside_fraction(_arena(), np.zeros((4, 2)), np.zeros(4), (640, 480)) == 0.0


class TestPostHoc:
    def test_it_preserves_the_original_and_its_sidecar(self, tmp_path):
        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())
        assert (tmp_path / f"{csv.stem}_ungated.csv").exists()
        assert (tmp_path / f"{csv.stem}_ungated.meta.json").exists()

    def test_the_sidecar_keeps_its_resolution(self, tmp_path):
        """to_dlc_csv rebuilds the sidecar from pose.metadata and from_dlc_csv
        populates none, so a naive round trip destroys resolution -- breaking
        the viewer and making a second pass refuse."""
        csv = _write_track(tmp_path, resolution=(640, 480))
        gate_pose_csv(csv, _arena())
        assert read_pose_meta(csv)["resolution"] == [640, 480]

    def test_it_keeps_the_fps(self, tmp_path):
        csv = _write_track(tmp_path, fps=25.0)
        gate_pose_csv(csv, _arena())
        assert read_pose_meta(csv)["fps"] == 25.0

    def test_a_second_pass_with_the_same_settings_is_skipped(self, tmp_path):
        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())
        before = csv.read_bytes()
        gate_pose_csv(csv, _arena())
        assert csv.read_bytes() == before

    def test_the_skip_compares_corners_by_value(self, tmp_path):
        """arena_corners is declared list[tuple] but comes back from JSON as a
        list of lists, so an identity comparison never matches and the skip
        never fires. Asserted against a block that has actually been through
        the sidecar, since that round trip is what breaks the comparison."""
        from glider.vision.arena_gate import _same_gate

        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())
        stored = read_pose_meta(csv)["arena_gate"]
        assert _same_gate(stored, ArenaGateSettings(), _arena())

    def test_the_skip_does_not_re_read_the_track(self, tmp_path, monkeypatch):
        """The one observable consequence of the skip actually firing.

        Re-gating from ``_ungated`` with the same settings produces byte-
        identical output, so no comparison of file contents can tell a skip
        from a repeat -- which is exactly how a broken comparison would hide,
        rewriting every CSV in the cohort on every pass while looking correct.
        """
        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())

        def boom(*args, **kwargs):
            raise AssertionError("re-read the track instead of skipping it")

        monkeypatch.setattr("glider.vision.pose.dlc.from_dlc_csv", boom)
        gate_pose_csv(csv, _arena())

    def test_a_skipped_pass_returns_the_recorded_report(self, tmp_path):
        """Rehydrated from the block, so a caller logging blanked_fraction gets
        the same number on the second pass as on the first."""
        csv = _write_track(tmp_path, outside=True)
        first = gate_pose_csv(csv, _arena())
        assert gate_pose_csv(csv, _arena()) == first

    def test_changed_settings_regate_from_the_original(self, tmp_path):
        """The documented workflow: run with defaults, read the report,
        escalate. The second run must not eat the true original."""
        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())
        original = (tmp_path / f"{csv.stem}_ungated.csv").read_bytes()
        gate_pose_csv(csv, _arena(), settings=ArenaGateSettings(min_detected_fraction=1.0))
        assert (tmp_path / f"{csv.stem}_ungated.csv").read_bytes() == original

    def test_it_refuses_an_inference_gated_primary(self, tmp_path):
        """A primary gated by run_batch has a gate block and no _ungated twin.
        Renaming it would make 'the original survives' false."""
        csv = _write_track(tmp_path, gate_block={"gated": True, "settings": {}})
        with pytest.raises(ValueError, match="_raw"):
            gate_pose_csv(csv, _arena(), settings=ArenaGateSettings(min_detected_fraction=1.0))

    def test_it_refuses_an_ungated_that_outlived_its_primary(self, tmp_path):
        """Re-running inference leaves the old ``_ungated`` beside a primary it
        no longer describes, and "always gate from the pristine original" would
        then gate the *previous* run and write it over the new one."""
        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())
        _write_track(tmp_path, pad=(200.0, 200.0))  # inference re-run, overwrite=True
        with pytest.raises(ValueError, match="_ungated"):
            gate_pose_csv(csv, _arena())

    def test_the_refusal_leaves_the_new_run_where_it_was(self, tmp_path):
        """The damage the refusal exists to prevent: an hour of inference
        silently replaced by the gated remains of the run before it."""
        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())
        fresh = _write_track(tmp_path, pad=(200.0, 200.0)).read_bytes()
        with pytest.raises(ValueError):
            gate_pose_csv(csv, _arena())
        assert csv.read_bytes() == fresh

    def test_a_sidecar_less_csv_still_gates(self, tmp_path):
        """write_pose_meta is best-effort and DEFAULT_FPS exists for exactly
        these files, so a missing sidecar must not end the pass."""
        csv = _write_track(tmp_path)
        meta_path(csv).unlink()
        gate_pose_csv(csv, _arena(), settings=ArenaGateSettings(margin_cm=7.5))
        assert (tmp_path / f"{csv.stem}_ungated.csv").exists()
