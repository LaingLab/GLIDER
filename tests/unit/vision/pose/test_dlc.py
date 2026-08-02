"""Round-trip tests for the DLC CSV converter."""

from __future__ import annotations

import numpy as np
import pytest

from glider.vision.pose import dlc as dlc_io


def test_dlc_csv_roundtrip(synthetic_pose, tmp_path):
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(synthetic_pose, out)
    assert out.exists()

    restored = dlc_io.from_dlc_csv(out, fps=synthetic_pose.fps)
    np.testing.assert_allclose(restored.xy, synthetic_pose.xy, rtol=1e-5)
    np.testing.assert_allclose(restored.confidence, synthetic_pose.confidence, rtol=1e-5)
    assert restored.keypoint_names == synthetic_pose.keypoint_names
    assert restored.source == synthetic_pose.source


def test_dlc_csv_header_format(synthetic_pose, tmp_path):
    """Verify the 3 header rows match DLC convention."""
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(synthetic_pose, out)
    with out.open() as f:
        lines = [next(f) for _ in range(4)]
    assert lines[0].startswith("scorer,")
    assert lines[1].startswith("bodyparts,")
    assert lines[2].startswith("coords,")
    # First data row starts with frame index "0"
    assert lines[3].startswith("0,")


def test_dlc_h5_roundtrip(synthetic_pose, tmp_path):
    pytest.importorskip("tables")
    out = tmp_path / "out.h5"
    dlc_io.to_dlc_h5(synthetic_pose, out)
    assert out.exists()


def test_dlc_roundtrip_preserves_nans(kpt_names, tmp_path):
    """NaNs (no-detection frames) must survive a round trip."""
    from glider.vision.pose.core import PoseData

    xy = np.full((10, len(kpt_names), 2), 100.0)
    xy[3] = np.nan
    cf = np.ones((10, len(kpt_names)))
    cf[3] = 0.0
    pose = PoseData(xy=xy, confidence=cf, keypoint_names=kpt_names)

    out = tmp_path / "nans.csv"
    dlc_io.to_dlc_csv(pose, out)
    restored = dlc_io.from_dlc_csv(out)
    assert np.isnan(restored.xy[3]).all()
    assert (restored.confidence[3] == 0.0).all()


# --------------------------------------------------------------------------
# Frame-rate sidecar
#
# The DLC header has no room for a frame rate, but every windowed feature
# downstream is specified in seconds, so a lost rate silently rescales the
# analysis. to_dlc_csv drops a sidecar; from_dlc_csv reads it back.
# --------------------------------------------------------------------------


def test_to_dlc_csv_records_the_frame_rate_in_a_sidecar(kpt_names, tmp_path):
    from glider.vision.pose.core import PoseData

    pose = PoseData(
        xy=np.zeros((10, len(kpt_names), 2)),
        confidence=np.ones((10, len(kpt_names))),
        keypoint_names=kpt_names,
        fps=59.94,
    )
    out = tmp_path / "session01DLC_exp-6.csv"
    dlc_io.to_dlc_csv(pose, out)

    meta = dlc_io.read_pose_meta(out)
    assert meta is not None
    assert meta["fps"] == pytest.approx(59.94)
    assert meta["keypoint_names"] == kpt_names
    assert dlc_io.meta_path(out) == tmp_path / "session01DLC_exp-6.meta.json"


def test_from_dlc_csv_recovers_the_recorded_rate(kpt_names, tmp_path):
    """A 60 fps recording must not read back as 30."""
    from glider.vision.pose.core import PoseData

    pose = PoseData(
        xy=np.zeros((10, len(kpt_names), 2)),
        confidence=np.ones((10, len(kpt_names))),
        keypoint_names=kpt_names,
        fps=60.0,
    )
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(pose, out)

    assert dlc_io.from_dlc_csv(out).fps == pytest.approx(60.0)


def test_explicit_fps_overrides_the_sidecar(synthetic_pose, tmp_path):
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(synthetic_pose, out)

    assert dlc_io.from_dlc_csv(out, fps=25.0).fps == pytest.approx(25.0)


def test_csv_without_a_sidecar_still_reads_at_the_old_default(synthetic_pose, tmp_path):
    """CSVs written before the sidecar existed must behave exactly as before."""
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(synthetic_pose, out, write_meta=False)

    assert not dlc_io.meta_path(out).exists()
    assert dlc_io.from_dlc_csv(out).fps == pytest.approx(dlc_io.DEFAULT_FPS)


def test_corrupt_sidecar_is_treated_as_absent(synthetic_pose, tmp_path):
    """The CSV is still readable, so a bad sidecar must not fail the read."""
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(synthetic_pose, out)
    dlc_io.meta_path(out).write_text("{not json")

    assert dlc_io.fps_for_csv(out) is None
    assert dlc_io.from_dlc_csv(out).fps == pytest.approx(dlc_io.DEFAULT_FPS)


def test_fps_for_csv_distinguishes_unknown_from_thirty(synthetic_pose, tmp_path):
    """None means 'nobody recorded it' — callers warn on that, not on 30 fps."""
    known = tmp_path / "known.csv"
    unknown = tmp_path / "unknown.csv"
    dlc_io.to_dlc_csv(synthetic_pose, known)
    dlc_io.to_dlc_csv(synthetic_pose, unknown, write_meta=False)

    assert dlc_io.fps_for_csv(known) == pytest.approx(30.0)
    assert dlc_io.fps_for_csv(unknown) is None


# --------------------------------------------------------------------------
# Frame-size sidecar
#
# Pose coordinates are pixels in the source video. Anything drawing them
# *without* that video — the session review window — needs the canvas they
# were measured on, and must be told when nobody recorded it rather than be
# handed a guess.
# --------------------------------------------------------------------------


def _pose(kpt_names, **metadata):
    from glider.vision.pose.core import PoseData

    return PoseData(
        xy=np.zeros((10, len(kpt_names), 2)),
        confidence=np.ones((10, len(kpt_names))),
        keypoint_names=kpt_names,
        fps=30.0,
        metadata=metadata,
    )


def test_the_resolution_is_recorded_when_the_producer_knew_it(kpt_names, tmp_path):
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(_pose(kpt_names, resolution=(1280, 960)), out)

    assert dlc_io.read_pose_meta(out)["resolution"] == [1280, 960]
    assert dlc_io.resolution_for_csv(out) == (1280, 960)


def test_an_unknown_resolution_is_omitted_rather_than_guessed(kpt_names, tmp_path):
    """Inferring it from the coordinate range would shrink the arena to
    wherever the animal happened to go."""
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(_pose(kpt_names), out)

    assert "resolution" not in dlc_io.read_pose_meta(out)
    assert dlc_io.resolution_for_csv(out) is None


@pytest.mark.parametrize("bad", [None, (0, 480), (640, 0), ("a", "b"), (640,), 640])
def test_a_nonsense_resolution_is_refused_not_written(kpt_names, tmp_path, bad):
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(_pose(kpt_names, resolution=bad), out)

    assert dlc_io.resolution_for_csv(out) is None


def test_resolution_for_a_csv_with_no_sidecar_is_none(synthetic_pose, tmp_path):
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(synthetic_pose, out, write_meta=False)

    assert dlc_io.resolution_for_csv(out) is None


def test_backfill_adds_a_resolution_to_an_older_sidecar(kpt_names, tmp_path):
    """Re-running hours of inference to recover a number the video already
    knows would be absurd."""
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(_pose(kpt_names), out)

    assert dlc_io.backfill_resolution(out, (800, 600)) is True
    assert dlc_io.resolution_for_csv(out) == (800, 600)
    # The rest of the sidecar survives the rewrite.
    assert dlc_io.fps_for_csv(out) == pytest.approx(30.0)
    assert dlc_io.read_pose_meta(out)["keypoint_names"] == kpt_names


def test_backfill_refuses_when_there_is_no_sidecar_to_amend(synthetic_pose, tmp_path):
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(synthetic_pose, out, write_meta=False)

    assert dlc_io.backfill_resolution(out, (800, 600)) is False
    assert not dlc_io.meta_path(out).exists()


@pytest.mark.parametrize("bad", [(0, 600), (800, -1), ("x", "y"), None])
def test_backfill_refuses_a_nonsense_resolution(kpt_names, tmp_path, bad):
    out = tmp_path / "out.csv"
    dlc_io.to_dlc_csv(_pose(kpt_names, resolution=(640, 480)), out)

    assert dlc_io.backfill_resolution(out, bad) is False
    assert dlc_io.resolution_for_csv(out) == (640, 480)  # unchanged
