"""Live identity run -> Batch Pose layout."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from glider.vision.identity_export import export_live_identity
from glider.vision.pose.batch import animals_dir, dlc_output_path, find_pose_csvs
from glider.vision.pose.dlc import from_dlc_csv

KP = ["nose", "tail"]
TRACK_HDR = (
    "frame,timestamp,elapsed_ms,flow_elapsed_ms,object_id,class,x,y,w,h,confidence,"
    "center_x,center_y,distance_px,distance_mm,cumulative_mm,zone_ids,velocity_px_frame,"
    "animal,identity_flag"
)


T0 = 1_700_000_000.0  # capture time of the video's first frame
FRAME_TS = [T0 + i / 30.0 for i in range(4)]  # a 4-frame video at 30 fps


def _iso_at(ts):
    return datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")


def _iso(video_frame):
    """Logger-style ISO timestamp of 0-based video frame ``video_frame`` at 30 fps."""
    return _iso_at(T0 + video_frame / 30.0)


def _write(tmp_path, rows, offset=0, ts_of=None):
    """rows: (log frame, object_id, animal, flag, nose_x).

    Log frame ``f`` was captured at video frame ``f - 1 + offset``.
    """
    t = ["# GLIDER Tracking Data", "# Experiment,x", "", TRACK_HDR]
    k = ["frame,timestamp,elapsed_ms,flow_elapsed_ms,object_id,keypoint,x,y,confidence"]
    for frame, oid, animal, flag, nx in rows:
        ts = ts_of(frame) if ts_of else _iso(frame - 1 + offset)
        t.append(f"{frame},{ts},0,0,{oid},mouse,0,0,1,1,.9,0,0,0,0,0,,0,{animal},{flag}")
        for name, dx in (("nose", nx), ("tail", nx + 1)):
            k.append(f"{frame},{ts},0,0,{oid},{name},{dx},2.0,0.9")
    t.append("# Total Frames,4")
    (tmp_path / "r_tracking.csv").write_text("\n".join(t) + "\n")
    (tmp_path / "r_tracking_keypoints.csv").write_text("\n".join(k) + "\n")
    video = tmp_path / "r.mp4"
    video.write_bytes(b"")
    return tmp_path / "r_tracking.csv", tmp_path / "r_tracking_keypoints.csv", video


def _run(tmp_path, rows, names, offset=0, frame_ts=FRAME_TS, ts_of=None):
    tr, kp, video = _write(tmp_path, rows, offset, ts_of)
    model = tmp_path / "m.pt"
    out = export_live_identity(tr, kp, video, model, names, KP, 30.0, frame_ts)
    return out, video, model


ROWS = [
    (1, 0, "A", "", 10),
    (1, 1, "B", "close", 50),
    (2, 0, "A", "close+stitched", 11),  # B hidden on frame 2
    (4, 0, "A", "", 13),
    (4, 1, "B", "", 53),
    (4, 2, "extra", "", 99),  # unnamed: not exported
]


def test_single_name_writes_primary_only(tmp_path):
    out, video, model = _run(tmp_path, [r for r in ROWS if r[1] == 0], ["A"])
    primary = dlc_output_path(video, model)
    assert out == [primary] and primary.exists()
    assert not animals_dir(primary).exists()


def test_two_names_batch_layout_and_alignment(tmp_path):
    out, video, model = _run(tmp_path, ROWS, ["A", "B"])
    primary = dlc_output_path(video, model)
    found = find_pose_csvs(video)
    assert found == [animals_dir(primary) / "animal0.csv", animals_dir(primary) / "animal1.csv"]
    b = from_dlc_csv(found[1])
    assert b.xy.shape[0] == 4
    assert b.xy[0, 0, 0] == 50 and b.xy[3, 0, 0] == 53  # frame - 1
    assert np.isnan(b.xy[1]).all() and np.isnan(b.xy[2]).all()  # no row


def test_identity_sidecar(tmp_path):
    out, video, model = _run(tmp_path, ROWS, ["A", "B"])
    side = (tmp_path / "rDLC_m_identity.csv").read_text().splitlines()
    assert side[0] == "frame,individual,identity_flag"
    assert set(side[1:]) == {
        "0,animal1,close",
        "1,animal0,close+stitched",
        "1,animal1,gap",
        "2,animal0,gap",
        "2,animal1,gap",
    }


def test_failure_is_logged_not_raised(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        out = export_live_identity(
            tmp_path / "no.csv",
            tmp_path / "no2.csv",
            tmp_path / "v.mp4",
            tmp_path / "m.pt",
            ["A"],
            KP,
            30.0,
            FRAME_TS,
        )
    assert out == [] and "Could not export" in caplog.text


def test_rows_land_on_video_frames_by_timestamp(tmp_path):
    # The logger started 2 frames before the video's first frame and skipped
    # some frames: log frame numbers no longer match video frames.
    rows = [
        (1, 0, "A", "", 1),  # video frame -2: before the video, dropped
        (2, 0, "A", "", 2),  # video frame -1: dropped
        (3, 0, "A", "", 10),  # video frame 0
        (5, 0, "A", "close", 12),  # video frame 2 (log frame 4 never logged)
        (7, 0, "A", "", 14),  # video frame 4: past the 4-frame video, dropped
    ]
    out, video, model = _run(tmp_path, rows, ["A"], offset=-2)
    a = from_dlc_csv(dlc_output_path(video, model))
    assert a.xy.shape[0] == 4
    assert a.xy[0, 0, 0] == 10 and a.xy[2, 0, 0] == 12
    assert np.isnan(a.xy[1]).all() and np.isnan(a.xy[3]).all()


def test_flags_land_on_video_frames_by_timestamp(tmp_path):
    rows = [(1, 0, "A", "", 1), (2, 0, "A", "close", 10), (2, 1, "B", "", 50)]
    _run(tmp_path, rows, ["A", "B"], offset=1)
    side = (tmp_path / "rDLC_m_identity.csv").read_text().splitlines()
    assert set(side[1:]) == {
        "0,animal0,gap",
        "0,animal1,gap",
        "1,animal1,gap",
        "2,animal0,close",
        "3,animal0,gap",
        "3,animal1,gap",
    }


def test_paused_run_rows_follow_the_video_frames(tmp_path):
    # Video frames 0-1, a 2 s pause (the recorder writes nothing), then 2-3.
    # The logger kept logging through the pause; those rows have no frame.
    frame_ts = [T0, T0 + 1 / 30, T0 + 2 + 2 / 30, T0 + 2 + 3 / 30]
    log_ts = {1: frame_ts[0], 2: frame_ts[1], 3: T0 + 1.0, 4: frame_ts[2], 5: frame_ts[3]}
    rows = [(f, 0, "A", "close" if f == 4 else "", 10 + f) for f in log_ts]
    rows.append((3, 1, "B", "close", 77))  # logged mid-pause only
    _run(tmp_path, rows, ["A", "B"], frame_ts=frame_ts, ts_of=lambda f: _iso_at(log_ts[f]))
    a = from_dlc_csv(tmp_path / "rDLC_m_animals" / "animal0.csv")
    assert a.xy.shape[0] == 4
    assert [a.xy[i, 0, 0] for i in range(4)] == [11, 12, 14, 15]  # log frame 3 dropped
    b = from_dlc_csv(tmp_path / "rDLC_m_animals" / "animal1.csv")
    assert np.isnan(b.xy).all()
    side = set((tmp_path / "rDLC_m_identity.csv").read_text().splitlines()[1:])
    assert "2,animal0,close" in side and not any(",animal1,close" in r for r in side)
