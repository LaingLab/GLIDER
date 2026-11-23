"""Live identity run -> Batch Pose layout."""

from __future__ import annotations

import logging

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


def _write(tmp_path, rows):
    """rows: (frame, object_id, animal, flag, nose_x)."""
    t = ["# GLIDER Tracking Data", "# Experiment,x", "", TRACK_HDR]
    k = ["frame,timestamp,elapsed_ms,flow_elapsed_ms,object_id,keypoint,x,y,confidence"]
    for frame, oid, animal, flag, nx in rows:
        t.append(f"{frame},0,0,0,{oid},mouse,0,0,1,1,.9,0,0,0,0,0,,0,{animal},{flag}")
        for name, dx in (("nose", nx), ("tail", nx + 1)):
            k.append(f"{frame},0,0,0,{oid},{name},{dx},2.0,0.9")
    t.append("# Total Frames,4")
    (tmp_path / "r_tracking.csv").write_text("\n".join(t) + "\n")
    (tmp_path / "r_tracking_keypoints.csv").write_text("\n".join(k) + "\n")
    video = tmp_path / "r.mp4"
    video.write_bytes(b"")
    return tmp_path / "r_tracking.csv", tmp_path / "r_tracking_keypoints.csv", video


def _run(tmp_path, rows, names):
    tr, kp, video = _write(tmp_path, rows)
    model = tmp_path / "m.pt"
    out = export_live_identity(tr, kp, video, model, names, KP, 30.0, 4)
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
            4,
        )
    assert out == [] and "Could not export" in caplog.text
