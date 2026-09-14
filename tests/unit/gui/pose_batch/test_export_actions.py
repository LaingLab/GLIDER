"""export_actions: deriving the four-row DLC CSV from per-animal files.

Qt-free -- these exercise the same walk-over-videos idiom as
``arena_actions.regate_videos`` (see ``test_window_arena.py``), just with no
window or qtbot needed.
"""

from __future__ import annotations

import numpy as np

from glider.gui.pose_batch.export_actions import export_sessions, exportable
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import from_dlc_csv, to_dlc_csv

NAMES = ["snout", "tail"]


def _pose(fill, n_frames=5, keypoints=NAMES, fps=30.0):
    return PoseData(
        xy=np.full((n_frames, len(keypoints), 2), float(fill)),
        confidence=np.full((n_frames, len(keypoints)), 0.9),
        keypoint_names=list(keypoints),
        fps=fps,
        source="yolo_test",
    )


def _video(tmp_path, name="session01.mp4"):
    path = tmp_path / name
    path.write_bytes(b"")
    return path


def _write_animals(video, poses, *, model="exp-7"):
    """Per-animal CSVs where ``find_pose_csvs`` looks for them."""
    animal_dir = video.parent / f"{video.stem}DLC_{model}_animals"
    for slot, pose in enumerate(poses):
        to_dlc_csv(pose, animal_dir / f"animal{slot}.csv")
    return animal_dir


def test_two_animals_export_a_four_row_file_matching_the_per_animal_coords(tmp_path):
    video = _video(tmp_path)
    poses = [_pose(1), _pose(2)]
    animal_dir = _write_animals(video, poses)

    exported, skipped = export_sessions([video], on_log=lambda m: None)

    assert (exported, skipped) == (1, 0)
    # The primary path animals_dir was named from -- never written by the
    # tracking path itself, so this is the only writer of it.
    target = animal_dir.with_name(animal_dir.name.removesuffix("_animals") + ".csv")
    assert target.exists()

    for slot, pose in enumerate(poses):
        back = from_dlc_csv(target, individual=slot)
        assert np.allclose(back.xy, pose.xy)
        assert back.keypoint_names == pose.keypoint_names


def test_export_logs_what_it_wrote(tmp_path):
    video = _video(tmp_path)
    _write_animals(video, [_pose(1), _pose(2)])

    logged = []
    export_sessions([video], on_log=logged.append)

    assert len(logged) == 1
    assert video.name in logged[0]
    assert "wrote" in logged[0]


def test_a_single_animal_session_reports_nothing_to_export(tmp_path):
    video = _video(tmp_path)
    # An untracked video: no _animals directory, nothing to combine.
    video.parent.mkdir(parents=True, exist_ok=True)

    logged = []
    exported, skipped = export_sessions([video], on_log=logged.append)

    assert (exported, skipped) == (0, 1)
    assert "nothing to export" in logged[0]
    assert not list(video.parent.glob("*DLC_*.csv"))


def test_exportable_excludes_single_animal_sessions(tmp_path):
    multi = _video(tmp_path, "multi.mp4")
    _write_animals(multi, [_pose(1), _pose(2)])
    untracked = _video(tmp_path, "untracked.mp4")

    assert exportable([multi, untracked]) == [multi]


def test_disagreeing_per_animal_files_are_reported_not_raised(tmp_path):
    """Different frame counts -- PoseTracks refuses; this must surface that
    in English, not as a bare traceback, and must not write a bad file."""
    video = _video(tmp_path)
    animal_dir = _write_animals(video, [_pose(1, n_frames=5), _pose(2, n_frames=7)])

    logged = []
    exported, skipped = export_sessions([video], on_log=logged.append)

    assert (exported, skipped) == (0, 1)
    assert video.name in logged[0]
    assert "n_frames" in logged[0]  # PoseTracks' own message, carried through
    target = animal_dir.with_name(animal_dir.name.removesuffix("_animals") + ".csv")
    assert not target.exists()


def test_one_bad_session_does_not_end_the_pass(tmp_path):
    """The whole point of a batch walk is that it finishes."""
    good = _video(tmp_path, "good.mp4")
    _write_animals(good, [_pose(1), _pose(2)])
    bad = _video(tmp_path, "bad.mp4")
    _write_animals(bad, [_pose(1, n_frames=5), _pose(2, n_frames=7)])

    exported, skipped = export_sessions([bad, good], on_log=lambda m: None)

    assert (exported, skipped) == (1, 1)


def test_progress_is_reported_per_video(tmp_path):
    good = _video(tmp_path, "good.mp4")
    _write_animals(good, [_pose(1), _pose(2)])
    untracked = _video(tmp_path, "untracked.mp4")

    seen = []
    export_sessions([good, untracked], on_progress=lambda i, n: seen.append((i, n)))

    assert seen == [(1, 2), (2, 2)]
