"""infer_video over a DeepLabCut / SLEAP model folder.

Unlike the ultralytics path (which streams the whole video itself), this route
decodes frame by frame with OpenCV, so these tests write a real short video
rather than stubbing the reader. Only the onnxruntime session is faked — that
keeps the decode loop, the frame counting and the cancel path honest while
still running on a machine with no onnxruntime installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from glider.vision.pose import backend as backend_mod
from glider.vision.pose import core

N_FRAMES = 6
NAMES = ["snout", "left_ear", "right_ear"]


@pytest.fixture
def tiny_video(tmp_path):
    """A short MJPG clip; MJPG/.avi is the most portable writer across OSes."""
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(N_FRAMES):
        frame = np.full((24, 32, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def dlc_folder(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "glider_pose.json").write_text(
        json.dumps(
            {
                "kind": "dlc",
                "onnx": "model.onnx",
                "keypoint_names": NAMES,
                "output_stride": 8.0,
                "locref_stdev": 1.0,
            }
        )
    )
    (root / "model.onnx").write_bytes(b"stub")
    return root


class _FakeInput:
    name = "input"


class _FakeSession:
    """Returns a peak at (row 1, col 2) for every keypoint, on every frame."""

    def __init__(self):
        self.runs = 0

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, feed):
        self.runs += 1
        heat = np.zeros((1, len(NAMES), 3, 4), np.float32)
        heat[0, :, 1, 2] = 1.0
        locref = np.zeros((1, 2 * len(NAMES), 3, 4), np.float32)
        return [heat, locref]


@pytest.fixture
def fake_session(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(backend_mod, "_make_session", lambda spec: session)
    return session


def test_infer_video_runs_a_dlc_folder(tiny_video, dlc_folder, fake_session):
    pose = core.infer_video(dlc_folder, tiny_video, progress=False)

    assert pose.keypoint_names == NAMES
    assert pose.n_frames == N_FRAMES
    assert pose.n_keypoints == len(NAMES)
    assert fake_session.runs == N_FRAMES
    # Grid cell (1, 2) at stride 8 decodes to the cell centre.
    assert pose.xy[0, 0] == pytest.approx([2 * 8 + 4, 1 * 8 + 4])


def test_names_come_from_the_model_not_the_caller(tiny_video, dlc_folder, fake_session):
    pose = core.infer_video(dlc_folder, tiny_video, ["wrong", "names", "here"], progress=False)
    assert pose.keypoint_names == NAMES


def test_metadata_records_the_model_and_backend(tiny_video, dlc_folder, fake_session):
    pose = core.infer_video(dlc_folder, tiny_video, progress=False)
    assert pose.metadata["backend"] == "dlc"
    assert pose.metadata["model_path"].endswith("model.onnx")
    assert pose.metadata["resolution"] == (32, 24)
    assert pose.source.startswith("dlc_")


def test_fps_is_read_from_the_video(tiny_video, dlc_folder, fake_session):
    pose = core.infer_video(dlc_folder, tiny_video, progress=False)
    assert pose.fps == pytest.approx(10.0, abs=0.5)


def test_explicit_fps_overrides_the_container(tiny_video, dlc_folder, fake_session):
    pose = core.infer_video(dlc_folder, tiny_video, fps=60.0, progress=False)
    assert pose.fps == pytest.approx(60.0)


def test_progress_cb_fires_once_per_frame(tiny_video, dlc_folder, fake_session):
    seen: list[tuple[int, int]] = []
    core.infer_video(
        dlc_folder, tiny_video, progress=False, progress_cb=lambda d, t: seen.append((d, t))
    )
    assert [d for d, _ in seen] == list(range(1, N_FRAMES + 1))


def test_cancel_cb_raises_pose_cancelled(tiny_video, dlc_folder, fake_session):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] >= 3

    with pytest.raises(core.PoseCancelledError):
        core.infer_video(dlc_folder, tiny_video, progress=False, cancel_cb=cancel)
    # Cancel is checked before the frame is decoded, so it stops early.
    assert fake_session.runs < N_FRAMES


def test_low_confidence_keypoints_are_masked_to_nan(tiny_video, dlc_folder, monkeypatch):
    class _WeakSession(_FakeSession):
        def run(self, output_names, feed):
            self.runs += 1
            heat = np.zeros((1, len(NAMES), 3, 4), np.float32)
            heat[0, :, 1, 2] = 0.1  # below the 0.25 default threshold
            locref = np.zeros((1, 2 * len(NAMES), 3, 4), np.float32)
            return [heat, locref]

    monkeypatch.setattr(backend_mod, "_make_session", lambda spec: _WeakSession())
    pose = core.infer_video(dlc_folder, tiny_video, progress=False)
    assert np.isnan(pose.xy).all()
    assert pose.confidence == pytest.approx(np.full_like(pose.confidence, 0.1))


def test_yolo_without_names_is_still_an_error(tmp_path, tiny_video):
    from glider.vision.pose.spec import PoseModelError

    with pytest.raises(PoseModelError, match="keypoint_names"):
        core.infer_video(tmp_path / "best.pt", tiny_video, progress=False)
