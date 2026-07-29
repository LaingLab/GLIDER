"""infer_video's optional progress_cb / cancel_cb hooks.

Ultralytics is stubbed into sys.modules, so these run without torch, a GPU, or
a real video file — infer_video imports YOLO inside the function body, which is
what makes the injection work.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from glider.vision.pose import core

NAMES = ["a", "b", "c"]


class _FakeTensor:
    """Minimal stand-in for a torch tensor: supports indexing, .cpu(), .numpy()."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def __getitem__(self, index):
        return _FakeTensor(self._array[index])

    @property
    def shape(self):
        return self._array.shape

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeKeypoints:
    def __init__(self, n_kpts):
        self.xy = _FakeTensor(np.zeros((1, n_kpts, 2)))
        self.conf = _FakeTensor(np.ones((1, n_kpts)))


class _FakeResult:
    def __init__(self, n_kpts):
        self.keypoints = _FakeKeypoints(n_kpts)
        self.boxes = None


class _FakeYOLO:
    frames = 5

    def __init__(self, path):
        self.path = path

    def to(self, device):
        return self

    def predict(self, **kwargs):
        for _ in range(self.frames):
            yield _FakeResult(len(NAMES))


@pytest.fixture
def stub_yolo(monkeypatch):
    """Install a fake ultralytics module and neutralize device/fps probing."""
    module = types.ModuleType("ultralytics")
    module.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(core, "_video_fps", lambda path: 30.0)

    import glider.vision.pose.device as device_mod

    monkeypatch.setattr(device_mod, "resolve_device", lambda d, require_gpu=False: "cpu")
    return module


def _run(tmp_path, **kwargs):
    return core.infer_video(
        tmp_path / "model.pt",
        tmp_path / "video.mp4",
        NAMES,
        progress=False,
        echo_device=False,
        **kwargs,
    )


def test_progress_cb_receives_monotonic_frame_counts(stub_yolo, tmp_path):
    seen: list[int] = []
    _run(tmp_path, progress_cb=lambda done, total: seen.append(done))
    assert seen == [1, 2, 3, 4, 5]


def test_progress_cb_reports_a_frame_total(stub_yolo, tmp_path):
    totals: list[int] = []
    _run(tmp_path, progress_cb=lambda done, total: totals.append(total))
    # The video does not exist, so OpenCV cannot report a count: 0 means
    # "indeterminate" and callers must render it as such.
    assert set(totals) == {0}


def test_cancel_cb_raises_pose_cancelled(stub_yolo, tmp_path):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] >= 3

    with pytest.raises(core.PoseCancelledError):
        _run(tmp_path, cancel_cb=cancel)


def test_cancel_cb_returning_false_completes_normally(stub_yolo, tmp_path):
    pose = _run(tmp_path, cancel_cb=lambda: False)
    assert pose.n_frames == 5


def test_without_callbacks_behaviour_is_unchanged(stub_yolo, tmp_path):
    pose = _run(tmp_path)
    assert pose.n_frames == 5
    assert pose.n_keypoints == 3
    assert pose.keypoint_names == NAMES


def test_undetected_frames_still_advance_progress(stub_yolo, tmp_path, monkeypatch):
    """A frame with no detection must count toward progress like any other."""

    class _EmptyResult:
        keypoints = None
        boxes = None

    class _MixedYOLO(_FakeYOLO):
        def predict(self, **kwargs):
            yield _FakeResult(len(NAMES))
            yield _EmptyResult()
            yield _FakeResult(len(NAMES))

    stub_yolo.YOLO = _MixedYOLO
    seen: list[int] = []
    pose = _run(tmp_path, progress_cb=lambda done, total: seen.append(done))
    assert seen == [1, 2, 3]
    assert pose.n_frames == 3
    assert np.isnan(pose.xy[1]).all()
