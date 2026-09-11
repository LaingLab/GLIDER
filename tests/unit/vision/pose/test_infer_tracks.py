"""Multi-animal inference, with ultralytics replaced by a stub.

The seam is _track_stream, matching how backend.py already isolates _load_yolo:
the streaming loop, the candidate ranking and the hand-off to consolidation are
all testable without a model, a GPU or a video file."""

import numpy as np
import pytest

from glider.vision.pose import core
from glider.vision.pose.core import _rank_candidates, infer_video_tracks

NAMES = ["snout", "tail"]


class FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __getitem__(self, i):
        return FakeTensor(self._array[i])

    @property
    def shape(self):
        return self._array.shape

    def __len__(self):
        return len(self._array)


class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = FakeTensor(xy)
        self.conf = FakeTensor(conf)


class FakeBoxes:
    def __init__(self, conf, ids):
        self.conf = FakeTensor(np.asarray(conf))
        self.id = None if ids is None else FakeTensor(np.asarray(ids))


class FakeResult:
    """One frame: `animals` is [(track_id, x, y, box_conf), ...]."""

    def __init__(self, animals):
        if not animals:
            self.keypoints = FakeKeypoints(np.zeros((0, 2, 2)), np.zeros((0, 2)))
            self.boxes = FakeBoxes([], None)
            return
        xy = np.array([[[x, y], [x, y]] for _, x, y, _ in animals], dtype=float)
        self.keypoints = FakeKeypoints(xy, np.ones((len(animals), 2)))
        self.boxes = FakeBoxes([c for *_, c in animals], [t for t, *_ in animals])


class FakeSpec:
    kind = "yolo"


@pytest.fixture
def stub_stream(monkeypatch):
    """Install a fake frame stream and stub out everything that touches disk.

    identify_pose_model and resolve_device are imported *inside*
    infer_video_tracks, so they must be patched where they are defined --
    patching core's attributes would not be seen by a local import.
    """

    def install(frames):
        monkeypatch.setattr(core, "_track_stream", lambda *a, **k: iter(frames))
        monkeypatch.setattr(core, "_video_fps", lambda p: 30.0)
        monkeypatch.setattr("glider.vision.pose.spec.identify_pose_model", lambda p: FakeSpec())
        monkeypatch.setattr(
            "glider.vision.pose.device.resolve_device", lambda d, require_gpu=False: "cpu"
        )
        monkeypatch.setattr("glider.vision.video_source.video_resolution", lambda p: (640, 480))

    return install


def run(n_animals=2, **kw):
    return infer_video_tracks(
        "model.pt",
        "video.mp4",
        NAMES,
        n_animals=n_animals,
        fps=30.0,
        progress=False,
        echo_device=False,
        **kw,
    )


def test_two_animals_become_two_slots(stub_stream):
    stub_stream([FakeResult([(1, 10, 10, 0.9), (2, 300, 300, 0.8)]) for _ in range(20)])
    tracks = run()
    assert tracks.n_animals == 2
    assert tracks.n_frames == 20
    assert np.allclose(tracks[0].xy[0], 10.0)


def test_a_frame_with_no_detections_is_nan_for_everyone(stub_stream):
    frames = [FakeResult([(1, 10, 10, 0.9), (2, 300, 300, 0.8)]) for _ in range(20)]
    frames[5] = FakeResult([])
    stub_stream(frames)
    tracks = run()
    assert np.all(np.isnan(tracks[0].xy[5]))


def test_more_detections_than_animals_keeps_the_best_n(stub_stream):
    stub_stream(
        [FakeResult([(1, 10, 10, 0.9), (2, 300, 300, 0.8), (3, 600, 600, 0.2)]) for _ in range(20)]
    )
    tracks = run(n_animals=2)
    assert tracks.n_animals == 2
    xs = {float(tracks[s].xy[0, 0, 0]) for s in tracks}
    assert 600.0 not in xs


def test_a_detection_without_a_track_id_is_skipped(stub_stream):
    # ByteTrack leaves id None on unconfirmed detections. Inventing an id for
    # them would create a fragment per frame.
    frames = [FakeResult([(1, 10, 10, 0.9)]) for _ in range(20)]
    frames[3].boxes.id = None
    stub_stream(frames)
    tracks = run(n_animals=1)
    assert np.all(np.isnan(tracks[0].xy[3]))


def test_cancel_lands_within_one_frame(stub_stream):
    from glider.vision.pose.core import PoseCancelledError

    stub_stream([FakeResult([(1, 10, 10, 0.9)]) for _ in range(1000)])
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(PoseCancelledError):
        run(n_animals=1, cancel_cb=cancel)


def test_metadata_records_the_consolidation_knobs(stub_stream):
    stub_stream([FakeResult([(1, 10, 10, 0.9)]) for _ in range(20)])
    tracks = run(n_animals=1, max_travel_px_per_frame=25.0)
    assert tracks.metadata["consolidation"]["max_travel_px_per_frame"] == 25.0
    assert tracks.metadata["consolidation"]["n_animals"] == 1


def test_ranking_first_matches_argmax_including_on_a_tie():
    # _pick_candidate used to be argmax, which returns the FIRST maximum.
    # _rank_candidates[0] has to agree, or the single-animal path shifts.
    result = FakeResult([(1, 0, 0, 0.5), (2, 9, 9, 0.5), (3, 4, 4, 0.1)])
    confidences = np.array([0.5, 0.5, 0.1])
    assert _rank_candidates(result, confidences, None, None, None)[0] == int(confidences.argmax())
