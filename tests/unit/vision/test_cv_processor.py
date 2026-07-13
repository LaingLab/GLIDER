"""Tests for CVProcessor: keypoint rendering, NCNN loading, settings."""

import numpy as np

import glider.vision.cv_processor as cvp
from glider.vision.cv_processor import (
    CVProcessor,
    CVSettings,
    DetectionBackend,
    TrackedObject,
    resolve_model_path,
)


def test_cvsettings_keypoint_fields_round_trip():
    """New keypoint-rendering settings survive to_dict/from_dict."""
    settings = CVSettings(
        show_keypoints=False,
        keypoint_radius=5,
        keypoint_color=(255, 0, 0),
        keypoint_min_confidence=0.6,
    )

    restored = CVSettings.from_dict(settings.to_dict())

    assert restored.show_keypoints is False
    assert restored.keypoint_radius == 5
    assert restored.keypoint_color == (255, 0, 0)
    assert restored.keypoint_min_confidence == 0.6


def test_cvsettings_keypoint_defaults():
    """Keypoints render by default so the live view shows them without setup."""
    settings = CVSettings()
    assert settings.show_keypoints is True
    assert settings.keypoint_radius > 0
    assert 0.0 <= settings.keypoint_min_confidence <= 1.0


def _tracked_with_keypoints(keypoints: np.ndarray) -> TrackedObject:
    return TrackedObject(
        track_id=1,
        class_name="mouse",
        bbox=(10, 10, 20, 20),
        confidence=0.9,
        centroid=(20, 20),
        keypoints=keypoints,
    )


def test_draw_overlays_renders_keypoint_dots():
    """A confident keypoint paints a dot of keypoint_color at its location."""
    settings = CVSettings(
        keypoint_color=(0, 0, 255),
        keypoint_radius=4,
        show_labels=False,
        behavior_enabled=False,
    )
    proc = CVProcessor(settings)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Keypoint at (70, 70) with high confidence, well clear of the bbox.
    kp = np.array([[70.0, 70.0, 0.95]], dtype=np.float32)
    tracked = [_tracked_with_keypoints(kp)]

    out = proc.draw_overlays(frame, [], tracked)

    assert tuple(int(c) for c in out[70, 70]) == (0, 0, 255)


def test_draw_overlays_skips_low_confidence_keypoints():
    """Keypoints below keypoint_min_confidence are not drawn."""
    settings = CVSettings(
        keypoint_min_confidence=0.5,
        show_labels=False,
        behavior_enabled=False,
    )
    proc = CVProcessor(settings)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    kp = np.array([[70.0, 70.0, 0.1]], dtype=np.float32)  # below threshold
    tracked = [_tracked_with_keypoints(kp)]

    out = proc.draw_overlays(frame, [], tracked)

    assert tuple(int(c) for c in out[70, 70]) == (0, 0, 0)


def test_draw_overlays_skips_zero_keypoints():
    """(0, 0) placeholder keypoints (undetected) are not drawn at the origin."""
    settings = CVSettings(show_labels=False, behavior_enabled=False)
    proc = CVProcessor(settings)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    kp = np.array([[0.0, 0.0, 0.99]], dtype=np.float32)
    tracked = [_tracked_with_keypoints(kp)]

    out = proc.draw_overlays(frame, [], tracked)

    assert tuple(int(c) for c in out[0, 0]) == (0, 0, 0)


def test_draw_overlays_respects_show_keypoints_toggle():
    """show_keypoints=False leaves the keypoint location untouched."""
    settings = CVSettings(
        show_keypoints=False,
        show_labels=False,
        behavior_enabled=False,
    )
    proc = CVProcessor(settings)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    kp = np.array([[70.0, 70.0, 0.95]], dtype=np.float32)
    tracked = [_tracked_with_keypoints(kp)]

    out = proc.draw_overlays(frame, [], tracked)

    assert tuple(int(c) for c in out[70, 70]) == (0, 0, 0)


def test_draw_overlays_handles_ultralytics_instance_shape():
    """Real Ultralytics keypoints arrive as (1, K, 2) with a leading dim."""
    settings = CVSettings(
        keypoint_color=(0, 0, 255),
        keypoint_radius=4,
        show_labels=False,
        behavior_enabled=False,
    )
    proc = CVProcessor(settings)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Shape (1, 2, 2): one instance, two keypoints, xy only.
    kp = np.array([[[70.0, 70.0], [30.0, 30.0]]], dtype=np.float32)
    tracked = [_tracked_with_keypoints(kp)]

    out = proc.draw_overlays(frame, [], tracked)

    assert tuple(int(c) for c in out[70, 70]) == (0, 0, 255)
    assert tuple(int(c) for c in out[30, 30]) == (0, 0, 255)


def test_draw_overlays_handles_2d_keypoints_without_confidence():
    """Nx2 keypoints (no confidence column) still render."""
    settings = CVSettings(
        keypoint_color=(0, 0, 255),
        keypoint_radius=4,
        show_labels=False,
        behavior_enabled=False,
    )
    proc = CVProcessor(settings)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    kp = np.array([[70.0, 70.0]], dtype=np.float32)  # no confidence column
    tracked = [_tracked_with_keypoints(kp)]

    out = proc.draw_overlays(frame, [], tracked)

    assert tuple(int(c) for c in out[70, 70]) == (0, 0, 255)


# ---------------------------------------------------------------------------
# Keypoint extraction (Ultralytics result -> flat (K, C) array)
# ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeKeypoints:
    def __init__(self, data=None, xy=None):
        self.data = None if data is None else _FakeTensor(data)
        self.xy = None if xy is None else _FakeTensor(xy)


def test_keypoints_to_array_uses_data_with_confidence():
    """Prefers .data (x, y, conf) and squeezes the leading instance dim."""
    kp = _FakeKeypoints(
        data=[[[10.0, 20.0, 0.9], [30.0, 40.0, 0.1]]],  # (1, 2, 3)
        xy=[[[10.0, 20.0], [30.0, 40.0]]],
    )
    arr = cvp.keypoints_to_array(kp)
    assert arr.shape == (2, 3)
    assert arr[0].tolist() == [10.0, 20.0, 0.9]


def test_keypoints_to_array_falls_back_to_xy():
    """When .data is absent, uses .xy and returns (K, 2)."""
    kp = _FakeKeypoints(data=None, xy=[[[10.0, 20.0], [30.0, 40.0]]])
    arr = cvp.keypoints_to_array(kp)
    assert arr.shape == (2, 2)
    assert arr[1].tolist() == [30.0, 40.0]


# ---------------------------------------------------------------------------
# NCNN model loading
# ---------------------------------------------------------------------------


def test_resolve_model_path_param_file_returns_parent_dir(tmp_path):
    """Selecting the .param file resolves to its folder (what Ultralytics loads)."""
    ncnn_dir = tmp_path / "exp-6_ncnn_model"
    ncnn_dir.mkdir()
    param = ncnn_dir / "model.ncnn.param"
    param.write_text("stub")
    (ncnn_dir / "model.ncnn.bin").write_text("stub")

    resolved, is_ncnn = resolve_model_path(str(param))

    assert is_ncnn is True
    assert resolved == str(ncnn_dir)


def test_resolve_model_path_directory_with_param_is_ncnn(tmp_path):
    """A folder containing model.ncnn.param is recognized as NCNN."""
    ncnn_dir = tmp_path / "exp-6_ncnn_model"
    ncnn_dir.mkdir()
    (ncnn_dir / "model.ncnn.param").write_text("stub")

    resolved, is_ncnn = resolve_model_path(str(ncnn_dir))

    assert is_ncnn is True
    assert resolved == str(ncnn_dir)


def test_resolve_model_path_pt_is_not_ncnn(tmp_path):
    """A .pt weights file is left untouched and flagged as non-NCNN."""
    pt = tmp_path / "exp-6.pt"
    pt.write_text("stub")

    resolved, is_ncnn = resolve_model_path(str(pt))

    assert is_ncnn is False
    assert resolved == str(pt)


def test_load_yolo_ncnn_skips_device_move(tmp_path, monkeypatch):
    """NCNN backend is CPU-only: loading must not call model.to()."""
    ncnn_dir = tmp_path / "exp-6_ncnn_model"
    ncnn_dir.mkdir()
    (ncnn_dir / "model.ncnn.param").write_text("stub")
    (ncnn_dir / "model.ncnn.bin").write_text("stub")

    calls = {"to": 0, "loaded_path": None, "task": "unset"}

    class FakeYOLO:
        def __init__(self, path, task=None):
            calls["loaded_path"] = path
            calls["task"] = task

        def to(self, device):  # pragma: no cover - must not be called
            calls["to"] += 1
            return self

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)

    proc = CVProcessor(CVSettings(backend=DetectionBackend.YOLO_V8, model_path=str(ncnn_dir)))
    proc._load_yolo_model()

    assert calls["to"] == 0
    assert calls["loaded_path"] == str(ncnn_dir)
    assert proc._device == "cpu"
    # Backend must remain YOLO (not silently downgraded to bg-subtraction).
    assert proc._settings.backend == DetectionBackend.YOLO_V8


def test_read_ncnn_metadata_task(tmp_path):
    """The task field is read from an NCNN export's metadata.yaml."""
    ncnn_dir = tmp_path / "exp-6_ncnn_model"
    ncnn_dir.mkdir()
    (ncnn_dir / "metadata.yaml").write_text("task: pose\nimgsz: [640, 640]\n")

    assert cvp.read_ncnn_metadata_task(str(ncnn_dir)) == "pose"


def test_read_ncnn_metadata_task_missing_returns_none(tmp_path):
    """No metadata.yaml (or no task key) yields None so Ultralytics can guess."""
    ncnn_dir = tmp_path / "exp-6_ncnn_model"
    ncnn_dir.mkdir()

    assert cvp.read_ncnn_metadata_task(str(ncnn_dir)) is None


def test_load_yolo_ncnn_passes_task_from_metadata(tmp_path, monkeypatch):
    """NCNN pose models must load with task='pose' or keypoints are misread.

    Regression test for the KeyError seen when Ultralytics guesses task=detect
    for an NCNN pose model and interprets keypoint channels as class scores.
    """
    ncnn_dir = tmp_path / "exp-6_ncnn_model"
    ncnn_dir.mkdir()
    (ncnn_dir / "model.ncnn.param").write_text("stub")
    (ncnn_dir / "model.ncnn.bin").write_text("stub")
    (ncnn_dir / "metadata.yaml").write_text("task: pose\n")

    captured = {"task": "unset"}

    class FakeYOLO:
        def __init__(self, path, task=None):
            captured["task"] = task

        def to(self, device):  # pragma: no cover
            return self

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)

    proc = CVProcessor(CVSettings(backend=DetectionBackend.YOLO_V8, model_path=str(ncnn_dir)))
    proc._load_yolo_model()

    assert captured["task"] == "pose"


def test_load_yolo_pt_still_moves_to_device(tmp_path, monkeypatch):
    """The .pt path keeps its accelerator-move behavior (regression guard)."""
    pt = tmp_path / "exp-6.pt"
    pt.write_text("stub")

    calls = {"to": 0}

    class FakeYOLO:
        def __init__(self, path):
            pass

        def to(self, device):
            calls["to"] += 1
            return self

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(cvp, "_resolve_device_for_yolo", lambda: "cpu")

    proc = CVProcessor(CVSettings(backend=DetectionBackend.YOLO_V8, model_path=str(pt)))
    proc._load_yolo_model()

    assert calls["to"] == 1
