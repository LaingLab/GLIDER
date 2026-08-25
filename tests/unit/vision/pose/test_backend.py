"""Pose backends, exercised without ultralytics, torch, or onnxruntime.

Both real backends take their heavy object by injection — a YOLO net or an
onnxruntime session — so the doubles below cover the whole class. That is what
keeps these tests runnable in CI on a machine with neither installed.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from glider.vision.pose import backend as backend_mod
from glider.vision.pose.backend import (
    OnnxPoseBackend,
    UltralyticsBackend,
    load_pose_backend,
    preprocess_frame,
)
from glider.vision.pose.spec import PoseModelError, PoseModelSpec, identify_pose_model


class _FakeTensor:
    """Minimal stand-in for a torch tensor as extract_keypoints uses it."""

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape

    def __getitem__(self, i):
        return _FakeTensor(self._arr[i])

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = _FakeTensor(np.asarray([xy], dtype=float))
        self.conf = _FakeTensor(np.asarray([conf], dtype=float))


class _FakeResult:
    def __init__(self, xy=None, conf=None):
        self.keypoints = None if xy is None else _FakeKeypoints(xy, conf)


class _FakeYolo:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [self._result]


def test_predict_returns_xy_and_conf_in_pixels():
    yolo = _FakeYolo(_FakeResult([[10.0, 20.0], [30.0, 40.0]], [0.9, 0.8]))
    backend = UltralyticsBackend(yolo, ["a", "b"], conf_threshold=0.25)
    xy, conf = backend.predict(np.zeros((32, 32, 3), np.uint8))
    assert xy == pytest.approx(np.array([[10.0, 20.0], [30.0, 40.0]]))
    assert conf == pytest.approx([0.9, 0.8])


def test_low_confidence_keypoints_become_nan():
    yolo = _FakeYolo(_FakeResult([[10.0, 20.0], [30.0, 40.0]], [0.9, 0.1]))
    backend = UltralyticsBackend(yolo, ["a", "b"], conf_threshold=0.5)
    xy, conf = backend.predict(np.zeros((32, 32, 3), np.uint8))
    assert np.isnan(xy[1]).all()
    # Confidence is preserved even for masked keypoints: the overlay uses it
    # for per-dot fade.
    assert conf[1] == pytest.approx(0.1)


def test_no_detection_returns_all_nan():
    backend = UltralyticsBackend(_FakeYolo(_FakeResult()), ["a", "b"])
    xy, conf = backend.predict(np.zeros((32, 32, 3), np.uint8))
    assert np.isnan(xy).all()
    assert conf == pytest.approx([0.0, 0.0])


def test_confidence_threshold_is_forwarded_to_yolo():
    yolo = _FakeYolo(_FakeResult([[1.0, 2.0]], [0.9]))
    UltralyticsBackend(yolo, ["a"], conf_threshold=0.4).predict(np.zeros((8, 8, 3), np.uint8))
    assert yolo.calls[0]["conf"] == pytest.approx(0.4)
    assert yolo.calls[0]["verbose"] is False


def test_output_is_sized_by_names_not_by_the_model_head():
    # Head emits 2 keypoints, caller named 3: arrays stay (3, ...) so downstream
    # feature maths never sees a short row.
    yolo = _FakeYolo(_FakeResult([[1.0, 2.0], [3.0, 4.0]], [0.9, 0.9]))
    backend = UltralyticsBackend(yolo, ["a", "b", "c"])
    xy, conf = backend.predict(np.zeros((8, 8, 3), np.uint8))
    assert xy.shape == (3, 2)
    assert conf.shape == (3,)
    assert np.isnan(xy[2]).all()


def test_keypoint_names_are_copied_not_aliased():
    names = ["a", "b"]
    backend = UltralyticsBackend(_FakeYolo(_FakeResult()), names)
    names.append("c")
    assert backend.keypoint_names == ["a", "b"]


# --- preprocessing ----------------------------------------------------------


def _spec(**kw):
    base = {
        "kind": "dlc",
        "model_path": Path("m.onnx"),
        "root": Path("."),
        "keypoint_names": ["a"],
    }
    base.update(kw)
    return PoseModelSpec(**base)


def test_nchw_rgb_normalised_output_shape_and_range():
    frame = np.full((64, 48, 3), 255, np.uint8)
    tensor, scale = preprocess_frame(frame, _spec(input_layout="NCHW"))
    assert tensor.shape == (1, 3, 64, 48)
    assert tensor.dtype == np.float32
    assert scale == pytest.approx(1.0)
    assert tensor.max() <= 1.0001


def test_nhwc_layout_is_channels_last():
    tensor, _ = preprocess_frame(np.zeros((32, 32, 3), np.uint8), _spec(input_layout="NHWC"))
    assert tensor.shape == (1, 32, 32, 3)


def test_grayscale_collapses_to_one_channel():
    tensor, _ = preprocess_frame(np.zeros((16, 16, 3), np.uint8), _spec(color_mode="gray"))
    assert tensor.shape == (1, 1, 16, 16)


def test_scale_resizes_the_frame():
    tensor, scale = preprocess_frame(np.zeros((100, 200, 3), np.uint8), _spec(scale=0.5))
    assert tensor.shape == (1, 3, 50, 100)
    assert scale == pytest.approx(0.5)


def test_pad_to_stride_pads_bottom_right_only():
    # 30x30 padded up to a multiple of 16 -> 32x32. Padding goes bottom/right so
    # it never shifts the coordinate origin and needs no un-mapping.
    tensor, _ = preprocess_frame(np.zeros((30, 30, 3), np.uint8), _spec(pad_to_stride=16))
    assert tensor.shape == (1, 3, 32, 32)


def test_pad_is_a_noop_when_already_aligned():
    tensor, _ = preprocess_frame(np.zeros((32, 32, 3), np.uint8), _spec(pad_to_stride=16))
    assert tensor.shape == (1, 3, 32, 32)


def test_mean_std_normalisation_is_applied():
    frame = np.full((8, 8, 3), 255, np.uint8)
    spec = _spec(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    tensor, _ = preprocess_frame(frame, spec)
    assert tensor == pytest.approx(np.ones_like(tensor))


def test_divide_by_255_can_be_disabled():
    frame = np.full((8, 8, 3), 255, np.uint8)
    tensor, _ = preprocess_frame(frame, _spec(divide_by_255=False))
    assert tensor.max() == pytest.approx(255.0)


def test_bgr_is_converted_to_rgb():
    # Pure blue in BGR must land in the red-most channel after conversion.
    frame = np.zeros((4, 4, 3), np.uint8)
    frame[:, :, 0] = 255  # B
    tensor, _ = preprocess_frame(frame, _spec(divide_by_255=False))
    assert tensor[0, 2].max() == pytest.approx(255.0)  # channel 2 == B in RGB
    assert tensor[0, 0].max() == pytest.approx(0.0)


# --- OnnxPoseBackend --------------------------------------------------------


class _FakeInput:
    name = "images"


class _FakeSession:
    """Stands in for onnxruntime.InferenceSession."""

    def __init__(self, outputs):
        self._outputs = outputs
        self.feeds = []

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, feed):
        self.feeds.append(feed)
        return self._outputs


def _dlc_outputs(k=2, h=8, w=8, row=3, col=5):
    heat = np.zeros((1, k, h, w), np.float32)
    heat[0, :, row, col] = 1.0
    locref = np.zeros((1, 2 * k, h, w), np.float32)
    return [heat, locref]


def test_dlc_session_output_decodes_to_source_pixels():
    spec = _spec(
        kind="dlc", keypoint_names=["a", "b"], output_stride=8.0, locref_stdev=1.0, scale=1.0
    )
    backend = OnnxPoseBackend(_FakeSession(_dlc_outputs()), spec)
    xy, conf = backend.predict(np.zeros((64, 64, 3), np.uint8))
    assert xy[0] == pytest.approx([5 * 8 + 4, 3 * 8 + 4])
    assert conf == pytest.approx([1.0, 1.0])


def test_scale_is_divided_back_out():
    spec = _spec(
        kind="dlc", keypoint_names=["a", "b"], output_stride=8.0, locref_stdev=1.0, scale=0.5
    )
    backend = OnnxPoseBackend(_FakeSession(_dlc_outputs()), spec)
    xy, _ = backend.predict(np.zeros((64, 64, 3), np.uint8))
    # Model-space 44 px measured on a half-size frame is 88 px in the source.
    assert xy[0] == pytest.approx([88.0, 56.0])


def test_sleap_single_output_uses_the_confmap_decoder():
    cm = np.zeros((1, 1, 16, 16), np.float32)
    cm[0, 0, 8, 6] = 1.0
    spec = _spec(kind="sleap", keypoint_names=["a"], output_stride=2.0, locref_stdev=None)
    backend = OnnxPoseBackend(_FakeSession([cm]), spec)
    xy, _ = backend.predict(np.zeros((32, 32, 3), np.uint8))
    assert xy[0] == pytest.approx([12.0, 16.0])


def test_nhwc_model_output_is_transposed_before_decode():
    heat = np.zeros((1, 8, 8, 2), np.float32)  # NHWC
    heat[0, 3, 5, :] = 1.0
    spec = _spec(
        kind="sleap",
        keypoint_names=["a", "b"],
        output_stride=1.0,
        locref_stdev=None,
        input_layout="NHWC",
    )
    backend = OnnxPoseBackend(_FakeSession([heat]), spec)
    xy, _ = backend.predict(np.zeros((8, 8, 3), np.uint8))
    assert xy[0] == pytest.approx([5.0, 3.0])


def test_the_input_name_from_the_session_is_used():
    spec = _spec(kind="sleap", keypoint_names=["a"], output_stride=1.0, locref_stdev=None)
    session = _FakeSession([np.zeros((1, 1, 4, 4), np.float32)])
    OnnxPoseBackend(session, spec).predict(np.zeros((4, 4, 3), np.uint8))
    assert list(session.feeds[0]) == ["images"]


def test_dlc_without_a_locref_output_is_an_explicit_error():
    spec = _spec(kind="dlc", keypoint_names=["a", "b"], locref_stdev=7.28)
    heat = np.zeros((1, 2, 8, 8), np.float32)
    backend = OnnxPoseBackend(_FakeSession([heat]), spec)
    with pytest.raises(ValueError, match="locref"):
        backend.predict(np.zeros((64, 64, 3), np.uint8))


def test_keypoint_count_mismatch_is_an_explicit_error():
    spec = _spec(kind="sleap", keypoint_names=["a", "b", "c"], locref_stdev=None)
    cm = np.zeros((1, 2, 8, 8), np.float32)  # 2 channels, 3 names
    backend = OnnxPoseBackend(_FakeSession([cm]), spec)
    with pytest.raises(ValueError, match="3"):
        backend.predict(np.zeros((32, 32, 3), np.uint8))


# --- load_pose_backend dispatch ---------------------------------------------


def _sidecar_folder(root):
    (root / "glider_pose.json").write_text(
        json.dumps(
            {
                "kind": "dlc",
                "onnx": "model.onnx",
                "keypoint_names": ["snout", "left_ear", "right_ear"],
                "output_stride": 8.0,
                "locref_stdev": 7.2831,
            }
        )
    )
    (root / "model.onnx").write_bytes(b"stub")
    return root


def test_pt_path_builds_an_ultralytics_backend(tmp_path, monkeypatch):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"stub")
    monkeypatch.setattr(backend_mod, "_load_yolo", lambda p: _FakeYolo(_FakeResult()))
    b = load_pose_backend(pt, keypoint_names=["a", "b"])
    assert isinstance(b, UltralyticsBackend)
    assert b.keypoint_names == ["a", "b"]


def test_sidecar_folder_builds_an_onnx_backend(tmp_path, monkeypatch):
    root = _sidecar_folder(tmp_path)
    monkeypatch.setattr(backend_mod, "_make_session", lambda spec: _FakeSession([]))
    b = load_pose_backend(root)
    assert isinstance(b, OnnxPoseBackend)
    # Names come from the model, not the caller.
    assert b.keypoint_names == ["snout", "left_ear", "right_ear"]


def test_model_names_override_caller_supplied_names(tmp_path, monkeypatch):
    # The model's training order is the only order that is correct, so anything
    # the operator typed loses.
    root = _sidecar_folder(tmp_path)
    monkeypatch.setattr(backend_mod, "_make_session", lambda spec: _FakeSession([]))
    b = load_pose_backend(root, keypoint_names=["wrong", "names", "here"])
    assert b.keypoint_names == ["snout", "left_ear", "right_ear"]


def test_yolo_without_names_is_an_error(tmp_path, monkeypatch):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"stub")
    monkeypatch.setattr(backend_mod, "_load_yolo", lambda p: _FakeYolo(_FakeResult()))
    with pytest.raises(PoseModelError, match="keypoint names"):
        load_pose_backend(pt)


def test_an_already_built_spec_is_accepted(tmp_path, monkeypatch):
    root = _sidecar_folder(tmp_path)
    monkeypatch.setattr(backend_mod, "_make_session", lambda spec: _FakeSession([]))
    spec = identify_pose_model(root)
    b = load_pose_backend(spec)
    assert isinstance(b, OnnxPoseBackend)
    assert b.spec is spec


def test_conf_threshold_reaches_the_ultralytics_backend(tmp_path, monkeypatch):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"stub")
    monkeypatch.setattr(backend_mod, "_load_yolo", lambda p: _FakeYolo(_FakeResult()))
    b = load_pose_backend(pt, keypoint_names=["a"], conf_threshold=0.7)
    assert b.conf_threshold == pytest.approx(0.7)


# --- fitting a frame to a fixed-input model -----------------------------------


def _spec_for_fit(tmp_path, **over):
    from glider.vision.pose.spec import PoseModelSpec

    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"stub")
    kw = {
        "kind": "sleap",
        "model_path": onnx,
        "root": tmp_path,
        "keypoint_names": ["a", "b"],
        "source_label": "t",
        "input_layout": "NHWC",
        "output_stride": 1.0,
        "scale": 1.0,
        "pad_to_stride": 1,
        "divide_by_255": True,
    }
    kw.update(over)
    return PoseModelSpec(**kw)


def test_a_frame_is_fitted_to_a_models_fixed_input(tmp_path):
    """The model's ONNX fixes its spatial dims; the frame is what must give."""
    from glider.vision.pose.backend import preprocess_frame

    spec = _spec_for_fit(tmp_path)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    tensor, scale = preprocess_frame(frame, spec, fit_to=(320, 560))

    assert tensor.shape == (1, 320, 560, 3)
    # 560/1920 fits before 320/1080 does, so width is the binding constraint.
    assert scale == pytest.approx(560 / 1920)


def test_fitting_preserves_aspect_and_pads_rather_than_stretching(tmp_path):
    """A stretched animal is a differently-shaped animal, and the model was not
    trained on one. Fit inside, pad the remainder."""
    from glider.vision.pose.backend import preprocess_frame

    spec = _spec_for_fit(tmp_path)
    # A bright frame on a black pad makes the letterbox visible.
    frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)

    tensor, scale = preprocess_frame(frame, spec, fit_to=(320, 560))

    used_h = int(round(1080 * scale))
    assert tensor[0, used_h - 1, 0, 0] > 0.0  # inside the image
    assert tensor[0, -1, 0, 0] == 0.0  # in the pad below it


def test_keypoints_map_back_to_source_pixels(tmp_path):
    """The property that matters: a point the model reports must land where it
    is in the original frame, not in the resized one."""
    from glider.vision.pose.backend import preprocess_frame

    spec = _spec_for_fit(tmp_path)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _tensor, scale = preprocess_frame(frame, spec, fit_to=(320, 560))

    # A detection at the centre of the fitted image is the centre of the source.
    network_xy = np.array([[1920 * scale / 2, 1080 * scale / 2]])
    source_xy = network_xy / scale
    assert source_xy[0] == pytest.approx([960.0, 540.0], abs=1.0)


def test_fitting_composes_with_the_specs_own_scaling(tmp_path):
    """SLEAP's input_scaling already halves the frame; fitting must account for
    what that left, not re-derive from the source size."""
    from glider.vision.pose.backend import preprocess_frame

    spec = _spec_for_fit(tmp_path, scale=0.5)
    frame = np.zeros((640, 1120, 3), dtype=np.uint8)

    tensor, scale = preprocess_frame(frame, spec, fit_to=(320, 560))

    # 0.5 alone lands exactly on the target, so no further fitting is needed.
    assert tensor.shape == (1, 320, 560, 3)
    assert scale == pytest.approx(0.5)


def test_no_fit_target_leaves_the_old_behaviour_alone(tmp_path):
    """A dynamic-input model must keep padding to stride, not be resized."""
    from glider.vision.pose.backend import preprocess_frame

    spec = _spec_for_fit(tmp_path, pad_to_stride=32)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    tensor, scale = preprocess_frame(frame, spec)

    assert tensor.shape == (1, 128, 128, 3)  # padded up to the stride
    assert scale == pytest.approx(1.0)
