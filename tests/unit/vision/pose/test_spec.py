"""Identification of pose models on disk.

Every case here is a directory laid out on tmp_path — no real models, no torch,
no onnxruntime. Identification is pure path/JSON/YAML work by design, and these
tests are what hold it to that.
"""

import dataclasses
import json

import pytest

from glider.vision.pose.spec import (
    PoseModelError,
    PoseModelMeta,
    identify_pose_model,
    read_pose_model_meta,
)


def _write_sidecar(root, **overrides):
    payload = {
        "schema_version": 1,
        "kind": "dlc",
        "onnx": "model.onnx",
        "keypoint_names": ["snout", "left_ear", "right_ear"],
        "output_stride": 8.0,
        "locref_stdev": 7.2831,
        "input_layout": "NCHW",
        "color_mode": "rgb",
        "scale": 1.0,
        "divide_by_255": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "pad_to_stride": 32,
        "apply_sigmoid": False,
    }
    payload.update(overrides)
    (root / "glider_pose.json").write_text(json.dumps(payload))
    (root / "model.onnx").write_bytes(b"not a real model")
    return root


def test_sidecar_folder_is_identified_as_dlc(tmp_path):
    spec = identify_pose_model(_write_sidecar(tmp_path))
    assert spec.kind == "dlc"
    assert spec.keypoint_names == ["snout", "left_ear", "right_ear"]
    assert spec.locref_stdev == pytest.approx(7.2831)
    assert spec.model_path == tmp_path / "model.onnx"
    assert spec.root == tmp_path


def test_pt_file_is_yolo_with_no_names(tmp_path):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"stub")
    spec = identify_pose_model(pt)
    assert spec.kind == "yolo"
    assert spec.keypoint_names == []
    assert spec.model_path == pt


def test_sidecar_naming_a_missing_onnx_is_an_error(tmp_path):
    _write_sidecar(tmp_path)
    (tmp_path / "model.onnx").unlink()
    with pytest.raises(PoseModelError, match="model.onnx"):
        identify_pose_model(tmp_path)


def test_folder_with_onnx_but_no_config_names_the_helper(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"stub")
    with pytest.raises(PoseModelError, match="export_pose_onnx"):
        identify_pose_model(tmp_path)


def test_bare_onnx_without_a_sidecar_names_the_helper(tmp_path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"stub")
    with pytest.raises(PoseModelError, match="export_pose_onnx"):
        identify_pose_model(onnx)


def test_bare_onnx_beside_a_sidecar_resolves(tmp_path):
    _write_sidecar(tmp_path)
    spec = identify_pose_model(tmp_path / "model.onnx")
    assert spec.kind == "dlc"
    assert spec.keypoint_names == ["snout", "left_ear", "right_ear"]


def test_unknown_path_is_an_error(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("hello")
    with pytest.raises(PoseModelError):
        identify_pose_model(junk)


def test_missing_path_is_an_error(tmp_path):
    with pytest.raises(PoseModelError, match="does not exist"):
        identify_pose_model(tmp_path / "nope")


def test_a_missing_pt_still_classifies_as_yolo(tmp_path):
    """Suffix alone identifies a checkpoint; existence is the loader's problem.

    infer_video() has always let ultralytics report a missing model, and its
    error names the path far better than a generic "does not exist" would.
    """
    spec = identify_pose_model(tmp_path / "not_written_yet.pt")
    assert spec.kind == "yolo"


def test_duplicate_keypoint_names_are_rejected(tmp_path):
    _write_sidecar(tmp_path, keypoint_names=["snout", "snout"])
    with pytest.raises(PoseModelError, match="unique"):
        identify_pose_model(tmp_path)


def test_future_schema_version_is_refused(tmp_path):
    _write_sidecar(tmp_path, schema_version=99)
    with pytest.raises(PoseModelError, match="schema_version"):
        identify_pose_model(tmp_path)


def test_spec_is_frozen(tmp_path):
    spec = identify_pose_model(_write_sidecar(tmp_path))
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.kind = "sleap"


# --- DeepLabCut native-config fallback -------------------------------------


def _write_dlc(root, text):
    (root / "pose_cfg.yaml").write_text(text)
    (root / "snapshot.onnx").write_bytes(b"stub")
    return root


def test_dlc_pose_cfg_supplies_names_and_stride(tmp_path):
    _write_dlc(
        tmp_path,
        "all_joints_names:\n"
        "- snout\n"
        "- left_ear\n"
        "- right_ear\n"
        "stride: 8.0\n"
        "locref_stdev: 7.2831\n"
        "global_scale: 0.8\n",
    )
    spec = identify_pose_model(tmp_path)
    assert spec.kind == "dlc"
    assert spec.keypoint_names == ["snout", "left_ear", "right_ear"]
    assert spec.output_stride == pytest.approx(8.0)
    assert spec.locref_stdev == pytest.approx(7.2831)
    assert spec.scale == pytest.approx(0.8)
    assert spec.model_path == tmp_path / "snapshot.onnx"


def test_dlc_config_without_onnx_names_the_helper(tmp_path):
    _write_dlc(tmp_path, "all_joints_names:\n- snout\nstride: 8.0\n")
    (tmp_path / "snapshot.onnx").unlink()
    with pytest.raises(PoseModelError, match="export_pose_onnx"):
        identify_pose_model(tmp_path)


def test_dlc_config_with_two_onnx_files_refuses_to_guess(tmp_path):
    _write_dlc(tmp_path, "all_joints_names:\n- snout\nstride: 8.0\n")
    (tmp_path / "other.onnx").write_bytes(b"stub")
    with pytest.raises(PoseModelError, match="cannot tell which"):
        identify_pose_model(tmp_path)


def test_dlc_config_without_joint_names_is_an_error(tmp_path):
    _write_dlc(tmp_path, "stride: 8.0\n")
    with pytest.raises(PoseModelError, match="all_joints_names"):
        identify_pose_model(tmp_path)


def test_sidecar_wins_over_native_config(tmp_path):
    _write_dlc(tmp_path, "all_joints_names:\n- WRONG\nstride: 99.0\n")
    _write_sidecar(tmp_path)
    spec = identify_pose_model(tmp_path)
    assert spec.keypoint_names == ["snout", "left_ear", "right_ear"]
    assert spec.output_stride == pytest.approx(8.0)


# --- SLEAP native-config fallback ------------------------------------------


def _write_sleap(root, config):
    (root / "training_config.json").write_text(json.dumps(config))
    (root / "model.onnx").write_bytes(b"stub")
    return root


def test_sleap_training_config_supplies_names_and_stride(tmp_path):
    _write_sleap(
        tmp_path,
        {
            "model": {
                "heads": {
                    "single_instance": {
                        "part_names": ["snout", "tailbase"],
                        "output_stride": 4,
                    }
                },
                "backbone": {"unet": {"max_stride": 16}},
            },
            "data": {"preprocessing": {"input_scaling": 0.5, "ensure_grayscale": True}},
        },
    )
    spec = identify_pose_model(tmp_path)
    assert spec.kind == "sleap"
    assert spec.keypoint_names == ["snout", "tailbase"]
    assert spec.output_stride == pytest.approx(4.0)
    assert spec.scale == pytest.approx(0.5)
    assert spec.pad_to_stride == 16
    assert spec.color_mode == "gray"
    # No location-refinement head: this is what selects the confmap decoder.
    assert spec.locref_stdev is None


def test_sleap_defaults_to_rgb_when_not_grayscale(tmp_path):
    _write_sleap(
        tmp_path,
        {"model": {"heads": {"single_instance": {"part_names": ["a"], "output_stride": 2}}}},
    )
    spec = identify_pose_model(tmp_path)
    assert spec.color_mode == "rgb"
    assert spec.scale == pytest.approx(1.0)
    assert spec.pad_to_stride == 1


def test_sleap_multi_instance_head_is_rejected_by_name(tmp_path):
    _write_sleap(
        tmp_path,
        {"model": {"heads": {"centered_instance": {"part_names": ["a"]}}}},
    )
    with pytest.raises(PoseModelError, match="single_instance"):
        identify_pose_model(tmp_path)


def test_sleap_names_the_head_the_model_actually_has(tmp_path):
    """A real SLEAP config carries all four head keys, three of them null.

    The message used to list the dict's keys, so a centroid model was rejected
    with "has no single_instance head (found: centered_instance, centroid,
    multi_instance, single_instance)" -- naming single_instance as present in
    the same breath as calling it absent. It reads as a GLIDER bug rather than
    as the wrong model, which is the one thing an error like this must not do.
    """
    _write_sleap(
        tmp_path,
        {
            "model": {
                "heads": {
                    "single_instance": None,
                    "centroid": {"anchor_part": None, "sigma": 2.5},
                    "centered_instance": None,
                    "multi_instance": None,
                }
            }
        },
    )
    with pytest.raises(PoseModelError) as excinfo:
        identify_pose_model(tmp_path)

    message = str(excinfo.value)
    assert "found: centroid" in message
    # The whole defect: the head it says is missing must not also be listed
    # among the heads it says it found.
    assert "found: centroid, single_instance" not in message
    assert message.count("single_instance") == 1


def test_sleap_config_with_no_configured_head_says_none(tmp_path):
    """Every head null is a real shape -- an untrained or hand-edited config."""
    _write_sleap(
        tmp_path,
        {"model": {"heads": {"single_instance": None, "centroid": None}}},
    )
    with pytest.raises(PoseModelError, match="found: none"):
        identify_pose_model(tmp_path)


def test_sleap_config_without_part_names_is_an_error(tmp_path):
    _write_sleap(tmp_path, {"model": {"heads": {"single_instance": {"output_stride": 2}}}})
    with pytest.raises(PoseModelError, match="part_names"):
        identify_pose_model(tmp_path)


# --- read_pose_model_meta ---------------------------------------------------
#
# This fills the seam gui/pose_batch/window.py has carried since it was written:
# it imported glider.vision.pose.model_meta inside a try/except, but that module
# was never committed on any ref, so the except branch was the only one that
# ever ran and _MetaWorker always emitted None.


def test_meta_from_dlc_folder_carries_names(tmp_path):
    meta = read_pose_model_meta(_write_sidecar(tmp_path))
    assert isinstance(meta, PoseModelMeta)
    assert meta.keypoint_names == ["snout", "left_ear", "right_ear"]
    assert meta.n_keypoints == 3
    assert meta.kind == "dlc"


def test_meta_for_unreadable_path_is_none(tmp_path):
    assert read_pose_model_meta(tmp_path / "nope") is None


def test_meta_for_pt_reports_count_without_names(tmp_path, monkeypatch):
    # Ultralytics checkpoints carry class names, not keypoint names, so a .pt
    # can only ever tell us K.
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"stub")
    monkeypatch.setattr("glider.vision.pose.spec._read_yolo_kpt_count", lambda p: 7)
    meta = read_pose_model_meta(pt)
    assert meta.kind == "yolo"
    assert meta.keypoint_names is None
    assert meta.n_keypoints == 7


def test_meta_never_raises_when_the_checkpoint_is_unreadable(tmp_path, monkeypatch):
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"stub")

    def _boom(path):
        raise RuntimeError("torch exploded")

    monkeypatch.setattr("glider.vision.pose.spec._read_yolo_kpt_count", _boom)
    assert read_pose_model_meta(pt) is None


def test_a_slp_file_is_named_as_labels_not_a_model(tmp_path):
    """The natural mistake: .slp is what a SLEAP user works with all day, so
    it is what they reach for. A generic 'not a pose model' would send someone
    hunting for a corrupt file rather than telling them they picked the wrong
    one."""
    from glider.vision.pose.spec import PoseModelError, identify_pose_model

    labels = tmp_path / "my_project.slp"
    labels.write_bytes(b"not really a labels file")

    with pytest.raises(PoseModelError) as excinfo:
        identify_pose_model(labels)

    message = str(excinfo.value)
    assert "labels file, not a trained model" in message
    assert "training_config.json" in message, "it must say what to pick instead"
