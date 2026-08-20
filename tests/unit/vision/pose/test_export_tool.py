"""The standalone export helper's sidecar builders.

``tools/export_pose_onnx.py`` runs in the user's own DeepLabCut or SLEAP
environment and imports nothing from glider, so it is loaded here by path. The
conversion itself needs those frameworks and cannot be tested here; what *can*
be tested is the part that matters most — that it refuses to write a sidecar it
cannot fully populate, and that what it does write round-trips through
``identify_pose_model``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[4] / "tools" / "export_pose_onnx.py"


@pytest.fixture(scope="module")
def tool():
    if not _TOOL.is_file():
        pytest.skip(f"export helper not found at {_TOOL}")
    spec = importlib.util.spec_from_file_location("_export_pose_onnx", _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_export_pose_onnx"] = module
    spec.loader.exec_module(module)
    return module


def test_the_helper_imports_nothing_from_glider(tool):
    """It must be copyable into a DLC/SLEAP env on its own."""
    source = _TOOL.read_text(encoding="utf-8")
    assert "import glider" not in source
    assert "from glider" not in source


# --- DeepLabCut -------------------------------------------------------------


def _dlc_cfg(**over):
    cfg = {
        "all_joints_names": ["snout", "left_ear", "right_ear"],
        "stride": 8.0,
        "locref_stdev": 7.2831,
        "global_scale": 0.8,
    }
    cfg.update(over)
    return cfg


def test_dlc_sidecar_carries_every_field(tool, tmp_path):
    out = tool.build_dlc_sidecar(_dlc_cfg(), tmp_path / "pose_cfg.yaml", "model.onnx", "dlc_x")
    assert out["kind"] == "dlc"
    assert out["keypoint_names"] == ["snout", "left_ear", "right_ear"]
    assert out["output_stride"] == pytest.approx(8.0)
    assert out["locref_stdev"] == pytest.approx(7.2831)
    assert out["scale"] == pytest.approx(0.8)
    assert out["mean"] == tool.IMAGENET_MEAN


@pytest.mark.parametrize("missing", ["stride", "locref_stdev", "all_joints_names"])
def test_dlc_refuses_to_guess_a_missing_field(tool, tmp_path, missing):
    """A defaulted stride offsets every keypoint without ever failing."""
    cfg = _dlc_cfg()
    del cfg[missing]
    with pytest.raises(tool.ExportError, match=missing):
        tool.build_dlc_sidecar(cfg, tmp_path / "pose_cfg.yaml", "model.onnx", "dlc_x")


def test_dlc_rejects_duplicate_names(tool, tmp_path):
    cfg = _dlc_cfg(all_joints_names=["snout", "snout"])
    with pytest.raises(tool.ExportError, match="not unique"):
        tool.build_dlc_sidecar(cfg, tmp_path / "pose_cfg.yaml", "model.onnx", "dlc_x")


# --- SLEAP ------------------------------------------------------------------


def _sleap_cfg(**head_over):
    head = {"part_names": ["snout", "tailbase"], "output_stride": 4}
    head.update(head_over)
    return {
        "model": {"heads": {"single_instance": head}, "backbone": {"unet": {"max_stride": 16}}},
        "data": {"preprocessing": {"input_scaling": 0.5, "ensure_grayscale": True}},
    }


def test_sleap_sidecar_carries_every_field(tool, tmp_path):
    out = tool.build_sleap_sidecar(
        _sleap_cfg(), tmp_path / "training_config.json", "model.onnx", "sleap_x"
    )
    assert out["kind"] == "sleap"
    assert out["keypoint_names"] == ["snout", "tailbase"]
    assert out["output_stride"] == pytest.approx(4.0)
    assert out["scale"] == pytest.approx(0.5)
    assert out["pad_to_stride"] == 16
    assert out["color_mode"] == "gray"
    assert out["input_layout"] == "NHWC"
    assert out["locref_stdev"] is None


def test_sleap_rejects_a_multi_animal_head(tool, tmp_path):
    cfg = {"model": {"heads": {"centered_instance": {"part_names": ["a"]}}}}
    with pytest.raises(tool.ExportError, match="single_instance"):
        tool.build_sleap_sidecar(cfg, tmp_path / "training_config.json", "model.onnx", "s")


def test_sleap_refuses_a_missing_output_stride(tool, tmp_path):
    cfg = _sleap_cfg()
    del cfg["model"]["heads"]["single_instance"]["output_stride"]
    with pytest.raises(tool.ExportError, match="output_stride"):
        tool.build_sleap_sidecar(cfg, tmp_path / "training_config.json", "model.onnx", "s")


# --- round trip -------------------------------------------------------------


@pytest.mark.parametrize("kind", ["dlc", "sleap"])
def test_written_sidecar_is_readable_by_glider(tool, tmp_path, kind):
    """The whole point: what this writes is what identify_pose_model reads."""
    from glider.vision.pose.spec import identify_pose_model

    if kind == "dlc":
        payload = tool.build_dlc_sidecar(
            _dlc_cfg(), tmp_path / "pose_cfg.yaml", "model.onnx", "dlc_x"
        )
        expected = ["snout", "left_ear", "right_ear"]
    else:
        payload = tool.build_sleap_sidecar(
            _sleap_cfg(), tmp_path / "training_config.json", "model.onnx", "sleap_x"
        )
        expected = ["snout", "tailbase"]

    (tmp_path / "glider_pose.json").write_text(json.dumps(payload))
    (tmp_path / "model.onnx").write_bytes(b"stub")

    spec = identify_pose_model(tmp_path)
    assert spec.kind == kind
    assert spec.keypoint_names == expected


def test_cli_help_works(tool, capsys):
    with pytest.raises(SystemExit) as exc:
        tool.main(["--help"])
    assert exc.value.code == 0
    assert "DeepLabCut" in capsys.readouterr().out
