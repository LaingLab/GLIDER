"""Turning a sleap-nn training config into the sidecar GLIDER reads.

The fixture is Steven's real `training_config.yaml`, trimmed to the sections
that matter. Hand-written fixtures drift from what the tool actually emits;
this one cannot.
"""

from pathlib import Path

import pytest
import yaml
from glider_sleap_nn.convert import ConversionError, sidecar_for_config

DATA = Path(__file__).parent / "data"


def _cfg():
    return yaml.safe_load((DATA / "training_config.yaml").read_text(encoding="utf-8"))


def test_keypoint_names_come_from_part_names():
    assert sidecar_for_config(_cfg())["keypoint_names"] == [
        "left_ear",
        "right_ear",
        "nose",
        "body_center",
        "left_hip",
        "right_hip",
        "tail_base",
    ]


def test_strides_layout_and_kind():
    s = sidecar_for_config(_cfg())
    assert s["output_stride"] == 2  # the confmaps head, not the backbone
    assert s["pad_to_stride"] == 16  # UNet max_stride
    assert s["input_layout"] == "NCHW"  # PyTorch, always
    assert s["kind"] == "sleap"
    assert s["onnx"] == "model.onnx"


def test_the_sleap_nn_version_is_recorded():
    """The only thing in the folder saying which sleap-nn wrote it."""
    assert sidecar_for_config(_cfg())["source_label"] == "sleap_nn_0.3.3"


def test_backbone_is_found_past_the_null_siblings():
    """backbone_config lists unet/convnext/swint and nulls all but one."""
    cfg = _cfg()
    assert [k for k, v in cfg["model_config"]["backbone_config"].items() if v] == ["unet"]
    assert sidecar_for_config(cfg)["pad_to_stride"] == 16


def test_colour_mode_follows_the_preprocessing():
    cfg = _cfg()
    assert sidecar_for_config(cfg)["color_mode"] == "rgb"
    cfg["data_config"]["preprocessing"]["ensure_rgb"] = False
    cfg["data_config"]["preprocessing"]["ensure_grayscale"] = True
    assert sidecar_for_config(cfg)["color_mode"] == "gray"


@pytest.mark.parametrize(
    "head",
    [
        "centroid",
        "centered_instance",
        "bottomup",
        "multi_class_bottomup",
        "multi_class_topdown",
        "bottomup_segmentation",
        "semantic_segmentation",
    ],
)
def test_any_other_populated_head_is_refused_by_name(head):
    """Not a deny-list: anything but single_instance is refused.

    sleap-nn 0.3.3 ships nine head types and will add more. A list of the ones
    to reject would quietly start converting whatever comes next.
    """
    cfg = _cfg()
    cfg["model_config"]["head_configs"]["single_instance"] = None
    cfg["model_config"]["head_configs"][head] = {"confmaps": {"part_names": ["a"]}}
    with pytest.raises(ConversionError, match=head):
        sidecar_for_config(cfg)


def test_a_multi_head_model_names_every_offending_head():
    cfg = _cfg()
    cfg["model_config"]["head_configs"]["centroid"] = {"x": 1}
    cfg["model_config"]["head_configs"]["bottomup"] = {"x": 1}
    with pytest.raises(ConversionError) as exc:
        sidecar_for_config(cfg)
    assert "bottomup" in str(exc.value) and "centroid" in str(exc.value)


def test_a_config_with_no_usable_head_is_refused():
    cfg = _cfg()
    cfg["model_config"]["head_configs"] = {"single_instance": None}
    with pytest.raises(ConversionError, match="part names"):
        sidecar_for_config(cfg)


def test_divide_by_255_is_not_decided_here():
    """Only the conversion can know: it is a property of the traced graph."""
    assert "divide_by_255" not in sidecar_for_config(_cfg())
