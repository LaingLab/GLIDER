"""Fixtures shared by the whole ``glider-dlc`` suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from glider_dlc import POSE_CONVERTERS


@pytest.fixture(autouse=True)
def no_inherited_env(monkeypatch):
    """Unset ``GLIDER_DLC_ENV`` for every test.

    It changes where the environment is looked for and whether a missing one is
    ours to build, so a developer who has it set for their own DeepLabCut
    install would otherwise get different results from CI -- in the direction
    of passing locally and failing there.
    """
    monkeypatch.delenv("GLIDER_DLC_ENV", raising=False)


@pytest.fixture
def registered_plugin(monkeypatch):
    """Register this plugin's converter, and unregister afterwards.

    Ask for it when the test is about what core does with the converter rather
    than about the converter itself. ``monkeypatch.setitem`` restores the
    previous state either way, so nothing leaks into a later test.
    """
    from glider.plugins import plugin_manager as pm
    from glider.vision.pose import converters as core

    for name, cls in POSE_CONVERTERS.items():
        monkeypatch.setitem(core.POSE_CONVERTERS, name, cls)
        monkeypatch.setitem(pm._PLUGIN_COMPONENTS, ("pose", name), "glider-dlc")


#: A minimal but *real* DeepLabCut 3.x config: these are the keys DLC 3.0.1's
#: own templates use, and the ones the sidecar is read out of.
_DLC_CONFIG = {
    "metadata": {"bodyparts": ["snout", "leftear", "rightear", "tailbase"]},
    "model": {
        "backbone": {"type": "ResNet", "model_name": "resnet50_gn", "output_stride": 16},
        "backbone_output_channels": 2048,
        "heads": {
            "bodypart": {
                "type": "HeatmapHead",
                "predictor": {
                    "type": "HeatmapPredictor",
                    "apply_sigmoid": False,
                    "location_refinement": True,
                    "locref_std": 7.2801,
                },
                "heatmap_config": {"strides": [2]},
            }
        },
    },
    "data": {"inference": {"normalize_images": True}},
}


@pytest.fixture
def dlc_config():
    """A fresh copy of the config, so a test that edits one cannot reach another."""

    def _make(**overrides) -> dict:
        cfg = json.loads(json.dumps(_DLC_CONFIG))
        cfg.update(overrides)
        return cfg

    return _make


#: Distinguishes "give me the usual config" from an explicit ``config=None``,
#: which is how a test asks for a folder with a snapshot and no config at all.
_DEFAULT = object()


@pytest.fixture
def dlc_dir(tmp_path, dlc_config):
    """A folder shaped like the one DeepLabCut writes."""

    def _make(*, config=_DEFAULT, snapshots=("snapshot-200.pt",), sub="train") -> Path:
        if config is _DEFAULT:
            config = dlc_config()
        root = tmp_path / "project" / sub
        root.mkdir(parents=True)
        if config is not None:
            (root / "pytorch_config.yaml").write_text(yaml.safe_dump(config))
        for name in snapshots:
            (root / name).write_bytes(b"not really a snapshot")
        return root

    return _make


@pytest.fixture
def converted(dlc_dir):
    """A DLC folder with a conversion already in it, and its stamp."""
    from glider_dlc import convert as convert_module

    folder = dlc_dir()
    (folder / "model.onnx").write_bytes(b"onnx")
    (folder / convert_module.SIDECAR_NAME).write_text("{}")
    snapshot = convert_module.find_dlc_snapshot(folder)
    config = convert_module.find_dlc_config(folder)
    (folder / convert_module.STAMP_NAME).write_text(
        json.dumps(convert_module._stamp_for(snapshot, config))
    )
    return folder
