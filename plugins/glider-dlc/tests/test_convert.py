"""Finding, dating, and describing a DeepLabCut model.

None of this needs DeepLabCut, which is the point: it is what GLIDER asks about
a folder before anything is downloaded, and it runs in GLIDER's own
interpreter. The conversion itself is exercised by the slow test at the bottom,
which needs the provisioned environment and a real model.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml

from glider_dlc.convert import (
    SIDECAR_NAME,
    STAMP_NAME,
    ConversionError,
    build_sidecar,
    convert_dlc_to_onnx,
    find_dlc_config,
    find_dlc_snapshot,
    is_conversion_current,
    is_dlc_folder,
    needs_conversion,
)

# --- finding the model -------------------------------------------------------


def test_it_finds_a_config_beside_the_snapshot(dlc_dir):
    folder = dlc_dir()

    assert find_dlc_config(folder).name == "pytorch_config.yaml"
    assert find_dlc_snapshot(folder).name == "snapshot-200.pt"


def test_it_finds_a_model_nested_under_a_project(dlc_dir, tmp_path):
    """A researcher points at the project folder as often as at the train
    folder inside it, and refusing the first would be refusing the obvious
    thing to do."""
    dlc_dir()
    project = tmp_path / "project"

    assert find_dlc_config(project) is not None
    assert is_dlc_folder(project) is True


def test_the_best_snapshot_wins_over_a_later_one(dlc_dir):
    """DLC picks `best` on the validation metric. Taking a later but worse
    epoch because its number sorts higher is a silent downgrade."""
    folder = dlc_dir(snapshots=("snapshot-090.pt", "snapshot-best-050.pt", "snapshot-200.pt"))

    assert find_dlc_snapshot(folder).name == "snapshot-best-050.pt"


def test_epochs_are_compared_as_numbers(dlc_dir):
    """Lexically, snapshot-90 beats snapshot-100."""
    folder = dlc_dir(snapshots=("snapshot-90.pt", "snapshot-100.pt"))

    assert find_dlc_snapshot(folder).name == "snapshot-100.pt"


def test_a_config_without_weights_is_not_a_model(dlc_dir):
    assert is_dlc_folder(dlc_dir(snapshots=())) is False


def test_a_stray_checkpoint_is_not_a_model(dlc_dir):
    """Claiming other people's .pt files is how two plugins end up fighting
    over one folder."""
    assert is_dlc_folder(dlc_dir(config=None)) is False


def test_an_unrelated_folder_is_not_a_model(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"yolo")

    assert is_dlc_folder(tmp_path) is False


# --- staleness ---------------------------------------------------------------


def test_a_fresh_model_needs_converting(dlc_dir):
    assert needs_conversion(dlc_dir()) is True


def test_a_converted_model_does_not(converted):
    assert is_conversion_current(converted) is True
    assert needs_conversion(converted) is False


def test_a_retrained_snapshot_is_stale(converted):
    """The case this exists for. A stale ONNX runs perfectly well and answers
    with the network that was just replaced."""
    (converted / "snapshot-200.pt").write_bytes(b"retrained, and a different length")

    assert is_conversion_current(converted) is False


def test_a_same_length_retrain_is_still_stale(converted):
    """Size settles the common case cheaply; the digest is what makes it
    correct. A snapshot rewritten to the same length is the case where only the
    digest can tell."""
    (converted / "snapshot-200.pt").write_bytes(b"not really a snapshoT")

    assert is_conversion_current(converted) is False


def test_a_new_snapshot_beside_the_old_one_is_stale(converted):
    (converted / "snapshot-400.pt").write_bytes(b"newer")

    assert is_conversion_current(converted) is False


def test_a_missing_onnx_is_stale(converted):
    """An interrupted conversion writes the stamp last, but a deleted ONNX with
    a surviving stamp would otherwise read as current."""
    (converted / "model.onnx").unlink()

    assert is_conversion_current(converted) is False


def test_a_missing_sidecar_is_stale(converted):
    (converted / SIDECAR_NAME).unlink()

    assert is_conversion_current(converted) is False


def test_an_unreadable_stamp_is_stale(converted):
    (converted / STAMP_NAME).write_text("{ not json")

    assert is_conversion_current(converted) is False


# --- the sidecar -------------------------------------------------------------


def _sidecar(cfg, *, stride=8.0) -> dict:
    return build_sidecar(
        cfg,
        Path("pytorch_config.yaml"),
        onnx_name="model.onnx",
        label="dlc_test",
        stride=stride,
    )


def test_the_sidecar_describes_the_model(dlc_config):
    s = _sidecar(dlc_config())

    assert s["kind"] == "dlc"
    assert s["keypoint_names"] == ["snout", "leftear", "rightear", "tailbase"]
    assert s["output_stride"] == 8.0
    assert s["locref_stdev"] == pytest.approx(7.2801)
    assert s["input_layout"] == "NCHW"


def test_normalize_images_means_imagenet_statistics(dlc_config):
    """Read off DeepLabCut's own transform builder: a truthy normalize_images
    is albumentations' Normalize with ImageNet stats, which divides by 255
    first. Getting this wrong shifts every keypoint."""
    s = _sidecar(dlc_config())

    assert s["divide_by_255"] is True
    assert s["mean"] == pytest.approx([0.485, 0.456, 0.406])
    assert s["std"] == pytest.approx([0.229, 0.224, 0.225])


def test_without_normalize_images_the_network_sees_raw_bytes(dlc_config):
    """No Normalize means no rescaling at all -- ToTensorV2 alone does not
    divide by 255."""
    s = _sidecar(dlc_config(data={"inference": {"normalize_images": False}}))

    assert s["divide_by_255"] is False
    assert s["mean"] is None and s["std"] is None


def test_hrnet_padding_is_carried_into_the_sidecar(dlc_config):
    """HRNet cannot take a size it can't halve cleanly. Feeding it one does not
    raise -- it returns a heatmap a pixel short, which is a constant shift."""
    cfg = dlc_config(
        data={
            "inference": {
                "normalize_images": True,
                "auto_padding": {"pad_height_divisor": 32, "pad_width_divisor": 32},
            }
        }
    )

    assert _sidecar(cfg)["pad_to_stride"] == 32


def test_a_model_without_locref_selects_peak_refinement(dlc_config):
    cfg = dlc_config()
    cfg["model"]["heads"]["bodypart"]["predictor"]["location_refinement"] = False

    assert _sidecar(cfg)["locref_stdev"] is None


def test_a_missing_locref_std_is_refused_rather_than_guessed(dlc_config):
    """The discipline the whole module is built on: a defaulted locref rescales
    every sub-pixel offset and still produces a plausible skeleton."""
    cfg = dlc_config()
    del cfg["model"]["heads"]["bodypart"]["predictor"]["locref_std"]

    with pytest.raises(ConversionError, match="locref_std"):
        _sidecar(cfg)


def test_missing_bodyparts_are_refused(dlc_config):
    cfg = dlc_config(metadata={})

    with pytest.raises(ConversionError, match="bodyparts"):
        _sidecar(cfg)


def test_duplicate_bodyparts_are_refused(dlc_config):
    """Duplicates would make the keypoint-name check downstream pass while the
    behaviour model reads two different body parts as one."""
    cfg = dlc_config(metadata={"bodyparts": ["snout", "snout"]})

    with pytest.raises(ConversionError, match="not unique"):
        _sidecar(cfg)


def test_a_multi_animal_head_is_reported_by_name(dlc_config):
    cfg = dlc_config()
    cfg["model"]["heads"] = {"paf": {}}

    with pytest.raises(ConversionError, match="paf"):
        _sidecar(cfg)


def test_per_frame_rescaling_is_refused(dlc_config):
    """scale_to_unit_range normalises each frame against its own min and max.
    The sidecar describes a fixed normalisation and cannot express that, so
    converting would mean converting wrongly."""
    cfg = dlc_config(data={"inference": {"scale_to_unit_range": True}})

    with pytest.raises(ConversionError, match="scale_to_unit_range"):
        _sidecar(cfg)


# --- what a person sees when it cannot work ----------------------------------


def test_a_deeplabcut_2x_folder_says_so(tmp_path):
    """A 2.x folder is TensorFlow, not PyTorch. 'no pytorch_config.yaml found'
    would be true and useless."""
    folder = tmp_path / "train"
    folder.mkdir()
    (folder / "pose_cfg.yaml").write_text(yaml.safe_dump({"stride": 8}))
    (folder / "snapshot-100.index").write_bytes(b"tf")

    with pytest.raises(ConversionError, match="2.x"):
        convert_dlc_to_onnx(folder)


def test_a_config_with_no_weights_says_so(dlc_dir):
    with pytest.raises(ConversionError, match="no snapshot"):
        convert_dlc_to_onnx(dlc_dir(snapshots=()))


def test_a_folder_with_neither_says_so(tmp_path):
    with pytest.raises(ConversionError, match="pytorch_config.yaml"):
        convert_dlc_to_onnx(tmp_path)


# --- the stride --------------------------------------------------------------
#
# The number every decoded keypoint is multiplied by, and the one thing here
# that is measured from the network rather than read from its config -- because
# the effective stride is the backbone's divided by whatever the head's
# deconvolutions undo, and those move between backbones. A ResNet-50 and an
# HRNet-w32 trained on the same data differ by a factor of four.


def test_the_stride_is_a_difference_not_a_ratio():
    """Every DeepLabCut head adds one cell past the edge: out = in/stride + 1.
    Measured against real networks -- a ResNet-50 gives 17 cells at 128 px and
    65 at 512, an HRNet-w32 gives 65 and 257. A single ratio reads the first as
    7.53 and the second as 1.97; the difference gives 8 and 2 exactly."""
    from glider_dlc.convert import stride_from_sizes

    assert stride_from_sizes(128, 512, 17, 65) == 8.0
    assert stride_from_sizes(128, 512, 65, 257) == 2.0


def test_a_single_ratio_would_have_been_wrong():
    """Kept as the reason the function has the shape it does."""
    assert 128 / 17 == pytest.approx(7.53, abs=0.01)
    assert 512 / 65 == pytest.approx(7.88, abs=0.01)


def test_a_fractional_stride_is_refused():
    """GLIDER decodes against an integer stride. A model that does not have one
    would have every keypoint placed wrongly, quietly."""
    from glider_dlc.convert import stride_from_sizes

    with pytest.raises(ConversionError, match="whole number"):
        stride_from_sizes(128, 512, 17, 60)


def test_an_output_that_does_not_grow_is_refused():
    """A fixed-size output cannot be mapped back to pixels at all."""
    from glider_dlc.convert import stride_from_sizes

    with pytest.raises(ConversionError, match="does not grow"):
        stride_from_sizes(128, 512, 33, 33)


# --- the real thing ----------------------------------------------------------


def _need(module: str):
    """Import *module* or skip -- tolerating a broken install, not just a missing one.

    ``pytest.importorskip`` catches ImportError only, and GLIDER's own
    environment can hold a torch that raises OSError on import, from a partial
    wheel or a missing DLL. That is still "not available here", and failing on
    it would report somebody's environment as a defect in this plugin.
    """
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - any import failure means "not here"
        pytest.skip(f"{module} is not usable in this environment: {exc}")


@pytest.mark.slow
def test_a_real_model_converts_and_matches_deeplabcut(tmp_path):
    """The whole chain against DeepLabCut itself, when it is installed here.

    Skipped everywhere else, including CI: the point of this plugin is that
    DeepLabCut is *not* in GLIDER's environment. Run it inside the provisioned
    one, where `deeplabcut` imports:

        ~/.glider/envs/deeplabcut/Scripts/python -m pytest plugins/glider-dlc -m slow
    """
    torch = _need("torch")
    ort = _need("onnxruntime")
    np = _need("numpy")
    yaml_mod = _need("yaml")
    models = _need("deeplabcut.pose_estimation_pytorch.models")

    from glider_dlc.convert import convert_dlc_to_onnx

    bodyparts = ["snout", "leftear", "rightear", "tailbase"]
    cfg = {
        "metadata": {"bodyparts": bodyparts},
        "model": {
            "backbone": {"type": "ResNet", "model_name": "resnet50_gn", "output_stride": 16},
            "backbone_output_channels": 2048,
            "heads": {
                "bodypart": {
                    "type": "HeatmapHead",
                    "weight_init": "normal",
                    "predictor": {
                        "type": "HeatmapPredictor",
                        "apply_sigmoid": False,
                        "location_refinement": True,
                        "locref_std": 7.2801,
                    },
                    "target_generator": {
                        "type": "HeatmapGaussianGenerator",
                        "num_heatmaps": len(bodyparts),
                        "pos_dist_thresh": 17,
                        "generate_locref": True,
                        "locref_std": 7.2801,
                    },
                    "criterion": {
                        "heatmap": {"type": "WeightedMSECriterion", "weight": 1.0},
                        "locref": {"type": "WeightedHuberCriterion", "weight": 0.05},
                    },
                    "heatmap_config": {
                        "channels": [2048, len(bodyparts)],
                        "kernel_size": [3],
                        "strides": [2],
                    },
                    "locref_config": {
                        "channels": [2048, 2 * len(bodyparts)],
                        "kernel_size": [3],
                        "strides": [2],
                    },
                }
            },
        },
        "data": {"inference": {"normalize_images": True}},
    }

    folder = tmp_path / "train"
    folder.mkdir()
    (folder / "pytorch_config.yaml").write_text(yaml_mod.safe_dump(cfg))
    model = models.PoseModel.build(cfg["model"], pretrained_backbone=False)
    torch.save({"model": model.state_dict()}, folder / "snapshot-200.pt")

    result = convert_dlc_to_onnx(folder)

    assert result["output_stride"] == 8.0, "ResNet-50 with a stride-2 head is 8"

    sidecar = json.loads((folder / SIDECAR_NAME).read_text())
    assert sidecar["keypoint_names"] == bodyparts
    assert is_conversion_current(folder)

    # The exported graph has to be the same network, not merely a valid one.
    model.eval()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 256, 256), dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x))["bodypart"]["heatmap"].numpy()

    session = ort.InferenceSession(result["onnx"], providers=["CPUExecutionProvider"])
    actual, _ = session.run(None, {"image": x})

    # Relative: an untrained network's outputs can be enormous, and an absolute
    # tolerance would say nothing about whether the graph matches.
    error = np.abs(expected - actual).max() / max(np.abs(expected).max(), 1e-9)
    assert error < 1e-5, error


# --- looking for the model ---------------------------------------------------


def test_the_snapshot_comes_from_beside_its_own_config(tmp_path):
    """A project with two training runs has two configs and two sets of weights.
    Pairing one run's config with another's snapshot is not an error that
    surfaces: load_state_dict accepts it whenever the shapes happen to match,
    and the model then answers with a network nobody trained for this."""
    for run, epoch in (("runA", 100), ("runB", 900)):
        d = tmp_path / "project" / "dlc-models-pytorch" / run / "train"
        d.mkdir(parents=True)
        (d / "pytorch_config.yaml").write_text(yaml.safe_dump({"run": run}))
        (d / f"snapshot-{epoch}.pt").write_bytes(b"weights")

    config = find_dlc_config(tmp_path / "project")
    snapshot = find_dlc_snapshot(tmp_path / "project")

    assert snapshot.parent == config.parent, (config, snapshot)


def test_the_search_does_not_walk_the_whole_project(tmp_path):
    """`claims` runs on every model selection, and a DLC project also holds
    labeled-data with thousands of frames in it. Walking that to answer 'is
    this yours?' would stall the panel."""
    from glider_dlc.convert import MAX_SEARCH_DEPTH, _walk

    deep = tmp_path
    for i in range(MAX_SEARCH_DEPTH + 3):
        deep = deep / f"level{i}"
    deep.mkdir(parents=True)
    (deep / "pytorch_config.yaml").write_text("{}")

    assert find_dlc_config(tmp_path) is None
    assert next(_walk(tmp_path, "pytorch_config.yaml", depth=99), None) is not None
