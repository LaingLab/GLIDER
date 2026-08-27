"""The adapter GLIDER talks to.

The environment is mocked throughout: these assert the orchestration, not that
1.5 GB downloads.
"""

import json

import pytest
from glider_sleap_nn import convert as convert_module
from glider_sleap_nn import env as env_module
from glider_sleap_nn.converter import SleapNNConverter


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "training_config.yaml").write_text("model_config: {}\n")
    (tmp_path / "best.ckpt").write_bytes(b"PK\x03\x04stub")
    return tmp_path


def test_it_registers_under_the_pose_entry_point():
    import glider_sleap_nn

    assert glider_sleap_nn.POSE_CONVERTERS == {"sleap_nn": SleapNNConverter}


def test_the_label_distinguishes_it_from_classic_sleap():
    """Which SLEAP it is changes what happens next: one downloads 1.5 GB."""
    assert SleapNNConverter.label == "SLEAP (PyTorch)"


def test_claims_only_its_own_folders(model_dir, tmp_path):
    assert SleapNNConverter().claims(model_dir)
    classic = tmp_path / "classic"
    classic.mkdir()
    (classic / "training_config.json").write_text("{}")
    (classic / "best_model.h5").write_bytes(b"stub")
    assert not SleapNNConverter().claims(classic)


def test_not_current_until_everything_is_there(model_dir):
    c = SleapNNConverter()
    assert not c.is_current(model_dir)
    (model_dir / "model.onnx").write_bytes(b"stub")
    (model_dir / "glider_pose.json").write_text("{}")
    assert not c.is_current(model_dir)  # still no stamp
    (model_dir / convert_module.STAMP_NAME).write_text(
        json.dumps(convert_module._stamp_for(model_dir / "best.ckpt"))
    )
    assert c.is_current(model_dir)


def test_retraining_into_the_same_folder_goes_stale(model_dir):
    """The whole point of the stamp: a conversion must not answer with the
    network the researcher just replaced."""
    (model_dir / "model.onnx").write_bytes(b"stub")
    (model_dir / "glider_pose.json").write_text("{}")
    (model_dir / convert_module.STAMP_NAME).write_text(
        json.dumps(convert_module._stamp_for(model_dir / "best.ckpt"))
    )
    assert SleapNNConverter().is_current(model_dir)

    (model_dir / "best.ckpt").write_bytes(b"PK\x03\x04a much longer new checkpoint")
    assert not SleapNNConverter().is_current(model_dir)


def test_convert_provisions_before_running(model_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(env_module, "provision", lambda *a, **k: calls.append("provision"))
    monkeypatch.setattr(env_module, "run_converter", lambda *a, **k: (calls.append("run"), "{}")[1])
    SleapNNConverter().convert(model_dir)
    assert calls == ["provision", "run"]


def test_preflight_is_quiet_once_provisioned(model_dir, monkeypatch):
    monkeypatch.setattr(env_module, "is_provisioned", lambda *a, **k: True)
    assert SleapNNConverter().preflight(model_dir) is None


def test_preflight_warns_before_the_first_download(model_dir, monkeypatch):
    """Starting a 1.5 GB download behind a wait cursor is how a working
    conversion gets killed by someone who assumes GLIDER has hung."""
    monkeypatch.setattr(env_module, "is_provisioned", lambda *a, **k: False)
    message = SleapNNConverter().preflight(model_dir)
    assert "GB" in message and "own environment" in message


def test_describe_reads_the_stamp_and_tolerates_its_absence(model_dir):
    assert SleapNNConverter().describe(model_dir) is None
    (model_dir / convert_module.STAMP_NAME).write_text(json.dumps({"source": "best.ckpt"}))
    assert SleapNNConverter().describe(model_dir) == {"source": "best.ckpt"}


def test_the_adapter_imports_no_heavy_dependency(heavy_imports_after):
    """GLIDER imports this at startup to register the entry point.

    Pulling torch in at that moment would cost every launch, whether or not the
    lab has ever opened SLEAP.
    """
    assert heavy_imports_after("glider_sleap_nn") == []
