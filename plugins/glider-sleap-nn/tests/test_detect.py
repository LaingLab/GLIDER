"""Which folders this plugin claims.

`claims` is asked on every model selection, including for folders belonging to
other vendors, so these tests also pin that the answer is reachable without
importing torch or sleap-nn.
"""

import pytest
from glider_sleap_nn.convert import is_sleap_nn_folder


def _nn(tmp_path):
    (tmp_path / "training_config.yaml").write_text("head_configs: {}\n")
    (tmp_path / "best.ckpt").write_bytes(b"PK\x03\x04stub")
    return tmp_path


def test_claims_a_sleap_nn_folder(tmp_path):
    assert is_sleap_nn_folder(_nn(tmp_path))


def test_rejects_a_classic_sleap_folder(tmp_path):
    """The other generation. Shares no filename with this one."""
    (tmp_path / "training_config.json").write_text("{}")
    (tmp_path / "best_model.h5").write_bytes(b"stub")
    assert not is_sleap_nn_folder(tmp_path)


@pytest.mark.parametrize("drop", ["training_config.yaml", "best.ckpt"])
def test_needs_both_files(tmp_path, drop):
    """A config without weights describes a model; a stray .ckpt is not one."""
    _nn(tmp_path)
    (tmp_path / drop).unlink()
    assert not is_sleap_nn_folder(tmp_path)


def test_rejects_a_file_and_a_missing_path(tmp_path):
    labels = tmp_path / "labels.slp"
    labels.write_bytes(b"stub")
    assert not is_sleap_nn_folder(labels)
    assert not is_sleap_nn_folder(tmp_path / "nope")


def test_detection_does_not_import_torch(heavy_imports_after):
    """The whole point of keeping detection in path work.

    `claims` runs for every model the operator selects, whichever vendor wrote
    it. If importing this module pulled in torch, every selection would pay for
    it -- and on a machine with no sleap-nn environment it would fail outright.
    """
    assert heavy_imports_after("glider_sleap_nn.convert") == []
