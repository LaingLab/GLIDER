"""Converting a SLEAP model to ONNX inside GLIDER.

The premise this replaces was that conversion needs SLEAP, and that SLEAP's
Python ceiling therefore puts it out of GLIDER's reach. It needs *TensorFlow*,
which is a different package with a different ceiling. Verified against SLEAP's
own ``minimal_robot.UNet.single_instance`` fixture: it loads with plain Keras,
converts in one call, and the ONNX matches the original to 7e-7.

TensorFlow is not in CI, so the conversion itself is exercised by the one test
at the bottom, which skips without it. Everything else here -- detection,
staleness, the error paths a researcher will actually hit -- needs no TF and
runs everywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from glider.vision.pose.convert import (
    STAMP_NAME,
    ConversionError,
    convert_sleap_to_onnx,
    find_sleap_checkpoint,
    is_conversion_current,
    main,
    needs_conversion,
)


def _sleap_dir(tmp_path: Path, checkpoint="best_model.h5") -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "training_config.json").write_text(json.dumps({"model": {"heads": {}}}))
    if checkpoint:
        (d / checkpoint).write_bytes(b"not really a model")
    return d


def _stamp(model_dir: Path, onnx: Path) -> None:
    checkpoint = find_sleap_checkpoint(model_dir)
    stat = checkpoint.stat()
    onnx.with_name(STAMP_NAME).write_text(
        json.dumps({"source": checkpoint.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    )


# --- finding the checkpoint ---------------------------------------------------


def test_best_model_is_preferred(tmp_path):
    d = _sleap_dir(tmp_path)
    (d / "final_model.h5").write_bytes(b"x")

    assert find_sleap_checkpoint(d).name == "best_model.h5"


def test_final_model_is_accepted(tmp_path):
    """A run stopped early leaves only final_model.h5."""
    d = _sleap_dir(tmp_path, checkpoint="final_model.h5")

    assert find_sleap_checkpoint(d).name == "final_model.h5"


def test_any_h5_is_a_last_resort(tmp_path):
    d = _sleap_dir(tmp_path, checkpoint=None)
    (d / "something.h5").write_bytes(b"x")

    assert find_sleap_checkpoint(d).name == "something.h5"


def test_no_checkpoint_is_none(tmp_path):
    assert find_sleap_checkpoint(_sleap_dir(tmp_path, checkpoint=None)) is None


# --- when conversion is needed ------------------------------------------------


def test_a_sleap_folder_without_onnx_needs_conversion(tmp_path):
    assert needs_conversion(_sleap_dir(tmp_path)) is True


def test_a_converted_folder_does_not(tmp_path):
    """Otherwise every model selection would re-run a minute of TensorFlow."""
    d = _sleap_dir(tmp_path)
    onnx = d / "model.onnx"
    onnx.write_bytes(b"onnx")
    _stamp(d, onnx)

    assert needs_conversion(d) is False


def test_a_retrained_checkpoint_needs_reconversion(tmp_path):
    """The dangerous case. A stale ONNX runs fine and silently answers with the
    network the researcher just retrained away."""
    d = _sleap_dir(tmp_path)
    onnx = d / "model.onnx"
    onnx.write_bytes(b"onnx")
    _stamp(d, onnx)
    (d / "best_model.h5").write_bytes(b"a different, larger model")

    assert needs_conversion(d) is True


def test_an_onnx_with_no_stamp_is_not_trusted(tmp_path):
    """Could have come from anywhere, including a different model entirely."""
    d = _sleap_dir(tmp_path)
    (d / "model.onnx").write_bytes(b"onnx")

    assert needs_conversion(d) is True


def test_a_corrupt_stamp_is_not_trusted(tmp_path):
    d = _sleap_dir(tmp_path)
    onnx = d / "model.onnx"
    onnx.write_bytes(b"onnx")
    onnx.with_name(STAMP_NAME).write_text("{ this is not json")

    assert is_conversion_current(d, onnx) is False


def test_a_folder_with_no_sleap_config_is_not_our_business(tmp_path):
    d = tmp_path / "yolo"
    d.mkdir()
    (d / "best.pt").write_bytes(b"x")

    assert needs_conversion(d) is False


def test_a_missing_folder_is_not_our_business(tmp_path):
    assert needs_conversion(tmp_path / "nope") is False


def test_a_config_without_a_checkpoint_is_not_convertible(tmp_path):
    """An exported folder holding only the config has nothing to convert."""
    assert needs_conversion(_sleap_dir(tmp_path, checkpoint=None)) is False


# --- the errors a researcher will actually hit --------------------------------


def test_a_missing_checkpoint_is_reported_clearly(tmp_path):
    with pytest.raises(ConversionError, match="does not look like a folder SLEAP produced"):
        convert_sleap_to_onnx(_sleap_dir(tmp_path, checkpoint=None))


def test_a_missing_extra_names_the_install_command(tmp_path, monkeypatch):
    """The single most likely failure, and the one where a traceback would be
    useless: TensorFlow simply is not installed."""
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    monkeypatch.setattr(
        "builtins.__import__",
        _refuse_import("tensorflow"),
    )

    with pytest.raises(ConversionError, match=r"pip install 'glider\[sleap\]'"):
        convert_sleap_to_onnx(_sleap_dir(tmp_path))


def test_the_install_hint_does_not_claim_sleap_is_needed(tmp_path, monkeypatch):
    """It is not, and saying so would send people to install a package that
    will not even sit in the same environment."""
    monkeypatch.setattr("builtins.__import__", _refuse_import("tensorflow"))

    with pytest.raises(ConversionError) as excinfo:
        convert_sleap_to_onnx(_sleap_dir(tmp_path))

    assert "Nothing from SLEAP itself is required" in str(excinfo.value)


def _refuse_import(blocked: str):
    real = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake(name, *args, **kwargs):
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    return _fake


# --- the CLI, which is how GLIDER invokes it ----------------------------------


def test_the_cli_reports_failure_on_stderr_and_exits_nonzero(tmp_path, capsys):
    code = main([str(_sleap_dir(tmp_path, checkpoint=None))])

    assert code == 1
    captured = capsys.readouterr()
    assert "SLEAP produced" in captured.err
    assert captured.out == ""


def test_the_cli_writes_its_result_to_a_file(tmp_path, monkeypatch):
    """TensorFlow logs to stdout without asking, so a caller parsing stdout is
    parsing TensorFlow's mood. The file is not."""
    from glider.vision.pose import convert as convert_module

    out = tmp_path / "result.json"
    fake = convert_module.ConversionResult(
        onnx_path=tmp_path / "model.onnx",
        keypoint_count=2,
        input_shape=(None, 160, 280, 3),
        output_shape=(None, 40, 70, 2),
    )
    monkeypatch.setattr(convert_module, "convert_sleap_to_onnx", lambda *a, **k: fake)

    code = main([str(tmp_path), "--json-out", str(out)])

    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["keypoint_count"] == 2
    assert payload["output_shape"] == [None, 40, 70, 2]


# --- the real thing, where TensorFlow is available ----------------------------


@pytest.mark.slow
def test_a_real_sleap_model_converts_and_matches():
    """End to end against a genuine SLEAP checkpoint.

    Skipped unless TensorFlow, onnxruntime and a model are all present, because
    none of them are in CI. Point SLEAP_TEST_MODEL at a SLEAP model folder --
    e.g. talmolab/sleap's tests/data/models/minimal_robot.UNet.single_instance
    -- to run it.
    """
    import os

    pytest.importorskip("tensorflow", reason="conversion needs the [sleap] extra")
    pytest.importorskip("tf2onnx", reason="conversion needs the [sleap] extra")
    ort = pytest.importorskip("onnxruntime")

    model_dir = os.environ.get("SLEAP_TEST_MODEL")
    if not model_dir:
        pytest.skip("set SLEAP_TEST_MODEL to a SLEAP model folder")

    import numpy as np
    import tensorflow as tf

    result = convert_sleap_to_onnx(model_dir)
    assert result.onnx_path.is_file()
    assert needs_conversion(model_dir) is False, "a fresh conversion should be cached"

    checkpoint = find_sleap_checkpoint(Path(model_dir))
    keras = tf.keras.models.load_model(str(checkpoint), compile=False)
    shape = tuple(d if d is not None else 1 for d in result.input_shape)
    x = np.random.default_rng(0).random(shape, dtype=np.float32)

    session = ort.InferenceSession(str(result.onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {session.get_inputs()[0].name: x})[0]

    # The whole claim: converting does not change what the model says.
    assert np.allclose(keras.predict(x, verbose=0), onnx_out, atol=1e-4)
