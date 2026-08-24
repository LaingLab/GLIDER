"""The adapter GLIDER actually talks to.

``claims`` and ``is_current`` are asked on *every* model selection, including
for folders belonging to other vendors, so the important property here is what
they do **not** do: import TensorFlow. Only ``convert`` may, and it does so in a
child process.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from glider_sleap import POSE_CONVERTERS
from glider_sleap.converter import SleapConverter


def _sleap_dir(tmp_path: Path, config=True, checkpoint=True) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    if config:
        (d / "training_config.json").write_text(json.dumps({"model": {"heads": {}}}))
    if checkpoint:
        (d / "best_model.h5").write_bytes(b"not really a model")
    return d


# --- registration -------------------------------------------------------------


def test_the_plugin_exposes_a_converter():
    assert POSE_CONVERTERS == {"sleap": SleapConverter}


def test_it_satisfies_the_converter_protocol():
    """Structural check against core's Protocol, so a signature drifting apart
    from what GLIDER calls is caught here rather than at a researcher's bench."""
    from glider.vision.pose.converters import PoseConverter

    assert isinstance(SleapConverter(), PoseConverter)


# --- claiming -----------------------------------------------------------------


def test_it_claims_a_sleap_folder(tmp_path):
    assert SleapConverter().claims(_sleap_dir(tmp_path)) is True


def test_a_config_without_a_checkpoint_is_not_claimed(tmp_path):
    """An already-exported folder has nothing left to convert; claiming it would
    put a pointless 'Convert?' prompt in front of a working model."""
    assert SleapConverter().claims(_sleap_dir(tmp_path, checkpoint=False)) is False


def test_a_stray_checkpoint_is_not_claimed(tmp_path):
    """A .h5 on its own is not a SLEAP model, and claiming other people's files
    is how two plugins end up fighting over one folder."""
    assert SleapConverter().claims(_sleap_dir(tmp_path, config=False)) is False


def test_an_unrelated_folder_is_not_claimed(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"yolo")

    assert SleapConverter().claims(tmp_path) is False


def test_claiming_does_not_import_tensorflow(tmp_path, monkeypatch):
    """The property that keeps model selection instant. TensorFlow takes seconds
    to import, and this is asked every time anyone picks any model."""
    import builtins

    real = builtins.__import__

    def _refuse(name, *args, **kwargs):
        if name.split(".")[0] in {"tensorflow", "tf2onnx"}:
            raise AssertionError(f"claims() imported {name}")
        return real(name, *args, **kwargs)

    folder = _sleap_dir(tmp_path)
    monkeypatch.setattr(builtins, "__import__", _refuse)
    converter = SleapConverter()

    assert converter.claims(folder) is True
    assert converter.is_current(folder) is False


# --- converting ---------------------------------------------------------------


def test_convert_runs_the_module_as_a_child_process(tmp_path, monkeypatch):
    """A subprocess, not a call: TensorFlow would otherwise sit in the
    application process for the rest of the session."""
    calls: list[list[str]] = []

    def _run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    folder = _sleap_dir(tmp_path)

    SleapConverter().convert(folder)

    assert len(calls) == 1
    assert calls[0][1].endswith("convert.py")
    assert calls[0][2] == str(folder)


def test_a_failing_conversion_raises_the_childs_message(tmp_path, monkeypatch):
    """The child writes one actionable sentence to stderr. Anything else here
    would put a TensorFlow traceback in a dialog."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 1, "", "install glider-sleap first"),
    )

    with pytest.raises(RuntimeError, match="install glider-sleap first"):
        SleapConverter().convert(_sleap_dir(tmp_path))


def test_a_silent_failure_still_says_something(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 1, "", "   ")
    )

    with pytest.raises(RuntimeError, match="Conversion failed"):
        SleapConverter().convert(_sleap_dir(tmp_path))


def test_a_wedged_conversion_is_reported_in_minutes(tmp_path, monkeypatch):
    def _timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 900)

    monkeypatch.setattr(subprocess, "run", _timeout)

    with pytest.raises(RuntimeError, match="minutes"):
        SleapConverter().convert(_sleap_dir(tmp_path))
