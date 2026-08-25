"""The adapter GLIDER actually talks to.

``claims`` and ``is_current`` are asked on *every* model selection, including
for folders belonging to other vendors, so the important property here is what
they do **not** do: import torch or deeplabcut, or go looking for an
environment. Only ``convert`` does.
"""

from __future__ import annotations

import pytest

from glider_dlc import POSE_CONVERTERS
from glider_dlc import env as env_module
from glider_dlc.converter import DlcConverter

# --- registration -------------------------------------------------------------


def test_the_plugin_exposes_a_converter():
    assert POSE_CONVERTERS == {"deeplabcut": DlcConverter}


def test_it_satisfies_the_converter_protocol():
    """Structural check against core's Protocol, so a signature drifting apart
    from what GLIDER calls is caught here rather than at a researcher's bench."""
    from glider.vision.pose.converters import PoseConverter

    assert isinstance(DlcConverter(), PoseConverter)


# --- claiming -----------------------------------------------------------------


def test_it_claims_a_deeplabcut_folder(dlc_dir):
    assert DlcConverter().claims(dlc_dir()) is True


def test_it_leaves_other_vendors_alone(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"yolo")

    assert DlcConverter().claims(tmp_path) is False


def test_a_converted_folder_is_current(converted):
    assert DlcConverter().is_current(converted) is True


def test_claiming_imports_nothing_heavy(dlc_dir, monkeypatch):
    """The property that keeps model selection instant. torch takes seconds to
    import, and this is asked every time anyone picks any model."""
    import builtins

    real = builtins.__import__
    folder = dlc_dir()

    def _refuse(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "deeplabcut", "timm"}:
            raise AssertionError(f"claims() imported {name}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _refuse)
    converter = DlcConverter()

    assert converter.claims(folder) is True
    assert converter.is_current(folder) is False


# --- the disclosure -----------------------------------------------------------


def test_the_first_conversion_warns_about_the_download(dlc_dir, monkeypatch):
    """Starting a multi-gigabyte download behind a wait cursor with no warning
    is how a working conversion gets killed by someone who assumes GLIDER
    hung."""
    monkeypatch.setattr(env_module, "is_provisioned", lambda *a, **k: False)

    note = DlcConverter().preflight(dlc_dir())

    assert note and "GB" in note


def test_later_conversions_say_nothing(dlc_dir, monkeypatch):
    """Once the environment is there the conversion is a graph trace of a
    network already on disk, and a warning would be noise."""
    monkeypatch.setattr(env_module, "is_provisioned", lambda *a, **k: True)

    assert DlcConverter().preflight(dlc_dir()) is None


def test_core_shows_the_disclosure(dlc_dir, monkeypatch):
    """preflight is deliberately off the Protocol -- adding it there would make
    every converter without it stop satisfying the check -- so the way core
    finds it is worth pinning down."""
    from glider.vision.pose.converters import converter_preflight

    monkeypatch.setattr(env_module, "is_provisioned", lambda *a, **k: False)

    assert "GB" in converter_preflight(DlcConverter(), dlc_dir())


# --- converting ---------------------------------------------------------------


def test_convert_provisions_then_runs(dlc_dir, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(env_module, "provision", lambda *a, **k: order.append("provision"))
    monkeypatch.setattr(env_module, "run_converter", lambda *a, **k: order.append("run") or "{}")
    folder = dlc_dir()

    DlcConverter().convert(folder)

    assert order == ["provision", "run"]


def test_it_runs_the_standalone_script(dlc_dir, monkeypatch):
    """The script has to be the one on disk, not an import: the interpreter it
    runs in has neither glider nor glider_dlc installed."""
    seen: dict = {}
    monkeypatch.setattr(env_module, "provision", lambda *a, **k: None)
    monkeypatch.setattr(
        env_module,
        "run_converter",
        lambda script, model_dir, **k: seen.update(script=script, model=model_dir) or "{}",
    )
    folder = dlc_dir()

    DlcConverter().convert(folder)

    assert seen["script"].name == "convert.py"
    assert seen["model"] == folder


def test_a_provisioning_failure_reaches_the_caller(dlc_dir, monkeypatch):
    """The panel turns whatever comes out of here into the dialog, so the
    message from `uv` must not be swallowed on the way."""

    def _boom(*a, **k):
        raise env_module.ProvisioningError("uv is not installed")

    monkeypatch.setattr(env_module, "provision", _boom)

    with pytest.raises(env_module.ProvisioningError, match="uv is not installed"):
        DlcConverter().convert(dlc_dir())
