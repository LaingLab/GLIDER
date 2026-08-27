"""The private sleap-nn environment.

Nothing here downloads anything: provisioning is asserted through the argv it
would run, not by running it.
"""

import json

import pytest
from glider_sleap_nn import env


@pytest.fixture(autouse=True)
def _no_inherited_override(monkeypatch):
    """The developer's own GLIDER_SLEAP_NN_ENV must not decide these."""
    monkeypatch.delenv("GLIDER_SLEAP_NN_ENV", raising=False)


def test_default_lives_under_glider_envs():
    assert env.env_dir().parts[-3:] == (".glider", "envs", "sleap-nn")


def test_override_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("GLIDER_SLEAP_NN_ENV", str(tmp_path / "mine"))
    assert env.env_dir() == tmp_path / "mine"


def _fake_interpreter(tmp_path):
    interp = env.interpreter(tmp_path)
    interp.parent.mkdir(parents=True, exist_ok=True)
    interp.write_text("")
    return interp


def test_an_interpreter_alone_is_not_provisioned(tmp_path):
    """An interrupted download leaves a Python that imports nothing."""
    _fake_interpreter(tmp_path)
    assert not env.is_provisioned(tmp_path)


def test_provisioned_when_the_stamp_matches_the_current_spec(tmp_path):
    _fake_interpreter(tmp_path)
    (tmp_path / env.STAMP_NAME).write_text(
        json.dumps({"python": env.ENV_PYTHON, "packages": list(env.ENV_PACKAGES)})
    )
    assert env.is_provisioned(tmp_path)


def test_a_stale_stamp_forces_a_rebuild(tmp_path):
    """Otherwise a bumped sleap-nn pin would apply to new users only, silently."""
    _fake_interpreter(tmp_path)
    (tmp_path / env.STAMP_NAME).write_text(
        json.dumps({"python": env.ENV_PYTHON, "packages": ["sleap-nn==0.0.1"]})
    )
    assert not env.is_provisioned(tmp_path)


def test_a_lab_managed_env_is_taken_at_its_word(tmp_path, monkeypatch):
    """It has no stamp and is not ours to rebuild."""
    monkeypatch.setenv("GLIDER_SLEAP_NN_ENV", str(tmp_path))
    _fake_interpreter(tmp_path)
    assert env.is_provisioned(tmp_path)


def test_onnxruntime_is_in_the_spec():
    """sleap-nn's numerical parity check degrades to a warning without it.

    That check is the only thing that catches a graph which traces cleanly and
    is numerically wrong, so it must not be lost by omission.
    """
    assert any(p.startswith("onnxruntime") for p in env.ENV_PACKAGES)


def test_the_sleap_nn_pin_is_bounded():
    """A young 0.x package: an unbounded pin is a future silent breakage."""
    pin = next(p for p in env.ENV_PACKAGES if p.startswith("sleap-nn"))
    assert "<" in pin


def test_missing_uv_is_reported_with_a_way_out(tmp_path, monkeypatch):
    monkeypatch.setattr(env.shutil, "which", lambda _: None)
    with pytest.raises(env.ProvisioningError) as exc:
        env.provision(tmp_path)
    assert "uv" in str(exc.value) and "GLIDER_SLEAP_NN_ENV" in str(exc.value)


def test_an_override_pointing_at_nothing_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("GLIDER_SLEAP_NN_ENV", str(tmp_path / "empty"))
    with pytest.raises(env.ProvisioningError, match="GLIDER_SLEAP_NN_ENV"):
        env.provision(tmp_path / "empty")
