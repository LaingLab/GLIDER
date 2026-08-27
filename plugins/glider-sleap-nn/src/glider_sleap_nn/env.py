"""The sleap-nn environment GLIDER builds for itself.

Deliberately a near-copy of :mod:`glider_dlc.env`. The two plugins have the
same problem and the same answer, and they ship as independent packages -- so a
shared library between them would mean publishing a third one to save a hundred
lines. The repo already prefers the copy: ``release-glider-dlc.yml`` opens by
noting it is a near-copy of ``release-glider-harp.yml`` on purpose.

sleap-nn cannot live in GLIDER's environment. It is needed to *read* a
checkpoint at all -- a sleap-nn model is a PyTorch Lightning save, and
rebuilding the network from one needs sleap-nn's own model classes -- but it
brings torch, torchvision and lightning with it, which is not something to hand
a lab that tracks with YOLO.

Note this is the opposite of ``glider-sleap``'s situation, and the difference is
not arbitrary: a *classic* SLEAP model is an ordinary Keras checkpoint that
TensorFlow opens with nothing from ``sleap`` involved, so that plugin depends on
``tensorflow-cpu`` directly and never needs an environment of its own.

The alternative to putting sleap-nn in GLIDER's environment is not making the
researcher build one by hand. It is building one for them, once, the first time
they select a sleap-nn model, and keeping it. That is what this module does:
``uv`` creates a private virtualenv under ``~/.glider/envs`` and installs
sleap-nn into it, and the conversion runs in that interpreter.

A lab that already has a working sleap-nn environment can point at it with
``GLIDER_SLEAP_NN_ENV`` and nothing is downloaded at all.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

#: Python for the child environment. Pinned rather than inherited: GLIDER runs
#: on 3.11-3.13 and sleap-nn need not track that range, so following GLIDER's
#: interpreter would make provisioning succeed or fail depending on which
#: Python the researcher happened to install GLIDER under.
ENV_PYTHON = "3.12"

#: What goes in.
#:
#: onnx and onnxscript are listed because torch does not depend on either, and
#: ``torch.onnx.export`` fails at the end of a long export without them.
#:
#: onnxruntime is listed for a subtler reason: sleap-nn's exporter takes a
#: ``numerical_check`` flag that runs the exported graph and asserts parity
#: against PyTorch, and it *degrades to a warning* when onnxruntime is missing
#: rather than failing. That check is the one thing that catches a graph which
#: traces cleanly and is numerically wrong, so losing it silently is not
#: acceptable -- it is cheap here and it stays.
#:
#: The pin is conservative because sleap-nn is young (0.3.3, August 2026) and
#: this plugin uses exactly two entry points from it.
ENV_PACKAGES = (
    "sleap-nn>=0.3.3,<0.4",
    "onnx>=1.16",
    "onnxscript>=0.1",
    "onnxruntime>=1.17",
    "pyyaml",
)

#: Written into the environment once it is complete. Its absence means a
#: half-built environment -- an interrupted download leaves an interpreter that
#: imports nothing -- so provisioning keys off this rather than off the
#: directory existing.
STAMP_NAME = ".glider_env.json"

#: Roughly what lands on disk, for the sentence shown before the download
#: starts. Measured, not estimated: 1.5 GB for sleap-nn 0.3.3 with torch 2.13
#: on Windows, 2026-08-27.
INSTALLED_SIZE_GB = 1.5

#: Generous. This is a multi-gigabyte download on whatever connection the lab
#: has, and killing a working install is worse than waiting for a slow one.
PROVISION_TIMEOUT_S = 3600

#: A conversion itself is a graph trace of a network already on disk.
CONVERT_TIMEOUT_S = 900

NO_UV_HINT = (
    "Building the sleap-nn environment needs `uv`, which is what GLIDER's "
    "own installer uses and is usually already present.\n\n"
    "Install it from https://docs.astral.sh/uv/ and try again, or point "
    "GLIDER at a sleap-nn environment you already have by setting the "
    "GLIDER_SLEAP_NN_ENV environment variable to it."
)


class ProvisioningError(RuntimeError):
    """The sleap-nn environment could not be built.

    Carries a sentence meant for a researcher: it goes on screen verbatim.
    """


def env_dir() -> Path:
    """Where the sleap-nn environment lives.

    ``GLIDER_SLEAP_NN_ENV`` overrides it, which is the escape hatch for a lab that
    already has sleap-nn working somewhere and would rather not have a second
    copy: point it at that virtualenv and nothing is downloaded.
    """
    override = os.environ.get("GLIDER_SLEAP_NN_ENV")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".glider" / "envs" / "sleap-nn"


def interpreter(env: Path | None = None) -> Path:
    """The Python inside *env*."""
    env = Path(env) if env is not None else env_dir()
    if sys.platform == "win32":
        return env / "Scripts" / "python.exe"
    return env / "bin" / "python"


def _spec() -> dict:
    return {"python": ENV_PYTHON, "packages": list(ENV_PACKAGES)}


def is_provisioned(env: Path | None = None) -> bool:
    """Whether *env* is a complete environment built for the current spec.

    Compares against the recorded spec rather than merely checking that the
    interpreter exists. A plugin upgrade that changes ``ENV_PACKAGES`` has to
    rebuild -- otherwise a fix to the pinned sleap-nn version would apply to
    new users only, and silently, which is the shape of bug that gets blamed on
    the model.

    A ``GLIDER_SLEAP_NN_ENV`` the lab manages is exempt: it has no stamp and is not
    ours to rebuild, so an interpreter there is taken at its word.
    """
    env = Path(env) if env is not None else env_dir()
    if not interpreter(env).is_file():
        return False
    if os.environ.get("GLIDER_SLEAP_NN_ENV"):
        return True
    try:
        return json.loads((env / STAMP_NAME).read_text()) == _spec()
    except (OSError, ValueError):
        return False


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise ProvisioningError(NO_UV_HINT)
    return uv


def _run(args: list[str], *, what: str, timeout: int) -> None:
    logger.info("sleap-nn environment: %s", what)
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProvisioningError(
            f"{what} took longer than {timeout // 60} minutes and was stopped."
        ) from exc
    except OSError as exc:
        raise ProvisioningError(f"{what} could not be started: {exc}") from exc

    if completed.returncode != 0:
        # uv's last few lines say what actually went wrong -- an unresolvable
        # pin, no network, no matching Python. The whole log is on the logger
        # for anyone who wants it; the dialog gets the end of it.
        logger.error("sleap-nn environment: %s failed\n%s", what, completed.stderr)
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-6:])
        raise ProvisioningError(f"{what} failed.\n\n{tail}".strip())


def provision(env: Path | None = None, *, on_progress: Callable[[str], None] | None = None) -> Path:
    """Build the sleap-nn environment, and return its interpreter.

    Idempotent: an environment already built for the current spec is returned
    untouched. Otherwise a partial one is removed first -- resuming an install
    into a virtualenv whose interpreter is missing does not work, and the
    failure it produces is unreadable.

    Raises:
        ProvisioningError: with a sentence to show the researcher.
    """
    env = Path(env) if env is not None else env_dir()
    if is_provisioned(env):
        return interpreter(env)

    if os.environ.get("GLIDER_SLEAP_NN_ENV"):
        raise ProvisioningError(
            f"GLIDER_SLEAP_NN_ENV points at {env}, which has no Python in it. "
            "Point it at a virtualenv with sleap-nn installed, or unset it "
            "and let GLIDER build its own."
        )

    def say(message: str) -> None:
        logger.info("%s", message)
        if on_progress is not None:
            on_progress(message)

    uv = _uv()
    stamp = env / STAMP_NAME
    if stamp.is_file():
        stamp.unlink()
    if env.exists():
        say(f"Removing the incomplete environment at {env}")
        shutil.rmtree(env, ignore_errors=True)

    env.parent.mkdir(parents=True, exist_ok=True)
    say(f"Creating a Python {ENV_PYTHON} environment at {env}")
    _run(
        [uv, "venv", "--python", ENV_PYTHON, str(env)],
        what=f"Creating the Python {ENV_PYTHON} environment",
        timeout=PROVISION_TIMEOUT_S,
    )

    say(f"Installing sleap-nn (about {INSTALLED_SIZE_GB} GB, once)")
    _run(
        [uv, "pip", "install", "--python", str(interpreter(env)), *ENV_PACKAGES],
        what="Installing sleap-nn",
        timeout=PROVISION_TIMEOUT_S,
    )

    stamp.write_text(json.dumps(_spec(), indent=2))
    say("sleap-nn environment ready")
    return interpreter(env)


def run_converter(
    script: Path,
    model_dir: Path,
    *,
    env: Path | None = None,
    timeout: int = CONVERT_TIMEOUT_S,
) -> str:
    """Run *script* over *model_dir* in the sleap-nn environment.

    Raises:
        ProvisioningError: if the conversion fails, carrying the child's own
            message -- which is one sentence written for a person, because the
            converter prints only that to stderr.
    """
    python = interpreter(env)
    try:
        completed = subprocess.run(
            [str(python), str(script), str(model_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
            # torch's exporter prints a check mark on success, which raises
            # UnicodeEncodeError on a Windows console still defaulting to
            # cp1252 -- a conversion that worked, reported as a crash.
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired as exc:
        raise ProvisioningError(
            f"Converting {model_dir.name} took longer than {timeout // 60} "
            "minutes and was stopped."
        ) from exc

    if completed.returncode != 0:
        message = (completed.stderr or "").strip() or "Conversion failed."
        raise ProvisioningError(message)
    return completed.stdout
