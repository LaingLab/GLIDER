"""``glider doctor``: can this machine run GLIDER, and if not, why not.

Checks the install, not a recording -- for what is wrong with a project's
data, see ``tools/project.py doctor``.

Nothing here imports Qt at module level: ``glider.__main__`` dispatches to this
before it loads PyQt6, because a Qt that will not load is one of the things
this has to be able to report.

Every module is really *imported*, never looked up with
``importlib.util.find_spec``. The two disagree in the case that matters: a
packaged build can carry every torch file on disk and still fail to load
``c10.dll``, and find_spec would call that present.

Windows hands a frozen app a cp1252 stdout when it is piped, which is exactly
when someone is capturing this to send us. The report's own text is ASCII, but
the paths and error messages in it are not, so it writes UTF-8 instead.
"""

from __future__ import annotations

import importlib
import sys
import warnings

#: Feature stacks in the shape the user meets them: the feature, the extra that
#: installs it ("-" for core), and the modules that must import for it to work.
#: Grouped rather than flat because "hdbscan is missing" means nothing to
#: someone whose actual complaint is that the Behavior menu is greyed out.
#:
#: The first module is the anchor. If it is not installed at all, the extra was
#: simply never chosen, which is not a fault. If it is, every other module must
#: import too: a half-present stack is how a bundle missing hdbscan greyed out
#: the whole Behavior menu.
DEP_STACKS: list[tuple[str, str, list[str]]] = [
    ("Core", "-", ["PyQt6.QtWidgets", "cv2", "numpy", "ryvencore"]),
    ("Behavior analysis", "behavior", ["umap", "hdbscan", "sklearn", "yaml"]),
    ("Behavior plots", "behavior", ["lightgbm", "matplotlib"]),
    ("Pose tracking (YOLO)", "vision", ["ultralytics", "torch", "lap"]),
    ("Audio", "audio", ["sounddevice", "soundfile", "pydub"]),
]

_MARKS = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "absent": "[ -- ]"}

_FFMPEG = {
    "win32": 'winget install "FFmpeg (Essentials Build)"',
    "darwin": "brew install ffmpeg",
}


def _not_installed(module: str, error: ModuleNotFoundError) -> bool:
    """True when ``module`` itself is absent, not something it imports.

    ``import ultralytics`` raises ModuleNotFoundError for polars, too, when
    ultralytics is installed and polars is not -- and that is broken, not
    absent.
    """
    name = error.name or ""
    return bool(name) and (module == name or module.startswith(name + "."))


def check_stack(modules: list[str]) -> tuple[str, list[str]]:
    """``("ok" | "absent" | "broken", problems)`` for one stack."""
    problems: list[str] = []
    for i, module in enumerate(modules):
        try:
            importlib.import_module(module)
        except Exception as e:  # ImportError, OSError (a DLL), anything
            if i == 0 and isinstance(e, ModuleNotFoundError) and _not_installed(module, e):
                return "absent", [f"{module} is not installed"]
            problems.append(f"{module}: {e.__class__.__name__}: {e}")
    return ("broken" if problems else "ok"), problems


def _fix(extra: str) -> str:
    if getattr(sys, "frozen", False):
        return "this packaged build is missing it -- please report it"
    if extra == "-":
        return "re-run the GLIDER installer"
    return f"add it with: uv sync --inexact --extra {extra}"


def write_utf8_output() -> None:
    """Make stdout and stderr UTF-8, degrading what they cannot encode.

    A diagnostic must never be the thing that fails: on a cp1252 pipe, a
    username like "张伟" in a path raised UnicodeEncodeError mid-report.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - not every stream can be reconfigured
            pass


def _inference_device() -> str:
    from glider.vision.pose.device import resolve_device

    return resolve_device(None)


def run_doctor() -> int:
    """Print the report. 1 if anything is broken, else 0.

    Warnings, and extras that were never installed, do not fail it.
    """
    from glider._version import __version__

    write_utf8_output()
    # pydub's regexes print SyntaxWarnings the first time a fresh install
    # compiles it -- right above this report, where they read as the problem.
    warnings.simplefilter("ignore", SyntaxWarning)

    failures = 0

    def report(status: str, text: str, *details: str) -> None:
        nonlocal failures
        failures += status == "fail"
        print(f"  {_MARKS[status]} {text}")
        for line in details:
            print(f"           {line}")

    print(f"GLIDER {__version__} doctor")
    print("-" * 60)

    version = sys.version_info[:2]
    python = f"Python {sys.version.split()[0]}"
    if not (3, 11) <= version <= (3, 13):
        report("fail", f"{python} is not supported (3.11-3.13)", _fix("-"))
    elif version != (3, 12):
        report("warn", f"{python} works, but 3.12 is the tested version")
    else:
        report("ok", python)

    if getattr(sys, "frozen", False):
        report("ok", "Packaged build")
    elif sys.prefix != sys.base_prefix:
        report("ok", f"Virtual environment: {sys.prefix}")
    else:
        report(
            "warn",
            f"Not in a virtual environment: {sys.executable}",
            "start GLIDER with `uv run glider` from its folder",
        )

    vision = False
    for label, extra, modules in DEP_STACKS:
        status, problems = check_stack(modules)
        if status == "ok":
            report("ok", label)
        elif status == "absent" and extra != "-":
            report("absent", f"{label}: not installed", _fix(extra))
        else:
            report("fail", label, *problems, _fix(extra))
        vision = vision or (extra == "vision" and status == "ok")

    from glider.core.executables import find_executable

    ffmpeg = find_executable("ffmpeg")
    if ffmpeg:
        report("ok", f"FFmpeg: {ffmpeg}")
    else:
        report(
            "warn",
            "FFmpeg not found: video recording will not work",
            f"install it with: {_FFMPEG.get(sys.platform, 'sudo apt install ffmpeg')}",
        )

    if vision:
        try:
            report("ok", f"Inference device: {_inference_device()}")
        except Exception as e:  # noqa: BLE001 - a diagnostic must not fail on this
            report("warn", f"Inference device unknown: {e.__class__.__name__}: {e}")

    print("-" * 60)
    print(f"{failures} problem(s) found." if failures else "GLIDER is ready to run.")
    return 1 if failures else 0
