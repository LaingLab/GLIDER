# PyInstaller spec for the Windows build.
#
# One-folder (not one-file) because PyQt6 onefile has a slow cold-start — every
# launch unpacks the 300MB bundle to a temp dir — and occasionally races with
# OpenCV's DLL loader. One-folder is also easier to code-sign (we sign the
# produced executable directly rather than the self-extracting onefile stub).
#
# Build locally with:
#     pyinstaller packaging/windows/glider.spec --clean --noconfirm
# Output: dist/GLIDER/GLIDER.exe and the full runtime folder beside it.
#
# NOTE: this file is evaluated as Python — the build system imports it and
# executes it. Keep side effects minimal.

# ruff: noqa  -- PyInstaller spec files are not normal Python modules

import os

from PyInstaller.utils.hooks import (
    collect_all,
    collect_dynamic_libs,
    collect_submodules,
)

block_cipher = None

# --- Dependency collection ---------------------------------------------------

# Our own package. collect_all includes data files (schemas, stylesheets, icons)
# declared in pyproject.toml's package-data.
glider_datas, glider_binaries, glider_hiddenimports = collect_all("glider")

# Optional stacks the app imports lazily, inside the functions that need them.
# PyInstaller's static analysis walks imports from __main__ outward, so a
# module that is only ever imported inside a function body is invisible to it
# and silently absent from the bundle -- which is how hdbscan went missing
# while its own dependencies (numba, llvmlite, pynndescent) rode along via
# other packages. The failure is quiet: behavior_available() finds one module
# short and the Behavior Analysis menu disables itself with no error anywhere.
#
# collect_all rather than a bare hiddenimport because these ship more than
# Python: hdbscan is Cython (.pyd), and ultralytics loads its model and
# tracker configs from .yaml files at runtime.
_LAZY_STACKS = ["hdbscan", "umap", "ultralytics", "lap"]

lazy_datas, lazy_binaries, lazy_hiddenimports = [], [], []
for _pkg in _LAZY_STACKS:
    _d, _b, _h = collect_all(_pkg)
    lazy_datas += _d
    lazy_binaries += _b
    lazy_hiddenimports += _h

# ryvencore is the node-graph engine; its plugins are discovered by name so
# PyInstaller's static analysis doesn't find them.
ryvencore_hiddenimports = collect_submodules("ryvencore")

# qasync bridges Qt and asyncio; same dynamic-discovery problem.
qasync_hiddenimports = collect_submodules("qasync")

# OpenCV ships a pile of secondary DLLs (opencv_videoio_ffmpeg*, opencv_world*,
# etc.) that the upstream PyInstaller hook occasionally misses. Collecting
# dynamic libs explicitly turns a silent `ImportError: DLL load failed while
# importing cv2` into a predictable, reproducible binary on disk.
cv2_binaries = collect_dynamic_libs("cv2")

hiddenimports = (
    glider_hiddenimports
    + ryvencore_hiddenimports
    + qasync_hiddenimports
    + lazy_hiddenimports
    + [
        # Driver entry points are registered in pyproject.toml via the
        # "glider.driver" group. At runtime they're looked up by string, so
        # PyInstaller's import graph doesn't see them. List them explicitly.
        "glider.hal.boards.telemetrix_board",
        # Telemetrix-aio's submodules are dynamically imported by device type.
        "telemetrix_aio",
        "telemetrix_aio.telemetrix_aio",
    ]
)

# --- Exclusions --------------------------------------------------------------

excludes = [
    # Pi-only drivers — gpiozero/lgpio don't have Windows wheels and should
    # never be part of a Windows build.
    "glider.hal.boards.pi_gpio_board",
    "gpiozero",
    "lgpio",
    # ultralytics and torch are now BUNDLED, deliberately.
    #
    # They used to be excluded to keep the installer permissively licensed,
    # on the assumption that yolo_install.py would fetch ultralytics on first
    # use. It cannot: can_auto_install() returns `not sys.frozen`, so in a
    # packaged build that path is always off. The exclusion did not defer the
    # download, it removed pose tracking from the installer entirely.
    #
    # The cost is real and was accepted knowingly: ultralytics is AGPL-3.0, so
    # the distributed application is a combined AGPL work and recipients are
    # entitled to source. Ultralytics sells a commercial license if that
    # becomes a problem. torch is the CUDA build (~2.9 GB unpacked) because
    # CPU-only inference is too slow for hour-long recordings.
    #
    # To go back to a permissive, small installer, re-add "ultralytics",
    # "torch" and "torchvision" here AND give frozen builds a real way to
    # obtain them -- an add-on pack, or a bundled pip -- or tracking silently
    # disappears again.
    # Test + dev tooling should never ride along.
    "pytest",
    "pytest_asyncio",
    "pytest_qt",
    "black",
    "ruff",
]

# --- Analysis / build tree ---------------------------------------------------

a = Analysis(
    ["../../src/glider/__main__.py"],
    pathex=["../../src"],
    binaries=glider_binaries + cv2_binaries + lazy_binaries,
    datas=glider_datas + lazy_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    # Runs before the entry script -- see rthook_torch.py. __main__.py's own
    # torch-before-Qt guard is too late in a frozen build.
    #
    # Absolute via SPECPATH: runtime_hooks are resolved against the current
    # working directory, not this file's directory (unlike the entry script
    # above), so a relative path here builds only when pyinstaller happens to
    # be invoked from packaging/windows.
    runtime_hooks=[os.path.join(SPECPATH, "rthook_torch.py")],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GLIDER",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX tends to trip AV heuristics; not worth it.
    console=False,             # GUI app — no console window.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # The project's canonical icon lives at packaging/icons/glider.ico.
    # Relative to this spec file (which PyInstaller resolves from its own
    # directory), that's one level up then into icons/.
    icon="../icons/glider.ico",
    # File-properties metadata (File version / Product version / Copyright in
    # the Explorer "Details" tab) is intentionally omitted for v1 — PyInstaller
    # accepts a `version=` kwarg pointing at a generated resource file, but
    # we haven't wired the generator yet. The .exe works fine without it; add
    # a `version_info.txt` generator step when we want branded metadata.
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GLIDER",
)
