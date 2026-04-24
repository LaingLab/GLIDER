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
    # ultralytics is AGPL-3.0. We ship it as a lazy first-run download so the
    # base installer stays permissively-licensed. If a user opts in, the app
    # will pip-install it into the bundled venv at runtime.
    "ultralytics",
    "torch",
    "torchvision",
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
    binaries=glider_binaries + cv2_binaries,
    datas=glider_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
