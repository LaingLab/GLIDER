# Local Windows build: freeze GLIDER with PyInstaller and wrap it in the
# Inno Setup installer. One-command equivalent of release-windows.yml for
# building on your own Windows machine.
#
# Prerequisites:
#   - Python 3.11+ with the project installed:  uv sync --extra pc --extra dev
#   - Inno Setup 6 (https://jrsoftware.org/isdl.php) — ISCC.exe on PATH or at
#     the default install location.
#
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
#
# Output: Output\glider-setup-<version>.exe
#
# The installer ships UNSIGNED — SmartScreen will warn downloaders. See
# README.md in this directory for the code-signing plan.

$ErrorActionPreference = "Stop"

# Repo root, regardless of where the script is invoked from.
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

# Version from the single source of truth (same module the app reads).
$Version = & python -c "import sys; sys.path.insert(0, 'src'); from glider._version import __version__; print(__version__)"
Write-Host ">> Building GLIDER $Version (Windows)"

# Freeze. Prefer `uv run` when uv manages the env; fall back to plain python.
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv run pyinstaller packaging\windows\glider.spec --clean --noconfirm
} else {
    python -m PyInstaller packaging\windows\glider.spec --clean --noconfirm
}

Write-Host ">> Smoke-testing the frozen app"
& "dist\GLIDER\GLIDER.exe" --version
if ($LASTEXITCODE -ne 0) { throw "frozen GLIDER.exe --version failed ($LASTEXITCODE)" }

# Locate ISCC (Inno Setup compiler).
$Iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
if (-not $Iscc) {
    $Default = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if (Test-Path $Default) { $Iscc = $Default }
    else { throw "ISCC.exe not found. Install Inno Setup 6: https://jrsoftware.org/isdl.php" }
}

Write-Host ">> Building installer with Inno Setup"
# /O pins the output dir to the repo root (matching release-windows.yml);
# without it Inno resolves OutputDir relative to the .iss file.
& $Iscc "/DMyAppVersion=$Version" "/O$RepoRoot\Output" packaging\windows\installer.iss
if ($LASTEXITCODE -ne 0) { throw "ISCC exited with $LASTEXITCODE" }

$Installer = Get-ChildItem "Output\glider-setup-*.exe" | Select-Object -First 1
Write-Host ">> Done: $($Installer.FullName) ($([math]::Round($Installer.Length / 1MB)) MB)"
