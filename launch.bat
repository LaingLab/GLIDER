@echo off
REM GLIDER Launcher for Windows
REM Installs uv if needed, creates venv, syncs dependencies, and launches GLIDER.
setlocal

cd /d "%~dp0"

REM --- Install uv if not found ---
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing uv...
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"

    REM Refresh PATH
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

    where uv >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: uv installed but not found on PATH.
        echo Restart your terminal and run this script again.
        pause
        exit /b 1
    )
    echo uv installed successfully.
)

REM --- Create venv and sync dependencies ---
if not exist ".venv" (
    echo Creating virtual environment...
    uv venv
)

echo Syncing dependencies...
uv sync --extra pc

REM --- Launch GLIDER ---
echo Launching GLIDER...
uv run glider %*
