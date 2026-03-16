# GLIDER One-Click Installer for Windows (PowerShell)
# Clones the repo, installs uv, syncs dependencies, launches GLIDER, and creates a desktop shortcut.
$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/LaingLab/glider.git"
$InstallDir = "$env:USERPROFILE\GLIDER"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  GLIDER Installer" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- Install git if missing ---
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Installing git via winget..."
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: git installed but not found. Restart your terminal and re-run." -ForegroundColor Red
        exit 1
    }
}

# --- Install uv if missing ---
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    # Refresh PATH
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:PATH"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: uv installed but not found. Restart your terminal and re-run." -ForegroundColor Red
        exit 1
    }
}

# --- Clone or update repo ---
if (Test-Path "$InstallDir\.git") {
    Write-Host "Updating existing GLIDER installation..."
    Push-Location $InstallDir
    git pull --ff-only 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "Warning: could not fast-forward, using existing version." -ForegroundColor Yellow }
} else {
    Write-Host "Cloning GLIDER..."
    git clone $RepoUrl $InstallDir
    Push-Location $InstallDir
}

# --- Create venv and sync ---
Write-Host "Setting up environment..."
if (-not (Test-Path ".venv")) {
    uv venv
}
uv sync --extra pc

# --- Create desktop shortcut ---
Write-Host "Creating desktop shortcut..."
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = "$DesktopPath\GLIDER.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "$InstallDir\launch.bat"
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.Description = "GLIDER - General Laboratory Interface for Design, Experimentation, and Recording"
$Shortcut.Save()

Pop-Location

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  GLIDER installed successfully!" -ForegroundColor Green
Write-Host "  Location: $InstallDir" -ForegroundColor Green
Write-Host "  Shortcut: Desktop" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

# --- Launch ---
Write-Host "Launching GLIDER..."
Push-Location $InstallDir
uv run glider
Pop-Location
