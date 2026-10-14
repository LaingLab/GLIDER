# Install GLIDER from source on Windows, or update an install this made.
# (macOS, Linux and Raspberry Pi: install.sh.)
#
#   irm https://raw.githubusercontent.com/LaingLab/glider/main/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/LaingLab/glider/main/install.ps1))) -Extra vision,behavior
#
# It goes in ~\glider unless $env:GLIDER_DIR says otherwise. Run it again to
# update; pass every -Extra you want each time, since extras not named are removed.
param([string[]]$Extra = @())

# Everything happens inside a function: `irm | iex` runs a script in the
# caller's own session, and the settings below must not leak into it.
function Install-Glider([string[]]$Extra) {
    $ErrorActionPreference = "Stop"
    $repo = "https://github.com/LaingLab/glider.git"
    $dir = if ($env:GLIDER_DIR) { $env:GLIDER_DIR } else { Join-Path $HOME "glider" }

    function Say($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
    function Has($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }
    # Windows PowerShell does not stop on a failing program, only on a failing cmdlet.
    function Run {
        $exe, $rest = $args
        & $exe @rest
        if ($LASTEXITCODE -ne 0) { throw "'$exe $rest' failed (exit code $LASTEXITCODE)" }
    }
    # An installer edits PATH in the registry; this session still has the old one.
    function Update-Path {
        $env:Path = $env:Path + ";" + [Environment]::GetEnvironmentVariable("Path", "User") +
                    ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
    }

    if (-not (Has uv)) {
        Say "Installing uv"
        # In its own process: it is a script of its own, and an `exit` in it
        # must not close this window.
        Run powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
        Update-Path
    }

    if (-not (Has git)) {
        if (-not (Has winget)) {
            throw "Install Git from https://git-scm.com/download/win, then run this again."
        }
        Say "Installing git"
        Run winget install --id Git.Git --exact --source winget --accept-package-agreements --accept-source-agreements
        Update-Path
    }

    if (Test-Path (Join-Path $dir ".git")) {
        Say "Updating $dir"
        Run git -C $dir pull --ff-only
    } elseif (Test-Path $dir) {
        throw "$dir already exists and is not a GLIDER checkout. Set `$env:GLIDER_DIR to install somewhere else."
    } else {
        Say "Downloading GLIDER into $dir"
        Run git clone $repo $dir
    }

    Say "Installing GLIDER's Python packages"
    $sync = @("sync", "--directory", $dir, "--extra", "pc")
    foreach ($name in $Extra) { $sync += @("--extra", $name) }
    Run uv @sync

    Say "Checking the install"
    & uv run --directory $dir glider doctor
    if ($LASTEXITCODE -eq 0) {
        Say "Done. Start GLIDER with:"
    } else {
        Say "Installed, but doctor found problems (above). Once they are fixed, start GLIDER with:"
    }
    Write-Host "    cd $dir; uv run glider"
}

Install-Glider $Extra
