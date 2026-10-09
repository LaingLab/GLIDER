"""The one write rule every MCP tool shares."""

from pathlib import Path


def writable_path(path: str, suffix: str, overwrite: bool = False) -> Path:
    """Where a tool may write, or a ValueError telling the agent what to fix.

    Absolute only, so a relative path cannot land wherever the client happened
    to launch the server; one suffix per tool, so a tool can never overwrite a
    recording's CSV with a PNG or the like; and no silent replacement.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        raise ValueError(f"{path} is not an absolute path; pass the full path")
    if p.suffix.lower() != suffix:
        raise ValueError(f"{p.name} must end in {suffix}")
    if p.exists() and not overwrite:
        raise ValueError(
            f"{p} already exists; choose a new name"
            + (" or pass overwrite=true to replace it" if suffix == ".glider" else "")
        )
    if not p.parent.is_dir():
        raise ValueError(f"folder {p.parent} does not exist")
    return p
