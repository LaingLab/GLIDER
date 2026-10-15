"""Experiment presets: .glider files in the user's presets folder, plus built-ins.

A preset is an ordinary session file that also carries ``metadata.run_fields``.
A built-in preset is a protocol and its run fields, defined here rather than
in a file, and applied to whatever experiment is open.
Everything here is pure file/str logic so it can be tested without Qt.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from glider.core.config import get_config
from glider.core.experiment_session import RunField
from glider.core.experiment_types import OPEN_FIELD, PUP_RETRIEVAL

logger = logging.getLogger(__name__)

SUFFIX = ".glider"
_RUN_RE = re.compile(r"_run(\d+)$")


def presets_dir() -> Path:
    return get_config().paths.user_config_dir / "presets"


def slugify(text: str, sep: str = "-") -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", sep, ascii_text.lower()).strip(sep)
    return slug or "preset"


@dataclass(frozen=True)
class PresetInfo:
    path: Path
    name: str
    field_labels: tuple[str, ...]


@dataclass(frozen=True)
class BuiltinPreset:
    """A protocol GLIDER ships, and the fields it asks for before each run."""

    name: str
    fields: tuple[RunField, ...]


BUILTIN_PRESETS: tuple[BuiltinPreset, ...] = (
    BuiltinPreset(
        PUP_RETRIEVAL,
        (
            RunField("dam_id", "Dam ID", True),
            RunField("postnatal_day", "Postnatal day"),
            RunField("litter_size", "Litter size"),
        ),
    ),
    BuiltinPreset(
        OPEN_FIELD, (RunField("animal_id", "Animal ID", True), RunField("group", "Group"))
    ),
)


def apply_builtin(session, name: str) -> None:
    """Make the open experiment a *name* experiment: its protocol and run fields.

    The flow, devices and camera are the experiment's own and stay as they are.
    """
    preset = next((p for p in BUILTIN_PRESETS if p.name == name), None)
    if preset is None:
        raise KeyError(name)
    md = session.metadata
    md.experiment_type = preset.name
    md.run_fields = [replace(f) for f in preset.fields]
    md.reset_run_values()
    session.mark_dirty()


def list_presets(folder: Path | None = None) -> list[PresetInfo]:
    folder = folder or presets_dir()
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob(f"*{SUFFIX}")):
        try:
            md = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
            if not isinstance(md, dict):
                raise ValueError("metadata is not an object")
        except (OSError, ValueError, AttributeError):
            logger.warning("Skipping unreadable preset %s", path)
            continue
        fields = md.get("run_fields", [])
        if not isinstance(fields, list):
            fields = []
        labels = tuple(str(f.get("label", "")) for f in fields if isinstance(f, dict))
        found.append(PresetInfo(path, md.get("experiment_type") or path.stem, labels))
    return sorted(found, key=lambda p: p.name.lower())


def save_preset(session, name: str, folder: Path | None = None) -> Path:
    folder = folder or presets_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stem = slugify(name)
    path = folder / f"{stem}{SUFFIX}"
    n = 2
    while path.exists():
        path = folder / f"{stem}-{n}{SUFFIX}"
        n += 1
    session.metadata.experiment_type = name
    session.save(str(path))
    return path


def next_run_number(base: Path, preset_slug: str, day: date) -> int:
    folder = base / preset_slug
    stamp = f"{day:%Y%m%d}_run"
    highest = 0
    if folder.is_dir():
        for child in folder.iterdir():
            if child.is_dir() and stamp in child.name and (m := _RUN_RE.search(child.name)):
                highest = max(highest, int(m.group(1)))
    return highest + 1


def run_folder(base: Path, preset_slug: str, values: list[str], day: date) -> Path:
    parts = [slugify(v) for v in values if v.strip()]
    n = next_run_number(base, preset_slug, day)
    parts.append(f"{day:%Y%m%d}_run{n:02d}")
    return base / preset_slug / "_".join(parts)


def missing_required(fields, values: dict[str, str]) -> str | None:
    for f in fields:
        if f.required and not values.get(f.key, "").strip():
            return f.label
    return None


def derive_keys(labels: list[str]) -> list[str]:
    keys: list[str] = []
    for label in labels:
        base = slugify(label, sep="_")
        key, n = base, 2
        while key in keys:
            key, n = f"{base}_{n}", n + 1
        keys.append(key)
    return keys


def validate_labels(labels: list[str]) -> str | None:
    seen: set[str] = set()
    for label in labels:
        clean = label.strip().lower()
        if not clean:
            return "Every field needs a label."
        if clean in seen:
            return f"Two fields are both called “{clean}”."
        seen.add(clean)
    return None
