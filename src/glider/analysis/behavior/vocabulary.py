"""Behavior vocabulary: name ↔ hotkey ↔ color.

The vocabulary is set up before annotation begins. It maps a single-
character hotkey (e.g. ``"1"``) to a behavior name (``"rearing"``) and
assigns each behavior a stable display color.

Loading from disk
-----------------

A YAML or JSON file with a top-level ``behaviors`` list. Each entry is
a mapping with ``name``, ``hotkey``, and an optional ``color``::

    # behaviors.yaml
    behaviors:
      - name: rearing
        hotkey: "1"
        color: "#1d4ed8"
      - name: grooming
        hotkey: "2"
      - name: locomote
        hotkey: "3"

Colors omitted from the file are assigned from a built-in palette in
the order behaviors appear. Hotkeys must be unique; the loader raises
:class:`VocabularyError` if any conflict.

This module has no Qt dependency.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


class VocabularyError(ValueError):
    """Raised when a vocabulary violates an invariant (duplicate hotkey, etc.)."""


# Twelve-hue categorical palette. Same set used elsewhere in the project
# (the comparison docs + the deleted PyQt6 UI). Light-mode friendly.
DEFAULT_PALETTE: tuple[str, ...] = (
    "#1d4ed8",  # blue
    "#047857",  # emerald
    "#b45309",  # amber
    "#7c3aed",  # violet
    "#be185d",  # rose
    "#0f766e",  # teal
    "#a16207",  # yellow
    "#c2410c",  # orange
    "#4338ca",  # indigo
    "#15803d",  # green
    "#9d174d",  # pink
    "#b91c1c",  # red
)


@dataclass
class Behavior:
    """One entry in the vocabulary."""

    name: str
    hotkey: str
    color: str = ""
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise VocabularyError("behavior name must be non-empty")
        if not self.hotkey or len(self.hotkey) != 1:
            raise VocabularyError(f"hotkey must be a single character; got {self.hotkey!r}")
        if self.color and not _is_valid_hex(self.color):
            raise VocabularyError(f"color must be a hex string like '#1d4ed8'; got {self.color!r}")
        self.tags = frozenset(str(t) for t in self.tags)


class Vocabulary:
    """Ordered collection of :class:`Behavior` entries with hotkey index.

    Order matters: it determines the default color when none is set, and
    the display order in the UI.
    """

    def __init__(self, behaviors: Iterable[Behavior] | None = None):
        self._behaviors: list[Behavior] = []
        self._by_hotkey: dict[str, Behavior] = {}
        self._by_name: dict[str, Behavior] = {}
        if behaviors:
            for b in behaviors:
                self.add(b)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add(self, behavior: Behavior) -> None:
        """Append a behavior. Auto-assigns a default color if none was set."""
        if behavior.hotkey in self._by_hotkey:
            raise VocabularyError(
                f"hotkey {behavior.hotkey!r} already maps to "
                f"{self._by_hotkey[behavior.hotkey].name!r}"
            )
        if behavior.name in self._by_name:
            raise VocabularyError(f"behavior name {behavior.name!r} is already in the vocabulary")
        if not behavior.color:
            behavior.color = DEFAULT_PALETTE[len(self._behaviors) % len(DEFAULT_PALETTE)]
        self._behaviors.append(behavior)
        self._by_hotkey[behavior.hotkey] = behavior
        self._by_name[behavior.name] = behavior

    def remove(self, name: str) -> bool:
        """Remove a behavior by name. Returns True if anything was removed.

        Existing zones referencing this behavior are NOT cleaned up — the
        caller (UI / training pipeline) decides what to do with orphans.
        """
        b = self._by_name.pop(name, None)
        if b is None:
            return False
        self._by_hotkey.pop(b.hotkey, None)
        self._behaviors.remove(b)
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Behavior]:
        return iter(self._behaviors)

    def __len__(self) -> int:
        return len(self._behaviors)

    def __bool__(self) -> bool:
        return bool(self._behaviors)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def behaviors(self) -> list[Behavior]:
        return list(self._behaviors)

    def names(self) -> list[str]:
        return [b.name for b in self._behaviors]

    def hotkeys(self) -> list[str]:
        return [b.hotkey for b in self._behaviors]

    def behavior_for_hotkey(self, hotkey: str) -> Behavior | None:
        return self._by_hotkey.get(hotkey)

    def behavior_for_name(self, name: str) -> Behavior | None:
        return self._by_name.get(name)

    def color_for(self, name: str) -> str:
        b = self._by_name.get(name)
        return b.color if b else "#94a3b8"  # slate fallback

    def tag_map(self) -> dict[str, frozenset[str]]:
        """{behavior name: tag set} — the map the hybrid prior consumes."""
        return {b.name: b.tags for b in self._behaviors}

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "behaviors": [
                {"name": b.name, "hotkey": b.hotkey, "color": b.color, "tags": sorted(b.tags)}
                for b in self._behaviors
            ]
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Vocabulary:
        if "behaviors" not in payload:
            raise VocabularyError("expected a top-level 'behaviors' key in the vocabulary file")
        entries = payload["behaviors"]
        if not isinstance(entries, list):
            raise VocabularyError("'behaviors' must be a list")
        v = cls()
        for entry in entries:
            if not isinstance(entry, dict):
                raise VocabularyError(f"each behavior must be a mapping; got {entry!r}")
            v.add(
                Behavior(
                    name=str(entry["name"]),
                    hotkey=str(entry["hotkey"]),
                    color=str(entry.get("color", "")),
                    tags=set(entry.get("tags", [])),
                )
            )
        return v

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        """Load from a YAML (.yaml/.yml) or JSON (.json) file."""
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:  # pragma: no cover - optional dep
                raise VocabularyError(
                    "PyYAML is required to load .yaml vocabularies; "
                    "use a .json file or install yaml"
                ) from e
            payload = yaml.safe_load(text) or {}
        else:
            payload = json.loads(text)
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> Path:
        """Write to a YAML (.yaml/.yml) or JSON (.json) file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            path.write_text(yaml.safe_dump(payload, sort_keys=False))
        else:
            path.write_text(json.dumps(payload, indent=2))
        return path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _is_valid_hex(color: str) -> bool:
    """``#rgb`` or ``#rrggbb`` only — same convention as CSS."""
    if not color.startswith("#"):
        return False
    body = color[1:]
    if len(body) not in (3, 6):
        return False
    return all(c in "0123456789abcdefABCDEF" for c in body)
