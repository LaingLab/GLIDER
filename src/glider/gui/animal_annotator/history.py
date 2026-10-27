"""Undo and redo for labelling, across frames.

One entry per user action -- a click, a finished drag, a tilt, a deleted
point, a Skip toggle -- holding the frame it happened on and the state
before and after. Undo restores *before* (on that frame, going back to it if
need be); redo restores *after*. Qt-free, so it is tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Edit", "History"]

#: Kinds of edit: a frame's labels, or its skipped flag.
LABELS = "labels"
SKIP = "skip"


@dataclass(frozen=True)
class Edit:
    frame: int  # index into project.frames
    kind: str  # LABELS or SKIP
    before: Any
    after: Any


class History:
    """A bounded undo stack with redo. A new edit clears what could be redone."""

    def __init__(self, limit: int = 500):
        self._limit = limit
        self._undo: list[Edit] = []
        self._redo: list[Edit] = []

    def push(self, edit: Edit) -> None:
        self._undo.append(edit)
        if len(self._undo) > self._limit:
            del self._undo[0]
        self._redo.clear()

    def undo(self) -> Edit | None:
        if not self._undo:
            return None
        edit = self._undo.pop()
        self._redo.append(edit)
        return edit

    def redo(self) -> Edit | None:
        if not self._redo:
            return None
        edit = self._redo.pop()
        self._undo.append(edit)
        return edit

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
