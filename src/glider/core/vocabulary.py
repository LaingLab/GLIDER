"""
Lab vocabulary: the treatment terms a lab actually uses, defined once.

Subjects carry treatment metadata -- group, strain, solution, dose, route.
Left as free text, ``Control``, ``control`` and ``  CONTROL  `` become three
different treatment groups and no downstream analysis can tell that they were
meant to be one. This module is the store that prevents that: a single JSON
file beside the device library (``~/.glider/library/vocabulary.json``) holding
the five lists a subject form offers as choices.

De-duplication is case- and whitespace-insensitive, and the **first** spelling
a lab uses is the one kept, so the lab's own capitalisation survives.

Loading is forgiving by design, mirroring :mod:`glider.core.device_library`: a
malformed file is logged and skipped, never raised. A broken vocabulary must
not stop the application starting.

This module imports no Qt, so it can be used and tested headlessly.
"""

import json
import logging
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

VOCABULARY_FILENAME = "vocabulary.json"
SCHEMA_VERSION = "1.0"

#: The five lists a vocabulary holds, in the order a setup form shows them.
LISTS = ("groups", "strains", "solutions", "routes", "sexes")

# Defaults are duplicated from SEX_OPTIONS/ROUTE_OPTIONS in
# glider.gui.dialogs.subject_dialog rather than imported, because importing
# that module would drag Qt into this one. NOTHING PINS THE TWO TOGETHER YET:
# Task 2 of the lab-vocabulary plan adds the test that asserts they match, and
# until it lands, editing either copy can silently drift from the other.
# The leading "" of those constants is a combo-box affordance meaning
# "nothing chosen", not a vocabulary value, so it is deliberately absent here.
DEFAULT_SEXES = ["Male", "Female", "Unknown"]
DEFAULT_ROUTES = ["IP", "IV", "PO", "SC", "IM", "Topical", "Inhalation", "Other"]


def _key(value: str) -> str:
    """The comparison key for a vocabulary value: normalized, trimmed, folded.

    NFKC normalization matters as much as case folding here. macOS favours
    decomposed forms and Windows composed ones, so the same strain typed on
    two machines in one lab arrives as two different strings that render
    identically -- ``Bre`` + combining acute versus a precomposed ``é``.
    Without normalizing, that splits one cohort in two exactly as
    ``Control``/``control`` would, just through a different door.
    """
    return unicodedata.normalize("NFKC", value).strip().casefold()


@dataclass
class Vocabulary:
    """The terms one lab uses. Groups, strains and solutions start empty."""

    groups: list[str] = field(default_factory=list)
    strains: list[str] = field(default_factory=list)
    solutions: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=lambda: list(DEFAULT_ROUTES))
    sexes: list[str] = field(default_factory=lambda: list(DEFAULT_SEXES))

    def get(self, name: str) -> list[str]:
        """The live list called ``name``. Raises ``KeyError`` if there is none.

        This is the stored list, not a copy. Mutate it only through ``add``
        and ``remove`` -- appending to it directly bypasses the de-duplication
        that is the entire purpose of this module.
        """
        if name not in LISTS:
            raise KeyError(f"Unknown vocabulary list: {name!r}")
        return getattr(self, name)

    def add(self, name: str, value: str) -> bool:
        """Add ``value`` unless an equivalent spelling is already present.

        Returns whether anything changed, so callers can skip re-saving the
        file when nothing was learned.
        """
        values = self.get(name)
        cleaned = (value or "").strip()
        if not cleaned:
            return False
        key = _key(cleaned)
        if any(_key(existing) == key for existing in values):
            return False
        values.append(cleaned)
        return True

    def remove(self, name: str, value: str) -> bool:
        """Remove the entry equivalent to ``value``. Returns whether one went."""
        values = self.get(name)
        key = _key(value or "")
        for index, existing in enumerate(values):
            if _key(existing) == key:
                del values[index]
                return True
        return False

    def to_dict(self) -> dict:
        """A JSON-serialisable snapshot, including the schema version."""
        data: dict = {"schema_version": SCHEMA_VERSION}
        for name in LISTS:
            data[name] = list(self.get(name))
        return data


def vocabulary_path(library_dir: Path) -> Path:
    return Path(library_dir) / VOCABULARY_FILENAME


def load(library_dir: Path) -> Vocabulary:
    """Read the vocabulary from ``library_dir``, falling back to defaults.

    A missing, unreadable, malformed or structurally wrong file yields the
    defaults with a logged warning; it never raises.
    """
    path = vocabulary_path(library_dir)
    if not path.exists():
        return Vocabulary()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning("Ignoring invalid vocabulary file %s: %s", path, e)
        return Vocabulary()

    if not isinstance(data, dict):
        logger.warning("Ignoring vocabulary file %s: expected a JSON object", path)
        return Vocabulary()

    stored_version = data.get("schema_version")
    if stored_version is not None and stored_version != SCHEMA_VERSION:
        # Never fatal -- a future v2 file must still yield a usable vocabulary
        # rather than an empty subject form. The warning leaves the trace.
        logger.warning(
            "Vocabulary file %s declares schema_version %r, expected %r; reading it anyway",
            path,
            stored_version,
            SCHEMA_VERSION,
        )

    vocab = Vocabulary()
    for name in LISTS:
        stored = data.get(name)
        if not isinstance(stored, list):
            continue  # Absent or wrong-typed: keep this list's default.
        vocab.get(name).clear()
        for item in stored:
            if isinstance(item, str):
                vocab.add(name, item)
    return vocab


def save(vocab: Vocabulary, library_dir: Path) -> bool:
    """Write the vocabulary to ``library_dir``. Returns whether it was written.

    A read-only home directory must not take the application down, so an
    ``OSError`` is logged and reported as ``False`` for the caller to surface.

    The write goes to a temporary file and is then moved into place, so an
    interrupted save leaves the previous vocabulary intact. This file holds
    the lab's whole vocabulary and, once subjects learn novel values on save,
    is rewritten on every subject save -- a truncating write would put the
    entire vocabulary at risk on each one.
    """
    path = vocabulary_path(library_dir)
    tmp = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(vocab.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("Could not save vocabulary to %s: %s", path, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    logger.debug("Saved vocabulary to %s", path)
    return True
