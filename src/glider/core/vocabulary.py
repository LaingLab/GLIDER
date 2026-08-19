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
import re
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
# that module would drag Qt into this one. The two copies are pinned together
# by test_the_dialog_options_match_the_vocabulary_defaults in
# tests/unit/gui/test_subject_dialog_vocabulary.py -- edit one and that test
# fails. The leading "" of those constants is a combo-box affordance meaning
# "nothing chosen", not a vocabulary value, so it is deliberately absent here.
#
# The dialog no longer reads SEX_OPTIONS/ROUTE_OPTIONS: every one of its five
# fields is populated from a Vocabulary instead. They survive only as the other
# half of that pin, which is what keeps these defaults honest about what a
# subject form used to offer before any lab had defined a vocabulary.
DEFAULT_SEXES = ["Male", "Female", "Unknown"]
DEFAULT_ROUTES = ["IP", "IV", "PO", "SC", "IM", "Topical", "Inhalation", "Other"]


#: Zero-width and bidi-control characters: U+200B..U+200F and U+FEFF. None of
#: them is whitespace, so ``str.strip()`` leaves them alone and NFKC keeps
#: them. Without removing them a pasted ``Control`` carrying a stray U+200B
#: sits beside a typed ``Control`` as a second, pixel-identical entry -- the
#: duplicate-cohort failure this module exists to prevent, arriving through a
#: character nobody can see to delete. They are the commonest invisible
#: passenger on a paste out of Excel, a PDF or a web page, which is how a lab
#: vocabulary actually gets filled in.
_INVISIBLE = re.compile("[\u200b-\u200f\ufeff]")


def _clean(value: str) -> str:
    """The value as it should be stored: no invisible characters, trimmed."""
    return _INVISIBLE.sub("", value or "").strip()


def _key(value: str) -> str:
    """The comparison key for a vocabulary value: cleaned, normalized, folded.

    NFKC normalization matters as much as case folding here. macOS favours
    decomposed forms and Windows composed ones, so the same strain typed on
    two machines in one lab arrives as two different strings that render
    identically -- ``Bre`` + combining acute versus a precomposed ``é``.
    Without normalizing, that splits one cohort in two exactly as
    ``Control``/``control`` would, just through a different door.
    """
    return unicodedata.normalize("NFKC", _clean(value)).casefold()


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
        cleaned = _clean(value)
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
        if stored is None:
            continue  # Absent: keep this list's default, as documented.
        if not isinstance(stored, list):
            # Wrong-typed: the list silently reverts to its defaults, which
            # from the user's side looks like the terms they hand-edited in
            # simply vanishing. Say so; the docs promise a warning.
            logger.warning(
                "Vocabulary file %s: %r is %s, not a list; using the defaults for it",
                path,
                name,
                type(stored).__name__,
            )
            continue
        vocab.get(name).clear()
        dropped = 0
        for item in stored:
            if isinstance(item, str):
                vocab.add(name, item)
            else:
                dropped += 1
        if dropped:
            logger.warning(
                "Vocabulary file %s: dropped %d non-text entr%s from %r",
                path,
                dropped,
                "y" if dropped == 1 else "ies",
                name,
            )
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
