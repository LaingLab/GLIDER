"""Corners come from one scale, and stay there.

The app had grown fifteen distinct radii across 145 declarations - 2, 3, 5, 7,
9, 13 - not because anything needed them but because each was picked next to
whatever was on screen at the time. Corners that nearly match read as sloppier
than corners that plainly differ.

Snapping them once fixes today. This is what stops it coming back: a new
``border-radius: 7px`` fails here rather than surviving review because nobody
had the other 144 in their head.

Circles and pills are the deliberate exception. A radio indicator is 18px
across with a 9px radius because that is what makes it round, so those keep
their own value and say so.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from glider.gui.styles import STYLES_DIR, load_stylesheet, radius

#: Every stylesheet the app ships.
SHEETS = ("desktop", "tools", "touch")

#: A radius declaration and whatever trails it on the line.
_DECLARATION = re.compile(r"border-radius:\s*([^;]+);(.*)")

#: How a deliberate exception is marked, in QSS and in Python alike.
_GEOMETRIC = "geometric"

#: Tokens a stylesheet may use for a stylistic radius.
_TOKENS = {"@RADIUS_SMALL@", "@RADIUS_MEDIUM@", "@RADIUS_LARGE@"}

#: What any substitution token looks like, matching the loader.
_TOKEN_SHAPE = re.compile(r"@[A-Z_]+@")

#: The source tree: src/. Python files draw their own chrome with inline
#: stylesheets, and those must come from the same scale, so the whole app
#: moves together when it is retuned.
_SOURCE = STYLES_DIR.parents[2]


def _qss_declarations(sheet: str) -> list[tuple[int, str, str]]:
    """``(line number, value, trailing text)`` for each radius in the raw file."""
    path = STYLES_DIR / f"{sheet}.qss"
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _DECLARATION.search(line)
        if match:
            found.append((number, match.group(1).strip(), match.group(2)))
    return found


def _python_declarations() -> list[tuple[Path, int, str, str]]:
    found = []
    for path in sorted(_SOURCE.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = _DECLARATION.search(line)
            if match:
                found.append((path, number, match.group(1).strip(), match.group(2)))
    return found


class TestStylesheets:
    @pytest.mark.parametrize("sheet", SHEETS)
    def test_every_stylistic_radius_is_a_token(self, sheet):
        offenders = [
            f"{sheet}.qss:{number}: border-radius: {value}"
            for number, value, trailing in _qss_declarations(sheet)
            if value not in _TOKENS and value != "0" and _GEOMETRIC not in trailing
        ]
        assert not offenders, (
            "these radii are neither a scale token nor marked geometric:\n  "
            + "\n  ".join(offenders)
            + "\n\nUse @RADIUS_SMALL@ (controls), @RADIUS_MEDIUM@ (containers) or "
            "@RADIUS_LARGE@ (large surfaces). If the radius has to follow the "
            "element's size -- a circle or a pill -- mark the line /* geometric */."
        )

    @pytest.mark.parametrize("sheet", SHEETS)
    def test_a_geometric_radius_is_a_plain_number(self, sheet):
        # A token on a line marked geometric means someone marked it to quiet
        # the check rather than because the radius follows a size.
        offenders = [
            f"{sheet}.qss:{number}"
            for number, value, trailing in _qss_declarations(sheet)
            if _GEOMETRIC in trailing and value in _TOKENS
        ]
        assert not offenders, f"marked geometric but using a scale token: {offenders}"

    @pytest.mark.parametrize("sheet", SHEETS)
    def test_loading_resolves_every_token(self, sheet):
        # Qt discards a whole rule containing something it cannot parse, and
        # says nothing, so a token that survives loading removes styling that
        # nothing then reports as missing. Matched by shape rather than by a
        # bare "@", which appears legitimately in comments like /* ACCENT @ 13% */.
        assert not _TOKEN_SHAPE.findall(load_stylesheet(sheet))

    def test_a_missed_token_warns_rather_than_passing_silently(self, tmp_path, monkeypatch):
        import glider.gui.styles as styles

        monkeypatch.setattr(styles, "STYLES_DIR", tmp_path)
        (tmp_path / "made_up.qss").write_text("QWidget { border-radius: @RADIUS_HUGE@; }")
        with pytest.warns(UserWarning, match="RADIUS_HUGE"):
            styles.load_stylesheet("made_up")


class TestInlineStyles:
    def test_widgets_drawing_their_own_chrome_use_the_scale(self):
        allowed = {f"{{radius.{name}}}px" for name in ("NONE", "SMALL", "MEDIUM", "LARGE")}
        allowed |= {"0", "{CHIP_RADIUS}px"}
        offenders = [
            f"{path.relative_to(_SOURCE)}:{number}: border-radius: {value}"
            for path, number, value, trailing in _python_declarations()
            if value not in allowed and _GEOMETRIC not in trailing and "radius.pill(" not in value
        ]
        assert not offenders, (
            "inline stylesheets must take their radius from glider.gui.styles.radius:\n  "
            + "\n  ".join(offenders)
        )


class TestOneLoader:
    """Nothing reads a stylesheet except ``load_stylesheet``.

    Three places did: ``ViewManager.apply_stylesheet``, and two test fixtures.
    They worked only because the sheets happened to hold no tokens. The moment
    one did, Qt refused to parse the sheet, dropped *all* of it, and the
    failure surfaced as a widget being the wrong colour - several steps from
    the cause, and only because a test happened to assert on that colour.
    """

    def test_no_one_reads_a_qss_file_directly(self):
        # Naming a stylesheet path is fine -- ViewManager picks one by mode and
        # logs it. Reading one is what bypasses the loader, so that is what is
        # checked: a .qss path read on the spot, or bound to a name and read
        # through it, which is the shape both fixtures had.
        inline = re.compile(r"\.qss[\"']\s*\)?\s*\.(?:read_text|open)\(")
        bound = re.compile(r"^\s*(\w+)\s*=.*\.qss[\"']")
        offenders = []
        for root in (_SOURCE, _SOURCE.parent / "tests"):
            for path in sorted(root.rglob("*.py")):
                if path.parent == STYLES_DIR or path == Path(__file__):
                    continue  # the loader, and this test's own fixtures
                lines = path.read_text(encoding="utf-8").splitlines()
                names = {m.group(1) for line in lines if (m := bound.match(line))}
                reads = {f"{name}.read_text(" for name in names} | {
                    f"open({name}" for name in names
                }
                for number, line in enumerate(lines, start=1):
                    if inline.search(line) or any(r in line for r in reads):
                        offenders.append(
                            f"{path.relative_to(root.parent)}:{number}: {line.strip()}"
                        )
        assert not offenders, (
            "read a stylesheet only through load_stylesheet, which resolves its "
            "tokens -- Qt silently discards a whole sheet it cannot parse:\n  "
            + "\n  ".join(offenders)
        )


class TestScale:
    def test_the_steps_are_distinct_and_ordered(self):
        assert radius.NONE < radius.SMALL < radius.MEDIUM < radius.LARGE

    def test_pill_halves_its_height(self):
        assert radius.pill(18) == 9
        assert radius.pill(8) == 4

    def test_pill_refuses_a_negative_height(self):
        with pytest.raises(ValueError):
            radius.pill(-1)

    def test_the_scale_is_what_the_tokens_resolve_to(self):
        sheet = load_stylesheet("desktop")
        # Every token maps to its constant, so retuning the scale retunes the
        # stylesheets rather than only the Python side.
        assert f"{radius.SMALL}px" in sheet
        assert f"{radius.MEDIUM}px" in sheet
        assert f"{radius.LARGE}px" in sheet
