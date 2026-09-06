"""
GLIDER Styles - Qt Style Sheets for desktop and touch modes.
"""

import re
import warnings
from pathlib import Path

from PyQt6.QtWidgets import QWidget

from glider.gui.styles import (
    colors,  # noqa: F401
    radius,
)

STYLES_DIR = Path(__file__).parent


def restyle(widget: QWidget) -> None:
    """Make Qt re-evaluate *widget* after a dynamic property changed.

    Qt resolves ``[state="..."]`` and ``[role="..."]`` selectors at *polish*
    time, not when the property is set, so a rule that ought to fire once the
    property moves never takes effect without this. The symptom is a widget
    whose property is provably correct and whose colour never changed --
    plausible enough to survive a code review and invisible to a test that only
    reads the property back.

    This lives in the styles package because four widgets had grown their own
    private copy of it (``plugin_card``, ``tool_ui``, ``side_panel``,
    ``status_strip``) and a fifth was about to. Three of the four were
    identical; ``tool_ui``'s also called ``update()``, which is the superset and
    is what is kept here -- a repaint after a restyle is what the caller wanted
    in every case.
    """
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()


#: Pattern any substitution token matches, used to catch one that was missed.
_TOKEN_RE = re.compile(r"@[A-Z_]+@")


def _tokens() -> dict[str, str]:
    """Substitutions applied to every stylesheet as it is loaded.

    ``@ICONS@`` becomes the absolute path of ``styles/icons``: Qt resolves
    ``url()`` against the *application's* working directory rather than the
    stylesheet, and has no data-URI support, so a token is the only way to
    reference the arrow assets from a package that may be installed anywhere.
    Posix separators because QSS ``url()`` rejects Windows backslashes.

    ``@RADIUS_*@`` come from :mod:`glider.gui.styles.radius`, so the corner
    scale has one definition that both the stylesheets and the widgets drawing
    their own chrome read.
    """
    return {
        "@ICONS@": (STYLES_DIR / "icons").as_posix(),
        "@RADIUS_SMALL@": f"{radius.SMALL}px",
        "@RADIUS_MEDIUM@": f"{radius.MEDIUM}px",
        "@RADIUS_LARGE@": f"{radius.LARGE}px",
    }


def load_stylesheet(name: str) -> str:
    """
    Load a stylesheet by name, with its tokens resolved.

    Args:
        name: Stylesheet name without extension (e.g., 'desktop', 'touch')

    Returns:
        The stylesheet content as a string
    """
    style_path = STYLES_DIR / f"{name}.qss"
    if not style_path.exists():
        return ""
    text = style_path.read_text(encoding="utf-8")
    for token, value in _tokens().items():
        text = text.replace(token, value)
    # A token that survives is not a cosmetic problem: Qt discards the whole
    # rule containing it, silently, so a mistyped token removes styling that
    # nothing then reports as missing. Warn rather than raise - a lab machine
    # should still start - and let the scale test fail the build instead.
    missed = sorted(set(_TOKEN_RE.findall(text)))
    if missed:
        warnings.warn(
            f"{style_path.name} still contains unresolved token(s) {', '.join(missed)}. "
            f"Qt will drop every rule that uses one.",
            stacklevel=2,
        )
    return text


def get_desktop_stylesheet() -> str:
    """Get the desktop mode stylesheet."""
    return load_stylesheet("desktop")


def get_touch_stylesheet() -> str:
    """Get the touch/runner mode stylesheet."""
    return load_stylesheet("touch")
