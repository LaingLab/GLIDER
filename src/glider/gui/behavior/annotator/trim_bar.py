"""Per-clip trim editor: a timeline with draggable IN/OUT handles.

The annotator proposes a short clip; the labeler trims it to the exact
behavior boundary before assigning a label, which keeps each saved zone
single-behavior. The trim happens inside a *padded* seekable window
(the proposed clip plus context on each side) so the true start/end of
a behavior — which often spills just past the tiny proposed clip — is
reachable.

This module's geometry/clamping math is split into pure functions so it
can be tested without a Qt event loop. The :class:`TrimBar` widget is a
thin painted track that drives those functions from mouse/keyboard.
"""

from __future__ import annotations


def compute_window(
    clip_start: int,
    clip_end: int,
    pad: int,
    n_frames: int | None = None,
) -> tuple[int, int]:
    """Return the padded, clamped seekable window for a proposed clip.

    The window is ``[clip_start - pad, clip_end + pad)``, clamped low to
    ``0`` and (when ``n_frames`` is known) high to ``n_frames``. When the
    video length isn't known the high edge is left unclamped — the clip
    player stops at EOF anyway.
    """
    win_start = max(0, int(clip_start) - int(pad))
    win_end = int(clip_end) + int(pad)
    if n_frames is not None:
        win_end = min(win_end, int(n_frames))
    win_end = max(win_end, win_start + 1)
    return win_start, win_end


# Padding on each end of the painted track. Shared, because anything drawn
# beneath the trim bar has to use the same mapping or it will not line up.
TRACK_MARGIN = 10


def track_width(width: int, margin: int = TRACK_MARGIN) -> int:
    """Usable track width for a widget ``width`` px wide."""
    return max(1, int(width) - 2 * int(margin))


def frame_to_x(
    frame: int,
    win_start: int,
    win_end: int,
    width: int,
    margin: int = TRACK_MARGIN,
) -> int:
    """Pixel x for ``frame`` within the window ``[win_start, win_end)``.

    Pure, and shared by :class:`TrimBar` and the speed trace drawn under it.
    A second copy of this arithmetic is how two stacked timelines drift a few
    pixels apart and stop meaning the same thing.
    """
    span = max(1, int(win_end) - int(win_start))
    frac = (frame - int(win_start)) / span
    return int(int(margin) + frac * track_width(width, margin))


def clamp_trim_bounds(
    in_frame: int,
    out_frame: int,
    win_start: int,
    win_end: int,
) -> tuple[int, int]:
    """Keep ``in < out`` and both inside ``[win_start, win_end]``.

    Enforces a minimum one-frame span. ``win_end`` is exclusive (a valid
    out point), so ``in`` may range over ``[win_start, win_end - 1]``.
    """
    in_f = max(int(win_start), min(int(in_frame), int(win_end) - 1))
    out_f = max(in_f + 1, min(int(out_frame), int(win_end)))
    return in_f, out_f


# ---------------------------------------------------------------------------
# Widget (only defined when PyQt6 is available — the helpers above don't
# need it, so importing this module stays cheap and Qt-free for tests).
# ---------------------------------------------------------------------------
try:
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtGui import QColor, QPainter
    from PyQt6.QtWidgets import QWidget

    _HAVE_QT = True
except ImportError:  # pragma: no cover - exercised only in no-UI installs
    _HAVE_QT = False


if _HAVE_QT:

    _MARGIN = TRACK_MARGIN  # kept as a local alias; the constant is shared

    class TrimBar(QWidget):
        """Timeline track with draggable IN/OUT handles over a padded window.

        Emits :data:`bounds_changed` ``(in_frame, out_frame)`` whenever the
        selection changes — by drag, by :meth:`nudge_in` / :meth:`nudge_out`,
        or by :meth:`set_bounds`. The owner wires that signal to the clip
        player's loop bounds so the loop always previews the current trim.
        """

        bounds_changed = pyqtSignal(int, int)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFixedHeight(44)
            self.setMinimumWidth(200)
            self._win_start = 0
            self._win_end = 1
            self._in = 0
            self._out = 1
            self._clip_start = 0
            self._clip_end = 1
            self._dragging: str | None = None  # "in" | "out" | None

        # ---- public API ----
        def set_window(self, win_start: int, win_end: int) -> None:
            self._win_start = int(win_start)
            self._win_end = max(int(win_end), self._win_start + 1)
            self._in, self._out = clamp_trim_bounds(
                self._in, self._out, self._win_start, self._win_end
            )
            self.update()

        def set_clip_region(self, clip_start: int, clip_end: int) -> None:
            self._clip_start = int(clip_start)
            self._clip_end = int(clip_end)
            self.update()

        def set_bounds(self, in_frame: int, out_frame: int) -> None:
            self._in, self._out = clamp_trim_bounds(
                in_frame, out_frame, self._win_start, self._win_end
            )
            self.update()
            self.bounds_changed.emit(self._in, self._out)

        def bounds(self) -> tuple[int, int]:
            return (self._in, self._out)

        def nudge_in(self, delta: int) -> None:
            self.set_bounds(self._in + int(delta), self._out)

        def nudge_out(self, delta: int) -> None:
            self.set_bounds(self._in, self._out + int(delta))

        # ---- frame <-> pixel mapping ----
        def _track_width(self) -> int:
            return track_width(self.width())

        def _frame_to_x(self, frame: int) -> int:
            return frame_to_x(frame, self._win_start, self._win_end, self.width())

        def _x_to_frame(self, x: int) -> int:
            span = max(1, self._win_end - self._win_start)
            frac = (x - _MARGIN) / self._track_width()
            return int(round(self._win_start + frac * span))

        # ---- mouse drag ----
        def mousePressEvent(self, event) -> None:
            x = int(event.position().x())
            in_x, out_x = self._frame_to_x(self._in), self._frame_to_x(self._out)
            self._dragging = "in" if abs(x - in_x) <= abs(x - out_x) else "out"
            self._drag_to(x)

        def mouseMoveEvent(self, event) -> None:
            if self._dragging:
                self._drag_to(int(event.position().x()))

        def mouseReleaseEvent(self, event) -> None:
            self._dragging = None

        def _drag_to(self, x: int) -> None:
            frame = self._x_to_frame(x)
            if self._dragging == "in":
                self.set_bounds(frame, self._out)
            elif self._dragging == "out":
                self.set_bounds(self._in, frame)

        # ---- paint ----
        def paintEvent(self, event) -> None:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            h = self.height()
            mid = h // 2
            # Track baseline.
            p.fillRect(_MARGIN, mid - 3, self._track_width(), 6, QColor("#e5e5e5"))
            # Proposed-clip shaded region.
            cs, ce = self._frame_to_x(self._clip_start), self._frame_to_x(self._clip_end)
            p.fillRect(cs, mid - 3, max(1, ce - cs), 6, QColor("#cbd5e1"))
            # Selected span.
            ix, ox = self._frame_to_x(self._in), self._frame_to_x(self._out)
            p.fillRect(ix, mid - 5, max(1, ox - ix), 10, QColor("#2563eb"))
            # Handles.
            for hx in (ix, ox):
                p.fillRect(hx - 3, mid - 11, 6, 22, QColor("#1d4ed8"))
            p.end()
