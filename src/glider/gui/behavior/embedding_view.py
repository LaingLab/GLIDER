"""Rotatable 3D scatter of a model's feature-space embedding.

``fit_embedding`` reduces the training features to three dimensions and
stores the result in the model bundle — the "galaxy" of what the classifier
actually learned. Nothing in the GUI could draw it, so every bundle carrying
one had nothing to show for it.

Hand-painted with ``QPainter``, like :class:`~...annotator.speed_trace.SpeedTrace`,
the trim bar and the Review tab's confusion grid. A 3D scatter of a few
thousand points does not need OpenGL, and a new dependency for one panel
would cost more than it returns. Projection is orthographic: for judging
whether classes separate, parallel lines staying parallel is more useful
than perspective foreshortening.

What it is for: seeing whether the behaviours occupy distinct regions. Two
classes sitting on top of each other here is the same story the confusion
matrix tells in numbers, and it is usually visible at a glance long before
anyone reads a table.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QSizePolicy, QWidget

from glider.gui.styles import colors

#: Colour-blind-safe categorical palette (Okabe-Ito), matching the one the
#: report charts use so a class looks the same in both.
_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#999999",
]


class EmbeddingView(QWidget):
    """A draggable 3D scatter of ``EmbeddingArtifact.coords``."""

    #: Points drawn at most. The artifact already caps its own sample at
    #: 20,000; repainting that many on every drag frame makes rotation crawl,
    #: and the shape of a cloud is legible from a fraction of it.
    MAX_POINTS = 4000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._coords: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._names: list[str] = []
        self._colors: dict[str, QColor] = {}
        self._method = ""
        self._yaw = 0.6
        self._pitch = 0.35
        self._drag: QPoint | None = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def set_artifact(self, artifact) -> None:
        """Show ``artifact``; pass None to clear."""
        if artifact is None or getattr(artifact, "coords", None) is None:
            self._coords = self._labels = None
            self._names = []
            self.update()
            return

        coords = np.asarray(artifact.coords, dtype=np.float64)
        labels = np.asarray(artifact.labels)
        if coords.ndim != 2 or coords.shape[0] == 0:
            self._coords = self._labels = None
            self._names = []
            self.update()
            return
        # Pad to three columns so a 2D reducer still renders rather than
        # raising deep inside the projection.
        if coords.shape[1] < 3:
            coords = np.hstack([coords, np.zeros((len(coords), 3 - coords.shape[1]))])

        if len(coords) > self.MAX_POINTS:
            # Evenly spaced rather than random: deterministic across repaints,
            # and it cannot accidentally drop a whole small class.
            keep = np.linspace(0, len(coords) - 1, self.MAX_POINTS).astype(int)
            coords, labels = coords[keep], labels[keep]

        self._coords = coords
        self._labels = labels
        self._method = str(getattr(artifact, "method", "") or "")
        # Sorted, so a class keeps its colour whatever order the rows arrive
        # in — otherwise the legend means something different every refit.
        self._names = sorted({str(v) for v in labels})
        self._colors = {
            name: QColor(_PALETTE[i % len(_PALETTE)]) for i, name in enumerate(self._names)
        }
        self.update()

    def has_data(self) -> bool:
        return self._coords is not None

    def class_names(self) -> list[str]:
        return list(self._names)

    def color_for(self, name: str) -> QColor:
        return self._colors.get(str(name), QColor(colors.TEXT_SECONDARY))

    def drawn_point_count(self) -> int:
        return 0 if self._coords is None else len(self._coords)

    def rotation(self) -> tuple[float, float]:
        return (self._yaw, self._pitch)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._drag = event.position().toPoint()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag is None:
            return
        pos = event.position().toPoint()
        delta = pos - self._drag
        self._drag = pos
        self._yaw += delta.x() * 0.01
        # Clamped so the cloud cannot be tipped past vertical and come back
        # mirrored, which reads as the data having changed.
        self._pitch = max(-1.5, min(1.5, self._pitch + delta.y() * 0.01))
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._drag = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def _project(self) -> tuple[np.ndarray, np.ndarray]:
        """``(xy, depth)`` in widget pixels, orthographic."""
        pts = self._coords
        centred = pts - pts.mean(axis=0)

        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        cp, sp = np.cos(self._pitch), np.sin(self._pitch)
        x = centred[:, 0] * cy + centred[:, 2] * sy
        z = -centred[:, 0] * sy + centred[:, 2] * cy
        y = centred[:, 1] * cp - z * sp
        depth = centred[:, 1] * sp + z * cp

        pad = 26
        w, h = max(1, self.width() - 2 * pad), max(1, self.height() - 2 * pad)
        # A degenerate cloud (every point identical) has zero extent; scaling
        # to fit would divide by it.
        span = max(np.ptp(x), np.ptp(y), 1e-9)
        scale = min(w, h) / span * 0.9
        sx = x * scale + self.width() / 2
        sy_ = -y * scale + self.height() / 2
        return np.column_stack([sx, sy_]), depth

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------
    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(colors.SURFACE_1))

        if not self.has_data():
            p.setPen(QColor(colors.TEXT_SECONDARY))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No embedding in this run.\nTrain with an embedding to see it here.",
            )
            p.end()
            return

        xy, depth = self._project()
        # Far points first, so near ones land on top and the cloud reads as
        # having a front and a back.
        order = np.argsort(depth)
        lo, hi = float(depth.min()), float(depth.max())
        spread = max(hi - lo, 1e-9)

        for i in order:
            name = str(self._labels[i])
            color = QColor(self._colors.get(name, QColor(colors.TEXT_SECONDARY)))
            # Depth cue: nearer points are more opaque and slightly larger.
            near = (float(depth[i]) - lo) / spread
            color.setAlphaF(0.35 + 0.55 * near)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            r = 1.6 + 1.8 * near
            p.drawEllipse(QPointF(float(xy[i, 0]), float(xy[i, 1])), r, r)

        self._paint_legend(p)
        p.end()

    def _paint_legend(self, p: QPainter) -> None:
        p.setPen(QColor(colors.TEXT_SECONDARY))
        y = 16
        if self._method:
            p.drawText(10, y, f"{self._method.upper()} · drag to rotate")
            y += 16
        for name in self._names:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self._colors[name])
            p.drawEllipse(QPointF(14.0, y - 3.0), 4.0, 4.0)
            p.setPen(QColor(colors.TEXT_PRIMARY))
            p.drawText(24, y, name)
            y += 15
