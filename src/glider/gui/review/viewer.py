"""The viewer: the session's video with poses, zones and a HUD over it.

The canvas draws poses over the video frame and the heads-up display names
what is happening on this frame -- the behaviour, and which outputs the rig is
driving -- so scrubbing reads like watching the session back with its
annotations on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from glider.analysis.behavior.session_view import SessionView
from glider.gui.review.timeline import behavior_qcolor
from glider.gui.styles import colors
from glider.gui.widgets.pastel_glyphs import lucide_icon
from glider.gui.widgets.tool_ui import ElidedLabel, data_font, set_button_role, set_text_role

__all__ = ["TRAIL_DEFAULT_S", "KeypointCanvas", "Transport"]

logger = logging.getLogger(__name__)

TRAIL_DEFAULT_S = 5.0


class KeypointCanvas(QWidget):
    """The animal drawn from its poses, with a trailing centroid track.

    Coordinates are pixels in the source video, so the canvas needs that
    video's resolution to place them. Without it the arena's true extent is
    unknown, and stretching the points to fit their own range would silently
    redraw the enclosure as whatever the animal happened to visit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view: SessionView | None = None
        self._frame = 0
        self._trail_s = TRAIL_DEFAULT_S
        self._show_trail = True
        self._show_video = True
        self._zones = None
        self._heatmap = None
        self._reader = None  # VideoFileSource, opened lazily
        self._cached: tuple[int, QImage] | None = None
        self._show_poses = True
        self._show_zones = True
        self._show_hud = True
        self._hud: tuple[tuple[str, QColor] | None, list[tuple[str, QColor]]] | None = None
        self._hud_samples: list[str] = []  # what sizes the behaviour chip

    def set_view(self, view: SessionView | None) -> None:
        self._close_reader()
        self._view = view
        self._frame = 0
        self.update()

    # ------------------------------------------------------------------
    # video
    # ------------------------------------------------------------------

    def set_show_video(self, enabled: bool) -> None:
        self._show_video = bool(enabled)
        self.update()

    def set_zones(self, zones) -> None:
        """Outline a zone configuration over the arena."""
        self._zones = zones
        self.update()

    def current_frame(self):
        """The decoded BGR frame on screen, or None without a video.

        Decoded regardless of the video toggle: the zone editor wants the
        arena whether or not the operator is looking at it right now.
        """
        if self._view is None or self._view.video_path is None:
            return None
        was_showing, self._show_video = self._show_video, True
        try:
            self._frame_image(self._frame)  # populates the cache and the reader
            if self._reader is None:
                return None
            index = self._frame - self._view.first_video_frame
            return None if index < 0 else self._reader.read_frame(index)
        finally:
            self._show_video = was_showing

    def set_heatmap(self, grid) -> None:
        """Show (or clear, with None) an occupancy histogram over the arena.

        ``grid`` is the ``(nx, ny)`` array ``compute_occupancy`` returns, in
        the same pixel space the arena is drawn in.
        """
        self._heatmap = None
        if grid is None or not getattr(grid, "size", 0) or not np.isfinite(grid).any():
            self.update()
            return
        peak = float(grid.max())
        if peak <= 0:
            self.update()
            return
        # Normalised to its own peak, so a short window is still readable;
        # this is a picture of where time went, not an absolute count.
        normalised = np.clip(grid / peak, 0.0, 1.0)
        nx, ny = normalised.shape
        rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        accent = QColor(colors.ACCENT)
        rgba[..., 0] = accent.red()
        rgba[..., 1] = accent.green()
        rgba[..., 2] = accent.blue()
        # Transposed because histogram2d's first axis is x and an image's is y.
        alpha = (np.sqrt(normalised.T) * 210).astype(np.uint8)
        alpha[normalised.T <= 0] = 0  # never-visited cells stay clear
        rgba[..., 3] = alpha
        self._heatmap = QImage(rgba.tobytes(), nx, ny, 4 * nx, QImage.Format.Format_RGBA8888).copy()
        self.update()

    def has_video(self) -> bool:
        return self._view is not None and self._view.video_path is not None

    def has_heatmap(self) -> bool:
        """Whether an overlay is actually on screen.

        Not the same as "the checkbox is on": set_heatmap refuses a grid whose
        peak is <= 0 or that is all-NaN, so the checkbox can be checked with
        nothing drawn.
        """
        return self._heatmap is not None

    def _close_reader(self) -> None:
        if self._reader is not None:
            self._reader.release()
            self._reader = None
        self._cached = None

    def _frame_image(self, index: int) -> QImage | None:
        """The video frame for *index*, as a QImage, or None.

        Decoded on demand and cached by index: a repaint from resizing or a
        selection change must not cost another decode, and scrubbing one frame
        at a time is a sequential read rather than a seek.
        """
        if not self._show_video or self._view is None or self._view.video_path is None:
            return None
        if self._cached is not None and self._cached[0] == index:
            return self._cached[1]
        if self._reader is None:
            from glider.vision.video_source import VideoFileSource

            reader = VideoFileSource()
            if not reader.load(self._view.video_path):
                logger.info("could not open %s for playback", self._view.video_path)
                self._view.video_path = None  # stop retrying every repaint
                return None
            self._reader = reader
        # A live recording's tracking counts frames from 1 and its video from
        # 0; without the offset every frame shows its neighbour.
        video_index = index - self._view.first_video_frame
        if video_index < 0:
            return None
        frame = self._reader.read_frame(video_index)
        if frame is None:
            return None
        # cv2 gives BGR; copy because the QImage must own its buffer once the
        # numpy array goes out of scope.
        height, width = frame.shape[:2]
        image = QImage(frame.data, width, height, 3 * width, QImage.Format.Format_BGR888).copy()
        self._cached = (index, image)
        return image

    def set_frame(self, frame: int) -> None:
        self._frame = int(frame)
        self.update()

    def set_trail(self, seconds: float, enabled: bool) -> None:
        self._trail_s, self._show_trail = float(seconds), bool(enabled)
        self.update()

    def _transform(self):
        """Scale and offset mapping video pixels onto the widget, or None."""
        if self._view is None or not self._view.resolution:
            return None
        width, height = self._view.resolution
        scale = min(self.width() / width, self.height() / height)
        return scale, (self.width() - width * scale) / 2, (self.height() - height * scale) / 2

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(colors.CANVAS))

        transform = self._transform()
        if (
            self._view is None
            or transform is None
            or (self._view.xy is None and not self.has_video())
        ):
            painter.setPen(QPen(QColor(colors.TEXT_MUTED)))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._why_blank(),
            )
            return

        scale, dx, dy = transform
        width, height = self._view.resolution
        arena = QRectF(dx, dy, width * scale, height * scale)

        image = self._frame_image(self._frame)
        if image is not None:
            painter.drawImage(arena, image)
        if self._heatmap is not None:
            painter.drawImage(arena, self._heatmap)
        painter.setPen(QPen(QColor(colors.BORDER), 1))
        painter.drawRect(arena)
        if self._show_zones:
            self._paint_zones(painter, arena)

        def to_widget(point):
            return QPointF(point[0] * scale + dx, point[1] * scale + dy)

        if self._show_trail and self._view.xy is not None:
            trail = self._view.trail(self._frame, self._trail_s)
            if trail is not None and len(trail) > 1:
                # Fade the tail so recent travel reads as the leading edge.
                for i in range(1, len(trail)):
                    alpha = 0.15 + 0.65 * (i / len(trail))
                    painter.setPen(QPen(colors.qcolor_with_alpha(QColor(colors.ACCENT), alpha), 2))
                    painter.drawLine(to_widget(trail[i - 1]), to_widget(trail[i]))

        if self._show_poses:
            if self._view.xy is not None and 0 <= self._frame < len(self._view.xy):
                points = self._view.xy[self._frame]
                names = self._view.keypoint_names
                for i, point in enumerate(points):
                    if not np.isfinite(point).all():
                        continue
                    # Keyed on the keypoint list, so seven body parts get seven
                    # different colours rather than whatever a hash of each name
                    # happened to pick.
                    painter.setBrush(
                        QBrush(behavior_qcolor(names[i] if i < len(names) else str(i), names))
                    )
                    painter.setPen(QPen(QColor(colors.CANVAS), 1))
                    painter.drawEllipse(to_widget(point), 5, 5)

        if self._hud is None:
            label = self._view.label_at(self._frame)
            painter.setPen(QPen(QColor(colors.TEXT_PRIMARY)))
            painter.drawText(
                QRectF(8, 6, self.width() - 16, 20),
                Qt.AlignmentFlag.AlignLeft,
                f"frame {self._frame}   {label or '(unscored)'}",
            )
        else:
            self._paint_hud(painter)

    def _paint_zones(self, painter, arena: QRectF) -> None:
        """Outline each zone in the colour the live overlay draws it in.

        Zone geometry is normalised, so it maps onto whatever rectangle the
        arena occupies on screen without knowing the resolution.
        """
        if self._zones is None:
            return
        for zone in getattr(self._zones, "zones", []):
            if not zone.vertices:
                continue
            b, g, r = zone.color
            painter.setPen(QPen(QColor(r, g, b), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            points = [
                QPointF(arena.x() + vx * arena.width(), arena.y() + vy * arena.height())
                for vx, vy in zone.vertices
            ]
            name = str(getattr(zone.shape, "value", zone.shape)).lower()
            if name == "rectangle" and len(points) >= 2:
                painter.drawRect(QRectF(points[0], points[1]).normalized())
            elif name == "circle" and len(points) >= 2:
                radius = (
                    (points[1].x() - points[0].x()) ** 2 + (points[1].y() - points[0].y()) ** 2
                ) ** 0.5
                painter.drawEllipse(points[0], radius, radius)
            elif len(points) >= 3:
                painter.drawPolygon(*points)

    def _why_blank(self) -> str:
        view = self._view
        if view is None:
            return "Load a session"
        # A recording's position is its tracked centroid: there is no pose
        # CSV to choose and no sidecar to repair, so neither button is shown.
        recording = view.keypoint_names == ["centroid"] or (
            view.source is not None and Path(view.source).is_dir()
        )
        if view.xy is None and recording:
            return (
                "This recording's tracking CSV has no centroid (center_x, "
                "center_y), so there is no position to draw."
            )
        if view.xy is None:
            return (
                "No pose CSV could be found for this session.\n\n"
                "Looked at the path recorded in run.json, then beside the "
                "ethogram, then for a CSV named after this session in the "
                "folders above.\n\n"
                "Use “Choose pose CSV…” on the inspector's Session tab to point at it."
            )
        if recording:
            return (
                "This recording's frame size is unknown: no calibration header "
                "records it and no readable video was found, so the arena cannot "
                "be sized."
            )
        return (
            "This session's pose sidecar records no resolution, so the arena "
            "cannot be sized.\n\n"
            "Use “Set arena size from video…” on the inspector's Session tab to "
            "read it from the source video — it is stored, so this is a one-off."
        )

    def set_show_poses(self, enabled: bool) -> None:
        self._show_poses = bool(enabled)
        self.update()

    def set_show_zones(self, enabled: bool) -> None:
        self._show_zones = bool(enabled)
        self.update()

    def set_hud(self, behavior, chips) -> None:
        """What this frame is: ``(text, colour)`` for the behaviour, and a chip
        per active output as ``(text, colour)``."""
        self._hud = (behavior, list(chips))
        self.update()

    def set_hud_names(self, names) -> None:
        """The behaviours this session can show; they fix the behaviour chip's width.

        Sized to the widest name with a long "seconds in", so the chip holds
        still while scrubbing instead of twitching with every digit.
        """
        self._hud_samples = [f"{name}  000.00 s in" for name in names]
        self.update()

    def behaviour_chip_rect(self, text: str) -> QRectF:
        """Where the behaviour chip showing ``text`` is drawn."""
        return self._chip_rect(12.0, 10.0, text, right=False, large=True)

    def set_show_hud(self, enabled: bool) -> None:
        self._show_hud = bool(enabled)
        self.update()

    _MAX_CHIPS = 4

    def _paint_hud(self, painter: QPainter) -> None:
        if not self._show_hud or self._hud is None:
            return
        behavior, chips = self._hud
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setFont(self.font())
        if behavior is not None:
            # The one thing a reviewer reads while scrubbing, so it is set large.
            self._chip(painter, 12.0, 10.0, *behavior, right=False, large=True)
        y = 10.0
        for text, colour in chips[: self._MAX_CHIPS]:
            self._chip(painter, self.width() - 12.0, y, text, colour, right=True)
            y += 24
        if len(chips) > self._MAX_CHIPS:
            extra = f"+{len(chips) - self._MAX_CHIPS}"
            self._chip(
                painter, self.width() - 12.0, y, extra, QColor(colors.TEXT_MUTED), right=True
            )

    def _chip_font(self, large: bool) -> QFont:
        font = QFont(self.font())
        if large:
            size = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
            font.setPointSizeF(size * 1.45)
            font.setBold(True)
        return font

    def _chip_rect(self, x, y, text, *, right, large) -> QRectF:
        height = BEHAVIOUR_CHIP_H if large else CHIP_H
        pad, dot = height / 2, height / 5
        metrics = QFontMetricsF(self._chip_font(large))
        # The behaviour chip is as wide as the session's widest behaviour, so
        # its edge stays put as the label and the seconds change under it.
        samples = [text, *self._hud_samples] if large else [text]
        width = max(metrics.horizontalAdvance(t) for t in samples) + pad + dot + 8 + pad * 0.8
        return QRectF(x - width if right else x, y, width, height)

    def _chip(self, painter, x, y, text, colour, *, right, large=False) -> None:
        rect = self._chip_rect(x, y, text, right=right, large=large)
        pad, dot = rect.height() / 2, rect.height() / 5
        text_left = pad + dot + 8
        painter.setFont(self._chip_font(large))
        painter.setPen(QPen(colors.qcolor_with_alpha(QColor(colors.TEXT_PRIMARY), 0.08), 1))
        painter.setBrush(colors.qcolor_with_alpha(QColor(colors.CANVAS), 0.78))
        painter.drawRoundedRect(rect, pad, pad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawEllipse(QPointF(rect.left() + pad, rect.top() + pad), dot, dot)
        painter.setPen(QColor(colors.TEXT_PRIMARY))
        painter.drawText(
            QRectF(rect.left() + text_left, rect.top(), rect.width() - text_left, rect.height()),
            Qt.AlignmentFlag.AlignVCenter,
            text,
        )


#: Heights of the HUD chips: the behaviour chip top-left, the output chips top-right.
CHIP_H = 22.0
BEHAVIOUR_CHIP_H = 36.0

#: Transport glyphs are Lucide (``gui/styles/icons/lucide/``, ISC).
_ICON_SIZE = QSize(18, 18)


def _text_width(label: QLabel, text: str) -> int:
    return int(QFontMetricsF(label.font()).horizontalAdvance(text)) + 1


def _icon_button(icon: str, tip: str) -> QPushButton:
    button = QPushButton()
    button.setIcon(lucide_icon(icon, colors.TEXT_SECONDARY, _ICON_SIZE.width()))
    button.setIconSize(_ICON_SIZE)
    button.setToolTip(tip)
    button.setFixedWidth(36)
    set_button_role(button, "icon")
    return button


def _toggle(text: str, tip: str, *, checked: bool) -> QCheckBox:
    box = QCheckBox(text)
    box.setObjectName("ToggleChip")
    box.setToolTip(tip)
    box.setChecked(checked)
    return box


class Transport(QFrame):
    """Play controls, the timecode, and what the viewer draws."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Transport")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(4)
        self.to_start = _icon_button("skip-back", "Session start  (Home)")
        self.back = _icon_button("step-back", "Back one frame  (←)")
        self.play = QPushButton()
        self.play.setMinimumWidth(92)
        self.play.setIconSize(QSize(16, 16))
        self.play.setToolTip("Play / pause  (Space)")
        set_button_role(self.play, "primary")
        self.set_playing(False)
        self.forward = _icon_button("step-forward", "Forward one frame  (→)")
        self.to_end = _icon_button("skip-forward", "Session end  (End)")
        for button in (self.to_start, self.back, self.play, self.forward, self.to_end):
            row.addWidget(button)
        # Every read-out in this row changes as the playhead moves, and the
        # row's minimum width is the viewer panel's: a read-out that grew with
        # its text pushed the splitter, and the video slid sideways. The clock
        # and counter are fixed to their widest text, and the bout read-out
        # takes what is left and elides.
        self.clock = QLabel("—")
        self.clock.setObjectName("TransportClock")
        self.clock.setFont(data_font(15))
        self.clock.setFixedWidth(_text_width(self.clock, "-00:00:00:00") + 14)
        self.position = QLabel("")
        self.position.setFont(data_font(9))
        set_text_role(self.position, "muted")
        self.position.setFixedWidth(_text_width(self.position, "000,000 / 000,000") + 4)
        row.addSpacing(8)
        row.addWidget(self.clock)
        row.addWidget(self.position)
        self.bout = ElidedLabel("—")
        self.bout.setFont(data_font(9))
        set_text_role(self.bout, "caption")
        row.addSpacing(8)
        row.addWidget(self.bout, 1)
        self.video_on = _toggle("Video", "Draw the session's video", checked=True)
        self.poses_on = _toggle("Poses", "Draw the tracked keypoints", checked=True)
        self.trail_on = _toggle("Trail", "Draw the recent centroid track", checked=True)
        self.heatmap_on = _toggle(
            "Heatmap", "Where the animal spent the selected range", checked=False
        )
        self.zones_on = _toggle("Zones", "Outline the loaded zones", checked=True)
        self.hud_on = _toggle(
            "HUD", "Name the behaviour and active outputs on each frame", checked=True
        )
        for toggle in (
            self.video_on,
            self.poses_on,
            self.trail_on,
            self.heatmap_on,
            self.zones_on,
            self.hud_on,
        ):
            row.addWidget(toggle)
        self.rate = QLabel("1×")
        self.rate.setFont(data_font(9))
        set_text_role(self.rate, "muted")
        row.addSpacing(6)
        row.addWidget(self.rate)

    def set_playing(self, playing: bool) -> None:
        """Show Pause while playing and Play otherwise, icon and word together."""
        # Dark on the accent-filled primary button.
        self.play.setIcon(lucide_icon("pause" if playing else "play", colors.CANVAS, 16))
        self.play.setText("Pause" if playing else "Play")
