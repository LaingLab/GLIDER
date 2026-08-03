"""
A mouse running in an exercise wheel, for use as a loading indicator.

Drawn with QPainter rather than played back from a GIF or SVG: Qt's SVG
support is static, a GIF would need a binary asset per size and theme, and
the geometry here is cheap enough to evaluate per frame. That keeps it
resolution-independent and lets it pick the wheel colour up from
``glider.gui.styles.colors``.

The rig is a two-bone IK solve per limb over a ground curve. Three details
are load-bearing and easy to undo by accident:

* The ground is the wheel's *arc*, not a flat line. Feet solved against a
  flat ground pass through the rim as they get further from bottom-centre.
* Swing must begin from the pose stance ended in — up on the toe — and settle
  flat for the landing. Resetting to flat at toe-off makes the foot teleport.
* Limbs solve in world space from transformed hips. Rotating the legs and the
  ground together makes planted feet slide.

Limb lengths are short and the swing lift low on purpose. Lifting a foot near
its own hip drives the IK into an ill-conditioned region, where the knee
swings through a wide arc for almost no movement of the foot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QElapsedTimer, QPointF, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import colors

# ~30 fps. Smooth enough for a spinner, and it leaves headroom on a Pi where
# this may run beside live tracking.
_FRAME_MS = 33

# Local units: the mouse faces +x, the running surface is y = 0, and y grows
# downward. The body spans roughly x -22..+35, y -30..-5.
_MOUSE_UNITS = 46.0

# Below this widget size the rungs alias into a grey ring, so we draw a plain
# rim and let the mouse carry the animation.
_PLAIN_RIM_BELOW = 32

# Fur keyed to a fawn lab mouse. Specific to this drawing rather than theme
# colours, so they live here instead of in the palette module.
_FUR = "#d9b98a"
_FUR_LO = "#a98c63"
_BELLY = "#e8d4b6"
_PINK = "#dfa79e"
_PINK_LO = "#be857d"
_EYE = "#100d0b"
_OUTLINE = "#6e5636"


@dataclass(frozen=True)
class _Limb:
    """One limb. ``lf`` is the foot, which for the hind leg is nearly as long
    as the shin — mice are plantigrade behind and stand on the whole foot."""

    x: float
    y: float
    l1: float  # femur / humerus
    l2: float  # tibia / radius
    lf: float  # foot
    ankle_h: float  # ankle height above the surface with the foot flat
    stride: float
    lift: float  # extra height at mid-swing, on top of the toe-off pose
    w_up: float
    w_lo: float
    w_foot: float
    bend: int  # +1 elbow points back, -1 knee points forward


# A short femur keeps the knee tucked against the body and shrinks the arc it
# sweeps, since the knee always sits l1 from the hip.
_HIND = _Limb(
    x=-10.5,
    y=-11.5,
    l1=5.6,
    l2=8.6,
    lf=9.2,
    ankle_h=2.3,
    stride=10.5,
    lift=2.6,
    w_up=6.4,
    w_lo=3.6,
    w_foot=3.0,
    bend=-1,
)
_FORE = _Limb(
    x=13.5,
    y=-13.0,
    l1=4.8,
    l2=7.0,
    lf=3.6,
    ankle_h=1.3,
    stride=8.5,
    lift=2.2,
    w_up=4.6,
    w_lo=2.8,
    w_foot=2.4,
    bend=1,
)


@dataclass(frozen=True)
class _Gait:
    fore: tuple[float, float]  # phase offsets, near limb then far
    hind: tuple[float, float]
    stance: float  # fraction of the cycle a foot is planted
    bob: float
    bob_f: int
    flex: float  # spine flex, counter-shifting shoulders and hips
    flex_ph: float  # offset from the bob, so the two do not stack on the hip


GAITS: dict[str, _Gait] = {
    "bound": _Gait((0.0, 0.07), (0.45, 0.52), 0.46, 1.8, 1, 1.1, 0.25),
    "gallop": _Gait((0.0, 0.15), (0.42, 0.60), 0.44, 1.6, 1, 0.9, 0.25),
    "trot": _Gait((0.0, 0.50), (0.50, 0.00), 0.58, 1.5, 2, 0.4, 0.0),
}

_body_cache: QPainterPath | None = None


def _body_path() -> QPainterPath:
    """The body outline, built once and reused."""
    global _body_cache
    if _body_cache is None:
        path = QPainterPath()
        path.moveTo(-20, -16)
        path.cubicTo(-23, -26, -8, -30.5, 6, -29.5)
        path.cubicTo(15, -28.5, 21, -26, 25, -22)
        path.cubicTo(29, -19, 33, -17, 34.2, -14.5)
        path.cubicTo(34.8, -12.8, 33, -11.4, 30.4, -11.4)
        path.cubicTo(27, -11.4, 24, -12, 21.5, -12.6)
        path.cubicTo(13, -7.2, -2, -5.6, -13, -8)
        path.cubicTo(-17.5, -9.6, -19.5, -12, -20, -16)
        path.closeSubpath()
        _body_cache = path
    return _body_cache


def _knee(hx: float, hy: float, fx: float, fy: float, l1: float, l2: float, bend: int) -> QPointF:
    """Elbow/knee position for a two-bone chain from (hx, hy) to (fx, fy)."""
    dx, dy = fx - hx, fy - hy
    d = min(math.hypot(dx, dy), l1 + l2 - 0.01)
    a = math.atan2(dy, dx)
    cos_b = (d * d + l1 * l1 - l2 * l2) / (2 * d * l1)
    b = math.acos(max(-1.0, min(1.0, cos_b)))
    k = a + bend * b
    return QPointF(hx + math.cos(k) * l1, hy + math.sin(k) * l1)


def _toe_down_angle(limb: _Limb, ankle_x: float, ankle_y: float, gy) -> float:
    """Absolute foot angle that rests the toe on the running surface.

    On a flat ground this would just be ``asin(ankle_h / lf)`` — drop the toe
    by the ankle height and it lands. The wheel curves away under the foot, so
    the toe has to be found *on the arc* instead; assuming flat ground puts a
    9-unit hind foot roughly a unit through the rim. Converges in a couple of
    passes because the surface is shallow over one foot length.
    """
    toe_x = ankle_x + limb.lf
    for _ in range(4):
        dy = gy(toe_x) - ankle_y
        dx = math.sqrt(max(1e-6, limb.lf * limb.lf - dy * dy))
        toe_x = ankle_x + dx
    return math.atan2(gy(toe_x) - ankle_y, toe_x - ankle_x)


def _ankle_at(limb: _Limb, base_x: float, p: float, stance: float, gy) -> tuple[QPointF, float]:
    """Ankle position and foot angle at cycle phase ``p``.

    The two branches agree at both boundaries: swing starts in the rolled-up
    pose stance ends in, and finishes flat where stance begins.
    """
    p -= math.floor(p)
    h_lift = limb.ankle_h + limb.lf * 0.42
    back_x = base_x - limb.stride / 2
    front_x = base_x + limb.stride / 2

    if p < stance:
        u = p / stance
        x = front_x - u * limb.stride
        # Smoothstep the heel roll so it eases off rather than hinging.
        rr = (u - 0.66) / 0.34 if u > 0.66 else 0.0
        roll = rr * rr * (3 - 2 * rr)
        h = limb.ankle_h + roll * (h_lift - limb.ankle_h)
        ankle_y = gy(x) - h
        # Through all of stance the toe stays planted — rolling off the toe
        # raises the heel, which this produces for free as h grows.
        return QPointF(x, ankle_y), _toe_down_angle(limb, x, ankle_y, gy)

    v = (p - stance) / (1 - stance)
    arc = math.sin(math.pi * v)
    x = back_x + v * limb.stride
    h = h_lift + (limb.ankle_h - h_lift) * v + limb.lift * arc
    ankle_y = gy(x) - h

    # Blend between the pose it lifted off in and the pose it has to land in,
    # both solved on the surface, then lift the toe through mid-swing so it
    # clears the rungs.
    #
    # Deliberately NOT surface-tracking here. Mid-swing the ankle rises above
    # the foot's own length (5.0 against a 3.6 forefoot), so "point the toe at
    # the ground" has no solution and saturates at vertical — a visible snap.
    # Blending the two endpoint poses keeps the foot inside the rim anyway,
    # provided _toe_down_angle has actually converged; at three iterations it
    # had not, and the toe grazed through by 0.05 units.
    a_lift = _toe_down_angle(limb, back_x, gy(back_x) - h_lift, gy)
    a_land = _toe_down_angle(limb, front_x, gy(front_x) - limb.ankle_h, gy)
    a = a_lift + (a_land - a_lift) * v - 0.45 * arc
    return QPointF(x, ankle_y), a


def _tail_points(t: float, root: QPointF, base_ang: float, rim_r: float) -> list[QPointF]:
    """Tail joints, every one clamped inside the wheel.

    The tail is in a cage and cannot leave one, so it rides up the trailing
    face instead of passing through the rungs. Wave amplitude grows toward the
    tip — applied evenly it accumulates down the chain and flails.
    """
    n, seg = 11, 5.4
    cy = -rim_r
    max_r = rim_r - 2.4
    x, y = root.x(), root.y()
    ang = base_ang
    pts = [QPointF(x, y)]
    for i in range(n):
        amp = 0.075 * (i / (n - 1))
        ang += 0.115 + amp * math.sin(2 * math.pi * t - i * 0.42)
        nx = x + math.cos(ang) * seg
        ny = y + math.sin(ang) * seg
        dx, dy = nx, ny - cy
        d = math.hypot(dx, dy)
        if d > max_r:
            nx, ny = (dx / d) * max_r, cy + (dy / d) * max_r
        ang = math.atan2(ny - y, nx - x)
        x, y = nx, ny
        pts.append(QPointF(x, y))
    return pts


class MouseRunner(QWidget):
    """A mouse running in a wheel, animating only while visible.

    The timer stops on hide, so an indicator left in a hidden tab costs
    nothing.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        gait: str = "bound",
        speed: float = 1.30,
        pitch_deg: float = -14.0,
        lean_deg: float = 0.0,
    ) -> None:
        super().__init__(parent)
        if gait not in GAITS:
            raise ValueError(f"unknown gait {gait!r}; expected one of {sorted(GAITS)}")
        self._gait = gait
        self._speed = speed
        self._pitch = math.radians(pitch_deg)
        self._lean = math.radians(lean_deg)
        self._t = 0.1

        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(96, 96)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(24, 24)

    # -- lifecycle -----------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._clock.restart()
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._timer.stop()
        super().hideEvent(event)

    def is_animating(self) -> bool:
        return self._timer.isActive()

    def _tick(self) -> None:
        self.advance(self._clock.restart() / 1000.0)
        self.update()

    def advance(self, seconds: float) -> None:
        """Step the animation by wall-clock seconds, without an event loop."""
        self._t += seconds * 1.05 * self._speed

    # -- painting ------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        side = min(self.width(), self.height())
        if side <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_wheel(
            painter,
            self.width() / 2,
            self.height() * 0.46,
            side * 0.40,
            plain_rim=side < _PLAIN_RIM_BELOW,
        )
        painter.end()

    def _draw_wheel(self, p: QPainter, cx: float, cy: float, r: float, *, plain_rim: bool) -> None:
        ri = r - r * 0.10
        ms = r / _MOUSE_UNITS
        rim_local = ri / ms

        def gy(x: float) -> float:
            return -(rim_local - math.sqrt(max(0.0, rim_local * rim_local - x * x)))

        pen = QPen(QColor(colors.TEXT_MUTED))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        p.save()
        p.translate(cx, cy)
        p.setBrush(Qt.BrushStyle.NoBrush)

        p.setOpacity(0.85)
        pen.setWidthF(r * 0.035)
        p.setPen(pen)
        p.drawEllipse(QPointF(0, 0), r, r)

        if not plain_rim:
            pen.setWidthF(r * 0.028)
            p.setPen(pen)
            p.drawEllipse(QPointF(0, 0), ri, ri)

            p.setOpacity(0.6)
            pen.setWidthF(r * 0.022)
            p.setPen(pen)
            spin = -self._t * (math.pi * 2) / 9
            for i in range(34):
                a = spin + (i / 34) * math.pi * 2
                p.drawLine(
                    QPointF(math.cos(a) * r, math.sin(a) * r),
                    QPointF(math.cos(a) * ri, math.sin(a) * ri),
                )

            p.setOpacity(0.8)
            pen.setWidthF(r * 0.028)
            p.setPen(pen)
            hub = r * 0.08
            p.drawEllipse(QPointF(0, 0), hub, hub)
            p.drawLine(QPointF(0, hub), QPointF(0, r * 1.18))
            p.drawLine(QPointF(-r * 0.46, r * 1.18), QPointF(r * 0.46, r * 1.18))

        p.setOpacity(1.0)

        p.save()
        p.rotate(math.degrees(-self._lean))
        p.translate(0, ri)
        p.scale(ms, ms)
        self._draw_mouse(p, gy, rim_local)
        p.restore()
        p.restore()

    def _draw_mouse(self, p: QPainter, gy, rim_local: float) -> None:
        g = GAITS[self._gait]
        t = self._t - math.floor(self._t)
        sw = math.sin(2 * math.pi * t)
        fx = math.sin(2 * math.pi * (t + g.flex_ph))
        bob = -g.bob * math.sin(2 * math.pi * g.bob_f * t)
        pitch = self._pitch + 0.05 * sw
        cs, sn = math.cos(pitch), math.sin(pitch)

        def to_world(px: float, py: float) -> QPointF:
            return QPointF(px * cs - py * sn, px * sn + py * cs + bob)

        # Hips ride the body; the limbs solve in world space.
        hip_h = to_world(_HIND.x, _HIND.y - g.flex * fx)
        hip_f = to_world(_FORE.x, _FORE.y + g.flex * fx)

        fur, fur_lo = QColor(_FUR), QColor(_FUR_LO)
        pink, pink_lo = QColor(_PINK), QColor(_PINK_LO)
        body = _body_path()

        # Far side first, in shadow.
        self._draw_limb(p, _HIND, hip_h, t + g.hind[1], g.stance, gy, fur_lo, pink_lo)
        self._draw_limb(p, _FORE, hip_f, t + g.fore[1], g.stance, gy, fur_lo, pink_lo)

        pts = _tail_points(t, to_world(-19.5, -16), math.pi * 1.05 + pitch, rim_local)
        tail = QPen(pink)
        tail.setCapStyle(Qt.PenCapStyle.RoundCap)
        for i in range(1, len(pts)):
            tail.setWidthF(4.2 * (1 - i / (len(pts) + 1.2)))
            p.setPen(tail)
            p.drawLine(pts[i - 1], pts[i])

        p.save()
        p.translate(0, bob)
        p.rotate(math.degrees(pitch))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fur_lo)
        p.drawEllipse(QPointF(18.5, -27.5), 7.0, 7.4)
        p.setBrush(pink)
        p.drawEllipse(QPointF(18.8, -27.2), 4.2, 4.6)

        p.setBrush(fur)
        p.drawPath(body)

        p.save()
        p.setClipPath(body)
        p.setBrush(QColor(_BELLY))
        p.drawEllipse(QRectF(-22.0, -13.0, 44.0, 15.0))
        p.restore()

        outline = QPen(QColor(_OUTLINE))
        outline.setWidthF(1.3)
        p.setPen(outline)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(body)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(_EYE))
        p.drawEllipse(QPointF(27.5, -19.4), 1.9, 1.9)
        p.setBrush(pink)
        p.drawEllipse(QPointF(33.6, -12.6), 1.35, 1.35)
        p.restore()

        # Near side, full colour, over the body.
        self._draw_limb(p, _HIND, hip_h, t + g.hind[0], g.stance, gy, fur, pink)
        self._draw_limb(p, _FORE, hip_f, t + g.fore[0], g.stance, gy, fur, pink)

    def _draw_limb(
        self,
        p: QPainter,
        limb: _Limb,
        hip: QPointF,
        phase: float,
        stance: float,
        gy,
        color: QColor,
        foot_color: QColor,
    ) -> None:
        ankle, ang = _ankle_at(limb, limb.x, phase, stance, gy)

        # A leg cannot fold past |l2 - l1|. Keeping a margin clear of that
        # limit keeps the knee well behaved; it only bites mid-swing, where
        # the foot is airborne and free to move.
        dx, dy = ankle.x() - hip.x(), ankle.y() - hip.y()
        d = math.hypot(dx, dy)
        min_d = abs(limb.l2 - limb.l1) + (limb.l1 + limb.l2) * 0.10
        if 1e-4 < d < min_d:
            ankle = QPointF(hip.x() + (dx / d) * min_d, hip.y() + (dy / d) * min_d)

        knee = _knee(hip.x(), hip.y(), ankle.x(), ankle.y(), limb.l1, limb.l2, limb.bend)
        toe = QPointF(
            ankle.x() + math.cos(ang) * limb.lf,
            ankle.y() + math.sin(ang) * limb.lf,
        )

        pen = QPen(color)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        pen.setWidthF(limb.w_up)
        p.setPen(pen)
        p.drawLine(hip, knee)

        pen.setWidthF(limb.w_lo)
        p.setPen(pen)
        p.drawLine(knee, ankle)

        foot = QPen(foot_color)
        foot.setCapStyle(Qt.PenCapStyle.RoundCap)
        foot.setWidthF(limb.w_foot)
        p.setPen(foot)
        p.drawLine(ankle, toe)


class BusyIndicator(QWidget):
    """A :class:`MouseRunner` over a caption, for "this is working" states.

    Use where there is no meaningful percentage to report. Where there is one,
    keep the progress bar — a spinner throws away information the user wants.
    """

    def __init__(
        self,
        text: str = "Working…",
        parent: QWidget | None = None,
        *,
        size: int = 96,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._runner = MouseRunner(self)
        self._runner.setFixedSize(size, size)
        layout.addWidget(self._runner, 0, Qt.AlignmentFlag.AlignHCenter)

        self._label = QLabel(text)
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(f"color: {colors.TEXT_SECONDARY};")
        layout.addWidget(self._label)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt naming)
        self._label.setText(text)

    def text(self) -> str:
        return self._label.text()

    @property
    def runner(self) -> MouseRunner:
        return self._runner


class BusyProgress(QWidget):
    """A progress bar that shows the running mouse while indeterminate.

    Several jobs here start without knowing their total — they set a range of
    ``(0, 0)`` and switch to a real range once a frame or step count arrives.
    This swaps the mouse in for that first state and hands back to the bar for
    the second, so a percentage is never discarded once it is known.

    Implements the slice of the :class:`QProgressBar` API those call sites
    use, so it drops in where one was.
    """

    def __init__(self, parent: QWidget | None = None, *, runner_size: int = 44) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._runner = MouseRunner(self)
        self._runner.setFixedSize(runner_size, runner_size)
        layout.addWidget(self._runner)

        self._bar = QProgressBar(self)
        layout.addWidget(self._bar, 1)

        # Fixed height so the row does not jump when the two swap.
        self.setFixedHeight(runner_size)
        self._set_indeterminate(True)

    def _set_indeterminate(self, on: bool) -> None:
        self._runner.setVisible(on)
        self._bar.setVisible(not on)

    def is_indeterminate(self) -> bool:
        return self._runner.isVisible()

    # -- the QProgressBar surface these call sites use ------------------

    def setRange(self, minimum: int, maximum: int) -> None:  # noqa: N802 (Qt naming)
        self._set_indeterminate(maximum == 0)
        if maximum != 0:
            self._bar.setRange(minimum, maximum)

    def setValue(self, value: int) -> None:  # noqa: N802 (Qt naming)
        self._bar.setValue(value)

    def value(self) -> int:
        return self._bar.value()

    def maximum(self) -> int:
        return self._bar.maximum()

    def setFormat(self, fmt: str) -> None:  # noqa: N802 (Qt naming)
        self._bar.setFormat(fmt)

    @property
    def runner(self) -> MouseRunner:
        return self._runner
