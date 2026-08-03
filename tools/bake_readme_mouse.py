"""Bake the running-mouse loading indicator into an animated SVG for the README.

GitHub strips <script> from anything it renders, but an SVG referenced as an
image still runs the CSS animations inside it — which is what lets this work
without JavaScript. The rig is procedural, so there is nothing to hand-animate:
we sample it at N phases, emit each as a group, and cycle their opacity with a
stepped keyframe.

Geometry comes from ``glider.gui.widgets.mouse_runner`` rather than being
restated, so the banner and the in-app indicator can never drift apart.

    uv run --extra dev python tools/bake_readme_mouse.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from PyQt6.QtGui import QPainterPath  # noqa: E402

from glider.gui.widgets.mouse_runner import (  # noqa: E402
    _BELLY,
    _EYE,
    _FORE,
    _FUR,
    _FUR_LO,
    _HIND,
    _OUTLINE,
    _PINK,
    _PINK_LO,
    GAITS,
    _ankle_at,
    _body_path,
    _knee,
    _tail_points,
)

OUT = REPO / "docs-site" / "assets" / "mouse-run.svg"

# Matches MouseRunner's defaults, which are the settings signed off on.
GAIT = "bound"
SPEED = 1.30
PITCH = math.radians(-14.0)
LEAN = math.radians(0.0)

FRAMES = 20
# MouseRunner advances t by 1.05 * speed per second, so one stride is:
PERIOD = 1.0 / (1.05 * SPEED)

SIZE = 200.0
CX, CY = SIZE / 2, SIZE * 0.46
R = SIZE * 0.40
RI = R - R * 0.10
MS = R / 46.0
RIM_LOCAL = RI / MS

# A neutral that holds up on both GitHub themes; the page background behind a
# README image can be near-white or near-black and we only get one asset.
WHEEL = "#8b949e"


def n(v: float) -> str:
    """Trim coordinates — full float repr triples the file size."""
    return f"{v:.2f}".rstrip("0").rstrip(".") or "0"


def gy(x: float) -> float:
    return -(RIM_LOCAL - math.sqrt(max(0.0, RIM_LOCAL * RIM_LOCAL - x * x)))


def body_path_d() -> str:
    """Convert the shared QPainterPath to an SVG path, so there is one source."""
    path: QPainterPath = _body_path()
    out: list[str] = []
    i = 0
    while i < path.elementCount():
        e = path.elementAt(i)
        if e.type == QPainterPath.ElementType.MoveToElement:
            out.append(f"M{n(e.x)} {n(e.y)}")
            i += 1
        elif e.type == QPainterPath.ElementType.LineToElement:
            out.append(f"L{n(e.x)} {n(e.y)}")
            i += 1
        elif e.type == QPainterPath.ElementType.CurveToElement:
            c2, to = path.elementAt(i + 1), path.elementAt(i + 2)
            out.append(f"C{n(e.x)} {n(e.y)} {n(c2.x)} {n(c2.y)} {n(to.x)} {n(to.y)}")
            i += 3
        else:
            i += 1
    return " ".join(out) + " Z"


def limb_elements(limb, hip, phase: float, stance: float, color: str, foot: str) -> list[str]:
    ankle, ang = _ankle_at(limb, limb.x, phase, stance, gy)

    dx, dy = ankle.x() - hip[0], ankle.y() - hip[1]
    d = math.hypot(dx, dy)
    min_d = abs(limb.l2 - limb.l1) + (limb.l1 + limb.l2) * 0.10
    if 1e-4 < d < min_d:
        ax, ay = hip[0] + (dx / d) * min_d, hip[1] + (dy / d) * min_d
    else:
        ax, ay = ankle.x(), ankle.y()

    knee = _knee(hip[0], hip[1], ax, ay, limb.l1, limb.l2, limb.bend)
    tx = ax + math.cos(ang) * limb.lf
    ty = ay + math.sin(ang) * limb.lf

    return [
        f'<path class="s" d="M{n(hip[0])} {n(hip[1])}L{n(knee.x())} {n(knee.y())}" '
        f'stroke="{color}" stroke-width="{n(limb.w_up)}"/>',
        f'<path class="s" d="M{n(knee.x())} {n(knee.y())}L{n(ax)} {n(ay)}" '
        f'stroke="{color}" stroke-width="{n(limb.w_lo)}"/>',
        f'<path class="s" d="M{n(ax)} {n(ay)}L{n(tx)} {n(ty)}" '
        f'stroke="{foot}" stroke-width="{n(limb.w_foot)}"/>',
    ]


def mouse_elements(t: float) -> list[str]:
    g = GAITS[GAIT]
    tf = t - math.floor(t)
    sw = math.sin(2 * math.pi * tf)
    fx = math.sin(2 * math.pi * (tf + g.flex_ph))
    bob = -g.bob * math.sin(2 * math.pi * g.bob_f * tf)
    pitch = PITCH + 0.05 * sw
    cs, sn = math.cos(pitch), math.sin(pitch)

    def to_world(px: float, py: float) -> tuple[float, float]:
        return px * cs - py * sn, px * sn + py * cs + bob

    hip_h = to_world(_HIND.x, _HIND.y - g.flex * fx)
    hip_f = to_world(_FORE.x, _FORE.y + g.flex * fx)

    out: list[str] = []
    out += limb_elements(_HIND, hip_h, tf + g.hind[1], g.stance, _FUR_LO, _PINK_LO)
    out += limb_elements(_FORE, hip_f, tf + g.fore[1], g.stance, _FUR_LO, _PINK_LO)

    from PyQt6.QtCore import QPointF

    root = to_world(-19.5, -16)
    pts = _tail_points(tf, QPointF(*root), math.pi * 1.05 + pitch, RIM_LOCAL)
    for i in range(1, len(pts)):
        w = 4.2 * (1 - i / (len(pts) + 1.2))
        out.append(
            f'<path class="s" d="M{n(pts[i - 1].x())} {n(pts[i - 1].y())}'
            f'L{n(pts[i].x())} {n(pts[i].y())}" '
            f'stroke="{_PINK}" stroke-width="{n(w)}"/>'
        )

    # The body is the longest path in the file and appears twice per frame, so
    # it lives in <defs> and is referenced rather than repeated 40 times.
    deg = math.degrees(pitch)
    out.append(f'<g transform="translate(0 {n(bob)}) rotate({n(deg)})">')
    out.append(f'<ellipse cx="18.5" cy="-27.5" rx="7" ry="7.4" fill="{_FUR_LO}"/>')
    out.append(f'<ellipse cx="18.8" cy="-27.2" rx="4.2" ry="4.6" fill="{_PINK}"/>')
    out.append(f'<use href="#b" fill="{_FUR}"/>')
    out.append(
        f'<ellipse cx="0" cy="-5.5" rx="22" ry="7.5" fill="{_BELLY}" ' f'clip-path="url(#bc)"/>'
    )
    out.append(f'<use href="#b" fill="none" stroke="{_OUTLINE}" stroke-width="1.3"/>')
    out.append(f'<circle cx="27.5" cy="-19.4" r="1.9" fill="{_EYE}"/>')
    out.append(f'<circle cx="33.6" cy="-12.6" r="1.35" fill="{_PINK}"/>')
    out.append("</g>")

    out += limb_elements(_HIND, hip_h, tf + g.hind[0], g.stance, _FUR, _PINK)
    out += limb_elements(_FORE, hip_f, tf + g.fore[0], g.stance, _FUR, _PINK)
    return out


def wheel_static() -> str:
    """Rim, hub and stand. None of it moves, so it is drawn once for the whole
    animation rather than baked into every frame."""
    hub = R * 0.08
    return "".join(
        [
            f'<g transform="translate({n(CX)} {n(CY)})">',
            f'<circle class="s" cx="0" cy="0" r="{n(R)}" stroke="{WHEEL}" '
            f'stroke-width="{n(R * 0.035)}" opacity="0.85"/>',
            f'<circle class="s" cx="0" cy="0" r="{n(RI)}" stroke="{WHEEL}" '
            f'stroke-width="{n(R * 0.028)}" opacity="0.85"/>',
            f'<circle class="s" cx="0" cy="0" r="{n(hub)}" stroke="{WHEEL}" '
            f'stroke-width="{n(R * 0.028)}" opacity="0.8"/>',
            f'<path class="s" d="M0 {n(hub)}L0 {n(R * 1.18)}M{n(-R * 0.46)} '
            f'{n(R * 1.18)}L{n(R * 0.46)} {n(R * 1.18)}" stroke="{WHEEL}" '
            f'stroke-width="{n(R * 0.028)}" opacity="0.8"/>',
            "</g>",
        ]
    )


def wheel_rungs() -> str:
    """The rungs are the only part of the wheel that turns, and they turn at a
    constant rate — so one group spun by CSS replaces 34 paths per frame."""
    rungs = []
    for i in range(34):
        a = (i / 34) * math.pi * 2
        rungs.append(
            f"M{n(math.cos(a) * R)} {n(math.sin(a) * R)}"
            f"L{n(math.cos(a) * RI)} {n(math.sin(a) * RI)}"
        )
    return (
        f'<g class="rung" transform="translate({n(CX)} {n(CY)})">'
        f'<path class="s" d="{"".join(rungs)}" stroke="{WHEEL}" '
        f'stroke-width="{n(R * 0.022)}" opacity="0.6"/></g>'
    )


def build() -> str:
    step = 100.0 / FRAMES
    frames = []
    for i in range(FRAMES):
        t = i / FRAMES
        inner = [
            f'<g transform="translate({n(CX)} {n(CY)}) '
            f"rotate({n(math.degrees(-LEAN))}) "
            f'translate(0 {n(RI)}) scale({n(MS)})">'
        ]
        inner += mouse_elements(t)
        inner.append("</g>")
        frames.append(f'<g class="f f{i}">' + "".join(inner) + "</g>")

    delays = "".join(f".f{i}{{animation-delay:{n(PERIOD * i / FRAMES)}s}}" for i in range(FRAMES))
    # One full turn of the rungs takes t = 9 by the rig's own spin rate.
    rung_period = 9 * PERIOD
    body = body_path_d()

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n(SIZE)} {n(SIZE)}" \
width="{n(SIZE)}" height="{n(SIZE)}" role="img" \
aria-label="A mouse running in an exercise wheel">
<title>A mouse running in an exercise wheel</title>
<defs>
<path id="b" d="{body}"/>
<clipPath id="bc"><path d="{body}"/></clipPath>
</defs>
<style>
/* Generated by tools/bake_readme_mouse.py — edit that, not this.
   One mouse frame is visible at a time, each taking its turn on a delay.
   Note the stroke-only rule is scoped to .s: a bare `fill:none` on every
   element would beat the fill presentation attributes and hollow out the
   mouse, because CSS always outranks those. */
.s{{fill:none;stroke-linecap:round;stroke-linejoin:round}}
.f{{opacity:0;animation:cyc {n(PERIOD)}s steps(1,end) infinite}}
@keyframes cyc{{0%,{n(step)}%{{opacity:1}}{n(step + 0.001)}%,100%{{opacity:0}}}}
{delays}
.rung{{transform-origin:{n(CX)}px {n(CY)}px;animation:spin {n(rung_period)}s linear infinite}}
@keyframes spin{{to{{transform:rotate(-360deg)}}}}
@media (prefers-reduced-motion:reduce){{
.f{{animation:none}}
.f0{{opacity:1}}
.rung{{animation:none}}
}}
</style>
{wheel_static()}
{wheel_rungs()}
{"".join(frames)}
</svg>
"""


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = build()
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(svg) / 1024:.1f} KB, {FRAMES} frames)")
