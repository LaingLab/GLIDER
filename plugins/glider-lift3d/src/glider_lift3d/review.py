"""Session Review panels: the 3-D mouse beside the video, and posture traces.

Registered through the ``glider.review_panel`` entry point (see
``glider.gui.review.extensions``). Both panels share one controller that finds the
session's saved lift (:mod:`glider_lift3d.session`), offers to compute it when it is
missing, and seeks the window when a trace is clicked.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from PyQt6.QtCore import QObject, QPointF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .curvature import MAX_REAR_WEIGHT, spine_profile
from .session import GRID, LiftSession, load_session

logger = logging.getLogger(__name__)

# Height colour ramp (cm-like): floor-dark to rear-bright.
_RAMP = [(0.0, (40, 52, 90)), (1.5, (40, 120, 160)), (3.0, (90, 190, 140)), (6.0, (250, 220, 90))]
MAX_Z = 8.0
#: Spine-profile axes (cm): a long mouse's neck-to-tail run, and a high back.
PROFILE_X_CM, PROFILE_Z_CM = 10.0, 5.0


def _colour(z: float) -> QColor:
    z = min(max(z, _RAMP[0][0]), _RAMP[-1][0])
    for (z0, c0), (z1, c1) in zip(_RAMP, _RAMP[1:], strict=False):
        if z <= z1:
            t = (z - z0) / (z1 - z0)
            return QColor(*(int(a + t * (b - a)) for a, b in zip(c0, c1, strict=True)))
    return QColor(*_RAMP[-1][1])


# ----------------------------------------------------------------------------------------
# compute in the background


class _Worker(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str)  # "" on success, else the reason

    def __init__(self, video, pose_csv, config):
        super().__init__()
        self.video, self.pose_csv, self.config = video, pose_csv, config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from .pipeline import Lift3D
        from .session import compute_session

        try:
            from glider.vision.pose.dlc import from_dlc_csv

            names = from_dlc_csv(self.pose_csv).keypoint_names
            try:
                lift = Lift3D(names, self.config)
            except RuntimeError as e:  # depth model present but Ultralytics too old
                logger.warning("3-D lift without depth: %s", e)
                self.config.use_depth = False
                lift = Lift3D(names, self.config)
            done = compute_session(
                self.video,
                self.pose_csv,
                self.config,
                lift=lift,
                progress=lambda i, n: self.progress.emit(i, n) if i % 30 == 0 or i == n else None,
                should_stop=lambda: self._stop,
            )
            self.finished.emit("" if done else "stopped")
        except Exception as e:  # noqa: BLE001 - shown in the panel
            logger.exception("3-D lift failed")
            self.finished.emit(str(e))


class _Controller(QObject):
    """Which session is shown, its saved lift, and the current frame."""

    session_changed = pyqtSignal()
    frame_changed = pyqtSignal(int)
    status = pyqtSignal(str)
    busy = pyqtSignal(bool)
    progress = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.context = None
        self.lift: LiftSession | None = None
        self.frame = 0
        self._thread: QThread | None = None
        self._worker: _Worker | None = None

    # -- session ---------------------------------------------------------------------
    def set_session(self, context) -> None:
        self.context = context
        self.lift = None
        if context is not None and context.view.pose_path is not None:
            try:
                self.lift = load_session(context.view.pose_path)
            except Exception:  # noqa: BLE001
                logger.exception("could not read the saved 3-D lift")
        self.status.emit(self._describe())
        self.session_changed.emit()

    def set_frame(self, frame: int) -> None:
        self.frame = int(frame)
        self.frame_changed.emit(self.frame)

    def seek(self, frame: int) -> None:
        if self.context is not None:
            self.context.seek(int(frame))

    def ppc(self) -> float | None:
        view = self.context.view
        if view.px_per_mm:
            return float(view.px_per_mm) * 10.0
        arena = self.context.arena
        if arena is not None:
            try:
                return float(arena.px_per_cm_centre)
            except Exception:  # noqa: BLE001
                return None
        return None

    def missing(self) -> str | None:
        """Why the lift cannot be computed for this session, or None."""
        if self.context is None:
            return "No session."
        view = self.context.view
        if view.pose_path is None:
            return "No pose CSV for this session."
        if view.video_path is None:
            return "No video for this session."
        if self.ppc() is None:
            return "No scale: set the arena or scale first (Arena menu)."
        return None

    def _describe(self) -> str:
        if self.context is None:
            return ""
        if self.lift is not None:
            m = self.lift.meta
            parts = ["mask" if m.get("used_mask_model") else "no mask model"]
            parts.append("depth" if m.get("used_depth") else "no depth")
            if not m.get("has_neck"):
                parts.append("no neck keypoint: arch needs a 10-keypoint model")
            return "3-D lift: " + ", ".join(parts)
        return self.missing() or "Not computed for this session yet."

    # -- compute ---------------------------------------------------------------------
    def compute(self) -> None:
        if self._thread is not None or self.missing():
            return
        from .pipeline import Lift3DConfig

        view = self.context.view
        arena = None
        if self.context.arena is not None and view.resolution:
            try:
                arena = np.asarray(self.context.arena.outline(*view.resolution), float)
            except Exception:  # noqa: BLE001
                arena = None
        cfg = Lift3DConfig(ppc=self.ppc(), fps=view.fps or 30.0, arena_corners=arena)
        self._worker = _Worker(view.video_path, view.pose_path, cfg)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self._done)
        self.busy.emit(True)
        self.status.emit("Lifting every frame (GPU recommended)…")
        self._thread.start()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    def _done(self, error: str) -> None:
        self._thread.quit()
        self._thread.wait()
        self._thread = self._worker = None
        self.busy.emit(False)
        if error:
            self.status.emit(f"3-D lift: {error}")
            return
        self.set_session(self.context)
        self.set_frame(self.frame)
        # The epoch table and exports read the new frames.csv (glider_lift3d.metrics).
        changed = getattr(self.context, "data_changed", None)
        if changed is not None:
            changed()


# ----------------------------------------------------------------------------------------
# the 3-D view


class SurfaceView(QWidget):
    """Oblique view of a height grid, drag to turn. Painter's algorithm, no OpenGL."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.grid: np.ndarray | None = None
        self.spine: np.ndarray | None = None
        self.cell_cm = 20.0 / GRID
        self.yaw, self.pitch = math.radians(30), math.radians(35)
        self._drag: QPointF | None = None
        self.message = ""

    def set_data(self, grid, spine, cell_cm, message="") -> None:
        self.grid, self.spine, self.cell_cm, self.message = grid, spine, cell_cm, message
        self.update()

    def mousePressEvent(self, e):
        self._drag = e.position()

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        d = e.position() - self._drag
        self._drag = e.position()
        self.yaw += d.x() * 0.01
        self.pitch = min(max(self.pitch + d.y() * 0.01, math.radians(5)), math.radians(85))
        self.update()

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def _project(self, x, y, z):
        """Grid cell coords (centred) and height (cm) -> screen offset (cm units)."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        xr = x * cy - y * sy
        yr = x * sy + y * cy
        sx = xr
        sv = yr * math.sin(self.pitch) - z * math.cos(self.pitch)
        depth = yr * math.cos(self.pitch) + z * math.sin(self.pitch)
        return sx, sv, depth

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(18, 22, 30))
        g = self.grid
        if g is None or not np.isfinite(g).any():
            p.setPen(QColor(150, 160, 175))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message or "No mouse")
            return
        n = g.shape[0]
        c = self.cell_cm
        half = n / 2
        w, h = self.width(), self.height()
        scale = min(w, h) / (n * c * 0.9)

        def to_screen(i, j, z):
            sx, sv, d = self._project((j - half) * c, (i - half) * c, z)
            return QPointF(w / 2 + sx * scale, h * 0.55 + sv * scale), d

        # Floor outline.
        corners = [to_screen(i, j, 0)[0] for i, j in ((0, 0), (0, n), (n, n), (n, 0))]
        p.setPen(QPen(QColor(60, 70, 88), 1))
        p.setBrush(QColor(28, 34, 46))
        p.drawPolygon(QPolygonF(corners))

        # Heights at cell corners: the mean of the (up to four) cells sharing each
        # corner, so neighbouring cells meet instead of floating as separate steps.
        filled = np.where(np.isfinite(g), g, 0.0)
        count = np.isfinite(g).astype(float)
        pad_v = np.pad(filled, 1)
        pad_c = np.pad(count, 1)
        vsum = pad_v[:-1, :-1] + pad_v[1:, :-1] + pad_v[:-1, 1:] + pad_v[1:, 1:]
        vcnt = pad_c[:-1, :-1] + pad_c[1:, :-1] + pad_c[:-1, 1:] + pad_c[1:, 1:]
        vz = np.where(vcnt > 0, vsum / np.maximum(vcnt, 1), 0.0)  # (n+1, n+1)

        quads = []
        for i, j in zip(*np.nonzero(np.isfinite(g)), strict=True):
            pts = []
            depth = 0.0
            for di, dj in ((0, 0), (0, 1), (1, 1), (1, 0)):
                q, d = to_screen(i + di, j + dj, float(vz[i + di, j + dj]))
                pts.append(q)
                depth += d
            quads.append((depth, float(g[i, j]), pts))
        quads.sort(key=lambda t: t[0])
        p.setPen(QPen(QColor(18, 22, 30, 90), 0.5))
        for _d, zz, pts in quads:
            p.setBrush(_colour(zz))
            p.drawPolygon(QPolygonF(pts))

        if self.spine is not None and np.isfinite(self.spine).all():
            path = QPainterPath()
            for k, (gx, gy) in enumerate(self.spine):
                i, j = int(np.clip(round(gy), 0, n)), int(np.clip(round(gx), 0, n))
                q, _ = to_screen(gy, gx, float(vz[i, j]) + 0.05)
                path.moveTo(q) if k == 0 else path.lineTo(q)
            p.setPen(QPen(QColor(255, 90, 90), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)


# ----------------------------------------------------------------------------------------
# panels


class SurfacePanel:
    title = "3-D mouse"
    placement = "side"

    def __init__(self, ctl: _Controller):
        import pyqtgraph as pg

        self.ctl = ctl
        self._w = QWidget()
        col = QVBoxLayout(self._w)
        col.setContentsMargins(6, 6, 6, 6)
        self.status = QLabel()
        self.status.setWordWrap(True)
        col.addWidget(self.status)
        row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute 3-D for this session")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setVisible(False)
        row.addWidget(self.compute_btn)
        row.addWidget(self.stop_btn)
        col.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        col.addWidget(self.bar)
        self.view = SurfaceView()
        col.addWidget(self.view, 3)
        self.readout = QLabel()
        col.addWidget(self.readout)
        self.profile = pg.PlotWidget()
        self.profile.setMinimumHeight(140)
        self.profile.setLabel("bottom", "along spine, neck → tail base (cm)")
        self.profile.setLabel("left", "height (cm)")
        # Fixed axes at true proportions, so frames compare by eye: a hunch
        # shows as the line bowing up, not as the axes rescaling.
        self.profile.setAspectLocked(True)
        self.profile.setXRange(0, PROFILE_X_CM, padding=0.02)
        self.profile.setYRange(0, PROFILE_Z_CM, padding=0.02)
        self.profile.disableAutoRange()
        self.profile.setMouseEnabled(x=False, y=False)
        self._line = self.profile.plot([], [], pen=pg.mkPen((255, 90, 90), width=2))
        col.addWidget(self.profile, 2)

        self.compute_btn.clicked.connect(ctl.compute)
        self.stop_btn.clicked.connect(ctl.stop)
        ctl.status.connect(self.status.setText)
        ctl.busy.connect(self._busy)
        ctl.progress.connect(self._progress)
        ctl.session_changed.connect(self._refresh_buttons)
        ctl.frame_changed.connect(self._show)

    def widget(self):
        return self._w

    def set_session(self, context):
        self.ctl.set_session(context)

    def set_frame(self, frame):
        self.ctl.set_frame(frame)

    def _busy(self, on):
        self.bar.setVisible(on)
        self.stop_btn.setVisible(on)
        self.compute_btn.setEnabled(not on)

    def _progress(self, i, n):
        self.bar.setMaximum(n)
        self.bar.setValue(i)

    def _refresh_buttons(self):
        ctl = self.ctl
        self.compute_btn.setVisible(ctl.context is not None and ctl.lift is None)
        self.compute_btn.setEnabled(ctl.missing() is None)
        self._show(ctl.frame)

    def _show(self, frame):
        lift = self.ctl.lift
        if lift is None or not 0 <= frame < lift.n_frames:
            self.view.set_data(None, None, 20.0 / GRID, "" if lift else "Not computed")
            self.readout.setText("")
            self._line.setData([], [])
            return
        grid = np.asarray(lift.surface[frame], float)
        spine = np.asarray(lift.spine[frame], float)
        self.view.set_data(grid, spine, lift.cell_cm)
        row = lift.table.loc[frame] if frame in lift.table.index else None
        self.readout.setText(_readout(row))
        if np.isfinite(spine).all():
            s, z = spine_profile(grid, spine, 1.0 / lift.cell_cm)
            self._line.setData(s, z, connect="finite")
        else:
            self._line.setData([], [])


def _readout(row) -> str:
    if row is None:
        return ""

    def f(key, fmt):
        v = row.get(key)
        return "–" if v is None or not np.isfinite(v) else fmt.format(v)

    rearing = (row.get("rear_weight") or 0) >= MAX_REAR_WEIGHT
    arch = (
        "arch e  – (rearing)"
        if rearing
        else f"arch e  surface {f('ecc_surface', '{:.3f}')}  depth {f('ecc_depth', '{:.3f}')}"
    )
    return arch + (
        f"   ·   body {f('body_length_cm', '{:.1f}')} cm ({f('body_ratio', '{:.2f}')})"
        f"   ·   rear {f('rear_weight', '{:.2f}')}"
    )


class PostureTracePanel:
    """Arch eccentricity and body length over the session; click to seek."""

    title = "Posture (3-D)"
    placement = "timeline"
    SERIES = (
        ("ecc_surface", "arch e (surface)", (255, 90, 90)),
        ("ecc_depth", "arch e (depth only)", (90, 180, 255)),
        ("body_ratio", "body length / full", (200, 200, 200)),
        ("rear_weight", "rear weight", (250, 210, 90)),
    )

    def __init__(self, ctl: _Controller):
        import pyqtgraph as pg

        self.ctl = ctl
        self._w = pg.PlotWidget()
        self._w.addLegend(offset=(10, 5))
        self._w.setLabel("bottom", "frame")
        self._w.setYRange(0, 1.2)
        self._w.setClipToView(True)
        self._w.setDownsampling(auto=True, mode="peak")
        self.curves = {
            key: self._w.plot([], [], pen=pg.mkPen(colour, width=1), name=label)
            for key, label, colour in self.SERIES
        }
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 140)))
        self._w.addItem(self.cursor)
        self._w.scene().sigMouseClicked.connect(self._clicked)
        ctl.session_changed.connect(self._load)
        ctl.frame_changed.connect(lambda f: self.cursor.setValue(f))

    def widget(self):
        return self._w

    def set_session(self, context):
        pass  # the side panel's controller already received it

    def set_frame(self, frame):
        pass

    def _load(self):
        lift = self.ctl.lift
        for key, curve in self.curves.items():
            if lift is None or key not in lift.table:
                curve.setData([], [])
            else:
                col = lift.table[key]
                curve.setData(col.index.to_numpy(), col.to_numpy(dtype=float), connect="finite")

    def _clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = self._w.getPlotItem().vb.mapSceneToView(event.scenePos()).x()
        self.ctl.seek(round(x))


def panels():
    """Entry point: the two panels, sharing one controller."""
    ctl = _Controller()
    return [SurfacePanel(ctl), PostureTracePanel(ctl)]


__all__ = ["panels", "SurfaceView", "SurfacePanel", "PostureTracePanel"]
