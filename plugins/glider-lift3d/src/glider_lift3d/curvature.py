"""Back-arch eccentricity from the lifted surface, after Mittal et al. 2016.

Mittal et al. (Blood Cells Mol Dis 57:58-66) scored pain in sickle mice from a side-view
photo: points along the arch of the spine, from the neck to just before the tail base,
were fitted with an ellipse, and its eccentricity ``e = sqrt(a^2 - b^2) / a`` was the
curvature measure. Lower ``e`` = rounder, more hunched back = more pain. Body length was
the companion measure (and correlated better with grimace: R -0.72 vs -0.58).

Here there is no side camera. The side view is read off the lifted height surface:
heights sampled along the spine polyline (neck -> withers -> centre -> lumbar -> tail
base, from a 10-keypoint model) give ``(distance along spine, height)``, the profile
the paper's points traced by hand. Two profiles are fitted:

* ``surface``: the lift's full surface. For low poses that is 75% an outline dome whose
  height is set by how short the body looks, so this eccentricity largely re-states
  body length. Shown because it is what the 3-D view draws.
* ``depth``: the depth model alone, the only part that sees the back. It under-reads
  height on the phantoms, so its absolute value is not cm, but it is the one that can
  disagree with body length. Compare the two before trusting either.

Uniform scale and translation do not change eccentricity, so the profile's units
(cm-like estimates) do not matter as long as both axes share them.
"""

from __future__ import annotations

import numpy as np

N_PROFILE = 16  # points along the spine, as hand-placed points along an arch would be
MIN_POINTS = 6
SAMPLE_RADIUS_CM = 0.35  # top of the back within this of the spine line


def resample_polyline(points: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points evenly spaced along a polyline, and their arc length from the start."""
    pts = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    if cum[-1] <= 0:
        return np.repeat(pts[:1], n, axis=0), np.zeros(n)
    s = np.linspace(0.0, cum[-1], n)
    out = np.c_[np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])]
    return out, s


def spine_profile(
    height: np.ndarray, spine_c: np.ndarray, px_per_cm: float, n: int = N_PROFILE
) -> tuple[np.ndarray, np.ndarray]:
    """``(s_cm, z)`` along neck -> tail base; samples with no surface underneath dropped.

    ``z`` is the 75th percentile of the surface within :data:`SAMPLE_RADIUS_CM` of each
    point: the top of the back, robust to a pixel or two of mask edge.
    """
    pts, s_px = resample_polyline(spine_c, n)
    r = max(SAMPLE_RADIUS_CM * px_per_cm, 1.0)
    h, w = height.shape
    zs = np.full(n, np.nan)
    for i, (u, v) in enumerate(pts):
        y0, y1 = int(max(v - r, 0)), int(min(v + r + 1, h))
        x0, x1 = int(max(u - r, 0)), int(min(u + r + 1, w))
        if y0 >= y1 or x0 >= x1:
            continue
        gv, gu = np.mgrid[y0:y1, x0:x1]
        patch = height[y0:y1, x0:x1][(gu - u) ** 2 + (gv - v) ** 2 <= r * r]
        patch = patch[np.isfinite(patch)]
        if patch.size:
            zs[i] = np.percentile(patch, 75)
    ok = np.isfinite(zs)
    return s_px[ok] / px_per_cm, zs[ok]


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Semi-axes ``(a, b)``, ``a >= b``, of the least-squares ellipse through the points."""
    params = fit_ellipse_params(x, y)
    return None if params is None else (params[2], params[3])


def fit_ellipse_params(x: np.ndarray, y: np.ndarray):
    """``(cx, cy, a, b, theta)`` of the least-squares ellipse through the points.

    ``a >= b`` are the semi-axes, ``theta`` the major axis's angle (radians).

    Direct least squares (Fitzgibbon et al. 1999) in Halir & Flusser's (1998) stable
    form: it always returns an ellipse, so a partial arc (as here) or a near-straight
    profile gives a long thin one (``e`` near 1) rather than a hyperbola. None when the
    points cannot constrain a conic.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 5:
        return None
    # Uniform normalisation for conditioning; eccentricity is scale-invariant.
    cx, cy = x.mean(), y.mean()
    scale = max(float(np.sqrt(((x - cx) ** 2 + (y - cy) ** 2).mean())), 1e-9)
    x, y = (x - cx) / scale, (y - cy) / scale
    d1 = np.c_[x * x, x * y, y * y]
    d2 = np.c_[x, y, np.ones_like(x)]
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    try:
        t = -np.linalg.solve(s3, s2.T)
    except np.linalg.LinAlgError:
        return None
    m = s1 + s2 @ t
    m = np.array([m[2] / 2, -m[1], m[0] / 2])
    try:
        vals, vecs = np.linalg.eig(m)
    except np.linalg.LinAlgError:
        return None
    vecs = np.real(vecs)
    cond = 4 * vecs[0] * vecs[2] - vecs[1] ** 2
    good = np.nonzero(cond > 0)[0]
    if good.size == 0:
        return None
    a1 = vecs[:, good[0]]
    A, B, C = a1  # noqa: N806 - conic coefficients
    D, E, F = t @ a1  # noqa: N806
    conic = np.array([[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]])
    quad = conic[:2, :2]
    det_q = np.linalg.det(quad)
    if det_q <= 0:
        return None
    k = -np.linalg.det(conic) / det_q
    lam, vec = np.linalg.eigh(quad)
    axes_sq = k / lam
    if np.any(axes_sq <= 0):
        return None
    major = int(np.argmax(axes_sq))
    a, b = np.sqrt(axes_sq[major]) * scale, np.sqrt(axes_sq[1 - major]) * scale
    centre = np.linalg.solve(2 * quad, -np.array([D, E]))
    theta = float(np.arctan2(vec[1, major], vec[0, major]))
    return (
        float(centre[0] * scale + cx),
        float(centre[1] * scale + cy),
        float(a),
        float(b),
        theta,
    )


def ellipse_points(params, n: int = 120) -> np.ndarray:
    """``(n, 2)`` points around a fitted ellipse, for drawing."""
    cx, cy, a, b, th = params
    t = np.linspace(0, 2 * np.pi, n)
    c, s = np.cos(th), np.sin(th)
    return np.c_[
        cx + a * np.cos(t) * c - b * np.sin(t) * s, cy + a * np.cos(t) * s + b * np.sin(t) * c
    ]


def eccentricity(a: float, b: float) -> float:
    return float(np.sqrt(max(a * a - b * b, 0.0)) / a) if a > 0 else float("nan")


def profile_eccentricity(s: np.ndarray, z: np.ndarray) -> float:
    if len(s) < MIN_POINTS:
        return float("nan")
    axes = fit_ellipse(s, z)
    return float("nan") if axes is None else eccentricity(*axes)


#: Rear weight at and above which eccentricity is left blank. Mittal's measure is the arch
#: of a mouse on four paws; a rear's profile is a ramp, which fits as a needle (e ~ 1).
MAX_REAR_WEIGHT = 0.5


def back_arch(frame, crop_px_per_cm: float) -> dict:
    """Per-frame arch measures from a :class:`~glider_lift3d.pipeline.Lift3DFrame`.

    All NaN when the model has no neck keypoint (``frame.spine_c is None``);
    eccentricities NaN while rearing (rear weight >= :data:`MAX_REAR_WEIGHT`).
    """
    out = {
        "ecc_surface": float("nan"),
        "ecc_depth": float("nan"),
        "spine_length_cm": float("nan"),
        "arch_peak": float("nan"),
    }
    if frame is None or frame.spine_c is None:
        return out
    seg = np.linalg.norm(np.diff(frame.spine_c, axis=0), axis=1)
    out["spine_length_cm"] = float(seg.sum() / crop_px_per_cm)
    s, z = spine_profile(frame.height, frame.spine_c, crop_px_per_cm)
    if len(z):
        out["arch_peak"] = float(z.max())
    if frame.rear_weight >= MAX_REAR_WEIGHT:
        return out
    out["ecc_surface"] = profile_eccentricity(s, z)
    if frame.rel is not None and frame.depth_scale:
        depth = np.where(frame.mask, np.clip(frame.rel, 0, None) * frame.depth_scale, np.nan)
        sd, zd = spine_profile(depth, frame.spine_c, crop_px_per_cm)
        out["ecc_depth"] = profile_eccentricity(sd, zd)
    return out
