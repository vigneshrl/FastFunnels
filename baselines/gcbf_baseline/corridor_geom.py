"""Corridor geometry recovered from a FastFunnels eval map.

The variants under ``eval_maps_matched/`` and ``eval_maps_heldout/`` are all the
same clean ``open_narrow`` corridor with K rectangular obstacles rasterised into
the occupancy image (see ``make_obstacle_variants.py``).  Everything this
baseline needs comes out of the ``.pgm`` plus the ``narrow_regions`` entry of the
sidecar ``<name>_obs_pos.yaml``:

  * the walkable corridor -- the free connected component containing the
    centreline, traced as a shapely polygon.  Obstacles that sit clear of the
    walls come back as holes; obstacles that were rasterised into a wall are
    absorbed into the shell, which is exactly what the f1tenth sim saw.
  * the scored narrow zone -- ``narrow_regions[0].center_m``, matched to the
    nearest centreline waypoint.

Tracing is the expensive part (a 1600x1600 label + contour pass), so the result
is memoised on a hash of the occupancy image.  Variants differ in their
obstacles and so get their own trace; the cache is what stops the runner paying
twice when it walks the splits once to size the obstacle arrays and again to
roll out.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.ops import unary_union

__all__ = ["CorridorGeom", "ZoneWindow", "load_corridor", "wall_rects"]

# f1tenth_gym binarisation: a pixel value <= 128 is occupied.
_OCC_THRESH = 128

_TRACE_CACHE: dict[str, Polygon] = {}


@dataclass(frozen=True)
class ZoneWindow:
    """The stretch of centreline a single GCBF+ episode has to cover.

    ``exit_wp`` is the waypoint the run must reach for the map to count as
    cleared.  It mirrors ``make_rebuttal_figure.py`` / ``record_patch_
    generalization.py``, which both score clearance as "reached the narrow
    waypoint + 8".
    """

    start_wp: int
    narrow_wp: int
    exit_wp: int
    goal_wp: int
    exit_clamped: bool = False
    """True when the corridor ends before ``narrow_wp + exit_offset``.

    A couple of variants put their tightest centreline point within a few
    waypoints of the end of the track, where the centreline runs out past the
    walls.  Those maps are scored on the last waypoint that is actually inside
    the corridor, which is flagged here and carried into the results so the
    shortened test is visible rather than silent.
    """


@dataclass
class CorridorGeom:
    """Walkable corridor for one eval map, in metres, in map frame."""

    name: str
    walkable: Polygon  # corridor interior; obstacle islands appear as holes
    centerline: np.ndarray  # (N, 2)
    arclength: np.ndarray  # (N,) cumulative metres along the centreline
    narrow_xy: tuple[float, float]
    gap_width_m: float
    resolution: float
    _clearances: np.ndarray | None = None

    # -- centreline helpers ------------------------------------------------

    def nearest_wp(self, x: float, y: float) -> int:
        d = np.hypot(self.centerline[:, 0] - x, self.centerline[:, 1] - y)
        return int(np.argmin(d))

    def tangent(self, wp: int) -> np.ndarray:
        """Unit tangent of the centreline at ``wp``, pointing forwards."""
        n = len(self.centerline)
        i0, i1 = max(0, wp - 1), min(n - 1, wp + 1)
        t = self.centerline[i1] - self.centerline[i0]
        norm = float(np.hypot(t[0], t[1]))
        return t / norm if norm > 1e-9 else np.array([1.0, 0.0])

    def normal(self, wp: int) -> np.ndarray:
        """Unit left-normal of the centreline at ``wp``."""
        t = self.tangent(wp)
        return np.array([-t[1], t[0]])

    @property
    def clearances(self) -> np.ndarray:
        """Per-waypoint distance to the nearest wall or obstacle, in metres."""
        if self._clearances is None:
            import shapely

            pts = shapely.points(self.centerline)
            d = shapely.distance(pts, self.walkable.boundary)
            # A waypoint outside the corridor is not usable at any width; the
            # last few waypoints of these maps run past the end of the walls.
            inside = shapely.contains_xy(
                self.walkable, self.centerline[:, 0], self.centerline[:, 1]
            )
            self._clearances = np.where(inside, d, 0.0)
        return self._clearances

    def clearance(self, wp: int) -> float:
        """Distance from centreline waypoint ``wp`` to the nearest wall."""
        return float(self.clearances[wp])

    def window(
        self,
        pre_wp: int = 10,
        exit_offset: int = 8,
        post_wp: int = 12,
        min_clearance: float = 0.0,
    ) -> ZoneWindow:
        """Waypoints bounding one episode, clamped to the drivable corridor.

        The exit test is ``narrow_wp + exit_offset``, matching
        ``make_rebuttal_figure.py``.  It is only pulled in when the corridor
        physically ends first -- see :attr:`ZoneWindow.exit_clamped`.
        """
        n = len(self.centerline)
        narrow_wp = self.nearest_wp(*self.narrow_xy)
        usable = self.clearances >= min_clearance
        if not usable.any():
            raise ValueError(f"{self.name}: no waypoint clears {min_clearance} m")

        if not usable[narrow_wp]:
            narrow_wp = int(np.flatnonzero(usable)[
                np.argmin(np.abs(np.flatnonzero(usable) - narrow_wp))
            ])

        # Grow outwards from the pinch while the corridor stays drivable.
        first = narrow_wp
        while first > 0 and usable[first - 1]:
            first -= 1
        last = narrow_wp
        while last < n - 1 and usable[last + 1]:
            last += 1

        wanted_exit = min(n - 1, narrow_wp + exit_offset)
        exit_wp = min(last, wanted_exit)
        return ZoneWindow(
            start_wp=max(first, narrow_wp - pre_wp),
            narrow_wp=narrow_wp,
            exit_wp=exit_wp,
            # The goal always sits at or beyond the exit test, so a run is never
            # asked to clear a line its own goal stops short of.
            goal_wp=max(exit_wp, min(last, narrow_wp + post_wp)),
            exit_clamped=exit_wp < wanted_exit,
        )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _trace_free_component(
    occ: np.ndarray, res: float, origin: tuple[float, float],
    seed_px: tuple[int, int], simplify_m: float,
) -> Polygon:
    """Trace the free connected component containing ``seed_px`` (col, row)."""
    free = np.logical_not(occ)
    labels, _ = ndimage.label(free)
    col, row = seed_px
    seed_label = int(labels[row, col])
    if seed_label == 0:
        raise ValueError(
            "the narrow-zone centre falls on an occupied pixel -- the map or "
            "the narrow_regions metadata is inconsistent"
        )
    mask = (labels == seed_label).astype(np.uint8)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        raise ValueError("no contour found for the corridor component")
    hierarchy = hierarchy[0]

    def to_world(c: np.ndarray) -> np.ndarray:
        pts = c.reshape(-1, 2).astype(float)  # (col, row)
        return np.stack(
            [origin[0] + (pts[:, 0] + 0.5) * res,
             origin[1] + (pts[:, 1] + 0.5) * res],
            axis=1,
        )

    outer_idx = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    shell_idx = max(outer_idx, key=lambda i: cv2.contourArea(contours[i]))
    shell = to_world(contours[shell_idx])
    holes = [
        to_world(contours[i])
        for i, h in enumerate(hierarchy)
        if h[3] == shell_idx and len(contours[i]) >= 4
    ]

    poly = Polygon(shell, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    # Contours run through the centres of free pixels, so the true wall face is
    # half a cell further out.  Growing by res/2 also shrinks the holes by the
    # same amount, which is the right correction for both.
    poly = poly.buffer(res / 2.0, join_style=2)
    poly = poly.simplify(simplify_m, preserve_topology=True)
    if isinstance(poly, Polygon):
        return poly
    return max(poly.geoms, key=lambda g: g.area)  # type: ignore[attr-defined]


def load_corridor(
    map_dir: str, name: str | None = None, simplify_m: float = 0.10
) -> CorridorGeom:
    """Load one eval-map variant as a :class:`CorridorGeom`."""
    name = name or os.path.basename(os.path.normpath(map_dir))

    with open(os.path.join(map_dir, f"{name}_map.yaml")) as f:
        spec = yaml.safe_load(f)
    res = float(spec["resolution"])
    origin = (float(spec["origin"][0]), float(spec["origin"][1]))

    # FLIP_TOP_BOTTOM so the row index grows with +y, matching the gym.
    img = Image.open(os.path.join(map_dir, spec["image"])).transpose(
        Transpose.FLIP_TOP_BOTTOM
    )
    occ = np.asarray(img).astype(np.uint8) <= _OCC_THRESH

    cl = np.loadtxt(
        os.path.join(map_dir, f"{name}_centerline.csv"), delimiter=",", comments="#"
    )[:, :2].astype(float)
    seg = np.hypot(np.diff(cl[:, 0]), np.diff(cl[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])

    with open(os.path.join(map_dir, f"{name}_obs_pos.yaml")) as f:
        meta = yaml.safe_load(f)
    region = meta["narrow_regions"][0]
    narrow_xy = (float(region["center_m"][0]), float(region["center_m"][1]))
    gap_width = float(region["gap_width_m"])

    seed_px = (
        int((narrow_xy[0] - origin[0]) / res),
        int((narrow_xy[1] - origin[1]) / res),
    )
    key = hashlib.md5(np.packbits(occ)).hexdigest() + f"|{simplify_m}"
    poly = _TRACE_CACHE.get(key)
    if poly is None:
        poly = _trace_free_component(occ, res, origin, seed_px, simplify_m)
        _TRACE_CACHE[key] = poly

    return CorridorGeom(
        name=name,
        walkable=poly,
        centerline=cl,
        arclength=s,
        narrow_xy=narrow_xy,
        gap_width_m=gap_width,
        resolution=res,
    )


# ---------------------------------------------------------------------------
# GCBF+ obstacle set
# ---------------------------------------------------------------------------


def wall_rects(
    geom: CorridorGeom,
    window: ZoneWindow,
    thickness_m: float = 0.6,
    reach_m: float = 12.0,
    wp_pad: int = 4,
) -> np.ndarray:
    """Approximate the corridor walls near ``window`` as oriented rectangles.

    GCBF+ only understands rectangles, so every boundary segment of the
    walkable polygon (shell *and* holes) that lies within ``reach_m`` of the
    scored stretch of centreline becomes one rectangle.  Each rectangle is
    pushed ``thickness_m / 2`` to its occupied side, so its inner face sits on
    the true wall rather than intruding into the corridor.

    Segments far from the window are dropped: they cost lidar raytracing time
    and no agent ever gets near them.

    Returns
    -------
    (M, 5) array of ``(cx, cy, width, height, theta)`` in metres / radians.
    """
    lo = max(0, window.start_wp - wp_pad)
    hi = min(len(geom.centerline) - 1, window.goal_wp + wp_pad)
    corridor_pts = geom.centerline[lo : hi + 1]

    rings: list[Sequence[tuple[float, float]]] = [geom.walkable.exterior.coords]
    rings += [r.coords for r in geom.walkable.interiors]

    rects: list[tuple[float, float, float, float, float]] = []
    for ring in rings:
        pts = np.asarray(ring, dtype=float)
        for a, b in zip(pts[:-1], pts[1:]):
            mid = 0.5 * (a + b)
            if np.min(np.hypot(corridor_pts[:, 0] - mid[0],
                               corridor_pts[:, 1] - mid[1])) > reach_m:
                continue
            d = b - a
            length = float(np.hypot(d[0], d[1]))
            if length < 1e-6:
                continue
            theta = float(np.arctan2(d[1], d[0]))
            # Push the rectangle to whichever side is *not* walkable, so that
            # its inner face lands on the wall instead of eating the corridor.
            nrm = np.array([-d[1], d[0]]) / length
            probe = 0.5 * geom.resolution + 1e-6
            from shapely.geometry import Point

            if geom.walkable.contains(Point(*(mid + nrm * probe))):
                nrm = -nrm
            centre = mid + nrm * (thickness_m / 2.0)
            # Overlap the joints so corners do not leak.
            rects.append(
                (float(centre[0]), float(centre[1]),
                 length + thickness_m, thickness_m, theta)
            )

    if not rects:
        raise ValueError(f"{geom.name}: no wall segments near the scored window")
    return np.asarray(rects, dtype=np.float64)


def free_space(geom: CorridorGeom, obstacles: Sequence[Polygon] = ()) -> Polygon:
    """Walkable polygon with any extra obstacle polygons removed."""
    if not obstacles:
        return geom.walkable
    return geom.walkable.difference(unary_union(list(obstacles)))
