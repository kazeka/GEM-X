"""
SOMA landmark definitions for body measurements.

Landmarks are resolved dynamically from the T-pose mesh and joint positions, so
they work correctly for any body shape and both LOD levels.

Landmark specs
--------------
MaxY()                  → vertex with the highest Y coordinate (head top)
MinY()                  → vertex with the lowest Y coordinate (heel)
Ring(joint, n)          → list of n vertex indices evenly distributed angularly around
                          the body's Y-axis at the height of the given joint.
AnchorVerticesRing(...)  → same as Ring but Y is pinned to the mean Y of given vertex IDs.
ScanNarrowest(...)       → ring at the Y with the *minimum* horizontal cross-section
                          perimeter between two joints (waist).
ScanWidest(...)          → ring at the Y with the *maximum* horizontal cross-section
                          perimeter between two joints (hips).

GEM-X joint indices (0-indexed, joint 0 = Hips):
  0 = Hips          1 = Spine1       2 = Spine2      3 = Chest
  4 = Neck          5 = Head         ...  (77 total)

To add a new landmark:
  1. Add an entry to LANDMARK_SPECS with any name.
  2. Use that name in a Measurement in measurements.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Landmark spec dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MaxY:
    """Vertex with the max Y coordinate (e.g. crown of the head)."""


@dataclass
class MinY:
    """Vertex with the min Y coordinate (e.g. heel)."""


@dataclass
class Ring:
    """
    N vertices evenly distributed by angle around the body Y-axis at a given height.

    Parameters
    ----------
    joint_idx:
        GEM-X joint index (0 = Hips). The ring is placed at the Y coordinate of
        this joint plus ``y_offset``.
    y_offset:
        Signed vertical offset in metres applied to the joint's Y position.
    n:
        Number of vertices in the ring.
    band_width:
        Half-width of the Y-band used to candidate vertices (metres).
    """

    joint_idx: int
    y_offset: float = 0.0
    n: int = 8
    band_width: float = 0.04


@dataclass
class AnchorVerticesRing:
    """
    Like ``Ring`` but the Y level is pinned to the mean Y of specific vertex IDs.

    Use this when known anatomical vertices should define the measurement plane
    (e.g. chest guidance vertices 84, 1561, 3765 on the SOMA-X mesh).

    Parameters
    ----------
    vertex_ids:
        Vertex indices whose mean Y defines the measurement plane.
    y_offset:
        Additional vertical offset in metres.
    n:
        Number of ring vertices for visualisation.
    band_width:
        Half-width of the Y-band for candidate vertices (metres).
    """

    vertex_ids: list[int] = field(default_factory=list)
    y_offset: float = 0.0
    n: int = 8
    band_width: float = 0.04


@dataclass
class ScanNarrowest:
    """
    Place the ring at the Y level with the *minimum* horizontal cross-section
    perimeter between ``joint_bottom`` and ``joint_top``.

    Useful for finding the anatomical waist (narrowest torso cross-section).

    Parameters
    ----------
    joint_bottom / joint_top:
        GEM-X joint indices that bound the scan range (Y of bottom < Y of top).
    margin:
        Fraction of the range to trim from each end before scanning (avoids
        measuring at the joints themselves where the cross-section is ill-defined).
    steps:
        Number of equally-spaced Y levels to evaluate.
    n:
        Number of ring vertices for visualisation.
    band_width:
        Half-width of the Y-band for candidate vertices (metres).
    """

    joint_bottom: int
    joint_top: int
    margin: float = 0.05
    steps: int = 60
    n: int = 8
    band_width: float = 0.04


@dataclass
class ScanWidest:
    """
    Place the ring at the Y level with the *maximum* horizontal cross-section
    perimeter between ``joint_bottom`` and ``joint_top``.

    Useful for finding the anatomical hip (widest cross-section around the buttocks).

    Parameters
    ----------
    joint_bottom / joint_top:
        GEM-X joint indices that bound the scan range.
    margin:
        Fraction of the range to trim from each end before scanning.
    steps:
        Number of equally-spaced Y levels to evaluate.
    n:
        Number of ring vertices for visualisation.
    band_width:
        Half-width of the Y-band for candidate vertices (metres).
    """

    joint_bottom: int
    joint_top: int
    margin: float = 0.05
    steps: int = 60
    n: int = 8
    band_width: float = 0.04


# ---------------------------------------------------------------------------
# Default landmark specs
# ---------------------------------------------------------------------------

LANDMARK_SPECS: dict[str, MaxY | MinY | Ring | AnchorVerticesRing | ScanNarrowest | ScanWidest] = {
    # --- point landmarks ---
    "HEAD_TOP": MaxY(),
    "HEEL": MinY(),
    # --- circumference rings ---
    # Chest: anchored to known SOMA-X guidance vertices at the chest level.
    "CHEST_RING": AnchorVerticesRing(vertex_ids=[84, 1561, 3765], n=8),
    # Waist: narrowest cross-section between Spine1 (joint 1) and Chest (joint 3).
    "WAIST_RING": ScanNarrowest(joint_bottom=1, joint_top=3, n=8),
    # Hips: widest cross-section between Hips (joint 0) and Spine1 (joint 1).
    "HIP_RING": ScanWidest(joint_bottom=0, joint_top=1, n=8),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_ring(
    verts: np.ndarray,
    target_y: float,
    n: int,
    band_width: float,
) -> list[int]:
    """
    Return n vertex indices at angles 0, 2π/n, …, 2π(n-1)/n around the
    body's Y-axis at height *target_y*.

    For each angular direction the outermost vertex (largest projection onto
    the XZ direction vector) within the Y-band is chosen.
    """
    mask = np.abs(verts[:, 1] - target_y) < band_width
    cand = np.where(mask)[0]

    if len(cand) < n:
        # Fallback: take the closest vertices by Y distance
        order = np.argsort(np.abs(verts[:, 1] - target_y))
        cand = order[: max(n * 4, 64)]

    xz = verts[cand][:, [0, 2]]  # (M, 2) — project to horizontal plane

    result: list[int] = []
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        direction = np.array([np.cos(angle), np.sin(angle)])  # XZ direction
        projections = xz @ direction  # (M,) — signed distance along ray
        result.append(int(cand[int(np.argmax(projections))]))

    return result


def _mesh_slice_perimeter(
    verts: np.ndarray,
    faces: np.ndarray,
    y_level: float,
) -> float:
    """
    Return the total perimeter of the mesh cross-section at ``y = y_level``.

    Each triangular face that straddles the horizontal plane contributes one
    line segment (the intersection of the triangle with the plane).  Summing
    those segment lengths gives the correct horizontal circumference — the
    physical equivalent of a measuring tape held level around the body.
    """
    y = verts[:, 1]
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    y0, y1, y2 = y[v0], y[v1], y[v2]

    ymin = np.minimum(np.minimum(y0, y1), y2)
    ymax = np.maximum(np.maximum(y0, y1), y2)
    crossing = np.where((ymin < y_level) & (ymax > y_level))[0]

    total = 0.0
    for fi in crossing:
        a, b, c = faces[fi]
        pts: list[np.ndarray] = []
        for p, q in ((a, b), (b, c), (c, a)):
            yp, yq = y[p], y[q]
            if (yp < y_level) != (yq < y_level):
                t = (y_level - yp) / (yq - yp)
                pts.append(verts[p] + t * (verts[q] - verts[p]))
        if len(pts) == 2:
            total += float(np.linalg.norm(pts[1] - pts[0]))

    return total


# ---------------------------------------------------------------------------
# Landmark resolution
# ---------------------------------------------------------------------------


def find_landmarks(
    verts: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray | None = None,
    specs: dict[str, MaxY | MinY | Ring | AnchorVerticesRing | ScanNarrowest | ScanWidest] | None = None,
) -> tuple[dict[str, int | list[int]], dict[str, float]]:
    """
    Resolve landmark specs to vertex indices and plane heights for the given T-pose mesh.

    Parameters
    ----------
    verts:
        (V, 3) float32/float64 vertex positions in world space (metres, Y-up).
    joints:
        (77, 3) float32/float64 joint positions in world space.
    faces:
        (F, 3) int32 face indices.  Required for ``ScanNarrowest`` / ``ScanWidest``.
    specs:
        Landmark spec dict.  Defaults to ``LANDMARK_SPECS``.

    Returns
    -------
    landmarks:
        dict mapping landmark name → int (single vertex) or list[int] (ring).
    plane_heights:
        dict mapping landmark name → float Y level.  Only populated for ring-type
        landmarks; used by ``MeasurementType.PLANAR`` in the measurer.
    """
    if specs is None:
        specs = LANDMARK_SPECS

    landmarks: dict[str, int | list[int]] = {}
    plane_heights: dict[str, float] = {}

    for name, spec in specs.items():
        if isinstance(spec, MaxY):
            landmarks[name] = int(np.argmax(verts[:, 1]))

        elif isinstance(spec, MinY):
            landmarks[name] = int(np.argmin(verts[:, 1]))

        elif isinstance(spec, Ring):
            target_y = float(joints[spec.joint_idx, 1]) + spec.y_offset
            landmarks[name] = _find_ring(verts, target_y, spec.n, spec.band_width)
            plane_heights[name] = target_y

        elif isinstance(spec, AnchorVerticesRing):
            anchor_ys = verts[spec.vertex_ids, 1]
            target_y = float(np.mean(anchor_ys)) + spec.y_offset
            landmarks[name] = _find_ring(verts, target_y, spec.n, spec.band_width)
            plane_heights[name] = target_y

        elif isinstance(spec, (ScanNarrowest, ScanWidest)):
            if faces is None:
                raise ValueError(
                    f"Landmark '{name}' ({type(spec).__name__}) requires mesh faces "
                    "for cross-section scanning.  Pass faces= to find_landmarks()."
                )
            y_bot = float(joints[spec.joint_bottom, 1])
            y_top = float(joints[spec.joint_top, 1])
            # ensure bot < top regardless of joint ordering
            y_lo, y_hi = min(y_bot, y_top), max(y_bot, y_top)
            span = y_hi - y_lo
            y_lo += span * spec.margin
            y_hi -= span * spec.margin
            ys = np.linspace(y_lo, y_hi, spec.steps)
            perims = np.array([_mesh_slice_perimeter(verts, faces, y) for y in ys])
            if isinstance(spec, ScanNarrowest):
                target_y = float(ys[int(np.argmin(perims))])
            else:
                target_y = float(ys[int(np.argmax(perims))])
            landmarks[name] = _find_ring(verts, target_y, spec.n, spec.band_width)
            plane_heights[name] = target_y

        else:
            raise TypeError(f"Unknown landmark spec type: {type(spec)}")

    return landmarks, plane_heights
