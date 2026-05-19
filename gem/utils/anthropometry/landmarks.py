"""
SOMA landmark definitions for body measurements.

Landmarks are resolved dynamically from the T-pose mesh and joint positions, so
they work correctly for any body shape and both LOD levels.

Landmark specs
--------------
MaxY()           → vertex with the highest Y coordinate (head top)
MinY()           → vertex with the lowest Y coordinate (heel)
Ring(joint, n)   → list of n vertex indices evenly distributed angularly around
                   the body's Y-axis at the height of the given joint.

GEM-X joint indices (0-indexed, joint 0 = Hips):
  0 Hips   1 Spine1   2 Spine2   3 Chest   6 Head   7 HeadEnd
  68 LeftLeg  70 LeftFoot  73 RightLeg  75 RightFoot

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
    """Vertex with the maximum Y coordinate (e.g. crown of the head)."""


@dataclass
class MinY:
    """Vertex with the minimum Y coordinate (e.g. heel)."""


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
        Number of vertices in the ring (more → smoother geodesic circumference).
    band_width:
        Half-width of the Y-band used to candidate vertices (metres).
    """

    joint_idx: int
    y_offset: float = 0.0
    n: int = 8
    band_width: float = 0.04


# ---------------------------------------------------------------------------
# Default landmark specs
# ---------------------------------------------------------------------------

LANDMARK_SPECS: dict[str, MaxY | MinY | Ring] = {
    # --- point landmarks ---
    "HEAD_TOP": MaxY(),
    "HEEL": MinY(),
    # --- circumference rings ---
    # Chest: at the Chest joint (GEM-X joint 3)
    "CHEST_RING": Ring(joint_idx=3, y_offset=0.0, n=8),
    # Waist: at the Spine1 joint (GEM-X joint 1), the narrowest torso cross-section
    "WAIST_RING": Ring(joint_idx=1, y_offset=0.0, n=8),
    # Hip: slightly below the Hips joint (GEM-X joint 0) to hit the widest point
    "HIP_RING": Ring(joint_idx=0, y_offset=-0.05, n=8),
}


# ---------------------------------------------------------------------------
# Landmark resolution
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


def find_landmarks(
    verts: np.ndarray,
    joints: np.ndarray,
    specs: dict[str, MaxY | MinY | Ring] | None = None,
) -> dict[str, int | list[int]]:
    """
    Resolve landmark specs to vertex indices for the given T-pose mesh.

    Parameters
    ----------
    verts:
        (V, 3) float32/float64 vertex positions in world space (metres, Y-up).
    joints:
        (77, 3) float32/float64 joint positions in world space.
    specs:
        Landmark spec dict.  Defaults to ``LANDMARK_SPECS``.

    Returns
    -------
    dict mapping landmark name → int (single vertex) or list[int] (ring).
    """
    if specs is None:
        specs = LANDMARK_SPECS

    result: dict[str, int | list[int]] = {}

    for name, spec in specs.items():
        if isinstance(spec, MaxY):
            result[name] = int(np.argmax(verts[:, 1]))

        elif isinstance(spec, MinY):
            result[name] = int(np.argmin(verts[:, 1]))

        elif isinstance(spec, Ring):
            target_y = float(joints[spec.joint_idx, 1]) + spec.y_offset
            result[name] = _find_ring(verts, target_y, spec.n, spec.band_width)

        else:
            raise TypeError(f"Unknown landmark spec type: {type(spec)}")

    return result
