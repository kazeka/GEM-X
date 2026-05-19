"""
Body measurement definitions and measurer for the SOMA body model.

To add a new measurement
------------------------
1. (opt) add any new landmark to ``LANDMARK_SPECS`` in landmarks.py.
2. Append a ``Measurement`` entry to ``MEASUREMENTS``.

Measurement types
-----------------
EUCLIDEAN
    Straight-line distance between two landmark vertices (||A - B||).
    Use for: height, limb lengths.
GEODESIC
    Sum of shortest surface distances between consecutive landmarks.
    Use for: surface-following paths.
PLANAR
    Perimeter of the horizontal (y = constant) mesh cross-section at the
    plane height stored for the named ring landmark.  This is the physically
    correct model for a measuring tape held level around the body.
    Use for: chest / waist / hip circumferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from gem.utils.anthropometry.landmarks import LANDMARK_SPECS, find_landmarks, _mesh_slice_perimeter

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class MeasurementType(Enum):
    EUCLIDEAN = "euclidean"
    GEODESIC = "geodesic"
    PLANAR = "planar"


@dataclass
class Measurement:
    """
    Single body measurement definition.

    Parameters
    ----------
    name:
        Human-readable name (e.g. ``"chest circumference"``).
    type:
        ``EUCLIDEAN``, ``GEODESIC``, or ``PLANAR``.
    landmarks:
        For EUCLIDEAN: exactly two landmark names, e.g. ``["HEAD_TOP", "HEEL"]``.
        For GEODESIC: one or more landmark names.  Each name that resolves to a
        *list* of indices (Ring) is expanded in order, so
        ``["CHEST_RING"]`` expands to all 8 ring vertices.
        For PLANAR: exactly one ring landmark name whose ``plane_height`` defines
        the cross-section Y level.
    closed:
        If True (circumferences), append ``landmarks[0]`` at the end so the
        geodesic loop closes.  Not used for PLANAR.
    """

    name: str
    type: MeasurementType
    landmarks: list[str]
    closed: bool = False


# ---------------------------------------------------------------------------
# Default measurement list  (edit or extend freely)
# ---------------------------------------------------------------------------

MEASUREMENTS: list[Measurement] = [
    Measurement(
        "height",
        MeasurementType.EUCLIDEAN,
        ["HEAD_TOP", "HEEL"],
    ),
    Measurement(
        "chest circumference",
        MeasurementType.PLANAR,
        ["CHEST_RING"],
    ),
    Measurement(
        "waist circumference",
        MeasurementType.PLANAR,
        ["WAIST_RING"],
    ),
    Measurement(
        "hip circumference",
        MeasurementType.PLANAR,
        ["HIP_RING"],
    ),
]


# ---------------------------------------------------------------------------
# Measurer
# ---------------------------------------------------------------------------


class SomaMeasurer:
    """
    Compute body measurements from a SOMA T-pose mesh.

    Parameters
    ----------
    verts:
        (V, 3) numpy array of vertex positions in metres (Y-up world space).
    faces:
        (F, 3) int32 numpy array of triangle face indices.
    joints:
        (77, 3) numpy array of joint positions in metres.
    landmark_specs:
        Override the default landmark spec dict (``LANDMARK_SPECS``).
    """

    def __init__(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        joints: np.ndarray,
        landmark_specs=None,
    ) -> None:
        self.verts = np.asarray(verts, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.joints = np.asarray(joints, dtype=np.float64)
        self.landmarks, self.plane_heights = find_landmarks(
            self.verts, self.joints, self.faces, landmark_specs
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure_all(self, measurements: list[Measurement] | None = None) -> dict[str, float]:
        """Return {measurement_name: value_in_metres} for every measurement."""
        if measurements is None:
            measurements = MEASUREMENTS
        return {m.name: self._measure(m) for m in measurements}

    def resolve_indices(self, landmark_names: list[str]) -> list[int]:
        """Expand a list of landmark names to a flat list of vertex indices."""
        indices: list[int] = []
        for name in landmark_names:
            v = self.landmarks[name]
            if isinstance(v, list):
                indices.extend(v)
            else:
                indices.append(v)
        return indices

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _measure(self, m: Measurement) -> float:
        if m.type == MeasurementType.EUCLIDEAN:
            indices = self.resolve_indices(m.landmarks)
            if len(indices) != 2:
                raise ValueError(
                    f"EUCLIDEAN measurement '{m.name}' requires exactly 2 "
                    f"landmark vertices, got {len(indices)}."
                )
            return float(np.linalg.norm(self.verts[indices[0]] - self.verts[indices[1]]))

        elif m.type == MeasurementType.GEODESIC:
            indices = self.resolve_indices(m.landmarks)
            return self._geodesic_length(indices, closed=m.closed)

        elif m.type == MeasurementType.PLANAR:
            if len(m.landmarks) != 1:
                raise ValueError(
                    f"PLANAR measurement '{m.name}' requires exactly 1 landmark name, "
                    f"got {len(m.landmarks)}."
                )
            lm_name = m.landmarks[0]
            if lm_name not in self.plane_heights:
                raise ValueError(
                    f"Landmark '{lm_name}' has no associated plane height.  "
                    "Use a ring-type landmark spec (Ring, AnchorVerticesRing, "
                    "ScanNarrowest, or ScanWidest)."
                )
            return _mesh_slice_perimeter(self.verts, self.faces, self.plane_heights[lm_name])

        else:
            raise ValueError(f"Unknown measurement type: {m.type}")

    def _geodesic_length(self, indices: list[int], closed: bool) -> float:
        try:
            import gdist
        except ImportError as exc:
            raise ImportError(
                "gdist is required for GEODESIC measurements.\n"
                "Install with:  uv pip install gdist\n"
                "          or:  conda install conda-forge::tvb-gdist"
            ) from exc

        path = indices + ([indices[0]] if closed else [])
        total = 0.0
        for i in range(len(path) - 1):
            src = np.array([path[i]], dtype=np.int32)
            tgt = np.array([path[i + 1]], dtype=np.int32)
            dists = gdist.compute_gdist(self.verts, self.faces, src, tgt)
            total += float(np.min(dists))
        return total
