#!/usr/bin/env python3
"""
Interactive Plotly visualization of a GEM-X prediction in canonical T-pose.

Vertex ordering
---------------
Vertices are deterministically ordered by SOMA_neutral.npz:
  full LOD  → 18 056 vertices  (bind_shape row order)
  low LOD   → 4 505 vertices   (lod_mid_to_low subset of the above)
So vertex index i always refers to the same anatomical point for a given LOD.

Each vertex is coloured by its *dominant joint* — the joint with the highest
LBS skinning weight for that vertex.  Hovering shows the vertex index and the
name of that joint.

Usage:
    python scripts/viz/viz_canonical_pose.py outputs/demo_soma/<clip>/hpe_results.pt

Output:
    <same dir>/canonical_pose.html  — interactive 3D viewer

Requirements (add to your env if missing):
    uv pip install plotly
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _get_body_params_global(pred: dict) -> dict:
    if "body_params_global" in pred:
        return pred["body_params_global"]
    for k in pred:
        if k.endswith("_params_global"):
            return pred[k]
    raise KeyError(f"No *_params_global key in {list(pred.keys())}")


def _to_numpy(t) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    if isinstance(t, (list, tuple)):
        return np.array(t)
    return t.cpu().numpy()


def _dominant_joint_per_vertex(soma_inner) -> np.ndarray:
    """Return (V,) array of GEM-X joint indices (0-76) with highest skin weight."""
    sw = soma_inner.skinning_weights          # sparse CSR (V, 78)
    dense = sw.to_dense()                    # (V, 78)  -- col 0 = Root, cols 1..77 = body joints
    raw = dense.argmax(dim=1).numpy()        # dominant column index (0 = Root)
    # Remap to GEM-X joint space (skip Root at col 0 → subtract 1, clamp)
    return np.clip(raw - 1, 0, 76)


def _joint_names(soma_inner) -> list[str]:
    """Return 77 joint name strings for GEM-X joints 0-76 (Hips … RightToeEnd)."""
    names = list(soma_inner.rig_data["joint_names"])  # 78 entries, index 0 = Root
    return names[1:]                                   # drop Root


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical T-pose mesh viewer")
    parser.add_argument("hpe_results", help="Path to hpe_results.pt")
    parser.add_argument(
        "--soma_assets",
        default="inputs/soma_assets",
        help="Path to SOMA body model assets (default: inputs/soma_assets)",
    )
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument("--high_lod", action="store_true", help="Use full-resolution mesh (18 056 verts)")
    parser.add_argument("--no_skeleton", action="store_true", help="Hide skeleton overlay")
    parser.add_argument("--no_labels", action="store_true", help="Hide joint index/name labels")
    args = parser.parse_args()

    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit("plotly not found — install it with:  uv pip install plotly")

    # ── Load predictions ──────────────────────────────────────────────────────
    pred = torch.load(args.hpe_results, map_location="cpu", weights_only=False)
    params = _get_body_params_global(pred)

    # Shape params: (1, L, C) → mean over frames → (1, C)
    identity_coeffs = params["identity_coeffs"].float().squeeze(0).mean(0, keepdim=True)
    scale_params = params["scale_params"].float().squeeze(0).mean(0, keepdim=True)

    # ── Build SOMA body model ─────────────────────────────────────────────────
    from gem.utils.soma_utils.soma_layer import SomaLayer

    soma = SomaLayer(
        data_root=args.soma_assets,
        low_lod=not args.high_lod,
        device="cpu",
        identity_model_type="mhr",
        mode="warp",
    )

    # Canonical T-pose: all joint rotations = 0, translation = 0
    zero_poses = torch.zeros((1, 77, 3))
    zero_transl = torch.zeros((1, 3))

    with torch.no_grad():
        out = soma.static_forward(zero_poses, identity_coeffs, scale_params, zero_transl)

    verts = _to_numpy(out["vertices"][0])   # (V, 3)
    joints = _to_numpy(out["joints"][0])    # (77, 3)
    faces = _to_numpy(soma.faces)           # (F, 3)
    parents = _to_numpy(soma.parents)       # (77,)

    # ── Vertex annotations ────────────────────────────────────────────────────
    # Vertex order is fixed by SOMA_neutral.npz (bind_shape / lod_mid_to_low).
    # dominant[i] = GEM-X joint index (0-76) of the joint with the highest
    # linear blend skinning weight for vertex i.
    dominant = _dominant_joint_per_vertex(soma.soma)   # (V,) int
    jnames = _joint_names(soma.soma)                   # 77 strings

    vertex_hover = [
        f"v{i}<br>joint {dom}: {jnames[dom]}"
        for i, dom in enumerate(dominant)
    ]

    # ── Joint labels (index + name) ───────────────────────────────────────────
    joint_labels = [f"{i} {jnames[i]}" for i in range(77)]

    # ── Plotly figure ─────────────────────────────────────────────────────────
    fig = go.Figure()

    # Mesh coloured by dominant joint index (deterministic, per-vertex)
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=dominant.astype(float),
            intensitymode="vertex",
            colorscale="Turbo",
            cmin=0,
            cmax=76,
            colorbar=dict(title="joint idx", thickness=12, len=0.6),
            opacity=0.88,
            flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.85, fresnel=0.2, specular=0.05, roughness=0.9),
            lightposition=dict(x=1, y=3, z=2),
            text=vertex_hover,
            hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
            name="Body mesh",
        )
    )

    if not args.no_skeleton:
        fig.add_trace(
            go.Scatter3d(
                x=joints[:, 0],
                y=joints[:, 1],
                z=joints[:, 2],
                mode="markers",
                marker=dict(size=4, color="white", line=dict(color="black", width=1)),
                text=joint_labels,
                hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
                name="Joints",
            )
        )

        sx, sy, sz = [], [], []
        for j_idx, par in enumerate(parents):
            if par < 0:
                continue
            sx += [joints[j_idx, 0], joints[int(par), 0], None]
            sy += [joints[j_idx, 1], joints[int(par), 1], None]
            sz += [joints[j_idx, 2], joints[int(par), 2], None]

        fig.add_trace(
            go.Scatter3d(
                x=sx, y=sy, z=sz,
                mode="lines",
                line=dict(color="white", width=3),
                hoverinfo="skip",
                name="Skeleton",
            )
        )

    if not args.no_labels:
        fig.add_trace(
            go.Scatter3d(
                x=joints[:, 0],
                y=joints[:, 1],
                z=joints[:, 2],
                mode="text",
                text=joint_labels,
                textposition="top center",
                textfont=dict(size=8, color="white"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    lod_label = f"full LOD ({len(verts):,} verts)" if args.high_lod else f"low LOD ({len(verts):,} verts)"
    fig.update_layout(
        title=f"GEM-X — Canonical T-pose  [{lod_label}]",
        paper_bgcolor="#1a1a1a",
        scene=dict(
            bgcolor="#1a1a1a",
            aspectmode="data",
            xaxis=dict(title="X (m)", gridcolor="#333"),
            yaxis=dict(title="Y (m)", gridcolor="#333"),
            zaxis=dict(title="Z (m)", gridcolor="#333"),
            camera=dict(eye=dict(x=0, y=0.5, z=2.5), up=dict(x=0, y=1, z=0)),
        ),
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
    )

    out_path = (
        Path(args.output) if args.output
        else Path(args.hpe_results).parent / "canonical_pose.html"
    )
    fig.write_html(str(out_path))
    print(f"Saved → {out_path}")
    print(f"  {len(verts):,} vertices, {len(faces):,} faces, 77 joints")
    print(f"  Vertex colour = dominant LBS joint (Turbo scale, 0=Hips … 76=RightToeEnd)")


if __name__ == "__main__":
    main()
