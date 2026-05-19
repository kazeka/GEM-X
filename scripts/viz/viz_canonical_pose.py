#!/usr/bin/env python3
"""
Interactive Plotly visualization of a GEM-X prediction in canonical T-pose.

The script loads body shape/identity parameters from hpe_results.pt, averages
them across frames to get a representative body shape, then evaluates the SOMA
body model at zero pose (T-pose) to obtain the mesh and skeleton.

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
    """Return the *_params_global sub-dict, matching demo_soma.py logic."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical T-pose mesh viewer")
    parser.add_argument("hpe_results", help="Path to hpe_results.pt")
    parser.add_argument(
        "--soma_assets",
        default="inputs/soma_assets",
        help="Path to SOMA body model assets (default: inputs/soma_assets)",
    )
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument("--high_lod", action="store_true", help="Use full-resolution mesh")
    parser.add_argument("--no_skeleton", action="store_true", help="Hide skeleton overlay")
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

    verts = _to_numpy(out["vertices"][0])   # (V, 3)  world space, metres
    joints = _to_numpy(out["joints"][0])    # (77, 3)
    faces = _to_numpy(soma.faces)           # (F, 3)  int indices
    parents = _to_numpy(soma.parents)       # (77,)   -1 for root

    # ── Plotly figure ─────────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightsteelblue",
            opacity=0.88,
            flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.85, fresnel=0.2, specular=0.05, roughness=0.9),
            lightposition=dict(x=1, y=3, z=2),
            showscale=False,
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
                marker=dict(size=3, color="orangered"),
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
                x=sx,
                y=sy,
                z=sz,
                mode="lines",
                line=dict(color="orangered", width=4),
                name="Skeleton",
            )
        )

    # Front-facing camera: SOMA is Y-up, body faces -Z in canonical pose
    fig.update_layout(
        title="GEM-X — Canonical T-pose",
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m)"),
            camera=dict(eye=dict(x=0, y=0.5, z=2.5), up=dict(x=0, y=1, z=0)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.01, y=0.99),
    )

    out_path = Path(args.output) if args.output else Path(args.hpe_results).parent / "canonical_pose.html"
    fig.write_html(str(out_path))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
