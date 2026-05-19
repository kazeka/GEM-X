#!/usr/bin/env python3
"""
Measure body anthropometry from a GEM-X prediction.

Loads hpe_results.pt, evaluates the SOMA body model in canonical T-pose using
the predicted body shape, computes measurements, and saves:
  - a results table to stdout
  - <output_dir>/measurements.json
  - <output_dir>/measurements.html  (annotated 3D viewer)

Usage:
    python scripts/viz/measure_soma.py outputs/demo_soma/<clip>/hpe_results.pt

Requirements (in addition to the GEM-X env):
    uv pip install plotly gdist
    # or: conda install conda-forge::tvb-gdist

To add a new measurement:
    Edit gem/utils/anthropometry/landmarks.py  (add landmark spec if needed)
    Edit gem/utils/anthropometry/measurements.py  (add Measurement entry)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gem.utils.anthropometry.landmarks import LANDMARK_SPECS, Ring, AnchorVerticesRing, ScanNarrowest, ScanWidest, find_landmarks
from gem.utils.anthropometry.measurements import MEASUREMENTS, SomaMeasurer


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
    sw = soma_inner.skinning_weights
    dense = sw.to_dense()
    raw = dense.argmax(dim=1).numpy()
    return np.clip(raw - 1, 0, 76)


def _joint_names(soma_inner) -> list[str]:
    return list(soma_inner.rig_data["joint_names"])[1:]


def _print_table(measurements: dict[str, float]) -> None:
    print("\n┌─────────────────────────────┬───────────┬──────────┐")
    print("│ Measurement                 │   metres  │    cm    │")
    print("├─────────────────────────────┼───────────┼──────────┤")
    for name, val_m in measurements.items():
        val_cm = val_m * 100.0
        print(f"│ {name:<27}  │ {val_m:>7.4f}  │  {val_cm:>6.1f}  │")
    print("└─────────────────────────────┴───────────┴──────────┘\n")


def _build_html(
    verts: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    landmarks: dict,
    measurements: dict[str, float],
    dominant: np.ndarray,
    jnames: list[str],
) -> str:
    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit("plotly not found — install with:  uv pip install plotly")

    fig = go.Figure()

    # ── Base mesh coloured by dominant joint ──────────────────────────────────
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=dominant.astype(float),
            intensitymode="vertex",
            colorscale="Turbo",
            cmin=0, cmax=76,
            colorbar=dict(title="joint", thickness=10, len=0.5, x=1.02),
            opacity=0.55,
            flatshading=False,
            lighting=dict(ambient=0.35, diffuse=0.8, fresnel=0.1, specular=0.05, roughness=0.9),
            lightposition=dict(x=1, y=3, z=2),
            hoverinfo="skip",
            showlegend=False,
            name="mesh",
        )
    )

    # ── Colour palette for landmark groups ───────────────────────────────────
    _RING_COLOURS = {
        "CHEST_RING": "#ff6b6b",
        "WAIST_RING": "#ffd93d",
        "HIP_RING":   "#6bcb77",
    }
    _POINT_COLOURS = {
        "HEAD_TOP": "#a29bfe",
        "HEEL":     "#74b9ff",
    }

    # ── Landmark spheres ──────────────────────────────────────────────────────
    for lm_name, indices in landmarks.items():
        if isinstance(indices, list):
            colour = _RING_COLOURS.get(lm_name, "#ff9f43")
            lm_verts = verts[indices]
            fig.add_trace(
                go.Scatter3d(
                    x=lm_verts[:, 0], y=lm_verts[:, 1], z=lm_verts[:, 2],
                    mode="markers",
                    marker=dict(size=6, color=colour, line=dict(color="white", width=1)),
                    name=lm_name,
                    text=[f"v{idx}" for idx in indices],
                    hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra>" + lm_name + "</extra>",
                )
            )
            # Connect ring with a loop line
            loop_v = np.vstack([lm_verts, lm_verts[:1]])
            fig.add_trace(
                go.Scatter3d(
                    x=loop_v[:, 0], y=loop_v[:, 1], z=loop_v[:, 2],
                    mode="lines",
                    line=dict(color=colour, width=3, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        else:
            colour = _POINT_COLOURS.get(lm_name, "#dfe6e9")
            lv = verts[indices]
            fig.add_trace(
                go.Scatter3d(
                    x=[lv[0]], y=[lv[1]], z=[lv[2]],
                    mode="markers+text",
                    marker=dict(size=8, color=colour, symbol="diamond",
                                line=dict(color="white", width=1)),
                    text=[lm_name],
                    textposition="top center",
                    textfont=dict(size=9, color=colour),
                    name=lm_name,
                    hovertemplate=f"v{indices}<br>(%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}})<extra>{lm_name}</extra>",
                )
            )

    # ── Measurement annotations (text labels between landmark centroids) ───────
    for m_name, val_m in measurements.items():
        # Find the landmarks for this measurement
        m_def = next((m for m in MEASUREMENTS if m.name == m_name), None)
        if m_def is None:
            continue

        # Compute centroid of involved vertices
        all_idx = []
        for lname in m_def.landmarks:
            v = landmarks.get(lname)
            if v is None:
                continue
            all_idx.extend(v if isinstance(v, list) else [v])

        if not all_idx:
            continue

        centroid = verts[all_idx].mean(axis=0)
        label = f"{m_name}<br>{val_m*100:.1f} cm"
        # Offset label to the right (+X)
        fig.add_trace(
            go.Scatter3d(
                x=[centroid[0] + 0.20],
                y=[centroid[1]],
                z=[centroid[2]],
                mode="text",
                text=[label],
                textfont=dict(size=11, color="white"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="GEM-X — Body Measurements (Canonical T-pose)",
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

    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Body measurement from hpe_results.pt")
    parser.add_argument("hpe_results", help="Path to hpe_results.pt")
    parser.add_argument("--soma_assets", default="inputs/soma_assets")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for output files (default: same dir as hpe_results.pt)")
    parser.add_argument("--no_html", action="store_true", help="Skip HTML output")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.hpe_results).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load predictions ──────────────────────────────────────────────────────
    pred = torch.load(args.hpe_results, map_location="cpu", weights_only=False)
    params = _get_body_params_global(pred)

    identity_coeffs = params["identity_coeffs"].float().squeeze(0).mean(0, keepdim=True)
    scale_params = params["scale_params"].float().squeeze(0).mean(0, keepdim=True)

    # ── Build SOMA (full LOD for best ring sampling) ──────────────────────────
    from gem.utils.soma_utils.soma_layer import SomaLayer

    soma = SomaLayer(
        data_root=args.soma_assets,
        low_lod=False,
        device="cpu",
        identity_model_type="mhr",
        mode="warp",
    )

    zero_poses = torch.zeros((1, 77, 3))
    zero_transl = torch.zeros((1, 3))

    with torch.no_grad():
        out = soma.static_forward(zero_poses, identity_coeffs, scale_params, zero_transl)

    verts = _to_numpy(out["vertices"][0])   # (V, 3)
    joints = _to_numpy(out["joints"][0])    # (77, 3)
    faces = _to_numpy(soma.faces)           # (F, 3)

    print(f"Mesh: {len(verts):,} vertices, {len(faces):,} faces (full LOD)")

    # ── Measure ───────────────────────────────────────────────────────────────
    measurer = SomaMeasurer(verts, faces, joints)
    results = measurer.measure_all()

    _print_table(results)

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = out_dir / "measurements.json"
    with open(json_path, "w") as f:
        json.dump({k: round(v, 6) for k, v in results.items()}, f, indent=2)
    print(f"JSON  → {json_path}")

    # ── HTML ──────────────────────────────────────────────────────────────────
    if not args.no_html:
        dominant = _dominant_joint_per_vertex(soma.soma)
        jnames = _joint_names(soma.soma)
        html = _build_html(verts, faces, joints, measurer.landmarks, results, dominant, jnames)
        html_path = out_dir / "measurements.html"
        html_path.write_text(html)
        print(f"HTML  → {html_path}")


if __name__ == "__main__":
    main()
