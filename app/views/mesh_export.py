"""Mesh & export: project σ onto a ring mesh, show the triangles, export NPZ."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import figures, state, ui  # noqa: E402
from finger_sim.export import arrays_for_export, export_bytes  # noqa: E402
from finger_sim.mesh import mesh_points_mm  # noqa: E402
from finger_sim.simulation import simulate_grid, simulate_points  # noqa: E402

ui.page_header(
    "Step 03",
    "Mesh projection & export",
    "Sample the conductivity on a real PVI ring mesh, inspect the triangular grid drawn over "
    "the field, and download a GCNM-ready NPZ.",
)

model = state.get_model()
spec = state.get_waveform()

controls = st.columns([2, 2, 1, 1])
target = controls[0].selectbox("Ring mesh", ["Cartesian grid", *state.mesh_catalog().keys()])
view = controls[1].selectbox(
    "Map", ["Tissue labels", "Baseline conductivity", "Differential conductivity", "Dynamic conductivity"], index=2)
frame = controls[2].slider("Frame", 0, spec.frames - 1, 0)
show_edges = controls[3].checkbox("Triangles", value=True)

if target == "Cartesian grid":
    result = simulate_grid(model, spec, 96)
    mesh_id = None
    mesh_manifest = None
    if view == "Tissue labels":
        st.plotly_chart(figures.tissue_figure(result), use_container_width=True)
    elif view == "Baseline conductivity":
        st.plotly_chart(figures.baseline_figure(result), use_container_width=True)
    else:
        dyn = "Differential conductivity" if view == "Differential conductivity" else "Dynamic conductivity"
        st.plotly_chart(figures.dynamic_figure(result, dyn, True), use_container_width=True)
    export_result = result
else:
    mesh = state.load_mesh(target)
    result = simulate_points(mesh_points_mm(mesh, model), model, spec)
    mesh_id = mesh.mesh_id
    mesh_manifest = mesh.manifest
    export_result = result
    vlim = None
    if view == "Tissue labels":
        values, is_delta = result.tissue_labels, False
    elif view == "Baseline conductivity":
        values, is_delta = result.sigma_baseline, False
    elif view == "Differential conductivity":
        values, is_delta = result.delta_sigma[frame], True
        vlim = float(np.nanmax(np.abs(result.delta_sigma)))
    else:
        values, is_delta = result.sigma_dynamic[frame], False
    title = f"{target} · {view} · t={result.time_s[frame]:.3f} s"
    st.plotly_chart(
        figures.mesh_field_figure(mesh, model, values, title, is_delta, vlim=vlim, show_edges=show_edges),
        use_container_width=True,
    )
    status = mesh_manifest.get("selection", {}).get("status", "bundled")
    st.info(f"Mesh: {target} · {len(mesh.elements):,} elements · provenance: {status}")

arrays = arrays_for_export(export_result, model, spec, mesh_id=mesh_id, mesh_manifest=mesh_manifest)
st.download_button(
    "Download GCNM-ready NPZ",
    export_bytes(arrays),
    file_name=f"finger_delta_{mesh_id or 'grid'}_{spec.frames}frames.npz",
    mime="application/octet-stream",
    type="primary",
)
with st.expander("Export contents"):
    st.markdown(
        "The NPZ holds `sigma` (clean Δσ target), `sigma_baseline`, `sigma_dynamic`, `time_s`, "
        "`waveform`, `tissue_labels`, `points_mm`, and JSON metadata. It is a synthetic simulation "
        "target, not a PVI/Newton reconstruction."
    )
