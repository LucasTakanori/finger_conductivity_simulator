"""Mesh projection and single/batch GCNM conductivity export."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import figures, state, ui  # noqa: E402
from finger_sim.augmentation import AugmentationSpec  # noqa: E402
from finger_sim.dataset import generate_augmented_dataset  # noqa: E402
from finger_sim.export import arrays_for_export, export_bytes  # noqa: E402
from finger_sim.mesh import mesh_points_mm  # noqa: E402
from finger_sim.simulation import simulate_grid, simulate_points  # noqa: E402

ui.page_header(
    "Step 03",
    "Mesh projection and dataset export",
    "Project absolute or delta conductivity onto a real PVI FEM mesh, then export either "
    "the current beat or a reproducible batch of randomized anatomies and waveforms.",
)

model = state.get_model()
spec = state.get_waveform()

mesh_names = list(state.mesh_catalog())
default_mesh = mesh_names.index("subject006_US120") + 1 if "subject006_US120" in mesh_names else 0
controls = st.columns([2, 2, 1, 1])
target = controls[0].selectbox("Sampling mesh", ["Cartesian grid", *mesh_names], index=default_mesh)
view = controls[1].selectbox("Conductivity", ["Delta conductivity Δσ", "Absolute conductivity σ"])
frame = controls[2].slider("Time frame", 0, spec.frames - 1, 0)
show_edges = controls[3].checkbox("Show mesh", value=True)

mesh = None
mesh_id = None
mesh_manifest = None
if target == "Cartesian grid":
    result = simulate_grid(model, spec, 112)
    values = result.delta_sigma[frame] if view.startswith("Delta") else result.sigma_dynamic[frame]
    title = f"{view} · t={result.time_s[frame]:.3f} s"
    st.plotly_chart(
        figures.conductivity_figure(result, values, title, delta=view.startswith("Delta"), height=650),
        use_container_width=True,
    )
else:
    mesh = state.load_mesh(target)
    result = simulate_points(mesh_points_mm(mesh, model), model, spec)
    mesh_id = mesh.mesh_id
    mesh_manifest = mesh.manifest
    if view.startswith("Delta"):
        values = result.delta_sigma[frame]
        vlim = float(np.nanmax(np.abs(result.delta_sigma)))
        is_delta = True
    else:
        values = result.sigma_dynamic[frame]
        vlim = None
        is_delta = False
    title = f"{target} · {view} · t={result.time_s[frame]:.3f} s"
    st.plotly_chart(
        figures.mesh_field_figure(mesh, model, values, title, is_delta, vlim=vlim, show_edges=show_edges),
        use_container_width=True,
    )
    status = mesh_manifest.get("selection", {}).get("status", "bundled")
    st.info(
        f"{target}: {len(mesh.elements):,} inverse elements · provenance: {status}. "
        "Dark triangle edges show exactly where conductivity is sampled."
    )

single_tab, batch_tab = st.tabs(["Current model and beat", "Augmented GCNM dataset"])

with single_tab:
    arrays = arrays_for_export(result, model, spec, mesh_id=mesh_id, mesh_manifest=mesh_manifest)
    st.download_button(
        "Download current simulation NPZ",
        export_bytes(arrays),
        file_name=f"finger_{mesh_id or 'grid'}_{spec.frames}frames.npz",
        mime="application/octet-stream",
        type="primary",
    )
    st.markdown(
        "**What this saves:** one finger and one beat at 50 kHz. `delta_sigma` is the clean "
        "GCNM target Δσ; `absolute_conductivity` is σ₀+Δσ at every frame; "
        "`resting_absolute_conductivity` is σ₀; `waveform`, `time_s`, tissue labels, coordinates, "
        "mesh provenance, and the complete finger model are included. Compatibility keys used by "
        "the existing 2D_GCNM loader are retained."
    )

with batch_tab:
    st.markdown("#### Generate different beats and different finger anatomies")
    first = st.columns(3)
    samples = first[0].number_input("Samples", 2, 500, 10, 1)
    seed = first[1].number_input("Random seed", 0, 10_000_000, 0, 1)
    finger_pct = first[2].slider("Finger size variation (%)", 0, 25, 8, 1)

    anatomy = st.columns(4)
    artery_size_pct = anatomy[0].slider("Artery size variation (%)", 0, 50, 20, 1)
    artery_position = anatomy[1].slider("Artery position variation (mm)", 0.0, 3.0, 1.0, 0.1)
    artery_rotation = anatomy[2].slider("Artery rotation variation (°)", 0.0, 45.0, 15.0, 1.0)
    conductivity_pct = anatomy[3].slider("Conductivity variation (%)", 0, 40, 10, 1)

    pulse = st.columns(2)
    waveform_pct = pulse[0].slider("Waveform shape variation (%)", 0, 40, 12, 1)
    duration_pct = pulse[1].slider("Beat duration variation (%)", 0, 30, 8, 1)

    augmentation = AugmentationSpec(
        samples=int(samples),
        seed=int(seed),
        finger_size_fraction=float(finger_pct) / 100.0,
        artery_size_fraction=float(artery_size_pct) / 100.0,
        artery_position_mm=float(artery_position),
        artery_rotation_deg=float(artery_rotation),
        conductivity_fraction=float(conductivity_pct) / 100.0,
        waveform_shape_fraction=float(waveform_pct) / 100.0,
        duration_fraction=float(duration_pct) / 100.0,
    )

    if st.button("Generate augmented dataset", type="primary", use_container_width=True):
        with st.spinner(f"Generating {samples} distinct simulations…"):
            batch = generate_augmented_dataset(model, spec, augmentation, mesh=mesh, grid_size=96)
            st.session_state["augmented_dataset_bytes"] = export_bytes(batch)
            st.session_state["augmented_dataset_name"] = (
                f"finger_augmented_{mesh_id or 'grid'}_{samples}samples_seed{seed}.npz"
            )
            st.session_state["augmented_dataset_shape"] = tuple(batch["delta_sigma"].shape)

    if "augmented_dataset_bytes" in st.session_state:
        shape = st.session_state["augmented_dataset_shape"]
        st.success(
            f"Dataset ready: {shape[0]} different samples × {shape[1]} frames × "
            f"{shape[2]:,} conductivity elements."
        )
        st.download_button(
            "Download augmented dataset NPZ",
            st.session_state["augmented_dataset_bytes"],
            file_name=st.session_state["augmented_dataset_name"],
            mime="application/octet-stream",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        "Each sample independently perturbs finger width/height, artery size, ellipticity, position, "
        "rotation, tissue and arterial conductivity, pulse amplitude, waveform shape, and beat duration. "
        "The seed records the draw exactly, so a useful varied dataset is reproducible rather than ten copies of one beat."
    )
