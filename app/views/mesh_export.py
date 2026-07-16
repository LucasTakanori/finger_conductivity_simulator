"""Conductivity image export: one beat or a reproducible augmented batch."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import figures, state, ui  # noqa: E402
from finger_sim.augmentation import AugmentationSpec  # noqa: E402
from finger_sim.dataset import generate_augmented_dataset  # noqa: E402
from finger_sim.export import arrays_for_export, export_bytes  # noqa: E402
from finger_sim import project as project_io  # noqa: E402
from finger_sim.simulation import simulate_grid  # noqa: E402

ui.page_header(
    "Step 03",
    "Conductivity image export",
    "Export the conductivity as a regular image on a Cartesian grid — one beat, or a "
    "reproducible batch of randomized anatomies and waveforms. A FEM mesh can be drawn "
    "on top for reference; it does not change what is exported.",
)

model = state.get_model()
spec = state.get_waveform()

controls = st.columns([2, 2, 1, 1, 1])
view = controls[0].selectbox("Conductivity", ["Delta conductivity Δσ", "Absolute conductivity σ"])
resolution = controls[1].select_slider(
    "Image resolution (pixels per side)", [40, 48, 64, 80, 96, 112, 128], value=40,
    help="The exported image is this many pixels square. 40×40 matches the img_size the "
    "bundled ring meshes were mapped with.",
)
frame = controls[2].slider("Time frame", 0, spec.frames - 1, 0)
overlay_names = ["No mesh overlay", *state.mesh_catalog().keys()]
overlay_choice = controls[3].selectbox("FEM mesh overlay", overlay_names)
overlay = None if overlay_choice == overlay_names[0] else overlay_choice
animate = controls[4].toggle("▶ Animate", value=False, help="Play the whole beat in place.")

result = simulate_grid(model, spec, int(resolution))
mesh_overlay = state.mesh_overlay_payload(overlay, model) if overlay else None
if animate:
    preview = figures.conductivity_beat_figure(result, view, mesh_overlay=mesh_overlay, height=650)
else:
    values = result.delta_sigma[frame] if view.startswith("Delta") else result.sigma_dynamic[frame]
    preview = figures.conductivity_figure(
        result,
        values,
        f"{view} · t={result.time_s[frame]:.3f} s · {resolution}×{resolution} image",
        delta=view.startswith("Delta"),
        mesh_overlay=mesh_overlay,
        height=650,
    )
st.plotly_chart(preview, use_container_width=True)
if overlay:
    st.info(
        f"{overlay} is drawn for reference only. The export stays a {resolution}×{resolution} "
        "conductivity image — projecting onto the FEM mesh is a later step."
    )

single_tab, batch_tab, session_tab = st.tabs(
    ["Current model and beat", "Augmented GCNM dataset", "Save / load session"]
)

with single_tab:
    arrays = arrays_for_export(result, model, spec, mesh_id=None, mesh_manifest=None)
    st.download_button(
        "Download current simulation NPZ",
        export_bytes(arrays),
        file_name=f"finger_image{resolution}_{spec.frames}frames.npz",
        mime="application/octet-stream",
        type="primary",
    )
    st.markdown(
        f"**What this saves:** one finger and one beat at 50 kHz, as a "
        f"{resolution}×{resolution} image. `delta_sigma` is the clean GCNM target Δσ; "
        "`absolute_conductivity` is σ₀+Δσ at every frame; `resting_absolute_conductivity` is σ₀; "
        "`waveform`, `time_s`, tissue labels, pixel coordinates, and the complete finger model "
        "are included. Compatibility keys used by the existing 2D_GCNM loader are retained."
    )

with batch_tab:
    st.markdown("#### Generate different beats and different finger anatomies")
    st.caption(
        f"One sample is one complete beat plus its own anatomy. With the current "
        f"{spec.frames}-frame beat the export is shaped "
        f"(samples × {spec.frames} frames × conductivity elements)."
    )
    first = st.columns(3)
    samples = first[0].number_input(
        "Samples (distinct beats)",
        2,
        10_000,
        1000,
        1,
        help="One sample = one complete beat with its own finger anatomy. Each beat "
        "contains the frames-per-beat set on the Waveform page, so 1000 samples of a "
        "50-frame beat is 50,000 conductivity fields.",
    )
    seed = first[1].number_input("Random seed", 0, 10_000_000, 0, 1)
    finger_pct = first[2].slider(
        "Finger size variation (%)", 0, 25, 8, 1,
        help="Set to 0 to keep one fixed finger size and vary only the beats and arteries.",
    )

    anatomy = st.columns(4)
    artery_size_pct = anatomy[0].slider("Artery size variation (%)", 0, 50, 20, 1)
    artery_position = anatomy[1].slider("Artery position variation (mm)", 0.0, 3.0, 1.0, 0.1)
    artery_rotation = anatomy[2].slider("Artery rotation variation (°)", 0.0, 45.0, 15.0, 1.0)
    conductivity_pct = anatomy[3].slider("Conductivity variation (%)", 0, 40, 10, 1)

    extra = st.columns(4)
    ring_rotation = extra[0].slider("Ring rotation variation (°)", 0.0, 45.0, 10.0, 1.0)
    diffusion_pct = extra[1].slider("Muscle diffusion variation (%)", 0, 50, 15, 1)
    waveform_pct = extra[2].slider("Waveform shape variation (%)", 0, 40, 12, 1)
    duration_pct = extra[3].slider("Beat duration variation (%)", 0, 30, 8, 1)

    augmentation = AugmentationSpec(
        samples=int(samples),
        seed=int(seed),
        finger_size_fraction=float(finger_pct) / 100.0,
        finger_rotation_deg=float(ring_rotation),
        artery_size_fraction=float(artery_size_pct) / 100.0,
        artery_position_mm=float(artery_position),
        artery_rotation_deg=float(artery_rotation),
        conductivity_fraction=float(conductivity_pct) / 100.0,
        diffusion_fraction=float(diffusion_pct) / 100.0,
        waveform_shape_fraction=float(waveform_pct) / 100.0,
        duration_fraction=float(duration_pct) / 100.0,
    )

    if st.button("Generate augmented dataset", type="primary", use_container_width=True):
        bar = st.progress(0.0, text=f"Simulating beat 0 of {samples}…")

        def _report(done: int, total: int) -> None:
            bar.progress(done / total, text=f"Simulating beat {done:,} of {total:,}…")

        batch = generate_augmented_dataset(
            model, spec, augmentation, mesh=None, grid_size=int(resolution), progress=_report
        )
        with st.spinner("Compressing the NPZ…"):
            st.session_state["augmented_dataset_bytes"] = export_bytes(batch)
        st.session_state["augmented_dataset_name"] = (
            f"finger_augmented_image{resolution}_{samples}samples_seed{seed}.npz"
        )
        st.session_state["augmented_dataset_shape"] = tuple(batch["delta_sigma"].shape)
        bar.empty()

    if "augmented_dataset_bytes" in st.session_state:
        shape = st.session_state["augmented_dataset_shape"]
        st.success(
            f"Dataset ready: {shape[0]} different samples × {shape[1]} frames × "
            f"{shape[2]:,} pixels."
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
        "Each sample independently perturbs finger width/height, ring rotation, artery size, "
        "ellipticity, position and rotation, tissue and arterial conductivity, muscle diffusion, "
        "pulse amplitude, waveform shape, and beat duration. The seed records the draw exactly, so a "
        "varied dataset is reproducible rather than many copies of one beat."
    )

with session_tab:
    st.markdown("#### Save this model + waveform as a reusable project")
    st.markdown(
        "A project file captures the current finger model, the selected waveform, the image "
        "resolution, and the augmentation settings shown in the **Augmented GCNM dataset** tab. "
        "Hand it to the headless generator to build a dataset with no interface at all:"
    )
    st.code(
        "finger-sim-dataset --project finger_session.json --samples 1000 --seed 0 \\\n"
        "    --out exports/finger_dataset_1000.npz",
        language="bash",
    )
    project = project_io.build_project(
        model,
        spec,
        mesh=None,  # export the conductivity image; FEM projection comes later
        grid_size=int(resolution),
        augmentation=augmentation,
    )
    st.download_button(
        "Download project (.json)",
        project_io.dumps_project(project),
        file_name="finger_session.json",
        mime="application/json",
        type="primary",
    )

    st.divider()
    st.markdown("#### Load a saved project")
    uploaded = st.file_uploader("Upload a project .json", type=["json"])
    if uploaded is not None:
        loaded = project_io.loads_project(uploaded.getvalue().decode("utf-8"))
        state.set_model_dict(loaded["finger_model"])
        state.set_waveform(project_io.waveform_of(loaded))
        st.success("Project loaded — finger model and waveform updated across every page.")
