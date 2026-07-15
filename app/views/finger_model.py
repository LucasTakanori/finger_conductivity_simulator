"""Finger model: drag tissues on the GPU canvas, scale to a ring, see σ₀ live."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import figures, palette, state, ui  # noqa: E402
from finger_sim.models import Artery, WaveformSpec  # noqa: E402
from finger_sim.simulation import simulate_grid  # noqa: E402
from components.finger_canvas import finger_canvas  # noqa: E402

ui.page_header(
    "Step 01",
    "Finger model",
    "Drag the handles to move, resize, and rotate each tissue. Colours update on the GPU "
    "as you drag; hover anywhere to identify the tissue and read its conductivity.",
)


def default_artery() -> dict:
    return Artery(3.4, -2.0, 0.72, 0.60, -12.0, peak_delta_s_m=0.055).__dict__.copy()


base = state.get_model_dict()

# --- sidebar: everything that is not dragged directly on the canvas ---------- #
with st.sidebar:
    st.header("Model controls")
    color_by = st.radio("Canvas colour", ["tissue", "conductivity"], horizontal=True)

    st.subheader("Size")
    rings = ["Free (drag to resize)"] + list(state.mesh_catalog().keys())
    ring_choice = st.selectbox("Scale finger to ring size", rings)
    ring_selected = ring_choice if ring_choice != rings[0] else None
    fill = st.slider("Ring fill", 0.80, 1.00, 0.98, 0.01, disabled=ring_selected is None)

    base["rotation_deg"] = st.slider("Finger rotation (°)", -180.0, 180.0, float(base["rotation_deg"]), 1.0)
    base["skin_thickness_mm"] = st.slider("Skin thickness (mm)", 0.2, 2.0, float(base["skin_thickness_mm"]), 0.05)
    base["fat_thickness_mm"] = st.slider("Fat thickness (mm)", 0.2, 4.0, float(base["fat_thickness_mm"]), 0.05)

    with st.expander("Tissue conductivity (S/m)"):
        c = base["conductivities"]
        c["skin_s_m"] = st.number_input("Skin", 0.001, 2.0, float(c["skin_s_m"]), 0.01)
        c["fat_s_m"] = st.number_input("Fat", 0.001, 2.0, float(c["fat_s_m"]), 0.01)
        c["muscle_s_m"] = st.number_input("Muscle", 0.001, 2.0, float(c["muscle_s_m"]), 0.01)
        c["bone_s_m"] = st.number_input("Bone", 0.001, 2.0, float(c["bone_s_m"]), 0.01)
        c["ligament_s_m"] = st.number_input("Ligament / tendon", 0.001, 2.0, float(c["ligament_s_m"]), 0.01)

    with st.expander("Arteries"):
        count = st.radio("Number of arteries", [1, 2], index=len(base["arteries"]) - 1, horizontal=True)
        if count == 1:
            base["arteries"] = base["arteries"][:1]
        elif len(base["arteries"]) == 1:
            base["arteries"] = base["arteries"] + [default_artery()]
        for i, art in enumerate(base["arteries"]):
            st.markdown(f"**Artery {i + 1}**")
            art["baseline_conductivity_s_m"] = st.number_input(
                f"A{i + 1} baseline σ", 0.01, 2.0, float(art["baseline_conductivity_s_m"]), 0.01)
            art["peak_delta_s_m"] = st.number_input(
                f"A{i + 1} peak Δσ", -0.5, 0.5, float(art["peak_delta_s_m"]), 0.005)
            art["phase_delay_fraction"] = st.slider(
                f"A{i + 1} phase delay", 0.0, 0.5, float(art["phase_delay_fraction"]), 0.01)

    with st.expander("Muscle coupling"):
        base["muscle_diffusion_fraction"] = st.slider(
            "Diffusion fraction", 0.0, 0.5, float(base["muscle_diffusion_fraction"]), 0.01)
        base["muscle_diffusion_length_mm"] = st.slider(
            "Diffusion length (mm)", 0.2, 5.0, float(base["muscle_diffusion_length_mm"]), 0.05)

# --- ring scaling + mesh preview -------------------------------------------- #
mesh_overlay = None
if ring_selected:
    base = state.scale_model_to_ring(base, ring_selected, fill)

state.set_model_dict(base)
if ring_selected:
    mesh_overlay = state.mesh_overlay_payload(ring_selected, state.get_model())

# --- interactive canvas beside the absolute conductivity map ----------------- #
left, right = st.columns(2)
with left:
    st.markdown("**Interactive model**")
    edited = finger_canvas(
        model=base,
        palette=palette.tissue_palette_for_js(),
        color_by=color_by,
        mesh_overlay=mesh_overlay,
        height=560,
        key="finger_canvas",
    )
    if edited:
        state.set_model_dict(edited)

model = state.get_model()
with right:
    st.markdown("**Absolute conductivity σ₀**")
    grid = simulate_grid(model, WaveformSpec("heartbeat", 8, 1.0), 96)
    st.plotly_chart(figures.baseline_figure(grid), use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("Width", f"{model.width_mm:.1f} mm")
m2.metric("Height", f"{model.height_mm:.1f} mm")
m3.metric("Ring", ring_selected or "free")
if ring_selected:
    st.info(f"Finger scaled to {ring_selected} — mesh triangles previewed on the canvas.")
st.caption("Backend badge (bottom-right of the canvas) shows WebGPU or WebGL2. Both render the same field.")
