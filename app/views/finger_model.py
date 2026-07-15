"""Finger model: edit anatomy, conductivity, mesh overlay, and saved presets."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import figures, state, ui  # noqa: E402
from finger_sim.models import Artery, WaveformSpec  # noqa: E402
from finger_sim.simulation import simulate_grid  # noqa: E402

ui.page_header(
    "Step 01",
    "Finger model",
    "Choose a finger size, edit its tissues and arteries, and inspect the anatomy and "
    "50 kHz absolute conductivity with the selected FEM mesh drawn on top.",
)


def default_artery() -> dict:
    return Artery(3.4, -2.0, 0.72, 0.60, -12.0, peak_delta_s_m=0.055).__dict__.copy()


base = state.get_model_dict()

with st.sidebar:
    st.header("Finger and mesh")
    size_options = [*state.FINGER_SIZE_PRESETS, "Custom dimensions"]
    current_preset = st.selectbox("Finger size", size_options, index=1)
    applied = st.session_state.get("_applied_finger_size")
    if current_preset != "Custom dimensions" and current_preset != applied:
        width, height = state.FINGER_SIZE_PRESETS[current_preset]
        base = state.resize_model(base, width, height)
        st.session_state["_applied_finger_size"] = current_preset

    if current_preset == "Custom dimensions":
        size_cols = st.columns(2)
        width = size_cols[0].number_input("Width (mm)", 10.0, 35.0, float(base["width_mm"]), 0.5)
        height = size_cols[1].number_input("Height (mm)", 9.0, 32.0, float(base["height_mm"]), 0.5)
        if width != base["width_mm"] or height != base["height_mm"]:
            base = state.resize_model(base, float(width), float(height))
        st.session_state["_applied_finger_size"] = "Custom dimensions"

    mesh_options = ["No mesh overlay", *state.mesh_catalog().keys()]
    mesh_choice = st.selectbox("FEM mesh overlay", mesh_options)
    mesh_selected = None if mesh_choice == mesh_options[0] else mesh_choice

    st.subheader("Outer tissues")
    base["rotation_deg"] = st.slider("Finger rotation (°)", -180.0, 180.0, float(base["rotation_deg"]), 1.0)
    max_layer = max(0.8, min(base["width_mm"], base["height_mm"]) / 4)
    base["skin_thickness_mm"] = st.slider(
        "Skin thickness (mm)", 0.2, min(2.0, max_layer), float(base["skin_thickness_mm"]), 0.05
    )
    max_fat = max(0.3, min(base["width_mm"], base["height_mm"]) / 2 - base["skin_thickness_mm"] - 0.6)
    base["fat_thickness_mm"] = st.slider(
        "Fat thickness (mm)", 0.2, min(4.0, max_fat), min(float(base["fat_thickness_mm"]), min(4.0, max_fat)), 0.05
    )

    with st.expander("Absolute tissue conductivity · 50 kHz", expanded=True):
        c = base["conductivities"]
        c["skin_s_m"] = st.number_input("Skin (S/m)", 0.001, 2.0, float(c["skin_s_m"]), 0.01)
        c["fat_s_m"] = st.number_input("Fat (S/m)", 0.001, 2.0, float(c["fat_s_m"]), 0.01)
        c["muscle_s_m"] = st.number_input("Muscle (S/m)", 0.001, 2.0, float(c["muscle_s_m"]), 0.01)
        c["bone_s_m"] = st.number_input("Bone (S/m)", 0.001, 2.0, float(c["bone_s_m"]), 0.01)
        c["ligament_s_m"] = st.number_input("Ligament / tendon (S/m)", 0.001, 2.0, float(c["ligament_s_m"]), 0.01)

    with st.expander("Bone and ligament geometry"):
        bone = base["bone"]
        bone["center_x_mm"] = st.slider("Bone x (mm)", -5.0, 5.0, float(bone["center_x_mm"]), 0.1)
        bone["center_y_mm"] = st.slider("Bone y (mm)", -5.0, 5.0, float(bone["center_y_mm"]), 0.1)
        bone["radius_x_mm"] = st.slider("Bone radius x (mm)", 0.5, 5.0, float(bone["radius_x_mm"]), 0.1)
        bone["radius_y_mm"] = st.slider("Bone radius y (mm)", 0.5, 5.0, float(bone["radius_y_mm"]), 0.1)
        bone["rotation_deg"] = st.slider("Bone rotation (°)", -90.0, 90.0, float(bone["rotation_deg"]), 1.0)
        st.caption("Ligament geometry is preserved and scales with the selected finger size.")

    with st.expander("Arteries", expanded=True):
        count = st.radio("Number of arteries", [1, 2], index=len(base["arteries"]) - 1, horizontal=True)
        if count == 1:
            base["arteries"] = base["arteries"][:1]
        elif len(base["arteries"]) == 1:
            base["arteries"] = base["arteries"] + [default_artery()]
        xlim = max(2.0, base["width_mm"] / 2 - 1.0)
        ylim = max(2.0, base["height_mm"] / 2 - 1.0)
        for i, art in enumerate(base["arteries"]):
            st.markdown(f"**Artery {i + 1} geometry**")
            art["center_x_mm"] = st.slider(f"A{i + 1} x (mm)", -xlim, xlim, float(art["center_x_mm"]), 0.1)
            art["center_y_mm"] = st.slider(f"A{i + 1} y (mm)", -ylim, ylim, float(art["center_y_mm"]), 0.1)
            art["radius_x_mm"] = st.slider(f"A{i + 1} radius x (mm)", 0.2, 2.5, float(art["radius_x_mm"]), 0.05)
            art["radius_y_mm"] = st.slider(f"A{i + 1} radius y (mm)", 0.2, 2.5, float(art["radius_y_mm"]), 0.05)
            art["rotation_deg"] = st.slider(f"A{i + 1} rotation (°)", -90.0, 90.0, float(art["rotation_deg"]), 1.0)
            art["baseline_conductivity_s_m"] = st.number_input(
                f"A{i + 1} absolute conductivity (S/m)", 0.01, 2.0, float(art["baseline_conductivity_s_m"]), 0.01
            )
            art["peak_delta_s_m"] = st.number_input(
                f"A{i + 1} peak delta conductivity (S/m)", -0.5, 0.5, float(art["peak_delta_s_m"]), 0.005
            )
            art["phase_delay_fraction"] = st.slider(
                f"A{i + 1} pulse phase delay", 0.0, 0.5, float(art["phase_delay_fraction"]), 0.01
            )

    with st.expander("Muscle pulse spread"):
        base["muscle_diffusion_fraction"] = st.slider(
            "Artery-to-muscle coupling", 0.0, 0.5, float(base["muscle_diffusion_fraction"]), 0.01
        )
        base["muscle_diffusion_length_mm"] = st.slider(
            "Spread length (mm)", 0.2, 5.0, float(base["muscle_diffusion_length_mm"]), 0.05
        )

state.set_model_dict(base)
model = state.get_model()
mesh_overlay = state.mesh_overlay_payload(mesh_selected, model) if mesh_selected else None
grid = simulate_grid(model, WaveformSpec("heartbeat", 8, 1.0), 128)

anatomy, conductivity = st.columns(2, gap="large")
with anatomy:
    st.markdown("#### Interactive anatomical model")
    st.plotly_chart(
        figures.tissue_figure(grid, mesh_overlay=mesh_overlay, height=610),
        use_container_width=True,
    )
    st.caption("Edit geometry in the controls; hover the map to inspect coordinates. The dark lines are the selected FEM mesh.")
with conductivity:
    st.markdown("#### Absolute conductivity at rest")
    st.plotly_chart(
        figures.baseline_figure(grid, mesh_overlay=mesh_overlay, height=610),
        use_container_width=True,
    )
    st.caption("Sequential light-to-plum scale in S/m at 50 kHz. This is σ₀ before the pulse adds Δσ.")

st.divider()
st.markdown("### Save or load a finger model")
save_col, load_col, file_col = st.columns(3)
with save_col:
    model_name = st.text_input("Model name", "my finger model")
    if st.button("Save in this browser session", use_container_width=True):
        state.save_model(model_name)
        st.success(f"Saved {model_name.strip()}.")
with load_col:
    names = list(state.saved_models())
    selected_saved = st.selectbox("Saved session models", names, disabled=not names)
    if st.button("Load selected model", disabled=not names, use_container_width=True):
        state.load_saved_model(selected_saved)
        st.rerun()
with file_col:
    model_json = json.dumps({"finger_model": model.to_dict()}, indent=2)
    st.download_button(
        "Download model JSON",
        model_json,
        file_name=f"{model_name.strip().replace(' ', '_') or 'finger_model'}.json",
        mime="application/json",
        use_container_width=True,
    )
    uploaded_model = st.file_uploader("Load model JSON", type=["json"])
    if uploaded_model is not None and st.button("Apply uploaded model", use_container_width=True):
        payload = json.loads(uploaded_model.getvalue().decode("utf-8"))
        state.set_model_dict(payload.get("finger_model", payload))
        st.rerun()

if mesh_selected:
    st.info(f"Showing {mesh_selected} over both maps. Mesh selection changes sampling topology; it does not force the finger to become circular.")
