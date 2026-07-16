"""Waveform editor with synchronized pulse and conductivity playback."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import figures, state, ui  # noqa: E402
from finger_sim.augmentation import AugmentationSpec, augment_model, augment_waveform  # noqa: E402
from finger_sim.models import WaveformSpec  # noqa: E402
from finger_sim.simulation import simulate_grid  # noqa: E402

ui.page_header(
    "Step 02",
    "Waveform and conductivity replay",
    "Define one arterial pulse, optionally preview an augmented beat, and play the waveform "
    "and conductivity map on one shared time control.",
)

def _reset_waveform() -> None:
    state.reset_waveform()  # also bumps the widget nonce so the controls re-read it


model = state.get_model()
n = state.widget_nonce()

st.button("↺ Reset waveform to defaults", on_click=_reset_waveform)

current = state.get_waveform()
kinds = ["heartbeat", "sine", "custom"]
controls = st.columns([1, 1, 1, 2])
kind = controls[0].selectbox("Waveform", kinds, index=kinds.index(current.kind), key=f"wave_kind_{n}")
frames = controls[1].number_input("Frames per beat", 10, 200, current.frames, 5, key=f"wave_frames_{n}")
duration = controls[2].number_input("Beat duration (s)", 0.1, 10.0, current.duration_s, 0.1, key=f"wave_dur_{n}")
custom_values: list[float] = []
if kind == "custom":
    uploaded = controls[3].file_uploader("Upload CSV/TXT", type=["csv", "txt"])
    text = controls[3].text_area("Or paste samples", "0, 0.2, 0.9, 1.0, 0.6, 0.3, 0.15, 0.05, 0")
    if uploaded is not None:
        text = uploaded.getvalue().decode("utf-8")
    custom_values = np.fromstring(text.replace(";", ",").replace("\n", ","), sep=",").tolist()
    if len(custom_values) < 2:
        st.error("Custom waveform needs at least two comma-separated values.")
        st.stop()

spec = WaveformSpec(kind, int(frames), float(duration), custom_values)
state.set_waveform(spec)

with st.expander("Waveform data augmentation preview"):
    enabled = st.checkbox("Preview one randomized beat", value=False)
    aug_cols = st.columns(4)
    seed = aug_cols[0].number_input("Preview seed", 0, 1_000_000, 0, 1)
    shape_pct = aug_cols[1].slider("Shape variation (%)", 0, 40, 12, 1)
    duration_pct = aug_cols[2].slider("Duration variation (%)", 0, 30, 8, 1)
    amplitude_pct = aug_cols[3].slider("Delta amplitude variation (%)", 0, 40, 10, 1)
    st.caption("The seed makes the preview reproducible. Batch export draws a different beat and anatomy for every sample.")

display_model = model
display_spec = spec
if enabled:
    aug = AugmentationSpec(
        samples=1,
        seed=int(seed),
        finger_size_fraction=0.0,
        artery_size_fraction=0.0,
        artery_position_mm=0.0,
        artery_rotation_deg=0.0,
        conductivity_fraction=float(amplitude_pct) / 100.0,
        waveform_shape_fraction=float(shape_pct) / 100.0,
        duration_fraction=float(duration_pct) / 100.0,
    )
    rng = np.random.default_rng(aug.seed)
    display_model = augment_model(model, aug, rng)
    display_spec = augment_waveform(spec, aug, rng)

# Lower grid resolution keeps each animation frame's redraw fast; zsmooth in the
# figure interpolates it back to a smooth image, so playback stays fluid.
result = simulate_grid(display_model, display_spec, 72)
view = st.radio(
    "Conductivity map",
    ["Absolute conductivity σ", "Delta conductivity Δσ"],
    horizontal=True,
    help="Absolute σ shows the whole finger and pulses continuously. Delta Δσ is the "
    "clean GCNM target and is near zero except around systole, so it looks sparse.",
)
st.plotly_chart(
    figures.pulse_figure(result, view, animate=True, height=780),
    use_container_width=True,
)

st.caption(
    "The vertical waveform cursor, simulation frame, time label, and progress slider are one "
    "animation state. Absolute conductivity is σ=σ₀+Δσ; delta conductivity is the clean GCNM target. "
    "All tissue conductivities are specified at 50 kHz."
)
st.info("Pulse spread is restricted to arteries and the configured muscle halo. Skin, fat, bone, and ligament remain unchanged.")
