"""Waveform: shape the pulse and animate the differential conductivity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import figures, state, ui  # noqa: E402
from finger_sim.models import WaveformSpec  # noqa: E402
from finger_sim.simulation import simulate_grid  # noqa: E402

ui.page_header(
    "Step 02",
    "Waveform & pulse response",
    "Choose the arterial pulse, then scrub the beat. The colour legend stays fixed across "
    "every frame so you can compare magnitudes over time.",
)

model = state.get_model()

controls = st.columns([1, 1, 1, 2])
kind = controls[0].selectbox("Waveform", ["heartbeat", "sine", "custom"])
frames = controls[1].number_input("Frames", 10, 200, 50, 5)
duration = controls[2].number_input("Duration (s)", 0.1, 10.0, 1.0, 0.1)
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
result = simulate_grid(model, spec, 96)

st.plotly_chart(figures.waveform_figure(result), use_container_width=True)

view = st.radio("Map", ["Differential conductivity", "Dynamic conductivity"], horizontal=True)
animate = st.checkbox("Show play button", value=True)
st.plotly_chart(figures.dynamic_figure(result, view, animate), use_container_width=True)

st.caption("Δσ is masked to arteries plus a muscle-only Gaussian halo. Skin, fat, bone, and ligament receive no pulse.")
