"""Hub: preview the current cross-section and enter one of the three tools."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import figures, nav, state, ui  # noqa: E402
from finger_sim.simulation import simulate_grid  # noqa: E402

ui.page_header(
    "PVI · EIT · GCNM",
    "Finger conductivity simulator",
    "Build a parametric finger cross-section, drive its arteries with a pulse waveform, "
    "and project the result onto real PVI ring meshes for GCNM training data.",
)

model = state.get_model()
spec = state.get_waveform()
preview = simulate_grid(model, spec, 96)

model_side, pulse_side = st.columns([1, 1], gap="large")
with model_side:
    st.markdown("#### Current finger cross-section")
    drawing, rail = st.columns([4, 1])
    with drawing:
        st.plotly_chart(
            figures.tissue_figure(preview, height=480, title="Finger anatomy"),
            use_container_width=True,
        )
    with rail:
        st.markdown("**Tissues**")
        ui.tissue_rail()
    st.caption("Elliptical finger preview at 50 kHz. Edit geometry and conductivity on the Finger model page.")
with pulse_side:
    st.markdown("#### Pulse and conductivity replay")
    st.plotly_chart(
        figures.pulse_figure(preview, "Delta conductivity", height=590),
        use_container_width=True,
    )

st.write("")
st.markdown("#### Tools")
pages = nav.pages()
cards = [
    ("01", "Finger model", "Edit tissue and artery geometry and 50 kHz conductivity, with live maps.", pages["finger"]),
    ("02", "Waveform", "Shape the heartbeat/sine/custom pulse and replay the conductivity.", pages["waveform"]),
    ("03", "Image export", "Export the conductivity image — one beat, or a reproducible augmented batch.", pages["mesh"]),
    ("04", "Dataset viewer", "Reopen an exported NPZ: browse samples and beats, save frames and GIFs.", pages["viewer"]),
]
for col, (idx, name, desc, page) in zip(st.columns(len(cards)), cards):
    with col:
        st.page_link(page, label=f"{idx} · {name} →", help=desc, use_container_width=True)
        st.caption(desc)
