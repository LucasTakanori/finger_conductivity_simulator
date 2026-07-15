"""Hub: preview the current cross-section and enter one of the three tools."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import figures, nav, state, ui  # noqa: E402
from finger_sim.models import WaveformSpec  # noqa: E402
from finger_sim.simulation import simulate_grid  # noqa: E402

ui.page_header(
    "PVI · EIT · GCNM",
    "Finger conductivity simulator",
    "Build a parametric finger cross-section, drive its arteries with a pulse waveform, "
    "and project the result onto real PVI ring meshes for GCNM training data.",
)

hero, rail = st.columns([3, 1])
with hero:
    model = state.get_model()
    preview = simulate_grid(model, WaveformSpec("heartbeat", 8, 1.0), 96)
    st.plotly_chart(figures.tissue_figure(preview), use_container_width=True)
    st.caption("Live preview of the current finger model. Edit it on the Finger model page.")
with rail:
    st.markdown("**Tissue legend**")
    ui.tissue_rail()

st.write("")
st.markdown("#### Tools")
pages = nav.pages()
cards = [
    ("01", "Finger model", "Drag tissues to reshape, scale to a ring size, watch conductivity update live.", pages["finger"]),
    ("02", "Waveform", "Shape the heartbeat/sine/custom pulse and animate the differential conductivity.", pages["waveform"]),
    ("03", "Mesh & export", "Project onto a PVI ring mesh, see the triangles, export a GCNM-ready NPZ.", pages["mesh"]),
]
for col, (idx, name, desc, page) in zip(st.columns(3), cards):
    with col:
        st.markdown(
            f'<div class="nav-card"><div class="nav-index">{idx}</div>'
            f'<div class="nav-name">{name}</div><div class="nav-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )
        st.page_link(page, label=f"Open {name} →")
