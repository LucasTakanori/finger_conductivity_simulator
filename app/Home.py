"""Interactive Streamlit application for finger conductivity simulation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from finger_sim.export import arrays_for_export, export_bytes
from finger_sim.mesh import (
    discover_meshes,
    element_to_node_values,
    load_ring_mesh,
    mesh_nodes_mm,
    mesh_points_mm,
)
from finger_sim.models import (
    Artery,
    Ellipse,
    FingerModel,
    TissueConductivities,
    WaveformSpec,
)
from finger_sim.simulation import SimulationResult, simulate_grid, simulate_points


st.set_page_config(
    page_title="Finger conductivity simulator",
    page_icon="🫀",
    layout="wide",
)


def number(label: str, value: float, *, minimum: float, maximum: float, step: float) -> float:
    return float(
        st.number_input(label, min_value=minimum, max_value=maximum, value=value, step=step)
    )


def build_model() -> FingerModel:
    with st.sidebar:
        st.header("Finger anatomy")
        width = st.slider("Width (mm)", 10.0, 30.0, 18.0, 0.25)
        height = st.slider("Height (mm)", 8.0, 28.0, 16.0, 0.25)
        rotation = st.slider("Finger rotation (degrees)", -180.0, 180.0, 0.0, 1.0)
        skin = st.slider("Skin thickness (mm)", 0.2, 2.0, 0.6, 0.05)
        fat = st.slider("Fat thickness (mm)", 0.2, 4.0, 1.8, 0.05)

        with st.expander("Tissue conductivity (S/m)", expanded=False):
            conductivities = TissueConductivities(
                skin_s_m=number("Skin", 0.20, minimum=0.001, maximum=2.0, step=0.01),
                fat_s_m=number("Fat", 0.08, minimum=0.001, maximum=2.0, step=0.01),
                muscle_s_m=number("Muscle", 0.36, minimum=0.001, maximum=2.0, step=0.01),
                bone_s_m=number("Bone", 0.04, minimum=0.001, maximum=2.0, step=0.01),
                ligament_s_m=number("Ligament / tendon", 0.16, minimum=0.001, maximum=2.0, step=0.01),
            )

        with st.expander("Bone and ligament geometry", expanded=False):
            bone = Ellipse(
                st.slider("Bone x (mm)", -5.0, 5.0, 0.0, 0.1),
                st.slider("Bone y (mm)", -5.0, 5.0, 0.6, 0.1),
                st.slider("Bone radius x (mm)", 0.5, 5.0, 2.7, 0.1),
                st.slider("Bone radius y (mm)", 0.5, 5.0, 2.2, 0.1),
                st.slider("Bone rotation (degrees)", -180.0, 180.0, -8.0, 1.0),
            )
            ligament_radius = st.slider("Ligament/tendon radius (mm)", 0.3, 3.0, 1.15, 0.05)
            ligaments = [
                Ellipse(0.0, -4.0, 1.9 * ligament_radius, ligament_radius, 4.0),
                Ellipse(0.0, 4.1, 1.7 * ligament_radius, 0.75 * ligament_radius, -3.0),
            ]

        st.header("Arteries")
        artery_count = st.radio("Number of arteries", [1, 2], index=1, horizontal=True)
        arteries: list[Artery] = []
        defaults = [(-3.4, -1.5, 0.65, 0.55, 0.060), (3.4, -2.0, 0.72, 0.60, 0.055)]
        for index in range(artery_count):
            dx, dy, drx, dry, ddelta = defaults[index]
            with st.expander(f"Artery {index + 1}", expanded=index == 0):
                arteries.append(
                    Artery(
                        st.slider(f"A{index + 1} x (mm)", -7.0, 7.0, dx, 0.1),
                        st.slider(f"A{index + 1} y (mm)", -7.0, 7.0, dy, 0.1),
                        st.slider(f"A{index + 1} radius x (mm)", 0.2, 2.5, drx, 0.05),
                        st.slider(f"A{index + 1} radius y (mm)", 0.2, 2.5, dry, 0.05),
                        st.slider(f"A{index + 1} rotation (degrees)", -180.0, 180.0, 0.0, 1.0),
                        baseline_conductivity_s_m=number(
                            f"A{index + 1} baseline conductivity",
                            0.70,
                            minimum=0.01,
                            maximum=2.0,
                            step=0.01,
                        ),
                        peak_delta_s_m=number(
                            f"A{index + 1} peak delta",
                            ddelta,
                            minimum=-0.5,
                            maximum=0.5,
                            step=0.005,
                        ),
                        waveform_scale=1.0,
                        phase_delay_fraction=st.slider(
                            f"A{index + 1} phase delay", 0.0, 0.5, 0.0, 0.01
                        ),
                    )
                )

        st.header("Muscle coupling")
        diffusion_fraction = st.slider("Diffusion fraction", 0.0, 0.5, 0.12, 0.01)
        diffusion_length = st.slider("Diffusion length (mm)", 0.2, 5.0, 1.25, 0.05)

    model = FingerModel(
        width_mm=width,
        height_mm=height,
        rotation_deg=rotation,
        skin_thickness_mm=skin,
        fat_thickness_mm=fat,
        conductivities=conductivities,
        bone=bone,
        ligaments=ligaments,
        arteries=arteries,
        muscle_diffusion_fraction=diffusion_fraction,
        muscle_diffusion_length_mm=diffusion_length,
    )
    model.validate()
    return model


def waveform_controls() -> WaveformSpec:
    st.subheader("Pulse waveform")
    columns = st.columns([1, 1, 1, 2])
    kind = columns[0].selectbox("Waveform", ["heartbeat", "sine", "custom"])
    frames = columns[1].number_input("Frames", 10, 200, 50, 5)
    duration = columns[2].number_input("Duration (s)", 0.1, 10.0, 1.0, 0.1)
    custom_values: list[float] = []
    if kind == "custom":
        uploaded = columns[3].file_uploader("Upload CSV/TXT", type=["csv", "txt"])
        text = columns[3].text_area(
            "Or paste samples",
            "0, 0.2, 0.9, 1.0, 0.6, 0.3, 0.15, 0.05, 0",
        )
        if uploaded is not None:
            text = uploaded.getvalue().decode("utf-8")
        custom_values = np.fromstring(
            text.replace(";", ",").replace("\n", ","), sep=","
        ).tolist()
        if len(custom_values) < 2:
            st.error("Custom waveform needs at least two comma-separated values.")
            st.stop()
    return WaveformSpec(kind, int(frames), float(duration), custom_values)


def grid_figure(result: SimulationResult, view: str, frame: int, animate: bool) -> go.Figure:
    shape = result.grid_shape
    assert shape is not None
    if view == "Tissue labels":
        values = result.tissue_labels.reshape(shape)
        colorscale = "Turbo"
        midpoint = None
        title = "Tissue regions"
    elif view == "Baseline conductivity":
        values = result.sigma_baseline.reshape(shape)
        colorscale = "Viridis"
        midpoint = None
        title = "Baseline conductivity σ₀"
    elif view == "Differential conductivity":
        values = result.delta_sigma[frame].reshape(shape)
        colorscale = "RdBu_r"
        midpoint = 0.0
        title = f"Differential conductivity Δσ, t={result.time_s[frame]:.3f} s"
    else:
        values = result.sigma_dynamic[frame].reshape(shape)
        colorscale = "Viridis"
        midpoint = None
        title = f"Dynamic conductivity σ₀+Δσ, t={result.time_s[frame]:.3f} s"
    heatmap = go.Heatmap(
        z=values,
        x=result.grid_x_mm,
        y=result.grid_y_mm,
        colorscale=colorscale,
        zmid=midpoint,
        colorbar={"title": "S/m" if view != "Tissue labels" else "label"},
        hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>value=%{z:.4f}<extra></extra>",
    )
    figure = go.Figure(data=[heatmap])
    if animate and view in {"Differential conductivity", "Dynamic conductivity"}:
        source = result.delta_sigma if view == "Differential conductivity" else result.sigma_dynamic
        figure.frames = [
            go.Frame(
                data=[go.Heatmap(z=values_frame.reshape(shape))],
                name=str(index),
            )
            for index, values_frame in enumerate(source)
        ]
        figure.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play pulse",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 70, "redraw": True}}],
                        }
                    ],
                }
            ]
        )
    figure.update_layout(
        title=title,
        height=650,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis={"scaleanchor": "y", "title": "mm"},
        yaxis={"title": "mm"},
    )
    return figure


def mesh_figure(mesh, model: FingerModel, values: np.ndarray, title: str, delta: bool) -> go.Figure:
    nodes = mesh_nodes_mm(mesh, model)
    node_values = element_to_node_values(mesh, values)
    figure = go.Figure(
        go.Mesh3d(
            x=nodes[:, 0],
            y=nodes[:, 1],
            z=np.zeros(len(nodes)),
            i=mesh.elements[:, 0],
            j=mesh.elements[:, 1],
            k=mesh.elements[:, 2],
            intensity=node_values,
            colorscale="RdBu_r" if delta else "Viridis",
            cmin=(-np.nanmax(np.abs(node_values)) if delta else None),
            cmax=(np.nanmax(np.abs(node_values)) if delta else None),
            flatshading=True,
            showscale=True,
            colorbar={"title": "S/m"},
        )
    )
    figure.update_layout(
        title=title,
        height=650,
        scene={
            "camera": {"eye": {"x": 0, "y": 0, "z": 2.4}},
            "aspectmode": "data",
            "xaxis_title": "mm",
            "yaxis_title": "mm",
            "zaxis": {"visible": False},
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return figure


st.title("Interactive finger conductivity simulator")
st.caption(
    "Define a parametric finger slice, assign tissue conductivities, drive arterial "
    "Δσ with an arbitrary waveform, and project the result onto a bundled PVI ring mesh."
)

model = build_model()
waveform_spec = waveform_controls()
grid_size = st.select_slider("Preview resolution", [48, 64, 80, 96, 128], value=96)
grid_result = simulate_grid(model, waveform_spec, int(grid_size))

waveform_figure = go.Figure(
    go.Scatter(x=grid_result.time_s, y=grid_result.waveform, mode="lines", name="input")
)
waveform_figure.update_layout(
    height=220,
    margin={"l": 20, "r": 20, "t": 10, "b": 20},
    xaxis_title="Time (s)",
    yaxis_title="Normalized amplitude",
)
st.plotly_chart(waveform_figure, use_container_width=True)

controls = st.columns([2, 2, 1])
view = controls[0].selectbox(
    "Map",
    ["Tissue labels", "Baseline conductivity", "Differential conductivity", "Dynamic conductivity"],
    index=2,
)
frame = controls[1].slider("Time frame", 0, waveform_spec.frames - 1, 0)
animate = controls[2].checkbox("Animation", value=True)

mesh_paths = discover_meshes()
mesh_choice = st.selectbox("Projection/export target", ["Cartesian grid", *mesh_paths.keys()])

left, right = st.columns(2 if mesh_choice != "Cartesian grid" else 1)
left.plotly_chart(grid_figure(grid_result, view, frame, animate), use_container_width=True)

export_result = grid_result
mesh_id = None
mesh_manifest = None
if mesh_choice != "Cartesian grid":
    mesh = load_ring_mesh(mesh_paths[mesh_choice], mesh_choice)
    mesh_result = simulate_points(mesh_points_mm(mesh, model), model, waveform_spec)
    mesh_id = mesh.mesh_id
    mesh_manifest = mesh.manifest
    export_result = mesh_result
    if view == "Tissue labels":
        mesh_values = mesh_result.tissue_labels
        is_delta = False
    elif view == "Baseline conductivity":
        mesh_values = mesh_result.sigma_baseline
        is_delta = False
    elif view == "Differential conductivity":
        mesh_values = mesh_result.delta_sigma[frame]
        is_delta = True
    else:
        mesh_values = mesh_result.sigma_dynamic[frame]
        is_delta = False
    right.plotly_chart(
        mesh_figure(mesh, model, mesh_values, f"{mesh_choice} mesh projection", is_delta),
        use_container_width=True,
    )
    status = mesh_manifest.get("selection", {}).get("status", "bundled")
    st.info(
        f"Mesh: {mesh_choice} · {len(mesh.elements):,} elements · provenance: {status}"
    )

arrays = arrays_for_export(
    export_result,
    model,
    waveform_spec,
    mesh_id=mesh_id,
    mesh_manifest=mesh_manifest,
)
filename = f"finger_delta_{mesh_id or 'grid'}_{waveform_spec.frames}frames.npz"
st.download_button(
    "Download GCNM-ready NPZ",
    export_bytes(arrays),
    file_name=filename,
    mime="application/octet-stream",
    type="primary",
)

with st.expander("Export contents and scientific assumptions"):
    st.markdown(
        """
The NPZ contains `sigma` (clean Δσ target), `sigma_baseline`, `sigma_dynamic`,
`time_s`, `waveform`, `tissue_labels`, element/grid coordinates, and JSON metadata.
The muscle response is a configurable Gaussian spatial halo masked strictly to
muscle. Skin, fat, bone, and ligament/tendon receive no pulse diffusion.
This is a synthetic tissue model, not a validated physiological digital twin.
"""
    )

