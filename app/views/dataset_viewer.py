"""Inspect an exported dataset: browse samples and beats, export GIFs and figures."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import dataset_io, palette, ui  # noqa: E402

ui.page_header(
    "Step 04",
    "Dataset viewer",
    "Open an exported NPZ, step through its samples and beats, read back the finger model "
    "behind any sample, and save frames or animations.",
)

EXPORTS = Path(__file__).resolve().parents[2] / "exports"


def _resolve_source() -> str | None:
    """Return a local path to the NPZ, whether typed in or uploaded."""
    tab_path, tab_upload = st.tabs(["Open a file on disk", "Upload"])
    with tab_path:
        found = sorted(str(p) for p in EXPORTS.glob("*.npz")) if EXPORTS.is_dir() else []
        if found:
            picked = st.selectbox("Found in exports/", ["(type a path below)", *found])
            if picked != "(type a path below)":
                return picked
        typed = st.text_input("Path to .npz", placeholder="exports/finger_dataset_1000.npz")
        if typed.strip():
            candidate = Path(typed.strip()).expanduser()
            if not candidate.is_absolute():
                candidate = Path(__file__).resolve().parents[2] / candidate
            if candidate.is_file():
                return str(candidate)
            st.error(f"No such file: {candidate}")
        st.caption("Best for large batches — nothing is copied through the browser.")
    with tab_upload:
        uploaded = st.file_uploader("Upload .npz", type=["npz"])
        if uploaded is not None:
            temp = Path(tempfile.gettempdir()) / f"viewer_{uploaded.name}"
            temp.write_bytes(uploaded.getvalue())
            return str(temp)
        st.caption("Streamlit caps uploads (200 MB by default); use the path tab for big files.")
    return None


source = _resolve_source()
if not source:
    st.info("Choose an exported .npz to begin.")
    st.stop()

mtime = Path(source).stat().st_mtime
summary = dataset_io.read_summary(source, mtime)

top = st.columns(4)
top[0].metric("Samples (beats)", f"{summary['samples']:,}")
top[1].metric("Frames per beat", summary["frames"])
top[2].metric("Points per frame", f"{summary['points']:,}")
top[3].metric("Mesh", summary["mesh_id"] or "Cartesian grid")
st.caption(f"Schema `{summary['schema']}` · file `{Path(source).name}`")

# --- pick a sample ---------------------------------------------------------- #
index = 0
if summary["samples"] > 1:
    index = st.slider("Sample (beat)", 0, summary["samples"] - 1, 0)
sample = dataset_io.read_sample(source, mtime, index)

if sample["points"] is None:
    st.error("This file has no `points_mm`, so the fields cannot be placed in space.")
    st.stop()
plan = dataset_io.raster_plan(sample["points"])

field_names = ["Delta conductivity Δσ", "Absolute conductivity σ", "Resting σ₀", "Tissue map"]
available = [f for f in field_names if not (f == "Absolute conductivity σ" and sample["absolute"] is None)]
controls = st.columns([2, 3, 1, 1])
field = controls[0].selectbox("Field", available)
frame = controls[1].slider("Frame", 0, summary["frames"] - 1, min(summary["frames"] - 1, 0))
if sample["time_s"] is not None:
    controls[2].metric("Time", f"{sample['time_s'][frame]:.3f} s")


def frame_values(index_: int) -> np.ndarray:
    if field.startswith("Delta"):
        return sample["delta"][index_]
    if field.startswith("Absolute"):
        return sample["absolute"][index_]
    if field.startswith("Resting"):
        return sample["resting"]
    return sample["labels"]


def scale_for_field() -> tuple[str, float, float]:
    """Colour key and limits held constant across the whole beat."""
    if field.startswith("Delta"):
        peak = float(np.nanmax(np.abs(sample["delta"]))) or 1e-9
        return "delta", -peak, peak
    if field.startswith("Absolute"):
        return "absolute", float(np.nanmin(sample["absolute"])), float(np.nanmax(sample["absolute"]))
    return "absolute", float(np.nanmin(sample["resting"])), float(np.nanmax(sample["resting"]))


def render(values: np.ndarray) -> np.ndarray:
    grid = dataset_io.to_grid(values, plan)
    if field == "Tissue map":
        return dataset_io.tissue_rgb(grid)
    key, vmin, vmax = scale_for_field()
    return dataset_io.colorize(grid, key, vmin, vmax)


is_static = field in {"Resting σ₀", "Tissue map"}
animate = False
if not is_static:
    animate = controls[3].toggle("▶ Animate", value=False, help="Play the whole beat in place.")
rgb = render(frame_values(frame))
extent = plan["extent"]


def _uri(image: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(dataset_io.png_bytes(image, scale=1)).decode()


def _place(**kwargs) -> go.Image:
    return go.Image(
        x0=extent[0],
        dx=(extent[1] - extent[0]) / rgb.shape[1],
        y0=extent[3],
        dy=(extent[2] - extent[3]) / rgb.shape[0],
        **kwargs,
    )


left, right = st.columns([3, 2], gap="large")
with left:
    if animate:
        with st.spinner("Preparing the beat…"):
            uris = [_uri(render(frame_values(i))) for i in range(summary["frames"])]
        figure = go.Figure(data=[_place(source=uris[0])])
        figure.frames = [go.Frame(name=str(i), data=[go.Image(source=u)]) for i, u in enumerate(uris)]
        figure.update_layout(
            updatemenus=[{
                "type": "buttons", "x": 0.0, "y": -0.08,
                "buttons": [
                    {"label": "▶ Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
            }],
            sliders=[{
                "active": 0, "pad": {"t": 40},
                "currentvalue": {"prefix": "frame "},
                "steps": [
                    {"label": str(i), "method": "animate",
                     "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                    for i in range(summary["frames"])
                ],
            }],
        )
        title = f"{field} · full beat"
    else:
        figure = go.Figure(data=[_place(z=rgb)])
        title = field if is_static else f"{field} · frame {frame}"
    figure.update_layout(
        title={"text": f"Sample {index} · {title}", "font": {"size": 15}},
        height=560, margin={"l": 20, "r": 20, "t": 50, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "color": palette.INK},
    )
    figure.update_xaxes(title="mm", constrain="domain", range=[extent[0], extent[1]])
    # Image traces default to a reversed y-axis, which would draw the finger upside
    # down; pin an ascending range so +y stays at the top.
    figure.update_yaxes(title="mm", scaleanchor="x", autorange=False, range=[extent[2], extent[3]])
    st.plotly_chart(figure, use_container_width=True)
with right:
    if sample["waveform"] is not None:
        beat = go.Figure(
            go.Scatter(x=sample["time_s"], y=sample["waveform"], mode="lines",
                       line={"color": palette.ARTERIAL, "width": 2.5},
                       fill="tozeroy", fillcolor="rgba(192,48,74,0.10)")
        )
        if not is_static:
            beat.add_vline(x=float(sample["time_s"][frame]), line={"color": palette.INK, "width": 2, "dash": "dot"})
        beat.update_layout(
            title={"text": "This sample's beat", "font": {"size": 15}},
            height=260, margin={"l": 20, "r": 20, "t": 45, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis={"title": "Time (s)", "gridcolor": palette.LINE},
            yaxis={"title": "Amplitude", "gridcolor": palette.LINE},
        )
        st.plotly_chart(beat, use_container_width=True)
    st.markdown("**Tissue legend**")
    ui.tissue_rail()

# --- the model behind this sample ------------------------------------------- #
st.markdown("#### The finger model behind sample %d" % index)
model = sample["model"]
if not model:
    st.info("This export carries no per-sample finger model.")
else:
    cond = model.get("conductivities", {})
    facts = st.columns(4)
    facts[0].metric("Width", f"{model['width_mm']:.2f} mm")
    facts[1].metric("Height", f"{model['height_mm']:.2f} mm")
    facts[2].metric("Ring rotation", f"{model['rotation_deg']:.1f} °")
    facts[3].metric("Arteries", len(model.get("arteries", [])))
    rows = [{"parameter": f"conductivity · {k.replace('_s_m','')}", "value": v} for k, v in cond.items()]
    for i, artery in enumerate(model.get("arteries", [])):
        rows += [
            {"parameter": f"artery {i+1} · centre x (mm)", "value": artery["center_x_mm"]},
            {"parameter": f"artery {i+1} · centre y (mm)", "value": artery["center_y_mm"]},
            {"parameter": f"artery {i+1} · radius x (mm)", "value": artery["radius_x_mm"]},
            {"parameter": f"artery {i+1} · radius y (mm)", "value": artery["radius_y_mm"]},
            {"parameter": f"artery {i+1} · rotation (°)", "value": artery["rotation_deg"]},
            {"parameter": f"artery {i+1} · peak Δσ (S/m)", "value": artery["peak_delta_s_m"]},
        ]
    rows += [
        {"parameter": "muscle diffusion fraction", "value": model.get("muscle_diffusion_fraction")},
        {"parameter": "muscle diffusion length (mm)", "value": model.get("muscle_diffusion_length_mm")},
    ]
    with st.expander("Every parameter of this sample"):
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.download_button(
        "Download this sample's model JSON",
        json.dumps({"finger_model": model}, indent=2),
        file_name=f"sample{index}_finger_model.json",
        mime="application/json",
        help="Load it on the Finger model page to rebuild this exact anatomy.",
    )

# --- exports ---------------------------------------------------------------- #
st.markdown("#### Save figures and animations")
save = st.columns([1, 1, 2])
save[0].download_button(
    "Download this frame (PNG)",
    dataset_io.png_bytes(rgb),
    file_name=f"sample{index}_{'static' if is_static else f'frame{frame}'}.png",
    mime="image/png",
    use_container_width=True,
)
fps = save[1].number_input("GIF frames per second", 2, 30, 12, 1, disabled=is_static)
if is_static:
    save[2].caption("Pick Δσ or absolute σ to animate — the resting and tissue maps do not change over the beat.")
elif save[2].button("Build GIF of this beat", type="primary", use_container_width=True):
    bar = st.progress(0.0, text="Rendering frames…")
    frames = []
    for i in range(summary["frames"]):
        frames.append(render(frame_values(i)))
        bar.progress((i + 1) / summary["frames"], text=f"Rendering frame {i + 1} of {summary['frames']}…")
    st.session_state["viewer_gif"] = dataset_io.gif_bytes(frames, fps=int(fps))
    st.session_state["viewer_gif_name"] = f"sample{index}_{field.split()[0].lower()}.gif"
    bar.empty()

if st.session_state.get("viewer_gif"):
    st.image(st.session_state["viewer_gif"], caption="Preview")
    st.download_button(
        "Download GIF",
        st.session_state["viewer_gif"],
        file_name=st.session_state.get("viewer_gif_name", "beat.gif"),
        mime="image/gif",
        type="primary",
    )

st.caption(
    "Δσ and σ use a colour range fixed across the whole beat, so frames and the GIF are "
    "directly comparable. Charts have a camera icon for a PNG of the interactive view."
)
