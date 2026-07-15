"""Plotly figure builders shared by the pages, styled with the tissue palette."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lib  # noqa: F401
from finger_sim.mesh import element_to_node_values, mesh_nodes_mm
from finger_sim.models import FingerModel
from finger_sim.simulation import SimulationResult
from lib import palette

_FONT = {"family": "Inter, system-ui, sans-serif", "color": palette.INK}
_PAPER = "rgba(0,0,0,0)"


def _base_layout(figure: go.Figure, title: str, height: int = 520) -> None:
    figure.update_layout(
        title={"text": title, "font": {"size": 15}},
        height=height,
        font=_FONT,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )


def _square_axes(figure: go.Figure) -> None:
    figure.update_xaxes(scaleanchor="y", title="mm", gridcolor=palette.LINE, zeroline=False)
    figure.update_yaxes(title="mm", gridcolor=palette.LINE, zeroline=False)


def _heatmap(z, result, colorscale, zmin, zmax, zmid, unit) -> go.Heatmap:
    return go.Heatmap(
        z=z,
        x=result.grid_x_mm,
        y=result.grid_y_mm,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        zmid=zmid,
        colorbar={"title": unit, "outlinewidth": 0},
        hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>value=%{z:.4f}<extra></extra>",
    )


def _overlay_triangles(figure: go.Figure, mesh_overlay: dict | None) -> None:
    if not mesh_overlay or not mesh_overlay.get("count"):
        return
    nodes = np.asarray(mesh_overlay["nodes"], dtype=float).reshape(-1, 2)
    elements = np.asarray(mesh_overlay["elements"], dtype=int).reshape(-1, 3)
    xs, ys = _triangle_edges(nodes, elements)
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "rgba(19,35,59,0.62)", "width": 1.0},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def tissue_figure(
    result: SimulationResult,
    *,
    mesh_overlay: dict | None = None,
    height: int = 520,
    title: str = "Tissue regions",
) -> go.Figure:
    shape = result.grid_shape
    z = result.tissue_labels.reshape(shape).astype(float)
    z[z == palette.OUTSIDE] = np.nan
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=result.grid_x_mm,
            y=result.grid_y_mm,
            colorscale=palette.tissue_colorscale(),
            zmin=0,
            zmax=7,
            showscale=False,
            hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<extra></extra>",
        )
    )
    _overlay_triangles(figure, mesh_overlay)
    _base_layout(figure, title, height=height)
    _square_axes(figure)
    return figure


def baseline_figure(
    result: SimulationResult,
    *,
    mesh_overlay: dict | None = None,
    height: int = 520,
) -> go.Figure:
    shape = result.grid_shape
    z = result.sigma_baseline.reshape(shape)
    figure = go.Figure(
        _heatmap(z, result, palette.ABSOLUTE_SCALE, float(np.nanmin(z)), float(np.nanmax(z)), None, "S/m")
    )
    _overlay_triangles(figure, mesh_overlay)
    _base_layout(figure, "Resting absolute conductivity σ₀ · 50 kHz", height=height)
    _square_axes(figure)
    return figure


def conductivity_figure(
    result: SimulationResult,
    values: np.ndarray,
    title: str,
    *,
    delta: bool,
    mesh_overlay: dict | None = None,
    height: int = 560,
) -> go.Figure:
    """One registered conductivity frame on a Cartesian finger grid."""
    z = np.asarray(values).reshape(result.grid_shape)
    if delta:
        limit = max(float(np.nanmax(np.abs(z))), 1e-9)
        colorscale, zmin, zmax, zmid = "RdBu_r", -limit, limit, 0.0
    else:
        colorscale = palette.ABSOLUTE_SCALE
        zmin, zmax, zmid = float(np.nanmin(z)), float(np.nanmax(z)), None
    figure = go.Figure(_heatmap(z, result, colorscale, zmin, zmax, zmid, "S/m"))
    _overlay_triangles(figure, mesh_overlay)
    _base_layout(figure, title, height=height)
    _square_axes(figure)
    return figure


def waveform_figure(result: SimulationResult) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=result.time_s,
            y=result.waveform,
            mode="lines",
            line={"color": palette.ARTERIAL, "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(192,48,74,0.10)",
        )
    )
    figure.update_layout(
        height=220,
        font=_FONT,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        xaxis={"title": "Time (s)", "gridcolor": palette.LINE},
        yaxis={"title": "Normalized amplitude", "gridcolor": palette.LINE},
    )
    return figure


def pulse_figure(
    result: SimulationResult,
    view: str = "Delta conductivity",
    *,
    animate: bool = True,
    height: int = 720,
) -> go.Figure:
    """Waveform and map in one animation so cursor, slider, and frame cannot drift."""
    shape = result.grid_shape
    is_delta = view.startswith("Delta")
    if is_delta:
        source = result.delta_sigma
        peak = max(float(np.nanmax(np.abs(source))), 1e-9)
        zmin, zmax, zmid, colorscale = -peak, peak, 0.0, "RdBu_r"
        map_title = "Delta conductivity Δσ"
    else:
        source = result.sigma_dynamic
        zmin, zmax = float(np.nanmin(source)), float(np.nanmax(source))
        zmid, colorscale = None, palette.ABSOLUTE_SCALE
        map_title = "Absolute conductivity σ = σ₀ + Δσ"

    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.28, 0.72],
        vertical_spacing=0.10,
        subplot_titles=("Input pulse waveform", map_title),
    )
    figure.add_trace(
        go.Scatter(
            x=result.time_s,
            y=result.waveform,
            mode="lines",
            line={"color": palette.ARTERIAL, "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(192,48,74,0.10)",
            hovertemplate="t=%{x:.3f} s<br>amplitude=%{y:.3f}<extra></extra>",
            name="waveform",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[result.time_s[0], result.time_s[0]],
            y=[float(np.nanmin(result.waveform)), float(np.nanmax(result.waveform))],
            mode="lines",
            line={"color": palette.INK, "width": 2, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        _heatmap(source[0].reshape(shape), result, colorscale, zmin, zmax, zmid, "S/m"),
        row=2,
        col=1,
    )

    ymin = float(np.nanmin(result.waveform))
    ymax = float(np.nanmax(result.waveform))
    figure.frames = [
        go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=[result.time_s[i], result.time_s[i]], y=[ymin, ymax]),
                _heatmap(source[i].reshape(shape), result, colorscale, zmin, zmax, zmid, "S/m"),
            ],
            traces=[1, 2],
        )
        for i in range(len(source))
    ]
    steps = [
        {
            "method": "animate",
            "label": f"{result.time_s[i]:.2f}",
            "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        }
        for i in range(len(source))
    ]
    controls = []
    if animate:
        controls = [
            {
                "type": "buttons",
                "x": 0.0,
                "y": -0.04,
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 70, "redraw": True}, "fromcurrent": True}]},
                    {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
            }
        ]
    figure.update_layout(
        height=height,
        font=_FONT,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
        margin={"l": 20, "r": 20, "t": 45, "b": 110},
        showlegend=False,
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Simulation time: ", "suffix": " s"},
            "pad": {"t": 48},
            "steps": steps,
            "x": 0.0,
            "len": 1.0,
        }],
        updatemenus=controls,
    )
    figure.update_xaxes(title_text="Waveform time (s)", gridcolor=palette.LINE, row=1, col=1)
    figure.update_yaxes(title_text="Normalized amplitude", gridcolor=palette.LINE, row=1, col=1)
    figure.update_xaxes(title_text="mm", scaleanchor="y2", gridcolor=palette.LINE, row=2, col=1)
    figure.update_yaxes(title_text="mm", gridcolor=palette.LINE, row=2, col=1)
    return figure


def dynamic_figure(result: SimulationResult, view: str, animate: bool) -> go.Figure:
    """Animated pulse map: time scrubber + a legend fixed across the whole beat."""
    shape = result.grid_shape
    if view == "Differential conductivity":
        source = result.delta_sigma
        peak = float(np.nanmax(np.abs(source)))
        peak = peak if peak > 0 else 1e-9
        zmin, zmax, zmid, colorscale = -peak, peak, 0.0, "RdBu_r"
        title = "Differential conductivity Δσ"
    else:
        source = result.sigma_dynamic
        zmin, zmax = float(np.nanmin(source)), float(np.nanmax(source))
        zmid, colorscale = None, palette.ABSOLUTE_SCALE
        title = "Absolute conductivity σ = σ₀ + Δσ"

    def trace(index: int) -> go.Heatmap:
        return _heatmap(source[index].reshape(shape), result, colorscale, zmin, zmax, zmid, "S/m")

    figure = go.Figure(data=[trace(0)])
    figure.frames = [go.Frame(data=[trace(i)], name=str(i)) for i in range(len(source))]

    steps = [
        {
            "method": "animate",
            "label": f"{result.time_s[i]:.3f}",
            "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        }
        for i in range(len(source))
    ]
    sliders = [{"active": 0, "currentvalue": {"prefix": "t = ", "suffix": " s"}, "pad": {"t": 45}, "steps": steps}]
    updatemenus = []
    if animate:
        updatemenus = [
            {
                "type": "buttons",
                "x": 0.0,
                "y": -0.05,
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 70, "redraw": True}, "fromcurrent": True}]},
                    {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
            }
        ]
    _base_layout(figure, title, height=560)
    _square_axes(figure)
    figure.update_layout(sliders=sliders, updatemenus=updatemenus)
    return figure


def _triangle_edges(nodes: np.ndarray, elements: np.ndarray) -> tuple[list, list]:
    """Return x/y line-segment lists (None-separated) for every triangle edge."""
    xs: list = []
    ys: list = []
    for tri in elements:
        loop = [tri[0], tri[1], tri[2], tri[0]]
        xs.extend([nodes[i, 0] for i in loop] + [None])
        ys.extend([nodes[i, 1] for i in loop] + [None])
    return xs, ys


def mesh_field_figure(
    mesh,
    model: FingerModel,
    values: np.ndarray,
    title: str,
    delta: bool,
    vlim: float | None = None,
    show_edges: bool = True,
) -> go.Figure:
    """Conductivity coloured on the ring mesh with the triangle grid drawn over it."""
    nodes = mesh_nodes_mm(mesh, model)
    node_values = element_to_node_values(mesh, values)
    limit = vlim if vlim is not None else float(np.nanmax(np.abs(node_values)))
    figure = go.Figure(
        go.Mesh3d(
            x=nodes[:, 0],
            y=nodes[:, 1],
            z=np.zeros(len(nodes)),
            i=mesh.elements[:, 0],
            j=mesh.elements[:, 1],
            k=mesh.elements[:, 2],
            intensity=node_values,
            colorscale="RdBu_r" if delta else palette.ABSOLUTE_SCALE,
            cmin=(-limit if delta else None),
            cmax=(limit if delta else None),
            flatshading=True,
            showscale=True,
            colorbar={"title": "S/m", "outlinewidth": 0},
        )
    )
    if show_edges:
        xs, ys = _triangle_edges(nodes, mesh.elements)
        figure.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=[0.01] * len(xs),
                mode="lines",
                line={"color": "rgba(19,35,59,0.78)", "width": 2.2},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.update_layout(
        title={"text": title, "font": {"size": 15}},
        height=620,
        font=_FONT,
        paper_bgcolor=_PAPER,
        scene={
            "camera": {"eye": {"x": 0, "y": 0, "z": 2.2}, "up": {"x": 0, "y": 1, "z": 0}},
            "aspectmode": "data",
            "xaxis_title": "mm",
            "yaxis_title": "mm",
            "zaxis": {"visible": False},
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return figure
