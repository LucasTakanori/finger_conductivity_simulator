"""Plotly figure builders shared by the pages, styled with the tissue palette."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

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


def tissue_figure(result: SimulationResult) -> go.Figure:
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
    _base_layout(figure, "Tissue regions")
    _square_axes(figure)
    return figure


def baseline_figure(result: SimulationResult) -> go.Figure:
    shape = result.grid_shape
    z = result.sigma_baseline.reshape(shape)
    figure = go.Figure(
        _heatmap(z, result, "Cividis", float(np.nanmin(z)), float(np.nanmax(z)), None, "S/m")
    )
    _base_layout(figure, "Absolute conductivity σ₀")
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
        zmid, colorscale = None, "Cividis"
        title = "Dynamic conductivity σ₀ + Δσ"

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
            colorscale="RdBu_r" if delta else "Cividis",
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
                line={"color": "rgba(19,35,59,0.35)", "width": 1},
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
