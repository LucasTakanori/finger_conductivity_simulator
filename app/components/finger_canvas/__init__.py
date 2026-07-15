"""Bidirectional Streamlit component: draggable, GPU-rendered finger cross-section.

The frontend (``frontend/index.html`` + ``finger_canvas.js``) renders the tissue
field on the GPU (WebGPU when available, WebGL2 otherwise), lets the user drag
tissue ellipses in real time, and returns the edited model back to Python.

No npm build step: the frontend implements the Streamlit component postMessage
protocol in vanilla JS and is served straight from ``frontend/``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND = Path(__file__).parent / "frontend"
_component = components.declare_component("finger_canvas", path=str(_FRONTEND))


def finger_canvas(
    model: dict,
    palette: dict,
    *,
    color_by: str = "tissue",
    mesh_overlay: dict | None = None,
    height: int = 560,
    key: str | None = None,
):
    """Render the interactive canvas.

    Returns the edited model dict when the user changes geometry, else ``None``.
    """
    return _component(
        model=model,
        palette=palette,
        color_by=color_by,
        mesh_overlay=mesh_overlay,
        height=height,
        key=key,
        default=None,
    )
