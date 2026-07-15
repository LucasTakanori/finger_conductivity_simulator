"""Session-state model management and ring-size scaling shared across pages."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import streamlit as st

import lib  # noqa: F401  (adds src/ to sys.path)
from finger_sim.mesh import discover_meshes, load_ring_mesh
from finger_sim.models import FingerModel, WaveformSpec

_MODEL_KEY = "finger_model"
_WAVE_KEY = "waveform_spec"


# --------------------------------------------------------------------------- #
# Finger model
# --------------------------------------------------------------------------- #
def default_model_dict() -> dict:
    return FingerModel().to_dict()


def get_model_dict() -> dict:
    if _MODEL_KEY not in st.session_state:
        st.session_state[_MODEL_KEY] = default_model_dict()
    return st.session_state[_MODEL_KEY]


def set_model_dict(values: dict) -> None:
    # Validate before storing so a bad drag can never poison later pages.
    FingerModel.from_dict(values)
    st.session_state[_MODEL_KEY] = values


def get_model() -> FingerModel:
    return FingerModel.from_dict(get_model_dict())


# --------------------------------------------------------------------------- #
# Waveform
# --------------------------------------------------------------------------- #
def get_waveform() -> WaveformSpec:
    if _WAVE_KEY not in st.session_state:
        st.session_state[_WAVE_KEY] = {
            "kind": "heartbeat",
            "frames": 50,
            "duration_s": 1.0,
            "custom_values": [],
        }
    data = st.session_state[_WAVE_KEY]
    return WaveformSpec(
        kind=data["kind"],
        frames=int(data["frames"]),
        duration_s=float(data["duration_s"]),
        custom_values=list(data.get("custom_values", [])),
    )


def set_waveform(spec: WaveformSpec) -> None:
    st.session_state[_WAVE_KEY] = {
        "kind": spec.kind,
        "frames": spec.frames,
        "duration_s": spec.duration_s,
        "custom_values": list(spec.custom_values),
    }


# --------------------------------------------------------------------------- #
# Ring meshes and finger-to-ring scaling
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def mesh_catalog() -> dict:
    return {name: str(path) for name, path in discover_meshes().items()}


@lru_cache(maxsize=64)
def _load_cached(name: str):
    return load_ring_mesh(mesh_catalog()[name], name)


def load_mesh(name: str):
    return _load_cached(name)


def ring_diameter_mm(name: str) -> float:
    """Inner diameter of a ring mesh in millimetres (rings are circular)."""
    mesh = load_mesh(name)
    center = mesh.nodes.mean(axis=0, keepdims=True)
    radius = float(np.max(np.linalg.norm(mesh.nodes - center, axis=1)))
    return 2.0 * radius


def scale_model_to_ring(model_dict: dict, name: str, fill: float = 1.0) -> dict:
    """Resize the finger so its outer boundary fits the chosen ring."""
    diameter = ring_diameter_mm(name) * fill
    updated = dict(model_dict)
    updated["width_mm"] = round(diameter, 3)
    updated["height_mm"] = round(diameter, 3)
    return updated


def mesh_overlay_payload(name: str, model: FingerModel) -> dict:
    """Nodes (in finger coords) + triangle indices for canvas / Plotly overlay."""
    from finger_sim.mesh import mesh_nodes_mm

    mesh = load_mesh(name)
    nodes = mesh_nodes_mm(mesh, model)
    return {
        "nodes": nodes.astype(np.float32).ravel().tolist(),
        "elements": mesh.elements.astype(np.int32).ravel().tolist(),
        "count": int(len(mesh.elements)),
    }
