"""Session-state model management and ring-size scaling shared across pages."""

from __future__ import annotations

from functools import lru_cache
from copy import deepcopy

import numpy as np
import streamlit as st

import lib  # noqa: F401  (adds src/ to sys.path)
from finger_sim.mesh import discover_meshes, load_ring_mesh
from finger_sim.models import FingerModel, WaveformSpec

_MODEL_KEY = "finger_model"
_WAVE_KEY = "waveform_spec"
_SAVED_KEY = "saved_finger_models"

DEFAULT_WAVEFORM = {"kind": "heartbeat", "frames": 50, "duration_s": 1.0, "custom_values": []}

FINGER_SIZE_PRESETS = {
    "Small finger · 14 × 12 mm": (14.0, 12.0),
    "Medium finger · 18 × 16 mm": (18.0, 16.0),
    "Large finger · 22 × 19 mm": (22.0, 19.0),
    "Extra large finger · 26 × 22 mm": (26.0, 22.0),
}


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


_NONCE_KEY = "widget_nonce"


def widget_nonce() -> int:
    """Version stamped into editor widget keys.

    Streamlit derives a keyless widget's identity from its parameters, so a widget
    reverting to a value it already held reuses the old identity — and the stale
    edit with it. Bumping this nonce gives the widgets fresh keys, which is the
    only reliable way to make them re-read a programmatically changed model.
    """
    return int(st.session_state.setdefault(_NONCE_KEY, 0))


def bump_widget_nonce() -> None:
    st.session_state[_NONCE_KEY] = widget_nonce() + 1


def reset_model() -> None:
    """Restore the finger model to the built-in defaults."""
    st.session_state[_MODEL_KEY] = default_model_dict()
    bump_widget_nonce()


def resize_model(model_dict: dict, width_mm: float, height_mm: float) -> dict:
    """Resize the outer finger and every embedded structure coherently."""
    updated = deepcopy(model_dict)
    old_width = float(updated["width_mm"])
    old_height = float(updated["height_mm"])
    sx = float(width_mm) / old_width
    sy = float(height_mm) / old_height
    layer_scale = min(sx, sy)
    updated["width_mm"] = float(width_mm)
    updated["height_mm"] = float(height_mm)
    updated["skin_thickness_mm"] *= layer_scale
    updated["fat_thickness_mm"] *= layer_scale
    for item in [updated["bone"], *updated.get("ligaments", []), *updated["arteries"]]:
        item["center_x_mm"] *= sx
        item["center_y_mm"] *= sy
        item["radius_x_mm"] *= sx
        item["radius_y_mm"] *= sy
    FingerModel.from_dict(updated)
    return updated


def saved_models() -> dict[str, dict]:
    if _SAVED_KEY not in st.session_state:
        st.session_state[_SAVED_KEY] = {}
    return st.session_state[_SAVED_KEY]


def save_model(name: str) -> None:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("model name cannot be empty")
    saved_models()[clean_name] = deepcopy(get_model_dict())


def load_saved_model(name: str) -> None:
    if name not in saved_models():
        raise KeyError(name)
    set_model_dict(deepcopy(saved_models()[name]))


# --------------------------------------------------------------------------- #
# Waveform
# --------------------------------------------------------------------------- #
def get_waveform() -> WaveformSpec:
    if _WAVE_KEY not in st.session_state:
        st.session_state[_WAVE_KEY] = dict(DEFAULT_WAVEFORM)
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


def reset_waveform() -> None:
    """Restore the waveform to the built-in default heartbeat."""
    st.session_state[_WAVE_KEY] = dict(DEFAULT_WAVEFORM)
    bump_widget_nonce()


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
