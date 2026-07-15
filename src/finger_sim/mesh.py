"""Load bundled MATLAB-exported PVI ring meshes and project anatomy onto them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from finger_sim.models import FingerModel


@dataclass
class RingMesh:
    mesh_id: str
    nodes: np.ndarray
    elements: np.ndarray
    source_path: Path
    manifest: dict

    @property
    def centroids(self) -> np.ndarray:
        return self.nodes[self.elements].mean(axis=1)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bundled_mesh_root() -> Path:
    return repository_root() / "meshes"


def discover_meshes(root: Path | None = None) -> dict[str, Path]:
    root = root or bundled_mesh_root()
    meshes: dict[str, Path] = {}
    for path in sorted(root.glob("**/*_inv.h5")):
        meshes[path.parent.name] = path
    return meshes


def load_ring_mesh(path: Path, mesh_id: str | None = None) -> RingMesh:
    path = Path(path)
    with h5py.File(path, "r") as source:
        nodes = np.asarray(source["nodes"], dtype=np.float64)
        elements = np.asarray(source["elems"], dtype=np.int64)
    if nodes.shape[0] == 2:
        nodes = nodes.T
    if elements.shape[0] == 3:
        elements = elements.T
    if elements.min() == 1:
        elements = elements - 1
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return RingMesh(mesh_id or path.parent.name, nodes[:, :2], elements, path, manifest)


def _scale_mesh_coordinates(points: np.ndarray, mesh: RingMesh, model: FingerModel) -> np.ndarray:
    scaled = np.asarray(points, dtype=np.float64).copy()
    center = np.mean(mesh.nodes, axis=0, keepdims=True)
    scaled -= center
    node_centered = mesh.nodes - center
    x_scale = max(float(np.max(np.abs(node_centered[:, 0]))), 1e-12)
    y_scale = max(float(np.max(np.abs(node_centered[:, 1]))), 1e-12)
    scaled[:, 0] *= model.width_mm / (2.0 * x_scale)
    scaled[:, 1] *= model.height_mm / (2.0 * y_scale)
    return scaled


def mesh_points_mm(mesh: RingMesh, model: FingerModel) -> np.ndarray:
    """Map element centroids into the editable finger coordinate system.

    The ring mesh defines topology/electrode geometry. Its centered x/y extent is
    normalized and then scaled by the requested finger width and height.
    """
    return _scale_mesh_coordinates(mesh.centroids, mesh, model)


def mesh_nodes_mm(mesh: RingMesh, model: FingerModel) -> np.ndarray:
    return _scale_mesh_coordinates(mesh.nodes, mesh, model)


def element_to_node_values(mesh: RingMesh, values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(mesh.nodes), dtype=np.float64)
    counts = np.zeros(len(mesh.nodes), dtype=np.float64)
    finite_values = np.nan_to_num(values, nan=0.0)
    for corner in range(mesh.elements.shape[1]):
        np.add.at(result, mesh.elements[:, corner], finite_values)
        np.add.at(counts, mesh.elements[:, corner], np.isfinite(values))
    result /= np.maximum(counts, 1.0)
    result[counts == 0] = np.nan
    return result
