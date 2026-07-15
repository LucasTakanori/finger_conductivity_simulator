"""Versioned NPZ export shared by the webapp, CLI, and 2D_GCNM adapter."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np

from finger_sim.models import FingerModel, WaveformSpec
from finger_sim.simulation import SimulationResult


SCHEMA_VERSION = "finger-conductivity-v1"


def arrays_for_export(
    result: SimulationResult,
    model: FingerModel,
    waveform: WaveformSpec,
    *,
    mesh_id: str | None,
    mesh_manifest: dict | None = None,
) -> dict[str, np.ndarray]:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "target": "clean differential conductivity, not a PVI/Newton image",
        "coordinate_units": "mm",
        "conductivity_units": "S/m",
        "mesh_id": mesh_id,
        "mesh_manifest": mesh_manifest or {},
        "finger_model": model.to_dict(),
        "waveform": {
            "kind": waveform.kind,
            "frames": waveform.frames,
            "duration_s": waveform.duration_s,
            "normalize": waveform.normalize,
        },
    }
    return {
        "sigma": result.delta_sigma.astype(np.float32),
        "sigma_baseline": np.broadcast_to(
            result.sigma_baseline[None, :], result.delta_sigma.shape
        ).astype(np.float32),
        "sigma_dynamic": result.sigma_dynamic.astype(np.float32),
        "time_s": result.time_s.astype(np.float32),
        "waveform": result.waveform.astype(np.float32),
        "tissue_labels": result.tissue_labels.astype(np.uint8),
        "points_mm": result.points_mm.astype(np.float32),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }


def export_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def export_npz(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path
