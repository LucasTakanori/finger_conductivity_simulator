"""Generate one reproducible multi-anatomy, multi-beat conductivity dataset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import json

import numpy as np

from finger_sim.augmentation import AugmentationSpec, augment_model, augment_waveform
from finger_sim.mesh import RingMesh, mesh_points_mm
from finger_sim.models import FingerModel, WaveformSpec
from finger_sim.simulation import simulate_grid, simulate_points


def generate_augmented_dataset(
    baseline_model: FingerModel,
    baseline_waveform: WaveformSpec,
    augmentation: AugmentationSpec,
    *,
    mesh: RingMesh | None = None,
    grid_size: int = 40,  # matches the img_size the bundled ring meshes were mapped with
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Return a v2 NPZ payload whose first dimension is augmented sample.

    ``progress`` is called with (completed, total) after each sample so callers can
    report how far a long batch has run.
    """
    augmentation.validate()
    rng = np.random.default_rng(augmentation.seed)
    results = []
    models = []
    waveforms = []
    total = augmentation.samples
    for index in range(total):
        model = augment_model(baseline_model, augmentation, rng)
        waveform = augment_waveform(baseline_waveform, augmentation, rng)
        if mesh is None:
            result = simulate_grid(model, waveform, grid_size)
        else:
            result = simulate_points(mesh_points_mm(mesh, model), model, waveform)
        results.append(result)
        models.append(model)
        waveforms.append(waveform)
        if progress is not None:
            progress(index + 1, total)

    delta = np.stack([r.delta_sigma for r in results]).astype(np.float32)
    rest = np.stack([r.sigma_baseline for r in results]).astype(np.float32)
    absolute = np.stack([r.sigma_dynamic for r in results]).astype(np.float32)
    metadata = {
        "schema_version": "finger-conductivity-v2-batch",
        "target": "clean delta conductivity, not a PVI/Newton reconstruction",
        "frequency_hz": baseline_model.frequency_hz,
        "coordinate_units": "mm",
        "conductivity_units": "S/m",
        "mesh_id": mesh.mesh_id if mesh is not None else None,
        "mesh_manifest": mesh.manifest if mesh is not None else {},
        "augmentation": asdict(augmentation),
        "baseline_finger_model": baseline_model.to_dict(),
        "samples": [
            {
                "finger_model": model.to_dict(),
                "waveform": {
                    "kind": waveform.kind,
                    "frames": waveform.frames,
                    "duration_s": waveform.duration_s,
                    "normalize": waveform.normalize,
                },
            }
            for model, waveform in zip(models, waveforms)
        ],
    }
    return {
        # Clear v2 names.
        "delta_sigma": delta,
        "absolute_conductivity": absolute,
        "resting_absolute_conductivity": rest,
        # Compatibility names used by the existing 2D_GCNM adapter.
        "sigma": delta,
        "sigma_dynamic": absolute,
        "sigma_baseline": np.broadcast_to(rest[:, None, :], delta.shape).copy(),
        "time_s": np.stack([r.time_s for r in results]).astype(np.float32),
        "waveform": np.stack([r.waveform for r in results]).astype(np.float32),
        "tissue_labels": np.stack([r.tissue_labels for r in results]).astype(np.uint8),
        "points_mm": np.stack([r.points_mm for r in results]).astype(np.float32),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
