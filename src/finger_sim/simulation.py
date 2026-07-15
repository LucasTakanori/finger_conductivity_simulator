"""Absolute and waveform-driven differential conductivity simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finger_sim.geometry import MUSCLE, TissueField, classify_points, grid_points
from finger_sim.models import FingerModel, WaveformSpec
from finger_sim.waveforms import build_waveform, phase_shift


@dataclass
class SimulationResult:
    points_mm: np.ndarray
    time_s: np.ndarray
    waveform: np.ndarray
    tissue_labels: np.ndarray
    sigma_baseline: np.ndarray
    delta_sigma: np.ndarray
    sigma_dynamic: np.ndarray
    grid_x_mm: np.ndarray | None = None
    grid_y_mm: np.ndarray | None = None
    grid_shape: tuple[int, int] | None = None


def _delta_from_field(
    field: TissueField,
    model: FingerModel,
    waveform: np.ndarray,
) -> np.ndarray:
    delta = np.zeros((len(waveform), len(field.labels)), dtype=np.float64)
    muscle = field.labels == MUSCLE
    for artery, mask, halo in zip(
        model.arteries, field.artery_masks, field.artery_halos
    ):
        pulse = phase_shift(waveform, artery.phase_delay_fraction)
        pulse = pulse * artery.waveform_scale
        spatial = np.zeros(len(field.labels), dtype=np.float64)
        spatial[mask] = artery.peak_delta_s_m
        spatial[muscle] += (
            artery.peak_delta_s_m
            * model.muscle_diffusion_fraction
            * halo[muscle]
        )
        delta += pulse[:, None] * spatial[None, :]
    delta[:, field.labels == 0] = np.nan
    return delta


def simulate_points(
    points_mm: np.ndarray,
    model: FingerModel,
    waveform_spec: WaveformSpec,
) -> SimulationResult:
    field = classify_points(points_mm, model)
    time, waveform = build_waveform(waveform_spec)
    delta = _delta_from_field(field, model, waveform)
    dynamic = field.baseline[None, :] + delta
    return SimulationResult(
        points_mm=np.asarray(points_mm, dtype=np.float64),
        time_s=time,
        waveform=waveform,
        tissue_labels=field.labels,
        sigma_baseline=field.baseline,
        delta_sigma=delta,
        sigma_dynamic=dynamic,
    )


def simulate_grid(
    model: FingerModel,
    waveform_spec: WaveformSpec,
    size: int = 96,
) -> SimulationResult:
    x, y, points = grid_points(model, size)
    result = simulate_points(points, model, waveform_spec)
    result.grid_x_mm = x
    result.grid_y_mm = y
    result.grid_shape = (size, size)
    return result

