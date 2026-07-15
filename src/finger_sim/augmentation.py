"""Reproducible anatomy and pulse augmentation for GCNM dataset generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from finger_sim.models import FingerModel, WaveformSpec
from finger_sim.waveforms import build_waveform


@dataclass(frozen=True)
class AugmentationSpec:
    """Maximum independent perturbations around one baseline finger model."""

    samples: int = 10
    seed: int = 0
    finger_size_fraction: float = 0.08
    artery_size_fraction: float = 0.20
    artery_position_mm: float = 1.0
    artery_rotation_deg: float = 15.0
    conductivity_fraction: float = 0.10
    waveform_shape_fraction: float = 0.12
    duration_fraction: float = 0.08

    def validate(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be at least one")
        for name, value in asdict(self).items():
            if name not in {"samples", "seed"} and value < 0:
                raise ValueError(f"{name} cannot be negative")


def _factor(rng: np.random.Generator, fraction: float) -> float:
    return float(rng.uniform(1.0 - fraction, 1.0 + fraction))


def augment_model(
    baseline: FingerModel,
    spec: AugmentationSpec,
    rng: np.random.Generator,
) -> FingerModel:
    """Draw one plausible model variant while preserving valid nesting."""
    data = deepcopy(baseline.to_dict())
    sx = _factor(rng, spec.finger_size_fraction)
    sy = _factor(rng, spec.finger_size_fraction)
    data["width_mm"] *= sx
    data["height_mm"] *= sy
    layer_scale = min(sx, sy)
    data["skin_thickness_mm"] *= layer_scale
    data["fat_thickness_mm"] *= layer_scale

    embedded = [data["bone"], *data.get("ligaments", []), *data["arteries"]]
    for item in embedded:
        item["center_x_mm"] *= sx
        item["center_y_mm"] *= sy
        item["radius_x_mm"] *= sx
        item["radius_y_mm"] *= sy

    for key in list(data["conductivities"]):
        data["conductivities"][key] = max(
            1e-5,
            data["conductivities"][key] * _factor(rng, spec.conductivity_fraction),
        )

    inner_x = data["width_mm"] / 2 - data["skin_thickness_mm"] - data["fat_thickness_mm"]
    inner_y = data["height_mm"] / 2 - data["skin_thickness_mm"] - data["fat_thickness_mm"]
    for artery in data["arteries"]:
        artery["radius_x_mm"] *= _factor(rng, spec.artery_size_fraction)
        artery["radius_y_mm"] *= _factor(rng, spec.artery_size_fraction)
        artery["center_x_mm"] += float(rng.uniform(-spec.artery_position_mm, spec.artery_position_mm))
        artery["center_y_mm"] += float(rng.uniform(-spec.artery_position_mm, spec.artery_position_mm))
        artery["rotation_deg"] += float(rng.uniform(-spec.artery_rotation_deg, spec.artery_rotation_deg))
        artery["baseline_conductivity_s_m"] = max(
            1e-5,
            artery["baseline_conductivity_s_m"] * _factor(rng, spec.conductivity_fraction),
        )
        artery["peak_delta_s_m"] *= _factor(rng, spec.conductivity_fraction)
        margin_x = artery["radius_x_mm"] + 0.2
        margin_y = artery["radius_y_mm"] + 0.2
        artery["center_x_mm"] = float(np.clip(artery["center_x_mm"], -inner_x + margin_x, inner_x - margin_x))
        artery["center_y_mm"] = float(np.clip(artery["center_y_mm"], -inner_y + margin_y, inner_y - margin_y))

    return FingerModel.from_dict(data)


def augment_waveform(
    baseline: WaveformSpec,
    spec: AugmentationSpec,
    rng: np.random.Generator,
) -> WaveformSpec:
    """Perturb beat timing and morphology, then return the actual sampled pulse."""
    _, base = build_waveform(baseline)
    count = len(base)
    phase = np.linspace(0.0, 1.0, count, endpoint=False)

    # A positive exponent delays the systolic rise; a negative one advances it.
    exponent = float(np.exp(rng.uniform(-spec.waveform_shape_fraction, spec.waveform_shape_fraction)))
    warped = np.power(phase, exponent)
    values = np.interp(warped, phase, base, period=1.0)

    # Smooth, low-frequency morphology noise avoids implausible frame-to-frame jitter.
    anchors = max(5, min(12, count // 4))
    anchor_phase = np.linspace(0.0, 1.0, anchors)
    anchor_noise = rng.normal(0.0, spec.waveform_shape_fraction, anchors)
    anchor_noise[-1] = anchor_noise[0]
    noise = np.interp(phase, anchor_phase, anchor_noise)
    values = values * (1.0 + noise)
    if np.nanmin(base) >= 0:
        values = np.clip(values, 0.0, None)
    values -= values[0]
    peak = float(np.max(np.abs(values)))
    if peak > 0:
        values /= peak

    duration = baseline.duration_s * _factor(rng, spec.duration_fraction)
    return WaveformSpec(
        kind="custom",
        frames=baseline.frames,
        duration_s=duration,
        custom_values=values.tolist(),
        normalize="none",
    )
