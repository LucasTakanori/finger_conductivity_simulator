"""Waveform creation and resampling."""

from __future__ import annotations

import numpy as np

from finger_sim.models import WaveformSpec


def _normalize(values: np.ndarray, mode: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if mode == "none":
        return result
    if mode == "minmax":
        span = float(np.max(result) - np.min(result))
        return (result - np.min(result)) / max(span, 1e-12)
    if mode == "peak_abs":
        return result / max(float(np.max(np.abs(result))), 1e-12)
    raise ValueError(f"unknown waveform normalization: {mode}")


def build_waveform(spec: WaveformSpec) -> tuple[np.ndarray, np.ndarray]:
    spec.validate()
    time = np.linspace(0.0, spec.duration_s, spec.frames, endpoint=False)
    phase = time / spec.duration_s
    if spec.kind == "heartbeat":
        systolic = np.exp(-0.5 * ((phase - 0.20) / 0.075) ** 2)
        shoulder = 0.34 * np.exp(-0.5 * ((phase - 0.34) / 0.12) ** 2)
        dicrotic = 0.18 * np.exp(-0.5 * ((phase - 0.58) / 0.045) ** 2)
        values = systolic + shoulder + dicrotic
        values -= values[0]
    elif spec.kind == "sine":
        values = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
    else:
        source = np.asarray(spec.custom_values, dtype=np.float64)
        source_phase = np.linspace(0.0, 1.0, len(source))
        values = np.interp(phase, source_phase, source)
    return time, _normalize(values, spec.normalize)


def phase_shift(values: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    phase = np.arange(len(values), dtype=np.float64) / len(values)
    source = (phase - float(fraction)) % 1.0
    periodic_phase = np.append(phase, 1.0)
    periodic_values = np.append(values, values[0])
    return np.interp(source, periodic_phase, periodic_values)

