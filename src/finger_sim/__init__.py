"""Parametric finger conductivity and differential-pulse simulator."""

from finger_sim.models import FingerModel, WaveformSpec, default_finger_model
from finger_sim.simulation import SimulationResult, simulate_grid, simulate_points

__all__ = [
    "FingerModel",
    "SimulationResult",
    "WaveformSpec",
    "default_finger_model",
    "simulate_grid",
    "simulate_points",
]

__version__ = "0.1.0"

