"""Serializable parameter models for anatomy and waveform generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Ellipse:
    center_x_mm: float
    center_y_mm: float
    radius_x_mm: float
    radius_y_mm: float
    rotation_deg: float = 0.0


@dataclass
class Artery(Ellipse):
    baseline_conductivity_s_m: float = 0.70
    peak_delta_s_m: float = 0.06
    waveform_scale: float = 1.0
    phase_delay_fraction: float = 0.0


@dataclass
class TissueConductivities:
    skin_s_m: float = 0.20
    fat_s_m: float = 0.08
    muscle_s_m: float = 0.36
    bone_s_m: float = 0.04
    ligament_s_m: float = 0.16


@dataclass
class FingerModel:
    width_mm: float = 18.0
    height_mm: float = 16.0
    rotation_deg: float = 0.0
    skin_thickness_mm: float = 0.60
    fat_thickness_mm: float = 1.80
    conductivities: TissueConductivities = field(default_factory=TissueConductivities)
    bone: Ellipse = field(
        default_factory=lambda: Ellipse(0.0, 0.6, 2.7, 2.2, -8.0)
    )
    ligaments: list[Ellipse] = field(
        default_factory=lambda: [
            Ellipse(0.0, -4.0, 2.2, 1.15, 4.0),
            Ellipse(0.0, 4.1, 2.0, 0.85, -3.0),
        ]
    )
    arteries: list[Artery] = field(
        default_factory=lambda: [
            Artery(-3.4, -1.5, 0.65, 0.55, 8.0),
            Artery(3.4, -2.0, 0.72, 0.60, -12.0, peak_delta_s_m=0.055),
        ]
    )
    muscle_diffusion_fraction: float = 0.12
    muscle_diffusion_length_mm: float = 1.25

    def validate(self) -> None:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("finger width and height must be positive")
        if self.skin_thickness_mm < 0 or self.fat_thickness_mm < 0:
            raise ValueError("layer thicknesses cannot be negative")
        if 2 * (self.skin_thickness_mm + self.fat_thickness_mm) >= min(
            self.width_mm, self.height_mm
        ):
            raise ValueError("skin and fat layers leave no muscle region")
        if not self.arteries:
            raise ValueError("at least one artery is required")
        for artery in self.arteries:
            if artery.radius_x_mm <= 0 or artery.radius_y_mm <= 0:
                raise ValueError("artery radii must be positive")
        if not 0.0 <= self.muscle_diffusion_fraction <= 1.0:
            raise ValueError("muscle diffusion fraction must be in [0, 1]")
        if self.muscle_diffusion_length_mm <= 0:
            raise ValueError("muscle diffusion length must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FingerModel":
        data = dict(values)
        data["conductivities"] = TissueConductivities(**data.get("conductivities", {}))
        data["bone"] = Ellipse(**data["bone"])
        data["ligaments"] = [Ellipse(**item) for item in data.get("ligaments", [])]
        data["arteries"] = [Artery(**item) for item in data.get("arteries", [])]
        model = cls(**data)
        model.validate()
        return model


@dataclass
class WaveformSpec:
    kind: str = "heartbeat"
    frames: int = 50
    duration_s: float = 1.0
    custom_values: list[float] = field(default_factory=list)
    normalize: str = "peak_abs"

    def validate(self) -> None:
        if self.frames < 2:
            raise ValueError("waveform requires at least two frames")
        if self.duration_s <= 0:
            raise ValueError("waveform duration must be positive")
        if self.kind not in {"heartbeat", "sine", "custom"}:
            raise ValueError(f"unsupported waveform kind: {self.kind}")
        if self.kind == "custom" and len(self.custom_values) < 2:
            raise ValueError("custom waveform requires at least two values")


def default_finger_model() -> FingerModel:
    return FingerModel()

