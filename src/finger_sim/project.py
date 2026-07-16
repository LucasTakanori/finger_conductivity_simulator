"""Portable project bundle: one finger model + one waveform + generation settings.

A project is the single artifact the app saves and the headless dataset CLI
consumes, so a GCNM dataset can be regenerated with no graphical interface.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

from finger_sim.augmentation import AugmentationSpec
from finger_sim.models import FingerModel, WaveformSpec

SCHEMA = "finger-sim-project-v1"


def _waveform_dict(waveform: WaveformSpec) -> dict[str, Any]:
    return {
        "kind": waveform.kind,
        "frames": waveform.frames,
        "duration_s": waveform.duration_s,
        "custom_values": list(waveform.custom_values),
        "normalize": waveform.normalize,
    }


def build_project(
    model: FingerModel,
    waveform: WaveformSpec,
    *,
    mesh: str | None = None,
    grid_size: int = 96,
    augmentation: AugmentationSpec | None = None,
) -> dict[str, Any]:
    """Assemble a serializable project dict from live objects."""
    model.validate()
    waveform.validate()
    augmentation = augmentation or AugmentationSpec()
    augmentation.validate()
    return {
        "schema": SCHEMA,
        "finger_model": model.to_dict(),
        "waveform": _waveform_dict(waveform),
        "mesh": mesh,  # bundled mesh id, or None / "grid" for a Cartesian grid
        "grid_size": int(grid_size),
        "augmentation": asdict(augmentation),
    }


def dumps_project(project: dict[str, Any]) -> str:
    return json.dumps(project, indent=2)


def save_project(path: str | Path, project: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_project(project))
    return path


def loads_project(text: str) -> dict[str, Any]:
    project = json.loads(text)
    if "finger_model" not in project or "waveform" not in project:
        raise ValueError("not a finger-sim project: missing finger_model/waveform")
    # Validate the embedded objects early so bad files fail loudly.
    model_of(project)
    waveform_of(project)
    return project


def load_project(path: str | Path) -> dict[str, Any]:
    return loads_project(Path(path).read_text())


def model_of(project: dict[str, Any]) -> FingerModel:
    return FingerModel.from_dict(project["finger_model"])


def waveform_of(project: dict[str, Any]) -> WaveformSpec:
    data = project["waveform"]
    spec = WaveformSpec(
        kind=data["kind"],
        frames=int(data["frames"]),
        duration_s=float(data["duration_s"]),
        custom_values=list(data.get("custom_values", [])),
        normalize=data.get("normalize", "peak_abs"),
    )
    spec.validate()
    return spec


def augmentation_of(project: dict[str, Any], **overrides: Any) -> AugmentationSpec:
    """Rebuild the augmentation, applying any non-None overrides."""
    base = AugmentationSpec(**project.get("augmentation", {}))
    clean = {key: value for key, value in overrides.items() if value is not None}
    spec = replace(base, **clean) if clean else base
    spec.validate()
    return spec
