"""Analytic tissue geometry evaluated on arbitrary 2-D points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finger_sim.models import Artery, Ellipse, FingerModel


OUTSIDE = 0
SKIN = 1
FAT = 2
MUSCLE = 3
BONE = 4
LIGAMENT = 5
ARTERY = 6

TISSUE_NAMES = {
    OUTSIDE: "outside",
    SKIN: "skin",
    FAT: "fat",
    MUSCLE: "muscle",
    BONE: "bone",
    LIGAMENT: "ligament/tendon",
    ARTERY: "artery",
}


@dataclass
class TissueField:
    labels: np.ndarray
    baseline: np.ndarray
    local_points_mm: np.ndarray
    artery_masks: list[np.ndarray]
    artery_halos: list[np.ndarray]


def _rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, sine], [-sine, cosine]])
    return np.asarray(points, dtype=np.float64) @ rotation.T


def ellipse_radius(points: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    centered = np.asarray(points) - np.array(
        [ellipse.center_x_mm, ellipse.center_y_mm]
    )
    local = _rotate(centered, ellipse.rotation_deg)
    return np.sqrt(
        (local[:, 0] / ellipse.radius_x_mm) ** 2
        + (local[:, 1] / ellipse.radius_y_mm) ** 2
    )


def ellipse_mask(points: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    return ellipse_radius(points, ellipse) <= 1.0


def classify_points(points_mm: np.ndarray, model: FingerModel) -> TissueField:
    model.validate()
    local = _rotate(points_mm, model.rotation_deg)
    x, y = local[:, 0], local[:, 1]
    outer_r = np.sqrt((x / (model.width_mm / 2)) ** 2 + (y / (model.height_mm / 2)) ** 2)
    skin_rx = model.width_mm / 2 - model.skin_thickness_mm
    skin_ry = model.height_mm / 2 - model.skin_thickness_mm
    fat_rx = skin_rx - model.fat_thickness_mm
    fat_ry = skin_ry - model.fat_thickness_mm
    inside_skin = (x / skin_rx) ** 2 + (y / skin_ry) ** 2 <= 1.0
    inside_fat = (x / fat_rx) ** 2 + (y / fat_ry) ** 2 <= 1.0

    labels = np.full(len(local), OUTSIDE, dtype=np.uint8)
    labels[outer_r <= 1.0] = SKIN
    labels[inside_skin] = FAT
    labels[inside_fat] = MUSCLE
    labels[ellipse_mask(local, model.bone)] = BONE
    for ligament in model.ligaments:
        labels[ellipse_mask(local, ligament)] = LIGAMENT

    artery_masks: list[np.ndarray] = []
    artery_halos: list[np.ndarray] = []
    for artery in model.arteries:
        radius = ellipse_radius(local, artery)
        mask = radius <= 1.0
        effective_radius = np.sqrt(artery.radius_x_mm * artery.radius_y_mm)
        distance_from_wall = np.maximum(radius - 1.0, 0.0) * effective_radius
        halo = np.exp(
            -0.5 * (distance_from_wall / model.muscle_diffusion_length_mm) ** 2
        )
        artery_masks.append(mask)
        artery_halos.append(halo)
        labels[mask] = ARTERY

    c = model.conductivities
    baseline = np.full(len(local), np.nan, dtype=np.float64)
    baseline[labels == SKIN] = c.skin_s_m
    baseline[labels == FAT] = c.fat_s_m
    baseline[labels == MUSCLE] = c.muscle_s_m
    baseline[labels == BONE] = c.bone_s_m
    baseline[labels == LIGAMENT] = c.ligament_s_m
    for artery, mask in zip(model.arteries, artery_masks):
        baseline[mask] = artery.baseline_conductivity_s_m
    return TissueField(labels, baseline, local, artery_masks, artery_halos)


def grid_points(model: FingerModel, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if size < 16:
        raise ValueError("grid size must be at least 16")
    extent = 0.58 * np.hypot(model.width_mm, model.height_mm)
    x = np.linspace(-extent, extent, size)
    y = np.linspace(-extent, extent, size)
    xx, yy = np.meshgrid(x, y)
    return x, y, np.column_stack([xx.ravel(), yy.ravel()])

