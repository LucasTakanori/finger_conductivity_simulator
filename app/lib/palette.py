"""Single source of truth for the anatomical-atlas visual identity.

Every tissue owns one hue that is reused everywhere: the WebGPU canvas, the
Plotly maps, and the UI legend. Charts and interface never disagree about what
"muscle" looks like.
"""

from __future__ import annotations

# Brand ink / surface tokens (mirrored in CSS in lib.ui).
INK = "#13233B"
MUTED = "#5B6B7F"
PAPER = "#EEF1F4"
SURFACE = "#FFFFFF"
LINE = "#D3DCE3"
ARTERIAL = "#C0304A"  # primary action accent

# Absolute conductivity is sequential: pale neutral -> coral -> deep plum.
# Delta conductivity remains the conventional blue-negative/red-positive map.
ABSOLUTE_SCALE = [
    [0.00, "#F7F3F2"],
    [0.25, "#E8C5C0"],
    [0.50, "#CE817B"],
    [0.75, "#96465A"],
    [1.00, "#3D1938"],
]

# Tissue label integers must match finger_sim.geometry.
OUTSIDE, SKIN, FAT, MUSCLE, BONE, LIGAMENT, ARTERY = range(7)

TISSUE_COLORS = {
    OUTSIDE: "#EEF1F4",
    SKIN: "#D9A066",
    FAT: "#E8CE8A",
    MUSCLE: "#C0556B",
    BONE: "#C3CBD4",
    LIGAMENT: "#3E8E8A",
    ARTERY: "#B7263F",
}

TISSUE_LABELS = {
    OUTSIDE: "Outside",
    SKIN: "Skin",
    FAT: "Fat",
    MUSCLE: "Muscle",
    BONE: "Bone",
    LIGAMENT: "Ligament / tendon",
    ARTERY: "Artery",
}

# Ordered rows for the persistent tissue "spectrum rail" legend.
LEGEND_ORDER = [SKIN, FAT, MUSCLE, BONE, LIGAMENT, ARTERY]


def tissue_colorscale() -> list[list]:
    """Discrete Plotly colorscale spanning labels 0..6 as sharp bands."""
    stops = []
    n = 7
    for label in range(n):
        lo, hi = label / n, (label + 1) / n
        color = TISSUE_COLORS[label]
        stops.append([lo, color])
        stops.append([hi, color])
    return stops


def tissue_palette_for_js() -> dict:
    """Compact payload handed to the WebGPU/WebGL component."""
    return {
        "colors": {str(k): v for k, v in TISSUE_COLORS.items()},
        "names": {str(k): v for k, v in TISSUE_LABELS.items()},
        "ink": INK,
        "line": LINE,
        "arterial": ARTERIAL,
    }
