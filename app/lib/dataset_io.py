"""Read exported NPZ datasets back in, rasterize fields, and build GIFs/PNGs.

Handles both export schemas: the single-simulation export (2-D fields) and the
augmented batch export (3-D, sample-major). Everything is normalised to the
batch view so the viewer only deals with one shape.
"""

from __future__ import annotations

from io import BytesIO
import json

import numpy as np
from PIL import Image
from plotly.colors import sample_colorscale
import streamlit as st

import lib  # noqa: F401
from lib import palette

_MASK_RGB = (247, 249, 251)  # outside-the-finger pixels


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _metadata_of(archive) -> dict:
    if "metadata_json" not in archive.files:
        return {}
    raw = archive["metadata_json"]
    return json.loads(str(raw))


@st.cache_data(show_spinner=False)
def read_summary(path: str, mtime: float) -> dict:
    """Cheap header: shapes, schema, sample/frame counts, metadata."""
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        delta = archive["delta_sigma"] if "delta_sigma" in keys else archive["sigma"]
        batched = delta.ndim == 3
        samples = int(delta.shape[0]) if batched else 1
        frames = int(delta.shape[1]) if batched else int(delta.shape[0])
        points = int(delta.shape[-1])
        metadata = _metadata_of(archive)
    return {
        "keys": keys,
        "batched": batched,
        "samples": samples,
        "frames": frames,
        "points": points,
        "metadata": metadata,
        "schema": metadata.get("schema_version", "unknown"),
        "mesh_id": metadata.get("mesh_id"),
    }


@st.cache_data(show_spinner=False)
def read_sample(path: str, mtime: float, index: int) -> dict:
    """All arrays for one sample, normalised to (frames, points)."""
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)

        def pick(*names):
            for name in names:
                if name in keys:
                    return archive[name]
            return None

        delta = pick("delta_sigma", "sigma")
        absolute = pick("absolute_conductivity", "sigma_dynamic")
        resting = pick("resting_absolute_conductivity", "sigma_baseline")
        labels = pick("tissue_labels")
        points = pick("points_mm")
        time_s = pick("time_s")
        waveform = pick("waveform")
        metadata = _metadata_of(archive)

    batched = delta.ndim == 3
    take = (lambda a: a[index]) if batched else (lambda a: a)

    delta = take(delta)
    absolute = take(absolute) if absolute is not None else None
    resting = take(resting) if resting is not None else None
    # sigma_baseline is stored broadcast across frames in some exports.
    if resting is not None and resting.ndim == 2:
        resting = resting[0]
    labels = take(labels) if labels is not None else None
    points = take(points) if points is not None else None
    time_s = take(time_s) if time_s is not None else None
    waveform = take(waveform) if waveform is not None else None

    model = None
    entries = metadata.get("samples")
    if entries and index < len(entries):
        model = entries[index].get("finger_model")
    elif metadata.get("finger_model"):
        model = metadata["finger_model"]

    return {
        "delta": np.asarray(delta, dtype=np.float32),
        "absolute": None if absolute is None else np.asarray(absolute, dtype=np.float32),
        "resting": None if resting is None else np.asarray(resting, dtype=np.float32),
        "labels": None if labels is None else np.asarray(labels),
        "points": None if points is None else np.asarray(points, dtype=np.float64),
        "time_s": None if time_s is None else np.asarray(time_s, dtype=np.float64),
        "waveform": None if waveform is None else np.asarray(waveform, dtype=np.float64),
        "model": model,
    }


# --------------------------------------------------------------------------- #
# Rasterizing point fields to an image grid
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def raster_plan(points: np.ndarray, size: int = 240) -> dict:
    """Map sample points onto an image grid.

    Cartesian-grid exports reshape exactly. Mesh exports (element centroids) get a
    nearest-centroid lookup, which is geometry-only and therefore reused for every
    frame of the sample.
    """
    x, y = points[:, 0], points[:, 1]
    ux, uy = np.unique(x), np.unique(y)
    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    if ux.size * uy.size == points.shape[0] and ux.size > 1 and uy.size > 1:
        shape = (int(uy.size), int(ux.size))
        # Grid exports ravel y ascending, but images draw row 0 at the top. Read the
        # stored ordering rather than assuming it, then flip so both raster kinds
        # agree that row 0 is max y.
        rows_y = y.reshape(shape)
        flip = bool(rows_y[0, 0] < rows_y[-1, 0])
        return {"kind": "grid", "shape": shape, "extent": extent, "flip": flip}

    gx = np.linspace(extent[0], extent[1], size)
    gy = np.linspace(extent[3], extent[2], size)  # top row = max y
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    flat = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    nearest = np.empty(flat.shape[0], dtype=np.int32)
    distance = np.empty(flat.shape[0], dtype=np.float64)
    step = 4096  # chunked so the distance matrix stays small
    for start in range(0, flat.shape[0], step):
        block = flat[start : start + step]
        d2 = ((block[:, None, 0] - x[None, :]) ** 2) + ((block[:, None, 1] - y[None, :]) ** 2)
        nearest[start : start + step] = np.argmin(d2, axis=1)
        distance[start : start + step] = np.sqrt(d2[np.arange(len(block)), nearest[start : start + step]])

    # Mask only pixels that fall well outside the sampled geometry. The cutoff has
    # to follow the spacing *between sample points*, not the pixel size: mesh
    # centroids sit ~sqrt(area/count) apart, which is far coarser than one pixel.
    area = max((extent[1] - extent[0]) * (extent[3] - extent[2]), 1e-12)
    spacing = np.sqrt(area / max(points.shape[0], 1))
    cutoff = 1.5 * spacing
    return {
        "kind": "nearest",
        "shape": (size, size),
        "extent": extent,
        "index": nearest,
        "outside": distance > cutoff,
    }


def to_grid(values: np.ndarray, plan: dict) -> np.ndarray:
    """Point values -> 2-D array following the raster plan."""
    if plan["kind"] == "grid":
        grid = np.asarray(values, dtype=np.float64).reshape(plan["shape"])
        return grid[::-1] if plan.get("flip") else grid
    picked = np.asarray(values, dtype=np.float64)[plan["index"]]
    picked[plan["outside"]] = np.nan
    return picked.reshape(plan["shape"])


# --------------------------------------------------------------------------- #
# Colour + image output
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _lut(colorscale_key: str, steps: int = 256) -> np.ndarray:
    scale = palette.ABSOLUTE_SCALE if colorscale_key == "absolute" else "RdBu_r"
    colors = sample_colorscale(scale, np.linspace(0.0, 1.0, steps), colortype="tuple")
    return (np.asarray(colors) * 255.0).round().astype(np.uint8)


def colorize(grid: np.ndarray, colorscale_key: str, vmin: float, vmax: float) -> np.ndarray:
    lut = _lut(colorscale_key)
    span = max(vmax - vmin, 1e-12)
    normalized = np.clip((grid - vmin) / span, 0.0, 1.0)
    index = np.nan_to_num(normalized, nan=0.0) * (len(lut) - 1)
    rgb = lut[index.astype(np.int32)]
    rgb[~np.isfinite(grid)] = _MASK_RGB
    return rgb


def tissue_rgb(grid: np.ndarray) -> np.ndarray:
    """Colour a tissue-label grid with the app's tissue palette."""
    lut = np.array(
        [_hex_rgb(palette.TISSUE_COLORS[i]) for i in range(palette.N_TISSUES)], dtype=np.uint8
    )
    safe = np.nan_to_num(grid, nan=0.0).astype(np.int32)
    safe = np.clip(safe, 0, palette.N_TISSUES - 1)
    rgb = lut[safe]
    rgb[~np.isfinite(grid)] = _MASK_RGB
    return rgb


def _hex_rgb(value: str) -> tuple[int, int, int]:
    number = int(value.lstrip("#"), 16)
    return ((number >> 16) & 255, (number >> 8) & 255, number & 255)


def png_bytes(rgb: np.ndarray, scale: int = 2) -> bytes:
    image = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def gif_bytes(frames: list[np.ndarray], fps: int = 12, scale: int = 2) -> bytes:
    images = []
    for frame in frames:
        image = Image.fromarray(frame, mode="RGB")
        if scale > 1:
            image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
        images.append(image.convert("P", palette=Image.ADAPTIVE, colors=256))
    buffer = BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=max(int(1000 / max(fps, 1)), 20),
        loop=0,
        disposal=2,
    )
    return buffer.getvalue()
