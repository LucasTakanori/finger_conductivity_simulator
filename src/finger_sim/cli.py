"""Command-line dataset export for reproducible handoff to 2D_GCNM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finger_sim.augmentation import AugmentationSpec
from finger_sim.dataset import generate_augmented_dataset
from finger_sim.export import arrays_for_export, export_npz
from finger_sim.mesh import discover_meshes, load_ring_mesh, mesh_points_mm
from finger_sim.models import FingerModel, WaveformSpec
from finger_sim.simulation import simulate_grid, simulate_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default_finger.json"))
    parser.add_argument("--mesh", default="grid", help="grid or bundled mesh folder name")
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--waveform", choices=["heartbeat", "sine"], default="heartbeat")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=1, help="augmented samples in one NPZ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--finger-size-variation", type=float, default=0.08)
    parser.add_argument("--artery-size-variation", type=float, default=0.20)
    parser.add_argument("--artery-position-mm", type=float, default=1.0)
    parser.add_argument("--artery-rotation-deg", type=float, default=15.0)
    parser.add_argument("--conductivity-variation", type=float, default=0.10)
    parser.add_argument("--waveform-shape-variation", type=float, default=0.12)
    parser.add_argument("--duration-variation", type=float, default=0.08)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = FingerModel.from_dict(json.loads(args.config.read_text()))
    waveform = WaveformSpec(args.waveform, args.frames, args.duration)
    mesh_id = None
    manifest = None
    mesh = None
    if args.mesh != "grid":
        paths = discover_meshes()
        if args.mesh not in paths:
            raise KeyError(f"unknown mesh {args.mesh}; choices: {sorted(paths)}")
        mesh = load_ring_mesh(paths[args.mesh], args.mesh)
        mesh_id = mesh.mesh_id
        manifest = mesh.manifest

    if args.samples > 1:
        augmentation = AugmentationSpec(
            samples=args.samples,
            seed=args.seed,
            finger_size_fraction=args.finger_size_variation,
            artery_size_fraction=args.artery_size_variation,
            artery_position_mm=args.artery_position_mm,
            artery_rotation_deg=args.artery_rotation_deg,
            conductivity_fraction=args.conductivity_variation,
            waveform_shape_fraction=args.waveform_shape_variation,
            duration_fraction=args.duration_variation,
        )
        arrays = generate_augmented_dataset(
            model, waveform, augmentation, mesh=mesh, grid_size=args.grid_size
        )
    else:
        if mesh is None:
            result = simulate_grid(model, waveform, args.grid_size)
        else:
            result = simulate_points(mesh_points_mm(mesh, model), model, waveform)
        arrays = arrays_for_export(
            result, model, waveform, mesh_id=mesh_id, mesh_manifest=manifest
        )
    output = export_npz(args.out, arrays)
    print(output.resolve())


if __name__ == "__main__":
    main()
