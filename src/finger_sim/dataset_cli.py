"""Headless GCNM dataset generator driven by a saved project bundle.

Example — 1000 beats/anatomies from a session saved in the app, no GUI::

    finger-sim-dataset --project my_session.json --samples 1000 --seed 0 \\
        --out exports/finger_dataset_1000.npz

Every augmentation knob can be overridden on the command line; anything left
unset falls back to the value stored in the project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from finger_sim.dataset import generate_augmented_dataset
from finger_sim.export import export_npz
from finger_sim.mesh import discover_meshes, load_ring_mesh
from finger_sim import project as project_io


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, required=True, help="project JSON saved from the app")
    parser.add_argument("--out", type=Path, required=True, help="output NPZ path")
    parser.add_argument("--samples", type=int, default=None, help="number of distinct beats/anatomies")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for a reproducible draw")
    parser.add_argument("--mesh", default=None, help="'grid' or a bundled mesh id; overrides the project")
    parser.add_argument("--grid-size", type=int, default=None, help="Cartesian grid resolution when --mesh grid")
    # Per-variation overrides (all optional; None -> keep the project value).
    parser.add_argument("--finger-size-variation", type=float, default=None)
    parser.add_argument("--finger-rotation-deg", type=float, default=None)
    parser.add_argument("--artery-size-variation", type=float, default=None)
    parser.add_argument("--artery-position-mm", type=float, default=None)
    parser.add_argument("--artery-rotation-deg", type=float, default=None)
    parser.add_argument("--conductivity-variation", type=float, default=None)
    parser.add_argument("--diffusion-variation", type=float, default=None)
    parser.add_argument("--waveform-shape-variation", type=float, default=None)
    parser.add_argument("--duration-variation", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    project = project_io.load_project(args.project)
    model = project_io.model_of(project)
    waveform = project_io.waveform_of(project)
    augmentation = project_io.augmentation_of(
        project,
        samples=args.samples,
        seed=args.seed,
        finger_size_fraction=args.finger_size_variation,
        finger_rotation_deg=args.finger_rotation_deg,
        artery_size_fraction=args.artery_size_variation,
        artery_position_mm=args.artery_position_mm,
        artery_rotation_deg=args.artery_rotation_deg,
        conductivity_fraction=args.conductivity_variation,
        diffusion_fraction=args.diffusion_variation,
        waveform_shape_fraction=args.waveform_shape_variation,
        duration_fraction=args.duration_variation,
    )

    mesh_choice = args.mesh if args.mesh is not None else project.get("mesh")
    grid_size = args.grid_size if args.grid_size is not None else int(project.get("grid_size", 96))
    mesh = None
    if mesh_choice not in (None, "grid"):
        paths = discover_meshes()
        if mesh_choice not in paths:
            raise KeyError(f"unknown mesh {mesh_choice}; choices: {sorted(paths)}")
        mesh = load_ring_mesh(paths[mesh_choice], mesh_choice)

    arrays = generate_augmented_dataset(model, waveform, augmentation, mesh=mesh, grid_size=grid_size)
    output = export_npz(args.out, arrays)
    shape = arrays["delta_sigma"].shape
    print(f"{output.resolve()}  ->  {shape[0]} samples x {shape[1]} frames x {shape[2]} elements")


if __name__ == "__main__":
    main()
