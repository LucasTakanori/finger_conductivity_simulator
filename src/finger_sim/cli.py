"""Command-line dataset export for reproducible handoff to 2D_GCNM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = FingerModel.from_dict(json.loads(args.config.read_text()))
    waveform = WaveformSpec(args.waveform, args.frames, args.duration)
    mesh_id = None
    manifest = None
    if args.mesh == "grid":
        result = simulate_grid(model, waveform, args.grid_size)
    else:
        paths = discover_meshes()
        if args.mesh not in paths:
            raise KeyError(f"unknown mesh {args.mesh}; choices: {sorted(paths)}")
        mesh = load_ring_mesh(paths[args.mesh], args.mesh)
        result = simulate_points(mesh_points_mm(mesh, model), model, waveform)
        mesh_id = mesh.mesh_id
        manifest = mesh.manifest
    arrays = arrays_for_export(
        result, model, waveform, mesh_id=mesh_id, mesh_manifest=manifest
    )
    output = export_npz(args.out, arrays)
    print(output.resolve())


if __name__ == "__main__":
    main()
