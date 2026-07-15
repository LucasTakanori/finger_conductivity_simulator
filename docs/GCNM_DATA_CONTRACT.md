# 2D_GCNM data contract

## Export schema

Every exported `.npz` contains:

| Array | Shape for mesh export | Meaning |
|---|---:|---|
| `sigma` | `(T, K)` | Clean differential conductivity target \(\Delta\sigma_t\) |
| `sigma_baseline` | `(T, K)` | Absolute baseline conductivity repeated for each frame |
| `sigma_dynamic` | `(T, K)` | \(\sigma_0 + \Delta\sigma_t\) |
| `time_s` | `(T,)` | Frame time in seconds |
| `waveform` | `(T,)` | Normalized input waveform |
| `tissue_labels` | `(K,)` | Integer tissue class per element |
| `points_mm` | `(K, 2)` | Inverse-element centroid coordinates |
| `metadata_json` | scalar string | Schema, units, model parameters, and mesh provenance |

Here, `T` is the waveform frame count and `K` is the selected inverse-mesh
element count. For a Cartesian export, `K = H × W`; `grid_shape` can be
reconstructed from the chosen export configuration. Mesh exports are the
preferred handoff for 2D_GCNM.

## Required forward-model adapter

The simulator deliberately does not invent boundary voltages. The next adapter
must evaluate the matching PVI forward mesh and append:

| Array | Shape | Meaning |
|---|---:|---|
| `V` | `(T, M)` | Noisy differential voltages used by training |
| `V_clean` | `(T, M)` | Clean differential voltages |
| `newton` | `(T, K)` | Optional PVI-style physics feature, never the target |

The clean training target remains `sigma`. PVI/Newton images must not replace
it as supervision.

## Tissue labels

| ID | Tissue |
|---:|---|
| 0 | Outside |
| 1 | Skin |
| 2 | Fat |
| 3 | Muscle |
| 4 | Bone |
| 5 | Ligament/tendon |
| 6 | Artery |

## Compatibility checks before training

1. Confirm `K` matches the inverse mesh.
2. Confirm the forward and inverse meshes belong to the same bundle.
3. Confirm conductivity is in S/m and coordinates are in mm.
4. Confirm `sigma_dynamic - sigma_baseline == sigma` within float tolerance.
5. Group all frames from the same anatomy into one dataset split.
6. Preserve `metadata_json` when adding voltages.

