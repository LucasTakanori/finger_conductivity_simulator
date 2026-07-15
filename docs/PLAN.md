# Finger conductivity webapp and dataset-factory plan

## 1. Goal

Build a desktop web application that creates a controllable 2D finger slice,
assigns conductivity to every tissue, drives arterial differential
conductivity with an arbitrary waveform, projects the result onto a selected
PVI mesh, and exports clean training targets for 2D_GCNM.

The application separates three objects that must not be conflated:

1. **Anatomy:** geometry and absolute tissue conductivity \(\sigma_0\).
2. **Physiology:** waveform-driven conductivity change \(\Delta\sigma(t)\).
3. **Acquisition physics:** FEM voltages, Newton features, and noise generated
   later with the selected electrode/ring mesh.

The first two are implemented in this repository. The third is the explicit
integration boundary with the PVI solver and 2D_GCNM.

## 2. User workflow

1. Open the Streamlit application on a normal PC.
2. Choose a small/medium/large finger preset or custom elliptical dimensions.
3. Set skin/fat thickness and every tissue conductivity.
4. Move and resize bone, ligament/tendon, and artery regions.
5. Select a heartbeat/sine waveform or upload arbitrary CSV/TXT samples.
6. Set artery peak \(\Delta\sigma\), phase delay, and muscle coupling.
7. Inspect the waveform and conductivity map on one synchronized time control.
8. Select a bundled ring mesh and inspect the FEM element projection.
9. Download one sequence or an augmented batch of distinct beats/anatomies.
10. Feed that sequence to a PVI forward-model adapter to create voltages and
    then assemble a 2D_GCNM training pack.

## 3. Current architecture

```text
Streamlit controls
      |
      v
FingerModel + WaveformSpec
      |
      +--> analytic tissue classifier --> sigma_baseline
      |
      +--> arterial pulse + muscle-only halo --> delta_sigma(t)
      |
      +--> Cartesian preview
      |
      +--> FEM element-centroid projection
      |
      v
versioned NPZ export
      |
      v
PVI forward solver (next integration) --> V(t) --> 2D_GCNM pack
```

The simulation core evaluates arbitrary point coordinates. Cartesian grids and
FEM elements therefore use the same tissue equations rather than two separate
rasterization implementations.

## 4. Implemented MVP

### Anatomy

- Elliptical finger boundary with independent width/height and rotation.
- Skin and subcutaneous-fat shells.
- Muscle interior.
- Editable bone ellipse.
- Two editable ligament/tendon structures.
- One or two independently editable arteries.
- Independent conductivity for each tissue.

### Physiology

- Heartbeat, sine, and arbitrary sampled waveform.
- Per-artery peak conductivity change and phase delay.
- Muscle-only Gaussian halo with editable strength and length scale.
- Exact masking that prevents diffusion into skin, fat, bone, or ligament.

### Mesh and export

- All 38 b035/b045 ring variants, plus subject-selected US120 and US140 aliases,
  with inverse, forward, mapping, and provenance files.
- Element-centroid tissue projection.
- Interactive mesh preview.
- NPZ export and reproducible CLI export.
- Portable model JSON plus browser-session save/load.
- Reproducible augmented batches with independent finger size, artery geometry,
  conductivity, waveform morphology, and beat-duration variation.

## 5. Phase 2: make it a training dataset factory

### 5.1 Domain-randomization specification

Implemented distributions and UI/CLI controls:

- Finger size and eccentricity.
- Tissue and arterial conductivity.
- Artery count, location, radius, aspect ratio, and rotation.
- Pulse amplitude, duration, and smooth morphology variation.

Still to add:

- Non-elliptical boundary deformation and randomized tissue thickness.
- Bone/ligament geometry distributions.
- Ring rotation, electrode placement jitter, contact impedance, and gain.

Every generated sample must save both the sampled parameters and a deterministic
random seed.

### 5.2 Batch generator

The mesh/export page and CLI now generate many anatomies in one versioned NPZ.
A recommended
first learning curve is 512, 2,000, 8,000, and 32,000 unique anatomy/ring
combinations. Split by base anatomy and mesh/ring configuration, never by
cardiac frame, so waveform variants cannot leak into validation.

### 5.3 Forward physics adapter

Create a separate adapter that consumes the exported mesh-aligned
`sigma_baseline` and `sigma_dynamic` arrays and calls the existing complete
electrode PVI FEM solver:

\[
V_0 = F_m(\sigma_0), \qquad
V_t = F_m(\sigma_0 + \Delta\sigma_t), \qquad
\Delta V_t = V_t - V_0.
\]

The output pack should contain `sigma`, `sigma_baseline`, `V`, `V_clean`, and
optional `newton`, matching the current 2D_GCNM naming convention. FEM solving
should remain outside the browser process so long jobs can be parallelized.

## 6. Phase 3: finger- and ring-agnostic GCNM

The present 2D_GCNM trainer uses one runtime and one graph. Generalization
requires batches containing different `edge_index`, coordinates, and matching
physics. Recommended changes:

- Canonicalize coordinates relative to electrode 1.
- Attach electrode-distance or Jacobian-sensitivity node features.
- Train across 32--64 mesh/ring configurations.
- Consider sharing one GCN block across Newton stages.
- Hold out complete mesh geometries and ring positions for testing.
- Report clean synthetic accuracy separately from real PVI pseudo-reference
  agreement.

## 7. Phase 4: improved physiology

Replace the Gaussian muscle halo only after obtaining evidence for a better
model. Candidate upgrades are:

- An anisotropic diffusion equation inside muscle.
- Separate arterial wall, lumen, capillary bed, and venous compartments.
- Tissue-dependent phase delay.
- Pressure/volume-to-conductivity calibration from phantom measurements.
- Co-registration with ultrasound for vessel location and radius.

Any such model needs explicit units, boundary conditions, and parameter
provenance.

## 8. Validation gates

### Software invariants

- Tissue masks are mutually exclusive.
- Baseline conductivity is finite inside the finger.
- `sigma_dynamic = sigma_baseline + sigma` for every frame.
- Pulse diffusion is exactly zero in skin, fat, bone, and ligament/tendon.
- Mesh export length equals its inverse element count.
- Export metadata identifies mesh provenance and schema version.

### Scientific evidence

- Test on exact-nonlinear FEM voltages, not only linearized data.
- Use unseen anatomy and unseen ring geometry holdouts.
- Compare multiple random seeds.
- Validate physical phantoms with known vessel positions.
- Use real subjects only as pseudo-reference/physical-consistency evidence
  until co-registered anatomical ground truth is available.

## 9. GitHub milestones

1. **v0.1:** interactive model, bundled meshes, and NPZ export.
2. **v0.2:** parameter-distribution editor and batch generation (implemented).
3. **v0.3:** PVI forward-solver adapter and voltage/noise generation.
4. **v0.4:** multi-mesh dataset manifest and grouped splits.
5. **v1.0:** documented phantom validation and stable 2D_GCNM handoff.
