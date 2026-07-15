# Finger Conductivity Simulator

An interactive, parametric finger cross-section simulator for peripheral
vascular impedance imaging (PVI) and 2D graph convolutional Newton methods
(GCNM). The application creates an absolute tissue conductivity model and a
time-varying differential conductivity sequence driven by a heartbeat, sine,
or user-supplied waveform.

The webapp is intended to run on a normal Windows, macOS, or Linux PC. It is
not tied to an HPC cluster.

## What is implemented

The webapp is a hub (`app/Home.py`) plus three tool pages:

1. **Finger model** — a reliable high-resolution Plotly cross-section driven by
   explicit geometry controls. Choose a small/medium/large elliptical finger or
   custom dimensions; edit tissue/artery geometry and 50 kHz conductivity; draw
   any bundled FEM mesh directly over both the anatomy and conductivity maps;
   and save/load models in the browser or as portable JSON.
2. **Waveform** — heartbeat, sine, or arbitrary CSV/TXT pulse, with synchronized
   waveform cursor, playback frame, time label, and progress slider. Preview
   waveform and pulse-amplitude augmentation before export.
3. **Mesh & export** — projects conductivity onto a PVI ring mesh with strongly
   visible triangles, exports one simulation, or generates a seeded multi-sample
   NPZ with different finger sizes, artery geometry, conductivity, and beats.

Other capabilities:

- Independent skin, fat, muscle, bone, ligament/tendon, and arterial
  conductivities; editable skin/fat thickness and finger rotation.
- Artery position, elliptical radius, rotation, baseline conductivity, pulse
  amplitude, and phase delay.
- A configurable muscle-only pulse halo. Skin, fat, bone, and ligament/tendon
  are explicitly excluded from diffusion.
- Projection onto all 38 PVI FEM mesh variants from the b035/b045 collections,
  plus the subject-selected US120 and US140 aliases.
- A command-line exporter for reproducible single or augmented-batch generation.

The starting anatomy was informed by
[`docs/reference/Dit.png`](docs/reference/Dit.png), but the simulator uses
analytic geometry so every region can be resized, moved, rotated, randomized,
and evaluated on a FEM mesh.

## Install on a PC

```bash
git clone <your-github-url>/Finger-Conductivity-Simulator.git
cd Finger-Conductivity-Simulator
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install and launch:

```bash
python -m pip install --upgrade pip
pip install -e .
streamlit run app/Home.py
```

The browser normally opens automatically at `http://localhost:8501`.
Convenience launchers are also provided:

```bash
# macOS/Linux
bash scripts/run_app.sh

# Windows
scripts\run_app.bat
```

## Command-line export

Export a 50-frame heartbeat directly onto the confirmed subject-006 US120
inverse mesh:

```bash
finger-sim-export \
  --config configs/default_finger.json \
  --mesh subject006_US120 \
  --waveform heartbeat \
  --frames 50 \
  --out exports/subject006_US120_heartbeat.npz
```

Generate ten distinct finger/beat samples with reproducible default variation:

```bash
finger-sim-export \
  --config configs/default_finger.json \
  --mesh subject006_US120 \
  --waveform heartbeat \
  --frames 50 \
  --samples 10 \
  --seed 42 \
  --out exports/subject006_US120_augmented_10.npz
```

Use `--mesh grid` for a Cartesian grid export.

## Scientific definition

The baseline map is piecewise tissue conductivity:

\[
\sigma_0(\mathbf{x}) = \sigma_{c(\mathbf{x})},
\]

where the class function identifies skin, fat, muscle, bone,
ligament/tendon, or artery. Each artery follows the supplied normalized
waveform \(w(t)\):

\[
\Delta\sigma_a(\mathbf{x},t)
= A_a w(t-\tau_a)\,\mathbf{1}_{\Omega_a}(\mathbf{x}).
\]

A soft response may extend into muscle:

\[
\Delta\sigma_m(\mathbf{x},t)
= \eta A_a w(t-\tau_a)
\exp\!\left[-\frac{d_a(\mathbf{x})^2}{2\ell^2}\right]
\mathbf{1}_{\Omega_m}(\mathbf{x}).
\]

The muscle indicator is important: the halo is not applied to skin, fat,
bone, ligament/tendon, or the exterior. The current halo is a controllable
phenomenological model, not a validated perfusion PDE.

## Repository layout

```text
app/Home.py                 Entry point (st.navigation)
app/views/                  Hub, finger model, waveform, and mesh-export pages
app/lib/                    Shared navigation, state, tissue palette, and figures
app/components/finger_canvas Legacy experimental GPU drag canvas
src/finger_sim/             Scientific model, waveform, mesh, and export code
configs/default_finger.json Reproducible default anatomy
meshes/                     Bundled PVI ring mesh artifacts and provenance
docs/PLAN.md                Product/research implementation plan
docs/GCNM_DATA_CONTRACT.md  Exact NPZ handoff contract
tests/                      Lightweight scientific invariants
```

There is no frontend build step. The production editor uses Plotly so the full
tissue field renders consistently across browsers; the earlier GPU drag canvas
is kept as an experimental component but is no longer the primary editor.

## Important limitations

- The model is a 2D cross-section. Longitudinal ring displacement and
  out-of-plane current require a future 3D forward model.
- The exported conductivity is a clean simulation target. It is not a PVI
  Newton reconstruction and should not be described as measured anatomy.
- The generic b035/b045 choices are ring-size candidates, not subject labels.
  US120 is confirmed for subject 006. The bundled US140 subject-001 selection
  remains estimated; its manifest preserves that warning.

See [the implementation plan](docs/PLAN.md) before using the app as a dataset
factory for finger-agnostic GCNM training.
