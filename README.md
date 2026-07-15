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

1. **Finger model** — an interactive GPU cross-section. Each tissue is a
   draggable object: move its centre, drag the rim handles to stretch a circle
   into an ellipse, or rotate it, and the conductivity field re-renders in real
   time. Hover anywhere to identify the tissue and read its conductivity. The
   finger can be scaled to any bundled ring size, which also previews that ring's
   triangular mesh over the model. Rendering uses **WebGPU** when the browser
   provides it and falls back to **WebGL2** otherwise; both paths run the same
   per-pixel tissue classifier, verified point-for-point against the Python model.
2. **Waveform** — heartbeat, sine, or arbitrary CSV/TXT pulse, with an animated
   differential/dynamic conductivity map. A time scrubber selects any frame and
   the colour legend is held constant across the whole beat.
3. **Mesh & export** — projects the conductivity onto a PVI ring mesh with the
   triangular grid drawn over the field, and exports a versioned NPZ for 2D_GCNM.

Other capabilities:

- Independent skin, fat, muscle, bone, ligament/tendon, and arterial
  conductivities; editable skin/fat thickness and finger rotation.
- Artery position, elliptical radius, rotation, baseline conductivity, pulse
  amplitude, and phase delay.
- A configurable muscle-only pulse halo. Skin, fat, bone, and ligament/tendon
  are explicitly excluded from diffusion.
- Projection onto all 38 PVI FEM mesh variants from the b035/b045 collections,
  plus the subject-selected US120 and US140 aliases.
- A command-line exporter for reproducible dataset generation.

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
app/components/finger_canvas Custom WebGPU/WebGL2 drag-to-edit canvas (no build step)
src/finger_sim/             Scientific model, waveform, mesh, and export code
configs/default_finger.json Reproducible default anatomy
meshes/                     Bundled PVI ring mesh artifacts and provenance
docs/PLAN.md                Product/research implementation plan
docs/GCNM_DATA_CONTRACT.md  Exact NPZ handoff contract
tests/                      Lightweight scientific invariants
```

The interactive canvas implements the Streamlit component protocol in vanilla
JavaScript, so there is **no npm/build step**: the frontend under
`app/components/finger_canvas/frontend/` is served directly. WebGPU needs a
recent Chrome/Edge (or Chrome-based browser with WebGPU enabled); every other
browser automatically uses the WebGL2 path.

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
