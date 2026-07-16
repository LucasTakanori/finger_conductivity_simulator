import numpy as np

from finger_sim.augmentation import AugmentationSpec, augment_model
from finger_sim.dataset import generate_augmented_dataset
from finger_sim.geometry import ARTERY, ARTERY_WALL, BONE, BONE_MARROW
from finger_sim.models import FingerModel, WaveformSpec, default_finger_model
from finger_sim.simulation import simulate_grid
from finger_sim import project as project_io


def test_bone_and_artery_have_nested_layers():
    result = simulate_grid(default_finger_model(), WaveformSpec(frames=8), size=160)
    labels = result.tissue_labels
    # Cortical shell around a marrow core; vessel wall around a blood lumen.
    for tissue in (BONE, BONE_MARROW, ARTERY_WALL, ARTERY):
        assert (labels == tissue).any(), tissue
    # Only blood pulsates — the wall and the marrow never do.
    assert np.all(result.delta_sigma[:, labels == ARTERY_WALL] == 0.0)
    assert np.all(result.delta_sigma[:, labels == BONE_MARROW] == 0.0)
    assert np.nanmax(np.abs(result.delta_sigma[:, labels == ARTERY])) > 0


def test_layer_fractions_control_the_amounts():
    solid = default_finger_model()
    solid.bone_marrow_fraction = 0.0  # no medullary cavity
    for artery in solid.arteries:
        artery.lumen_fraction = 1.0  # no vessel wall
    labels = simulate_grid(solid, WaveformSpec(frames=4), size=160).tissue_labels
    assert not (labels == BONE_MARROW).any()
    assert not (labels == ARTERY_WALL).any()
    assert (labels == BONE).any() and (labels == ARTERY).any()


def test_older_models_without_layer_fields_still_load():
    data = default_finger_model().to_dict()
    data.pop("bone_marrow_fraction")
    data["conductivities"].pop("bone_marrow_s_m")
    data["conductivities"].pop("artery_wall_s_m")
    for artery in data["arteries"]:
        artery.pop("lumen_fraction")
    restored = FingerModel.from_dict(data)
    assert restored.bone_marrow_fraction == 0.6
    assert restored.arteries[0].lumen_fraction == 0.75


def test_augmentation_varies_ring_rotation_and_diffusion():
    model = default_finger_model()
    spec = AugmentationSpec(seed=0, finger_rotation_deg=12.0, diffusion_fraction=0.2)
    rng = np.random.default_rng(0)
    rotations, diffusions = set(), set()
    for _ in range(8):
        variant = augment_model(model, spec, rng)
        rotations.add(round(variant.rotation_deg, 4))
        diffusions.add(round(variant.muscle_diffusion_fraction, 5))
    assert len(rotations) > 1
    assert len(diffusions) > 1


def test_project_round_trip(tmp_path):
    model = default_finger_model()
    waveform = WaveformSpec("heartbeat", 16, 1.0)
    augmentation = AugmentationSpec(samples=12, seed=5, finger_rotation_deg=8.0)
    project = project_io.build_project(
        model, waveform, mesh="b035_US120", grid_size=64, augmentation=augmentation
    )
    path = project_io.save_project(tmp_path / "session.json", project)
    reloaded = project_io.load_project(path)

    assert reloaded["mesh"] == "b035_US120"
    assert project_io.model_of(reloaded).to_dict() == model.to_dict()
    assert project_io.waveform_of(reloaded).frames == 16
    restored = project_io.augmentation_of(reloaded)
    assert restored.samples == 12 and restored.finger_rotation_deg == 8.0
    # CLI-style overrides win over the stored values.
    overridden = project_io.augmentation_of(reloaded, samples=1000, seed=42)
    assert overridden.samples == 1000 and overridden.seed == 42


def test_dataset_seed_is_reproducible():
    model = default_finger_model()
    waveform = WaveformSpec("heartbeat", 10, 1.0)
    spec = AugmentationSpec(samples=4, seed=3)
    first = generate_augmented_dataset(model, waveform, spec, grid_size=40)
    second = generate_augmented_dataset(model, waveform, spec, grid_size=40)
    assert np.array_equal(first["delta_sigma"], second["delta_sigma"], equal_nan=True)

    different = generate_augmented_dataset(
        model, waveform, AugmentationSpec(samples=4, seed=4), grid_size=40
    )
    assert not np.array_equal(first["delta_sigma"], different["delta_sigma"], equal_nan=True)
