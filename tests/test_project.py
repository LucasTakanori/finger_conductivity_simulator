import numpy as np

from finger_sim.augmentation import AugmentationSpec, augment_model
from finger_sim.dataset import generate_augmented_dataset
from finger_sim.models import WaveformSpec, default_finger_model
from finger_sim import project as project_io


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
