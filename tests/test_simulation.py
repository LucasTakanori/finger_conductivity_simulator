import numpy as np

from finger_sim.geometry import ARTERY, BONE, FAT, LIGAMENT, SKIN
from finger_sim.models import WaveformSpec, default_finger_model
from finger_sim.simulation import simulate_grid
from finger_sim.augmentation import AugmentationSpec
from finger_sim.dataset import generate_augmented_dataset


def test_dynamic_identity_and_protected_tissues():
    model = default_finger_model()
    result = simulate_grid(model, WaveformSpec(frames=12), size=48)
    np.testing.assert_allclose(
        result.sigma_dynamic,
        result.sigma_baseline[None, :] + result.delta_sigma,
        equal_nan=True,
    )
    protected = np.isin(result.tissue_labels, [SKIN, FAT, BONE, LIGAMENT])
    assert np.all(result.delta_sigma[:, protected] == 0.0)
    assert np.nanmax(np.abs(result.delta_sigma[:, result.tissue_labels == ARTERY])) > 0


def test_custom_waveform_is_resampled():
    model = default_finger_model()
    waveform = WaveformSpec(
        kind="custom", frames=17, custom_values=[0.0, 1.0, 0.0]
    )
    result = simulate_grid(model, waveform, size=32)
    assert result.delta_sigma.shape[0] == 17
    assert np.isclose(np.max(result.waveform), 1.0)


def test_augmented_batch_is_reproducible_and_varied():
    model = default_finger_model()
    waveform = WaveformSpec(frames=12)
    augmentation = AugmentationSpec(samples=3, seed=17)
    first = generate_augmented_dataset(model, waveform, augmentation, grid_size=24)
    second = generate_augmented_dataset(model, waveform, augmentation, grid_size=24)
    assert first["delta_sigma"].shape == (3, 12, 24 * 24)
    np.testing.assert_allclose(first["delta_sigma"], second["delta_sigma"], equal_nan=True)
    assert not np.array_equal(first["waveform"][0], first["waveform"][1])
    assert not np.array_equal(first["points_mm"][0], first["points_mm"][1])


def test_frequency_is_fixed_at_50_khz():
    assert default_finger_model().frequency_hz == 50_000.0
