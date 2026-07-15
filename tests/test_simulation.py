import numpy as np

from finger_sim.geometry import ARTERY, BONE, FAT, LIGAMENT, SKIN
from finger_sim.models import WaveformSpec, default_finger_model
from finger_sim.simulation import simulate_grid


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

