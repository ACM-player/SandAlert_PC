from pathlib import Path
import sys

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from core.tdoa_engine import TDOAEngine


def circular_error(true_angle, predicted_angle):
    return abs((predicted_angle - true_angle + 180.0) % 360.0 - 180.0)


def test_physical_lag_windows_follow_pair_geometry():
    engine = TDOAEngine()
    assert engine._pair_max_lag_samples(1, 2000) == 97
    assert engine._pair_max_lag_samples(2, 2000) == 136
    assert engine._pair_max_lag_samples(3, 2000) == 97


def test_compute_doa_keeps_legacy_tuple_and_exposes_diagnostics():
    np.random.seed(7)
    engine = TDOAEngine()
    waveform, _ = engine.generate_mock_array_data(
        n_samples=2000,
        true_angle_deg=359.0,
        snr_db=10.0,
    )
    angle, tdoa, angles, costs = engine.compute_doa(waveform)

    assert circular_error(359.0, angle) <= 2.0
    assert tdoa.shape == (3,)
    assert angles.shape == costs.shape
    assert engine.last_diagnostics.candidate_count >= 1
    assert engine.last_diagnostics.signal_like


def test_fractional_interpolation_improves_band_limited_accuracy():
    simulator = TDOAEngine()
    integer_peak = TDOAEngine(
        peak_candidates=1,
        interpolate_peaks=False,
        bandpass_enabled=True,
    )
    interpolated = TDOAEngine(
        peak_candidates=1,
        interpolate_peaks=True,
        bandpass_enabled=True,
    )
    errors_integer = []
    errors_interpolated = []
    np.random.seed(19)
    for true_angle in np.linspace(3.7, 356.2, 24):
        waveform, _ = simulator.generate_mock_array_data(
            n_samples=2000,
            true_angle_deg=float(true_angle),
            snr_db=10.0,
            delay_mode="fractional",
        )
        predicted_integer = integer_peak.compute_doa(waveform)[0]
        predicted_interpolated = interpolated.compute_doa(waveform)[0]
        errors_integer.append(circular_error(true_angle, predicted_integer))
        errors_interpolated.append(
            circular_error(true_angle, predicted_interpolated)
        )

    assert np.mean(errors_interpolated) < np.mean(errors_integer)


def test_pure_noise_is_rejected_by_peak_confidence_gate():
    rng = np.random.default_rng(123)
    engine = TDOAEngine()
    for _ in range(25):
        result = engine.compute_doa_detailed(rng.standard_normal((4, 2000)))
        assert not result.diagnostics.signal_like


def test_invalid_enabled_frequency_band_is_rejected():
    with pytest.raises(ValueError, match="band_low_hz"):
        TDOAEngine(
            bandpass_enabled=True,
            band_low_hz=300_000.0,
            band_high_hz=100_000.0,
        )


def test_mock_generator_supports_integer_and_fractional_delays():
    engine = TDOAEngine()
    np.random.seed(11)
    integer_data, integer_delays = engine.generate_mock_array_data(
        true_angle_deg=33.0,
        delay_mode="integer",
    )
    np.random.seed(11)
    fractional_data, fractional_delays = engine.generate_mock_array_data(
        true_angle_deg=33.0,
        delay_mode="fractional",
    )

    assert integer_data.shape == fractional_data.shape == (4, 2000)
    np.testing.assert_allclose(integer_delays, fractional_delays)
    assert not np.array_equal(integer_data, fractional_data)
