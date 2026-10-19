import numpy as np
import pytest
from scipy.signal import sosfreqz

from core.dsp_engine import DSPEngine


@pytest.mark.parametrize("fs", [1000, 2_000_000])
def test_highpass_keeps_20_hz_cutoff_at_each_sampling_frequency(fs):
    dsp = DSPEngine(fs)
    _, response = sosfreqz(dsp.sos, worN=np.array([20.0]), fs=fs)
    assert 20 * np.log10(abs(response[0])) == pytest.approx(-3.0103, abs=0.001)
    signal = np.random.default_rng(42).normal(0, 0.01, 3000) - 2.85
    assert np.all(np.isfinite(dsp.process_window(signal, 0.008, 1.2)[0]))


def test_rates_and_mass_agree_for_one_and_two_second_windows():
    dsp = DSPEngine(1000)
    rates = []
    for seconds in (1, 2):
        signal = np.zeros(seconds * 1000)
        signal[50::100] = 0.2
        _, _, ips, mass_rate, energy, peaks = dsp.process_window(signal, 0.1, 2.0)
        assert len(peaks) == seconds * 10
        assert ips == 10
        assert mass_rate * seconds == pytest.approx(energy / 2.0)
        rates.append(mass_rate)
    assert rates[0] == pytest.approx(rates[1], rel=0.01)


@pytest.mark.parametrize("length", [0, 1, 14, 15])
def test_short_windows_always_return_six_values_without_filter_padding_error(length):
    result = DSPEngine().process_window(np.zeros(length), 0.008, 1.2)
    assert len(result) == 6
    assert result[2:] == (0, 0.0, 0.0, [])


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf])
def test_invalid_calibration_is_rejected(factor):
    with pytest.raises(ValueError, match="calibration factor"):
        DSPEngine().process_window(np.zeros(100), 0.008, factor)
