from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis


FEATURE_NAMES = (
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "zero_crossing_rate",
    "envelope_mean",
    "envelope_std",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_entropy",
    "band_low_fraction",
    "band_mid_fraction",
    "band_high_fraction",
)


def extract_features(waveform: np.ndarray, fs_hz: int) -> np.ndarray:
    signal = np.asarray(waveform, dtype=float).reshape(-1)
    signal = signal - np.median(signal)
    rms = float(np.sqrt(np.mean(np.square(signal))))
    peak = float(np.max(np.abs(signal)))
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / fs_hz)
    total_power = float(np.sum(spectrum)) + 1e-15
    weights = spectrum / total_power
    centroid = float(np.sum(frequencies * weights))
    bandwidth = float(
        np.sqrt(np.sum(np.square(frequencies - centroid) * weights))
    )
    nonzero = weights[weights > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(weights.size))
    nyquist = fs_hz / 2.0

    def band_fraction(low: float, high: float) -> float:
        mask = (frequencies >= low * nyquist) & (frequencies < high * nyquist)
        return float(np.sum(spectrum[mask]) / total_power)

    envelope = np.abs(hilbert(signal))
    zero_crossing_rate = float(
        np.mean(np.diff(np.signbit(signal)).astype(float))
    )
    values = (
        rms,
        peak,
        float(np.ptp(signal)),
        peak / max(rms, 1e-12),
        float(np.nan_to_num(kurtosis(signal, fisher=True, bias=False))),
        zero_crossing_rate,
        float(np.mean(envelope)),
        float(np.std(envelope)),
        centroid,
        bandwidth,
        entropy,
        band_fraction(0.0, 0.1),
        band_fraction(0.1, 0.3),
        band_fraction(0.3, 1.01),
    )
    return np.asarray(values, dtype=np.float64)


def extract_feature_matrix(waveforms: np.ndarray, fs_hz: int) -> np.ndarray:
    return np.vstack([
        extract_features(window[0] if window.ndim == 2 else window, fs_hz)
        for window in waveforms
    ])
