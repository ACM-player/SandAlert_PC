from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert


AUX_FEATURE_NAMES = (
    "analytic_relative_energy_db",
    "log_rms",
    "log_envelope_mean",
    "spectral_centroid_fraction",
    "spectral_bandwidth_fraction",
    "spectral_entropy",
    "band_low_fraction",
    "band_mid_fraction",
    "band_high_fraction",
    "clipping_fraction",
    "dropout_fraction",
    "air_velocity_scaled",
    "air_velocity_present",
)


@dataclass(frozen=True)
class SignalAuxFeatures:
    values: np.ndarray
    clipping_fraction: float
    dropout_fraction: float


def _quality_fractions(signal: np.ndarray) -> tuple[float, float]:
    span = float(np.ptp(signal))
    if span <= 1e-12:
        return 0.0, 1.0

    quantization_tolerance = max(span / 65535.0 * 1.5, 1e-9)
    at_low = np.abs(signal - float(np.min(signal))) <= quantization_tolerance
    at_high = np.abs(signal - float(np.max(signal))) <= quantization_tolerance
    clipping_fraction = float(np.mean(at_low | at_high))

    differences = np.abs(np.diff(signal))
    dropout_fraction = float(
        np.count_nonzero(differences <= quantization_tolerance)
        / max(differences.size, 1)
    )
    return clipping_fraction, dropout_fraction


def extract_shadow_aux_features(
    waveform: np.ndarray,
    fs_hz: int,
    analytic_relative_energy_db: float | None,
    air_velocity_m_s: float | None = None,
) -> SignalAuxFeatures:
    """Extract observable features shared by training and shadow inference."""
    signal = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if signal.size < 8 or not np.all(np.isfinite(signal)):
        raise ValueError("waveform must contain at least eight finite samples")

    centered = signal - np.median(signal)
    rms = float(np.sqrt(np.mean(np.square(centered))))
    envelope_mean = float(np.mean(np.abs(hilbert(centered))))
    spectrum = np.square(np.abs(np.fft.rfft(centered)))
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / fs_hz)
    total_power = float(np.sum(spectrum)) + 1e-15
    weights = spectrum / total_power
    centroid = float(np.sum(frequencies * weights))
    bandwidth = float(
        np.sqrt(np.sum(np.square(frequencies - centroid) * weights))
    )
    nonzero = weights[weights > 0]
    entropy = float(
        -np.sum(nonzero * np.log(nonzero)) / max(np.log(weights.size), 1.0)
    )
    nyquist = fs_hz / 2.0

    def band_fraction(low: float, high: float) -> float:
        mask = (frequencies >= low * nyquist) & (frequencies < high * nyquist)
        return float(np.sum(spectrum[mask]) / total_power)

    clipping_fraction, dropout_fraction = _quality_fractions(signal)
    air_present = air_velocity_m_s is not None and np.isfinite(air_velocity_m_s)
    values = np.asarray(
        (
            float(analytic_relative_energy_db or 0.0),
            float(np.log(max(rms, 1e-12))),
            float(np.log(max(envelope_mean, 1e-12))),
            centroid / max(nyquist, 1e-12),
            bandwidth / max(nyquist, 1e-12),
            entropy,
            band_fraction(0.0, 0.1),
            band_fraction(0.1, 0.3),
            band_fraction(0.3, 1.01),
            clipping_fraction,
            dropout_fraction,
            float(air_velocity_m_s) / 50.0 if air_present else 0.0,
            1.0 if air_present else 0.0,
        ),
        dtype=np.float32,
    )
    return SignalAuxFeatures(
        values=values,
        clipping_fraction=clipping_fraction,
        dropout_fraction=dropout_fraction,
    )
