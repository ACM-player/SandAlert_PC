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

MECHANICAL_SHAPE_FEATURE_NAMES = (
    "envelope_rise_fraction",
    "envelope_decay_fraction",
    "envelope_post_pre_ratio_log",
    "crest_factor_log",
    "excess_kurtosis",
    "autocorrelation_peak",
    "spectral_line_ratio",
)

QUALITY_AUX_FEATURE_NAMES = AUX_FEATURE_NAMES + MECHANICAL_SHAPE_FEATURE_NAMES

MECHANICAL_REJECTOR_FEATURE_NAMES = (
    "log_rms",
    "log_envelope_mean",
    "spectral_centroid_fraction",
    "spectral_bandwidth_fraction",
    "spectral_entropy",
    "band_low_fraction",
    "band_mid_fraction",
    "band_high_fraction",
) + MECHANICAL_SHAPE_FEATURE_NAMES


@dataclass(frozen=True)
class SignalAuxFeatures:
    values: np.ndarray
    clipping_fraction: float
    dropout_fraction: float
    feature_names: tuple[str, ...] = AUX_FEATURE_NAMES


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
    feature_names: tuple[str, ...] | list[str] | None = None,
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
    envelope = np.abs(hilbert(centered))
    peak_index = int(np.argmax(envelope))
    peak_value = max(float(envelope[peak_index]), 1e-12)
    before_peak = envelope[:peak_index + 1]
    below_10 = np.flatnonzero(before_peak <= 0.1 * peak_value)
    below_90 = np.flatnonzero(before_peak <= 0.9 * peak_value)
    rise_start = int(below_10[-1]) if below_10.size else 0
    rise_end = int(below_90[-1]) if below_90.size else peak_index
    rise_fraction = max(rise_end - rise_start, 0) / max(signal.size, 1)
    after_peak = envelope[peak_index:]
    below_decay = np.flatnonzero(after_peak <= peak_value / np.e)
    decay_samples = (
        int(below_decay[0]) if below_decay.size else signal.size - peak_index
    )
    decay_fraction = decay_samples / max(signal.size, 1)
    pre_energy = float(np.sum(np.square(envelope[:peak_index]))) + 1e-12
    post_energy = float(np.sum(np.square(envelope[peak_index + 1:]))) + 1e-12
    crest_factor = peak_value / max(rms, 1e-12)
    variance = float(np.mean(np.square(centered))) + 1e-15
    excess_kurtosis = float(
        np.mean(np.power(centered, 4)) / (variance * variance) - 3.0
    )
    normalized = centered / max(float(np.linalg.norm(centered)), 1e-12)
    min_lag = max(2, signal.size // 500)
    max_lag = min(signal.size // 8, max(min_lag + 1, signal.size // 20))
    autocorrelation_peak = 0.0
    if max_lag > min_lag:
        fft_size = 1 << int(np.ceil(np.log2(max(2, signal.size * 2 - 1))))
        fft_values = np.fft.rfft(normalized, n=fft_size)
        correlations = np.fft.irfft(
            fft_values * np.conjugate(fft_values), n=fft_size
        )[min_lag:max_lag]
        if correlations.size:
            autocorrelation_peak = float(np.max(np.abs(correlations)))
    ordered_power = np.sort(spectrum[1:] if spectrum.size > 1 else spectrum)
    spectral_line_ratio = float(
        np.sum(ordered_power[-min(8, ordered_power.size):]) / total_power
    )
    feature_values = {
        "analytic_relative_energy_db": float(
            analytic_relative_energy_db or 0.0
        ),
        "log_rms": float(np.log(max(rms, 1e-12))),
        "log_envelope_mean": float(np.log(max(envelope_mean, 1e-12))),
        "spectral_centroid_fraction": centroid / max(nyquist, 1e-12),
        "spectral_bandwidth_fraction": bandwidth / max(nyquist, 1e-12),
        "spectral_entropy": entropy,
        "band_low_fraction": band_fraction(0.0, 0.1),
        "band_mid_fraction": band_fraction(0.1, 0.3),
        "band_high_fraction": band_fraction(0.3, 1.01),
        "clipping_fraction": clipping_fraction,
        "dropout_fraction": dropout_fraction,
        "air_velocity_scaled": (
            float(air_velocity_m_s) / 50.0 if air_present else 0.0
        ),
        "air_velocity_present": 1.0 if air_present else 0.0,
        "envelope_rise_fraction": float(rise_fraction),
        "envelope_decay_fraction": float(decay_fraction),
        "envelope_post_pre_ratio_log": float(np.log(post_energy / pre_energy)),
        "crest_factor_log": float(np.log(max(crest_factor, 1e-12))),
        "excess_kurtosis": float(np.clip(excess_kurtosis, -3.0, 100.0)),
        "autocorrelation_peak": autocorrelation_peak,
        "spectral_line_ratio": spectral_line_ratio,
    }
    selected_names = tuple(feature_names or AUX_FEATURE_NAMES)
    unknown = [name for name in selected_names if name not in feature_values]
    if unknown:
        raise ValueError(f"unsupported auxiliary features: {unknown}")
    values = np.asarray(
        [feature_values[name] for name in selected_names],
        dtype=np.float32,
    )
    return SignalAuxFeatures(
        values=values,
        clipping_fraction=clipping_fraction,
        dropout_fraction=dropout_fraction,
        feature_names=selected_names,
    )


def extract_mechanical_rejector_features(
    waveform: np.ndarray,
    fs_hz: int,
) -> np.ndarray:
    return extract_shadow_aux_features(
        waveform,
        fs_hz,
        analytic_relative_energy_db=0.0,
        air_velocity_m_s=None,
        feature_names=MECHANICAL_REJECTOR_FEATURE_NAMES,
    ).values
