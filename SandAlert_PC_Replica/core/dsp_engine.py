"""Waveform window analysis with explicit sample-rate and time units."""

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


class DSPEngine:
    """Return a rectified window, threshold, integer IPS, g/s, energy and peaks.

    ``k_v2`` is the calibration coefficient in peak-energy units per gram.
    Energy / k_v2 gives the mass represented by the window; dividing by its
    duration gives a rate. IPS is rounded to an integer for protocol v1.
    """

    PEAK_AMPLITUDE_LIMIT = 2.0

    def __init__(self, sample_rate=1000, highpass_hz=20.0):
        self.highpass_hz = float(highpass_hz)
        self.set_sample_rate(sample_rate)

    def set_sample_rate(self, sample_rate):
        sample_rate = float(sample_rate)
        if not np.isfinite(sample_rate) or not 0 < self.highpass_hz < sample_rate / 2:
            raise ValueError("sample_rate must exceed twice the positive highpass_hz")
        self.fs = sample_rate
        self.nyq = sample_rate / 2
        # SOS avoids the ill-conditioned polynomial filter at 20 Hz / 2 MHz.
        self.sos = butter(4, self.highpass_hz, btype="high", fs=self.fs, output="sos")

    def process_window(self, raw_voltage_array, threshold_margin, k_v2):
        raw = np.asarray(raw_voltage_array, dtype=np.float64)
        if raw.ndim != 1 or not np.all(np.isfinite(raw)):
            raise ValueError("waveform must be a finite one-dimensional array")
        if not np.isfinite(k_v2) or k_v2 <= 0:
            raise ValueError("calibration factor must be positive and finite")
        if not np.isfinite(threshold_margin) or threshold_margin < 0:
            raise ValueError("threshold margin must be non-negative and finite")
        # A fourth-order SOS highpass needs more than 15 points for padding.
        if raw.size <= 15:
            return np.zeros_like(raw), float(threshold_margin), 0, 0.0, 0.0, []

        # Remove DC before rectifying. Never clip the raw, DC-biased waveform.
        rectified = np.abs(sosfiltfilt(self.sos, raw))
        background = float(np.mean(rectified))
        threshold = background + threshold_margin
        peaks, _ = find_peaks(rectified, height=threshold, distance=10)
        valid_peaks = peaks[rectified[peaks] < self.PEAK_AMPLITUDE_LIMIT].tolist()
        energy = float(np.sum((rectified[valid_peaks] - background) ** 2))
        duration = raw.size / self.fs
        ips = int(round(len(valid_peaks) / duration))
        mass_rate = energy / k_v2 / duration
        return rectified, threshold, ips, mass_rate, energy, valid_peaks
