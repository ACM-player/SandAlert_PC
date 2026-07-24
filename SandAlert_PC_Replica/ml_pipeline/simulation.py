from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, sosfilt


SCENARIOS = (
    "clean",
    "background",
    "low_snr",
    "gain_shift",
    "resonance_shift",
    "multipath",
    "clustered",
    "clipping",
    "dropout",
)

OOD_SCENARIOS = (
    "extreme_band_shift",
    "severe_multipath",
    "mechanical_interference",
    "heavy_clipping",
    "burst_dropout",
    "compound_extreme",
)

ALL_SCENARIOS = SCENARIOS + OOD_SCENARIOS


@dataclass(frozen=True)
class SimulationConfig:
    sample_rate_hz: int = 2_000_000
    window_samples: int = 8192
    channels: int = 1
    impact_rate_hz: float = 220.0
    noise_rms_min: float = 0.0005
    noise_rms_max: float = 0.006
    resonance_low_hz: float = 60_000.0
    resonance_high_hz: float = 300_000.0
    damping_low_us: float = 15.0
    damping_high_us: float = 180.0
    amplitude_min_v: float = 0.01
    amplitude_max_v: float = 0.35
    dc_offset_v: float = -2.85
    clip_limit_v: float = 5.0
    quantization_bits: int = 16
    array_radius_m: float = 0.05
    propagation_speed_m_s: float = 1500.0
    air_velocity_min_m_s: float = 4.0
    air_velocity_max_m_s: float = 32.0

    def validate(self) -> None:
        if self.sample_rate_hz <= 0 or self.window_samples < 128:
            raise ValueError("sample rate and window length must be positive")
        if not 1 <= self.channels <= 4:
            raise ValueError("channels must be between 1 and 4")
        if self.resonance_high_hz >= self.sample_rate_hz / 2:
            raise ValueError("resonance band must remain below Nyquist")
        if self.noise_rms_min <= 0 or self.noise_rms_max < self.noise_rms_min:
            raise ValueError("invalid noise range")
        if (
            self.air_velocity_min_m_s < 0
            or self.air_velocity_max_m_s < self.air_velocity_min_m_s
        ):
            raise ValueError("invalid air velocity range")


@dataclass(frozen=True)
class SyntheticWindow:
    waveform: np.ndarray
    labels: dict


def _colored_noise(
    rng: np.random.Generator,
    samples: int,
    fs_hz: int,
    cutoff_hz: float = 20_000.0,
) -> np.ndarray:
    white = rng.normal(0.0, 1.0, samples)
    cutoff = min(max(cutoff_hz, 100.0), fs_hz * 0.45)
    sos = butter(2, cutoff / (0.5 * fs_hz), btype="low", output="sos")
    colored = sosfilt(sos, white)
    std = float(np.std(colored))
    return colored / max(std, 1e-12)


def _event_times(
    rng: np.random.Generator,
    count: int,
    samples: int,
    scenario: str,
) -> np.ndarray:
    margin = max(8, samples // 100)
    if count == 0:
        return np.empty(0, dtype=int)
    if scenario == "clustered":
        center = int(rng.integers(margin, max(margin + 1, samples - margin)))
        spread = max(2, samples // 100)
        return np.clip(
            np.rint(rng.normal(center, spread, count)).astype(int),
            margin,
            samples - margin - 1,
        )
    return np.sort(rng.integers(margin, samples - margin, count))


def _single_impact(
    rng: np.random.Generator,
    config: SimulationConfig,
    scenario: str,
    severity: float,
) -> np.ndarray:
    fs_hz = config.sample_rate_hz
    frequency = rng.uniform(config.resonance_low_hz, config.resonance_high_hz)
    if scenario == "resonance_shift":
        frequency = rng.choice([
            rng.uniform(25_000.0, config.resonance_low_hz),
            rng.uniform(config.resonance_high_hz, fs_hz * 0.45),
        ])
    elif scenario in {"extreme_band_shift", "compound_extreme"}:
        low_high_boundary = max(5_000.0, config.resonance_low_hz * 0.6)
        high_low_boundary = min(
            fs_hz * 0.46,
            config.resonance_high_hz * (1.05 + 0.35 * severity),
        )
        if rng.random() < 0.5:
            frequency = rng.uniform(5_000.0, low_high_boundary)
        else:
            frequency = rng.uniform(high_low_boundary, fs_hz * 0.47)
    damping_us = rng.uniform(config.damping_low_us, config.damping_high_us)
    damping_seconds = damping_us * 1e-6
    duration = min(
        config.window_samples,
        max(32, int(math.ceil(8.0 * damping_seconds * fs_hz))),
    )
    t = np.arange(duration, dtype=float) / fs_hz
    phase = rng.uniform(0.0, 2.0 * math.pi)
    carrier = np.sin(2.0 * math.pi * frequency * t + phase)
    carrier += 0.25 * np.sin(4.0 * math.pi * frequency * t + phase * 0.5)
    envelope = np.exp(-t / max(damping_seconds, 1.0 / fs_hz))
    amplitude = math.exp(
        rng.uniform(math.log(config.amplitude_min_v), math.log(config.amplitude_max_v))
    )
    return (amplitude * envelope * carrier).astype(np.float32)


def _delayed_channel(
    source: np.ndarray,
    delay_samples: float,
    attenuation: float,
) -> np.ndarray:
    indices = np.arange(source.size, dtype=float) - delay_samples
    delayed = np.interp(indices, np.arange(source.size), source, left=0.0, right=0.0)
    return (attenuation * delayed).astype(np.float32)


def generate_window(
    config: SimulationConfig,
    seed: int,
    scenario: str = "clean",
    force_event_count: int | None = None,
    *,
    severity: float = 0.5,
    air_velocity_m_s: float | None = None,
) -> SyntheticWindow:
    """Generate one deterministic, labeled waveform window."""
    config.validate()
    if scenario not in ALL_SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    duration_s = config.window_samples / config.sample_rate_hz
    if air_velocity_m_s is None:
        air_velocity_m_s = float(
            rng.uniform(config.air_velocity_min_m_s, config.air_velocity_max_m_s)
        )
    if not math.isfinite(air_velocity_m_s) or air_velocity_m_s < 0:
        raise ValueError("air velocity must be finite and non-negative")

    if force_event_count is not None:
        event_count = int(force_event_count)
    elif scenario == "background":
        event_count = 0
    else:
        event_count = int(rng.poisson(config.impact_rate_hz * duration_s))
    event_count = max(0, min(event_count, 12))

    noise_rms = math.exp(
        rng.uniform(math.log(config.noise_rms_min), math.log(config.noise_rms_max))
    )
    if scenario == "low_snr":
        noise_rms *= 2.0 + 4.0 * severity
    elif scenario in {"mechanical_interference", "compound_extreme"}:
        noise_rms *= 3.0 + 8.0 * severity
    white = rng.normal(0.0, noise_rms, config.window_samples)
    air_scale = max(0.2, (air_velocity_m_s / 12.0) ** 1.25)
    flow_cutoff = 8_000.0 + 1_200.0 * air_velocity_m_s
    flow = _colored_noise(
        rng, config.window_samples, config.sample_rate_hz, flow_cutoff
    )
    background = white + flow * noise_rms * rng.uniform(0.25, 1.5) * air_scale

    if scenario in {"mechanical_interference", "compound_extreme"}:
        t = np.arange(config.window_samples, dtype=float) / config.sample_rate_hz
        fundamental = rng.uniform(600.0, 8_000.0)
        mechanical = np.zeros(config.window_samples, dtype=float)
        for harmonic in range(1, 5):
            mechanical += (
                noise_rms
                * severity
                * rng.uniform(1.0, 4.0)
                / harmonic
                * np.sin(
                    2.0 * math.pi * fundamental * harmonic * t
                    + rng.uniform(0.0, 2.0 * math.pi)
                )
            )
        for _ in range(1 + int(4 * severity)):
            start = int(rng.integers(0, max(1, config.window_samples - 16)))
            length = int(rng.integers(8, min(160, config.window_samples - start) + 1))
            mechanical[start:start + length] += (
                rng.normal(0.0, noise_rms * (4.0 + 12.0 * severity), length)
                * np.hanning(length)
            )
        background += mechanical

    impact = np.zeros(config.window_samples, dtype=np.float32)
    events = _event_times(
        rng, event_count, config.window_samples, scenario
    )
    for event_index in events:
        pulse = _single_impact(rng, config, scenario, severity)
        end = min(config.window_samples, event_index + pulse.size)
        impact[event_index:end] += pulse[: end - event_index]
        if scenario in {"multipath", "severe_multipath", "compound_extreme"}:
            if scenario == "multipath":
                echo_count = 2
                delay_max = 500
                scale_max = 0.5
            else:
                echo_count = 3 + int(5 * severity)
                delay_max = min(
                    config.window_samples // 2,
                    int(600 + severity * 1800),
                )
                scale_max = 0.55 + 0.4 * severity
            for _ in range(echo_count):
                delay = rng.integers(8, max(9, delay_max))
                scale = rng.uniform(0.08, scale_max)
                if scenario != "multipath" and rng.random() < 0.25:
                    scale *= -1.0
                start = min(config.window_samples, event_index + int(delay))
                end = min(config.window_samples, start + pulse.size)
                if end > start:
                    impact[start:end] += scale * pulse[: end - start]

    gain = rng.uniform(0.85, 1.15)
    if scenario == "gain_shift":
        gain *= rng.choice([rng.uniform(0.2, 0.5), rng.uniform(1.8, 3.0)])
    clean_source = gain * (background + impact)

    channels = np.empty(
        (config.channels, config.window_samples), dtype=np.float32
    )
    if config.channels == 1:
        channels[0] = clean_source
        angle_deg = None
    else:
        angle_deg = float(rng.uniform(0.0, 360.0))
        angle_rad = math.radians(angle_deg)
        for channel in range(config.channels):
            sensor_angle = 2.0 * math.pi * channel / config.channels
            delay_s = (
                config.array_radius_m
                * math.cos(angle_rad - sensor_angle)
                / config.propagation_speed_m_s
            )
            channels[channel] = _delayed_channel(
                clean_source,
                delay_s * config.sample_rate_hz,
                rng.uniform(0.9, 1.1),
            )

    if scenario in {"clipping", "heavy_clipping", "compound_extreme"}:
        if scenario == "clipping":
            clip = rng.uniform(0.03, 0.12)
        else:
            signal_peak = max(float(np.max(np.abs(channels))), 1e-6)
            clip = signal_peak * rng.uniform(
                max(0.05, 0.45 - 0.35 * severity),
                max(0.10, 0.75 - 0.45 * severity),
            )
        channels = np.clip(channels, -clip, clip)
    if scenario in {"dropout", "burst_dropout", "compound_extreme"}:
        if scenario == "dropout":
            segments = 1
            max_fraction = 0.06
        else:
            segments = 2 + int(6 * severity)
            max_fraction = 0.08 + 0.32 * severity
        total_budget = max(16, int(config.window_samples * max_fraction))
        for _ in range(segments):
            length = int(
                rng.integers(
                    8,
                    max(9, total_budget // max(segments, 1) + 1),
                )
            )
            start = int(
                rng.integers(0, max(1, config.window_samples - length + 1))
            )
            channels[:, start:start + length] = 0.0

    if scenario == "compound_extreme" and severity >= 0.6:
        jitter_samples = rng.normal(
            0.0, 0.05 + 0.55 * severity, config.window_samples
        )
        original_indices = np.arange(config.window_samples, dtype=float)
        for channel in range(config.channels):
            channels[channel] = np.interp(
                original_indices + jitter_samples,
                original_indices,
                channels[channel],
                left=channels[channel, 0],
                right=channels[channel, -1],
            )

    quantization_step = (
        2.0 * config.clip_limit_v / (2 ** config.quantization_bits - 1)
    )
    channels = np.round(channels / quantization_step) * quantization_step
    channels = np.clip(channels, -config.clip_limit_v, config.clip_limit_v)
    channels += config.dc_offset_v
    channels = channels.astype(np.float32)

    impact_energy = float(np.mean(np.square(gain * impact)))
    background_energy = float(np.mean(np.square(gain * background)))
    window_energy = float(np.mean(np.square(clean_source)))
    observed_centered = channels[0].astype(np.float64) - float(
        np.median(channels[0])
    )
    observed_window_energy = float(np.mean(np.square(observed_centered)))
    observed_background_energy = background_energy
    relative_energy_db = 10.0 * math.log10(
        (window_energy + 1e-12) / (background_energy + 1e-12)
    )
    analytic_relative_energy_db = 10.0 * math.log10(
        (observed_window_energy + 1e-12)
        / (observed_background_energy + 1e-12)
    )
    impact_to_background_db = 10.0 * math.log10(
        (impact_energy + 1e-12) / (background_energy + 1e-12)
    )
    signal_plus_noise = max(window_energy, 1e-12)
    snr_db = 10.0 * math.log10(
        (impact_energy + 1e-12) / (background_energy + 1e-12)
    )
    centered_span = float(np.ptp(channels[0]))
    if centered_span <= 1e-12:
        clipping_fraction = 0.0
        dropout_fraction = 1.0
    else:
        tolerance = max(centered_span / 65535.0 * 1.5, 1e-9)
        clipping_fraction = float(
            np.mean(
                (np.abs(channels[0] - float(np.min(channels[0]))) <= tolerance)
                | (np.abs(channels[0] - float(np.max(channels[0]))) <= tolerance)
            )
        )
        dropout_fraction = float(
            np.mean(np.abs(np.diff(channels[0])) <= tolerance)
        )
    labels = {
        "seed": int(seed),
        "scenario": scenario,
        "sample_rate_hz": config.sample_rate_hz,
        "window_samples": config.window_samples,
        "channels": config.channels,
        "event_count": event_count,
        "ips_hz": event_count / duration_s,
        "impact_energy": impact_energy,
        "background_energy": background_energy,
        "window_energy": window_energy,
        "observed_window_energy": observed_window_energy,
        "observed_background_energy": observed_background_energy,
        "relative_energy_db": relative_energy_db,
        "analytic_relative_energy_db": analytic_relative_energy_db,
        "impact_to_background_db": impact_to_background_db,
        "angle_deg": angle_deg,
        "event_indices": events.tolist(),
        "noise_rms": noise_rms,
        "gain": gain,
        "severity": severity,
        "air_velocity_m_s": float(air_velocity_m_s),
        "snr_db": snr_db,
        "clipping_fraction": clipping_fraction,
        "dropout_fraction": dropout_fraction,
        "signal_plus_noise_energy": signal_plus_noise,
        "config": asdict(config),
    }
    return SyntheticWindow(waveform=channels, labels=labels)
