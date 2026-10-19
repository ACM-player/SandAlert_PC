from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import numpy as np

from core.dsp_engine import DSPEngine
from core.tdoa_engine import TDOAEngine
from shared.contracts import AnalysisResult, WaveformMetadata
from shared.settings import ServerSettings


@dataclass
class SessionProcessor:
    session_id: UUID
    settings: ServerSettings

    def __post_init__(self) -> None:
        self.dsp = DSPEngine(sample_rate=self.settings.dsp_sample_rate)
        self.tdoa = TDOAEngine(
            fs=2_000_000,
            radius=0.05,
            c=1500.0,
            bandpass_enabled=self.settings.tdoa_bandpass_enabled,
            band_low_hz=self.settings.tdoa_band_low_hz,
            band_high_hz=self.settings.tdoa_band_high_hz,
        )
        self.window_len = 3000
        self.voltage = np.zeros(self.window_len, dtype=np.float32)
        self.multichannel = np.zeros((4, self.window_len), dtype=np.float32)
        self.samples_seen = 0
        self.cumulative_mass = 0.0
        self.last_tdoa_timestamp_ns = 0
        self.impact_active = False
        self.quiet_frames = 0
        self._stream_format = None

    def process(self, meta: WaveformMetadata, data: np.ndarray) -> AnalysisResult:
        if data.shape != (meta.channels, meta.samples) or not np.all(np.isfinite(data)):
            raise ValueError("waveform must match metadata and contain finite samples")
        stream_format = (meta.fs_hz, meta.channels)
        if stream_format != self._stream_format:
            self.dsp.set_sample_rate(meta.fs_hz)
            self.tdoa.fs = meta.fs_hz
            self.voltage.fill(data[0, 0])
            self.multichannel.fill(0)
            if meta.channels == 4:
                self.multichannel[:] = data[:, :1]
            self.samples_seen = 0
            self.impact_active = False
            self.quiet_frames = 0
            self.last_tdoa_timestamp_ns = 0
            self._stream_format = stream_format

        # Analyze every sample even if a caller sends a frame larger than the
        # rolling window. Analyze large frames as consecutive 500-sample pieces.
        weighted_mass_rate = 0.0
        weighted_ips = 0
        latest_tdoa = None
        for start in range(0, meta.samples, 500):
            chunk = data[:, start:start + 500]
            chunk_meta = meta.model_copy(update={
                "samples": chunk.shape[1],
                "timestamp_ns": meta.timestamp_ns + round(start / meta.fs_hz * 1e9),
            })
            result = self._process_chunk(chunk_meta, chunk)
            weighted_ips += result.ips * chunk.shape[1]
            weighted_mass_rate += result.mass_rate * chunk.shape[1]
            if result.tdoa_angle_deg is not None:
                latest_tdoa = result
        mass_rate = weighted_mass_rate / meta.samples
        return result.model_copy(update={
            "timestamp_ns": meta.timestamp_ns,
            "ips": int(round(weighted_ips / meta.samples)),
            "mass_rate": mass_rate,
            "alarm_level": self._alarm(mass_rate),
            "tdoa_angle_deg": latest_tdoa.tdoa_angle_deg if latest_tdoa else None,
            "tdoa_cost": latest_tdoa.tdoa_cost if latest_tdoa else None,
            "tdoa_reliable": latest_tdoa.tdoa_reliable if latest_tdoa else False,
        })

    def _alarm(self, mass_rate: float) -> str:
        if mass_rate >= self.settings.alarm_limit:
            return "critical"
        if mass_rate >= self.settings.caution_limit:
            return "caution"
        return "normal"

    def _process_chunk(self, meta: WaveformMetadata, data: np.ndarray) -> AnalysisResult:
        shift = min(data.shape[1], self.window_len)
        ch1 = data[0, -shift:]
        self.voltage = np.roll(self.voltage, -shift)
        self.voltage[-shift:] = ch1
        if data.shape[0] == 4:
            self.multichannel = np.roll(self.multichannel, -shift, axis=1)
            self.multichannel[:, -shift:] = data[:, -shift:]
        self.samples_seen += shift

        processed, threshold, ips, mass_rate, energy, peaks = self.dsp.process_window(
            self.voltage,
            self.settings.threshold_margin,
            self.settings.calibration_factor,
        )
        dt = meta.samples / float(meta.fs_hz)
        self.cumulative_mass += float(mass_rate) * dt

        alarm = self._alarm(mass_rate)

        angle = None
        cost = None
        reliable = False
        peak_idx = int(np.argmax(processed))
        if self.impact_active and peak_idx < 750:
            # 已定位的撞击已移出窗口左侧，允许下一个物理事件重新触发。
            self.impact_active = False
        if not peaks:
            self.quiet_frames += 1
            if self.quiet_frames >= 2:
                self.impact_active = False
        else:
            self.quiet_frames = 0

        if (
            self.settings.tdoa_enabled
            and data.shape[0] == 4
            and self.samples_seen >= 2000
            and len(peaks) > 0
            and not self.impact_active
            and 1250 <= peak_idx <= 1750
            and meta.timestamp_ns - self.last_tdoa_timestamp_ns >= 100_000_000
        ):
            end = min(peak_idx + 1000, self.window_len)
            start = end - 2000
            if start >= 0:
                slice_4ch = self.multichannel[:, start:end].copy()
                self.tdoa.fs = meta.fs_hz
                tdoa_result = self.tdoa.compute_doa_detailed(slice_4ch)
                angle = float(tdoa_result.angle_deg)
                cost = float(np.min(tdoa_result.cost_values))
                reliable = (
                    cost <= 1e-8
                    and tdoa_result.diagnostics.signal_like
                )
                self.last_tdoa_timestamp_ns = meta.timestamp_ns
                self.impact_active = True

        return AnalysisResult(
            session_id=meta.session_id,
            sequence=meta.sequence,
            timestamp_ns=meta.timestamp_ns,
            ips=int(ips),
            mass_rate=float(mass_rate),
            window_energy=float(energy),
            threshold=float(threshold),
            alarm_level=alarm,
            cumulative_mass=self.cumulative_mass,
            tdoa_angle_deg=angle,
            tdoa_cost=cost,
            tdoa_reliable=reliable,
        )


class ProcessorRegistry:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self._processors: dict[str, SessionProcessor] = {}

    def process(self, meta: WaveformMetadata, data: np.ndarray) -> AnalysisResult:
        key = str(meta.session_id)
        processor = self._processors.get(key)
        if processor is None:
            processor = SessionProcessor(meta.session_id, self.settings)
            self._processors[key] = processor
        return processor.process(meta, data)
