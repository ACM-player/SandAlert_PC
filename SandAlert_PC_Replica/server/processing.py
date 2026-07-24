from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import numpy as np

from core.dsp_engine import DSPEngine
from core.tdoa_engine import TDOAEngine
from server.ml_shadow import ONNXShadowModel, ShadowInferenceEngine
from shared.contracts import (
    AnalysisResult,
    MLContext,
    MLResult,
    MLStatus,
    WaveformMetadata,
)
from shared.settings import ServerSettings


@dataclass(frozen=True)
class ProcessingOutput:
    analysis: AnalysisResult
    ml_result: MLResult | None


@dataclass
class SessionProcessor:
    session_id: UUID
    settings: ServerSettings
    ml_model: ONNXShadowModel

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
        self.current_fs_hz: int | None = None
        self.last_signal_energy = 0.0
        self.shadow = ShadowInferenceEngine(self.settings, self.ml_model)

    def _set_sample_rate(self, fs_hz: int) -> None:
        if self.current_fs_hz == fs_hz:
            return
        self.current_fs_hz = int(fs_hz)
        self.dsp.set_sample_rate(fs_hz)
        self.tdoa.fs = fs_hz
        self.voltage.fill(0.0)
        self.multichannel.fill(0.0)
        self.samples_seen = 0
        self.last_tdoa_timestamp_ns = 0
        self.impact_active = False
        self.quiet_frames = 0
        self.shadow.reset(fs_hz)

    def process(self, meta: WaveformMetadata, data: np.ndarray) -> AnalysisResult:
        if data.ndim != 2:
            raise ValueError("waveform data must have shape (channels, samples)")
        self._set_sample_rate(meta.fs_hz)
        shift = min(data.shape[1], self.window_len)
        ch1 = data[0, -shift:]
        self.voltage = np.roll(self.voltage, -shift)
        self.voltage[-shift:] = ch1
        if data.shape[0] == 4:
            self.multichannel = np.roll(self.multichannel, -shift, axis=1)
            self.multichannel[:, -shift:] = data[:, -shift:]
        self.samples_seen += shift

        processed, threshold, ips, mass_rate, energy, _ = self.dsp.process_window(
            self.voltage,
            self.settings.threshold_margin,
            self.settings.calibration_factor,
        )
        self.last_signal_energy = float(np.mean(np.square(processed)))
        dt = meta.samples / float(meta.fs_hz)
        self.cumulative_mass += float(mass_rate) * dt

        if mass_rate >= self.settings.alarm_limit:
            alarm = "critical"
        elif mass_rate >= self.settings.caution_limit:
            alarm = "caution"
        else:
            alarm = "normal"

        angle = None
        cost = None
        reliable = False
        peak_idx = int(np.argmax(processed))
        if self.impact_active and peak_idx < 750:
            # 已定位的撞击已移出窗口左侧，允许下一个物理事件重新触发。
            self.impact_active = False
        if ips == 0:
            self.quiet_frames += 1
            if self.quiet_frames >= 2:
                self.impact_active = False
        else:
            self.quiet_frames = 0

        if (
            self.settings.tdoa_enabled
            and data.shape[0] == 4
            and self.samples_seen >= 2000
            and ips > 0
            and not self.impact_active
            and 1250 <= peak_idx <= 1750
            and meta.timestamp_ns - self.last_tdoa_timestamp_ns >= 100_000_000
        ):
            end = min(peak_idx + 1000, self.window_len)
            start = end - 2000
            if start >= 0:
                slice_4ch = self.multichannel[:, start:end].copy()
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
        self._ml_contexts: dict[str, MLContext] = {}
        self._ml_model = ONNXShadowModel(settings)

    def ml_status(self) -> MLStatus:
        return self._ml_model.status_event()

    def _get_processor(self, meta: WaveformMetadata) -> SessionProcessor:
        key = str(meta.session_id)
        processor = self._processors.get(key)
        if processor is None:
            processor = SessionProcessor(
                meta.session_id, self.settings, self._ml_model
            )
            context = self._ml_contexts.get(key)
            if context is not None:
                processor.shadow.set_context(
                    context.air_velocity_m_s, context.timestamp_ns
                )
            self._processors[key] = processor
        return processor

    def update_ml_context(self, context: MLContext) -> None:
        key = str(context.session_id)
        self._ml_contexts[key] = context
        processor = self._processors.get(key)
        if processor is not None:
            processor.shadow.set_context(
                context.air_velocity_m_s, context.timestamp_ns
            )

    def process(self, meta: WaveformMetadata, data: np.ndarray) -> AnalysisResult:
        return self._get_processor(meta).process(meta, data)

    def process_detailed(
        self, meta: WaveformMetadata, data: np.ndarray
    ) -> ProcessingOutput:
        processor = self._get_processor(meta)
        analysis = processor.process(meta, data)
        ml_result = None
        if self.settings.ml_enabled:
            ml_result = processor.shadow.process(
                meta, data, analysis, processor.last_signal_energy
            )
        return ProcessingOutput(analysis=analysis, ml_result=ml_result)
