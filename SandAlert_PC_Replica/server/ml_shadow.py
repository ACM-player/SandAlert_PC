from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from shared.contracts import AnalysisResult, MLResult, MLStatus, WaveformMetadata
from shared.ml_features import (
    AUX_FEATURE_NAMES,
    SignalAuxFeatures,
    extract_shadow_aux_features,
)
from shared.settings import ServerSettings


class ONNXShadowModel:
    """Optional ONNX model loader. Failure never stops the DSP service."""

    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.session = None
        self.model_id = "unavailable"
        self.target_fs_hz: int | None = None
        self.window_samples = 8192
        self.input_name = ""
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.aux_feature_names: list[str] = []
        self.aux_mean = np.empty(0, dtype=np.float32)
        self.aux_std = np.empty(0, dtype=np.float32)
        self.event_decision_threshold = 0.5
        self.reason = ""
        if settings.ml_enabled:
            self._load()

    def _load(self) -> None:
        model_path = self.settings.ml_model_path
        if model_path is None or not Path(model_path).is_file():
            self.reason = f"model not found: {model_path}"
            return
        try:
            import onnxruntime as ort
        except ImportError:
            self.reason = "onnxruntime is not installed"
            return

        manifest: dict = {}
        manifest_path = self.settings.ml_manifest_path
        if manifest_path is not None and Path(manifest_path).is_file():
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.reason = f"invalid model manifest: {exc}"
                return
        try:
            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self.input_names = [
                item.name for item in self.session.get_inputs()
            ]
            self.input_name = self.input_names[0]
            self.output_names = [item.name for item in self.session.get_outputs()]
            self.model_id = str(manifest.get("model_id") or Path(model_path).stem)
            target_fs = manifest.get("sample_rate_hz")
            self.target_fs_hz = int(target_fs) if target_fs else None
            self.window_samples = int(
                manifest.get("window_samples") or self.window_samples
            )
            self.aux_feature_names = list(
                manifest.get("aux_feature_names") or []
            )
            self.aux_mean = np.asarray(
                manifest.get("aux_mean") or [], dtype=np.float32
            )
            self.aux_std = np.asarray(
                manifest.get("aux_std") or [], dtype=np.float32
            )
            self.event_decision_threshold = float(
                manifest.get("event_decision_threshold") or 0.5
            )
        except Exception as exc:  # onnxruntime exposes provider-specific errors
            self.session = None
            self.reason = f"ONNX load failed: {exc}"

    @property
    def ready(self) -> bool:
        return self.session is not None

    def status_event(self) -> MLStatus:
        if not self.settings.ml_enabled:
            return MLStatus(status="disabled", reason="shadow inference disabled")
        if self.ready:
            return MLStatus(status="ready", model_id=self.model_id)
        return MLStatus(
            status="unavailable", model_id=self.model_id, reason=self.reason
        )

    def predict(
        self,
        waveform: np.ndarray,
        aux_features: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        if self.session is None:
            raise RuntimeError(self.reason or "ONNX model unavailable")
        feeds = {
            self.input_name: waveform.astype(
                np.float32, copy=False
            )[None, None, :]
        }
        if len(self.input_names) > 1:
            expected = len(self.aux_feature_names) or len(AUX_FEATURE_NAMES)
            if aux_features is None:
                aux_features = np.zeros(expected, dtype=np.float32)
            aux_array = np.asarray(aux_features, dtype=np.float32).reshape(-1)
            if aux_array.size != expected:
                raise ValueError(
                    f"model expects {expected} auxiliary features; "
                    f"received {aux_array.size}"
                )
            feeds[self.input_names[1]] = aux_array[None, :]
        outputs = self.session.run(
            self.output_names,
            feeds,
        )
        values = {
            name: float(np.asarray(value).reshape(-1)[0])
            for name, value in zip(self.output_names, outputs)
        }
        if {"event_probability", "ips_estimate", "relative_energy_db"} <= values.keys():
            return (
                float(np.clip(values["event_probability"], 0.0, 1.0)),
                max(0.0, values["ips_estimate"]),
                values["relative_energy_db"],
            )
        if len(outputs) < 3:
            raise RuntimeError("ONNX model must expose three shadow outputs")
        return (
            float(np.clip(np.asarray(outputs[0]).reshape(-1)[0], 0.0, 1.0)),
            max(0.0, float(np.asarray(outputs[1]).reshape(-1)[0])),
            float(np.asarray(outputs[2]).reshape(-1)[0]),
        )


class ShadowInferenceEngine:
    """Per-session background tracking, buffering and rate-limited inference."""

    EPSILON = 1e-12

    def __init__(self, settings: ServerSettings, model: ONNXShadowModel):
        self.settings = settings
        self.model = model
        self.current_fs_hz: int | None = None
        self.background_energies: deque[float] = deque()
        self.waveform = np.empty(0, dtype=np.float32)
        self.target_samples_seen = 0
        self.last_emit_timestamp_ns = 0
        self.coalesced_windows = 0
        self.air_velocity_m_s = settings.ml_air_velocity_m_s
        self.context_timestamp_ns = 0

    def reset(self, fs_hz: int) -> None:
        self.current_fs_hz = int(fs_hz)
        self.background_energies.clear()
        self.waveform = np.empty(0, dtype=np.float32)
        self.target_samples_seen = 0
        self.last_emit_timestamp_ns = 0
        self.coalesced_windows = 0

    def set_context(
        self,
        air_velocity_m_s: float | None,
        timestamp_ns: int,
    ) -> None:
        self.air_velocity_m_s = air_velocity_m_s
        self.context_timestamp_ns = int(timestamp_ns)

    def _context_for(self, timestamp_ns: int) -> tuple[float | None, list[str]]:
        if self.air_velocity_m_s is None:
            return None, ["air_velocity_missing"]
        if self.context_timestamp_ns <= 0:
            return self.air_velocity_m_s, []
        age_ns = max(0, timestamp_ns - self.context_timestamp_ns)
        if age_ns > int(self.settings.ml_context_max_age_seconds * 1e9):
            return None, ["air_velocity_context_stale"]
        return self.air_velocity_m_s, []

    def _quality_flags(self, features: SignalAuxFeatures) -> list[str]:
        flags = []
        if (
            features.clipping_fraction
            > self.settings.ml_max_clipping_fraction
        ):
            flags.append("clipping_excessive")
        if (
            features.dropout_fraction
            > self.settings.ml_max_dropout_fraction
        ):
            flags.append("dropout_excessive")
        if (
            self.model.aux_feature_names
            and self.model.aux_mean.size == len(self.model.aux_feature_names)
            and self.model.aux_std.size == len(self.model.aux_feature_names)
        ):
            spectral_names = {
                "spectral_centroid_fraction",
                "spectral_bandwidth_fraction",
                "band_low_fraction",
                "band_mid_fraction",
                "band_high_fraction",
            }
            indices = [
                index
                for index, name in enumerate(self.model.aux_feature_names)
                if name in spectral_names
            ]
            if indices:
                z = np.abs(
                    (
                        features.values[indices]
                        - self.model.aux_mean[indices]
                    )
                    / np.maximum(self.model.aux_std[indices], 1e-6)
                )
                if float(np.max(z)) > self.settings.ml_spectral_ood_z_limit:
                    flags.append("spectral_ood")
        return flags

    def _background_capacity(self, meta: WaveformMetadata) -> int:
        frames_per_second = meta.fs_hz / max(meta.samples, 1)
        return max(
            self.settings.ml_background_min_frames,
            int(math.ceil(frames_per_second * self.settings.ml_background_seconds)),
        )

    def _update_background(
        self, meta: WaveformMetadata, analysis: AnalysisResult, signal_energy: float
    ) -> tuple[float | None, float | None]:
        capacity = self._background_capacity(meta)
        if self.background_energies.maxlen != capacity:
            self.background_energies = deque(
                self.background_energies, maxlen=capacity
            )
        if analysis.ips == 0 and math.isfinite(signal_energy):
            self.background_energies.append(max(signal_energy, self.EPSILON))
        if len(self.background_energies) < self.settings.ml_background_min_frames:
            return None, None
        background = float(np.median(np.asarray(self.background_energies)))
        relative_db = 10.0 * math.log10(
            (max(signal_energy, 0.0) + self.EPSILON)
            / (background + self.EPSILON)
        )
        return background, relative_db

    def _append_waveform(self, meta: WaveformMetadata, data: np.ndarray) -> None:
        channel = np.asarray(data[0], dtype=np.float32)
        target_fs = self.model.target_fs_hz or meta.fs_hz
        if target_fs != meta.fs_hz:
            divisor = math.gcd(int(target_fs), int(meta.fs_hz))
            channel = resample_poly(
                channel, int(target_fs) // divisor, int(meta.fs_hz) // divisor
            ).astype(np.float32, copy=False)
        self.target_samples_seen += channel.size
        self.waveform = np.concatenate((self.waveform, channel))
        max_samples = max(self.model.window_samples * 2, self.model.window_samples)
        if self.waveform.size > max_samples:
            self.waveform = self.waveform[-max_samples:]

    @staticmethod
    def _normalize(waveform: np.ndarray) -> np.ndarray:
        centered = waveform - np.median(waveform)
        mad = float(np.median(np.abs(centered)))
        scale = max(1.4826 * mad, float(np.std(centered)) * 0.1, 1e-8)
        return np.clip(centered / scale, -20.0, 20.0).astype(np.float32)

    def process(
        self,
        meta: WaveformMetadata,
        data: np.ndarray,
        analysis: AnalysisResult,
        signal_energy: float,
    ) -> MLResult | None:
        if self.current_fs_hz != meta.fs_hz:
            self.reset(meta.fs_hz)

        background, relative_db = self._update_background(
            meta, analysis, signal_energy
        )
        self._append_waveform(meta, data)

        min_interval_ns = int(1_000_000_000 / self.settings.ml_max_emit_hz)
        if (
            self.last_emit_timestamp_ns
            and meta.timestamp_ns - self.last_emit_timestamp_ns < min_interval_ns
        ):
            self.coalesced_windows += 1
            return None
        self.last_emit_timestamp_ns = meta.timestamp_ns

        flags: list[str] = []
        if background is None:
            flags.append("background_warmup")
        if self.waveform.size < self.model.window_samples:
            flags.append("waveform_warmup")

        common = dict(
            session_id=meta.session_id,
            sequence=meta.sequence,
            timestamp_ns=meta.timestamp_ns,
            source=meta.source,
            sample_rate_hz=meta.fs_hz,
            model_id=self.model.model_id,
            relative_energy_db=relative_db,
            background_energy=background,
            coalesced_windows=self.coalesced_windows,
        )
        self.coalesced_windows = 0

        if flags:
            return MLResult(status="warmup", quality_flags=flags, **common)
        if not self.model.ready:
            return MLResult(
                status="unavailable",
                quality_flags=["model_unavailable", self.model.reason],
                **common,
            )

        raw_window = self.waveform[-self.model.window_samples:]
        air_velocity, context_flags = self._context_for(meta.timestamp_ns)
        try:
            aux_features = extract_shadow_aux_features(
                raw_window,
                self.model.target_fs_hz or meta.fs_hz,
                relative_db,
                air_velocity,
            )
        except ValueError as exc:
            return MLResult(
                status="skipped",
                quality_flags=[f"invalid_aux_features:{exc}"],
                **common,
            )
        quality_flags = context_flags + self._quality_flags(aux_features)
        hard_rejection_flags = {
            "clipping_excessive",
            "dropout_excessive",
            "spectral_ood",
        }
        if any(flag in hard_rejection_flags for flag in quality_flags):
            return MLResult(
                status="skipped",
                quality_flags=quality_flags,
                **common,
            )

        window = self._normalize(raw_window)
        started = time.perf_counter()
        try:
            event_probability, ips_estimate, energy_estimate = self.model.predict(
                window, aux_features.values
            )
        except Exception as exc:
            return MLResult(
                status="unavailable",
                inference_latency_ms=(time.perf_counter() - started) * 1000.0,
                quality_flags=[f"inference_failed:{exc}"],
                **common,
            )
        return MLResult(
            status="ready",
            event_probability=event_probability,
            ips_estimate=ips_estimate,
            ml_relative_energy_db=energy_estimate,
            inference_latency_ms=(time.perf_counter() - started) * 1000.0,
            quality_flags=quality_flags,
            **common,
        )
