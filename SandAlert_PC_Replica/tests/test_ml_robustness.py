from uuid import uuid4

import numpy as np

from server.ml_shadow import ShadowInferenceEngine
from shared.contracts import AnalysisResult, WaveformMetadata
from shared.settings import ServerSettings


class _ReadyModel:
    ready = True
    reason = ""
    model_id = "test-v2"
    target_fs_hz = None
    window_samples = 16
    aux_feature_names = []
    aux_mean = np.empty(0, dtype=np.float32)
    aux_std = np.empty(0, dtype=np.float32)

    @staticmethod
    def predict(waveform, aux_features=None):
        return 0.5, 0.0, 0.0


def _analysis(session_id, sequence):
    return AnalysisResult(
        session_id=session_id,
        sequence=sequence,
        timestamp_ns=(sequence + 1) * 1_000_000_000,
        ips=0,
        mass_rate=0.0,
        window_energy=0.0,
        threshold=0.1,
        alarm_level="normal",
        cumulative_mass=0.0,
    )


def test_corrupt_window_is_skipped_without_affecting_dsp_result():
    settings = ServerSettings(
        ml_enabled=True,
        ml_background_min_frames=4,
        ml_max_emit_hz=100.0,
        ml_max_dropout_fraction=0.1,
    )
    engine = ShadowInferenceEngine(settings, _ReadyModel())
    session_id = uuid4()
    result = None
    for sequence in range(4):
        meta = WaveformMetadata(
            session_id=session_id,
            sequence=sequence,
            timestamp_ns=(sequence + 1) * 1_000_000_000,
            fs_hz=1_000_000,
            channels=1,
            samples=16,
            source="quality-test",
        )
        result = engine.process(
            meta,
            np.zeros((1, 16), dtype=np.float32),
            _analysis(session_id, sequence),
            signal_energy=1e-6,
        )
    assert result is not None
    assert result.status == "skipped"
    assert "dropout_excessive" in result.quality_flags
    assert result.event_probability is None


def test_air_velocity_context_expires_independently():
    settings = ServerSettings(ml_context_max_age_seconds=1.0)
    engine = ShadowInferenceEngine(settings, _ReadyModel())
    engine.set_context(18.0, 1_000_000_000)
    assert engine._context_for(1_500_000_000) == (18.0, [])
    velocity, flags = engine._context_for(3_000_000_000)
    assert velocity is None
    assert flags == ["air_velocity_context_stale"]
