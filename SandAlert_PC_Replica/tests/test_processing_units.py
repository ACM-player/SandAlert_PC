from uuid import uuid4

import numpy as np
import pytest

from server.processing import SessionProcessor
from shared.contracts import WaveformMetadata
from shared.settings import ServerSettings


def metadata(session_id, data, fs=2_000_000, sequence=0):
    return WaveformMetadata(
        session_id=session_id, sequence=sequence, timestamp_ns=1_000_000_000 + sequence * 250_000_000,
        fs_hz=fs, channels=data.shape[0], samples=data.shape[1], source="test",
    )


def test_metadata_frequency_overrides_legacy_setting_and_format_changes_reset_window():
    session_id = uuid4()
    processor = SessionProcessor(session_id, ServerSettings(tdoa_enabled=False))
    data = np.full((1, 500), -2.85)
    first = processor.process(metadata(session_id, data), data)
    assert processor.dsp.fs == 2_000_000
    assert first.ips == 0
    assert first.mass_rate == 0
    second_data = np.full((4, 500), 0.3)
    second = processor.process(metadata(session_id, second_data, fs=1000), second_data)
    assert processor.dsp.fs == 1000
    assert second.mass_rate == 0
    assert processor.samples_seen == 500


def test_large_waveform_is_not_truncated_and_accumulation_matches_split_frames():
    session_id = uuid4()
    settings = ServerSettings(tdoa_enabled=False, threshold_margin=0.1)
    data = np.zeros((1, 6000), dtype=np.float32)
    data[0, 250:2000:500] = 0.2  # All impacts are in the first half.
    whole = SessionProcessor(session_id, settings)
    full_result = whole.process(metadata(session_id, data, fs=1000), data)
    split = SessionProcessor(session_id, settings)
    integral = 0.0
    for index, start in enumerate(range(0, data.shape[1], 500)):
        chunk = data[:, start:start + 500]
        result = split.process(metadata(session_id, chunk, fs=1000, sequence=index), chunk)
        integral += result.mass_rate * 0.5
    assert full_result.cumulative_mass > 0
    assert full_result.cumulative_mass == pytest.approx(split.cumulative_mass)
    assert integral == pytest.approx(split.cumulative_mass)
    assert full_result.mass_rate * 6 == pytest.approx(full_result.cumulative_mass)


def test_shape_mismatch_does_not_advance_processor_state():
    session_id = uuid4()
    processor = SessionProcessor(session_id, ServerSettings())
    data = np.zeros((4, 500))
    with pytest.raises(ValueError, match="match metadata"):
        processor.process(metadata(session_id, data), data[:, :499])
    assert processor.samples_seen == 0
