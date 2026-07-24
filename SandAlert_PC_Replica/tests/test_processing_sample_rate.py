from uuid import uuid4

import numpy as np

from server.processing import ProcessorRegistry
from shared.contracts import WaveformMetadata
from shared.settings import ServerSettings


def _meta(session_id, sequence, fs_hz):
    return WaveformMetadata(
        session_id=session_id,
        sequence=sequence,
        timestamp_ns=(sequence + 1) * 1_000_000_000,
        fs_hz=fs_hz,
        channels=1,
        samples=500,
        source="sample-rate-test",
    )


def test_processor_uses_metadata_sample_rate_and_resets_buffers():
    session_id = uuid4()
    registry = ProcessorRegistry(ServerSettings(dsp_sample_rate=1000))
    data = np.zeros((1, 500), dtype=np.float32)

    registry.process(_meta(session_id, 0, 2_000_000), data)
    processor = registry._processors[str(session_id)]
    assert processor.dsp.fs == 2_000_000
    assert processor.samples_seen == 500

    registry.process(_meta(session_id, 1, 1_000_000), data)
    assert processor.dsp.fs == 1_000_000
    assert processor.samples_seen == 500
    assert processor.shadow.current_fs_hz == 1_000_000
