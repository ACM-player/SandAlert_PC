from uuid import uuid4

import numpy as np
import pytest

from shared.contracts import WaveformMetadata
from shared.waveform_codec import decode_waveform, encode_waveform


def test_waveform_round_trip():
    data = np.arange(32, dtype=np.float32).reshape(4, 8)
    metadata = WaveformMetadata(
        session_id=uuid4(), sequence=7, fs_hz=2_000_000,
        channels=4, samples=8, source="test",
    )
    decoded_meta, decoded_data = decode_waveform(encode_waveform(metadata, data))
    assert decoded_meta == metadata
    np.testing.assert_array_equal(decoded_data, data)


def test_waveform_rejects_truncated_payload():
    metadata = WaveformMetadata(
        session_id=uuid4(), sequence=0, fs_hz=1000,
        channels=1, samples=4,
    )
    payload = encode_waveform(metadata, np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        decode_waveform(payload[:-1])
