from __future__ import annotations

import json
import struct

import numpy as np

from shared.contracts import WaveformMetadata


MAGIC = b"SAWF"
HEADER_PREFIX = struct.Struct(">4sI")
MAX_HEADER_BYTES = 16 * 1024


def encode_waveform(metadata: WaveformMetadata, data: np.ndarray) -> bytes:
    array = np.asarray(data, dtype="<f4", order="C")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    expected = (metadata.channels, metadata.samples)
    if array.shape != expected:
        raise ValueError(f"waveform shape {array.shape} does not match {expected}")
    header = metadata.model_dump_json().encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise ValueError("waveform header is too large")
    return HEADER_PREFIX.pack(MAGIC, len(header)) + header + array.tobytes(order="C")


def decode_waveform(payload: bytes) -> tuple[WaveformMetadata, np.ndarray]:
    if len(payload) < HEADER_PREFIX.size:
        raise ValueError("waveform frame is truncated")
    magic, header_len = HEADER_PREFIX.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("invalid waveform magic")
    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise ValueError("invalid waveform header length")
    header_end = HEADER_PREFIX.size + header_len
    if header_end > len(payload):
        raise ValueError("waveform header is truncated")
    metadata = WaveformMetadata.model_validate(
        json.loads(payload[HEADER_PREFIX.size:header_end].decode("utf-8"))
    )
    expected_bytes = metadata.channels * metadata.samples * 4
    raw = payload[header_end:]
    if len(raw) != expected_bytes:
        raise ValueError(
            f"waveform payload has {len(raw)} bytes; expected {expected_bytes}"
        )
    array = np.frombuffer(raw, dtype="<f4").reshape(
        metadata.channels, metadata.samples
    )
    return metadata, array.copy()
