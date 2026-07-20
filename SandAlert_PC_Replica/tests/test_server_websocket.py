from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient

from server.main import create_app
from shared.contracts import EdgeHello, WaveformMetadata
from shared.settings import ServerSettings
from shared.waveform_codec import encode_waveform


def test_websocket_ingest_ack_and_analysis(tmp_path):
    settings = ServerSettings(
        token="test-token",
        database_url=f"sqlite:///{tmp_path / 'server.sqlite3'}",
        waveform_dir=tmp_path / "waveforms",
    )
    session_id = uuid4()
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        with client.websocket_connect(
            "/v1/ingest?token=test-token"
        ) as websocket:
            websocket.send_text(EdgeHello(
                edge_id="test-edge", session_id=session_id
            ).model_dump_json())
            assert websocket.receive_json()["type"] == "edge.ready"
            data = np.zeros((4, 500), dtype=np.float32)
            meta = WaveformMetadata(
                session_id=session_id, sequence=0, fs_hz=2_000_000,
                channels=4, samples=500, source="test",
            )
            websocket.send_bytes(encode_waveform(meta, data))
            assert websocket.receive_json()["type"] == "ack"
            result = websocket.receive_json()
            assert result["type"] == "analysis.result"
            assert result["sequence"] == 0
