from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient

from server.main import create_app
from shared.contracts import EdgeHello, MLContext, WaveformMetadata
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
            ml_status = websocket.receive_json()
            assert ml_status["type"] == "ml.status"
            assert ml_status["status"] == "disabled"
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


def test_websocket_ml_unavailable_does_not_block_analysis(tmp_path):
    settings = ServerSettings(
        token="test-token",
        database_url=f"sqlite:///{tmp_path / 'server.sqlite3'}",
        waveform_dir=tmp_path / "waveforms",
        ml_enabled=True,
        ml_model_path=tmp_path / "missing.onnx",
        ml_background_min_frames=4,
        ml_max_emit_hz=100.0,
    )
    session_id = uuid4()
    app = create_app(settings)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/ingest?token=test-token"
        ) as websocket:
            websocket.send_text(EdgeHello(
                edge_id="test-edge", session_id=session_id
            ).model_dump_json())
            assert websocket.receive_json()["type"] == "edge.ready"
            status = websocket.receive_json()
            assert status["type"] == "ml.status"
            assert status["status"] == "unavailable"
            data = np.zeros((1, 500), dtype=np.float32)
            meta = WaveformMetadata(
                session_id=session_id,
                sequence=0,
                timestamp_ns=1_000_000_000,
                fs_hz=2_000_000,
                channels=1,
                samples=500,
                source="test",
            )
            websocket.send_bytes(encode_waveform(meta, data))
            assert websocket.receive_json()["type"] == "ack"
            assert websocket.receive_json()["type"] == "analysis.result"
            ml_result = websocket.receive_json()
            assert ml_result["type"] == "ml.result"
            assert ml_result["status"] == "warmup"


def test_websocket_accepts_optional_ml_air_velocity_context(tmp_path):
    settings = ServerSettings(
        token="test-token",
        database_url=f"sqlite:///{tmp_path / 'server.sqlite3'}",
        waveform_dir=tmp_path / "waveforms",
    )
    session_id = uuid4()
    app = create_app(settings)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/ingest?token=test-token"
        ) as websocket:
            websocket.send_text(EdgeHello(
                edge_id="test-edge", session_id=session_id
            ).model_dump_json())
            assert websocket.receive_json()["type"] == "edge.ready"
            assert websocket.receive_json()["type"] == "ml.status"
            websocket.send_text(MLContext(
                session_id=session_id,
                timestamp_ns=1_000_000_000,
                air_velocity_m_s=18.5,
                source="simulated-flow-meter",
            ).model_dump_json())
            ack = websocket.receive_json()
            assert ack["type"] == "ml.context.ack"
            assert ack["data"]["air_velocity_available"] is True
            context = app.state.sandalert.processors._ml_contexts[str(session_id)]
            assert context.air_velocity_m_s == 18.5
