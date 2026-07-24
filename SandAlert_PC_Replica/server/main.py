from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from server.processing import ProcessorRegistry
from server.storage import Database, HDF5Archive
from shared.contracts import (
    Ack,
    EdgeHello,
    MLContext,
    ModbusReading,
    parse_json_message,
)
from shared.settings import ServerSettings, load_server_settings
from shared.waveform_codec import decode_waveform


class AppState:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.database = Database(settings.database_url)
        self.archive = HDF5Archive(settings.waveform_dir)
        self.processors = ProcessorRegistry(settings)


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    resolved = settings or ServerSettings()
    state = AppState(resolved)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.database.create_schema()
        yield

    app = FastAPI(title="SandAlert Compute Server", version="1.0.0", lifespan=lifespan)
    app.state.sandalert = state

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "sandalert-server", "version": 1}

    @app.get("/v1/sessions/{session_id}/export.csv", response_class=PlainTextResponse)
    async def export_csv(session_id: str, token: str = Query(default="")):
        if token != resolved.token:
            raise HTTPException(status_code=401, detail="invalid token")
        rows = await asyncio.to_thread(state.database.analysis_rows, session_id)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Timestamp(ns)", "Sequence", "IPS", "MassRate(g/s)",
            "CumulativeMass(g)", "Alarm", "TDOA_Angle(deg)", "TDOA_Cost",
        ])
        for row in rows:
            writer.writerow([
                row.timestamp_ns, row.sequence, row.ips, row.mass_rate,
                row.cumulative_mass, row.alarm_level,
                row.tdoa_angle_deg if row.tdoa_angle_deg is not None else "N/A",
                row.tdoa_cost if row.tdoa_cost is not None else "N/A",
            ])
        return output.getvalue()

    @app.websocket("/v1/ingest")
    async def ingest(websocket: WebSocket, token: str = Query(default="")):
        auth_header = websocket.headers.get("authorization", "")
        bearer = auth_header.removeprefix("Bearer ").strip()
        if token != resolved.token and bearer != resolved.token:
            await websocket.close(code=4401, reason="invalid token")
            return
        await websocket.accept()
        edge_id = "unknown"
        try:
            first = await websocket.receive_text()
            hello = EdgeHello.model_validate(parse_json_message(first))
            edge_id = hello.edge_id
            await asyncio.to_thread(
                state.database.ensure_session, hello.session_id, hello.edge_id
            )
            await websocket.send_text(json.dumps({
                "type": "edge.ready", "version": 1, "edge_id": edge_id,
                "session_id": str(hello.session_id),
            }))
            await websocket.send_text(
                state.processors.ml_status().model_dump_json()
            )

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    meta, array = decode_waveform(message["bytes"])
                    duplicate = await asyncio.to_thread(
                        state.database.waveform_exists, meta.session_id, meta.sequence
                    )
                    if not duplicate:
                        archive_path = await asyncio.to_thread(
                            state.archive.write, meta, array
                        )
                        await asyncio.to_thread(
                            state.database.add_waveform, meta, archive_path
                        )
                        processing_output = await asyncio.to_thread(
                            state.processors.process_detailed, meta, array
                        )
                        result = processing_output.analysis
                        await asyncio.to_thread(state.database.add_analysis, result)
                    await websocket.send_text(Ack(
                        session_id=meta.session_id, sequence=meta.sequence,
                        kind="waveform", duplicate=duplicate,
                    ).model_dump_json())
                    if not duplicate:
                        await websocket.send_text(result.model_dump_json())
                        if processing_output.ml_result is not None:
                            await websocket.send_text(
                                processing_output.ml_result.model_dump_json()
                            )
                        if result.tdoa_angle_deg is not None:
                            await websocket.send_text(json.dumps({
                                "type": "tdoa.result", "version": 1,
                                "data": {
                                    "session_id": str(result.session_id),
                                    "sequence": result.sequence,
                                    "angle_deg": result.tdoa_angle_deg,
                                    "cost": result.tdoa_cost,
                                    "reliable": result.tdoa_reliable,
                                },
                            }))
                    continue

                raw_text = message.get("text")
                if raw_text is None:
                    continue
                payload = parse_json_message(raw_text)
                if payload["type"] == "modbus.reading":
                    reading = ModbusReading.model_validate(payload)
                    duplicate = await asyncio.to_thread(
                        state.database.modbus_exists,
                        reading.session_id,
                        reading.sequence,
                    )
                    if not duplicate:
                        await asyncio.to_thread(state.database.add_modbus, reading)
                    await websocket.send_text(Ack(
                        session_id=reading.session_id, sequence=reading.sequence,
                        kind="modbus", duplicate=duplicate,
                    ).model_dump_json())
                    await websocket.send_text(json.dumps({
                        "type": "modbus.result", "version": 1,
                        "data": reading.model_dump(mode="json"),
                    }))
                elif payload["type"] == "ml.context":
                    context = MLContext.model_validate(payload)
                    state.processors.update_ml_context(context)
                    await websocket.send_text(json.dumps({
                        "type": "ml.context.ack",
                        "version": 1,
                        "data": {
                            "session_id": str(context.session_id),
                            "timestamp_ns": context.timestamp_ns,
                            "air_velocity_available": (
                                context.air_velocity_m_s is not None
                            ),
                            "source": context.source,
                        },
                    }))
                elif payload["type"] == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "version": 1}))
        except WebSocketDisconnect:
            return
        except Exception as exc:
            try:
                await websocket.send_text(json.dumps({
                    "type": "service.error", "version": 1,
                    "data": {"message": str(exc), "edge_id": edge_id},
                }))
                await websocket.close(code=1011)
            except Exception:
                # 客户端已断开或服务正在关闭时，不再尝试二次写入连接。
                return

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="SandAlert compute server")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    settings = load_server_settings(args.config)
    uvicorn.run(
        create_app(settings), host=settings.host, port=settings.port,
        log_level="info", workers=1,
    )


if __name__ == "__main__":
    main()
