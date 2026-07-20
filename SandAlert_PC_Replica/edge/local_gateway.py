from __future__ import annotations

import asyncio
import json

from websockets.asyncio.server import serve


class LocalGateway:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.clients: set = set()
        self.latest_waveform: bytes | None = None
        self.latest_events: dict[str, str] = {}

    async def _handler(self, websocket) -> None:
        self.clients.add(websocket)
        try:
            await websocket.send(json.dumps({
                "type": "edge.local.ready", "version": 1,
            }))
            if self.latest_waveform is not None:
                await websocket.send(self.latest_waveform)
            for event in self.latest_events.values():
                await websocket.send(event)
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)

    async def serve(self, stop_event: asyncio.Event) -> None:
        async with serve(self._handler, self.host, self.port, max_size=None):
            await stop_event.wait()

    async def broadcast(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            self.latest_waveform = message
        else:
            try:
                event_type = json.loads(message).get("type")
                if event_type in {
                    "analysis.result", "tdoa.result", "modbus.result",
                    "edge.connection", "service.error",
                }:
                    self.latest_events[event_type] = message
            except (ValueError, AttributeError):
                pass
        if not self.clients:
            return
        dead = []
        for client in tuple(self.clients):
            try:
                await client.send(message)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)
