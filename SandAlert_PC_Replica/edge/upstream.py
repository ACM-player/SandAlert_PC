from __future__ import annotations

import asyncio
import json
import logging

from websockets.asyncio.client import connect

from edge.local_gateway import LocalGateway
from edge.spool import DurableSpool, spool_io
from shared.contracts import EdgeHello, parse_json_message
from shared.settings import EdgeSettings


LOG = logging.getLogger(__name__)


class UpstreamClient:
    def __init__(
        self,
        settings: EdgeSettings,
        session_id,
        spool: DurableSpool,
        gateway: LocalGateway,
    ):
        self.settings = settings
        self.session_id = session_id
        self.spool = spool
        self.gateway = gateway
        self._ack_events: dict[tuple[str, int, str], asyncio.Event] = {}

    async def run(self, stop_event: asyncio.Event) -> None:
        delay = 1.0
        while not stop_event.is_set():
            try:
                async with connect(
                    self.settings.server_url,
                    additional_headers={"Authorization": f"Bearer {self.settings.token}"},
                    max_size=None,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    await websocket.send(EdgeHello(
                        edge_id=self.settings.edge_id,
                        session_id=self.session_id,
                    ).model_dump_json())
                    await self.gateway.broadcast(json.dumps({
                        "type": "edge.connection", "version": 1,
                        "data": {"status": "online"},
                    }))
                    delay = 1.0
                    receiver = asyncio.create_task(self._receive(websocket, stop_event))
                    sender = asyncio.create_task(self._send(websocket, stop_event))
                    try:
                        done, _ = await asyncio.wait(
                            {receiver, sender}, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done:
                            task.result()
                        if not stop_event.is_set():
                            raise ConnectionError("server websocket closed")
                    finally:
                        # Cancellation must also join both tasks before the
                        # owner closes SQLite in run_edge's cleanup.
                        receiver.cancel()
                        sender.cancel()
                        await asyncio.gather(receiver, sender, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.warning("server connection unavailable: %s", exc)
                await self.gateway.broadcast(json.dumps({
                    "type": "edge.connection", "version": 1,
                    "data": {"status": "offline", "reason": str(exc)},
                }))
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, self.settings.reconnect_max_seconds)

    async def _send(self, websocket, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            item = await spool_io(self.spool.first_pending)
            if item is None:
                await asyncio.sleep(0.05)
                continue
            payload = await spool_io(self.spool.read, item)
            key = (item.session_id, item.sequence, item.kind)
            event = asyncio.Event()
            self._ack_events[key] = event
            if item.kind == "waveform":
                await websocket.send(payload)
            else:
                await websocket.send(payload.decode("utf-8"))
            try:
                await asyncio.wait_for(event.wait(), timeout=30.0)
            finally:
                self._ack_events.pop(key, None)

    async def _receive(self, websocket, stop_event: asyncio.Event) -> None:
        async for raw in websocket:
            if stop_event.is_set():
                return
            if isinstance(raw, bytes):
                continue
            payload = parse_json_message(raw)
            if payload["type"] == "ack":
                key = (payload["session_id"], payload["sequence"], payload["kind"])
                await spool_io(
                    self.spool.acknowledge,
                    payload["session_id"], payload["sequence"], payload["kind"],
                )
                event = self._ack_events.get(key)
                if event:
                    event.set()
            else:
                await self.gateway.broadcast(raw)
