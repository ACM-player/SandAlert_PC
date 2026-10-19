from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path
from uuid import uuid4

from edge.acquisition import FileReplaySource, MockModbusSource
from edge.local_gateway import LocalGateway
from edge.spool import DurableSpool, SpoolFullError, spool_io
from edge.upstream import UpstreamClient
from shared.settings import load_edge_settings
from shared.waveform_codec import encode_waveform


LOG = logging.getLogger(__name__)


async def run_edge(config_path: Path | None = None) -> None:
    settings = load_edge_settings(config_path)
    session_id = uuid4()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    spool = DurableSpool(settings.spool_dir, settings.max_spool_gib)
    gateway = LocalGateway(settings.local_host, settings.local_port)
    upstream = UpstreamClient(settings, session_id, spool, gateway)

    async def waveform_producer() -> None:
        source = FileReplaySource(
            settings.replay_path, session_id, settings.replay_fs_hz,
            settings.replay_chunk_samples, settings.replay_speed,
            settings.replay_loop,
        )
        await asyncio.sleep(settings.replay_start_delay_seconds)
        async for meta, array in source.frames():
            if stop_event.is_set():
                return
            payload = encode_waveform(meta, array)
            await gateway.broadcast(payload)
            try:
                await spool_io(
                    spool.enqueue, str(session_id), meta.sequence, "waveform", payload
                )
            except SpoolFullError as exc:
                LOG.error("%s; stopping acquisition", exc)
                stop_event.set()
                return
        LOG.info("file replay completed")

    async def modbus_producer() -> None:
        source = MockModbusSource(session_id)
        async for reading in source.readings():
            if stop_event.is_set():
                return
            payload = reading.model_dump_json().encode("utf-8")
            try:
                await spool_io(
                    spool.enqueue,
                    str(session_id), reading.sequence, "modbus", payload,
                )
            except SpoolFullError as exc:
                LOG.error("%s; stopping acquisition", exc)
                stop_event.set()
                return

    tasks = [
        asyncio.create_task(gateway.serve(stop_event)),
        asyncio.create_task(upstream.run(stop_event)),
    ]
    if settings.source in ("file", "file_and_mock_modbus"):
        tasks.append(asyncio.create_task(waveform_producer()))
    if settings.source in ("mock_modbus", "file_and_mock_modbus"):
        tasks.append(asyncio.create_task(modbus_producer()))
    try:
        await stop_event.wait()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        spool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SandAlert Mac edge service")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run_edge(args.config))


if __name__ == "__main__":
    main()
