import asyncio
import json
import threading
from uuid import uuid4

import pytest

from edge.spool import DurableSpool, spool_io
from edge.upstream import UpstreamClient
from shared.settings import EdgeSettings


def test_cancelling_disk_io_waits_until_the_thread_finishes():
    async def exercise():
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def disk_operation():
            started.set()
            release.wait(timeout=2)
            finished.set()

        task = asyncio.create_task(spool_io(disk_operation))
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(exercise())


class Socket:
    def __init__(self, close_immediately=False):
        self.close_immediately = close_immediately

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def send(self, _):
        pass

    async def __aiter__(self):
        if not self.close_immediately:
            await asyncio.Event().wait()
        if False:
            yield None


def test_cancelled_connection_does_not_leave_background_spool_tasks(tmp_path, monkeypatch):
    async def exercise():
        spool = DurableSpool(tmp_path)
        monkeypatch.setattr("edge.upstream.connect", lambda *a, **k: Socket())
        online = asyncio.Event()

        class Gateway:
            async def broadcast(self, _):
                online.set()

        upstream = UpstreamClient(EdgeSettings(), uuid4(), spool, Gateway())
        task = asyncio.create_task(upstream.run(asyncio.Event()))
        await asyncio.wait_for(online.wait(), 1)
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        spool.close()
        await asyncio.sleep(0.01)
        assert not [
            pending for pending in asyncio.all_tasks()
            if pending is not asyncio.current_task()
        ]

    asyncio.run(exercise())


def test_normal_server_close_is_detected_even_with_an_empty_spool(tmp_path, monkeypatch):
    async def exercise():
        spool = DurableSpool(tmp_path)
        stop = asyncio.Event()
        events = []
        monkeypatch.setattr("edge.upstream.connect", lambda *a, **k: Socket(True))

        class Gateway:
            async def broadcast(self, message):
                event = json.loads(message)
                events.append(event["data"]["status"])
                if events[-1] == "offline":
                    stop.set()

        try:
            upstream = UpstreamClient(EdgeSettings(), uuid4(), spool, Gateway())
            await asyncio.wait_for(upstream.run(stop), 1)
            assert events == ["online", "offline"]
        finally:
            spool.close()

    asyncio.run(exercise())
