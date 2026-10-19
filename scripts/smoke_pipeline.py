#!/usr/bin/env python3
"""Verify Edge -> compute server -> local gateway using disposable sample data.

Default: start an isolated local compute server. To test an SSH-forwarded server,
pass --server-url and set SANDALERT_SMOKE_TOKEN to its temporary test token.
"""

import argparse
import asyncio
import csv
import io
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

import numpy as np
import uvicorn
from websockets.asyncio.client import connect

APP_ROOT = Path(__file__).resolve().parents[1] / "SandAlert_PC_Replica"
sys.path.insert(0, str(APP_ROOT))
from core.tdoa_engine import TDOAEngine
from edge.main import run_edge
from server.main import create_app
from shared.settings import ServerSettings


def pending_count(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        return db.execute("SELECT COUNT(*) FROM spool_items").fetchone()[0]


def export_rows(server_url, token, session_id):
    parsed = urlsplit(server_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    url = f"{scheme}://{parsed.netloc}/v1/sessions/{session_id}/export.csv?{urlencode({'token': token})}"
    with urlopen(url, timeout=5) as response:
        return list(csv.DictReader(io.StringIO(response.read().decode())))


async def smoke(server_url=None, seconds=10):
    compute_server = None
    compute_task = None
    edge_task = None
    with tempfile.TemporaryDirectory(prefix="sandalert-smoke-") as directory:
        root = Path(directory)
        token = os.environ.get("SANDALERT_SMOKE_TOKEN", "local-disposable-smoke-token")
        if server_url and not os.environ.get("SANDALERT_SMOKE_TOKEN"):
            raise ValueError("set SANDALERT_SMOKE_TOKEN for the isolated external test server")
        try:
            if server_url is None:
                settings = ServerSettings(
                    token=token, database_url=f"sqlite:///{root / 'compute.sqlite3'}",
                    waveform_dir=root / "waveforms",
                )
                compute_server = uvicorn.Server(uvicorn.Config(
                    create_app(settings), host="127.0.0.1", port=0, log_level="warning",
                ))
                compute_task = asyncio.create_task(compute_server.serve())
                async def wait_server():
                    while not compute_server.started:
                        if compute_task.done():
                            await compute_task
                            raise RuntimeError("compute server exited before startup")
                        await asyncio.sleep(0.02)
                await asyncio.wait_for(wait_server(), timeout=5)
                port = compute_server.servers[0].sockets[0].getsockname()[1]
                server_url = f"ws://127.0.0.1:{port}/v1/ingest"

            # A new gateway and empty spool are used for every invocation.
            with socket.socket() as port_probe:
                port_probe.bind(("127.0.0.1", 0))
                gateway_port = port_probe.getsockname()[1]
            np.random.seed(42)
            waveform, _ = TDOAEngine().generate_mock_array_data(
                n_samples=6000, true_angle_deg=60, snr_db=10,
            )
            replay_path = root / "tdoa_60deg.txt"
            np.savetxt(replay_path, waveform.T, delimiter=",")
            spool_dir = root / "spool"
            values = {
                "server_url": server_url, "token": token,
                "local_port": gateway_port, "spool_dir": str(spool_dir),
                "source": "file_and_mock_modbus", "replay_path": str(replay_path),
                "replay_fs_hz": 2_000_000, "replay_chunk_samples": 500,
                "replay_speed": 0.02, "replay_loop": False,
                "replay_start_delay_seconds": 0.5,
            }
            config_path = root / "edge.toml"
            config_path.write_text("[edge]\n" + "\n".join(
                f"{key} = {json.dumps(value)}" for key, value in values.items()
            ), encoding="utf-8")
            edge_task = asyncio.create_task(run_edge(config_path))
            deadline = asyncio.get_running_loop().time() + seconds
            url = f"ws://127.0.0.1:{gateway_port}"
            while True:
                try:
                    websocket = await connect(url, max_size=None)
                    break
                except OSError:
                    if edge_task.done():
                        await edge_task
                        raise RuntimeError("edge exited before gateway startup")
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("gateway did not start")
                    await asyncio.sleep(0.05)
            results = {}
            angles = []
            modbus_count = 0
            waveforms = 0
            async with websocket:
                while asyncio.get_running_loop().time() < deadline:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=max(
                        0.1, deadline - asyncio.get_running_loop().time(),
                    ))
                    if isinstance(raw, bytes):
                        waveforms += 1
                        continue
                    event = json.loads(raw)
                    if event["type"] == "analysis.result":
                        results[event["sequence"]] = event
                    elif event["type"] == "tdoa.result" and event["data"]["reliable"]:
                        angles.append(event["data"]["angle_deg"])
                    elif event["type"] == "modbus.result":
                        modbus_count += 1
                    if len(results) == 12 and angles and modbus_count:
                        if await asyncio.to_thread(pending_count, spool_dir / "spool.sqlite3") == 0:
                            break
            assert len(results) == 12, f"expected 12 waveform analyses, got {len(results)}"
            assert waveforms > 0 and modbus_count > 0
            assert angles and abs((angles[-1] - 60 + 180) % 360 - 180) < 5, angles
            ordered = [results[index] for index in range(12)]
            integral = sum(result["mass_rate"] * 500 / 2_000_000 for result in ordered)
            assert math.isclose(integral, ordered[-1]["cumulative_mass"], rel_tol=1e-9)
            rows = await asyncio.to_thread(export_rows, server_url, token, ordered[0]["session_id"])
            assert len(rows) == 12
            assert math.isclose(float(rows[-1]["CumulativeMass(g)"]), integral, rel_tol=1e-9)
            print(json.dumps({
                "waveforms": waveforms, "analysis_results": len(results),
                "modbus_results": modbus_count, "tdoa_angles": angles,
                "csv_rows": len(rows), "pending_spool": 0,
                "mass_integral_matches": True,
            }, ensure_ascii=False))
        finally:
            if edge_task is not None:
                edge_task.cancel()
                await asyncio.gather(edge_task, return_exceptions=True)
            if compute_server is not None:
                compute_server.should_exit = True
                await compute_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url")
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    asyncio.run(smoke(args.server_url, args.seconds))
