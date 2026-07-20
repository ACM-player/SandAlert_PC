from __future__ import annotations

import argparse
import asyncio
import json
import time

from websockets.asyncio.client import connect

from shared.waveform_codec import decode_waveform


async def probe(url: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    waveform_count = 0
    event_counts: dict[str, int] = {}
    angles: list[float] = []
    async with connect(url, max_size=None) as websocket:
        while time.monotonic() < deadline:
            timeout = max(0.05, deadline - time.monotonic())
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            if isinstance(raw, bytes):
                _, data = decode_waveform(raw)
                waveform_count += 1
                if data.ndim != 2:
                    raise RuntimeError("local waveform is not two-dimensional")
                continue
            payload = json.loads(raw)
            event_type = payload.get("type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            angle = payload.get("tdoa_angle_deg")
            if event_type == "tdoa.result":
                angle = payload.get("data", {}).get("angle_deg")
            if angle is not None:
                angles.append(float(angle))
    print(json.dumps({
        "waveforms": waveform_count,
        "events": event_counts,
        "tdoa_angles": angles,
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the SandAlert local Edge gateway")
    parser.add_argument("--url", default="ws://127.0.0.1:8766")
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()
    asyncio.run(probe(args.url, args.seconds))


if __name__ == "__main__":
    main()
