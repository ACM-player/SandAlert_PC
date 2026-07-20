from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import AsyncIterator
from uuid import UUID

import numpy as np

from shared.contracts import ModbusReading, WaveformMetadata


class FileReplaySource:
    def __init__(
        self,
        path: str | Path,
        session_id: UUID,
        fs_hz: int,
        chunk_samples: int = 500,
        speed: float = 1.0,
        loop: bool = False,
    ):
        self.path = Path(path)
        self.session_id = session_id
        self.fs_hz = fs_hz
        self.chunk_samples = chunk_samples
        self.speed = speed
        self.loop = loop

    def _files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        files = sorted(self.path.glob("*.txt")) + sorted(self.path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no TXT/CSV replay files found in {self.path}")
        return files

    async def frames(self) -> AsyncIterator[tuple[WaveformMetadata, np.ndarray]]:
        sequence = 0
        buffer: list[list[float]] = []
        channels: int | None = None
        while True:
            for file_path in self._files():
                with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        parts = stripped.replace(",", " ").split()
                        try:
                            values = [float(value) for value in parts]
                        except ValueError:
                            continue
                        if len(values) not in (1, 4):
                            continue
                        if channels is None:
                            channels = len(values)
                        if len(values) != channels:
                            continue
                        buffer.append(values)
                        if len(buffer) < self.chunk_samples:
                            continue
                        array = np.asarray(buffer, dtype=np.float32).T
                        meta = WaveformMetadata(
                            session_id=self.session_id,
                            sequence=sequence,
                            timestamp_ns=time.time_ns(),
                            fs_hz=self.fs_hz,
                            channels=array.shape[0],
                            samples=array.shape[1],
                            source=f"file:{file_path.name}",
                        )
                        yield meta, array
                        sequence += 1
                        buffer.clear()
                        await asyncio.sleep(self.chunk_samples / self.fs_hz / self.speed)
            if buffer:
                array = np.asarray(buffer, dtype=np.float32).T
                yield WaveformMetadata(
                    session_id=self.session_id,
                    sequence=sequence,
                    timestamp_ns=time.time_ns(),
                    fs_hz=self.fs_hz,
                    channels=array.shape[0],
                    samples=array.shape[1],
                    source="file:tail",
                ), array
                sequence += 1
                buffer.clear()
            if not self.loop:
                break


class MockModbusSource:
    def __init__(self, session_id: UUID, interval_seconds: float = 0.5):
        self.session_id = session_id
        self.interval_seconds = interval_seconds

    async def readings(self) -> AsyncIterator[ModbusReading]:
        sequence = 0
        total_mass = 0.0
        while True:
            ips = random.randint(20, 200)
            mass_rate = random.uniform(1.0, 50.0)
            total_mass += mass_rate * self.interval_seconds
            relay = 0
            if ips > 100 or mass_rate > 20:
                relay |= 0x01
            if ips > 180 or mass_rate > 40:
                relay |= 0x02
            current = 4.0 + min(mass_rate / 50.0, 1.0) * 16.0
            raw = [
                ips, 0, int(round(mass_rate * 100)), 0,
                int(total_mass) >> 16, int(total_mass) & 0xFFFF,
                0, 0, relay, int(round(current * 100)), ips + 20,
            ]
            yield ModbusReading(
                session_id=self.session_id,
                sequence=sequence,
                ips=ips,
                mass_rate=mass_rate,
                total_mass=total_mass,
                relay_status=relay,
                current_ma=current,
                raw_registers=raw,
            )
            sequence += 1
            await asyncio.sleep(self.interval_seconds)
