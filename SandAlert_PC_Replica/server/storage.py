from __future__ import annotations

import json
import threading
from pathlib import Path
from uuid import UUID

import h5py
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from server.models import (
    AcquisitionSession,
    AlarmRecord,
    AnalysisRecord,
    Base,
    ModbusSample,
    WaveformSegment,
)
from shared.contracts import AnalysisResult, ModbusReading, WaveformMetadata


class Database:
    def __init__(self, url: str):
        if url.startswith("sqlite:///"):
            db_path = Path(url.removeprefix("sqlite:///"))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            url,
            future=True,
            connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
        )
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def ensure_session(self, session_id: UUID | str, edge_id: str = "unknown") -> None:
        key = str(session_id)
        with self.sessions.begin() as db:
            existing = db.get(AcquisitionSession, key)
            if existing is None:
                db.add(AcquisitionSession(id=key, edge_id=edge_id))
            elif edge_id != "unknown":
                existing.edge_id = edge_id

    def waveform_exists(self, session_id: UUID | str, sequence: int) -> bool:
        with self.sessions() as db:
            stmt = select(WaveformSegment.id).where(
                WaveformSegment.session_id == str(session_id),
                WaveformSegment.sequence == sequence,
            )
            return db.execute(stmt).scalar_one_or_none() is not None

    def add_waveform(self, meta: WaveformMetadata, archive_path: str) -> None:
        self.ensure_session(meta.session_id)
        with self.sessions.begin() as db:
            db.add(WaveformSegment(
                session_id=str(meta.session_id), sequence=meta.sequence,
                timestamp_ns=meta.timestamp_ns, fs_hz=meta.fs_hz,
                channels=meta.channels, samples=meta.samples, source=meta.source,
                archive_path=archive_path,
            ))

    def modbus_exists(self, session_id: UUID | str, sequence: int) -> bool:
        with self.sessions() as db:
            stmt = select(ModbusSample.id).where(
                ModbusSample.session_id == str(session_id),
                ModbusSample.sequence == sequence,
            )
            return db.execute(stmt).scalar_one_or_none() is not None

    def add_modbus(self, reading: ModbusReading) -> None:
        self.ensure_session(reading.session_id)
        with self.sessions.begin() as db:
            db.add(ModbusSample(
                session_id=str(reading.session_id), sequence=reading.sequence,
                timestamp_ns=reading.timestamp_ns, ips=reading.ips,
                mass_rate=reading.mass_rate, total_mass=reading.total_mass,
                relay_status=reading.relay_status, current_ma=reading.current_ma,
                raw_registers_json=json.dumps(reading.raw_registers),
            ))

    def add_analysis(self, result: AnalysisResult) -> None:
        with self.sessions.begin() as db:
            db.add(AnalysisRecord(
                session_id=str(result.session_id), sequence=result.sequence,
                timestamp_ns=result.timestamp_ns, ips=result.ips,
                mass_rate=result.mass_rate, window_energy=result.window_energy,
                threshold=result.threshold, alarm_level=result.alarm_level,
                cumulative_mass=result.cumulative_mass,
                tdoa_angle_deg=result.tdoa_angle_deg, tdoa_cost=result.tdoa_cost,
            ))
            if result.alarm_level != "normal":
                db.add(AlarmRecord(
                    session_id=str(result.session_id), sequence=result.sequence,
                    timestamp_ns=result.timestamp_ns, level=result.alarm_level,
                    message=f"mass rate {result.mass_rate:.6f} g/s",
                ))

    def analysis_rows(self, session_id: str) -> list[AnalysisRecord]:
        with self.sessions() as db:
            stmt = select(AnalysisRecord).where(
                AnalysisRecord.session_id == session_id
            ).order_by(AnalysisRecord.sequence)
            return list(db.scalars(stmt))


class HDF5Archive:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, meta: WaveformMetadata, data: np.ndarray) -> str:
        session_dir = self.root / str(meta.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        file_path = session_dir / "waveforms.h5"
        dataset_name = f"frames/{meta.sequence:020d}"
        with self._lock, h5py.File(file_path, "a") as archive:
            if dataset_name in archive:
                return f"{file_path}#{dataset_name}"
            dataset = archive.create_dataset(
                dataset_name,
                data=np.asarray(data, dtype=np.float32),
                compression="lzf",
                chunks=True,
            )
            for key, value in meta.model_dump(mode="json").items():
                dataset.attrs[key] = value
            archive.flush()
        return f"{file_path}#{dataset_name}"
