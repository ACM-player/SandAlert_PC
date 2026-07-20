from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AcquisitionSession(Base):
    __tablename__ = "acquisition_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    edge_id: Mapped[str] = mapped_column(String(128), default="unknown")
    started_at: Mapped[datetime] = mapped_column(default=utc_now)
    last_seen_at: Mapped[datetime] = mapped_column(default=utc_now)


class WaveformSegment(Base):
    __tablename__ = "waveform_segments"
    __table_args__ = (UniqueConstraint("session_id", "sequence"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("acquisition_sessions.id"))
    sequence: Mapped[int] = mapped_column(BigInteger)
    timestamp_ns: Mapped[int] = mapped_column(BigInteger)
    fs_hz: Mapped[int] = mapped_column(Integer)
    channels: Mapped[int] = mapped_column(Integer)
    samples: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(128))
    archive_path: Mapped[str] = mapped_column(Text)


class ModbusSample(Base):
    __tablename__ = "modbus_samples"
    __table_args__ = (UniqueConstraint("session_id", "sequence"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("acquisition_sessions.id"))
    sequence: Mapped[int] = mapped_column(BigInteger)
    timestamp_ns: Mapped[int] = mapped_column(BigInteger)
    ips: Mapped[int] = mapped_column(Integer)
    mass_rate: Mapped[float] = mapped_column(Float)
    total_mass: Mapped[float] = mapped_column(Float)
    relay_status: Mapped[int] = mapped_column(Integer)
    current_ma: Mapped[float] = mapped_column(Float)
    raw_registers_json: Mapped[str] = mapped_column(Text, default="[]")


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"
    __table_args__ = (UniqueConstraint("session_id", "sequence"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("acquisition_sessions.id"))
    sequence: Mapped[int] = mapped_column(BigInteger)
    timestamp_ns: Mapped[int] = mapped_column(BigInteger)
    ips: Mapped[int] = mapped_column(Integer)
    mass_rate: Mapped[float] = mapped_column(Float)
    window_energy: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    alarm_level: Mapped[str] = mapped_column(String(16))
    cumulative_mass: Mapped[float] = mapped_column(Float)
    tdoa_angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    tdoa_cost: Mapped[float | None] = mapped_column(Float, nullable=True)


class AlarmRecord(Base):
    __tablename__ = "alarm_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("acquisition_sessions.id"))
    sequence: Mapped[int] = mapped_column(BigInteger)
    timestamp_ns: Mapped[int] = mapped_column(BigInteger)
    level: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
