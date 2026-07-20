from __future__ import annotations

import json
import time
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


PROTOCOL_VERSION = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WaveformMetadata(StrictModel):
    type: Literal["waveform"] = "waveform"
    version: int = PROTOCOL_VERSION
    session_id: UUID
    sequence: int = Field(ge=0)
    timestamp_ns: int = Field(default_factory=time.time_ns, ge=0)
    fs_hz: int = Field(gt=0)
    channels: int = Field(ge=1, le=32)
    samples: int = Field(gt=0)
    dtype: Literal["float32"] = "float32"
    byte_order: Literal["little"] = "little"
    source: str = "unknown"


class ModbusReading(StrictModel):
    type: Literal["modbus.reading"] = "modbus.reading"
    version: int = PROTOCOL_VERSION
    session_id: UUID
    sequence: int = Field(ge=0)
    timestamp_ns: int = Field(default_factory=time.time_ns, ge=0)
    ips: int = Field(ge=0)
    mass_rate: float = Field(ge=0)
    total_mass: float = Field(ge=0)
    relay_status: int = Field(ge=0)
    current_ma: float
    raw_registers: list[int] = Field(default_factory=list)


class EdgeHello(StrictModel):
    type: Literal["edge.hello"] = "edge.hello"
    version: int = PROTOCOL_VERSION
    edge_id: str
    session_id: UUID


class Ack(StrictModel):
    type: Literal["ack"] = "ack"
    version: int = PROTOCOL_VERSION
    session_id: UUID
    sequence: int
    kind: Literal["waveform", "modbus"]
    duplicate: bool = False


class AnalysisResult(StrictModel):
    type: Literal["analysis.result"] = "analysis.result"
    version: int = PROTOCOL_VERSION
    session_id: UUID
    sequence: int
    timestamp_ns: int
    ips: int
    mass_rate: float
    window_energy: float
    threshold: float
    alarm_level: Literal["normal", "caution", "critical"]
    cumulative_mass: float
    tdoa_angle_deg: float | None = None
    tdoa_cost: float | None = None
    tdoa_reliable: bool = False


class ServiceEvent(StrictModel):
    type: str
    version: int = PROTOCOL_VERSION
    data: dict[str, Any] = Field(default_factory=dict)


def to_json(model: BaseModel) -> str:
    return model.model_dump_json()


def parse_json_message(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or "type" not in value:
        raise ValueError("message must be a JSON object containing type")
    return value
