from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 on the lab server
    import tomli as tomllib


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    token: str = "change-me"
    database_url: str = "sqlite:///./runtime/sandalert.sqlite3"
    waveform_dir: Path = Path("./runtime/waveforms")
    threshold_margin: float = 0.008
    calibration_factor: float = 1.2
    caution_limit: float = 0.05
    alarm_limit: float = 0.10
    dsp_sample_rate: int = 1000
    tdoa_enabled: bool = True
    tdoa_bandpass_enabled: bool = False
    tdoa_band_low_hz: float = Field(default=100_000.0, gt=0)
    tdoa_band_high_hz: float = Field(default=300_000.0, gt=0)
    ml_enabled: bool = False
    ml_model_path: Path | None = None
    ml_manifest_path: Path | None = None
    ml_max_emit_hz: float = Field(default=20.0, gt=0.0, le=100.0)
    ml_background_seconds: float = Field(default=1.0, gt=0.0, le=60.0)
    ml_background_min_frames: int = Field(default=32, ge=4, le=100_000)


class EdgeSettings(BaseModel):
    edge_id: str = "mac-edge-01"
    server_url: str = "ws://127.0.0.1:8765/v1/ingest"
    token: str = "change-me"
    local_host: str = "127.0.0.1"
    local_port: int = Field(default=8766, ge=1, le=65535)
    spool_dir: Path = Path("./runtime/edge-spool")
    max_spool_gib: float = Field(default=20.0, gt=0)
    source: Literal["file", "mock_modbus", "file_and_mock_modbus"] = (
        "file_and_mock_modbus"
    )
    replay_path: Path = Path("./sample_data")
    replay_fs_hz: int = 2_000_000
    replay_chunk_samples: int = 500
    replay_speed: float = Field(default=0.001, gt=0)
    replay_start_delay_seconds: float = Field(default=2.0, ge=0)
    replay_loop: bool = True
    reconnect_max_seconds: float = 30.0


class UISettings(BaseModel):
    development_tab_enabled: bool = True
    refresh_interval_ms: int = Field(default=75, ge=50, le=1000)


def load_server_settings(path: str | Path | None) -> ServerSettings:
    return _load(path, "server", ServerSettings)


def load_edge_settings(path: str | Path | None) -> EdgeSettings:
    return _load(path, "edge", EdgeSettings)


def load_ui_settings(path: str | Path | None) -> UISettings:
    return _load(path, "ui", UISettings)


def _load(path, section, model_type):
    if path is None:
        return model_type()
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    values = payload.get(section, payload)
    return model_type.model_validate(values)
