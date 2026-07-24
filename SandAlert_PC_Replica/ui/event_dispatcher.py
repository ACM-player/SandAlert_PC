from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot


class UiUpdateDispatcher(QObject):
    """Coalesce high-rate Edge data and update widgets only from the UI thread."""

    waveform_ready = Signal(object)
    event_ready = Signal(object)
    stats_ready = Signal(object)

    IMMEDIATE_TYPES = {"edge.connection", "service.error", "ml.status"}
    EVENT_PRIORITY = {
        "analysis.result": 0,
        "ml.result": 1,
        "tdoa.result": 2,
        "modbus.result": 3,
    }

    def __init__(self, refresh_interval_ms: int = 75, parent=None):
        super().__init__(parent)
        self.refresh_interval_ms = int(refresh_interval_ms)
        self._latest_waveform: np.ndarray | None = None
        self._pending_events: dict[tuple[str, str], dict] = {}
        self._rendered_sequences: dict[tuple[str, str], int] = {}
        self._coalesced_waveforms = 0
        self._coalesced_events = 0
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(self.refresh_interval_ms)
        self.timer.timeout.connect(self.flush)
        self.timer.start()

    @staticmethod
    def _event_session(event: Mapping) -> str:
        if event.get("session_id") is not None:
            return str(event["session_id"])
        data = event.get("data")
        if isinstance(data, Mapping) and data.get("session_id") is not None:
            return str(data["session_id"])
        return ""

    @staticmethod
    def _event_sequence(event: Mapping) -> int | None:
        value = event.get("sequence")
        if value is None and isinstance(event.get("data"), Mapping):
            value = event["data"].get("sequence")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @Slot(object)
    def enqueue_waveform(self, waveform) -> None:
        array = np.asarray(waveform)
        if array.size == 0:
            return
        snapshot = np.array(array, dtype=np.float32, copy=True, order="C")
        if self._latest_waveform is not None:
            self._coalesced_waveforms += 1
        self._latest_waveform = snapshot

    @Slot(object)
    def enqueue_event(self, event) -> None:
        if not isinstance(event, Mapping):
            return
        snapshot = dict(event)
        event_type = str(snapshot.get("type", ""))
        if event_type in self.IMMEDIATE_TYPES:
            self.event_ready.emit(snapshot)
            return

        key = (event_type, self._event_session(snapshot))
        sequence = self._event_sequence(snapshot)
        rendered = self._rendered_sequences.get(key)
        if sequence is not None and rendered is not None and sequence <= rendered:
            return
        pending = self._pending_events.get(key)
        if pending is not None:
            pending_sequence = self._event_sequence(pending)
            if (
                sequence is not None
                and pending_sequence is not None
                and sequence <= pending_sequence
            ):
                return
            self._coalesced_events += 1
        self._pending_events[key] = snapshot

    @Slot()
    def flush(self) -> None:
        waveform = self._latest_waveform
        self._latest_waveform = None
        pending = list(self._pending_events.items())
        self._pending_events.clear()

        if waveform is not None:
            self.waveform_ready.emit(waveform)

        pending.sort(
            key=lambda item: self.EVENT_PRIORITY.get(item[0][0], 99)
        )
        for key, event in pending:
            sequence = self._event_sequence(event)
            if sequence is not None:
                self._rendered_sequences[key] = sequence
            self.event_ready.emit(event)

        self.stats_ready.emit({
            "refresh_interval_ms": self.refresh_interval_ms,
            "coalesced_waveforms": self._coalesced_waveforms,
            "coalesced_events": self._coalesced_events,
        })

    @Slot()
    def reset(self) -> None:
        self._latest_waveform = None
        self._pending_events.clear()
        self._rendered_sequences.clear()
        self._coalesced_waveforms = 0
        self._coalesced_events = 0

    def stop(self) -> None:
        self.timer.stop()
