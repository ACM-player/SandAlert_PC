from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget


class DevelopmentTab(QWidget):
    """Read-only diagnostics for research builds."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._analysis: dict = {}
        self._ml: dict = {}
        self._labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        live_group = QGroupBox("DSP / ML Shadow Comparison")
        live_form = QFormLayout(live_group)
        fields = [
            ("source", "Data source"),
            ("session", "Session / sequence"),
            ("sample_rate", "Sample rate"),
            ("model", "Model"),
            ("ml_status", "ML status"),
            ("dsp_ips", "DSP IPS"),
            ("ml_ips", "ML IPS"),
            ("ips_delta", "IPS difference"),
            ("window_energy", "DSP peak energy"),
            ("background_energy", "Background energy"),
            ("relative_energy", "Relative energy"),
            ("ml_relative_energy", "ML relative energy"),
            ("probability", "Impact probability"),
            ("latency", "Inference latency"),
            ("quality", "Quality flags"),
        ]
        for key, title in fields:
            label = QLabel("—")
            label.setTextInteractionFlags(label.textInteractionFlags())
            self._labels[key] = label
            live_form.addRow(f"{title}:", label)
        layout.addWidget(live_group)

        ui_group = QGroupBox("UI Coalescing")
        ui_form = QFormLayout(ui_group)
        for key, title in [
            ("refresh", "Refresh interval"),
            ("waveform_drop", "Coalesced waveforms"),
            ("event_drop", "Coalesced events"),
        ]:
            label = QLabel("0")
            self._labels[key] = label
            ui_form.addRow(f"{title}:", label)
        layout.addWidget(ui_group)
        layout.addStretch()

    @staticmethod
    def _number(value, digits=3, suffix="") -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}f}{suffix}"
        except (TypeError, ValueError):
            return str(value)

    def _refresh_comparison(self) -> None:
        analysis = self._analysis
        ml = self._ml
        session_id = ml.get("session_id") or analysis.get("session_id") or "—"
        sequence = ml.get("sequence")
        if sequence is None:
            sequence = analysis.get("sequence", "—")
        self._labels["session"].setText(f"{session_id} / {sequence}")
        self._labels["source"].setText(
            str(ml.get("source") or analysis.get("source") or "Edge/Server")
        )
        self._labels["sample_rate"].setText(
            self._number(ml.get("sample_rate_hz"), 0, " Hz")
        )
        self._labels["model"].setText(str(ml.get("model_id") or "—"))
        self._labels["ml_status"].setText(str(ml.get("status") or "—"))
        self._labels["dsp_ips"].setText(str(analysis.get("ips", "—")))
        self._labels["ml_ips"].setText(self._number(ml.get("ips_estimate")))
        dsp_ips = analysis.get("ips")
        ml_ips = ml.get("ips_estimate")
        delta = None if dsp_ips is None or ml_ips is None else float(ml_ips) - float(dsp_ips)
        self._labels["ips_delta"].setText(self._number(delta))
        self._labels["window_energy"].setText(
            self._number(analysis.get("window_energy"), 6)
        )
        self._labels["background_energy"].setText(
            self._number(ml.get("background_energy"), 6)
        )
        self._labels["relative_energy"].setText(
            self._number(ml.get("relative_energy_db"), 3, " dB")
        )
        self._labels["ml_relative_energy"].setText(
            self._number(ml.get("ml_relative_energy_db"), 3, " dB")
        )
        self._labels["probability"].setText(
            self._number(ml.get("event_probability"), 3)
        )
        self._labels["latency"].setText(
            self._number(ml.get("inference_latency_ms"), 2, " ms")
        )
        flags = ml.get("quality_flags") or []
        self._labels["quality"].setText(", ".join(map(str, flags)) or "OK")

    @Slot(object)
    def update_from_edge_event(self, event) -> None:
        event_type = event.get("type", "") if isinstance(event, dict) else ""
        if event_type == "analysis.result":
            self._analysis = dict(event)
        elif event_type == "ml.result":
            self._ml = dict(event)
        elif event_type == "ml.status":
            self._ml.update({
                "status": event.get("status"),
                "model_id": event.get("model_id"),
                "quality_flags": [event.get("reason")] if event.get("reason") else [],
            })
        else:
            return
        self._refresh_comparison()

    @Slot(object)
    def update_dispatch_stats(self, stats) -> None:
        self._labels["refresh"].setText(
            f"{int(stats.get('refresh_interval_ms', 0))} ms"
        )
        self._labels["waveform_drop"].setText(
            str(stats.get("coalesced_waveforms", 0))
        )
        self._labels["event_drop"].setText(
            str(stats.get("coalesced_events", 0))
        )
