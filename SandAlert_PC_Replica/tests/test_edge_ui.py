import csv
import io
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest

pytest.importorskip("PySide6", reason="UI tests require the edge environment")
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.tabs.signal_traces import SignalTracesTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def analysis(sequence=1, session="session-a"):
    return {
        "type": "analysis.result", "session_id": session, "sequence": sequence,
        "ips": 2, "mass_rate": 0.08, "threshold": 0.01, "alarm_level": "caution",
        "tdoa_angle_deg": 60.5, "tdoa_cost": 1e-10, "tdoa_reliable": True,
    }


def modbus():
    return {"type": "modbus.result", "data": {"ips": 120, "mass_rate": 30.0, "relay_status": 3}}


def test_dual_source_display_uses_one_selected_source(app):
    tab = SignalTracesTab()
    try:
        tab.set_edge_mode(True)
        tab.update_from_edge_event(modbus())
        assert "30.00000" in tab.mass_label.text()
        tab.update_from_edge_event(analysis())
        tab.update_from_edge_event(modbus())
        assert "0.08000" in tab.mass_label.text()
        assert tab.alarm_indicator.text() == "CAUTION"
        tab.metric_source_combo.setCurrentIndex(tab.metric_source_combo.findData("modbus"))
        assert "30.00000" in tab.mass_label.text()
        assert tab.alarm_indicator.text() == "CRITICAL ALARM"
        tab.update_from_edge_event(analysis(2))
        assert "30.00000" in tab.mass_label.text()
        tab.update_from_edge_event({"type": "edge.local.ready"})
        assert tab.alarm_indicator.text() == "WAITING"
    finally:
        tab.stop_all_workers()
        tab.close()


def test_tdoa_deduplication_respects_session_sequence_and_toggle(app):
    tab = SignalTracesTab()
    try:
        tab.set_edge_mode(True)
        event = analysis()
        tdoa = {"type": "tdoa.result", "data": {
            "session_id": "session-a", "sequence": 1, "angle_deg": 60.5,
            "cost": 1e-10, "reliable": True,
        }}
        tab.update_from_edge_event(tdoa)  # Both delivery orders are supported.
        tab.update_from_edge_event(event)
        assert tab.tdoa_history == [60.5]
        tab.update_from_edge_event(analysis(2))
        tab.update_from_edge_event(analysis(1, "session-b"))
        assert tab.tdoa_history == [60.5, 60.5, 60.5]
        tab.cb_tdoa.setChecked(False)
        tab.update_from_edge_event(analysis(3))
        assert tab.tdoa_history == []
        tab.cb_tdoa.setChecked(True)
        tab.update_from_edge_event(analysis(4))
        assert tab.tdoa_history == [60.5]
    finally:
        tab.stop_all_workers()
        tab.close()


def test_calibration_is_blocked_for_edge_and_instrument_sources(app):
    window = MainWindow()
    try:
        for mode in ("edge", "modbus", "idle"):
            window.signal_tab.set_source_mode(mode)
            assert not window.param_tab.btn_start_calib.isEnabled()
            assert not window.param_tab.factor_input.isEnabled()
            window.start_calibration()  # Guard against programmatic activation too.
            assert not window.signal_tab.is_calibrating
        window.signal_tab.set_source_mode("waveform")
        assert window.param_tab.btn_start_calib.isEnabled()
        window.start_calibration()
        assert window.signal_tab.is_calibrating
        window.signal_tab.set_source_mode("edge")
        assert not window.signal_tab.is_calibrating
    finally:
        window.close()


def test_calibration_energy_matches_csv_mass_for_overlapping_windows(app, monkeypatch):
    window = MainWindow()
    tab = window.signal_tab
    try:
        tab.set_source_mode("waveform")
        monkeypatch.setattr(tab.dsp, "process_window", lambda *a: (
            np.zeros(3000), 0.1, 2, 6.0 / tab.K_v2 / 3.0, 6.0, [],
        ))
        tab.csv_writer = csv.writer(io.StringIO())
        tab.is_recording = True
        window.start_calibration()
        for _ in range(10):
            tab.update_from_serial(np.zeros(50))
        assert tab.accumulated_energy == pytest.approx(1.0)
        assert tab.total_mass_recorded == pytest.approx(tab.accumulated_energy / tab.K_v2)
        window.param_tab.calib_mass_input.setText("2")
        window.stop_calibration()
        assert tab.K_v2 == pytest.approx(0.5)
    finally:
        window.close()


def test_local_sample_rate_controls_dsp_tdoa_and_is_locked_during_calibration(app):
    window = MainWindow()
    try:
        window.signal_tab.set_source_mode("waveform")
        window.param_tab.sample_rate_input.setText("2000000")
        assert window.signal_tab.dsp.fs == 2_000_000
        assert window.signal_tab.tdoa_worker._engine.fs == 2_000_000
        window.start_calibration()
        assert not window.param_tab.sample_rate_input.isEnabled()
        window.signal_tab.set_source_mode("edge")
        assert not window.param_tab.sample_rate_input.isEnabled()
    finally:
        window.close()
