import csv
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="UI test requires the Mac edge environment")
from PySide6.QtWidgets import QApplication, QFileDialog

from ui.tabs.signal_traces import SignalTracesTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def deliver(tab, total_mass):
    tab.update_from_serial({
        "ips": 100, "mass_rate": 10.0, "total_mass": total_mass,
        "relay_status": 1, "current_ma": 8.0,
    })


def read_masses(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [float(row["CumulativeMass(g)"]) for row in csv.DictReader(handle)]


def test_csv_uses_counter_deltas_across_duplicates_and_reset(app, tmp_path, monkeypatch):
    tab = SignalTracesTab()
    path = tmp_path / "modbus.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(path), ""))
    try:
        deliver(tab, 65530)
        tab.toggle_csv_recording()
        for value in (65535, 65540, 65540, 0, 3):
            deliver(tab, value)
        tab.toggle_csv_recording()
        assert read_masses(path) == [5.0, 10.0, 10.0, 10.0, 13.0]
        assert tab.alarm_indicator.text() == "CAUTION"
        assert "10.00000 g/s" in tab.mass_label.text()
    finally:
        if tab.is_recording:
            tab.toggle_csv_recording()
        tab.close()


def test_csv_rebaselines_on_new_recording_and_connection(app, tmp_path, monkeypatch):
    tab = SignalTracesTab()
    paths = iter([tmp_path / "first.csv", tmp_path / "second.csv"])
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(next(paths)), ""))
    try:
        tab.toggle_csv_recording()
        deliver(tab, 100)  # No earlier reading: establish a baseline.
        deliver(tab, 105)
        tab.toggle_csv_recording()
        assert read_masses(tmp_path / "first.csv") == [0.0, 5.0]

        deliver(tab, 120)  # Mass produced while not recording is excluded.
        tab.toggle_csv_recording()
        deliver(tab, 122)
        tab.set_edge_mode(False)  # A newly selected connection has its own counter.
        deliver(tab, 9000)
        deliver(tab, 9004)
        tab.toggle_csv_recording()
        assert read_masses(tmp_path / "second.csv") == [2.0, 2.0, 6.0]
    finally:
        if tab.is_recording:
            tab.toggle_csv_recording()
        tab.close()
