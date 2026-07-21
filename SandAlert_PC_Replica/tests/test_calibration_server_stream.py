import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    main_window = MainWindow()
    yield main_window
    main_window.close()
    app.processEvents()


def test_server_analysis_stream_updates_calibration_and_factor(window):
    window.param_tab.calib_mass_input.setText("2")
    window.start_calibration()

    window.signal_tab.update_from_edge_event({
        "type": "analysis.result",
        "impact_rate": 10,
        "mass_rate": 0.25,
        "window_energy": 4.0,
        "threshold": 0.008,
        "alarm_level": "normal",
    })
    window.signal_tab.update_from_edge_event({
        "type": "analysis.result",
        "ips": 5,
        "mass_rate": 0.15,
        "window_energy": 2.0,
        "threshold": 0.008,
        "alarm_level": "normal",
    })

    assert window.signal_tab.accumulated_impacts == 15
    assert window.signal_tab.accumulated_energy == 6
    assert "Total Impacts: 15" in window.param_tab.lbl_calib_status.text()

    window.stop_calibration()

    assert float(window.param_tab.factor_input.text()) == 3.0
    assert "Status: Success!" in window.param_tab.lbl_calib_status.text()


def test_impact_only_server_stream_uses_impact_fallback(window):
    window.param_tab.calib_mass_input.setText("10")
    window.start_calibration()

    window.signal_tab.update_from_edge_event({
        "type": "analysis.result",
        "impact_rate": 20,
        "mass_rate": 0.0,
        "threshold": 0.008,
        "alarm_level": "normal",
    })
    window.stop_calibration()

    assert float(window.param_tab.factor_input.text()) == 2.0
    assert "from Impacts" in window.param_tab.lbl_calib_status.text()
