import os
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ui.tabs.signal_traces import SignalTracesTab


def test_rate_labels_and_waveform_curves_are_stable_and_optimized():
    app = QApplication.instance() or QApplication([])
    tab = SignalTracesTab()
    try:
        tab.resize(1600, 900)
        tab.show()
        app.processEvents()

        for label in (tab.impact_label, tab.mass_label):
            assert label.minimumWidth() == tab.RATE_LABEL_WIDTH
            assert label.maximumWidth() == tab.RATE_LABEL_WIDTH
            assert label.alignment() & Qt.AlignLeft
            assert label.alignment() & Qt.AlignVCenter

        tab.impact_label.setText("Impact rate = 999999 /s  [Server]")
        tab.mass_label.setText("Mass rate = 999999.99999 g/s  [Server]")
        app.processEvents()
        fluid_flow_x = tab.fluid_flow_input.geometry().x()
        tab.impact_label.setText("Impact rate = 999999 /s  [Server Modbus]")
        tab.mass_label.setText("Mass rate = 999999.99999 g/s  [Server Modbus]")
        app.processEvents()
        assert tab.fluid_flow_input.geometry().x() == fluid_flow_x

        for curve in (tab.raw_curve, tab.avg_curve, tab.th_curve):
            assert curve.opts["antialias"] is False
            assert curve.opts["autoDownsample"] is True
            assert curve.opts["downsampleMethod"] == "peak"
            assert curve.opts["clipToView"] is True

        view_box = tab.plot_widget.getViewBox()
        assert view_box.state["autoRange"] == [False, False]
        assert tab.plot_widget.viewRange()[1] == pytest.approx([-3.5, 2.5])
        assert tab.ui_refresh_timer.interval() == 33
        assert tab.ui_refresh_timer.isActive()
    finally:
        tab.stop_all_workers()
        tab.close()
        app.processEvents()


def test_edge_callbacks_only_cache_and_timer_performs_set_data():
    app = QApplication.instance() or QApplication([])
    tab = SignalTracesTab()
    tab.ui_refresh_timer.stop()
    try:
        raw_set_data = Mock(wraps=tab.raw_curve.setData)
        threshold_set_data = Mock(wraps=tab.th_curve.setData)
        tab.raw_curve.setData = raw_set_data
        tab.th_curve.setData = threshold_set_data

        base = np.arange(80, dtype=np.float64).reshape(4, 20)
        for sequence in range(100):
            source = base + sequence
            tab.update_edge_waveform(source)
            tab.update_from_edge_event({
                "type": "analysis.result",
                "ips": sequence,
                "mass_rate": 0.25,
                "threshold": 0.01,
                "alarm_level": "normal",
            })

        assert raw_set_data.call_count == 0
        assert threshold_set_data.call_count == 0
        assert isinstance(tab._latest_waveform, np.ndarray)
        assert tab._latest_waveform.ndim == 1
        assert tab._latest_waveform.dtype == np.float32
        assert tab._latest_waveform.flags.c_contiguous
        np.testing.assert_array_equal(tab._latest_waveform, (base + 99)[0])

        tab._refresh_edge_ui()

        assert raw_set_data.call_count == 1
        assert threshold_set_data.call_count == 1
        assert tab._latest_waveform is None
        np.testing.assert_array_equal(tab.voltage_data[-20:], (base + 99)[0])
    finally:
        tab.stop_all_workers()
        tab.close()
        app.processEvents()
