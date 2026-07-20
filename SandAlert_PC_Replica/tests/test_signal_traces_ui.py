import os

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
    finally:
        tab.stop_all_workers()
        tab.close()
        app.processEvents()
