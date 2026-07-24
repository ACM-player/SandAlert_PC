import os
from uuid import uuid4

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from shared.settings import UISettings
from ui.event_dispatcher import UiUpdateDispatcher
from ui.tabs.parameters import ParametersTab


def _app():
    return QApplication.instance() or QApplication([])


def test_dispatcher_coalesces_and_rejects_stale_sequences():
    _app()
    dispatcher = UiUpdateDispatcher(refresh_interval_ms=75)
    assert dispatcher.timer.interval() == 75
    waveforms = []
    events = []
    stats = []
    dispatcher.waveform_ready.connect(waveforms.append)
    dispatcher.event_ready.connect(events.append)
    dispatcher.stats_ready.connect(stats.append)
    session_id = str(uuid4())

    dispatcher.enqueue_waveform(np.zeros((1, 8), dtype=np.float32))
    dispatcher.enqueue_waveform(np.ones((1, 8), dtype=np.float32))
    dispatcher.enqueue_event({
        "type": "analysis.result", "session_id": session_id, "sequence": 2
    })
    dispatcher.enqueue_event({
        "type": "analysis.result", "session_id": session_id, "sequence": 1
    })
    dispatcher.flush()

    assert len(waveforms) == 1
    np.testing.assert_array_equal(waveforms[0], np.ones((1, 8)))
    assert [event["sequence"] for event in events] == [2]
    assert stats[-1]["refresh_interval_ms"] == 75
    assert stats[-1]["coalesced_waveforms"] == 1

    dispatcher.enqueue_event({
        "type": "analysis.result", "session_id": session_id, "sequence": 1
    })
    dispatcher.flush()
    assert [event["sequence"] for event in events] == [2]
    dispatcher.stop()


def test_parameters_development_tab_visibility():
    _app()
    visible = ParametersTab(development_enabled=True)
    hidden = ParametersTab(development_enabled=False)
    assert visible.development_tab is not None
    assert visible.inner_tabs.indexOf(visible.development_tab) >= 0
    assert hidden.development_tab is None
    assert UISettings().refresh_interval_ms == 75
