from edge.spool import DurableSpool


def test_spool_persists_and_acknowledges(tmp_path):
    spool = DurableSpool(tmp_path, max_gib=0.01)
    spool.enqueue("session", 3, "waveform", b"payload")
    item = spool.first_pending()
    assert item is not None
    assert item.sequence == 3
    assert spool.read(item) == b"payload"
    spool.acknowledge("session", 3, "waveform")
    assert spool.first_pending() is None
    spool.close()
