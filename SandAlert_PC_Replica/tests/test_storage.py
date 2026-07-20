from uuid import uuid4

import h5py
import numpy as np

from server.storage import Database, HDF5Archive
from shared.contracts import AnalysisResult, WaveformMetadata


def test_hdf5_and_sqlite_store_waveform_and_result(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    database.create_schema()
    archive = HDF5Archive(tmp_path / "waveforms")
    session_id = uuid4()
    meta = WaveformMetadata(
        session_id=session_id, sequence=1, fs_hz=2_000_000,
        channels=4, samples=16, source="test",
    )
    data = np.ones((4, 16), dtype=np.float32)
    path = archive.write(meta, data)
    database.add_waveform(meta, path)
    assert database.waveform_exists(session_id, 1)

    file_path, dataset_name = path.split("#", 1)
    with h5py.File(file_path, "r") as handle:
        np.testing.assert_array_equal(handle[dataset_name][:], data)

    result = AnalysisResult(
        session_id=session_id, sequence=1, timestamp_ns=meta.timestamp_ns,
        ips=2, mass_rate=0.02, window_energy=0.024, threshold=0.008,
        alarm_level="normal", cumulative_mass=0.001,
    )
    database.add_analysis(result)
    rows = database.analysis_rows(str(session_id))
    assert len(rows) == 1
    assert rows[0].mass_rate == 0.02
