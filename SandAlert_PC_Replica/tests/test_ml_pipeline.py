import json

import h5py
import numpy as np

from ml_pipeline.dataset import generate_dataset, load_manifest
from ml_pipeline.simulation import SCENARIOS, SimulationConfig, generate_window


def test_simulation_is_deterministic_and_labeled():
    config = SimulationConfig(
        sample_rate_hz=1_000_000, window_samples=2048, channels=1
    )
    first = generate_window(config, 1234, "multipath", force_event_count=2)
    second = generate_window(config, 1234, "multipath", force_event_count=2)

    np.testing.assert_array_equal(first.waveform, second.waveform)
    assert first.labels == second.labels
    assert first.waveform.shape == (1, 2048)
    assert first.labels["event_count"] == 2
    assert np.isfinite(first.labels["relative_energy_db"])


def test_dataset_groups_never_cross_splits(tmp_path):
    config = SimulationConfig(
        sample_rate_hz=1_000_000, window_samples=1024, channels=1
    )
    summary = generate_dataset(
        tmp_path, windows=90, seed=55, config=config, scenarios=SCENARIOS
    )
    records = load_manifest(tmp_path)
    groups = {}
    for row in records:
        groups.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert set(summary["splits"]) == {"train", "validation", "test"}
    assert sum(summary["splits"].values()) == 90
    with h5py.File(tmp_path / "waveforms.h5", "r") as handle:
        assert handle["waveforms"].shape == (90, 1, 1024)
        json.loads(handle.attrs["simulation_config"])
