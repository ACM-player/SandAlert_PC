import json

import h5py
import numpy as np

from ml_pipeline.dataset import (
    generate_dataset,
    generate_stress_dataset,
    load_manifest,
)
from ml_pipeline.simulation import (
    OOD_SCENARIOS,
    SCENARIOS,
    SimulationConfig,
    generate_window,
)
from shared.ml_features import AUX_FEATURE_NAMES, extract_shadow_aux_features


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
    assert "impact_to_background_db" in first.labels

    quiet = generate_window(config, 4321, "clean", force_event_count=0)
    assert abs(quiet.labels["relative_energy_db"]) < 1e-6


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

    for group_id in groups:
        members = [row for row in records if row["group_id"] == group_id]
        assert len({row["air_velocity_m_s"] for row in members}) == 1
        assert all("particle_size" not in row for row in members)
        assert all(np.isfinite(row["energy_residual_db"]) for row in members)


def test_stress_dataset_is_ood_only(tmp_path):
    config = SimulationConfig(
        sample_rate_hz=1_000_000,
        window_samples=1024,
        channels=1,
        resonance_high_hz=300_000.0,
    )
    summary = generate_stress_dataset(
        tmp_path, windows=60, seed=77, config=config
    )
    records = load_manifest(tmp_path)
    assert summary["splits"] == {"test": 60}
    assert {row["scenario"] for row in records} == set(OOD_SCENARIOS)
    assert min(row["severity"] for row in records) >= 0.75
    assert all(row["split"] == "test" for row in records)


def test_aux_features_encode_optional_air_velocity():
    signal = np.sin(np.linspace(0, 40, 1024, dtype=np.float32))
    missing = extract_shadow_aux_features(signal, 1_000_000, 3.0, None)
    present = extract_shadow_aux_features(signal, 1_000_000, 3.0, 20.0)
    present_index = AUX_FEATURE_NAMES.index("air_velocity_present")
    velocity_index = AUX_FEATURE_NAMES.index("air_velocity_scaled")
    assert missing.values[present_index] == 0.0
    assert missing.values[velocity_index] == 0.0
    assert present.values[present_index] == 1.0
    assert present.values[velocity_index] == 0.4
