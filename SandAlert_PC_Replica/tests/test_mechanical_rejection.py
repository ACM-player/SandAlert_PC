import json

import numpy as np
import pytest

from ml_pipeline.dataset import (
    assert_calibration_isolated,
    generate_dataset,
    generate_mechanical_calibration_dataset,
    generate_stress_dataset,
    load_manifest,
    merge_adversarial_training_datasets,
)
from ml_pipeline.mechanical_evaluation import (
    create_mechanical_candidate_lock,
    evaluate_locked_mechanical_candidates,
)
from ml_pipeline.mechanical_rejector import train_mechanical_rejector
from ml_pipeline.simulation import (
    MECHANICAL_CALIBRATION_SCENARIOS,
    SimulationConfig,
    generate_window,
)
from ml_pipeline.training import train_feature_baseline, train_tcn
from server.ml_shadow import ONNXShadowModel
from shared.settings import ServerSettings


def _config():
    return SimulationConfig(
        sample_rate_hz=1_000_000,
        window_samples=1024,
        channels=1,
        resonance_high_hz=300_000.0,
    )


def test_mechanical_artifacts_are_deterministic_non_sand_sources():
    for scenario in MECHANICAL_CALIBRATION_SCENARIOS:
        first = generate_window(
            _config(),
            seed=99,
            scenario=scenario,
            force_event_count=0,
            severity=0.8,
        )
        second = generate_window(
            _config(),
            seed=99,
            scenario=scenario,
            force_event_count=0,
            severity=0.8,
        )
        np.testing.assert_array_equal(first.waveform, second.waveform)
        assert first.labels == second.labels
        assert first.labels["event_count"] == 0
        assert first.labels["mechanical_interference_type"] == scenario
        assert first.labels["mechanical_artifact_rms"] > 0.0


def test_calibration_merge_and_frozen_isolation(tmp_path):
    base = tmp_path / "base"
    calibration = tmp_path / "calibration"
    frozen = tmp_path / "frozen"
    merged = tmp_path / "merged"
    generate_dataset(base, 120, 1000, _config())
    generate_mechanical_calibration_dataset(
        calibration, 120, 3000, _config()
    )
    generate_stress_dataset(frozen, 60, 5000, _config())
    summary = merge_adversarial_training_datasets(
        base, calibration, merged
    )
    assert summary["frozen_ood_used"] is False
    rows = load_manifest(merged)
    hard_negatives = [
        row for row in rows if row["mechanical_reject_target"]
    ]
    mechanical_positives = [
        row
        for row in rows
        if row["mechanical_interference"] and row["event_count"] > 0
    ]
    assert hard_negatives
    assert mechanical_positives
    audit = assert_calibration_isolated(calibration, frozen)
    assert audit["group_overlap"] == 0
    assert audit["seed_overlap"] == 0
    assert audit["calibration_fingerprint"] != audit["frozen_ood_fingerprint"]


def test_routes_lock_and_frozen_evaluation_smoke(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    base = tmp_path / "base"
    calibration = tmp_path / "calibration"
    frozen = tmp_path / "frozen"
    merged = tmp_path / "merged"
    baseline = tmp_path / "baseline"
    route_a = tmp_path / "route-a"
    route_b = tmp_path / "route-b"
    rejector_dir = tmp_path / "rejector"
    generate_dataset(base, 140, 10100, _config())
    generate_mechanical_calibration_dataset(
        calibration, 200, 20300, _config()
    )
    generate_stress_dataset(frozen, 60, 30500, _config())
    merge_adversarial_training_datasets(base, calibration, merged)
    train_feature_baseline(base, baseline)
    train_tcn(
        merged,
        route_a,
        epochs=1,
        batch_size=32,
        threads=2,
        workers=0,
        seed=11,
        quality_head_enabled=True,
    )
    train_tcn(
        merged,
        route_b,
        epochs=1,
        batch_size=32,
        threads=2,
        workers=0,
        seed=12,
        quality_head_enabled=False,
    )
    train_mechanical_rejector(merged, rejector_dir, seed=13)
    route_a_runtime = ONNXShadowModel(
        ServerSettings(
            ml_enabled=True,
            ml_model_path=route_a / "candidate.onnx",
            ml_manifest_path=route_a / "candidate.manifest.json",
        )
    )
    assert route_a_runtime.ready
    aux_count = len(route_a_runtime.aux_feature_names)
    outputs = route_a_runtime.predict_extended(
        np.zeros(1024, dtype=np.float32),
        np.zeros(aux_count, dtype=np.float32),
    )
    assert len(outputs) == 4
    assert outputs[3] is not None

    lock_path = tmp_path / "candidate-lock.json"
    create_mechanical_candidate_lock(
        route_a / "candidate.onnx",
        route_a / "candidate.manifest.json",
        route_b / "candidate.onnx",
        route_b / "candidate.manifest.json",
        rejector_dir / "mechanical_rejector.json",
        calibration,
        lock_path,
    )
    report_path = tmp_path / "mechanical-frozen.json"
    report = evaluate_locked_mechanical_candidates(
        frozen,
        calibration,
        baseline / "feature_baseline.joblib",
        lock_path,
        report_path,
    )
    assert report["decision"]["deployment_scope"] == "shadow_only"
    assert report_path.is_file()
    assert report_path.with_suffix(".md").is_file()
    for route in ("route_a", "route_b"):
        values = report["candidates"][route]["mechanical_interference"]
        assert 0.0 <= values["false_positive_rate"] <= 1.0
        assert 0.0 <= values["false_negative_rate"] <= 1.0
    with pytest.raises(FileExistsError):
        evaluate_locked_mechanical_candidates(
            frozen,
            calibration,
            baseline / "feature_baseline.joblib",
            lock_path,
            report_path,
        )
    json.loads(report_path.read_text(encoding="utf-8"))
