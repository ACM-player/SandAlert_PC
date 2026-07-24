import json

import pytest

from ml_pipeline.dataset import generate_dataset, generate_stress_dataset
from ml_pipeline.evaluation import evaluate_stress
from ml_pipeline.simulation import SimulationConfig
from ml_pipeline.training import train_feature_baseline, train_tcn


def _dataset(path):
    generate_dataset(
        path,
        windows=120,
        seed=2026,
        config=SimulationConfig(
            sample_rate_hz=1_000_000,
            window_samples=1024,
            channels=1,
            resonance_high_hz=300_000.0,
        ),
    )


def test_feature_baseline_smoke(tmp_path):
    pytest.importorskip("sklearn")
    dataset = tmp_path / "dataset"
    output = tmp_path / "baseline"
    _dataset(dataset)
    report = train_feature_baseline(dataset, output)
    assert report["splits"]["test"]["samples"] > 0
    assert (output / "feature_baseline.joblib").is_file()
    json.loads((output / "feature_baseline_report.json").read_text())


def test_tcn_cpu_and_onnx_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    dataset = tmp_path / "dataset"
    output = tmp_path / "tcn"
    _dataset(dataset)
    report = train_tcn(
        dataset,
        output,
        epochs=1,
        batch_size=16,
        device="cpu",
        threads=2,
        workers=0,
        seed=7,
    )
    assert report["artifacts"]["onnx"] is not None
    assert (output / "candidate.onnx").is_file()
    manifest = json.loads((output / "candidate.manifest.json").read_text())
    assert manifest["device"] == "cpu"
    assert manifest["onnx_sha256"]
    assert manifest["promotion"]["scope"] == "synthetic_holdout_only"
    assert manifest["energy_mode"] == "analytic_residual"
    assert manifest["air_velocity_optional"] is True
    assert manifest["particle_size_input"] is False
    assert manifest["aux_feature_names"]
    assert manifest["event_decision_threshold"] >= 0.5

    from server.ml_shadow import ONNXShadowModel
    from shared.settings import ServerSettings

    runtime = ONNXShadowModel(ServerSettings(
        ml_enabled=True,
        ml_model_path=output / "candidate.onnx",
        ml_manifest_path=output / "candidate.manifest.json",
    ))
    assert runtime.ready
    probability, ips, energy = runtime.predict(
        __import__("numpy").zeros(1024, dtype="float32")
    )
    assert 0.0 <= probability <= 1.0
    assert ips >= 0.0
    assert isinstance(energy, float)


def test_ood_stress_evaluation_smoke(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    dataset = tmp_path / "dataset"
    stress = tmp_path / "stress"
    baseline = tmp_path / "baseline"
    tcn = tmp_path / "tcn"
    _dataset(dataset)
    generate_stress_dataset(
        stress,
        windows=60,
        seed=2027,
        config=SimulationConfig(
            sample_rate_hz=1_000_000,
            window_samples=1024,
            channels=1,
            resonance_high_hz=300_000.0,
        ),
    )
    train_feature_baseline(dataset, baseline)
    train_tcn(
        dataset,
        tcn,
        epochs=1,
        batch_size=16,
        device="cpu",
        threads=2,
        workers=0,
        seed=8,
    )
    report = evaluate_stress(
        stress,
        baseline / "feature_baseline.joblib",
        tcn / "candidate.onnx",
        tcn / "candidate.manifest.json",
        tmp_path / "stress_report.json",
        reference_dataset_dir=dataset,
    )
    assert report["scope"] == "synthetic_ood_stress_only"
    assert report["dataset"]["samples"] == 60
    assert 0.0 <= report["rejection"]["coverage"] <= 1.0
    assert report["air_velocity_ablation"]["available_fraction"] > 0.0
    assert 0.0 <= report["reference_quality_gate"]["coverage"] <= 1.0
