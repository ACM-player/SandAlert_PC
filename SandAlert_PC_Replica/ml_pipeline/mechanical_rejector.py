from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ml_pipeline.dataset import (
    dataset_fingerprint,
    load_manifest,
    load_waveforms,
)
from ml_pipeline.training import (
    _calibrate_rejection_threshold,
    _rejection_metrics,
)
from shared.mechanical_rejector import MechanicalInterferenceRejector
from shared.ml_features import (
    MECHANICAL_REJECTOR_FEATURE_NAMES,
    extract_mechanical_rejector_features,
)


def _feature_matrix(
    waveforms: np.ndarray,
    fs_hz: int,
) -> np.ndarray:
    return np.vstack(
        [
            extract_mechanical_rejector_features(waveform[0], fs_hz)
            for waveform in waveforms
        ]
    ).astype(np.float64)


def train_mechanical_rejector(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    max_positive_false_rejection_rate: float = 0.05,
    max_ordinary_negative_false_rejection_rate: float = 0.05,
) -> dict:
    """Train Route B without introducing scikit-learn into server runtime."""
    from sklearn.linear_model import LogisticRegression

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(
        (dataset_dir / "dataset_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("dataset_role") != "adversarial_training":
        raise ValueError("rejector requires an adversarial training dataset")
    if summary.get("frozen_ood_used") is not False:
        raise ValueError("training summary does not prove frozen OOD isolation")
    rows = load_manifest(dataset_dir)
    waveforms = load_waveforms(
        dataset_dir, [int(row["index"]) for row in rows]
    )
    fs_hz = int(rows[0]["sample_rate_hz"])
    features = _feature_matrix(waveforms, fs_hz)
    target = np.asarray(
        [bool(row.get("mechanical_reject_target", False)) for row in rows],
        dtype=int,
    )
    event = np.asarray(
        [int(row.get("event_count", 0)) > 0 for row in rows], dtype=int
    )
    splits = np.asarray([row["split"] for row in rows])
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    if np.unique(target[train_mask]).size < 2:
        raise ValueError("rejector training split requires both target classes")

    mean = np.mean(features[train_mask], axis=0)
    scale = np.maximum(np.std(features[train_mask], axis=0), 1e-8)
    normalized_clip = 20.0
    normalized = np.clip(
        (features - mean) / scale,
        -normalized_clip,
        normalized_clip,
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
        solver="liblinear",
    ).fit(normalized[train_mask], target[train_mask])
    probabilities = classifier.predict_proba(normalized)[:, 1]
    threshold, calibration_metrics = _calibrate_rejection_threshold(
        target[validation_mask],
        probabilities[validation_mask],
        event[validation_mask],
        max_positive_false_rejection_rate=max_positive_false_rejection_rate,
        max_ordinary_negative_false_rejection_rate=(
            max_ordinary_negative_false_rejection_rate
        ),
    )
    dataset_hash = dataset_fingerprint(dataset_dir)
    model_id = (
        f"sandalert-mechanical-rejector-v1-"
        f"{dataset_dir.name}-seed{seed}"
    )
    artifact = {
        "model_id": model_id,
        "model_type": "standardized_logistic_regression",
        "feature_names": list(MECHANICAL_REJECTOR_FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": classifier.coef_.reshape(-1).tolist(),
        "intercept": float(classifier.intercept_[0]),
        "decision_threshold": threshold,
        "normalized_clip": normalized_clip,
        "sample_rate_hz": fs_hz,
        "dataset_id": dataset_dir.name,
        "dataset_fingerprint": dataset_hash,
        "seed": seed,
        "calibration_metrics": calibration_metrics,
        "frozen_ood_used_for_training_or_calibration": False,
    }
    artifact_path = output_dir / "mechanical_rejector.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    runtime = MechanicalInterferenceRejector.load(artifact_path)
    runtime_probabilities = np.asarray(
        [runtime.predict_features(row) for row in features]
    )
    np.testing.assert_allclose(
        runtime_probabilities, probabilities, rtol=1e-7, atol=1e-8
    )
    report = {
        "model": model_id,
        "route": "B_dedicated_mechanical_prefilter",
        "dataset_fingerprint": dataset_hash,
        "calibration": calibration_metrics,
        "splits": {},
        "artifact": str(artifact_path),
        "artifact_sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
    }
    for split, mask in (
        ("validation", validation_mask),
        ("test", test_mask),
    ):
        report["splits"][split] = _rejection_metrics(
            target[mask],
            runtime_probabilities[mask],
            event[mask],
            threshold,
        )
    (output_dir / "mechanical_rejector_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
