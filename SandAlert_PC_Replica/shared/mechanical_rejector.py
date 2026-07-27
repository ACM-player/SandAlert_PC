from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.ml_features import (
    MECHANICAL_REJECTOR_FEATURE_NAMES,
    extract_mechanical_rejector_features,
)


@dataclass(frozen=True)
class MechanicalRejectorPrediction:
    probability: float
    rejected: bool


class MechanicalInterferenceRejector:
    """Small NumPy-only prefilter loaded from a portable JSON artifact."""

    def __init__(self, artifact: dict):
        if artifact.get("model_type") != "standardized_logistic_regression":
            raise ValueError("unsupported mechanical rejector model type")
        self.model_id = str(artifact["model_id"])
        self.feature_names = tuple(artifact["feature_names"])
        if self.feature_names != tuple(MECHANICAL_REJECTOR_FEATURE_NAMES):
            raise ValueError("mechanical rejector feature contract mismatch")
        self.mean = np.asarray(artifact["mean"], dtype=np.float64)
        self.scale = np.asarray(artifact["scale"], dtype=np.float64)
        self.coefficients = np.asarray(
            artifact["coefficients"], dtype=np.float64
        )
        self.intercept = float(artifact["intercept"])
        self.threshold = float(artifact["decision_threshold"])
        self.normalized_clip = float(artifact.get("normalized_clip", 20.0))
        expected = len(self.feature_names)
        if not (
            self.mean.size
            == self.scale.size
            == self.coefficients.size
            == expected
        ):
            raise ValueError("mechanical rejector coefficient size mismatch")
        if not 0.0 <= self.threshold <= 1.000001:
            raise ValueError("mechanical rejector threshold is invalid")

    @classmethod
    def load(cls, path: str | Path) -> "MechanicalInterferenceRejector":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict_features(self, features: np.ndarray) -> float:
        values = np.asarray(features, dtype=np.float64).reshape(-1)
        if values.size != self.coefficients.size:
            raise ValueError("mechanical feature vector size mismatch")
        normalized = np.clip(
            (values - self.mean) / np.maximum(self.scale, 1e-12),
            -self.normalized_clip,
            self.normalized_clip,
        )
        logit = float(np.dot(normalized, self.coefficients) + self.intercept)
        if logit >= 0:
            probability = 1.0 / (1.0 + np.exp(-logit))
        else:
            exp_value = np.exp(logit)
            probability = exp_value / (1.0 + exp_value)
        return float(np.clip(probability, 0.0, 1.0))

    def predict(
        self,
        waveform: np.ndarray,
        fs_hz: int,
    ) -> MechanicalRejectorPrediction:
        probability = self.predict_features(
            extract_mechanical_rejector_features(waveform, fs_hz)
        )
        return MechanicalRejectorPrediction(
            probability=probability,
            rejected=probability >= self.threshold,
        )
