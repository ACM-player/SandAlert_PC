from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ml_pipeline.dataset import load_manifest, load_waveforms
from ml_pipeline.features import extract_feature_matrix
from ml_pipeline.training import _dsp_predictions, _metrics
from server.ml_shadow import ONNXShadowModel, ShadowInferenceEngine
from shared.ml_features import extract_shadow_aux_features
from shared.settings import ServerSettings


def _subset_metrics(
    rows: list[dict],
    indices: np.ndarray,
    probability: np.ndarray,
    ips: np.ndarray,
    energy: np.ndarray,
) -> dict | None:
    if not np.any(indices):
        return None
    selected = np.flatnonzero(indices)
    target_event = np.asarray(
        [rows[index]["event_count"] > 0 for index in selected], dtype=int
    )
    target_ips = np.asarray(
        [rows[index]["ips_hz"] for index in selected], dtype=float
    )
    target_energy = np.asarray(
        [rows[index]["relative_energy_db"] for index in selected], dtype=float
    )
    return _metrics(
        target_event,
        probability[selected],
        target_ips,
        ips[selected],
        target_energy,
        energy[selected],
    )


def _quality_gate_summary(
    rows: list[dict],
    waveforms: np.ndarray,
    fs_hz: int,
    quality_engine: ShadowInferenceEngine,
) -> dict:
    accepted = []
    hard_corruption = []
    flags_counter: Counter[str] = Counter()
    scenario_total: Counter[str] = Counter()
    scenario_accepted: Counter[str] = Counter()
    settings = quality_engine.settings
    for row, waveform in zip(rows, waveforms):
        air_velocity = (
            row.get("air_velocity_m_s")
            if row.get("air_velocity_available", False)
            else None
        )
        features = extract_shadow_aux_features(
            waveform[0],
            fs_hz,
            row["analytic_relative_energy_db"],
            air_velocity,
        )
        flags = quality_engine._quality_flags(features)
        is_accepted = not flags
        accepted.append(is_accepted)
        flags_counter.update(flags)
        scenario_total[row["scenario"]] += 1
        if is_accepted:
            scenario_accepted[row["scenario"]] += 1
        hard_corruption.append(
            row.get("clipping_fraction", 0.0)
            > settings.ml_max_clipping_fraction
            or row.get("dropout_fraction", 0.0)
            > settings.ml_max_dropout_fraction
        )
    accepted_array = np.asarray(accepted, dtype=bool)
    corruption_array = np.asarray(hard_corruption, dtype=bool)
    return {
        "coverage": float(np.mean(accepted_array)),
        "rejected": int(np.count_nonzero(~accepted_array)),
        "quality_flags": dict(flags_counter),
        "hard_corruption_recall": float(
            np.mean(~accepted_array[corruption_array])
        ) if np.any(corruption_array) else None,
        "clean_or_nonhard_false_rejection_rate": float(
            np.mean(~accepted_array[~corruption_array])
        ) if np.any(~corruption_array) else None,
        "scenario_coverage": {
            scenario: (
                scenario_accepted[scenario] / max(scenario_total[scenario], 1)
            )
            for scenario in sorted(scenario_total)
        },
    }


def evaluate_stress(
    dataset_dir: str | Path,
    feature_model_path: str | Path,
    onnx_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    settings: ServerSettings | None = None,
    reference_dataset_dir: str | Path | None = None,
) -> dict:
    """Evaluate DSP, feature and TCN models on a frozen OOD dataset."""
    from joblib import load

    dataset_dir = Path(dataset_dir)
    rows = load_manifest(dataset_dir)
    waveforms = load_waveforms(
        dataset_dir, [int(row["index"]) for row in rows]
    )
    fs_hz = int(rows[0]["sample_rate_hz"])
    target_event = np.asarray(
        [row["event_count"] > 0 for row in rows], dtype=int
    )
    target_ips = np.asarray([row["ips_hz"] for row in rows], dtype=float)
    target_energy = np.asarray(
        [row["relative_energy_db"] for row in rows], dtype=float
    )

    dsp_probability, dsp_ips = _dsp_predictions(waveforms, fs_hz)
    analytic_energy = np.asarray(
        [row["analytic_relative_energy_db"] for row in rows], dtype=float
    )

    feature_artifact = load(feature_model_path)
    feature_matrix = extract_feature_matrix(waveforms, fs_hz)
    feature_probability = feature_artifact["classifier"].predict_proba(
        feature_matrix
    )[:, 1]
    feature_ips = np.maximum(
        0.0, feature_artifact["ips_regressor"].predict(feature_matrix)
    )
    feature_energy = feature_artifact["energy_regressor"].predict(
        feature_matrix
    )

    runtime_settings = settings or ServerSettings(
        ml_enabled=True,
        ml_model_path=Path(onnx_path),
        ml_manifest_path=Path(manifest_path),
    )
    runtime = ONNXShadowModel(runtime_settings)
    if not runtime.ready:
        raise RuntimeError(runtime.reason or "ONNX runtime unavailable")
    quality_engine = ShadowInferenceEngine(runtime_settings, runtime)

    tcn_probability = []
    tcn_ips = []
    tcn_energy = []
    tcn_energy_without_air = []
    accepted = []
    flags_counter: Counter[str] = Counter()
    per_scenario_total: Counter[str] = Counter()
    per_scenario_accepted: Counter[str] = Counter()

    for row, waveform in zip(rows, waveforms):
        raw = waveform[0]
        normalized = ShadowInferenceEngine._normalize(raw)
        air_velocity = (
            row.get("air_velocity_m_s")
            if row.get("air_velocity_available", False)
            else None
        )
        actual_aux = extract_shadow_aux_features(
            raw,
            fs_hz,
            row["analytic_relative_energy_db"],
            air_velocity,
        )
        no_air_aux = extract_shadow_aux_features(
            raw,
            fs_hz,
            row["analytic_relative_energy_db"],
            None,
        )
        flags = quality_engine._quality_flags(actual_aux)
        is_accepted = not flags
        accepted.append(is_accepted)
        flags_counter.update(flags)
        per_scenario_total[row["scenario"]] += 1
        if is_accepted:
            per_scenario_accepted[row["scenario"]] += 1

        probability, ips, energy = runtime.predict(
            normalized, actual_aux.values
        )
        _, _, energy_without_air = runtime.predict(
            normalized, no_air_aux.values
        )
        tcn_probability.append(probability)
        tcn_ips.append(ips)
        tcn_energy.append(energy)
        tcn_energy_without_air.append(energy_without_air)

    tcn_probability = np.asarray(tcn_probability)
    tcn_ips = np.asarray(tcn_ips)
    tcn_energy = np.asarray(tcn_energy)
    tcn_energy_without_air = np.asarray(tcn_energy_without_air)
    accepted = np.asarray(accepted, dtype=bool)
    hard_corruption = np.asarray([
        row.get("clipping_fraction", 0.0)
        > runtime_settings.ml_max_clipping_fraction
        or row.get("dropout_fraction", 0.0)
        > runtime_settings.ml_max_dropout_fraction
        for row in rows
    ])

    report = {
        "scope": "synthetic_ood_stress_only",
        "dataset": {
            "samples": len(rows),
            "sample_rate_hz": fs_hz,
            "window_samples": int(rows[0]["window_samples"]),
            "scenarios": dict(Counter(row["scenario"] for row in rows)),
            "severity": {
                "minimum": float(min(row["severity"] for row in rows)),
                "maximum": float(max(row["severity"] for row in rows)),
            },
        },
        "overall": {
            "dsp": _metrics(
                target_event,
                dsp_probability,
                target_ips,
                dsp_ips,
                target_energy,
                np.zeros_like(target_energy),
            ),
            "feature_model": _metrics(
                target_event,
                feature_probability,
                target_ips,
                feature_ips,
                target_energy,
                feature_energy,
            ),
            "analytic_energy_baseline_mae_db": float(
                np.mean(np.abs(target_energy - analytic_energy))
            ),
            "tcn_raw": _metrics(
                target_event,
                tcn_probability,
                target_ips,
                tcn_ips,
                target_energy,
                tcn_energy,
            ),
            "tcn_accepted": _subset_metrics(
                rows,
                accepted,
                tcn_probability,
                tcn_ips,
                tcn_energy,
            ),
        },
        "rejection": {
            "coverage": float(np.mean(accepted)),
            "rejected": int(np.count_nonzero(~accepted)),
            "quality_flags": dict(flags_counter),
            "hard_corruption_recall": float(
                np.mean(~accepted[hard_corruption])
            ) if np.any(hard_corruption) else None,
            "clean_or_nonhard_false_rejection_rate": float(
                np.mean(~accepted[~hard_corruption])
            ) if np.any(~hard_corruption) else None,
            "scenario_coverage": {
                scenario: (
                    per_scenario_accepted[scenario]
                    / max(per_scenario_total[scenario], 1)
                )
                for scenario in sorted(per_scenario_total)
            },
        },
        "air_velocity_ablation": {
            "available_fraction": float(
                np.mean([
                    bool(row.get("air_velocity_available", False))
                    for row in rows
                ])
            ),
            "with_air_energy_mae_db": float(
                np.mean(np.abs(target_energy - tcn_energy))
            ),
            "without_air_energy_mae_db": float(
                np.mean(np.abs(target_energy - tcn_energy_without_air))
            ),
        },
        "per_scenario": {},
        "limitations": [
            "All inputs and labels are synthetic.",
            "The OOD suite validates rejection and software robustness only.",
            "No field g/s accuracy claim is permitted.",
        ],
    }
    if reference_dataset_dir is not None:
        reference_rows = load_manifest(reference_dataset_dir)
        reference_waveforms = load_waveforms(
            reference_dataset_dir,
            [int(row["index"]) for row in reference_rows],
        )
        report["reference_quality_gate"] = _quality_gate_summary(
            reference_rows,
            reference_waveforms,
            int(reference_rows[0]["sample_rate_hz"]),
            quality_engine,
        )
    for scenario in sorted(per_scenario_total):
        mask = np.asarray([row["scenario"] == scenario for row in rows])
        report["per_scenario"][scenario] = {
            "samples": int(np.count_nonzero(mask)),
            "dsp": _subset_metrics(
                rows, mask, dsp_probability, dsp_ips, np.zeros_like(target_energy)
            ),
            "feature_model": _subset_metrics(
                rows, mask, feature_probability, feature_ips, feature_energy
            ),
            "tcn_raw": _subset_metrics(
                rows, mask, tcn_probability, tcn_ips, tcn_energy
            ),
            "tcn_accepted": _subset_metrics(
                rows, mask & accepted, tcn_probability, tcn_ips, tcn_energy
            ),
            "coverage": float(np.mean(accepted[mask])),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
