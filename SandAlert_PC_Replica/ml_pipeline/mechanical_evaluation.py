from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml_pipeline.dataset import (
    assert_calibration_isolated,
    dataset_fingerprint,
    load_manifest,
    load_waveforms,
)
from ml_pipeline.features import extract_feature_matrix
from ml_pipeline.training import _dsp_predictions
from server.ml_shadow import ONNXShadowModel, ShadowInferenceEngine
from shared.mechanical_rejector import MechanicalInterferenceRejector
from shared.ml_features import extract_shadow_aux_features
from shared.settings import ServerSettings


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_artifact(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": _sha256(path)}


def create_mechanical_candidate_lock(
    route_a_onnx: str | Path,
    route_a_manifest: str | Path,
    route_b_onnx: str | Path,
    route_b_manifest: str | Path,
    route_b_rejector: str | Path,
    calibration_dataset_dir: str | Path,
    output_path: str | Path,
) -> dict:
    """Seal candidates before any frozen OOD path is supplied."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            "candidate lock already exists; create a new versioned lock"
        )
    route_a_payload = json.loads(
        Path(route_a_manifest).read_text(encoding="utf-8")
    )
    route_b_payload = json.loads(
        Path(route_b_manifest).read_text(encoding="utf-8")
    )
    rejector_payload = json.loads(
        Path(route_b_rejector).read_text(encoding="utf-8")
    )
    if not route_a_payload.get("quality_head_enabled"):
        raise ValueError("Route A candidate must expose a quality head")
    if route_b_payload.get("quality_head_enabled"):
        raise ValueError("Route B TCN must not use the Route A quality head")
    payloads = (route_a_payload, route_b_payload, rejector_payload)
    if any(
        item.get("frozen_ood_used_for_training_or_calibration") is not False
        for item in payloads
    ):
        raise ValueError("candidate provenance does not prove frozen isolation")
    lock = {
        "lock_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "mechanical_calibration_only",
        "calibration_dataset_fingerprint": dataset_fingerprint(
            calibration_dataset_dir
        ),
        "frozen_ood_path_seen_during_lock": False,
        "criteria_locked_before_frozen_evaluation": {
            "mechanical_false_positive_rate_strictly_below": 0.20,
            "mechanical_false_negative_rate_maximum": 0.25,
            "maximum_fnr_increase_over_route_raw": 0.05,
        },
        "route_a": {
            "description": "TCN quality head plus adversarial event training",
            "model_id": route_a_payload["model_id"],
            "onnx": _locked_artifact(route_a_onnx),
            "manifest": _locked_artifact(route_a_manifest),
            "calibration_evidence": {
                "quality_calibration": route_a_payload.get(
                    "quality_calibration"
                ),
                "adversarial_holdout": route_a_payload.get("test_metrics"),
            },
        },
        "route_b": {
            "description": (
                "dedicated mechanical prefilter plus adversarial TCN"
            ),
            "model_id": route_b_payload["model_id"],
            "rejector_model_id": rejector_payload["model_id"],
            "onnx": _locked_artifact(route_b_onnx),
            "manifest": _locked_artifact(route_b_manifest),
            "rejector": _locked_artifact(route_b_rejector),
            "calibration_evidence": {
                "tcn_adversarial_holdout": route_b_payload.get("test_metrics"),
                "prefilter_calibration": rejector_payload.get(
                    "calibration_metrics"
                ),
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return lock


def _verify_lock_artifact(entry: dict) -> Path:
    path = Path(entry["path"])
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"locked artifact changed after selection: {path}")
    return path


def _operational_event_metrics(
    rows: list[dict],
    mask: np.ndarray,
    probability: np.ndarray,
    accepted: np.ndarray,
    threshold: float,
) -> dict:
    selected = np.flatnonzero(mask)
    truth = np.asarray(
        [int(rows[index]["event_count"]) > 0 for index in selected],
        dtype=bool,
    )
    selected_accepted = accepted[selected]
    prediction = selected_accepted & (probability[selected] >= threshold)
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "samples": int(selected.size),
        "event_threshold": float(threshold),
        "coverage": float(np.mean(selected_accepted)) if selected.size else None,
        "positive_skip_rate": (
            float(np.mean(~selected_accepted[truth]))
            if np.any(truth)
            else None
        ),
        "negative_skip_rate": (
            float(np.mean(~selected_accepted[~truth]))
            if np.any(~truth)
            else None
        ),
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "event_f1": (
            2.0 * precision * recall / max(precision + recall, 1e-12)
        ),
        "confusion": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
    }


def _runtime_predictions(
    rows: list[dict],
    waveforms: np.ndarray,
    runtime: ONNXShadowModel,
    engine: ShadowInferenceEngine,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = []
    ips_values = []
    energy_values = []
    quality_probabilities = []
    generic_accepted = []
    fs_hz = int(rows[0]["sample_rate_hz"])
    for row, waveform in zip(rows, waveforms):
        raw = waveform[0]
        air_velocity = (
            row.get("air_velocity_m_s")
            if row.get("air_velocity_available", False)
            else None
        )
        aux = extract_shadow_aux_features(
            raw,
            fs_hz,
            row["analytic_relative_energy_db"],
            air_velocity,
            feature_names=(
                tuple(runtime.aux_feature_names)
                if runtime.aux_feature_names
                else None
            ),
        )
        generic_accepted.append(not engine._quality_flags(aux))
        probability, ips, energy, quality_probability = (
            runtime.predict_extended(
                ShadowInferenceEngine._normalize(raw), aux.values
            )
        )
        probabilities.append(probability)
        ips_values.append(ips)
        energy_values.append(energy)
        quality_probabilities.append(
            np.nan if quality_probability is None else quality_probability
        )
    return (
        np.asarray(probabilities),
        np.asarray(ips_values),
        np.asarray(energy_values),
        np.asarray(quality_probabilities),
        np.asarray(generic_accepted, dtype=bool),
    )


def _candidate_result(
    rows: list[dict],
    probability: np.ndarray,
    accepted: np.ndarray,
    threshold: float,
    raw_accepted: np.ndarray,
    criteria: dict,
) -> dict:
    all_mask = np.ones(len(rows), dtype=bool)
    mechanical_mask = np.asarray(
        [row["scenario"] == "mechanical_interference" for row in rows]
    )
    overall = _operational_event_metrics(
        rows, all_mask, probability, accepted, threshold
    )
    mechanical = _operational_event_metrics(
        rows, mechanical_mask, probability, accepted, threshold
    )
    mechanical_raw = _operational_event_metrics(
        rows, mechanical_mask, probability, raw_accepted, threshold
    )
    fpr_pass = (
        mechanical["false_positive_rate"]
        < criteria["mechanical_false_positive_rate_strictly_below"]
    )
    fnr_pass = (
        mechanical["false_negative_rate"]
        <= criteria["mechanical_false_negative_rate_maximum"]
        and mechanical["false_negative_rate"]
        <= mechanical_raw["false_negative_rate"]
        + criteria["maximum_fnr_increase_over_route_raw"]
    )
    return {
        "overall": overall,
        "mechanical_interference": mechanical,
        "mechanical_interference_raw_without_specialized_rejection": (
            mechanical_raw
        ),
        "goal": {
            "mechanical_fpr_below_20_percent": bool(fpr_pass),
            "mechanical_fnr_within_locked_limit": bool(fnr_pass),
            "candidate_pass": bool(fpr_pass and fnr_pass),
        },
    }


def _markdown_report(report: dict) -> str:
    lines = [
        "# 机械干扰专项冻结集评估",
        "",
        "本报告只用于模拟数据下的影子模型候选比较，不构成现场 g/s、报警或继电器控制依据。",
        "",
        "## 数据隔离审计",
        "",
        f"- 校准集 SHA-256：`{report['isolation']['calibration_fingerprint']}`",
        f"- 冻结 OOD SHA-256：`{report['isolation']['frozen_ood_fingerprint']}`",
        f"- 组重叠：{report['isolation']['group_overlap']}",
        f"- 种子重叠：{report['isolation']['seed_overlap']}",
        f"- 冻结角色校验：`{report['isolation']['frozen_role_verification']}`",
        "",
        "## 候选对比",
        "",
        "| 路线 | 机械误报率 | 机械漏报率 | 正样本跳过率 | 机械覆盖率 | 目标结论 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, label in (
        ("route_a", "A：TCN Quality Head"),
        ("route_b", "B：机械前置拒绝器"),
    ):
        values = report["candidates"][key]["mechanical_interference"]
        goal = report["candidates"][key]["goal"]
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    f"{100 * values['false_positive_rate']:.1f}%",
                    f"{100 * values['false_negative_rate']:.1f}%",
                    f"{100 * (values['positive_skip_rate'] or 0.0):.1f}%",
                    f"{100 * (values['coverage'] or 0.0):.1f}%",
                    "通过" if goal["candidate_pass"] else "未通过",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## 约束与结论",
            "",
            "- 候选在读取冻结集前已经由锁文件固化，冻结集没有参与训练、阈值校准或候选选择。",
            "- 被拒绝的正样本按漏报计入，不能通过大量 `skipped` 人为降低误报率。",
            "- DSP、`analysis.result`、g/s、报警和继电器链路均未被候选模型修改。",
            "- 所有结论仍限于模拟波形；真实探头与气固试验台数据到位后必须重新标定。",
            "",
        )
    )
    return "\n".join(lines)


def evaluate_locked_mechanical_candidates(
    frozen_ood_dataset_dir: str | Path,
    calibration_dataset_dir: str | Path,
    feature_model_path: str | Path,
    candidate_lock_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Run the one-shot blind comparison after candidate selection is sealed."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            "frozen evaluation output already exists; do not retune or overwrite"
        )
    lock = json.loads(
        Path(candidate_lock_path).read_text(encoding="utf-8")
    )
    current_calibration_hash = dataset_fingerprint(calibration_dataset_dir)
    if (
        current_calibration_hash
        != lock["calibration_dataset_fingerprint"]
    ):
        raise ValueError("calibration dataset changed after candidate lock")
    isolation = assert_calibration_isolated(
        calibration_dataset_dir, frozen_ood_dataset_dir
    )
    route_a_onnx = _verify_lock_artifact(lock["route_a"]["onnx"])
    route_a_manifest = _verify_lock_artifact(lock["route_a"]["manifest"])
    route_b_onnx = _verify_lock_artifact(lock["route_b"]["onnx"])
    route_b_manifest = _verify_lock_artifact(lock["route_b"]["manifest"])
    route_b_rejector_path = _verify_lock_artifact(
        lock["route_b"]["rejector"]
    )

    rows = load_manifest(frozen_ood_dataset_dir)
    waveforms = load_waveforms(
        frozen_ood_dataset_dir, [int(row["index"]) for row in rows]
    )
    fs_hz = int(rows[0]["sample_rate_hz"])
    route_a_settings = ServerSettings(
        ml_enabled=True,
        ml_model_path=route_a_onnx,
        ml_manifest_path=route_a_manifest,
    )
    route_b_settings = ServerSettings(
        ml_enabled=True,
        ml_model_path=route_b_onnx,
        ml_manifest_path=route_b_manifest,
    )
    route_a_runtime = ONNXShadowModel(route_a_settings)
    route_b_runtime = ONNXShadowModel(route_b_settings)
    if not route_a_runtime.ready or not route_b_runtime.ready:
        raise RuntimeError("one or more locked ONNX candidates are unavailable")
    route_a_values = _runtime_predictions(
        rows,
        waveforms,
        route_a_runtime,
        ShadowInferenceEngine(route_a_settings, route_a_runtime),
    )
    route_b_values = _runtime_predictions(
        rows,
        waveforms,
        route_b_runtime,
        ShadowInferenceEngine(route_b_settings, route_b_runtime),
    )
    (
        route_a_probability,
        _,
        _,
        route_a_quality_probability,
        route_a_generic_accepted,
    ) = route_a_values
    if np.any(~np.isfinite(route_a_quality_probability)):
        raise ValueError("Route A ONNX output lacks quality probabilities")
    route_a_accepted = route_a_generic_accepted & (
        route_a_quality_probability
        < route_a_runtime.quality_decision_threshold
    )

    (
        route_b_probability,
        _,
        _,
        _,
        route_b_generic_accepted,
    ) = route_b_values
    rejector = MechanicalInterferenceRejector.load(route_b_rejector_path)
    route_b_prefilter_probability = np.asarray(
        [
            rejector.predict(waveform[0], fs_hz).probability
            for waveform in waveforms
        ]
    )
    route_b_accepted = route_b_generic_accepted & (
        route_b_prefilter_probability < rejector.threshold
    )

    criteria = lock["criteria_locked_before_frozen_evaluation"]
    candidates = {
        "route_a": _candidate_result(
            rows,
            route_a_probability,
            route_a_accepted,
            route_a_runtime.event_decision_threshold,
            route_a_generic_accepted,
            criteria,
        ),
        "route_b": _candidate_result(
            rows,
            route_b_probability,
            route_b_accepted,
            route_b_runtime.event_decision_threshold,
            route_b_generic_accepted,
            criteria,
        ),
    }
    dsp_probability, _ = _dsp_predictions(waveforms, fs_hz)
    feature_artifact = __import__("joblib").load(feature_model_path)
    feature_probability = feature_artifact["classifier"].predict_proba(
        extract_feature_matrix(waveforms, fs_hz)
    )[:, 1]
    all_accepted = np.ones(len(rows), dtype=bool)
    mechanical_mask = np.asarray(
        [row["scenario"] == "mechanical_interference" for row in rows]
    )
    report = {
        "scope": "synthetic_frozen_ood_mechanical_blind_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_lock_sha256": _sha256(candidate_lock_path),
        "candidate_lock": lock,
        "isolation": isolation,
        "dataset": {
            "samples": len(rows),
            "scenarios": dict(Counter(row["scenario"] for row in rows)),
            "sample_rate_hz": fs_hz,
            "fingerprint": isolation["frozen_ood_fingerprint"],
        },
        "candidates": candidates,
        "references": {
            "dsp_mechanical_interference": _operational_event_metrics(
                rows,
                mechanical_mask,
                dsp_probability,
                all_accepted,
                0.5,
            ),
            "feature_model_mechanical_interference": (
                _operational_event_metrics(
                    rows,
                    mechanical_mask,
                    feature_probability,
                    all_accepted,
                    0.5,
                )
            ),
        },
        "decision": {
            "passing_routes": [
                key
                for key, value in candidates.items()
                if value["goal"]["candidate_pass"]
            ],
            "deployment_scope": "shadow_only",
            "formal_dsp_path_unchanged": True,
        },
        "limitations": [
            "All waveforms and labels are synthetic.",
            "The frozen suite was read only after candidates were locked.",
            "No result is evidence of field g/s accuracy.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_path.with_suffix(".md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    return report
