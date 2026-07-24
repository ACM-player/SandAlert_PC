from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np

from core.dsp_engine import DSPEngine
from ml_pipeline.dataset import load_manifest, load_waveforms
from ml_pipeline.features import FEATURE_NAMES, extract_feature_matrix
from shared.ml_features import AUX_FEATURE_NAMES, extract_shadow_aux_features


def _metrics(
    y_true_event,
    probability,
    y_true_ips,
    ips_estimate,
    y_true_energy,
    energy,
    *,
    event_threshold: float = 0.5,
):
    from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error

    predicted_event = np.asarray(probability) >= event_threshold
    true_event = np.asarray(y_true_event, dtype=bool)
    event_classes = np.unique(y_true_event)
    true_positive = int(np.count_nonzero(predicted_event & true_event))
    false_positive = int(np.count_nonzero(predicted_event & ~true_event))
    true_negative = int(np.count_nonzero(~predicted_event & ~true_event))
    false_negative = int(np.count_nonzero(~predicted_event & true_event))
    return {
        "event_f1": float(f1_score(y_true_event, predicted_event, zero_division=0)),
        "event_threshold": float(event_threshold),
        "event_average_precision": (
            float(average_precision_score(y_true_event, probability))
            if event_classes.size > 1
            else None
        ),
        "false_positive_rate": float(
            false_positive / max(false_positive + true_negative, 1)
        ),
        "false_negative_rate": float(
            false_negative / max(false_negative + true_positive, 1)
        ),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "ips_mae": float(mean_absolute_error(y_true_ips, ips_estimate)),
        "relative_energy_mae_db": float(
            mean_absolute_error(y_true_energy, energy)
        ),
    }


def _targets(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event = np.asarray([row["event_count"] > 0 for row in records], dtype=int)
    ips = np.asarray([row["ips_hz"] for row in records], dtype=float)
    energy = np.asarray([row["relative_energy_db"] for row in records], dtype=float)
    return event, ips, energy


def _aux_feature_matrix(
    records: list[dict],
    waveforms: np.ndarray,
    fs_hz: int,
) -> np.ndarray:
    values = []
    for row, waveform in zip(records, waveforms):
        air_velocity = (
            row.get("air_velocity_m_s")
            if row.get("air_velocity_available", False)
            else None
        )
        values.append(
            extract_shadow_aux_features(
                waveform[0] if waveform.ndim == 2 else waveform,
                fs_hz,
                row.get("analytic_relative_energy_db"),
                air_velocity,
            ).values
        )
    return np.vstack(values).astype(np.float32)


def _dsp_predictions(
    waveforms: np.ndarray, fs_hz: int, threshold_margin: float = 0.008
) -> tuple[np.ndarray, np.ndarray]:
    engine = DSPEngine(sample_rate=fs_hz)
    duration = waveforms.shape[-1] / fs_hz
    probabilities = []
    ips = []
    for item in waveforms:
        _, _, peak_count, _, _, _ = engine.process_window(
            item[0], threshold_margin, 1.0
        )
        probabilities.append(float(peak_count > 0))
        ips.append(float(peak_count) / duration)
    return np.asarray(probabilities), np.asarray(ips)


def _promotion_decision(model_metrics: dict, dsp_metrics: dict) -> dict:
    f1_not_worse = model_metrics["event_f1"] >= dsp_metrics["event_f1"]
    ips_not_worse = model_metrics["ips_mae"] <= dsp_metrics["ips_mae"]
    f1_gain = (
        (model_metrics["event_f1"] - dsp_metrics["event_f1"])
        / max(abs(dsp_metrics["event_f1"]), 1e-12)
    )
    ips_gain = (
        (dsp_metrics["ips_mae"] - model_metrics["ips_mae"])
        / max(abs(dsp_metrics["ips_mae"]), 1e-12)
    )
    promoted = f1_not_worse and ips_not_worse and max(f1_gain, ips_gain) >= 0.10
    return {
        "eligible_for_shadow": bool(promoted),
        "event_f1_not_worse": bool(f1_not_worse),
        "ips_mae_not_worse": bool(ips_not_worse),
        "event_f1_relative_gain": float(f1_gain),
        "ips_mae_relative_gain": float(ips_gain),
        "required_relative_gain": 0.10,
        "scope": "synthetic_holdout_only",
    }


def train_feature_baseline(
    dataset_dir: str | Path, output_dir: str | Path
) -> dict:
    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(dataset_dir)
    config = records[0]
    fs_hz = int(config["sample_rate_hz"])

    subsets: dict[str, tuple[list[dict], np.ndarray, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in records if row["split"] == split]
        indices = np.asarray([row["index"] for row in rows], dtype=int)
        waveforms = load_waveforms(dataset_dir, indices.tolist())
        subsets[split] = (
            rows,
            waveforms,
            extract_feature_matrix(waveforms, fs_hz),
        )

    train_rows, _, train_features = subsets["train"]
    train_event, train_ips, train_energy = _targets(train_rows)
    classifier = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, random_state=42
    ).fit(train_features, train_event)
    ips_regressor = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.08, loss="absolute_error", random_state=43
    ).fit(train_features, train_ips)
    energy_regressor = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.08, loss="absolute_error", random_state=44
    ).fit(train_features, train_energy)

    artifact = {
        "classifier": classifier,
        "ips_regressor": ips_regressor,
        "energy_regressor": energy_regressor,
        "feature_names": FEATURE_NAMES,
        "sample_rate_hz": fs_hz,
    }
    artifact_path = output_dir / "feature_baseline.joblib"
    dump(artifact, artifact_path)

    report = {"model": "feature-baseline-v1", "splits": {}}
    for split in ("validation", "test"):
        rows, waveforms, features = subsets[split]
        event, ips, energy = _targets(rows)
        probability = classifier.predict_proba(features)[:, 1]
        ips_estimate = np.maximum(0.0, ips_regressor.predict(features))
        energy_estimate = energy_regressor.predict(features)
        dsp_probability, dsp_ips = _dsp_predictions(waveforms, fs_hz)
        report["splits"][split] = {
            "model": _metrics(
                event, probability, ips, ips_estimate, energy, energy_estimate
            ),
            "dsp": _metrics(
                event,
                dsp_probability,
                ips,
                dsp_ips,
                energy,
                np.zeros_like(energy),
            ),
            "samples": len(rows),
        }
    report["promotion"] = _promotion_decision(
        report["splits"]["test"]["model"],
        report["splits"]["test"]["dsp"],
    )
    report["artifact"] = str(artifact_path)
    report_path = output_dir / "feature_baseline_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_tcn(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    threads: int = 32,
    workers: int = 8,
    seed: int = 42,
    energy_loss_weight: float = 1.0,
    event_target_false_positive_rate: float = 0.05,
    event_positive_weight: float = 1.0,
) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from ml_pipeline.models import SandAlertONNXWrapper, SandAlertTCN

    if device != "cpu":
        raise ValueError("this phase is intentionally restricted to device='cpu'")
    if energy_loss_weight <= 0:
        raise ValueError("energy loss weight must be positive")
    if not 0.0 <= event_target_false_positive_rate < 1.0:
        raise ValueError("target false-positive rate must remain within [0, 1)")
    if event_positive_weight <= 0:
        raise ValueError("event positive weight must be positive")
    _seed_everything(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_num_interop_threads(max(1, min(4, int(threads))))
    except RuntimeError:
        pass

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(dataset_dir)
    dataset_id = dataset_dir.name
    fs_hz = int(records[0]["sample_rate_hz"])
    window_samples = int(records[0]["window_samples"])
    hdf5_path = dataset_dir / "waveforms.h5"

    all_waveforms = load_waveforms(
        dataset_dir, [int(row["index"]) for row in records]
    )
    all_aux = _aux_feature_matrix(records, all_waveforms, fs_hz)
    aux_by_index = {
        int(row["index"]): all_aux[position]
        for position, row in enumerate(records)
    }
    train_positions = [
        position
        for position, row in enumerate(records)
        if row["split"] == "train"
    ]
    train_aux = all_aux[np.asarray(train_positions, dtype=int)]
    aux_mean = np.mean(train_aux, axis=0).astype(np.float32)
    aux_std = np.std(train_aux, axis=0).astype(np.float32)
    aux_std = np.maximum(aux_std, 1e-6)
    train_residuals = np.asarray(
        [
            float(row["relative_energy_db"])
            - float(row.get("analytic_relative_energy_db", 0.0))
            for row in records
            if row["split"] == "train"
        ],
        dtype=np.float32,
    )
    energy_residual_scale_db = max(float(np.std(train_residuals)), 1.0)

    class WaveformDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
            self._handle = None

        def __len__(self):
            return len(self.rows)

        def _waveforms(self):
            if self._handle is None:
                self._handle = h5py.File(hdf5_path, "r")
            return self._handle["waveforms"]

        def __getitem__(self, index):
            row = self.rows[index]
            waveform = np.asarray(
                self._waveforms()[row["index"], 0], dtype=np.float32
            )
            waveform -= np.median(waveform)
            mad = float(np.median(np.abs(waveform)))
            scale = max(1.4826 * mad, float(np.std(waveform)) * 0.1, 1e-8)
            waveform = np.clip(waveform / scale, -20.0, 20.0)
            raw_aux = aux_by_index[int(row["index"])]
            normalized_aux = (raw_aux - aux_mean) / aux_std
            target_energy = float(row["relative_energy_db"])
            analytic_energy = float(
                row.get("analytic_relative_energy_db", 0.0)
            )
            energy_residual = (
                target_energy - analytic_energy
            ) / energy_residual_scale_db
            return (
                torch.from_numpy(waveform[None, :].copy()),
                torch.from_numpy(normalized_aux.copy()),
                torch.from_numpy(raw_aux.copy()),
                torch.tensor(float(row["event_count"] > 0), dtype=torch.float32),
                torch.tensor(np.log1p(row["ips_hz"]), dtype=torch.float32),
                torch.tensor(target_energy, dtype=torch.float32),
                torch.tensor(energy_residual, dtype=torch.float32),
            )

    split_rows = {
        split: [row for row in records if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        WaveformDataset(split_rows["train"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(0, int(workers)),
        generator=generator,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        WaveformDataset(split_rows["validation"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(workers)),
        persistent_workers=workers > 0,
    )
    test_loader = DataLoader(
        WaveformDataset(split_rows["test"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(workers)),
        persistent_workers=workers > 0,
    )

    model = SandAlertTCN(aux_feature_count=len(AUX_FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    event_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            float(event_positive_weight), dtype=torch.float32, device=device
        )
    )
    regression_loss = nn.SmoothL1Loss()

    def run_epoch(
        loader,
        train: bool,
        *,
        event_threshold: float = 0.5,
        return_predictions: bool = False,
    ):
        model.train(train)
        total = 0.0
        count = 0
        event_values = []
        probability_values = []
        ips_values = []
        ips_estimates = []
        energy_values = []
        energy_estimates = []
        for (
            waveform,
            normalized_aux,
            raw_aux,
            event,
            log_ips,
            energy,
            energy_residual,
        ) in loader:
            waveform = waveform.to(device)
            normalized_aux = normalized_aux.to(device)
            raw_aux = raw_aux.to(device)
            event = event.to(device)
            log_ips = log_ips.to(device)
            energy = energy.to(device)
            energy_residual = energy_residual.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                (
                    event_logit,
                    predicted_log_ips,
                    predicted_energy_residual,
                ) = model(waveform, normalized_aux)
                predicted_energy = (
                    raw_aux[:, AUX_FEATURE_NAMES.index(
                        "analytic_relative_energy_db"
                    )]
                    + predicted_energy_residual * energy_residual_scale_db
                )
                loss = (
                    event_loss(event_logit, event)
                    + regression_loss(predicted_log_ips, log_ips)
                    + energy_loss_weight
                    * regression_loss(
                        predicted_energy_residual, energy_residual
                    )
                )
                if train:
                    loss.backward()
                    optimizer.step()
            total += float(loss.detach()) * waveform.shape[0]
            count += waveform.shape[0]
            event_values.extend(event.detach().cpu().numpy())
            probability_values.extend(torch.sigmoid(event_logit).detach().cpu().numpy())
            ips_values.extend(torch.expm1(log_ips).detach().cpu().numpy())
            ips_estimates.extend(
                torch.expm1(torch.clamp_min(predicted_log_ips, 0.0))
                .detach()
                .cpu()
                .numpy()
            )
            energy_values.extend(energy.detach().cpu().numpy())
            energy_estimates.extend(predicted_energy.detach().cpu().numpy())
        metrics = _metrics(
            np.asarray(event_values),
            np.asarray(probability_values),
            np.asarray(ips_values),
            np.asarray(ips_estimates),
            np.asarray(energy_values),
            np.asarray(energy_estimates),
            event_threshold=event_threshold,
        )
        metrics["loss"] = total / max(count, 1)
        if return_predictions:
            return metrics, {
                "event": np.asarray(event_values),
                "probability": np.asarray(probability_values),
                "ips": np.asarray(ips_values),
                "ips_estimate": np.asarray(ips_estimates),
                "energy": np.asarray(energy_values),
                "energy_estimate": np.asarray(energy_estimates),
            }
        return metrics

    history = []
    best_state = None
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(train_loader, train=True)
        validation_metrics = run_epoch(validation_loader, train=False)
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        })
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    _, validation_predictions = run_epoch(
        validation_loader, train=False, return_predictions=True
    )
    negative_probabilities = np.sort(
        validation_predictions["probability"][
            validation_predictions["event"] < 0.5
        ]
    )
    if negative_probabilities.size:
        allowed_false_positives = int(
            np.floor(
                event_target_false_positive_rate
                * negative_probabilities.size
            )
        )
        boundary_index = max(
            negative_probabilities.size - allowed_false_positives - 1,
            0,
        )
        event_decision_threshold = float(
            np.nextafter(
                negative_probabilities[boundary_index],
                np.float32(np.inf),
            )
        )
        event_decision_threshold = max(0.5, event_decision_threshold)
    else:
        event_decision_threshold = 0.5
    calibrated_validation_metrics = _metrics(
        validation_predictions["event"],
        validation_predictions["probability"],
        validation_predictions["ips"],
        validation_predictions["ips_estimate"],
        validation_predictions["energy"],
        validation_predictions["energy_estimate"],
        event_threshold=event_decision_threshold,
    )
    test_metrics = run_epoch(
        test_loader,
        train=False,
        event_threshold=event_decision_threshold,
    )
    test_rows = split_rows["test"]
    test_waveforms = load_waveforms(
        dataset_dir, [row["index"] for row in test_rows]
    )
    test_event, test_ips, test_energy = _targets(test_rows)
    dsp_probability, dsp_ips = _dsp_predictions(test_waveforms, fs_hz)
    dsp_test_metrics = _metrics(
        test_event,
        dsp_probability,
        test_ips,
        dsp_ips,
        test_energy,
        np.zeros_like(test_energy),
    )
    promotion = _promotion_decision(test_metrics, dsp_test_metrics)

    model_id = (
        f"sandalert-tcn-v2-{dataset_id}-energy-residual-seed{seed}"
    )
    checkpoint_path = output_dir / "candidate.pt"
    torch.save({
        "model_id": model_id,
        "state_dict": model.state_dict(),
        "sample_rate_hz": fs_hz,
        "window_samples": window_samples,
        "seed": seed,
        "dataset_id": dataset_id,
        "aux_feature_names": list(AUX_FEATURE_NAMES),
        "aux_mean": aux_mean,
        "aux_std": aux_std,
        "energy_residual_scale_db": energy_residual_scale_db,
        "event_decision_threshold": event_decision_threshold,
        "event_positive_weight": event_positive_weight,
    }, checkpoint_path)

    onnx_path = output_dir / "candidate.onnx"
    onnx_status = "exported"
    try:
        wrapper = SandAlertONNXWrapper(
            model.eval(),
            aux_mean,
            aux_std,
            AUX_FEATURE_NAMES.index("analytic_relative_energy_db"),
            energy_residual_scale_db,
        )
        example = torch.zeros(1, 1, window_samples, dtype=torch.float32)
        example_aux = torch.zeros(
            1, len(AUX_FEATURE_NAMES), dtype=torch.float32
        )
        torch.onnx.export(
            wrapper,
            (example, example_aux),
            onnx_path,
            input_names=["waveform", "aux_features"],
            output_names=[
                "event_probability",
                "ips_estimate",
                "relative_energy_db",
            ],
            dynamic_axes={
                "waveform": {0: "batch"},
                "aux_features": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    except Exception as exc:
        onnx_status = f"failed:{exc}"

    manifest = {
        "model_id": model_id,
        "sample_rate_hz": fs_hz,
        "window_samples": window_samples,
        "device": "cpu",
        "threads": threads,
        "workers": workers,
        "seed": seed,
        "dataset_id": dataset_id,
        "epochs": epochs,
        "energy_mode": "analytic_residual",
        "energy_loss_weight": energy_loss_weight,
        "event_decision_threshold": event_decision_threshold,
        "event_target_false_positive_rate": (
            event_target_false_positive_rate
        ),
        "event_positive_weight": event_positive_weight,
        "calibrated_validation_metrics": calibrated_validation_metrics,
        "energy_residual_scale_db": energy_residual_scale_db,
        "aux_feature_names": list(AUX_FEATURE_NAMES),
        "aux_mean": aux_mean.tolist(),
        "aux_std": aux_std.tolist(),
        "air_velocity_optional": True,
        "particle_size_input": False,
        "onnx_status": onnx_status,
        "test_metrics": test_metrics,
        "dsp_test_metrics": dsp_test_metrics,
        "promotion": promotion,
    }
    if onnx_path.is_file():
        manifest["onnx_sha256"] = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    manifest_path = output_dir / "candidate.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "model": model_id,
        "history": history,
        "test": test_metrics,
        "calibrated_validation": calibrated_validation_metrics,
        "dsp_test": dsp_test_metrics,
        "promotion": promotion,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "onnx": str(onnx_path) if onnx_path.is_file() else None,
            "manifest": str(manifest_path),
        },
    }
    (output_dir / "tcn_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
