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


def _metrics(y_true_event, probability, y_true_ips, ips_estimate, y_true_energy, energy):
    from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error

    predicted_event = np.asarray(probability) >= 0.5
    return {
        "event_f1": float(f1_score(y_true_event, predicted_event, zero_division=0)),
        "event_average_precision": float(
            average_precision_score(y_true_event, probability)
        ),
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
) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from ml_pipeline.models import SandAlertONNXWrapper, SandAlertTCN

    if device != "cpu":
        raise ValueError("this phase is intentionally restricted to device='cpu'")
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
    fs_hz = int(records[0]["sample_rate_hz"])
    window_samples = int(records[0]["window_samples"])
    hdf5_path = dataset_dir / "waveforms.h5"

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
            return (
                torch.from_numpy(waveform[None, :].copy()),
                torch.tensor(float(row["event_count"] > 0), dtype=torch.float32),
                torch.tensor(np.log1p(row["ips_hz"]), dtype=torch.float32),
                torch.tensor(row["relative_energy_db"], dtype=torch.float32),
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

    model = SandAlertTCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    event_loss = nn.BCEWithLogitsLoss()
    regression_loss = nn.SmoothL1Loss()

    def run_epoch(loader, train: bool):
        model.train(train)
        total = 0.0
        count = 0
        event_values = []
        probability_values = []
        ips_values = []
        ips_estimates = []
        energy_values = []
        energy_estimates = []
        for waveform, event, log_ips, energy in loader:
            waveform = waveform.to(device)
            event = event.to(device)
            log_ips = log_ips.to(device)
            energy = energy.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                event_logit, predicted_log_ips, predicted_energy = model(waveform)
                loss = (
                    event_loss(event_logit, event)
                    + regression_loss(predicted_log_ips, log_ips)
                    + 0.5 * regression_loss(predicted_energy, energy)
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
        )
        metrics["loss"] = total / max(count, 1)
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
    test_metrics = run_epoch(test_loader, train=False)
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

    model_id = f"sandalert-tcn-v1-seed{seed}"
    checkpoint_path = output_dir / "candidate.pt"
    torch.save({
        "model_id": model_id,
        "state_dict": model.state_dict(),
        "sample_rate_hz": fs_hz,
        "window_samples": window_samples,
        "seed": seed,
    }, checkpoint_path)

    onnx_path = output_dir / "candidate.onnx"
    onnx_status = "exported"
    try:
        wrapper = SandAlertONNXWrapper(model.eval())
        example = torch.zeros(1, 1, window_samples, dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            example,
            onnx_path,
            input_names=["waveform"],
            output_names=[
                "event_probability",
                "ips_estimate",
                "relative_energy_db",
            ],
            dynamic_axes={"waveform": {0: "batch"}},
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
        "epochs": epochs,
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
