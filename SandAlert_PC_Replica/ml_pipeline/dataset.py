from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from ml_pipeline.simulation import (
    MECHANICAL_CALIBRATION_SCENARIOS,
    OOD_SCENARIOS,
    SCENARIOS,
    SimulationConfig,
    SyntheticWindow,
    generate_window,
)


def split_for_group(group_id: str) -> str:
    digest = hashlib.blake2b(group_id.encode("utf-8"), digest_size=8).digest()
    fraction = int.from_bytes(digest, "big") / float(2**64)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "validation"
    return "test"


def generate_dataset(
    output_dir: str | Path,
    windows: int,
    seed: int,
    config: SimulationConfig,
    scenarios: tuple[str, ...] = SCENARIOS,
    group_size: int = 5,
    severity_range: tuple[float, float] = (0.15, 0.55),
    context_missing_probability: float = 0.25,
    split_override: str | None = None,
    dataset_role: str = "training",
    group_namespace: str = "software",
) -> dict:
    if windows < len(scenarios):
        raise ValueError("window count must cover every requested scenario")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hdf5_path = output / "waveforms.h5"
    manifest_path = output / "manifest.jsonl"

    if not 0.0 <= severity_range[0] <= severity_range[1] <= 1.0:
        raise ValueError("severity range must remain within [0, 1]")
    if not 0.0 <= context_missing_probability <= 1.0:
        raise ValueError("context missing probability must remain within [0, 1]")

    generated: list[tuple[str, str, SyntheticWindow, bool, int]] = []
    group_context: dict[str, tuple[float, float]] = {}
    for index in range(windows):
        scenario = scenarios[index % len(scenarios)]
        scenario_iteration = index // len(scenarios)
        group_index = scenario_iteration // max(group_size, 1)
        group_id = f"{group_namespace}:{scenario}-{group_index:06d}"
        run_position = scenario_iteration % max(group_size, 1)
        split = split_override or split_for_group(group_id)
        if group_id not in group_context:
            digest = hashlib.blake2b(
                f"{seed}:{group_id}".encode("utf-8"), digest_size=8
            ).digest()
            context_rng = np.random.default_rng(int.from_bytes(digest, "big"))
            severity = float(
                context_rng.uniform(severity_range[0], severity_range[1])
            )
            air_velocity = float(
                context_rng.uniform(
                    config.air_velocity_min_m_s,
                    config.air_velocity_max_m_s,
                )
            )
            group_context[group_id] = (severity, air_velocity)
        severity, air_velocity = group_context[group_id]

        if scenario == "background" or run_position < 2:
            forced_count = 0
        else:
            forced_count = 1 + (run_position % 3)
        item = generate_window(
            config,
            seed=seed + index,
            scenario=scenario,
            force_event_count=forced_count,
            severity=severity,
            air_velocity_m_s=air_velocity,
        )
        available_rng = np.random.default_rng(seed * 1_000_003 + index)
        air_velocity_available = (
            available_rng.random() >= context_missing_probability
        )
        generated.append(
            (group_id, split, item, air_velocity_available, run_position)
        )

    references: dict[str, tuple[float, float]] = {}
    for group_id in group_context:
        members = [
            item
            for candidate_group, _, item, _, _ in generated
            if candidate_group == group_id
        ]
        quiet = [item for item in members if item.labels["event_count"] == 0]
        source = quiet or members
        clean_reference = float(
            np.median([item.labels["background_energy"] for item in source])
        )
        observed_reference = clean_reference
        references[group_id] = (
            max(clean_reference, 1e-12),
            max(observed_reference, 1e-12),
        )

    counts = {"train": 0, "validation": 0, "test": 0}
    if split_override is not None:
        counts = {split_override: 0}
    with h5py.File(hdf5_path, "w") as handle, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest:
        waveform_ds = handle.create_dataset(
            "waveforms",
            shape=(windows, config.channels, config.window_samples),
            dtype="float32",
            chunks=(1, config.channels, config.window_samples),
            compression="gzip",
            compression_opts=4,
        )
        handle.attrs["simulation_config"] = json.dumps(
            asdict(config), ensure_ascii=False, sort_keys=True
        )
        handle.attrs["base_seed"] = int(seed)
        for index, (
            group_id,
            split,
            item,
            air_velocity_available,
            run_position,
        ) in enumerate(generated):
            clean_reference, observed_reference = references[group_id]
            item.labels["quiet_background_energy"] = clean_reference
            item.labels["observed_quiet_background_energy"] = observed_reference
            item.labels["relative_energy_db"] = float(
                10.0
                * np.log10(
                    (item.labels["window_energy"] + 1e-12)
                    / (clean_reference + 1e-12)
                )
            )
            item.labels["analytic_relative_energy_db"] = float(
                10.0
                * np.log10(
                    (item.labels["observed_window_energy"] + 1e-12)
                    / (observed_reference + 1e-12)
                )
            )
            item.labels["energy_residual_db"] = (
                item.labels["relative_energy_db"]
                - item.labels["analytic_relative_energy_db"]
            )
            item.labels["air_velocity_available"] = air_velocity_available
            is_mechanical_calibration = (
                scenario in MECHANICAL_CALIBRATION_SCENARIOS
            )
            item.labels["dataset_role"] = dataset_role
            item.labels["mechanical_interference"] = (
                is_mechanical_calibration
            )
            item.labels["mechanical_reject_target"] = bool(
                is_mechanical_calibration
                and item.labels["event_count"] == 0
            )
            waveform_ds[index] = item.waveform
            record = {
                "index": index,
                "group_id": group_id,
                "run_id": group_id,
                "run_position": run_position,
                "split": split,
                **{key: value for key, value in item.labels.items() if key != "config"},
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split] += 1

    summary = {
        "waveforms": str(hdf5_path),
        "manifest": str(manifest_path),
        "windows": windows,
        "splits": counts,
        "seed": seed,
        "config": asdict(config),
        "scenarios": list(scenarios),
        "severity_range": list(severity_range),
        "context_missing_probability": context_missing_probability,
        "dataset_role": dataset_role,
        "group_namespace": group_namespace,
        "relative_energy_definition": (
            "10*log10((clean_window_energy+eps)/"
            "(quiet_background_median_energy+eps))"
        ),
        "analytic_energy_definition": (
            "10*log10((observed_window_energy+eps)/"
            "(observed_quiet_background_energy+eps))"
        ),
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def generate_stress_dataset(
    output_dir: str | Path,
    windows: int,
    seed: int,
    config: SimulationConfig,
    *,
    group_size: int = 5,
) -> dict:
    return generate_dataset(
        output_dir,
        windows=windows,
        seed=seed,
        config=config,
        scenarios=OOD_SCENARIOS,
        group_size=group_size,
        severity_range=(0.75, 1.0),
        context_missing_probability=0.5,
        split_override="test",
        dataset_role="frozen_ood_evaluation",
        group_namespace="frozen-ood",
    )


def generate_mechanical_calibration_dataset(
    output_dir: str | Path,
    windows: int,
    seed: int,
    config: SimulationConfig,
    *,
    group_size: int = 5,
) -> dict:
    """Generate a tuning-only mechanical hard-negative dataset.

    The namespace, role and seed family are intentionally separate from the
    frozen OOD suite. Mechanical-only windows are rejection targets, while
    windows containing simulated sand impacts remain positive non-reject
    controls so that a reject-all solution is penalized.
    """
    return generate_dataset(
        output_dir,
        windows=windows,
        seed=seed,
        config=config,
        scenarios=MECHANICAL_CALIBRATION_SCENARIOS,
        group_size=group_size,
        severity_range=(0.35, 0.9),
        context_missing_probability=0.5,
        dataset_role="mechanical_calibration_only",
        group_namespace="mechanical-calibration",
    )


def load_manifest(dataset_dir: str | Path) -> list[dict]:
    path = Path(dataset_dir) / "manifest.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_waveforms(
    dataset_dir: str | Path, indices: list[int] | np.ndarray
) -> np.ndarray:
    with h5py.File(Path(dataset_dir) / "waveforms.h5", "r") as handle:
        return np.asarray(handle["waveforms"][np.asarray(indices, dtype=int)])


def dataset_fingerprint(dataset_dir: str | Path) -> str:
    """Hash immutable dataset content used by candidate locks and reports."""
    dataset_dir = Path(dataset_dir)
    digest = hashlib.sha256()
    for name in ("dataset_summary.json", "manifest.jsonl", "waveforms.h5"):
        path = dataset_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def assert_calibration_isolated(
    calibration_dir: str | Path,
    frozen_ood_dir: str | Path,
) -> dict:
    """Prove that calibration and frozen OOD datasets are distinct.

    This check is only for the final audit/evaluation stage. Training commands
    never accept a frozen dataset path and therefore cannot call this function.
    """
    calibration_dir = Path(calibration_dir).resolve()
    frozen_ood_dir = Path(frozen_ood_dir).resolve()
    if calibration_dir == frozen_ood_dir:
        raise ValueError("calibration and frozen OOD paths must differ")
    calibration_summary = json.loads(
        (calibration_dir / "dataset_summary.json").read_text(encoding="utf-8")
    )
    frozen_summary = json.loads(
        (frozen_ood_dir / "dataset_summary.json").read_text(encoding="utf-8")
    )
    if calibration_summary.get("dataset_role") != "mechanical_calibration_only":
        raise ValueError("calibration dataset role is not tuning-only")
    frozen_role = frozen_summary.get("dataset_role")
    if frozen_role != "frozen_ood_evaluation":
        legacy_signature = (
            set(frozen_summary.get("scenarios", [])) == set(OOD_SCENARIOS)
            and set(frozen_summary.get("splits", {})) == {"test"}
        )
        if not legacy_signature:
            raise ValueError("frozen dataset role is not evaluation-only")
        frozen_role = "legacy_frozen_ood_signature"
    calibration_rows = load_manifest(calibration_dir)
    frozen_rows = load_manifest(frozen_ood_dir)
    calibration_groups = {row["group_id"] for row in calibration_rows}
    frozen_groups = {row["group_id"] for row in frozen_rows}
    group_overlap = calibration_groups & frozen_groups
    if group_overlap:
        raise ValueError("calibration and frozen datasets share group ids")
    calibration_seeds = {int(row["seed"]) for row in calibration_rows}
    frozen_seeds = {int(row["seed"]) for row in frozen_rows}
    seed_overlap = calibration_seeds & frozen_seeds
    if seed_overlap:
        raise ValueError("calibration and frozen datasets share window seeds")
    return {
        "calibration_fingerprint": dataset_fingerprint(calibration_dir),
        "frozen_ood_fingerprint": dataset_fingerprint(frozen_ood_dir),
        "calibration_groups": len(calibration_groups),
        "frozen_ood_groups": len(frozen_groups),
        "group_overlap": 0,
        "seed_overlap": 0,
        "frozen_role_verification": frozen_role,
    }


def merge_adversarial_training_datasets(
    base_dataset_dir: str | Path,
    calibration_dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Create a training artifact without ever reading the frozen OOD suite."""
    base_dataset_dir = Path(base_dataset_dir)
    calibration_dataset_dir = Path(calibration_dataset_dir)
    output_dir = Path(output_dir)
    calibration_summary = json.loads(
        (calibration_dataset_dir / "dataset_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if calibration_summary.get("dataset_role") != "mechanical_calibration_only":
        raise ValueError("only a mechanical calibration dataset may be merged")
    base_summary = json.loads(
        (base_dataset_dir / "dataset_summary.json").read_text(encoding="utf-8")
    )
    if base_summary.get("dataset_role") == "frozen_ood_evaluation":
        raise ValueError("frozen OOD data cannot enter adversarial training")
    base_records = load_manifest(base_dataset_dir)
    calibration_records = load_manifest(calibration_dataset_dir)
    if not base_records or not calibration_records:
        raise ValueError("both source datasets must contain records")
    expected_shape = (
        int(base_records[0]["channels"]),
        int(base_records[0]["window_samples"]),
    )
    calibration_shape = (
        int(calibration_records[0]["channels"]),
        int(calibration_records[0]["window_samples"]),
    )
    if expected_shape != calibration_shape:
        raise ValueError("source waveform shapes must match")
    if int(base_records[0]["sample_rate_hz"]) != int(
        calibration_records[0]["sample_rate_hz"]
    ):
        raise ValueError("source sample rates must match")
    base_groups = {row["group_id"] for row in base_records}
    calibration_groups = {row["group_id"] for row in calibration_records}
    if base_groups & calibration_groups:
        raise ValueError("source datasets share group ids")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = (
        ("base", base_dataset_dir, base_records),
        ("mechanical_calibration", calibration_dataset_dir, calibration_records),
    )
    total = len(base_records) + len(calibration_records)
    counts: dict[str, int] = {}
    with h5py.File(output_dir / "waveforms.h5", "w") as output_h5, (
        output_dir / "manifest.jsonl"
    ).open("w", encoding="utf-8") as output_manifest:
        waveform_ds = output_h5.create_dataset(
            "waveforms",
            shape=(total, *expected_shape),
            dtype="float32",
            chunks=(1, *expected_shape),
            compression="gzip",
            compression_opts=4,
        )
        output_h5.attrs["simulation_config"] = json.dumps(
            base_summary["config"], ensure_ascii=False, sort_keys=True
        )
        output_h5.attrs["dataset_role"] = "adversarial_training"
        output_index = 0
        for source_name, source_dir, records in combined:
            with h5py.File(source_dir / "waveforms.h5", "r") as source_h5:
                for row in records:
                    waveform_ds[output_index] = source_h5["waveforms"][
                        int(row["index"])
                    ]
                    merged = dict(row)
                    merged["index"] = output_index
                    merged["source_dataset"] = source_name
                    merged["source_index"] = int(row["index"])
                    merged.setdefault("mechanical_reject_target", False)
                    merged.setdefault("mechanical_interference", False)
                    output_manifest.write(
                        json.dumps(merged, ensure_ascii=False) + "\n"
                    )
                    counts[merged["split"]] = counts.get(merged["split"], 0) + 1
                    output_index += 1
    summary = {
        "dataset_role": "adversarial_training",
        "windows": total,
        "splits": counts,
        "config": base_summary["config"],
        "sources": {
            "base": {
                "path": str(base_dataset_dir),
                "fingerprint": dataset_fingerprint(base_dataset_dir),
                "windows": len(base_records),
            },
            "mechanical_calibration": {
                "path": str(calibration_dataset_dir),
                "fingerprint": dataset_fingerprint(calibration_dataset_dir),
                "windows": len(calibration_records),
            },
        },
        "frozen_ood_used": False,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
