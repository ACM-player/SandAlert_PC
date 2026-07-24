from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from ml_pipeline.simulation import (
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
        group_id = f"{scenario}-{group_index:06d}"
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
