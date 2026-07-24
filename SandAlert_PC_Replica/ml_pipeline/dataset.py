from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from ml_pipeline.simulation import SCENARIOS, SimulationConfig, generate_window


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
) -> dict:
    if windows < len(scenarios):
        raise ValueError("window count must cover every requested scenario")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hdf5_path = output / "waveforms.h5"
    manifest_path = output / "manifest.jsonl"

    counts = {"train": 0, "validation": 0, "test": 0}
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
        for index in range(windows):
            scenario = scenarios[index % len(scenarios)]
            scenario_iteration = index // len(scenarios)
            group_id = (
                f"{scenario}-{scenario_iteration // max(group_size, 1):06d}"
            )
            split = split_for_group(group_id)
            # Alternate explicit negatives and positives while retaining stress presets.
            if scenario == "background":
                forced_count = 0
            elif index % 2:
                forced_count = 1 + (index % 3)
            else:
                forced_count = 0
            item = generate_window(
                config,
                seed=seed + index,
                scenario=scenario,
                force_event_count=forced_count,
            )
            waveform_ds[index] = item.waveform
            record = {
                "index": index,
                "group_id": group_id,
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
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def load_manifest(dataset_dir: str | Path) -> list[dict]:
    path = Path(dataset_dir) / "manifest.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_waveforms(
    dataset_dir: str | Path, indices: list[int] | np.ndarray
) -> np.ndarray:
    with h5py.File(Path(dataset_dir) / "waveforms.h5", "r") as handle:
        return np.asarray(handle["waveforms"][np.asarray(indices, dtype=int)])
