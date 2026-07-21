#!/usr/bin/env python3
"""Monte Carlo accuracy evaluation for the SandAlert TDOA engine."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "SandAlert_PC_Replica"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from core.tdoa_engine import TDOAEngine  # noqa: E402


DEFAULT_TRIALS = 200
DEFAULT_SEED = 42
DEFAULT_SNR_DB = 10.0
DEFAULT_SAMPLES = 2000
DEFAULT_OUTPUT = PROJECT_ROOT / "tdoa_evaluation_results.csv"
RESULT_COLUMNS = [
    "Test_ID",
    "True_Angle",
    "Predicted_Angle",
    "Absolute_Error",
    "Cost",
]


def circular_absolute_error(true_angle, predicted_angle):
    """Return the shortest absolute angular distance in degrees."""
    true_values = np.asarray(true_angle, dtype=np.float64)
    predicted_values = np.asarray(predicted_angle, dtype=np.float64)
    signed_delta = (predicted_values - true_values + 180.0) % 360.0 - 180.0
    absolute_error = np.abs(signed_delta)
    if absolute_error.ndim == 0:
        return float(absolute_error)
    return absolute_error


def run_evaluation(
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    snr_db: float = DEFAULT_SNR_DB,
    n_samples: int = DEFAULT_SAMPLES,
    output_path: Path = DEFAULT_OUTPUT,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate simulated waveforms, run TDOA, export rows, and return metrics."""
    if trials < 1:
        raise ValueError("trials must be positive")
    if n_samples < 2000:
        raise ValueError("n_samples must be at least 2000")

    rng = np.random.default_rng(seed)
    # TDOAEngine.generate_mock_array_data currently uses NumPy's legacy RNG.
    np.random.seed(seed)
    true_angles = rng.uniform(0.0, 360.0, size=trials)
    engine = TDOAEngine()
    rows: list[dict[str, float | int]] = []

    started = time.perf_counter()
    progress_step = max(1, trials // 10)
    for index, true_angle in enumerate(true_angles, start=1):
        waveform, _ = engine.generate_mock_array_data(
            n_samples=n_samples,
            true_angle_deg=float(true_angle),
            snr_db=snr_db,
        )
        predicted_angle, _, _, costs = engine.compute_doa(waveform)
        predicted_angle = float(predicted_angle % 360.0)
        absolute_error = circular_absolute_error(true_angle, predicted_angle)
        cost = float(np.min(costs))
        rows.append({
            "Test_ID": index,
            "True_Angle": float(true_angle),
            "Predicted_Angle": predicted_angle,
            "Absolute_Error": absolute_error,
            "Cost": cost,
        })
        if index % progress_step == 0 or index == trials:
            print(f"Progress: {index}/{trials}")

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    errors = results["Absolute_Error"].to_numpy(dtype=np.float64)
    metrics = {
        "MAE": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "Max_Error": float(np.max(errors)),
    }

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    elapsed = time.perf_counter() - started
    print("\nSandAlert TDOA Evaluation Summary")
    print(f"Trials: {trials}")
    print(f"Seed: {seed}")
    print(f"SNR: {snr_db:.1f} dB")
    print(f"MAE: {metrics['MAE']:.6f} deg")
    print(f"RMSE: {metrics['RMSE']:.6f} deg")
    print(f"Max Error: {metrics['Max_Error']:.6f} deg")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"CSV: {output_path}")
    return results, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SandAlert TDOA accuracy with simulated waveforms."
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--snr-db", type=float, default=DEFAULT_SNR_DB)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.trials < DEFAULT_TRIALS:
        parser.error(f"--trials must be at least {DEFAULT_TRIALS}")
    return args


def main() -> None:
    args = parse_args()
    run_evaluation(
        trials=args.trials,
        seed=args.seed,
        snr_db=args.snr_db,
        n_samples=args.samples,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
