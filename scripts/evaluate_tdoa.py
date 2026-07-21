#!/usr/bin/env python3
"""Monte Carlo accuracy and robustness evaluation for SandAlert TDOA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

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
    "Method",
    "Seed",
    "SNR_dB",
    "Delay_Mode",
    "True_Angle",
    "Predicted_Angle",
    "Absolute_Error",
    "Cost",
    "Candidate_Count",
    "Peak_Confidence",
    "Selected_Peak_Strength",
    "Signal_Like",
    "Compute_Time_ms",
]


@dataclass(frozen=True)
class EvaluationMethod:
    name: str
    engine_options: dict[str, object]


def evaluation_methods(include_bandpass: bool = False) -> list[EvaluationMethod]:
    methods = [
        EvaluationMethod("legacy", {
            "angle_step": 0.5,
            "physical_lag_enabled": False,
            "peak_candidates": 1,
            "interpolate_peaks": False,
        }),
        EvaluationMethod("physical_window", {
            "angle_step": 0.5,
            "physical_lag_enabled": True,
            "peak_candidates": 1,
            "interpolate_peaks": False,
        }),
        EvaluationMethod("physical_multi_peak", {
            "angle_step": 0.5,
            "physical_lag_enabled": True,
            "peak_candidates": 3,
            "relative_peak_threshold": 0.70,
            "interpolate_peaks": False,
        }),
        EvaluationMethod("optimized", {
            "angle_step": 0.25,
            "physical_lag_enabled": True,
            "peak_candidates": 3,
            "relative_peak_threshold": 0.70,
            "interpolate_peaks": True,
        }),
    ]
    if include_bandpass:
        methods.append(EvaluationMethod("optimized_band_100_300k", {
            "angle_step": 0.25,
            "physical_lag_enabled": True,
            "peak_candidates": 3,
            "relative_peak_threshold": 0.70,
            "interpolate_peaks": True,
            "bandpass_enabled": True,
            "band_low_hz": 100_000.0,
            "band_high_hz": 300_000.0,
        }))
    return methods


def circular_absolute_error(true_angle, predicted_angle):
    """Return the shortest absolute angular distance in degrees."""
    true_values = np.asarray(true_angle, dtype=np.float64)
    predicted_values = np.asarray(predicted_angle, dtype=np.float64)
    signed_delta = (predicted_values - true_values + 180.0) % 360.0 - 180.0
    absolute_error = np.abs(signed_delta)
    if absolute_error.ndim == 0:
        return float(absolute_error)
    return absolute_error


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return one metric row per method/SNR combination."""
    rows = []
    for (method, snr_db), group in results.groupby(["Method", "SNR_dB"], sort=False):
        errors = group["Absolute_Error"].to_numpy(dtype=np.float64)
        timings = group["Compute_Time_ms"].to_numpy(dtype=np.float64)
        rows.append({
            "Method": method,
            "SNR_dB": float(snr_db),
            "Trials": len(group),
            "MAE": float(np.mean(errors)),
            "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
            "Max_Error": float(np.max(errors)),
            "P95_Error": float(np.percentile(errors, 95)),
            "Over_10deg": int(np.sum(errors > 10.0)),
            "Reliable_Rate": float(np.mean(group["Signal_Like"])) * 100.0,
            "Mean_Time_ms": float(np.mean(timings)),
            "P95_Time_ms": float(np.percentile(timings, 95)),
        })
    return pd.DataFrame(rows)


def run_evaluation(
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    snr_db: float = DEFAULT_SNR_DB,
    n_samples: int = DEFAULT_SAMPLES,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    seeds: list[int] | None = None,
    snr_values: list[float] | None = None,
    delay_mode: str = "fractional",
    pulse_center_fraction: float = 0.5,
    pulse_sigma_sec: float | None = None,
    include_bandpass: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run matched-waveform method comparisons and export a long-form CSV."""
    if trials < 1:
        raise ValueError("trials must be positive")
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10")
    seeds = list(seeds) if seeds is not None else [seed]
    snr_values = list(snr_values) if snr_values is not None else [snr_db]
    methods = evaluation_methods(include_bandpass)
    engines = {
        method.name: TDOAEngine(**method.engine_options)
        for method in methods
    }
    simulator = TDOAEngine()
    rows: list[dict[str, float | int | str | bool]] = []
    test_id = 0

    for current_seed in seeds:
        angle_rng = np.random.default_rng(current_seed)
        true_angles = angle_rng.uniform(0.0, 360.0, size=trials)
        for current_snr in snr_values:
            # The waveform generator uses NumPy's legacy random stream for noise.
            np.random.seed(current_seed)
            for true_angle in true_angles:
                test_id += 1
                waveform, _ = simulator.generate_mock_array_data(
                    n_samples=n_samples,
                    true_angle_deg=float(true_angle),
                    snr_db=float(current_snr),
                    delay_mode=delay_mode,
                    pulse_center_fraction=pulse_center_fraction,
                    pulse_sigma_sec=pulse_sigma_sec,
                )
                for method in methods:
                    started = time.perf_counter()
                    result = engines[method.name].compute_doa_detailed(waveform)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    predicted_angle = float(result.angle_deg % 360.0)
                    diagnostics = result.diagnostics
                    rows.append({
                        "Test_ID": test_id,
                        "Method": method.name,
                        "Seed": current_seed,
                        "SNR_dB": float(current_snr),
                        "Delay_Mode": delay_mode,
                        "True_Angle": float(true_angle),
                        "Predicted_Angle": predicted_angle,
                        "Absolute_Error": circular_absolute_error(
                            true_angle, predicted_angle
                        ),
                        "Cost": float(np.min(result.cost_values)),
                        "Candidate_Count": diagnostics.candidate_count,
                        "Peak_Confidence": diagnostics.peak_confidence,
                        "Selected_Peak_Strength": diagnostics.selected_peak_strength,
                        "Signal_Like": diagnostics.signal_like,
                        "Compute_Time_ms": elapsed_ms,
                    })

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    summary = summarize_results(results)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\nSandAlert TDOA Evaluation Summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nCSV: {output_path}")

    optimized = results[results["Method"] == "optimized"]
    errors = optimized["Absolute_Error"].to_numpy(dtype=np.float64)
    metrics = {
        "MAE": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "Max_Error": float(np.max(errors)),
    }
    return results, metrics


def evaluate_noise_rejection(
    trials: int = 200,
    seed: int = DEFAULT_SEED,
    n_samples: int = DEFAULT_SAMPLES,
) -> tuple[int, float]:
    """Measure how often pure independent channel noise passes peak gating."""
    rng = np.random.default_rng(seed)
    engine = TDOAEngine()
    accepted = 0
    for _ in range(trials):
        noise = rng.standard_normal((engine.m, n_samples))
        result = engine.compute_doa_detailed(noise)
        accepted += int(result.diagnostics.signal_like)
    rate = accepted / trials
    print(
        f"Pure-noise reliability gate: {accepted}/{trials} "
        f"accepted ({rate * 100:.3f}%)"
    )
    return accepted, rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SandAlert TDOA variants with simulated waveforms."
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--snr-db", type=float, nargs="+", default=[DEFAULT_SNR_DB])
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--delay-mode", choices=("integer", "fractional"), default="fractional"
    )
    parser.add_argument("--pulse-center-fraction", type=float, default=0.5)
    parser.add_argument("--pulse-sigma-sec", type=float)
    parser.add_argument("--include-bandpass", action="store_true")
    parser.add_argument("--noise-only-trials", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    return args


def main() -> None:
    args = parse_args()
    run_evaluation(
        trials=args.trials,
        seeds=args.seeds,
        snr_values=args.snr_db,
        n_samples=args.samples,
        output_path=args.output,
        delay_mode=args.delay_mode,
        pulse_center_fraction=args.pulse_center_fraction,
        pulse_sigma_sec=args.pulse_sigma_sec,
        include_bandpass=args.include_bandpass,
    )
    if args.noise_only_trials:
        evaluate_noise_rejection(
            trials=args.noise_only_trials,
            seed=args.seeds[0],
            n_samples=args.samples,
        )


if __name__ == "__main__":
    main()
