from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_pipeline.dataset import generate_dataset, generate_stress_dataset
from ml_pipeline.evaluation import evaluate_stress
from ml_pipeline.simulation import SimulationConfig
from ml_pipeline.training import train_feature_baseline, train_tcn


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SandAlert ML research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="generate labeled HDF5 data")
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--windows", type=int, default=900)
    simulate.add_argument("--seed", type=int, default=20260724)
    simulate.add_argument("--sample-rate", type=int, default=2_000_000)
    simulate.add_argument("--window-samples", type=int, default=8192)
    simulate.add_argument("--channels", type=int, default=1)

    stress = subparsers.add_parser(
        "simulate-stress", help="generate an OOD-only stress dataset"
    )
    stress.add_argument("--output", type=Path, required=True)
    stress.add_argument("--windows", type=int, default=600)
    stress.add_argument("--seed", type=int, default=20260725)
    stress.add_argument("--sample-rate", type=int, default=2_000_000)
    stress.add_argument("--window-samples", type=int, default=8192)
    stress.add_argument("--channels", type=int, default=1)

    baseline = subparsers.add_parser("train-baseline")
    baseline.add_argument("--dataset", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)

    tcn = subparsers.add_parser("train-tcn")
    tcn.add_argument("--dataset", type=Path, required=True)
    tcn.add_argument("--output", type=Path, required=True)
    tcn.add_argument("--epochs", type=int, default=10)
    tcn.add_argument("--batch-size", type=int, default=32)
    tcn.add_argument("--learning-rate", type=float, default=1e-3)
    tcn.add_argument("--device", choices=["cpu"], default="cpu")
    tcn.add_argument("--threads", type=int, default=32)
    tcn.add_argument("--workers", type=int, default=8)
    tcn.add_argument("--seed", type=int, default=42)
    tcn.add_argument("--energy-loss-weight", type=float, default=1.0)
    tcn.add_argument(
        "--event-target-fpr", type=float, default=0.05
    )
    tcn.add_argument(
        "--event-positive-weight", type=float, default=1.0
    )

    evaluate = subparsers.add_parser("evaluate-stress")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--feature-model", type=Path, required=True)
    evaluate.add_argument("--onnx", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--reference-dataset", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "simulate":
        result = generate_dataset(
            args.output,
            windows=args.windows,
            seed=args.seed,
            config=SimulationConfig(
                sample_rate_hz=args.sample_rate,
                window_samples=args.window_samples,
                channels=args.channels,
            ),
        )
    elif args.command == "simulate-stress":
        result = generate_stress_dataset(
            args.output,
            windows=args.windows,
            seed=args.seed,
            config=SimulationConfig(
                sample_rate_hz=args.sample_rate,
                window_samples=args.window_samples,
                channels=args.channels,
            ),
        )
    elif args.command == "train-baseline":
        result = train_feature_baseline(args.dataset, args.output)
    elif args.command == "train-tcn":
        result = train_tcn(
            args.dataset,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            threads=args.threads,
            workers=args.workers,
            seed=args.seed,
            energy_loss_weight=args.energy_loss_weight,
            event_target_false_positive_rate=args.event_target_fpr,
            event_positive_weight=args.event_positive_weight,
        )
    else:
        result = evaluate_stress(
            args.dataset,
            args.feature_model,
            args.onnx,
            args.manifest,
            args.output,
            reference_dataset_dir=args.reference_dataset,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
