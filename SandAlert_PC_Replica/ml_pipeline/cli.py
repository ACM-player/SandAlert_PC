from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_pipeline.dataset import generate_dataset
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
    elif args.command == "train-baseline":
        result = train_feature_baseline(args.dataset, args.output)
    else:
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
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
