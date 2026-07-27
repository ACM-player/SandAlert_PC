from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_pipeline.dataset import (
    generate_dataset,
    generate_mechanical_calibration_dataset,
    generate_stress_dataset,
    merge_adversarial_training_datasets,
)
from ml_pipeline.evaluation import evaluate_stress
from ml_pipeline.mechanical_evaluation import (
    create_mechanical_candidate_lock,
    evaluate_locked_mechanical_candidates,
)
from ml_pipeline.mechanical_rejector import train_mechanical_rejector
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

    mechanical = subparsers.add_parser(
        "simulate-mechanical-calibration",
        help="generate tuning-only mechanical hard negatives and controls",
    )
    mechanical.add_argument("--output", type=Path, required=True)
    mechanical.add_argument("--windows", type=int, default=800)
    mechanical.add_argument("--seed", type=int, default=20270001)
    mechanical.add_argument("--sample-rate", type=int, default=2_000_000)
    mechanical.add_argument("--window-samples", type=int, default=8192)
    mechanical.add_argument("--channels", type=int, default=1)

    merge = subparsers.add_parser(
        "merge-adversarial",
        help="merge base and calibration data without reading frozen OOD",
    )
    merge.add_argument("--base-dataset", type=Path, required=True)
    merge.add_argument("--calibration-dataset", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

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
    tcn.add_argument("--quality-head", action="store_true")
    tcn.add_argument("--quality-loss-weight", type=float, default=1.0)
    tcn.add_argument("--hard-negative-event-weight", type=float, default=2.0)
    tcn.add_argument("--finetune-checkpoint", type=Path)

    rejector = subparsers.add_parser("train-mechanical-rejector")
    rejector.add_argument("--dataset", type=Path, required=True)
    rejector.add_argument("--output", type=Path, required=True)
    rejector.add_argument("--seed", type=int, default=42)
    rejector.add_argument(
        "--max-positive-false-rejection", type=float, default=0.05
    )
    rejector.add_argument(
        "--max-ordinary-negative-false-rejection",
        type=float,
        default=0.05,
    )
    rejector.add_argument("--rbf-c", type=float, default=4.0)

    evaluate = subparsers.add_parser("evaluate-stress")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--feature-model", type=Path, required=True)
    evaluate.add_argument("--onnx", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--reference-dataset", type=Path)

    lock = subparsers.add_parser(
        "lock-mechanical-candidates",
        help="seal Route A/B artifacts before frozen evaluation",
    )
    lock.add_argument("--route-a-onnx", type=Path, required=True)
    lock.add_argument("--route-a-manifest", type=Path, required=True)
    lock.add_argument("--route-b-onnx", type=Path, required=True)
    lock.add_argument("--route-b-manifest", type=Path, required=True)
    lock.add_argument("--route-b-rejector", type=Path, required=True)
    lock.add_argument("--calibration-dataset", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)

    mechanical_evaluate = subparsers.add_parser(
        "evaluate-mechanical-frozen",
        help="one-shot evaluation of locked candidates on frozen OOD",
    )
    mechanical_evaluate.add_argument(
        "--frozen-dataset", type=Path, required=True
    )
    mechanical_evaluate.add_argument(
        "--calibration-dataset", type=Path, required=True
    )
    mechanical_evaluate.add_argument(
        "--feature-model", type=Path, required=True
    )
    mechanical_evaluate.add_argument(
        "--candidate-lock", type=Path, required=True
    )
    mechanical_evaluate.add_argument("--output", type=Path, required=True)
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
    elif args.command == "simulate-mechanical-calibration":
        result = generate_mechanical_calibration_dataset(
            args.output,
            windows=args.windows,
            seed=args.seed,
            config=SimulationConfig(
                sample_rate_hz=args.sample_rate,
                window_samples=args.window_samples,
                channels=args.channels,
            ),
        )
    elif args.command == "merge-adversarial":
        result = merge_adversarial_training_datasets(
            args.base_dataset,
            args.calibration_dataset,
            args.output,
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
            quality_head_enabled=args.quality_head,
            quality_loss_weight=args.quality_loss_weight,
            hard_negative_event_weight=args.hard_negative_event_weight,
            finetune_checkpoint=args.finetune_checkpoint,
        )
    elif args.command == "train-mechanical-rejector":
        result = train_mechanical_rejector(
            args.dataset,
            args.output,
            seed=args.seed,
            max_positive_false_rejection_rate=(
                args.max_positive_false_rejection
            ),
            max_ordinary_negative_false_rejection_rate=(
                args.max_ordinary_negative_false_rejection
            ),
            rbf_c=args.rbf_c,
        )
    elif args.command == "evaluate-stress":
        result = evaluate_stress(
            args.dataset,
            args.feature_model,
            args.onnx,
            args.manifest,
            args.output,
            reference_dataset_dir=args.reference_dataset,
        )
    elif args.command == "lock-mechanical-candidates":
        result = create_mechanical_candidate_lock(
            args.route_a_onnx,
            args.route_a_manifest,
            args.route_b_onnx,
            args.route_b_manifest,
            args.route_b_rejector,
            args.calibration_dataset,
            args.output,
        )
    else:
        result = evaluate_locked_mechanical_candidates(
            args.frozen_dataset,
            args.calibration_dataset,
            args.feature_model,
            args.candidate_lock,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
