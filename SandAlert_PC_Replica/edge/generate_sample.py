from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.tdoa_engine import TDOAEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate four-channel TDOA replay data")
    parser.add_argument("output", type=Path)
    parser.add_argument("--angle", type=float, default=60.0)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--snr", type=float, default=10.0)
    args = parser.parse_args()
    engine = TDOAEngine(fs=2_000_000, radius=0.05, c=1500.0)
    data, _ = engine.generate_mock_array_data(
        n_samples=args.samples,
        true_angle_deg=args.angle,
        snr_db=args.snr,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, data.T, delimiter=",", fmt="%.7f")
    print(f"generated {data.shape[1]} samples at {args.angle:.1f} deg: {args.output}")


if __name__ == "__main__":
    main()
