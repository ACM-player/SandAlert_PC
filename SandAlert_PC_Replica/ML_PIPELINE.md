# SandAlert ML research pipeline

This package is deliberately separate from the production DSP path. Synthetic
results are research evidence only and must not be described as calibrated
field accuracy.

## Environments

- Local checks: `conda activate playground`
- Server jobs: `/home/lks/miniconda3/envs/core/bin/python`
- Current server training device: CPU. Do not install NVIDIA drivers or reboot
  the server during this phase.
- `requirements/ml.txt` pins the official Linux CPU-only PyTorch wheel. A
  dependency dry-run must contain no `cuda-*` or `nvidia-*` packages.

## Commands

Run commands from `SandAlert_PC_Replica`:

```bash
python -m ml_pipeline simulate \
  --output runtime/ml-data/software-v1 \
  --windows 900 --seed 20260724

python -m ml_pipeline train-baseline \
  --dataset runtime/ml-data/software-v1 \
  --output runtime/ml-reports/feature-v1

nice -n 10 python -m ml_pipeline train-tcn \
  --dataset runtime/ml-data/software-v1 \
  --output runtime/ml-models/tcn-v1 \
  --device cpu --threads 32 --workers 8
```

The TCN command writes a checkpoint, an optional ONNX model, a strict runtime
manifest, and JSON metrics. Large HDF5 files and weights remain under
`runtime/`, which is ignored by Git.

## Shadow deployment

Enable the optional server path only after an ONNX artifact exists:

```toml
[server]
ml_enabled = true
ml_model_path = "./runtime/ml-models/tcn-v1/candidate.onnx"
ml_manifest_path = "./runtime/ml-models/tcn-v1/candidate.manifest.json"
ml_max_emit_hz = 20.0
```

`ml.result` never changes `analysis.result`, mass-rate alarms, or relay output.
