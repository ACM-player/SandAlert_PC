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
  --output runtime/ml-data/software-v2 \
  --windows 900 --seed 20260724

python -m ml_pipeline simulate-stress \
  --output runtime/ml-data/stress-v2 \
  --windows 600 --seed 20260725

python -m ml_pipeline train-baseline \
  --dataset runtime/ml-data/software-v2 \
  --output runtime/ml-reports/feature-v2

nice -n 10 python -m ml_pipeline train-tcn \
  --dataset runtime/ml-data/software-v2 \
  --output runtime/ml-models/tcn-v2-weight1 \
  --device cpu --threads 32 --workers 8 \
  --energy-loss-weight 1.0 --event-target-fpr 0.05

python -m ml_pipeline evaluate-stress \
  --dataset runtime/ml-data/stress-v2 \
  --feature-model runtime/ml-reports/feature-v2/feature_baseline.joblib \
  --onnx runtime/ml-models/tcn-v2-weight1/candidate.onnx \
  --manifest runtime/ml-models/tcn-v2-weight1/candidate.manifest.json \
  --reference-dataset runtime/ml-data/software-v2 \
  --output runtime/ml-reports/stress-v2.json
```

The TCN command writes a checkpoint, an optional ONNX model, a strict runtime
manifest, and JSON metrics. Large HDF5 files and weights remain under
`runtime/`, which is ignored by Git.

## Relative-energy v2

The production-facing target is now the clean window energy relative to the
quiet-background median for the same synthetic run. The former oracle ratio of
impact-only energy to background energy remains available as
`impact_to_background_db`, but is not used as `relative_energy_db`.

The ONNX model receives a normalized waveform plus observable auxiliary
features. Its energy head predicts a correction to the analytic
window/background ratio. Air velocity is optional and is always paired with a
presence mask; particle size is not an input.

The event decision threshold is calibrated only from in-domain validation
negatives to meet the configured target false-positive rate. The frozen OOD
suite is never used to choose that threshold.

The stress dataset contains only unseen, high-severity disturbance families.
Operational metrics report both raw predictions and accepted-only predictions,
along with coverage and rejection flags. Synthetic stress results are not field
accuracy evidence.

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
Optional air velocity can be supplied by an independent `ml.context` v1 text
message. Missing or stale context is non-fatal. Excessive clipping, dropout, or
spectral OOD produces `ml.result.status="skipped"` while the DSP path continues.
