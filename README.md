# DTAF

This repository is a compact, executable PyTorch project for DTAF. The original
benchmark framework, JSON configs, experiment shell scripts, dashboards, and
figures have been removed. The project now has one runtime entry point:
`run.py`.

## Install

```bash
pip install -r requirements.txt
```

## Data Format

Datasets follow the iTransformer CSV convention:

```csv
date,OT,HUFL,HULL,MUFL,MULL
2016-07-01 00:00:00,1.0,2.0,3.0,4.0,5.0
2016-07-01 01:00:00,1.1,2.1,3.1,4.1,5.1
```

Rules:

- The first column must be `date`.
- All remaining columns must be numeric variables.
- `--features S` trains on only `--target`.
- `--features M` trains and evaluates all variables.
- `--features MS` trains with all variables but evaluates only `--target`.

Place data under `./dataset` or pass another `--root_path`.

## Train And Test

```bash
python run.py \
  --root_path ./dataset \
  --data_path ETTh1.csv \
  --features M \
  --target OT \
  --seq_len 96 \
  --pred_len 96 \
  --train_epochs 10 \
  --batch_size 32
```

The best checkpoint is saved to `./checkpoints/dtaf_best.pt`, and metrics are
saved to `./checkpoints/metrics.json`. Validation and test MAE/MSE are reported
on the standardized scale used for training, matching the usual benchmark
reporting convention.

To also write test-window predictions:

```bash
python run.py --root_path ./dataset --data_path ETTh1.csv --save_predictions
```

## Evaluate Or Predict From A Checkpoint

```bash
python run.py --mode test --root_path ./dataset --data_path ETTh1.csv
python run.py --mode predict --root_path ./dataset --data_path ETTh1.csv
```

`predict` writes the next `--pred_len` steps to
`./checkpoints/future_predictions.csv`.

## Files

- `run.py`: CLI, data loading, train/validation/test loop, checkpointing.
- `dtaf/model.py`: standalone DTAF model implementation.
