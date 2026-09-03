# IMU speed and dead-reckoning pipeline

## Run the complete pipeline

Build the lightweight wrapper image once:

```bash
./build.sh
```

Then run the complete pre-training pipeline:

```bash
./run.sh prepare
```

Project files and data are bind-mounted into `/workspace`, so editing Python
code or adding data does not require rebuilding the image.

`./run.sh` with no argument defaults to `prepare` and performs these reproducible preparation stages:

1. audit all 72 synchronized S/V pairs;
2. verify row alignment without automatic GPS-lag shifting;
3. prepare ECU-supervised archives from verified sessions;
4. validate checksums, continuous segments, split isolation, and history purge.

It deliberately does **not** train a model. Training begins only after
`artifacts/ecu_training_dataset/validation.json` reports `status: passed`.

## Train the ECU-supervised stateful model

The dataset is already prepared and validated, so the recommended full CPU run is:

```bash
./run.sh train \
  --horizon 1.0 \
  --iterations 1500 \
  --depth 8 \
  --learning-rate 0.04 \
  --early-stopping-rounds 150 \
  --threads -1
```

The 1.0-second target is the first full experiment because its velocity change
has a stronger signal-to-noise ratio than the 0.5-second target, while still
updating the state frequently. This is an explicit pre-training choice from the
target audit, not a choice made using held-out Driver B results.

To repeat preparation and then train in one command, use `./run.sh all` with
the same arguments. This is unnecessary unless the raw data or preparation
code changed.

Before a long run, the end-to-end smoke check is:

```bash
./run.sh smoke
```

It intentionally uses one truncated session per split and at most 30 trees;
its accuracy is not a model result. It verifies fitting, stateful blackout
rollouts, distance integration, reports, and plots.

Full results are written separately to `artifacts/ecu_delta_v_catboost/` and
include evaluation/deployment models, static delta-v metrics, 10/30/60/120 s
blackout metrics, a hold-last-speed baseline, per-episode/sample CSVs, feature
importance, an experiment report, and all requested plots. Existing absolute
speed and Y1 stateful artifacts are not touched.

## Train the gated causal CNN

The temporal follow-up is a separate CPU PyTorch experiment. It removes the
unverified Euler orientation transform, uses a finite causal accelerometer-bias
state, predicts delta velocity from five seconds of IMU, and selects its
confidence gate only on Driver A blackouts. Run:

```bash
./run.sh train-cnn \
  --device cuda \
  --horizon 1.0 \
  --sequence-seconds 5 \
  --bias-seconds 30 \
  --epochs 35 \
  --batch-size 512 \
  --channels 48 \
  --blocks 4 \
  --learning-rate 0.0008 \
  --early-stopping-rounds 6 \
  --workers 4
```

`run.sh` automatically passes the NVIDIA GPU into Docker when the NVIDIA
runtime is available. Set `USE_GPU=0` only when a CPU-only run is intentional.
No image rebuild is required. Use `./run.sh smoke-cnn` only to verify the
pipeline; its accuracy is not a result. Full artifacts are written to
`artifacts/ecu_delta_v_causal_cnn/`, leaving every CatBoost artifact intact.

## Train the context-calibrated unrolled INS

This model trains on complete simulated blackouts rather than independent
delta-velocity windows. A 30-second pre-blackout encoder observes phone IMU and
causally held 1 Hz speed/heading fixes to infer mounting and bias. After
initialization, a regularized phone-to-vehicle projection is frozen and its
recurrent decoder receives phone IMU-derived signals only. Speed, heading,
travelled distance, and 2D position are integrated inside the training graph
and all contribute to the loss.

First verify the code and CUDA path (the two-epoch score is not a result):

```bash
./run.sh smoke-ins --device cuda --threads 16
```

Then run the full experiment:

```bash
./run.sh train-ins \
  --device cuda \
  --context-seconds 30 \
  --bias-seconds 30 \
  --train-durations 10,30,60 \
  --blackout-durations 10,30,60,120 \
  --train-stride-seconds 10 \
  --epochs 30 \
  --batch-size 32 \
  --hidden-size 96 \
  --layers 2 \
  --learning-rate 0.0008 \
  --early-stopping-rounds 7 \
  --workers 4 \
  --threads 16
```

It uses `artifacts/ecu_training_dataset_inertial_aligned/` by default: Driver E
trains the model, Driver A selects the checkpoint, and Driver B remains the
untouched test driver. Results go to `artifacts/context_unrolled_ins/`; all
absolute-speed, CatBoost, and CNN artifacts remain unchanged.

After that model has been trained, select causal drift controls on Driver A and
apply the frozen hybrid estimator to Driver B with:

```bash
./run.sh hybrid-ins --device cuda --workers 4 --threads 16
```

The candidate set includes hold-speed, correction gains and limits, causal
acceleration smoothing, correction decay, a stationary constraint, and yaw
smoothing. Selection tables and final metrics are saved separately under
`artifacts/hybrid_drift_controlled_ins/`.

## Training-ready ECU dataset

The prepared dataset is in `artifacts/ecu_training_dataset/`:

- `sessions/`: one compressed aligned NPZ archive per verified drive;
- `sessions.csv`: archive checksums, exclusions, and split assignments;
- `segments.csv`: exact continuous source/archive row ranges;
- `training_eligibility.csv`: leakage-safe example ranges for 0.5 s and 1.0 s Δv;
- `blackout_eligibility.csv`: non-overlapping 10/30/60/120 s episode capacity;
- `config.json`: feature/reference policy, target statistics, and split configuration;
- `driver_folds.json`: leave-one-driver-out folds;
- `validation.json`: independent integrity result.

The primary driver holdout is Driver E for training, Driver A for validation,
and Driver B as the untouched test driver. A second whole-drive split and
leave-one-driver-out folds are also supplied.

## Preserve/rerun the absolute-speed baseline

The existing `artifacts/speed_model/` results are not overwritten by the
stateful experiment. Rerun that baseline explicitly with:

```bash
./run.sh python3 scripts/train_speed_model.py
```

## Fast stateful development run

```bash
./run.sh python3 scripts/train_dead_reckoning.py \
  --max-rows 15000 \
  --iterations 40 \
  --skip-refit \
  --output-dir artifacts/dead_reckoning_smoke
```

## Strict chronological evaluation

The stateful default holds out randomly selected complete 300-second blocks,
purged by the maximum feature history. It never random-splits individual rows.
For the harder test where the final part of the drive is entirely unseen, run:

```bash
./run.sh python3 scripts/train_dead_reckoning.py \
  --split-strategy chronological \
  --output-dir artifacts/dead_reckoning_chronological
```

## Stateful outputs

The stateful experiment writes to `artifacts/dead_reckoning_catboost/`. Its
`experiment_report.md` contains the main held-out blackout table, while
`target_investigation.md` documents the GPS update-rate and Δv target problem.
The evaluation and deployment models, episode/sample CSVs, leakage metadata,
feature importance, and all requested plots are kept in the same directory.

## Adding synchronized sessions

The stateful script accepts repeatable `DRIVE_ID=PATH` arguments:

```bash
./run.sh python3 scripts/train_dead_reckoning.py \
  --session 'Y1=data/path/to/S-Y1.txt' \
  --session 'X1=data/path/to/S-X1.txt'
```

With at least three sessions, use `--split-strategy drive` for complete
drive-level holdouts. Driver IDs can be encoded into the supplied drive names
and grouped explicitly when more sessions become available.

## Absolute-speed outputs

The default output directory is `artifacts/speed_model/` and contains:

- `evaluation_model.cbm`: model fitted only on the training partition and used
  for reported validation/test metrics.
- `speed_model.cbm`: deployment model refitted on all usable Y1 samples using
  the selected iteration count.
- `preprocessing.joblib`: exact feature names and rolling-window settings.
- `metrics.json`: split, model, dataset, and accuracy metadata.
- `predictions.csv`: held-out predictions used to calculate the metrics.
- `feature_importance.csv`: ranked CatBoost feature importance.
- `test_predictions.png`: target and prediction trace for the test partition.

## Important interpretation

The model inputs contain only accelerometer and gyroscope signals. Smartphone
GPS speed is converted from m/s to km/h and used only as the offline label.

Y1 contains one driver and one phone placement. Its held-out score measures
within-session performance; it is not evidence of cross-driver or cross-phone
generalization. More synchronized drivers are required for that evaluation.
