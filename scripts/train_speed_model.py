#!/usr/bin/env python3
"""Train an IMU-only vehicle-speed model on the clean Y1 phone segment.

GPS speed is used only as an offline supervision label. GPS position, speed,
orientation, timestamps, and vehicle/ECU fields are never model features.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_INPUT = Path(
    "data/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/Y (Driver D)/Y1/S-Y1.txt"
)
DEFAULT_OUTPUT = Path("artifacts/speed_model")
IMU_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-strategy",
        choices=["blocked-random", "chronological"],
        default="blocked-random",
        help="Use whole holdout blocks or a strict final-time holdout.",
    )
    parser.add_argument("--block-seconds", type=float, default=60.0)
    parser.add_argument("--speed-scale", type=float, default=3.6)
    parser.add_argument("--rolling-windows", default="5,10,20,50,100")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument(
        "--skip-refit",
        action="store_true",
        help="Do not refit the deployable model on all rows after evaluation.",
    )
    args = parser.parse_args()

    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be between 0 and 1")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.train_fraction + args.validation_fraction >= 1:
        parser.error("train and validation fractions must leave a test partition")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be at least 1")
    if args.block_seconds <= 0:
        parser.error("--block-seconds must be positive")
    return args


def load_phone_data(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    if raw.shape[1] < 24:
        raise ValueError(f"Expected at least 24 columns in {path}, found {raw.shape[1]}")

    frame = pd.DataFrame(
        {
            "source_row": np.arange(len(raw), dtype=np.int64),
            "timestamp": pd.to_datetime(
                raw.iloc[:, 8], format="%Y-%m-%d %H:%M:%S:%f", errors="coerce"
            ),
            "gps_speed_raw": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "acc_x": pd.to_numeric(raw.iloc[:, 9], errors="coerce"),
            "acc_y": pd.to_numeric(raw.iloc[:, 10], errors="coerce"),
            "acc_z": pd.to_numeric(raw.iloc[:, 11], errors="coerce"),
            "gyro_yaw": pd.to_numeric(raw.iloc[:, 15], errors="coerce"),
            "gyro_pitch": pd.to_numeric(raw.iloc[:, 16], errors="coerce"),
            "gyro_roll": pd.to_numeric(raw.iloc[:, 17], errors="coerce"),
        }
    )
    return frame


def select_longest_continuous_segment(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    delta = frame["timestamp"].diff().dt.total_seconds()
    # Y1 is nominally 10 Hz. A break over 0.5 s is not a normal sample jitter.
    starts_new_segment = delta.isna() | (delta <= 0) | (delta > 0.5)
    segment_id = starts_new_segment.cumsum()
    segment_sizes = frame.groupby(segment_id, sort=False).size()
    selected_id = int(segment_sizes.idxmax())
    selected = frame.loc[segment_id == selected_id].copy().reset_index(drop=True)
    selected_delta = selected["timestamp"].diff().dt.total_seconds().dropna()
    metadata = {
        "segment_count": int(len(segment_sizes)),
        "segment_lengths": [int(value) for value in segment_sizes.tolist()],
        "selected_segment_id": selected_id,
        "selected_rows": int(len(selected)),
        "source_row_start": int(selected["source_row"].iloc[0]),
        "source_row_end": int(selected["source_row"].iloc[-1]),
        "start_timestamp": selected["timestamp"].iloc[0].isoformat(),
        "end_timestamp": selected["timestamp"].iloc[-1].isoformat(),
        "duration_seconds": float(
            (selected["timestamp"].iloc[-1] - selected["timestamp"].iloc[0]).total_seconds()
        ),
        "median_period_seconds": float(selected_delta.median()),
    }
    return selected, metadata


def build_causal_features(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    base = frame[IMU_COLUMNS].copy()
    base["acc_norm"] = np.sqrt(
        base["acc_x"] ** 2 + base["acc_y"] ** 2 + base["acc_z"] ** 2
    )
    base["gyro_norm"] = np.sqrt(
        base["gyro_yaw"] ** 2
        + base["gyro_pitch"] ** 2
        + base["gyro_roll"] ** 2
    )

    period = frame["timestamp"].diff().dt.total_seconds().clip(lower=0.05, upper=0.2)
    period = period.fillna(0.1)
    generated: dict[str, pd.Series] = {}
    for column in IMU_COLUMNS:
        delta = base[column].diff().fillna(0.0)
        generated[f"{column}_delta"] = delta
        generated[f"{column}_rate"] = delta / period

    rolling_sources = IMU_COLUMNS + ["acc_norm", "gyro_norm"]
    for window in windows:
        for column in rolling_sources:
            rolling = base[column].rolling(window=window, min_periods=window)
            generated[f"{column}_mean_{window}"] = rolling.mean()
            generated[f"{column}_std_{window}"] = rolling.std(ddof=0)
            generated[f"{column}_range_{window}"] = rolling.max() - rolling.min()
        generated[f"acc_energy_{window}"] = (
            base["acc_norm"].pow(2).rolling(window, min_periods=window).mean()
        )
        generated[f"gyro_energy_{window}"] = (
            base["gyro_norm"].pow(2).rolling(window, min_periods=window).mean()
        )

    return pd.concat([base, pd.DataFrame(generated, index=frame.index)], axis=1)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.abs(actual - predicted)
    return {
        "mae_kmh": float(mean_absolute_error(actual, predicted)),
        "rmse_kmh": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "within_1_kmh_percent": float(np.mean(error <= 1.0) * 100),
        "within_2_kmh_percent": float(np.mean(error <= 2.0) * 100),
        "within_5_kmh_percent": float(np.mean(error <= 5.0) * 100),
        "bias_kmh": float(np.mean(predicted - actual)),
    }


def make_partitions(
    row_count: int,
    train_fraction: float,
    validation_fraction: float,
    strategy: str,
    block_rows: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if strategy == "chronological":
        train_end = int(row_count * train_fraction)
        validation_end = int(row_count * (train_fraction + validation_fraction))
        return {
            "train": np.arange(0, train_end),
            "validation": np.arange(train_end, validation_end),
            "test": np.arange(validation_end, row_count),
        }

    block_ids = np.arange(row_count) // block_rows
    unique_blocks = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    shuffled_blocks = rng.permutation(unique_blocks)
    train_block_end = int(len(shuffled_blocks) * train_fraction)
    validation_block_end = int(
        len(shuffled_blocks) * (train_fraction + validation_fraction)
    )
    assigned = {
        "train": shuffled_blocks[:train_block_end],
        "validation": shuffled_blocks[train_block_end:validation_block_end],
        "test": shuffled_blocks[validation_block_end:],
    }
    return {
        name: np.flatnonzero(np.isin(block_ids, selected_blocks))
        for name, selected_blocks in assigned.items()
    }


def plot_predictions(predictions: pd.DataFrame, output: Path) -> None:
    test = predictions.loc[predictions["split"] == "test"]
    # Plot at most about 4,000 points to keep the artifact lightweight.
    stride = max(1, len(test) // 4_000)
    shown = test.iloc[::stride]
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(shown["timestamp"], shown["actual_speed_kmh"], label="GPS target", linewidth=1.2)
    axis.plot(shown["timestamp"], shown["predicted_speed_kmh"], label="IMU prediction", linewidth=1.0)
    axis.set_xlabel("Time")
    axis.set_ylabel("Speed (km/h)")
    axis.set_title("Y1 chronological test partition")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    windows = sorted({int(value) for value in args.rolling_windows.split(",") if value})
    if not windows or min(windows) < 2:
        raise ValueError("--rolling-windows must contain integers of at least 2")

    started = time.monotonic()
    phone = load_phone_data(args.input)
    segment, segment_metadata = select_longest_continuous_segment(phone)
    if args.max_rows:
        segment = segment.iloc[: args.max_rows].copy()

    segment["target_speed_kmh"] = segment["gps_speed_raw"] * args.speed_scale
    features = build_causal_features(segment, windows)
    usable = (
        features.notna().all(axis=1)
        & segment["target_speed_kmh"].between(0, 160, inclusive="both")
    )
    features = features.loc[usable].iloc[:: args.sample_stride].reset_index(drop=True)
    examples = segment.loc[usable].iloc[:: args.sample_stride].reset_index(drop=True)
    target = examples["target_speed_kmh"].astype(float)

    minimum_rows = 1_000
    if len(features) < minimum_rows:
        raise ValueError(f"Only {len(features)} usable rows; at least {minimum_rows} are required")

    median_period = segment_metadata["median_period_seconds"] * args.sample_stride
    block_rows = max(1, round(args.block_seconds / median_period))
    partitions = make_partitions(
        len(features),
        args.train_fraction,
        args.validation_fraction,
        args.split_strategy,
        block_rows,
        args.seed,
    )

    model = CatBoostRegressor(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="RMSE",
        eval_metric="MAE",
        random_seed=args.seed,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        task_type=args.task_type,
        thread_count=args.threads,
        allow_writing_files=False,
        verbose=max(1, args.iterations // 20),
    )
    model.fit(
        features.iloc[partitions["train"]],
        target.iloc[partitions["train"]],
        eval_set=(
            features.iloc[partitions["validation"]],
            target.iloc[partitions["validation"]],
        ),
        early_stopping_rounds=args.early_stopping_rounds,
        use_best_model=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output_dir / "evaluation_model.cbm")
    joblib.dump(
        {
            "feature_names": features.columns.tolist(),
            "rolling_windows": windows,
            "imu_columns": IMU_COLUMNS,
            "speed_scale": args.speed_scale,
            "model_type": "CatBoostRegressor",
        },
        args.output_dir / "preprocessing.joblib",
    )

    metrics = {}
    prediction_frames = []
    for split, indices in partitions.items():
        actual = target.iloc[indices].to_numpy()
        predicted = np.clip(model.predict(features.iloc[indices]), 0, None)
        metrics[split] = regression_metrics(actual, predicted)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "source_row": examples["source_row"].iloc[indices].to_numpy(),
                    "timestamp": examples["timestamp"].iloc[indices].to_numpy(),
                    "actual_speed_kmh": actual,
                    "predicted_speed_kmh": predicted,
                    "split": split,
                }
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    plot_predictions(predictions, args.output_dir / "test_predictions.png")

    best_iteration = model.get_best_iteration()
    deployment_iterations = best_iteration + 1 if best_iteration >= 0 else args.iterations
    if not args.skip_refit:
        deployment_model = CatBoostRegressor(
            iterations=deployment_iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            loss_function="RMSE",
            random_seed=args.seed,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            task_type=args.task_type,
            thread_count=args.threads,
            allow_writing_files=False,
            verbose=max(1, deployment_iterations // 10),
        )
        deployment_model.fit(features, target)
        deployment_model.save_model(args.output_dir / "speed_model.cbm")

    run_metadata = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "label": {
            "source": "smartphone GPS speed column",
            "raw_unit_interpretation": "m/s despite source header saying km/h",
            "conversion_to_kmh": args.speed_scale,
            "used_as_model_feature": False,
        },
        "features": {
            "sources": IMU_COLUMNS,
            "causal_only": True,
            "count": int(features.shape[1]),
            "rolling_windows_samples": windows,
        },
        "segment": segment_metadata,
        "rows_after_feature_warmup": int(len(features)),
        "split_rows": {name: int(len(indices)) for name, indices in partitions.items()},
        "split_strategy": args.split_strategy,
        "block_seconds": args.block_seconds if args.split_strategy == "blocked-random" else None,
        "model": {
            "type": "CatBoostRegressor",
            "requested_iterations": args.iterations,
            "best_iteration": int(best_iteration),
            "deployment_iterations": int(deployment_iterations),
            "deployment_refit_on_all_rows": not args.skip_refit,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "task_type": args.task_type,
            "seed": args.seed,
        },
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )

    test_metrics = metrics["test"]
    print("Training complete")
    print(f"Usable rows: {len(features):,}; features: {features.shape[1]}")
    print(f"Best iteration: {best_iteration}")
    print(
        "Test: "
        f"MAE={test_metrics['mae_kmh']:.3f} km/h, "
        f"RMSE={test_metrics['rmse_kmh']:.3f} km/h, "
        f"R2={test_metrics['r2']:.3f}, "
        f"within 5 km/h={test_metrics['within_5_kmh_percent']:.1f}%"
    )
    if not args.skip_refit:
        print(f"Refit deployment model on all {len(features):,} usable rows")
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
