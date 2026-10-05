#!/usr/bin/env python3
"""Train and evaluate a leakage-safe, stateful IMU delta-velocity model.

Each simulated blackout starts from the last valid GNSS speed. After that
initialization, state updates use only causal smartphone IMU-derived features.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import matplotlib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DEFAULT_Y1 = Path(
    "data/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/Y (Driver D)/Y1/S-Y1.txt"
)
DEFAULT_OUTPUT = Path("artifacts/dead_reckoning_catboost")


@dataclass
class Session:
    drive_id: str
    frame: pd.DataFrame
    features: pd.DataFrame
    metadata: dict


class DeltaModel(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class ZeroDeltaModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        action="append",
        metavar="DRIVE_ID=PATH",
        help="Repeat to add synchronized phone sessions. Defaults to Y1.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", default="0.5,1.0")
    parser.add_argument("--blackout-durations", default="10,30,60,120")
    parser.add_argument("--rolling-windows", default="5,10,20,50,100")
    parser.add_argument("--block-seconds", type=float, default=300.0)
    parser.add_argument(
        "--split-strategy",
        choices=["blocked-random", "chronological", "drive"],
        default="blocked-random",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--skip-refit", action="store_true")
    args = parser.parse_args()

    if args.train_fraction <= 0 or args.validation_fraction <= 0:
        parser.error("split fractions must be positive")
    if args.train_fraction + args.validation_fraction >= 1:
        parser.error("split fractions must leave a test partition")
    if args.block_seconds <= 0:
        parser.error("--block-seconds must be positive")
    return args


def parse_number_list(raw: str, cast: type) -> list:
    values = sorted({cast(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError(f"Expected a comma-separated list, received {raw!r}")
    return values


def session_specs(raw_specs: list[str] | None) -> list[tuple[str, Path]]:
    if not raw_specs:
        return [("Y1", DEFAULT_Y1)]
    parsed = []
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"Session must use DRIVE_ID=PATH syntax: {spec!r}")
        drive_id, raw_path = spec.split("=", 1)
        parsed.append((drive_id.strip(), Path(raw_path)))
    return parsed


def load_phone(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    if raw.shape[1] < 24:
        raise ValueError(f"Expected at least 24 columns in {path}, found {raw.shape[1]}")
    return pd.DataFrame(
        {
            "source_row": np.arange(len(raw), dtype=np.int64),
            "timestamp": pd.to_datetime(
                raw.iloc[:, 8], format="%Y-%m-%d %H:%M:%S:%f", errors="coerce"
            ),
            "gps_speed_raw_ms": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "acc_x": pd.to_numeric(raw.iloc[:, 9], errors="coerce"),
            "acc_y": pd.to_numeric(raw.iloc[:, 10], errors="coerce"),
            "acc_z": pd.to_numeric(raw.iloc[:, 11], errors="coerce"),
            "gravity_x": pd.to_numeric(raw.iloc[:, 12], errors="coerce"),
            "gravity_y": pd.to_numeric(raw.iloc[:, 13], errors="coerce"),
            "gravity_z": pd.to_numeric(raw.iloc[:, 14], errors="coerce"),
            "gyro_x": pd.to_numeric(raw.iloc[:, 15], errors="coerce"),
            "gyro_y": pd.to_numeric(raw.iloc[:, 16], errors="coerce"),
            "gyro_z": pd.to_numeric(raw.iloc[:, 17], errors="coerce"),
            "orientation_yaw": pd.to_numeric(raw.iloc[:, 21], errors="coerce"),
            "orientation_pitch": pd.to_numeric(raw.iloc[:, 22], errors="coerce"),
            "orientation_roll": pd.to_numeric(raw.iloc[:, 23], errors="coerce"),
        }
    )


def select_longest_segment(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    delta = frame["timestamp"].diff().dt.total_seconds()
    segment_id = (delta.isna() | (delta <= 0) | (delta > 0.5)).cumsum()
    sizes = frame.groupby(segment_id, sort=False).size()
    selected_id = int(sizes.idxmax())
    selected = frame.loc[segment_id == selected_id].copy().reset_index(drop=True)
    periods = selected["timestamp"].diff().dt.total_seconds().dropna()
    return selected, {
        "all_segment_lengths": [int(value) for value in sizes.tolist()],
        "selected_segment_id": selected_id,
        "selected_rows": int(len(selected)),
        "source_row_start": int(selected["source_row"].iloc[0]),
        "source_row_end": int(selected["source_row"].iloc[-1]),
        "start_timestamp": selected["timestamp"].iloc[0].isoformat(),
        "end_timestamp": selected["timestamp"].iloc[-1].isoformat(),
        "median_period_seconds": float(periods.median()),
    }


def reconstruct_speed_target(frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    held = frame["gps_speed_raw_ms"].interpolate().bfill().ffill().to_numpy(float)
    changes = np.flatnonzero(np.diff(held) != 0) + 1
    update_indices = np.unique(np.r_[0, changes, len(held) - 1])
    reconstructed = np.interp(np.arange(len(held)), update_indices, held[update_indices])
    run_lengths = np.diff(np.r_[0, changes, len(held)])
    duration = max(
        1e-9,
        (frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]).total_seconds(),
    )
    return reconstructed, {
        "method": "linear interpolation between observed smartphone GPS speed updates",
        "future_gps_used_as_feature": False,
        "observed_speed_change_count": int(len(changes)),
        "observed_update_rate_hz": float(len(changes) / duration),
        "constant_run_samples_p50": float(np.percentile(run_lengths, 50)),
        "constant_run_samples_p90": float(np.percentile(run_lengths, 90)),
        "constant_run_samples_max": int(run_lengths.max()),
    }


def build_imu_features(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    acc = frame[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
    gravity = frame[["gravity_x", "gravity_y", "gravity_z"]].to_numpy(float)
    gyro = frame[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(float)
    linear = acc - gravity
    gravity_unit = gravity / np.maximum(np.linalg.norm(gravity, axis=1)[:, None], 1e-6)
    linear_vertical = np.sum(linear * gravity_unit, axis=1)
    linear_horizontal_vector = linear - linear_vertical[:, None] * gravity_unit
    gyro_vertical = np.sum(gyro * gravity_unit, axis=1)
    gyro_horizontal_vector = gyro - gyro_vertical[:, None] * gravity_unit

    angles = (
        frame[["orientation_yaw", "orientation_pitch", "orientation_roll"]]
        .ffill()
        .fillna(0.0)
        .to_numpy(float)
    )
    # Android orientation is conventionally azimuth(Z), pitch(X), roll(Y).
    # Angles themselves are not exposed to the model; they only rotate linear
    # acceleration into an exploratory world-oriented representation.
    world_linear = Rotation.from_euler("zxy", angles, degrees=True).apply(linear)

    base = pd.DataFrame(
        {
            "acc_x": acc[:, 0],
            "acc_y": acc[:, 1],
            "acc_z": acc[:, 2],
            "gyro_x": gyro[:, 0],
            "gyro_y": gyro[:, 1],
            "gyro_z": gyro[:, 2],
            "linear_x": linear[:, 0],
            "linear_y": linear[:, 1],
            "linear_z": linear[:, 2],
            "acc_norm": np.linalg.norm(acc, axis=1),
            "linear_norm": np.linalg.norm(linear, axis=1),
            "gyro_norm": np.linalg.norm(gyro, axis=1),
            "linear_vertical": linear_vertical,
            "linear_horizontal": np.linalg.norm(linear_horizontal_vector, axis=1),
            "gyro_vertical": gyro_vertical,
            "gyro_horizontal": np.linalg.norm(gyro_horizontal_vector, axis=1),
            "world_linear_x": world_linear[:, 0],
            "world_linear_y": world_linear[:, 1],
            "world_linear_z": world_linear[:, 2],
        },
        index=frame.index,
    )

    generated: dict[str, pd.Series] = {}
    period = frame["timestamp"].diff().dt.total_seconds().clip(0.05, 0.2).fillna(0.1)
    derivative_columns = [
        "linear_x",
        "linear_y",
        "linear_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ]
    for column in derivative_columns:
        delta = base[column].diff().fillna(0.0)
        generated[f"{column}_delta"] = delta
        generated[f"{column}_rate"] = delta / period

    for window in windows:
        for column in base.columns:
            rolling = base[column].rolling(window, min_periods=window)
            generated[f"{column}_mean_{window}"] = rolling.mean()
            generated[f"{column}_std_{window}"] = rolling.std(ddof=0)
            generated[f"{column}_range_{window}"] = rolling.max() - rolling.min()
        generated[f"linear_energy_{window}"] = (
            base["linear_norm"].pow(2).rolling(window, min_periods=window).mean()
        )
        generated[f"gyro_energy_{window}"] = (
            base["gyro_norm"].pow(2).rolling(window, min_periods=window).mean()
        )
    return pd.concat([base, pd.DataFrame(generated, index=frame.index)], axis=1)


def load_sessions(
    specs: list[tuple[str, Path]], windows: list[int], max_rows: int | None
) -> list[Session]:
    sessions = []
    for drive_id, path in specs:
        selected, metadata = select_longest_segment(load_phone(path))
        if max_rows:
            selected = selected.iloc[:max_rows].copy().reset_index(drop=True)
            metadata["max_rows_applied"] = max_rows
        speed, speed_metadata = reconstruct_speed_target(selected)
        selected["target_speed_ms"] = speed
        metadata["speed_target"] = speed_metadata
        metadata["path"] = str(path)
        features = build_imu_features(selected, windows)
        sessions.append(Session(drive_id, selected, features, metadata))
    return sessions


def assign_blocks(
    sessions: list[Session],
    block_seconds: float,
    strategy: str,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    units: list[tuple[str, int]] = []
    for session in sessions:
        period = session.metadata["median_period_seconds"]
        block_rows = max(1, round(block_seconds / period))
        local_index = np.arange(len(session.frame))
        block_id = local_index // block_rows
        for value in np.unique(block_id):
            units.append((session.drive_id, int(value)))
        rows.append(
            pd.DataFrame(
                {
                    "drive_id": session.drive_id,
                    "local_index": local_index,
                    "block_id": block_id,
                    "position_in_block": local_index % block_rows,
                }
            )
        )
    layout = pd.concat(rows, ignore_index=True)

    unit_to_split: dict[tuple[str, int], str] = {}
    if strategy == "drive":
        drives = np.array([session.drive_id for session in sessions], dtype=object)
        if len(drives) < 3:
            raise ValueError("Drive-level splitting requires at least three sessions")
        shuffled = np.random.default_rng(seed).permutation(drives)
        train_end = max(1, int(len(shuffled) * train_fraction))
        validation_end = max(train_end + 1, int(len(shuffled) * (train_fraction + validation_fraction)))
        drive_split = {
            drive: "train" if index < train_end else "validation" if index < validation_end else "test"
            for index, drive in enumerate(shuffled)
        }
        for unit in units:
            unit_to_split[unit] = drive_split[unit[0]]
    elif strategy == "chronological":
        for session in sessions:
            drive_units = [unit for unit in units if unit[0] == session.drive_id]
            train_end = int(len(drive_units) * train_fraction)
            validation_end = int(len(drive_units) * (train_fraction + validation_fraction))
            for index, unit in enumerate(drive_units):
                unit_to_split[unit] = (
                    "train" if index < train_end else "validation" if index < validation_end else "test"
                )
    else:
        shuffled = np.random.default_rng(seed).permutation(len(units))
        train_end = int(len(units) * train_fraction)
        validation_end = int(len(units) * (train_fraction + validation_fraction))
        for order, unit_index in enumerate(shuffled):
            unit_to_split[units[int(unit_index)]] = (
                "train" if order < train_end else "validation" if order < validation_end else "test"
            )

    layout["split"] = [
        unit_to_split[(drive, int(block))]
        for drive, block in zip(layout["drive_id"], layout["block_id"])
    ]
    counts = layout.groupby("split").size().to_dict()
    if any(name not in counts for name in ("train", "validation", "test")):
        raise ValueError(f"Split produced an empty partition: {counts}")
    return layout, {
        "strategy": strategy,
        "block_seconds": block_seconds,
        "unit_assignments": {
            f"{drive}:{block}": split
            for (drive, block), split in sorted(unit_to_split.items())
        },
        "row_counts_before_purge": {key: int(value) for key, value in counts.items()},
    }


def combine_sessions(sessions: list[Session]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    features = []
    for session in sessions:
        frame = session.frame.copy()
        frame["drive_id"] = session.drive_id
        frame["local_index"] = np.arange(len(frame))
        frames.append(frame)
        feature = session.features.copy()
        feature.index = pd.RangeIndex(len(feature))
        features.append(feature)
    return pd.concat(frames, ignore_index=True), pd.concat(features, ignore_index=True)


def target_investigation(
    sessions: list[Session], horizons: list[float], base_feature_names: list[str]
) -> dict:
    output = {"sessions": {}, "horizons": {}}
    for session in sessions:
        output["sessions"][session.drive_id] = session.metadata
    for horizon in horizons:
        raw_targets = []
        reconstructed_targets = []
        correlations: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            name: [] for name in base_feature_names
        }
        for session in sessions:
            period = session.metadata["median_period_seconds"]
            k = max(1, round(horizon / period))
            raw_speed = session.frame["gps_speed_raw_ms"].interpolate().bfill().ffill()
            raw_targets.append(raw_speed.diff(k).iloc[k:].to_numpy())
            target = session.frame["target_speed_ms"].diff(k).iloc[k:].to_numpy()
            reconstructed_targets.append(target)
            for feature in base_feature_names:
                summary = session.features[feature].rolling(k, min_periods=k).mean().iloc[k:].to_numpy()
                correlations[feature].append((summary, target))
        raw = np.concatenate(raw_targets)
        reconstructed = np.concatenate(reconstructed_targets)
        ranked = []
        for feature, pairs in correlations.items():
            left = np.concatenate([pair[0] for pair in pairs])
            right = np.concatenate([pair[1] for pair in pairs])
            valid = np.isfinite(left) & np.isfinite(right)
            value = float(np.corrcoef(left[valid], right[valid])[0, 1])
            ranked.append({"feature": feature, "correlation": value})
        ranked.sort(key=lambda item: abs(item["correlation"]), reverse=True)
        output["horizons"][str(horizon)] = {
            "samples": int(len(reconstructed)),
            "raw_held_target": distribution(raw),
            "interpolated_target": distribution(reconstructed),
            "top_interval_mean_correlations": ranked[:10],
        }
    return output


def distribution(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "mean_absolute": float(np.mean(np.abs(finite))),
        "zero_percent": float(np.mean(np.abs(finite) < 1e-12) * 100),
        "percentiles": {
            str(percentile): float(np.percentile(finite, percentile))
            for percentile in (0, 1, 10, 25, 50, 75, 90, 99, 100)
        },
    }


def make_model(args: argparse.Namespace, iterations: int | None = None, verbose_divisor: int = 20) -> CatBoostRegressor:
    count = iterations or args.iterations
    return CatBoostRegressor(
        iterations=count,
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
        verbose=max(1, count // verbose_divisor),
    )


def eligible_indices(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    layout: pd.DataFrame,
    horizon_rows: int,
    purge_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = frame.groupby("drive_id", sort=False)["target_speed_ms"].diff(horizon_rows)
    eligible = (
        (layout["position_in_block"] >= purge_rows)
        & features.notna().all(axis=1)
        & target.notna()
    )
    return np.flatnonzero(eligible.to_numpy()), target.to_numpy(float)


def audit_boundaries(
    eligible: np.ndarray,
    layout: pd.DataFrame,
    horizon_rows: int,
    max_history_rows: int,
) -> dict:
    failures = []
    for index in eligible:
        history_start = int(index) - max_history_rows + 1
        target_start = int(index) - horizon_rows
        current = layout.iloc[int(index)]
        for role, boundary_index in (("history", history_start), ("target", target_start)):
            if boundary_index < 0:
                failures.append((int(index), role, boundary_index, "negative index"))
                continue
            boundary = layout.iloc[boundary_index]
            if (
                boundary["drive_id"] != current["drive_id"]
                or int(boundary["block_id"]) != int(current["block_id"])
                or boundary["split"] != current["split"]
            ):
                failures.append((int(index), role, boundary_index, "crossed partition boundary"))
    if failures:
        raise AssertionError(f"Feature/target boundary leakage detected: {failures[:5]}")
    return {
        "eligible_rows_checked": int(len(eligible)),
        "history_rows_checked_per_example": max_history_rows,
        "target_horizon_rows_checked": horizon_rows,
        "cross_boundary_failures": 0,
    }


def static_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mae_delta_ms": float(mean_absolute_error(actual, predicted)),
        "rmse_delta_ms": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "zero_delta_baseline_mae_ms": float(np.mean(np.abs(actual))),
    }


def evaluate_blackouts(
    model: DeltaModel,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    layout: pd.DataFrame,
    split: str,
    horizon_seconds: float,
    durations: list[int],
    max_history_rows: int,
    session_periods: dict[str, float],
    model_name: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    episodes = []
    velocity_samples = []
    selected = layout.loc[layout["split"] == split]
    for (drive_id, block_id), block_layout in selected.groupby(["drive_id", "block_id"], sort=False):
        block_indices = block_layout.index.to_numpy()
        block_start, block_end = int(block_indices[0]), int(block_indices[-1])
        period = session_periods[str(drive_id)]
        horizon_rows = max(1, round(horizon_seconds / period))
        for duration in durations:
            duration_rows = round(duration / period)
            first_start = block_start + max_history_rows
            last_start = block_end - duration_rows
            if first_start > last_start:
                continue
            # Non-overlapping episodes keep the reported sample count interpretable.
            for start in range(first_start, last_start + 1, duration_rows):
                end = start + duration_rows
                endpoints = np.arange(start + horizon_rows, end + 1, horizon_rows)
                endpoints = endpoints[endpoints <= end]
                if not len(endpoints):
                    continue
                deltas = np.asarray(model.predict(features.iloc[endpoints]), dtype=float)
                predicted_speed = [float(frame["target_speed_ms"].iloc[start])]
                predicted_distance = 0.0
                for delta in deltas:
                    previous = predicted_speed[-1]
                    current = max(0.0, previous + float(delta))
                    predicted_distance += 0.5 * (previous + current) * horizon_rows * period
                    predicted_speed.append(current)
                truth_indices = np.r_[start, endpoints]
                true_speed = frame["target_speed_ms"].iloc[truth_indices].to_numpy(float)
                predicted_speed_array = np.asarray(predicted_speed)
                error_kmh = (predicted_speed_array - true_speed) * 3.6
                dense_true = frame["target_speed_ms"].iloc[start : end + 1].to_numpy(float)
                true_distance = float(np.trapezoid(dense_true, dx=period))
                distance_error = predicted_distance - true_distance
                episode_id = f"{model_name}:{drive_id}:{block_id}:{duration}:{start}"
                episodes.append(
                    {
                        "model": model_name,
                        "episode_id": episode_id,
                        "drive_id": drive_id,
                        "block_id": int(block_id),
                        "duration_seconds": duration,
                        "start_global_row": start,
                        "end_global_row": end,
                        "velocity_mae_kmh": float(np.mean(np.abs(error_kmh[1:]))),
                        "velocity_rmse_kmh": float(np.sqrt(np.mean(error_kmh[1:] ** 2))),
                        "final_velocity_error_kmh": float(error_kmh[-1]),
                        "final_velocity_absolute_error_kmh": float(abs(error_kmh[-1])),
                        "within_5_kmh_percent": float(np.mean(np.abs(error_kmh[1:]) <= 5) * 100),
                        "true_distance_m": true_distance,
                        "predicted_distance_m": predicted_distance,
                        "distance_error_m": distance_error,
                        "absolute_distance_error_m": abs(distance_error),
                        "distance_drift_percent": (
                            float(abs(distance_error) / true_distance * 100)
                            if true_distance >= 1.0
                            else math.nan
                        ),
                    }
                )
                for position, (row_index, truth, prediction, error) in enumerate(
                    zip(truth_indices, true_speed, predicted_speed_array, error_kmh)
                ):
                    velocity_samples.append(
                        {
                            "model": model_name,
                            "episode_id": episode_id,
                            "duration_seconds": duration,
                            "elapsed_seconds": position * horizon_rows * period,
                            "global_row": int(row_index),
                            "true_speed_kmh": truth * 3.6,
                            "predicted_speed_kmh": prediction * 3.6,
                            "velocity_error_kmh": error,
                        }
                    )
    episode_frame = pd.DataFrame(episodes)
    velocity_frame = pd.DataFrame(velocity_samples)
    if episode_frame.empty:
        raise ValueError(f"No {split} blackout episodes could be generated")
    summary = {}
    for duration, group in episode_frame.groupby("duration_seconds"):
        sample_group = velocity_frame.loc[
            (velocity_frame["duration_seconds"] == duration)
            & (velocity_frame["elapsed_seconds"] > 0)
        ]
        summary[str(int(duration))] = {
            "episodes": int(len(group)),
            "velocity_mae_kmh": float(np.mean(np.abs(sample_group["velocity_error_kmh"]))),
            "velocity_rmse_kmh": float(np.sqrt(np.mean(sample_group["velocity_error_kmh"] ** 2))),
            "final_velocity_absolute_error_kmh": float(group["final_velocity_absolute_error_kmh"].mean()),
            "final_velocity_error_bias_kmh": float(group["final_velocity_error_kmh"].mean()),
            "within_5_kmh_percent": float(np.mean(np.abs(sample_group["velocity_error_kmh"]) <= 5) * 100),
            "absolute_distance_error_m": float(group["absolute_distance_error_m"].mean()),
            "distance_error_bias_m": float(group["distance_error_m"].mean()),
            "distance_drift_percent": float(group["distance_drift_percent"].mean()),
        }
    return summary, episode_frame, velocity_frame


def write_investigation(investigation: dict, output_dir: Path) -> None:
    (output_dir / "target_investigation.json").write_text(
        json.dumps(investigation, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Delta-velocity target investigation",
        "",
        "The phone IMU is approximately 10 Hz, but Y1 smartphone GPS speed updates",
        "only about once every nine seconds. Raw 0.5 s and 1.0 s differences are",
        "therefore dominated by zero-order holds and discontinuous update jumps.",
        "",
        "For this experiment only, the ground-truth speed trace is linearly",
        "interpolated between observed GPS speed updates. This uses future GPS only",
        "to construct offline labels and evaluation truth; it is never a feature.",
        "",
        "| Horizon | Raw zero % | Interpolated zero % | Interpolated Δv MAE (m/s) | Interpolated std (m/s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon, details in investigation["horizons"].items():
        raw = details["raw_held_target"]
        smooth = details["interpolated_target"]
        lines.append(
            f"| {horizon} s | {raw['zero_percent']:.1f} | {smooth['zero_percent']:.1f} | "
            f"{smooth['mean_absolute']:.3f} | {smooth['std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Simple linear correlations remain weak after gravity and orientation",
            "handling. CatBoost is retained to test nonlinear relationships, with a",
            "constant-last-speed rollout reported as the required baseline.",
            "",
        ]
    )
    (output_dir / "target_investigation.md").write_text("\n".join(lines), encoding="utf-8")


def plot_outputs(
    test_episodes: pd.DataFrame,
    test_velocity: pd.DataFrame,
    summary: dict,
    baseline_summary: dict,
    importance: pd.DataFrame,
    output_dir: Path,
) -> None:
    durations = sorted(test_episodes["duration_seconds"].unique())
    figure, axes = plt.subplots(len(durations), 1, figsize=(13, 3.2 * len(durations)), squeeze=False)
    for axis, duration in zip(axes[:, 0], durations):
        episode_id = test_episodes.loc[test_episodes["duration_seconds"] == duration, "episode_id"].iloc[0]
        shown = test_velocity.loc[test_velocity["episode_id"] == episode_id]
        axis.plot(shown["elapsed_seconds"], shown["true_speed_kmh"], label="Ground truth")
        axis.plot(shown["elapsed_seconds"], shown["predicted_speed_kmh"], label="Stateful IMU estimate")
        axis.set_title(f"Representative {duration:g} s blackout")
        axis.set_ylabel("Velocity (km/h)")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Blackout time (s)")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "blackout_velocity_examples.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5))
    grouped = (
        test_velocity.assign(abs_error=lambda value: value["velocity_error_kmh"].abs())
        .groupby(["duration_seconds", "elapsed_seconds"])["abs_error"]
        .mean()
        .reset_index()
    )
    for duration, values in grouped.groupby("duration_seconds"):
        axis.plot(values["elapsed_seconds"], values["abs_error"], label=f"{duration:g} s")
    axis.set_xlabel("Time since blackout start (s)")
    axis.set_ylabel("Mean absolute velocity error (km/h)")
    axis.set_title("Velocity error accumulation")
    axis.grid(alpha=0.25)
    axis.legend(title="Blackout")
    figure.tight_layout()
    figure.savefig(output_dir / "velocity_error_over_time.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 7))
    for duration, values in test_episodes.groupby("duration_seconds"):
        axis.scatter(values["true_distance_m"], values["predicted_distance_m"], label=f"{duration:g} s", alpha=0.75)
    maximum = max(test_episodes["true_distance_m"].max(), test_episodes["predicted_distance_m"].max())
    axis.plot([0, maximum], [0, maximum], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Ground-truth distance (m)")
    axis.set_ylabel("Predicted distance (m)")
    axis.set_title("Distance travelled during held-out blackouts")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "distance_ground_truth_vs_prediction.png", dpi=140)
    plt.close(figure)

    x = np.array(sorted(int(value) for value in summary))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x, [summary[str(value)]["velocity_mae_kmh"] for value in x], marker="o", label="Δv model")
    axes[0].plot(x, [baseline_summary[str(value)]["velocity_mae_kmh"] for value in x], marker="o", label="Hold last speed")
    axes[0].set_ylabel("Velocity MAE (km/h)")
    axes[1].plot(x, [summary[str(value)]["distance_drift_percent"] for value in x], marker="o", label="Δv model")
    axes[1].plot(x, [baseline_summary[str(value)]["distance_drift_percent"] for value in x], marker="o", label="Hold last speed")
    axes[1].set_ylabel("Mean distance drift (%)")
    for axis in axes:
        axis.set_xlabel("Blackout duration (s)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Error and drift versus blackout duration")
    figure.tight_layout()
    figure.savefig(output_dir / "error_drift_vs_duration.png", dpi=140)
    plt.close(figure)

    top = importance.head(25).sort_values("importance")
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.barh(top["feature"], top["importance"])
    axis.set_xlabel("CatBoost importance")
    axis.set_title("Δv model feature importance")
    figure.tight_layout()
    figure.savefig(output_dir / "delta_v_feature_importance.png", dpi=140)
    plt.close(figure)


def write_experiment_report(metadata: dict, output_dir: Path) -> None:
    lines = [
        "# Stateful velocity/dead-reckoning CatBoost experiment",
        "",
        f"Selected Δv horizon: **{metadata['selected_horizon_seconds']:.1f} s**, chosen on validation blackouts only.",
        "",
        "All held-out rollouts begin from the true speed at blackout start. Every",
        "subsequent update uses causal IMU features only. Blocks are purged by the",
        "maximum feature history before samples become eligible.",
        "",
        "## Held-out test results",
        "",
        "| Blackout | Episodes | Velocity MAE | Velocity RMSE | Final | Within ±5 | Distance error | Drift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for duration, result in metadata["test_blackouts"].items():
        lines.append(
            f"| {duration} s | {result['episodes']} | {result['velocity_mae_kmh']:.2f} km/h | "
            f"{result['velocity_rmse_kmh']:.2f} km/h | {result['final_velocity_absolute_error_kmh']:.2f} km/h | "
            f"{result['within_5_kmh_percent']:.1f}% | {result['absolute_distance_error_m']:.1f} m | "
            f"{result['distance_drift_percent']:.1f}% |"
        )
    lines.extend(["", "## Constant-last-speed comparison", "", "| Blackout | Velocity MAE | Distance error | Drift |", "|---:|---:|---:|---:|"])
    for duration, result in metadata["constant_speed_baseline"].items():
        lines.append(
            f"| {duration} s | {result['velocity_mae_kmh']:.2f} km/h | "
            f"{result['absolute_distance_error_m']:.1f} m | {result['distance_drift_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Leakage and interpretation",
            "",
            f"- Maximum causal history: {metadata['feature_policy']['maximum_history_rows']} rows.",
            f"- Purge at every block start: {metadata['purge']['rows_removed_at_start_of_every_block']} rows.",
            "- Every eligible feature history and Δv target boundary was checked to remain in one block and split.",
            "- Horizon selection used validation blackouts only; test blocks were read after selection.",
            "- The learned model only marginally improves on holding the initial speed, so this result is not operationally sufficient.",
            "",
            "These results are a one-drive Y1 baseline. GPS interpolation supplies",
            "offline supervision because phone GPS speed updates at only ~0.1 Hz.",
            "Cross-driver conclusions require additional synchronized sessions.",
            "",
        ]
    )
    (output_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    horizons = parse_number_list(args.horizons, float)
    durations = parse_number_list(args.blackout_durations, int)
    windows = parse_number_list(args.rolling_windows, int)
    if min(horizons) <= 0 or min(durations) <= 0 or min(windows) < 2:
        raise ValueError("Horizons/durations must be positive and rolling windows at least 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(session_specs(args.session), windows, args.max_rows)
    frame, features = combine_sessions(sessions)
    layout, split_metadata = assign_blocks(
        sessions,
        args.block_seconds,
        args.split_strategy,
        args.train_fraction,
        args.validation_fraction,
        args.seed,
    )
    session_periods = {
        session.drive_id: session.metadata["median_period_seconds"] for session in sessions
    }
    base_feature_names = list(sessions[0].features.columns[:19])
    investigation = target_investigation(sessions, horizons, base_feature_names)
    write_investigation(investigation, args.output_dir)

    max_history_rows = max(windows)
    purge_rows = max(max_history_rows, max(round(h / min(session_periods.values())) for h in horizons))
    candidate_results = {}
    candidate_models: dict[float, CatBoostRegressor] = {}
    candidate_indices: dict[float, np.ndarray] = {}
    candidate_targets: dict[float, np.ndarray] = {}

    for horizon in horizons:
        # Current Y1 sessions are all 10 Hz. Per-drive horizons are represented
        # in metadata; identical periods are required for a shared CatBoost target.
        horizon_rows_by_drive = {
            drive: max(1, round(horizon / period)) for drive, period in session_periods.items()
        }
        if len(set(horizon_rows_by_drive.values())) != 1:
            raise ValueError("Sessions with different rates must be resampled before joint training")
        horizon_rows = next(iter(horizon_rows_by_drive.values()))
        eligible, target = eligible_indices(frame, features, layout, horizon_rows, purge_rows)
        boundary_audit = audit_boundaries(
            eligible, layout, horizon_rows, max_history_rows
        )
        train_indices = eligible[layout["split"].iloc[eligible].to_numpy() == "train"]
        validation_indices = eligible[layout["split"].iloc[eligible].to_numpy() == "validation"]
        if not len(train_indices) or not len(validation_indices):
            raise ValueError("Purging left an empty train or validation partition")
        model = make_model(args)
        model.fit(
            features.iloc[train_indices],
            target[train_indices],
            eval_set=(features.iloc[validation_indices], target[validation_indices]),
            early_stopping_rounds=args.early_stopping_rounds,
            use_best_model=True,
        )
        key = f"{horizon:.1f}"
        model.save_model(args.output_dir / f"candidate_{int(round(horizon * 1000))}ms_evaluation.cbm")
        validation_prediction = model.predict(features.iloc[validation_indices])
        validation_blackouts, _, _ = evaluate_blackouts(
            model,
            frame,
            features,
            layout,
            "validation",
            horizon,
            durations,
            max_history_rows,
            session_periods,
            f"delta_v_{key}s",
        )
        score = float(np.mean([value["velocity_mae_kmh"] for value in validation_blackouts.values()]))
        candidate_results[key] = {
            "horizon_rows": horizon_rows,
            "eligible_rows": int(len(eligible)),
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "best_iteration": int(model.get_best_iteration()),
            "boundary_leakage_audit": boundary_audit,
            "validation_static_delta": static_metrics(target[validation_indices], validation_prediction),
            "validation_blackouts": validation_blackouts,
            "selection_score_mean_velocity_mae_kmh": score,
        }
        candidate_models[horizon] = model
        candidate_indices[horizon] = eligible
        candidate_targets[horizon] = target

    selected_horizon = min(
        horizons,
        key=lambda value: candidate_results[f"{value:.1f}"]["selection_score_mean_velocity_mae_kmh"],
    )
    selected_model = candidate_models[selected_horizon]
    selected_model.save_model(args.output_dir / "evaluation_delta_v_model.cbm")
    eligible = candidate_indices[selected_horizon]
    target = candidate_targets[selected_horizon]
    test_indices = eligible[layout["split"].iloc[eligible].to_numpy() == "test"]
    test_static = static_metrics(target[test_indices], selected_model.predict(features.iloc[test_indices]))
    test_summary, test_episodes, test_velocity = evaluate_blackouts(
        selected_model,
        frame,
        features,
        layout,
        "test",
        selected_horizon,
        durations,
        max_history_rows,
        session_periods,
        "delta_v_model",
    )
    baseline_summary, baseline_episodes, baseline_velocity = evaluate_blackouts(
        ZeroDeltaModel(),
        frame,
        features,
        layout,
        "test",
        selected_horizon,
        durations,
        max_history_rows,
        session_periods,
        "hold_last_speed",
    )

    test_episodes.to_csv(args.output_dir / "blackout_episodes.csv", index=False)
    test_velocity.to_csv(args.output_dir / "blackout_velocity_samples.csv", index=False)
    baseline_episodes.to_csv(args.output_dir / "constant_speed_blackout_episodes.csv", index=False)
    baseline_velocity.to_csv(args.output_dir / "constant_speed_velocity_samples.csv", index=False)
    importance = pd.DataFrame(
        {"feature": features.columns, "importance": selected_model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.output_dir / "delta_v_feature_importance.csv", index=False)

    deployment_iterations = max(1, selected_model.get_best_iteration() + 1)
    if not args.skip_refit:
        deployment = make_model(args, deployment_iterations, verbose_divisor=10)
        deployment.fit(features.iloc[eligible], target[eligible])
        deployment.save_model(args.output_dir / "delta_v_model.cbm")

    joblib.dump(
        {
            "feature_names": features.columns.tolist(),
            "rolling_windows_samples": windows,
            "selected_horizon_seconds": selected_horizon,
            "maximum_history_rows": max_history_rows,
            "uses_speed_feature": False,
            "uses_orientation_angles_directly": False,
            "speed_initialization": "last valid GNSS speed at blackout start",
        },
        args.output_dir / "preprocessing.joblib",
    )
    metadata = {
        "experiment": "stateful_delta_velocity_catboost",
        "sessions": {session.drive_id: session.metadata for session in sessions},
        "label": {
            "source": "linearly reconstructed smartphone GPS speed, metres/second",
            "future_gps_used_as_feature": False,
            "ground_truth_speed_used_after_blackout_initialization": False,
        },
        "feature_policy": {
            "imu_only": True,
            "causal": True,
            "speed_or_position_features": [],
            "orientation_angles_direct_features": [],
            "orientation_use": "rotation of linear acceleration only",
            "feature_count": int(features.shape[1]),
            "rolling_windows_samples": windows,
            "maximum_history_rows": max_history_rows,
        },
        "split": split_metadata,
        "purge": {
            "rows_removed_at_start_of_every_block": purge_rows,
            "seconds_approximately": purge_rows * min(session_periods.values()),
            "covers_maximum_feature_history": True,
            "covers_all_candidate_delta_horizons": True,
        },
        "candidate_validation_results": candidate_results,
        "selected_horizon_seconds": selected_horizon,
        "selection_rule": "lowest mean validation velocity MAE across requested blackout durations",
        "test_static_delta": test_static,
        "test_blackouts": test_summary,
        "constant_speed_baseline": baseline_summary,
        "deployment_refit_on_all_eligible_rows": not args.skip_refit,
        "deployment_iterations": deployment_iterations,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    plot_outputs(test_episodes, test_velocity, test_summary, baseline_summary, importance, args.output_dir)
    write_experiment_report(metadata, args.output_dir)

    print("Stateful dead-reckoning experiment complete")
    print(f"Selected Δv horizon: {selected_horizon:.1f} s (validation-only selection)")
    for duration in durations:
        result = test_summary[str(duration)]
        baseline = baseline_summary[str(duration)]
        print(
            f"{duration:>3}s: velocity MAE={result['velocity_mae_kmh']:.2f} km/h "
            f"(hold={baseline['velocity_mae_kmh']:.2f}), "
            f"distance error={result['absolute_distance_error_m']:.1f} m "
            f"({result['distance_drift_percent']:.1f}%)"
        )
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
