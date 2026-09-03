#!/usr/bin/env python3
"""Train CatBoost Δv from the validated ECU-supervised NPZ dataset."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_DATASET = Path("artifacts/ecu_training_dataset")
DEFAULT_OUTPUT = Path("artifacts/ecu_delta_v_catboost")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-mode", choices=["driver", "drive"], default="driver")
    parser.add_argument("--driver", help="Restrict to one driver when --split-mode=drive")
    parser.add_argument("--horizon", type=float, choices=[0.5, 1.0], default=1.0)
    parser.add_argument("--rolling-windows", default="5,10,20,50,100")
    parser.add_argument("--blackout-durations", default="10,30,60,120")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-session", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.sample_stride < 1:
        parser.error("--sample-stride must be at least 1")
    if args.split_mode == "drive" and not args.driver:
        parser.error("--driver is required when --split-mode=drive")
    return args


def select_sessions(args: argparse.Namespace) -> pd.DataFrame:
    sessions = pd.read_csv(args.dataset_dir / "sessions.csv")
    if args.split_mode == "driver":
        sessions["active_split"] = sessions["driver_holdout_split"]
    else:
        sessions = sessions.loc[sessions["driver_id"] == args.driver].copy()
        sessions["active_split"] = sessions["drive_holdout_split"]
        missing = {"train", "validation", "test"} - set(sessions["active_split"])
        if missing:
            raise ValueError(f"Driver {args.driver} lacks whole-drive partitions: {sorted(missing)}")
    return sessions


def build_features(archive: dict[str, np.ndarray], windows: list[int]) -> pd.DataFrame:
    acc = np.column_stack([archive["acc_x_ms2"], archive["acc_y_ms2"], archive["acc_z_ms2"]])
    gravity = np.column_stack([archive["gravity_x_ms2"], archive["gravity_y_ms2"], archive["gravity_z_ms2"]])
    gyro = np.column_stack([archive["gyro_x_rads"], archive["gyro_y_rads"], archive["gyro_z_rads"]])
    linear = acc - gravity
    gravity_unit = gravity / np.maximum(np.linalg.norm(gravity, axis=1)[:, None], 1e-6)
    linear_vertical = np.sum(linear * gravity_unit, axis=1)
    linear_horizontal_vector = linear - linear_vertical[:, None] * gravity_unit
    gyro_vertical = np.sum(gyro * gravity_unit, axis=1)
    gyro_horizontal_vector = gyro - gyro_vertical[:, None] * gravity_unit
    angles = np.column_stack([
        archive["orientation_yaw_deg"],
        archive["orientation_pitch_deg"],
        archive["orientation_roll_deg"],
    ])
    angles = pd.DataFrame(angles).ffill().fillna(0).to_numpy()
    world_linear = Rotation.from_euler("zxy", angles, degrees=True).apply(linear)

    base = pd.DataFrame({
        "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
        "gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
        "linear_x": linear[:, 0], "linear_y": linear[:, 1], "linear_z": linear[:, 2],
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
    })
    generated = {}
    for name in ["linear_x", "linear_y", "linear_z", "gyro_x", "gyro_y", "gyro_z"]:
        delta = base[name].diff().fillna(0)
        generated[f"{name}_delta"] = delta
        generated[f"{name}_rate"] = delta / 0.1
    rolling_names = [
        "linear_x", "linear_y", "linear_z", "gyro_x", "gyro_y", "gyro_z",
        "linear_norm", "gyro_norm", "linear_vertical", "linear_horizontal",
        "world_linear_x", "world_linear_y", "world_linear_z",
    ]
    for window in windows:
        for name in rolling_names:
            rolling = base[name].rolling(window, min_periods=window)
            generated[f"{name}_mean_{window}"] = rolling.mean()
            generated[f"{name}_std_{window}"] = rolling.std(ddof=0)
            generated[f"{name}_range_{window}"] = rolling.max() - rolling.min()
        generated[f"linear_energy_{window}"] = base["linear_norm"].pow(2).rolling(window, min_periods=window).mean()
        generated[f"gyro_energy_{window}"] = base["gyro_norm"].pow(2).rolling(window, min_periods=window).mean()
    return pd.concat([base, pd.DataFrame(generated)], axis=1).astype(np.float32)


def load_examples(args: argparse.Namespace, windows: list[int]):
    config = json.loads((args.dataset_dir / "config.json").read_text())
    validation = json.loads((args.dataset_dir / "validation.json").read_text())
    if validation["status"] != "passed":
        raise RuntimeError("Prepared dataset validation has not passed")
    sessions = select_sessions(args)
    if args.smoke:
        sessions = sessions.sort_values(["active_split", "session_key"]).groupby(
            "active_split", as_index=False
        ).head(1)
    horizon_rows = round(args.horizon * 10)
    history_rows = max(windows)
    parts = {name: [[], []] for name in ("train", "validation", "test")}
    feature_names = None
    row_counts = {name: 0 for name in parts}

    for session in sessions.itertuples(index=False):
        with np.load(session.archive_path) as source:
            archive = {name: source[name] for name in source.files}
        if args.max_rows_per_session:
            archive = {name: value[: args.max_rows_per_session] for name, value in archive.items()}
        for segment_id in np.unique(archive["segment_id"]):
            index = np.flatnonzero(archive["segment_id"] == segment_id)
            if len(index) <= history_rows:
                continue
            local = {name: value[index] for name, value in archive.items()}
            features = build_features(local, windows)
            velocity_ms = local["ecu_velocity_kmh"].astype(float) / 3.6
            target = np.full(len(index), np.nan, dtype=np.float32)
            target[horizon_rows:] = velocity_ms[horizon_rows:] - velocity_ms[:-horizon_rows]
            eligible = np.arange(history_rows, len(index), args.sample_stride)
            valid = features.iloc[eligible].notna().all(axis=1).to_numpy() & np.isfinite(target[eligible])
            eligible = eligible[valid]
            if not len(eligible):
                continue
            split = session.active_split
            parts[split][0].append(features.iloc[eligible].to_numpy(np.float32))
            parts[split][1].append(target[eligible])
            row_counts[split] += len(eligible)
            feature_names = features.columns.tolist()
    combined = {
        split: (np.concatenate(values[0]), np.concatenate(values[1]))
        for split, values in parts.items()
    }
    return combined, feature_names, row_counts, config


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mae_delta_ms": float(mean_absolute_error(actual, predicted)),
        "rmse_delta_ms": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "zero_delta_baseline_mae_ms": float(np.mean(np.abs(actual))),
        "improvement_over_zero_percent": float(
            (1 - mean_absolute_error(actual, predicted) / np.mean(np.abs(actual))) * 100
        ),
    }


def summarize_blackouts(episodes: pd.DataFrame, samples: pd.DataFrame) -> dict:
    summary = {}
    for duration, group in episodes.groupby("duration_seconds"):
        errors = samples.loc[
            (samples["duration_seconds"] == duration) & (samples["elapsed_seconds"] > 0),
            "velocity_error_kmh",
        ].to_numpy(float)
        finite_drift = group["distance_drift_percent"].replace([np.inf, -np.inf], np.nan)
        summary[str(int(duration))] = {
            "episodes": int(len(group)),
            "velocity_mae_kmh": float(np.mean(np.abs(errors))),
            "velocity_rmse_kmh": float(np.sqrt(np.mean(errors**2))),
            "final_velocity_absolute_error_kmh": float(group["final_velocity_absolute_error_kmh"].mean()),
            "final_velocity_error_bias_kmh": float(group["final_velocity_error_kmh"].mean()),
            "within_5_kmh_percent": float(np.mean(np.abs(errors) <= 5) * 100),
            "absolute_distance_error_m": float(group["absolute_distance_error_m"].mean()),
            "distance_error_bias_m": float(group["distance_error_m"].mean()),
            "distance_drift_percent": float(finite_drift.mean()),
        }
    return summary


def evaluate_blackouts(
    args: argparse.Namespace,
    model: CatBoostRegressor,
    windows: list[int],
    durations: list[int],
    split: str,
    model_name: str,
    hold_speed: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Roll out non-overlapping episodes; truth is never passed to the model."""
    sessions = select_sessions(args)
    sessions = sessions.loc[sessions["active_split"] == split]
    if args.smoke:
        sessions = sessions.sort_values("session_key").head(1)
    horizon_rows = round(args.horizon * 10)
    history_rows = max(windows)
    episode_records: list[dict] = []
    sample_records: list[dict] = []

    for session in sessions.itertuples(index=False):
        with np.load(session.archive_path) as source:
            archive = {name: source[name] for name in source.files}
        if args.max_rows_per_session:
            archive = {name: value[: args.max_rows_per_session] for name, value in archive.items()}
        for segment_id in np.unique(archive["segment_id"]):
            rows = np.flatnonzero(archive["segment_id"] == segment_id)
            if len(rows) <= history_rows:
                continue
            local = {name: value[rows] for name, value in archive.items()}
            features = build_features(local, windows)
            true_speed_ms = local["ecu_velocity_kmh"].astype(float) / 3.6
            elapsed = local["phone_elapsed_seconds"].astype(float)
            for duration in durations:
                duration_rows = duration * 10
                last_start = len(rows) - duration_rows - 1
                if history_rows > last_start:
                    continue
                for start in range(history_rows, last_start + 1, duration_rows):
                    end = start + duration_rows
                    endpoints = np.arange(start + horizon_rows, end + 1, horizon_rows)
                    if hold_speed:
                        deltas = np.zeros(len(endpoints), dtype=float)
                    else:
                        deltas = np.asarray(model.predict(features.iloc[endpoints]), dtype=float)
                    predicted = np.empty(len(endpoints) + 1, dtype=float)
                    predicted[0] = true_speed_ms[start]  # the one permitted GNSS/ECU initialization
                    for position, delta in enumerate(deltas, start=1):
                        predicted[position] = max(0.0, predicted[position - 1] + delta)
                    truth_rows = np.r_[start, endpoints]
                    truth = true_speed_ms[truth_rows]
                    times = elapsed[truth_rows] - elapsed[start]
                    errors = (predicted - truth) * 3.6
                    predicted_distance = float(np.trapezoid(predicted, x=times))
                    dense_times = elapsed[start : end + 1] - elapsed[start]
                    true_distance = float(np.trapezoid(true_speed_ms[start : end + 1], x=dense_times))
                    distance_error = predicted_distance - true_distance
                    episode_id = f"{model_name}:{session.driver_id}:{session.session_key}:{segment_id}:{duration}:{start}"
                    episode_records.append({
                        "model": model_name,
                        "episode_id": episode_id,
                        "driver_id": session.driver_id,
                        "session_key": session.session_key,
                        "segment_id": int(segment_id),
                        "duration_seconds": duration,
                        "start_archive_row": int(rows[start]),
                        "end_archive_row": int(rows[end]),
                        "velocity_mae_kmh": float(np.mean(np.abs(errors[1:]))),
                        "velocity_rmse_kmh": float(np.sqrt(np.mean(errors[1:] ** 2))),
                        "final_velocity_error_kmh": float(errors[-1]),
                        "final_velocity_absolute_error_kmh": float(abs(errors[-1])),
                        "within_5_kmh_percent": float(np.mean(np.abs(errors[1:]) <= 5) * 100),
                        "true_distance_m": true_distance,
                        "predicted_distance_m": predicted_distance,
                        "distance_error_m": distance_error,
                        "absolute_distance_error_m": abs(distance_error),
                        "distance_drift_percent": abs(distance_error) / true_distance * 100 if true_distance >= 1 else math.nan,
                    })
                    for row, seconds, actual, estimate, error in zip(truth_rows, times, truth, predicted, errors):
                        sample_records.append({
                            "model": model_name,
                            "episode_id": episode_id,
                            "duration_seconds": duration,
                            "elapsed_seconds": float(seconds),
                            "archive_row": int(rows[row]),
                            "true_speed_kmh": float(actual * 3.6),
                            "predicted_speed_kmh": float(estimate * 3.6),
                            "velocity_error_kmh": float(error),
                        })
    episodes = pd.DataFrame(episode_records)
    samples = pd.DataFrame(sample_records)
    if episodes.empty:
        raise ValueError(f"No {split} blackout episodes could be generated")
    return summarize_blackouts(episodes, samples), episodes, samples


def plot_outputs(
    episodes: pd.DataFrame,
    samples: pd.DataFrame,
    summary: dict,
    baseline: dict,
    importance: pd.DataFrame,
    output_dir: Path,
) -> None:
    durations = sorted(episodes["duration_seconds"].unique())
    figure, axes = plt.subplots(len(durations), 1, figsize=(13, 3.1 * len(durations)), squeeze=False)
    for axis, duration in zip(axes[:, 0], durations):
        candidates = episodes.loc[episodes["duration_seconds"] == duration].sort_values("velocity_mae_kmh")
        episode_id = candidates.iloc[len(candidates) // 2]["episode_id"]
        shown = samples.loc[samples["episode_id"] == episode_id]
        axis.plot(shown["elapsed_seconds"], shown["true_speed_kmh"], label="Ground truth")
        axis.plot(shown["elapsed_seconds"], shown["predicted_speed_kmh"], label="Stateful IMU estimate")
        axis.set_title(f"Representative {duration:g} s blackout (median episode MAE)")
        axis.set_ylabel("Velocity (km/h)")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time since blackout start (s)")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "blackout_velocity_examples.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5))
    grouped = samples.loc[samples["elapsed_seconds"] > 0].assign(
        absolute_error=lambda value: value["velocity_error_kmh"].abs()
    ).groupby(["duration_seconds", "elapsed_seconds"])["absolute_error"].mean().reset_index()
    for duration, values in grouped.groupby("duration_seconds"):
        axis.plot(values["elapsed_seconds"], values["absolute_error"], label=f"{duration:g} s")
    axis.set(xlabel="Time since blackout start (s)", ylabel="Mean absolute velocity error (km/h)", title="Velocity error accumulation")
    axis.grid(alpha=0.25); axis.legend(title="Blackout")
    figure.tight_layout(); figure.savefig(output_dir / "velocity_error_over_time.png", dpi=140); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 7))
    for duration, values in episodes.groupby("duration_seconds"):
        axis.scatter(values["true_distance_m"], values["predicted_distance_m"], label=f"{duration:g} s", alpha=0.65)
    maximum = max(episodes["true_distance_m"].max(), episodes["predicted_distance_m"].max())
    axis.plot([0, maximum], [0, maximum], "k--", linewidth=1)
    axis.set(xlabel="Ground-truth distance (m)", ylabel="Predicted distance (m)", title="Distance travelled during held-out blackouts")
    axis.grid(alpha=0.25); axis.legend()
    figure.tight_layout(); figure.savefig(output_dir / "distance_ground_truth_vs_prediction.png", dpi=140); plt.close(figure)

    x = sorted(int(value) for value in summary)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x, [summary[str(v)]["velocity_mae_kmh"] for v in x], "o-", label="Delta-v model")
    axes[0].plot(x, [baseline[str(v)]["velocity_mae_kmh"] for v in x], "o-", label="Hold last speed")
    axes[0].set_ylabel("Velocity MAE (km/h)")
    axes[1].plot(x, [summary[str(v)]["distance_drift_percent"] for v in x], "o-", label="Delta-v model")
    axes[1].plot(x, [baseline[str(v)]["distance_drift_percent"] for v in x], "o-", label="Hold last speed")
    axes[1].set_ylabel("Mean distance drift (%)")
    for axis in axes:
        axis.set_xlabel("Blackout duration (s)"); axis.grid(alpha=0.25); axis.legend()
    figure.tight_layout(); figure.savefig(output_dir / "error_drift_vs_duration.png", dpi=140); plt.close(figure)

    top = importance.head(25).sort_values("importance")
    figure, axis = plt.subplots(figsize=(9, 8)); axis.barh(top["feature"], top["importance"])
    axis.set(xlabel="CatBoost importance", title="Delta-v feature importance")
    figure.tight_layout(); figure.savefig(output_dir / "delta_v_feature_importance.png", dpi=140); plt.close(figure)


def write_report(report: dict, output_dir: Path) -> None:
    held_out_label = "Driver B" if report["split_mode"] == "driver" else f"Driver {report['selected_driver']} whole-drive test partition"
    lines = [
        "# ECU-supervised stateful velocity experiment", "",
        f"Delta-v horizon: **{report['horizon_seconds']:.1f} s**.", "",
        "Each blackout starts from the last valid ground-truth speed. All later model inputs are causal smartphone IMU features only.", "",
        f"## Held-out {held_out_label} blackouts", "",
        "| Duration | Episodes | Velocity MAE | Velocity RMSE | Final abs. error | Within +/-5 | Distance abs. error | Distance drift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for duration, value in report["test_blackouts"].items():
        lines.append(f"| {duration}s | {value['episodes']} | {value['velocity_mae_kmh']:.2f} km/h | {value['velocity_rmse_kmh']:.2f} km/h | {value['final_velocity_absolute_error_kmh']:.2f} km/h | {value['within_5_kmh_percent']:.1f}% | {value['absolute_distance_error_m']:.1f} m | {value['distance_drift_percent']:.1f}% |")
    split_statement = (
        "- Driver E trains the model, Driver A controls early stopping, and Driver B is untouched until final evaluation."
        if report["split_mode"] == "driver"
        else f"- Complete Driver {report['selected_driver']} sessions are assigned to train, validation, and test; rows are never randomly split."
    )
    lines += ["", "## Leakage controls", "", split_statement, "- Causal rolling features are rebuilt independently inside each continuous segment.", "- The first 100 rows of each segment are purged, covering the maximum feature history.", "- ECU/GNSS speed is used only as the offline target, initial blackout speed, and evaluation truth.", "- The deployment refit excludes the whole-drive test partition.", ""]
    (output_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def model_for(args: argparse.Namespace, iterations: int | None = None) -> CatBoostRegressor:
    count = iterations or args.iterations
    return CatBoostRegressor(
        iterations=count,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="RMSE",
        eval_metric="MAE",
        random_seed=args.seed,
        l2_leaf_reg=6,
        random_strength=0.5,
        task_type=args.task_type,
        thread_count=args.threads,
        allow_writing_files=False,
        verbose=max(1, count // 25),
    )


def main() -> None:
    args = arguments()
    if args.smoke:
        args.iterations = min(args.iterations, 30)
        args.early_stopping_rounds = min(args.early_stopping_rounds, 10)
        args.max_rows_per_session = args.max_rows_per_session or 5000
        if args.output_dir == DEFAULT_OUTPUT:
            args.output_dir = Path("artifacts/ecu_delta_v_smoke")
    windows = sorted({int(value) for value in args.rolling_windows.split(",")})
    durations = sorted({int(value) for value in args.blackout_durations.split(",")})
    started = time.monotonic()
    data, feature_names, row_counts, dataset_config = load_examples(args, windows)
    train_x, train_y = data["train"]
    validation_x, validation_y = data["validation"]
    test_x, test_y = data["test"]
    model = model_for(args)
    model.fit(
        train_x,
        train_y,
        eval_set=(validation_x, validation_y),
        early_stopping_rounds=args.early_stopping_rounds,
        use_best_model=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output_dir / "evaluation_delta_v_model.cbm")
    validation_prediction = model.predict(validation_x)
    test_prediction = model.predict(test_x)
    best_iteration = max(0, model.get_best_iteration())

    deployment = model_for(args, best_iteration + 1)
    deployment.fit(np.concatenate([train_x, validation_x]), np.concatenate([train_y, validation_y]))
    deployment.save_model(args.output_dir / "delta_v_model.cbm")
    joblib.dump(
        {
            "feature_names": feature_names,
            "rolling_windows": windows,
            "horizon_seconds": args.horizon,
            "history_rows": max(windows),
            "orientation_angles_direct_features": False,
            "reference_features": [],
        },
        args.output_dir / "preprocessing.joblib",
    )
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    test_blackouts, blackout_episodes, blackout_samples = evaluate_blackouts(
        args, model, windows, durations, "test", "delta_v_model"
    )
    constant_blackouts, constant_episodes, constant_samples = evaluate_blackouts(
        args, model, windows, durations, "test", "hold_last_speed", hold_speed=True
    )
    blackout_episodes.to_csv(args.output_dir / "blackout_episodes.csv", index=False)
    blackout_samples.to_csv(args.output_dir / "blackout_velocity_samples.csv", index=False)
    constant_episodes.to_csv(args.output_dir / "constant_speed_blackout_episodes.csv", index=False)
    constant_samples.to_csv(args.output_dir / "constant_speed_velocity_samples.csv", index=False)
    report = {
        "experiment": "ECU-supervised CatBoost delta velocity",
        "dataset_dir": str(args.dataset_dir),
        "dataset_validation": "passed",
        "horizon_seconds": args.horizon,
        "split": (
            "driver holdout: E train, A validation, B test"
            if args.split_mode == "driver"
            else f"whole-drive split within Driver {args.driver}"
        ),
        "split_mode": args.split_mode,
        "selected_driver": args.driver,
        "row_counts": row_counts,
        "feature_count": len(feature_names),
        "best_iteration": best_iteration,
        "validation": metrics(validation_y, validation_prediction),
        "test": metrics(test_y, test_prediction),
        "test_blackouts": test_blackouts,
        "constant_speed_blackout_baseline": constant_blackouts,
        "blackout_initialization": "one ground-truth speed at episode start",
        "ground_truth_speed_used_as_post_initialization_feature": False,
        "purge": {
            "rows_at_start_of_each_continuous_segment": max(windows),
            "covers_maximum_causal_history": True,
        },
        "deployment_refit": "train plus validation; test excluded",
        "elapsed_seconds": time.monotonic() - started,
        "source_target": dataset_config["target"],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_outputs(blackout_episodes, blackout_samples, test_blackouts, constant_blackouts, importance, args.output_dir)
    write_report(report, args.output_dir)
    print("Training complete")
    print(f"Rows: {row_counts}")
    print(f"Features: {len(feature_names)}; best iteration: {best_iteration}")
    print(f"Validation: {report['validation']}")
    print(f"Test: {report['test']}")
    for duration in durations:
        result = test_blackouts[str(duration)]
        baseline = constant_blackouts[str(duration)]
        print(
            f"{duration:>3}s blackout: velocity MAE={result['velocity_mae_kmh']:.2f} km/h "
            f"(hold={baseline['velocity_mae_kmh']:.2f}), distance error={result['absolute_distance_error_m']:.1f} m"
        )
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
