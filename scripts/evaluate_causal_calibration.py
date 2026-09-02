#!/usr/bin/env python3
"""Evaluate causal pre-blackout phone-axis calibration without test tuning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/ecu_training_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/causal_phone_calibration"))
    parser.add_argument("--calibration-seconds", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--durations", default="10,30,60,120")
    return parser.parse_args()


def features(archive: dict[str, np.ndarray], rows: np.ndarray, horizon: int) -> np.ndarray:
    linear = np.column_stack([
        archive["acc_x_ms2"][rows] - archive["gravity_x_ms2"][rows],
        archive["acc_y_ms2"][rows] - archive["gravity_y_ms2"][rows],
        archive["acc_z_ms2"][rows] - archive["gravity_z_ms2"][rows],
    ])
    frame = pd.DataFrame(linear)
    mean = frame.rolling(horizon, min_periods=horizon).mean().to_numpy()
    std = frame.rolling(horizon, min_periods=horizon).std(ddof=0).to_numpy()
    return np.column_stack([mean, std])


def evaluate(args: argparse.Namespace, split: str, durations: list[int]) -> tuple[dict, pd.DataFrame]:
    sessions = pd.read_csv(args.dataset_dir / "sessions.csv")
    sessions = sessions.loc[sessions["driver_holdout_split"] == split]
    horizon = 10
    calibration_rows = args.calibration_seconds * 10
    records = []
    for session in sessions.itertuples(index=False):
        with np.load(session.archive_path) as source:
            archive = {name: source[name] for name in source.files}
        for segment_id in np.unique(archive["segment_id"]):
            rows = np.flatnonzero(archive["segment_id"] == segment_id)
            if len(rows) <= calibration_rows + 100:
                continue
            x = features(archive, rows, horizon)
            speed = archive["ecu_velocity_kmh"][rows].astype(float) / 3.6
            elapsed = archive["phone_elapsed_seconds"][rows].astype(float)
            target = np.r_[np.full(horizon, np.nan), speed[horizon:] - speed[:-horizon]]
            for duration in durations:
                duration_rows = duration * 10
                first_start = max(100, calibration_rows + horizon)
                last_start = len(rows) - duration_rows - 1
                for start in range(first_start, last_start + 1, duration_rows):
                    calibration = np.arange(start - calibration_rows + horizon, start + 1)
                    endpoints = np.arange(start + horizon, start + duration_rows + 1, horizon)
                    model = Ridge(alpha=args.alpha).fit(x[calibration], target[calibration])
                    delta = model.predict(x[endpoints])
                    predicted = np.empty(len(endpoints) + 1)
                    predicted[0] = speed[start]
                    for position, change in enumerate(delta, start=1):
                        predicted[position] = max(0.0, predicted[position - 1] + change)
                    truth_rows = np.r_[start, endpoints]
                    truth = speed[truth_rows]
                    times = elapsed[truth_rows] - elapsed[start]
                    dense_times = elapsed[start : start + duration_rows + 1] - elapsed[start]
                    true_distance = float(np.trapezoid(speed[start : start + duration_rows + 1], x=dense_times))
                    predicted_distance = float(np.trapezoid(predicted, x=times))
                    for name, estimate in (("causal_ridge", predicted), ("hold_last_speed", np.full(len(truth), truth[0]))):
                        error = (estimate - truth) * 3.6
                        distance = predicted_distance if name == "causal_ridge" else float(truth[0] * times[-1])
                        distance_error = distance - true_distance
                        records.append({
                            "model": name,
                            "driver_id": session.driver_id,
                            "session_key": session.session_key,
                            "segment_id": int(segment_id),
                            "duration_seconds": duration,
                            "start_archive_row": int(rows[start]),
                            "velocity_mae_kmh": float(np.mean(np.abs(error[1:]))),
                            "velocity_rmse_kmh": float(np.sqrt(np.mean(error[1:] ** 2))),
                            "final_velocity_absolute_error_kmh": float(abs(error[-1])),
                            "within_5_kmh_percent": float(np.mean(np.abs(error[1:]) <= 5) * 100),
                            "absolute_distance_error_m": float(abs(distance_error)),
                            "distance_drift_percent": float(abs(distance_error) / true_distance * 100) if true_distance >= 1 else math.nan,
                        })
    episodes = pd.DataFrame(records)
    summary = {}
    for (model, duration), group in episodes.groupby(["model", "duration_seconds"]):
        summary.setdefault(model, {})[str(int(duration))] = {
            "episodes": int(len(group)),
            "mean_episode_velocity_mae_kmh": float(group["velocity_mae_kmh"].mean()),
            "mean_episode_velocity_rmse_kmh": float(group["velocity_rmse_kmh"].mean()),
            "final_velocity_absolute_error_kmh": float(group["final_velocity_absolute_error_kmh"].mean()),
            "within_5_kmh_percent": float(group["within_5_kmh_percent"].mean()),
            "absolute_distance_error_m": float(group["absolute_distance_error_m"].mean()),
            "distance_drift_percent": float(group["distance_drift_percent"].replace([np.inf, -np.inf], np.nan).mean()),
        }
    return summary, episodes


def main() -> None:
    args = arguments()
    durations = sorted({int(value) for value in args.durations.split(",")})
    validation, validation_episodes = evaluate(args, "validation", durations)
    test, test_episodes = evaluate(args, "test", durations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_episodes.to_csv(args.output_dir / "validation_episodes.csv", index=False)
    test_episodes.to_csv(args.output_dir / "test_episodes.csv", index=False)
    report = {
        "experiment": "causal pre-blackout phone-axis ridge calibration",
        "selection": "120 seconds and alpha 100 selected on Driver A 30-second blackouts",
        "calibration_seconds": args.calibration_seconds,
        "ridge_alpha": args.alpha,
        "features": "causal 1-second mean and standard deviation of gravity-corrected phone XYZ",
        "post_blackout_reference_features": [],
        "validation_driver_A": validation,
        "untouched_test_driver_B": test,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, label in (("causal_ridge", "Causal calibration"), ("hold_last_speed", "Hold speed")):
        axes[0].plot(durations, [test[model][str(v)]["mean_episode_velocity_mae_kmh"] for v in durations], "o-", label=label)
        axes[1].plot(durations, [test[model][str(v)]["distance_drift_percent"] for v in durations], "o-", label=label)
    axes[0].set_ylabel("Mean episode velocity MAE (km/h)")
    axes[1].set_ylabel("Mean distance drift (%)")
    for axis in axes:
        axis.set_xlabel("Blackout duration (s)"); axis.grid(alpha=0.25); axis.legend()
    figure.tight_layout(); figure.savefig(args.output_dir / "calibration_vs_hold.png", dpi=140); plt.close(figure)
    print(json.dumps(test, indent=2))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
