#!/usr/bin/env python3
"""Select causal INS drift controls on Driver A and evaluate once on Driver B."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_unrolled_ins import (
    BlackoutDataset, ContextUnrolledINS, build_examples, collate, device_for,
    fit_scaler, load_sessions, to_device,
)


DEFAULT_MODEL = Path("artifacts/context_unrolled_ins/evaluation_model.pt")
DEFAULT_OUTPUT = Path("artifacts/hybrid_drift_controlled_ins")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/ecu_training_dataset_inertial_aligned"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blackout-durations", default="10,30,60,120")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@dataclass
class Episode:
    driver: str
    session_key: str
    start: int
    duration: int
    initial_speed: float
    initial_heading: float
    acceleration: np.ndarray
    yaw: np.ndarray
    calibrated_acceleration: np.ndarray
    gyro_norm: np.ndarray
    true_speed: np.ndarray
    true_heading: np.ndarray
    true_east: np.ndarray
    true_north: np.ndarray
    true_distance: np.ndarray


@torch.no_grad()
def infer(model: torch.nn.Module, loader: DataLoader, device: torch.device,
          mean: np.ndarray, std: np.ndarray) -> list[Episode]:
    model.eval(); result: list[Episode] = []
    for raw in loader:
        batch = to_device(raw, device)
        acceleration, yaw = model(batch["context_imu"], batch["context_reference"], batch["blackout_imu"])
        for index, length_value in enumerate(raw["lengths"].tolist()):
            length = int(length_value)
            normalized_gyro_norm = raw["blackout_imu"][index, :length, 16].numpy()
            result.append(Episode(
                driver=str(raw["driver"][index]), session_key=str(raw["session_key"][index]),
                start=int(raw["start"][index]), duration=length // 10,
                initial_speed=float(raw["initial_speed"][index]), initial_heading=float(raw["initial_heading"][index]),
                acceleration=acceleration[index, :length].cpu().numpy(), yaw=yaw[index, :length].cpu().numpy(),
                calibrated_acceleration=raw["blackout_imu"][index, :length, -2].numpy() * 4.0,
                gyro_norm=normalized_gyro_norm * float(std[16]) + float(mean[16]),
                true_speed=raw["true_speed"][index, :length].numpy(),
                true_heading=raw["true_heading"][index, :length].numpy(),
                true_east=raw["true_east"][index, :length].numpy(),
                true_north=raw["true_north"][index, :length].numpy(),
                true_distance=raw["true_distance"][index, :length].numpy(),
            ))
    return result


def ema(values: np.ndarray, seconds: float) -> np.ndarray:
    if seconds <= 0:
        return values.copy()
    alpha = 1.0 - math.exp(-0.1 / seconds)
    output = np.empty_like(values); state = float(values[0])
    for index, value in enumerate(values):
        state += alpha * (float(value) - state); output[index] = state
    return output


def estimate(episode: Episode, speed: dict, yaw: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = len(episode.acceleration); elapsed = np.arange(1, length + 1) * 0.1
    acceleration = ema(episode.acceleration, speed["acceleration_smoothing_seconds"])
    model_delta = np.cumsum(acceleration * 0.1)
    limit = speed["maximum_correction_kmh"] / 3.6
    model_delta = np.clip(model_delta, -limit, limit)
    decay = np.ones(length) if math.isinf(speed["decay_seconds"]) else np.exp(-elapsed / speed["decay_seconds"])
    predicted_speed = np.maximum(0.0, episode.initial_speed + speed["gain"] * decay * model_delta)
    if speed["stationary_constraint"] and episode.initial_speed < 0.75:
        quiet = (np.abs(ema(episode.calibrated_acceleration, 0.5)) < 0.18) & (ema(episode.gyro_norm, 0.5) < 0.04)
        predicted_speed[quiet] = 0.0

    yaw_rate = ema(episode.yaw, yaw["smoothing_seconds"]) * yaw["gain"]
    predicted_heading = episode.initial_heading + np.cumsum(yaw_rate * 0.1)
    predicted_east = np.cumsum(predicted_speed * np.sin(predicted_heading) * 0.1)
    predicted_north = np.cumsum(predicted_speed * np.cos(predicted_heading) * 0.1)
    predicted_distance = np.cumsum(predicted_speed * 0.1)
    return predicted_speed, predicted_heading, predicted_east, predicted_north, predicted_distance


def speed_score(episodes: list[Episode], parameters: dict) -> tuple[float, dict]:
    by_duration: dict[int, dict[str, list[float] | float]] = {}
    for episode in episodes:
        predicted, _, _, _, distance = estimate(episode, parameters, {"gain": 1.0, "smoothing_seconds": 0.0})
        group = by_duration.setdefault(episode.duration, {"absolute_velocity": [], "distance_error": [], "true_distance": []})
        group["absolute_velocity"].extend(np.abs(predicted - episode.true_speed) * 3.6)
        group["distance_error"].append(abs(float(distance[-1] - episode.true_distance[-1])))
        group["true_distance"].append(float(episode.true_distance[-1]))
    details = {}; components = []
    for duration, group in by_duration.items():
        velocity_mae = float(np.mean(group["absolute_velocity"]))
        aggregate_drift = float(np.sum(group["distance_error"]) / max(np.sum(group["true_distance"]), 1.0) * 100)
        details[str(duration)] = {"velocity_mae_kmh": velocity_mae, "aggregate_distance_drift_percent": aggregate_drift}
        components.append(velocity_mae / 10.0 + aggregate_drift / 100.0)
    return float(np.mean(components)), details


def choose_speed(episodes: list[Episode]) -> tuple[dict, pd.DataFrame]:
    candidates = []
    for gain in (0.0, 0.25, 0.5, 0.75, 1.0):
        for decay in (20.0, 40.0, 80.0, math.inf):
            for correction in (5.0, 10.0, 20.0, math.inf):
                for smoothing in (0.0, 0.5, 1.0):
                    for stationary in (False, True):
                        if gain == 0 and (decay != 20 or correction != 5 or smoothing != 0 or stationary):
                            continue
                        parameters = {"gain": gain, "decay_seconds": decay,
                                      "maximum_correction_kmh": correction,
                                      "acceleration_smoothing_seconds": smoothing,
                                      "stationary_constraint": stationary}
                        score, details = speed_score(episodes, parameters)
                        candidates.append({**parameters, "validation_score": score,
                                           "validation_metrics": json.dumps(details, sort_keys=True)})
    frame = pd.DataFrame(candidates).sort_values("validation_score")
    selected = frame.iloc[0].to_dict(); selected.pop("validation_score"); selected.pop("validation_metrics")
    return selected, frame


def yaw_score(episodes: list[Episode], speed: dict, yaw: dict) -> tuple[float, dict]:
    by_duration: dict[int, dict[str, list[float]]] = {}
    for episode in episodes:
        _, _, east, north, _ = estimate(episode, speed, yaw)
        error = float(np.hypot(east[-1] - episode.true_east[-1], north[-1] - episode.true_north[-1]))
        group = by_duration.setdefault(episode.duration, {"error": [], "distance": []})
        group["error"].append(error); group["distance"].append(float(episode.true_distance[-1]))
    details = {}; components = []
    for duration, group in by_duration.items():
        drift = float(np.sum(group["error"]) / max(np.sum(group["distance"]), 1.0) * 100)
        details[str(duration)] = {"aggregate_endpoint_position_drift_percent": drift}
        components.append(drift)
    return float(np.mean(components)), details


def choose_yaw(episodes: list[Episode], speed: dict) -> tuple[dict, pd.DataFrame]:
    candidates = []
    for gain in (0.5, 0.75, 1.0, 1.25):
        for smoothing in (0.0, 0.25, 0.5, 1.0):
            parameters = {"gain": gain, "smoothing_seconds": smoothing}
            score, details = yaw_score(episodes, speed, parameters)
            candidates.append({**parameters, "validation_score": score,
                               "validation_metrics": json.dumps(details, sort_keys=True)})
    frame = pd.DataFrame(candidates).sort_values("validation_score")
    selected = frame.iloc[0].to_dict(); selected.pop("validation_score"); selected.pop("validation_metrics")
    return selected, frame


def evaluate(episodes: list[Episode], speed: dict, yaw: dict, model_name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    episode_rows, sample_rows = [], []
    for episode in episodes:
        predicted_speed, _, east, north, distance = estimate(episode, speed, yaw)
        error = (predicted_speed - episode.true_speed) * 3.6
        endpoint = float(np.hypot(east[-1] - episode.true_east[-1], north[-1] - episode.true_north[-1]))
        true_distance = float(episode.true_distance[-1]); distance_error = float(distance[-1] - true_distance)
        episode_id = f"{model_name}:{episode.driver}:{episode.session_key}:{episode.duration}:{episode.start}"
        episode_rows.append({"model": model_name, "episode_id": episode_id, "driver_id": episode.driver,
            "session_key": episode.session_key, "start_row": episode.start, "duration_seconds": episode.duration,
            "velocity_mae_kmh": float(np.mean(abs(error))), "velocity_rmse_kmh": float(np.sqrt(np.mean(error ** 2))),
            "final_velocity_error_kmh": float(error[-1]), "final_velocity_absolute_error_kmh": float(abs(error[-1])),
            "within_5_kmh_percent": float(np.mean(abs(error) <= 5) * 100), "true_distance_m": true_distance,
            "predicted_distance_m": float(distance[-1]), "distance_error_m": distance_error,
            "absolute_distance_error_m": abs(distance_error), "distance_drift_percent": abs(distance_error) / max(true_distance, 1) * 100,
            "endpoint_position_error_m": endpoint, "endpoint_position_drift_percent": endpoint / max(true_distance, 1) * 100})
        for index in range(len(error)):
            sample_rows.append({"model": model_name, "episode_id": episode_id, "duration_seconds": episode.duration,
                "elapsed_seconds": (index + 1) / 10, "true_speed_kmh": float(episode.true_speed[index] * 3.6),
                "predicted_speed_kmh": float(predicted_speed[index] * 3.6), "velocity_error_kmh": float(error[index]),
                "true_east_m": float(episode.true_east[index]), "true_north_m": float(episode.true_north[index]),
                "predicted_east_m": float(east[index]), "predicted_north_m": float(north[index])})
    frame, samples = pd.DataFrame(episode_rows), pd.DataFrame(sample_rows); summary = {}
    for duration, group in frame.groupby("duration_seconds"):
        errors = samples[samples.duration_seconds == duration].velocity_error_kmh.to_numpy()
        summary[str(int(duration))] = {"episodes": int(len(group)), "velocity_mae_kmh": float(np.mean(abs(errors))),
            "velocity_rmse_kmh": float(np.sqrt(np.mean(errors ** 2))),
            "final_velocity_absolute_error_kmh": float(group.final_velocity_absolute_error_kmh.mean()),
            "within_5_kmh_percent": float(np.mean(abs(errors) <= 5) * 100),
            "absolute_distance_error_m": float(group.absolute_distance_error_m.mean()),
            "median_distance_drift_percent": float(group.distance_drift_percent.median()),
            "aggregate_distance_drift_percent": float(group.absolute_distance_error_m.sum() / max(group.true_distance_m.sum(), 1) * 100),
            "endpoint_position_error_m": float(group.endpoint_position_error_m.mean()),
            "median_endpoint_position_drift_percent": float(group.endpoint_position_drift_percent.median()),
            "aggregate_endpoint_position_drift_percent": float(group.endpoint_position_error_m.sum() / max(group.true_distance_m.sum(), 1) * 100)}
    return summary, frame, samples


def plot(summary: dict, hold: dict, samples: pd.DataFrame, episodes: pd.DataFrame, output: Path) -> None:
    durations = sorted(int(x) for x in summary); figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fields = (("velocity_mae_kmh", "Velocity MAE (km/h)"),
              ("aggregate_distance_drift_percent", "Aggregate distance drift (%)"),
              ("aggregate_endpoint_position_drift_percent", "Aggregate position drift (%)"))
    for axis, (field, label) in zip(axes, fields):
        axis.plot(durations, [summary[str(d)][field] for d in durations], "o-", label="Hybrid")
        axis.plot(durations, [hold[str(d)][field] for d in durations], "o--", label="Hold")
        axis.set(xlabel="Blackout duration (s)", ylabel=label); axis.grid(alpha=.25); axis.legend()
    figure.tight_layout(); figure.savefig(output / "hybrid_vs_hold.png", dpi=140); plt.close(figure)
    figure, axes = plt.subplots(len(durations), 1, figsize=(12, 3 * len(durations)), squeeze=False)
    for axis, duration in zip(axes[:, 0], durations):
        choices = episodes[episodes.duration_seconds == duration].sort_values("velocity_mae_kmh")
        shown = samples[samples.episode_id == choices.iloc[len(choices) // 2].episode_id]
        axis.plot(shown.elapsed_seconds, shown.true_speed_kmh, label="Ground truth")
        axis.plot(shown.elapsed_seconds, shown.predicted_speed_kmh, label="Hybrid")
        axis.set_title(f"Representative {duration}s blackout"); axis.set_ylabel("km/h"); axis.grid(alpha=.25)
    axes[-1, 0].set_xlabel("Blackout time (s)"); axes[0, 0].legend(); figure.tight_layout()
    figure.savefig(output / "hybrid_velocity_examples.png", dpi=140); plt.close(figure)


def main() -> None:
    args = arguments(); np.random.seed(args.seed)
    if args.threads > 0: torch.set_num_threads(args.threads)
    device = device_for(args); args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = ContextUnrolledINS(config["imu_features"], config["hidden"], config["layers"], config["dropout"])
    model.load_state_dict(checkpoint["model_state"]); model.to(device)
    class LoadArgs: pass
    load_args = LoadArgs(); load_args.dataset_dir = args.dataset_dir; load_args.smoke = False; load_args.max_rows_per_session = None
    sessions, _ = load_sessions(load_args, checkpoint["bias_rows"])
    mean, std = checkpoint["mean"], checkpoint["std"]
    durations = sorted({int(x) for x in args.blackout_durations.split(",")})
    def make(split: str) -> BlackoutDataset:
        examples = []
        for duration in durations:
            examples += build_examples(sessions, split, [duration], checkpoint["context_rows"], checkpoint["bias_rows"], duration * 10)
        return BlackoutDataset(sessions, examples, checkpoint["context_rows"], mean, std)
    validation_dataset, test_dataset = make("validation"), make("test")
    print(f"device={device}; episodes: validation={len(validation_dataset)}, test={len(test_dataset)}", flush=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                   collate_fn=collate, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers,
                             collate_fn=collate, pin_memory=device.type == "cuda")
    print("running Driver A inference and selecting drift controls...", flush=True)
    validation_episodes = infer(model, validation_loader, device, mean, std)
    selected_speed, speed_candidates = choose_speed(validation_episodes)
    selected_yaw, yaw_candidates = choose_yaw(validation_episodes, selected_speed)
    speed_candidates.to_csv(args.output_dir / "speed_control_selection.csv", index=False)
    yaw_candidates.to_csv(args.output_dir / "yaw_control_selection.csv", index=False)
    validation_summary, _, _ = evaluate(validation_episodes, selected_speed, selected_yaw, "hybrid")
    print(f"selected speed={selected_speed}; yaw={selected_yaw}", flush=True)
    print("parameters frozen; running Driver B once...", flush=True)
    test_episodes = infer(model, test_loader, device, mean, std)
    test_summary, episode_frame, samples = evaluate(test_episodes, selected_speed, selected_yaw, "hybrid")
    hold_speed = {"gain": 0.0, "decay_seconds": 20.0, "maximum_correction_kmh": 5.0,
                  "acceleration_smoothing_seconds": 0.0, "stationary_constraint": False}
    hold_yaw = {"gain": 0.0, "smoothing_seconds": 0.0}
    hold_summary, hold_episodes, hold_samples = evaluate(test_episodes, hold_speed, hold_yaw, "hold_speed_heading")
    episode_frame.to_csv(args.output_dir / "blackout_episodes.csv", index=False)
    samples.to_csv(args.output_dir / "blackout_samples.csv", index=False)
    hold_episodes.to_csv(args.output_dir / "hold_blackout_episodes.csv", index=False)
    hold_samples.to_csv(args.output_dir / "hold_blackout_samples.csv", index=False)
    metrics = {"experiment": "validation-selected hybrid drift-controlled INS", "source_model": str(args.model),
               "selection_split": "Driver A", "evaluation_split": "Driver B", "selected_speed_control": selected_speed,
               "selected_yaw_control": selected_yaw, "validation_blackouts": validation_summary,
               "test_blackouts": test_summary, "hold_baseline": hold_summary,
               "test_ground_truth_used_for_parameter_selection": False,
               "caveat": "Driver B has been inspected in earlier experiments and is not a pristine final benchmark."}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot(test_summary, hold_summary, samples, episode_frame, args.output_dir)
    lines = ["# Hybrid drift-controlled INS", "", f"Speed control: `{selected_speed}`", f"Yaw control: `{selected_yaw}`", "",
             "All parameters were selected on Driver A before applying the frozen estimator to Driver B.", "",
             "| Duration | Velocity MAE | Hold MAE | Aggregate distance drift | Aggregate position drift |", "|---:|---:|---:|---:|---:|"]
    for duration, value in test_summary.items():
        hold = hold_summary[duration]; lines.append(f"| {duration}s | {value['velocity_mae_kmh']:.2f} km/h | {hold['velocity_mae_kmh']:.2f} km/h | {value['aggregate_distance_drift_percent']:.1f}% | {value['aggregate_endpoint_position_drift_percent']:.1f}% |")
    (args.output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n")
    print("Hybrid evaluation complete")
    for duration in durations:
        value, hold = test_summary[str(duration)], hold_summary[str(duration)]
        print(f"{duration:>3}s: MAE={value['velocity_mae_kmh']:.2f} (hold={hold['velocity_mae_kmh']:.2f}) km/h, "
              f"distance={value['aggregate_distance_drift_percent']:.1f}%, position={value['aggregate_endpoint_position_drift_percent']:.1f}%")
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
