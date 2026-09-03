#!/usr/bin/env python3
"""Train a context-calibrated recurrent INS with differentiable blackout rollout.

The context encoder may see causal, 1 Hz-held GNSS state before a blackout.  The
blackout decoder receives smartphone IMU features only.  Speed, heading and
position are integrated inside the training graph so the loss penalizes drift,
not merely one-step acceleration error.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_causal_cnn import build_segment_features, device_for


DEFAULT_DATASET = Path("artifacts/ecu_training_dataset_inertial_aligned")
DEFAULT_OUTPUT = Path("artifacts/context_unrolled_ins")
EARTH_RADIUS_M = 6_371_000.0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-seconds", type=int, default=30)
    parser.add_argument("--bias-seconds", type=int, default=30)
    parser.add_argument("--train-durations", default="10,30,60")
    parser.add_argument("--blackout-durations", default="10,30,60,120")
    parser.add_argument("--train-stride-seconds", type=float, default=10.0)
    parser.add_argument("--eval-stride-factor", type=float, default=1.0,
                        help="Evaluation stride as a multiple of blackout duration")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-rounds", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-rows-per-session", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


@dataclass
class Session:
    driver: str
    key: str
    split: str
    features: np.ndarray
    speed: np.ndarray
    heading: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    elapsed: np.ndarray
    segment: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sessions(args: argparse.Namespace, bias_rows: int) -> tuple[list[Session], list[str]]:
    validation = json.loads((args.dataset_dir / "validation.json").read_text())
    if validation.get("status") != "passed":
        raise RuntimeError("Prepared dataset validation has not passed")
    table = pd.read_csv(args.dataset_dir / "sessions.csv")
    if args.smoke:
        table = table.sort_values(["driver_holdout_split", "session_key"]).groupby(
            "driver_holdout_split", as_index=False
        ).head(1)
    sessions: list[Session] = []
    feature_names: list[str] = []
    for number, record in enumerate(table.itertuples(index=False), 1):
        with np.load(record.archive_path) as source:
            archive = {name: source[name] for name in source.files}
        if args.max_rows_per_session:
            archive = {name: value[:args.max_rows_per_session] for name, value in archive.items()}
        features = np.full((len(archive["segment_id"]), 19), np.nan, np.float32)
        for segment_id in np.unique(archive["segment_id"]):
            rows = np.flatnonzero(archive["segment_id"] == segment_id)
            features[rows], feature_names = build_segment_features(archive, rows, bias_rows)
        sessions.append(Session(
            driver=str(record.driver_id), key=str(record.session_key),
            split=str(record.driver_holdout_split), features=features,
            speed=archive["ecu_velocity_kmh"].astype(np.float32) / 3.6,
            heading=np.unwrap(np.deg2rad(archive["ecu_heading_deg"].astype(np.float64))).astype(np.float32),
            latitude=archive["ecu_latitude_deg"].astype(np.float64),
            longitude=archive["ecu_longitude_deg"].astype(np.float64),
            elapsed=archive["phone_elapsed_seconds"].astype(np.float64),
            segment=archive["segment_id"].astype(np.int32),
        ))
        if number % 5 == 0 or number == len(table):
            print(f"loaded and causally featurized {number}/{len(table)} sessions", flush=True)
    return sessions, feature_names


def fit_scaler(sessions: list[Session]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([s.features for s in sessions if s.split == "train"])
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def build_examples(sessions: list[Session], split: str, durations: list[int], context_rows: int,
                   bias_rows: int, stride_rows: int) -> list[tuple[int, int, int]]:
    examples: list[tuple[int, int, int]] = []
    for session_index, session in enumerate(sessions):
        if session.split != split:
            continue
        for segment_id in np.unique(session.segment):
            rows = np.flatnonzero(session.segment == segment_id)
            first, last = int(rows[0]), int(rows[-1])
            earliest = first + bias_rows + context_rows
            for duration in durations:
                length = duration * 10
                for start in range(earliest, last - length + 2, stride_rows):
                    examples.append((session_index, start, length))
    return examples


def rotate_phone_frame(values: np.ndarray, angle: float) -> None:
    cosine, sine = math.cos(angle), math.sin(angle)
    for start in (0, 3, 6, 9, 12):
        x, y = values[:, start].copy(), values[:, start + 1].copy()
        values[:, start] = cosine * x - sine * y
        values[:, start + 1] = sine * x + cosine * y


def calibrated_projection(context_features: np.ndarray, blackout_features: np.ndarray,
                          fix_speed: np.ndarray, fix_heading: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a regularized phone-to-vehicle projection from pre-blackout fixes only."""
    blocks = len(fix_speed)
    if blocks < 3:
        return np.zeros((len(context_features), 2), np.float32), np.zeros((len(blackout_features), 2), np.float32)
    acceleration_axes = context_features[:, 3:6]  # causally de-biased linear acceleration
    gyro_axes = context_features[:, 9:12]
    acceleration_blocks = np.stack([acceleration_axes[i * 10:(i + 1) * 10].mean(0) for i in range(blocks)])
    gyro_blocks = np.stack([gyro_axes[i * 10:(i + 1) * 10].mean(0) for i in range(blocks)])
    target_acceleration = np.diff(fix_speed)  # one-second fix spacing
    target_yaw = np.arctan2(np.sin(np.diff(fix_heading)), np.cos(np.diff(fix_heading)))

    def fit_apply(x: np.ndarray, y: np.ndarray, context_x: np.ndarray, blackout_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # diff(fix[i+1], fix[i]) belongs to IMU block i (the interval after fix i).
        x = x[:-1]
        center, scale = x.mean(0), x.std(0)
        scale[scale < 1e-4] = 1.0
        standardized = (x - center) / scale
        centered_target = y - y.mean()
        ridge = 1.0 * np.eye(standardized.shape[1])
        coefficient = np.linalg.solve(standardized.T @ standardized + ridge, standardized.T @ centered_target)
        return ((context_x - center) / scale @ coefficient + y.mean(),
                (blackout_x - center) / scale @ coefficient + y.mean())

    context_acc, blackout_acc = fit_apply(acceleration_blocks, target_acceleration, acceleration_axes, blackout_features[:, 3:6])
    context_yaw, blackout_yaw = fit_apply(gyro_blocks, target_yaw, gyro_axes, blackout_features[:, 9:12])
    context_projection = np.column_stack([np.clip(context_acc, -4, 4) / 4, np.clip(context_yaw, -1, 1)])
    blackout_projection = np.column_stack([np.clip(blackout_acc, -4, 4) / 4, np.clip(blackout_yaw, -1, 1)])
    return context_projection.astype(np.float32), blackout_projection.astype(np.float32)


class BlackoutDataset(Dataset):
    def __init__(self, sessions: list[Session], examples: list[tuple[int, int, int]],
                 context_rows: int, mean: np.ndarray, std: np.ndarray,
                 augment: bool = False, seed: int = 42):
        self.sessions = sessions
        self.examples = examples
        self.context_rows = context_rows
        self.mean = mean
        self.std = std
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str | int]:
        session_index, start, length = self.examples[item]
        session = self.sessions[session_index]
        context_slice = slice(start - self.context_rows, start)
        blackout_slice = slice(start, start + length)
        imu = session.features[start - self.context_rows:start + length].copy()
        if self.augment:
            # Rotation forces the network to infer phone-to-vehicle alignment from context.
            rotate_phone_frame(imu, np.random.uniform(-math.pi / 6, math.pi / 6))
        # Emulate a 1 Hz GNSS receiver: use only past fixes and causal zero-order hold.
        context_speed = session.speed[context_slice].copy()
        context_heading = session.heading[context_slice].copy()
        fix_speed = context_speed[::10]
        fix_heading = context_heading[::10]
        if self.augment:
            fix_speed += np.random.normal(0.0, 0.25, len(fix_speed))
            fix_heading += np.random.normal(0.0, np.deg2rad(1.5), len(fix_heading))
        context_projection, blackout_projection = calibrated_projection(
            imu[:self.context_rows], imu[self.context_rows:], fix_speed, fix_heading
        )
        imu = ((imu - self.mean) / self.std).astype(np.float32)
        imu = np.column_stack([imu, np.vstack([context_projection, blackout_projection])]).astype(np.float32)
        held_speed = np.repeat(fix_speed, 10)[:self.context_rows]
        held_heading = np.repeat(fix_heading, 10)[:self.context_rows]
        fix_acceleration = np.r_[0.0, np.diff(fix_speed)]
        held_acceleration = np.repeat(fix_acceleration, 10)[:self.context_rows]
        reference = np.column_stack([
            held_speed / 30.0, held_acceleration / 4.0,
            np.sin(held_heading), np.cos(held_heading),
        ]).astype(np.float32)

        previous = start - 1
        true_speed = session.speed[blackout_slice].copy()
        true_heading = session.heading[blackout_slice].copy()
        lat0 = math.radians(float(session.latitude[previous]))
        east = np.deg2rad(session.longitude[blackout_slice] - session.longitude[previous]) * EARTH_RADIUS_M * math.cos(lat0)
        north = np.deg2rad(session.latitude[blackout_slice] - session.latitude[previous]) * EARTH_RADIUS_M
        previous_speeds = np.r_[session.speed[previous], true_speed[:-1]]
        previous_headings = np.r_[session.heading[previous], true_heading[:-1]]
        target_acceleration = (true_speed - previous_speeds) * 10.0
        target_yaw = np.arctan2(np.sin(true_heading - previous_headings), np.cos(true_heading - previous_headings)) * 10.0
        true_distance = np.cumsum(true_speed * 0.1)
        return {
            "context_imu": torch.from_numpy(imu[:self.context_rows]),
            "context_reference": torch.from_numpy(reference),
            "blackout_imu": torch.from_numpy(imu[self.context_rows:]),
            "initial_speed": torch.tensor(session.speed[previous], dtype=torch.float32),
            "initial_heading": torch.tensor(session.heading[previous], dtype=torch.float32),
            "true_speed": torch.from_numpy(true_speed),
            "true_heading": torch.from_numpy(true_heading),
            "true_east": torch.from_numpy(east.astype(np.float32)),
            "true_north": torch.from_numpy(north.astype(np.float32)),
            "true_distance": torch.from_numpy(true_distance.astype(np.float32)),
            "target_acceleration": torch.from_numpy(target_acceleration.astype(np.float32)),
            "target_yaw": torch.from_numpy(target_yaw.astype(np.float32)),
            "length": length, "session_index": session_index, "start": start,
            "session_key": session.key, "driver": session.driver,
        }


def collate(batch: list[dict]) -> dict:
    maximum = max(int(x["length"]) for x in batch)
    result: dict = {}
    fixed = ("context_imu", "context_reference", "initial_speed", "initial_heading")
    for key in fixed:
        result[key] = torch.stack([x[key] for x in batch])
    padded = ("blackout_imu", "true_speed", "true_heading", "true_east", "true_north",
              "true_distance", "target_acceleration", "target_yaw")
    for key in padded:
        sample = batch[0][key]
        shape = (len(batch), maximum) + tuple(sample.shape[1:])
        value = torch.zeros(shape, dtype=sample.dtype)
        for index, item in enumerate(batch):
            value[index, :int(item["length"])] = item[key]
        result[key] = value
    result["lengths"] = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    result["mask"] = torch.arange(maximum)[None, :] < result["lengths"][:, None]
    for key in ("session_index", "start", "session_key", "driver"):
        result[key] = [x[key] for x in batch]
    return result


class ContextUnrolledINS(nn.Module):
    def __init__(self, imu_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.context = nn.GRU(imu_features + 4, hidden, layers, batch_first=True,
                              dropout=recurrent_dropout)
        self.calibration = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.blackout = nn.GRU(imu_features + hidden, hidden, layers, batch_first=True,
                               dropout=recurrent_dropout)
        self.initial_state = nn.Linear(hidden, hidden * layers)
        self.layers = layers
        self.hidden = hidden
        self.motion = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        # The untrained estimator is the safe zero-motion-update (hold) model.
        # Learning therefore adds evidence-backed corrections instead of beginning
        # with a large random acceleration/yaw drift.
        nn.init.zeros_(self.motion[-1].weight)
        nn.init.zeros_(self.motion[-1].bias)

    def forward(self, context_imu: torch.Tensor, context_reference: torch.Tensor,
                blackout_imu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, context_state = self.context(torch.cat([context_imu, context_reference], dim=-1))
        calibration = self.calibration(context_state[-1])
        repeated = calibration[:, None, :].expand(-1, blackout_imu.shape[1], -1)
        initial = self.initial_state(calibration).view(-1, self.layers, self.hidden).transpose(0, 1).contiguous()
        encoded, _ = self.blackout(torch.cat([blackout_imu, repeated], dim=-1), initial)
        motion = self.motion(encoded)
        return torch.tanh(motion[..., 0]) * 5.0, torch.tanh(motion[..., 1]) * 1.5


def integrate(acceleration: torch.Tensor, yaw_rate: torch.Tensor, initial_speed: torch.Tensor,
              initial_heading: torch.Tensor) -> dict[str, torch.Tensor]:
    speed = torch.relu(initial_speed[:, None] + torch.cumsum(acceleration * 0.1, dim=1))
    heading = initial_heading[:, None] + torch.cumsum(yaw_rate * 0.1, dim=1)
    east = torch.cumsum(speed * torch.sin(heading) * 0.1, dim=1)
    north = torch.cumsum(speed * torch.cos(heading) * 0.1, dim=1)
    distance = torch.cumsum(speed * 0.1, dim=1)
    return {"speed": speed, "heading": heading, "east": east, "north": north, "distance": distance}


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def compute_loss(outputs: dict[str, torch.Tensor], acceleration: torch.Tensor, yaw: torch.Tensor,
                 batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"].float()
    speed = masked_mean(nn.functional.smooth_l1_loss(outputs["speed"] / 5, batch["true_speed"] / 5, reduction="none"), mask)
    position_error = torch.sqrt((outputs["east"] - batch["true_east"]) ** 2 + (outputs["north"] - batch["true_north"]) ** 2 + 1e-4)
    trajectory = masked_mean(nn.functional.smooth_l1_loss(position_error / 30, torch.zeros_like(position_error), reduction="none"), mask)
    distance = masked_mean(nn.functional.smooth_l1_loss(outputs["distance"] / 30, batch["true_distance"] / 30, reduction="none"), mask)
    heading = masked_mean(1.0 - torch.cos(outputs["heading"] - batch["true_heading"]), mask)
    auxiliary_acceleration = masked_mean(nn.functional.smooth_l1_loss(acceleration / 4, batch["target_acceleration"] / 4, reduction="none"), mask)
    auxiliary_yaw = masked_mean(nn.functional.smooth_l1_loss(yaw / 1.0, batch["target_yaw"] / 1.0, reduction="none"), mask)
    indices = (batch["lengths"] - 1).view(-1, 1)
    endpoint = position_error.gather(1, indices).mean() / 30
    loss = speed + trajectory + 0.5 * distance + 0.2 * heading + 0.15 * auxiliary_acceleration + 0.1 * auxiliary_yaw + endpoint
    parts = {"speed": speed, "trajectory": trajectory, "distance": distance, "heading": heading,
             "acceleration": auxiliary_acceleration, "yaw": auxiliary_yaw, "endpoint": endpoint}
    return loss, {name: float(value.detach()) for name, value in parts.items()}


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def validation_score(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval(); losses, velocity_errors = [], []
    for raw in loader:
        batch = to_device(raw, device)
        acceleration, yaw = model(batch["context_imu"], batch["context_reference"], batch["blackout_imu"])
        outputs = integrate(acceleration, yaw, batch["initial_speed"], batch["initial_heading"])
        loss, _ = compute_loss(outputs, acceleration, yaw, batch)
        losses.append(float(loss) * len(batch["lengths"]))
        velocity_errors.append(torch.abs(outputs["speed"] - batch["true_speed"])[batch["mask"]].cpu().numpy())
    count = len(loader.dataset)
    return sum(losses) / max(count, 1), float(np.mean(np.concatenate(velocity_errors))) * 3.6


def train(args: argparse.Namespace, model: nn.Module, train_loader: DataLoader,
          validation_loader: DataLoader, device: torch.device) -> tuple[list[dict], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    best, stale, best_epoch, best_state, history = math.inf, 0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; seen = 0; started = time.monotonic()
        for batch_number, raw in enumerate(train_loader, 1):
            batch = to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            acceleration, yaw = model(batch["context_imu"], batch["context_reference"], batch["blackout_imu"])
            outputs = integrate(acceleration, yaw, batch["initial_speed"], batch["initial_heading"])
            loss, _ = compute_loss(outputs, acceleration, yaw, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch["lengths"]); seen += len(batch["lengths"])
            if batch_number % args.log_every == 0 or batch_number == len(train_loader):
                rate = batch_number / max(time.monotonic() - started, 1e-6)
                eta = (len(train_loader) - batch_number) / max(rate, 1e-6)
                print(f"epoch {epoch:02d} batch {batch_number}/{len(train_loader)} loss={total/seen:.5f} ETA={eta:.0f}s", flush=True)
        validation_loss, validation_mae = validation_score(model, validation_loader, device)
        scheduler.step(validation_loss)
        record = {"epoch": epoch, "train_loss": total / seen, "validation_loss": validation_loss,
                  "validation_velocity_mae_kmh": validation_mae, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(f"epoch {epoch:02d}: train={total/seen:.5f} val={validation_loss:.5f} val velocity MAE={validation_mae:.2f} km/h", flush=True)
        if validation_loss < best - 1e-5:
            best, stale, best_epoch = validation_loss, 0, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state": best_state, "history": history}, args.output_dir / "best_training_checkpoint.pt")
        else:
            stale += 1
            if stale >= args.early_stopping_rounds:
                print(f"early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return history, best_epoch


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, sessions: list[Session], model_name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    model.eval(); episodes, samples = [], []
    for raw in loader:
        batch = to_device(raw, device)
        acceleration, yaw = model(batch["context_imu"], batch["context_reference"], batch["blackout_imu"])
        outputs = integrate(acceleration, yaw, batch["initial_speed"], batch["initial_heading"])
        for index, length_value in enumerate(batch["lengths"].cpu().tolist()):
            length = int(length_value); duration = length // 10
            true_speed = batch["true_speed"][index, :length].cpu().numpy()
            predicted_speed = outputs["speed"][index, :length].cpu().numpy()
            speed_error = (predicted_speed - true_speed) * 3.6
            true_east = batch["true_east"][index, :length].cpu().numpy()
            true_north = batch["true_north"][index, :length].cpu().numpy()
            predicted_east = outputs["east"][index, :length].cpu().numpy()
            predicted_north = outputs["north"][index, :length].cpu().numpy()
            endpoint_error = float(np.hypot(predicted_east[-1] - true_east[-1], predicted_north[-1] - true_north[-1]))
            true_distance = float(batch["true_distance"][index, length - 1].cpu())
            predicted_distance = float(outputs["distance"][index, length - 1].cpu())
            distance_error = predicted_distance - true_distance
            session = sessions[int(raw["session_index"][index])]
            start = int(raw["start"][index])
            episode_id = f"{model_name}:{session.driver}:{session.key}:{duration}:{start}"
            episodes.append({
                "model": model_name, "episode_id": episode_id, "driver_id": session.driver,
                "session_key": session.key, "start_row": start, "duration_seconds": duration,
                "velocity_mae_kmh": float(np.mean(np.abs(speed_error))),
                "velocity_rmse_kmh": float(np.sqrt(np.mean(speed_error ** 2))),
                "final_velocity_error_kmh": float(speed_error[-1]),
                "final_velocity_absolute_error_kmh": float(abs(speed_error[-1])),
                "within_5_kmh_percent": float(np.mean(np.abs(speed_error) <= 5) * 100),
                "true_distance_m": true_distance, "predicted_distance_m": predicted_distance,
                "distance_error_m": distance_error, "absolute_distance_error_m": abs(distance_error),
                "distance_drift_percent": abs(distance_error) / max(true_distance, 1.0) * 100,
                "endpoint_position_error_m": endpoint_error,
                "endpoint_position_drift_percent": endpoint_error / max(true_distance, 1.0) * 100,
            })
            for row in range(length):
                samples.append({
                    "model": model_name, "episode_id": episode_id, "duration_seconds": duration,
                    "elapsed_seconds": (row + 1) / 10, "true_speed_kmh": float(true_speed[row] * 3.6),
                    "predicted_speed_kmh": float(predicted_speed[row] * 3.6),
                    "velocity_error_kmh": float(speed_error[row]), "true_east_m": float(true_east[row]),
                    "true_north_m": float(true_north[row]), "predicted_east_m": float(predicted_east[row]),
                    "predicted_north_m": float(predicted_north[row]),
                })
    episode_frame, sample_frame = pd.DataFrame(episodes), pd.DataFrame(samples)
    summary = summarize(episode_frame, sample_frame)
    return summary, episode_frame, sample_frame


def summarize(episodes: pd.DataFrame, samples: pd.DataFrame) -> dict:
    result = {}
    for duration, group in episodes.groupby("duration_seconds"):
        sample_group = samples[samples.duration_seconds == duration]
        errors = sample_group.velocity_error_kmh.to_numpy()
        result[str(int(duration))] = {
            "episodes": int(len(group)), "velocity_mae_kmh": float(np.mean(np.abs(errors))),
            "velocity_rmse_kmh": float(np.sqrt(np.mean(errors ** 2))),
            "final_velocity_absolute_error_kmh": float(group.final_velocity_absolute_error_kmh.mean()),
            "final_velocity_error_bias_kmh": float(group.final_velocity_error_kmh.mean()),
            "within_5_kmh_percent": float(np.mean(np.abs(errors) <= 5) * 100),
            "absolute_distance_error_m": float(group.absolute_distance_error_m.mean()),
            "distance_drift_percent": float(group.distance_drift_percent.mean()),
            "median_distance_drift_percent": float(group.distance_drift_percent.median()),
            "aggregate_distance_drift_percent": float(group.absolute_distance_error_m.sum() / max(group.true_distance_m.sum(), 1.0) * 100),
            "endpoint_position_error_m": float(group.endpoint_position_error_m.mean()),
            "endpoint_position_drift_percent": float(group.endpoint_position_drift_percent.mean()),
            "median_endpoint_position_drift_percent": float(group.endpoint_position_drift_percent.median()),
            "aggregate_endpoint_position_drift_percent": float(group.endpoint_position_error_m.sum() / max(group.true_distance_m.sum(), 1.0) * 100),
        }
    return result


def hold_baseline(dataset: BlackoutDataset, sessions: list[Session]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    episodes, samples = [], []
    for number, (session_index, start, length) in enumerate(dataset.examples):
        session = sessions[session_index]; duration = length // 10; initial_speed = float(session.speed[start - 1])
        initial_heading = float(session.heading[start - 1]); true_speed = session.speed[start:start + length]
        predicted_speed = np.full(length, initial_speed); predicted_heading = np.full(length, initial_heading)
        predicted_east = np.cumsum(predicted_speed * np.sin(predicted_heading) * 0.1)
        predicted_north = np.cumsum(predicted_speed * np.cos(predicted_heading) * 0.1)
        lat0 = math.radians(float(session.latitude[start - 1]))
        true_east = np.deg2rad(session.longitude[start:start + length] - session.longitude[start - 1]) * EARTH_RADIUS_M * math.cos(lat0)
        true_north = np.deg2rad(session.latitude[start:start + length] - session.latitude[start - 1]) * EARTH_RADIUS_M
        errors = (predicted_speed - true_speed) * 3.6
        true_distance = float(np.sum(true_speed) * 0.1); predicted_distance = initial_speed * length * 0.1
        endpoint_error = float(np.hypot(predicted_east[-1] - true_east[-1], predicted_north[-1] - true_north[-1]))
        episode_id = f"hold:{session.driver}:{session.key}:{duration}:{start}"
        episodes.append({"model": "hold_speed_heading", "episode_id": episode_id, "driver_id": session.driver,
                         "session_key": session.key, "start_row": start, "duration_seconds": duration,
                         "velocity_mae_kmh": float(np.mean(abs(errors))), "velocity_rmse_kmh": float(np.sqrt(np.mean(errors ** 2))),
                         "final_velocity_error_kmh": float(errors[-1]), "final_velocity_absolute_error_kmh": float(abs(errors[-1])),
                         "within_5_kmh_percent": float(np.mean(abs(errors) <= 5) * 100), "true_distance_m": true_distance,
                         "predicted_distance_m": predicted_distance, "distance_error_m": predicted_distance - true_distance,
                         "absolute_distance_error_m": abs(predicted_distance - true_distance),
                         "distance_drift_percent": abs(predicted_distance - true_distance) / max(true_distance, 1) * 100,
                         "endpoint_position_error_m": endpoint_error,
                         "endpoint_position_drift_percent": endpoint_error / max(true_distance, 1) * 100})
        for row in range(length):
            samples.append({"model": "hold_speed_heading", "episode_id": episode_id, "duration_seconds": duration,
                            "elapsed_seconds": (row + 1) / 10, "true_speed_kmh": float(true_speed[row] * 3.6),
                            "predicted_speed_kmh": float(predicted_speed[row] * 3.6), "velocity_error_kmh": float(errors[row]),
                            "true_east_m": float(true_east[row]), "true_north_m": float(true_north[row]),
                            "predicted_east_m": float(predicted_east[row]), "predicted_north_m": float(predicted_north[row])})
    episode_frame, sample_frame = pd.DataFrame(episodes), pd.DataFrame(samples)
    return summarize(episode_frame, sample_frame), episode_frame, sample_frame


@torch.no_grad()
def channel_importance(model: nn.Module, dataset: BlackoutDataset, names: list[str], device: torch.device,
                       batch_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(dataset), size=min(160, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, chosen.tolist())
    loader = DataLoader(subset, batch_size=batch_size, collate_fn=collate)
    baseline, _ = validation_score(model, loader, device); records = []
    for channel, name in enumerate(names):
        changed = []
        for raw in loader:
            raw["context_imu"][:, :, channel] = 0
            raw["blackout_imu"][:, :, channel] = 0
            batch = to_device(raw, device)
            acceleration, yaw = model(batch["context_imu"], batch["context_reference"], batch["blackout_imu"])
            outputs = integrate(acceleration, yaw, batch["initial_speed"], batch["initial_heading"])
            loss, _ = compute_loss(outputs, acceleration, yaw, batch)
            changed.append(float(loss) * len(batch["lengths"]))
        altered = sum(changed) / len(subset)
        records.append({"feature": name, "validation_loss_increase": altered - baseline, "ablated_loss": altered,
                        "baseline_loss": baseline})
    return pd.DataFrame(records).sort_values("validation_loss_increase", ascending=False)


def plot_outputs(episodes: pd.DataFrame, samples: pd.DataFrame, summary: dict, baseline: dict,
                 history: list[dict], importance: pd.DataFrame, output: Path) -> None:
    durations = sorted(episodes.duration_seconds.unique())
    figure, axes = plt.subplots(len(durations), 1, figsize=(12, 3 * len(durations)), squeeze=False)
    for axis, duration in zip(axes[:, 0], durations):
        choices = episodes[episodes.duration_seconds == duration].sort_values("velocity_mae_kmh")
        episode_id = choices.iloc[len(choices) // 2].episode_id
        shown = samples[samples.episode_id == episode_id]
        axis.plot(shown.elapsed_seconds, shown.true_speed_kmh, label="Ground truth")
        axis.plot(shown.elapsed_seconds, shown.predicted_speed_kmh, label="Context-unrolled INS")
        axis.set_title(f"Representative {duration}s blackout"); axis.set_ylabel("km/h"); axis.grid(alpha=.25)
    axes[-1, 0].set_xlabel("Blackout time (s)"); axes[0, 0].legend(); figure.tight_layout()
    figure.savefig(output / "blackout_velocity_examples.png", dpi=140); plt.close(figure)

    figure, axes = plt.subplots(1, len(durations), figsize=(5 * len(durations), 4.5), squeeze=False)
    for axis, duration in zip(axes[0], durations):
        choices = episodes[episodes.duration_seconds == duration].sort_values("endpoint_position_error_m")
        shown = samples[samples.episode_id == choices.iloc[len(choices) // 2].episode_id]
        axis.plot(shown.true_east_m, shown.true_north_m, label="Ground truth")
        axis.plot(shown.predicted_east_m, shown.predicted_north_m, label="Prediction")
        axis.axis("equal"); axis.grid(alpha=.25); axis.set_title(f"{duration}s trajectory")
    axes[0, 0].legend(); figure.tight_layout(); figure.savefig(output / "blackout_trajectory_examples.png", dpi=140); plt.close(figure)

    x = durations; figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(x, [summary[str(v)]["velocity_mae_kmh"] for v in x], "o-", label="INS")
    axes[0].plot(x, [baseline[str(v)]["velocity_mae_kmh"] for v in x], "o--", label="Hold")
    axes[1].plot(x, [summary[str(v)]["distance_drift_percent"] for v in x], "o-", label="INS")
    axes[1].plot(x, [baseline[str(v)]["distance_drift_percent"] for v in x], "o--", label="Hold")
    axes[2].plot(x, [summary[str(v)]["endpoint_position_drift_percent"] for v in x], "o-", label="INS")
    axes[2].plot(x, [baseline[str(v)]["endpoint_position_drift_percent"] for v in x], "o--", label="Hold")
    for axis, ylabel in zip(axes, ("Velocity MAE (km/h)", "Distance drift (%)", "Position drift (%)")):
        axis.set_xlabel("Blackout duration (s)"); axis.set_ylabel(ylabel); axis.grid(alpha=.25); axis.legend()
    figure.tight_layout(); figure.savefig(output / "error_drift_vs_duration.png", dpi=140); plt.close(figure)

    frame = pd.DataFrame(history); figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame.epoch, frame.train_loss, label="train"); axes[0].plot(frame.epoch, frame.validation_loss, label="validation")
    axes[1].plot(frame.epoch, frame.validation_velocity_mae_kmh)
    axes[0].legend(); axes[0].set_ylabel("Unrolled loss"); axes[1].set_ylabel("Validation velocity MAE (km/h)")
    for axis in axes: axis.set_xlabel("Epoch"); axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(output / "training_history.png", dpi=140); plt.close(figure)

    shown = importance.head(19).sort_values("validation_loss_increase")
    figure, axis = plt.subplots(figsize=(9, 7)); axis.barh(shown.feature, shown.validation_loss_increase)
    axis.set_xlabel("Validation loss increase when channel is zeroed"); axis.set_title("IMU channel ablation importance")
    figure.tight_layout(); figure.savefig(output / "feature_importance.png", dpi=140); plt.close(figure)


def write_report(metrics: dict, output: Path) -> None:
    lines = ["# Context-calibrated unrolled INS", "",
             "A recurrent context encoder observes smartphone IMU plus causally held 1 Hz reference fixes before the blackout. "
             "Those fixes also fit a regularized per-blackout phone-to-vehicle acceleration/yaw projection. The blackout decoder "
             "receives IMU and that fixed IMU-derived projection only, predicts longitudinal acceleration and yaw rate, and is trained "
             "through differentiable speed/heading/position integration.", "",
             f"Best epoch: **{metrics['best_epoch']}**. Split: **{metrics['split']}**.", "",
             "| Blackout | Episodes | Velocity MAE | Hold MAE | Final abs. velocity | Within +/-5 | Distance drift | Endpoint position drift |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for duration, value in metrics["test_blackouts"].items():
        hold = metrics["hold_baseline"][duration]
        lines.append(f"| {duration}s | {value['episodes']} | {value['velocity_mae_kmh']:.2f} km/h | {hold['velocity_mae_kmh']:.2f} km/h | {value['final_velocity_absolute_error_kmh']:.2f} km/h | {value['within_5_kmh_percent']:.1f}% | {value['distance_drift_percent']:.1f}% | {value['endpoint_position_drift_percent']:.1f}% |")
    lines += ["", "## Leakage controls", "",
              "- Driver E trains, Driver A selects the checkpoint, and Driver B is evaluated only after selection.",
              "- Bias and context windows are finite and fully contained inside each continuous segment.",
              "- Context reference speed/heading is downsampled to 1 Hz and causally held.",
              "- Phone-to-vehicle projection coefficients are fitted once from pre-blackout context and then frozen.",
              "- After blackout initialization, the recurrent decoder receives smartphone IMU features only.",
              "- Ground-truth speed, heading and position after blackout are loss/evaluation targets only.",
              "- Euler orientation and phone GPS are not model inputs.", ""]
    (output / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = arguments(); seed_everything(args.seed)
    if args.smoke:
        args.epochs = min(args.epochs, 2); args.max_rows_per_session = args.max_rows_per_session or 7000
        args.hidden_size = min(args.hidden_size, 32); args.layers = 1; args.batch_size = min(args.batch_size, 8)
        args.train_durations = "10,30"; args.blackout_durations = "10,30"
        args.train_stride_seconds = 30
        if args.output_dir == DEFAULT_OUTPUT:
            args.output_dir = Path("artifacts/context_unrolled_ins_smoke")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = device_for(args); args.output_dir.mkdir(parents=True, exist_ok=True)
    bias_rows, context_rows = args.bias_seconds * 10, args.context_seconds * 10
    train_durations = sorted({int(x) for x in args.train_durations.split(",")})
    eval_durations = sorted({int(x) for x in args.blackout_durations.split(",")})
    print(f"device={device}; loading inertially aligned sessions...", flush=True)
    sessions, feature_names = load_sessions(args, bias_rows); mean, std = fit_scaler(sessions)
    train_examples = build_examples(sessions, "train", train_durations, context_rows, bias_rows,
                                    max(1, round(args.train_stride_seconds * 10)))
    validation_examples = build_examples(sessions, "validation", train_durations, context_rows, bias_rows,
                                         max(1, round(args.train_stride_seconds * 10)))
    test_examples: list[tuple[int, int, int]] = []
    for duration in eval_durations:
        stride = max(1, round(duration * args.eval_stride_factor * 10))
        test_examples.extend(build_examples(sessions, "test", [duration], context_rows, bias_rows, stride))
    if not train_examples or not validation_examples or not test_examples:
        raise RuntimeError(f"Insufficient episodes: train={len(train_examples)}, validation={len(validation_examples)}, test={len(test_examples)}")
    train_dataset = BlackoutDataset(sessions, train_examples, context_rows, mean, std, augment=True, seed=args.seed)
    validation_dataset = BlackoutDataset(sessions, validation_examples, context_rows, mean, std)
    test_dataset = BlackoutDataset(sessions, test_examples, context_rows, mean, std)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=collate, generator=generator, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                   collate_fn=collate, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers,
                             collate_fn=collate, pin_memory=device.type == "cuda")
    print(f"episodes: train={len(train_dataset)}, validation={len(validation_dataset)}, test={len(test_dataset)}", flush=True)
    model_feature_names = feature_names + ["context_calibrated_forward_acceleration", "context_calibrated_yaw_rate"]
    model = ContextUnrolledINS(len(model_feature_names), args.hidden_size, args.layers, args.dropout).to(device)
    history, best_epoch = train(args, model, train_loader, validation_loader, device)
    print("best checkpoint selected on Driver A; evaluating untouched Driver B...", flush=True)
    test_summary, episodes, samples = evaluate(model, test_loader, device, sessions, "context_unrolled_ins")
    baseline, baseline_episodes, baseline_samples = hold_baseline(test_dataset, sessions)
    importance = channel_importance(model, validation_dataset, model_feature_names, device, args.batch_size, args.seed)
    torch.save({"model_state": model.state_dict(), "model_config": {"imu_features": len(model_feature_names),
               "hidden": args.hidden_size, "layers": args.layers, "dropout": args.dropout},
               "feature_names": model_feature_names, "mean": mean, "std": std,
               "context_rows": context_rows, "bias_rows": bias_rows}, args.output_dir / "evaluation_model.pt")
    episodes.to_csv(args.output_dir / "blackout_episodes.csv", index=False)
    samples.to_csv(args.output_dir / "blackout_samples.csv", index=False)
    baseline_episodes.to_csv(args.output_dir / "hold_blackout_episodes.csv", index=False)
    baseline_samples.to_csv(args.output_dir / "hold_blackout_samples.csv", index=False)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    metrics = {"experiment": "context-calibrated recurrent unrolled INS", "device": str(device),
               "split": "Driver E train, Driver A validation, Driver B untouched test",
               "dataset_dir": str(args.dataset_dir), "context_seconds": args.context_seconds,
               "bias_seconds": args.bias_seconds, "reference_rate_hz": 1,
               "train_durations_seconds": train_durations, "test_durations_seconds": eval_durations,
               "episode_counts": {"train": len(train_dataset), "validation": len(validation_dataset), "test": len(test_dataset)},
               "features": model_feature_names, "best_epoch": best_epoch, "test_blackouts": test_summary,
               "hold_baseline": baseline, "ground_truth_after_blackout_used_as_input": False}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_outputs(episodes, samples, test_summary, baseline, history, importance, args.output_dir)
    write_report(metrics, args.output_dir)
    print(f"Training complete; best epoch {best_epoch}")
    for duration in eval_durations:
        value, hold = test_summary[str(duration)], baseline[str(duration)]
        print(f"{duration:>3}s: velocity MAE={value['velocity_mae_kmh']:.2f} km/h (hold={hold['velocity_mae_kmh']:.2f}), "
              f"distance drift={value['distance_drift_percent']:.1f}%, position drift={value['endpoint_position_drift_percent']:.1f}%")
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
