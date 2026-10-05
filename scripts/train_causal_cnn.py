#!/usr/bin/env python3
"""Train a leakage-safe causal CNN for gated stateful delta-velocity rollout."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET = Path("artifacts/ecu_training_dataset")
DEFAULT_OUTPUT = Path("artifacts/ecu_delta_v_causal_cnn")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=float, choices=[0.5, 1.0], default=1.0)
    parser.add_argument("--sequence-seconds", type=float, default=5.0)
    parser.add_argument("--bias-seconds", type=float, default=30.0)
    parser.add_argument("--blackout-durations", default="10,30,60,120")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-rounds", type=int, default=6)
    parser.add_argument("--dynamic-threshold", type=float, default=0.15)
    parser.add_argument("--dynamic-weight", type=float, default=2.0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-session", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.sample_stride < 1:
        parser.error("--sample-stride must be >= 1")
    return args


@dataclass
class Session:
    driver: str
    key: str
    split: str
    features: np.ndarray
    velocity_ms: np.ndarray
    elapsed: np.ndarray
    segment_id: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_segment_features(archive: dict[str, np.ndarray], rows: np.ndarray, bias_rows: int) -> tuple[np.ndarray, list[str]]:
    acc = np.column_stack([archive["acc_x_ms2"][rows], archive["acc_y_ms2"][rows], archive["acc_z_ms2"][rows]]).astype(float)
    gravity = np.column_stack([archive["gravity_x_ms2"][rows], archive["gravity_y_ms2"][rows], archive["gravity_z_ms2"][rows]]).astype(float)
    gyro = np.column_stack([archive["gyro_x_rads"][rows], archive["gyro_y_rads"][rows], archive["gyro_z_rads"][rows]]).astype(float)
    linear = acc - gravity
    gravity_norm = np.maximum(np.linalg.norm(gravity, axis=1, keepdims=True), 1e-6)
    gravity_unit = gravity / gravity_norm
    vertical = np.sum(linear * gravity_unit, axis=1, keepdims=True)
    horizontal = np.linalg.norm(linear - vertical * gravity_unit, axis=1, keepdims=True)
    # Finite, strictly causal bias: samples t-bias_rows ... t-1 only.
    bias = pd.DataFrame(linear).rolling(bias_rows, min_periods=1).mean().shift(1).fillna(0).to_numpy()
    debiased = linear - bias
    values = np.column_stack([
        linear,
        debiased,
        bias,
        gyro,
        gravity_unit,
        np.linalg.norm(linear, axis=1),
        np.linalg.norm(gyro, axis=1),
        vertical[:, 0],
        horizontal[:, 0],
    ]).astype(np.float32)
    names = [
        "linear_x", "linear_y", "linear_z",
        "debiased_x", "debiased_y", "debiased_z",
        "bias_x", "bias_y", "bias_z",
        "gyro_x", "gyro_y", "gyro_z",
        "gravity_unit_x", "gravity_unit_y", "gravity_unit_z",
        "linear_norm", "gyro_norm", "linear_vertical", "linear_horizontal",
    ]
    return values, names


def load_sessions(args: argparse.Namespace, bias_rows: int) -> tuple[list[Session], list[str]]:
    validation = json.loads((args.dataset_dir / "validation.json").read_text())
    if validation["status"] != "passed":
        raise RuntimeError("Prepared dataset validation has not passed")
    table = pd.read_csv(args.dataset_dir / "sessions.csv")
    if args.smoke:
        table = table.sort_values(["driver_holdout_split", "session_key"]).groupby("driver_holdout_split", as_index=False).head(1)
    result = []
    feature_names = None
    for position, record in enumerate(table.itertuples(index=False), start=1):
        with np.load(record.archive_path) as source:
            archive = {name: source[name] for name in source.files}
        if args.max_rows_per_session:
            archive = {name: value[: args.max_rows_per_session] for name, value in archive.items()}
        all_features = np.full((len(archive["segment_id"]), 19), np.nan, dtype=np.float32)
        for segment_id in np.unique(archive["segment_id"]):
            rows = np.flatnonzero(archive["segment_id"] == segment_id)
            segment_features, feature_names = build_segment_features(archive, rows, bias_rows)
            all_features[rows] = segment_features
        result.append(Session(
            driver=str(record.driver_id), key=str(record.session_key), split=str(record.driver_holdout_split),
            features=all_features, velocity_ms=archive["ecu_velocity_kmh"].astype(np.float32) / 3.6,
            elapsed=archive["phone_elapsed_seconds"].astype(np.float64), segment_id=archive["segment_id"].astype(np.int32),
        ))
        if position % 10 == 0 or position == len(table):
            print(f"loaded and featurized {position}/{len(table)} sessions", flush=True)
    return result, feature_names or []


def fit_scaler(sessions: list[Session]) -> tuple[np.ndarray, np.ndarray]:
    train = np.concatenate([session.features for session in sessions if session.split == "train"])
    mean = np.nanmean(train, axis=0).astype(np.float32)
    std = np.nanstd(train, axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def normalize(sessions: list[Session], mean: np.ndarray, std: np.ndarray) -> None:
    for session in sessions:
        session.features = ((session.features - mean) / std).astype(np.float32)


def eligible_examples(sessions: list[Session], split: str, history_rows: int, horizon_rows: int, stride: int) -> np.ndarray:
    examples = []
    for session_index, session in enumerate(sessions):
        if session.split != split:
            continue
        for segment in np.unique(session.segment_id):
            rows = np.flatnonzero(session.segment_id == segment)
            if len(rows) <= history_rows:
                continue
            endpoints = rows[history_rows::stride]
            examples.extend((session_index, int(endpoint)) for endpoint in endpoints if endpoint - horizon_rows >= rows[0])
    return np.asarray(examples, dtype=np.int32)


class SequenceDataset(Dataset):
    def __init__(self, sessions: list[Session], examples: np.ndarray, sequence_rows: int, horizon_rows: int, dynamic_threshold: float):
        self.sessions = sessions
        self.examples = examples
        self.sequence_rows = sequence_rows
        self.horizon_rows = horizon_rows
        self.dynamic_threshold = dynamic_threshold

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item: int):
        session_index, endpoint = self.examples[item]
        session = self.sessions[int(session_index)]
        x = session.features[endpoint - self.sequence_rows + 1 : endpoint + 1].T
        y = session.velocity_ms[endpoint] - session.velocity_ms[endpoint - self.horizon_rows]
        active = float(abs(y) >= self.dynamic_threshold)
        return torch.from_numpy(x), torch.tensor(y), torch.tensor(active)


class CausalConv(nn.Module):
    def __init__(self, incoming: int, outgoing: int, kernel: int, dilation: int):
        super().__init__()
        self.crop = (kernel - 1) * dilation
        self.conv = nn.Conv1d(incoming, outgoing, kernel, padding=self.crop, dilation=dilation)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.conv(value)
        return result[:, :, :-self.crop] if self.crop else result


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.network = nn.Sequential(
            CausalConv(channels, channels, kernel, dilation), nn.GroupNorm(4, channels), nn.GELU(), nn.Dropout(0.1),
            CausalConv(channels, channels, kernel, dilation), nn.GroupNorm(4, channels), nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class DeltaVelocityCNN(nn.Module):
    def __init__(self, inputs: int, channels: int, blocks: int, kernel: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(inputs, channels, 1)]
        layers.extend(ResidualBlock(channels, kernel, 2**index) for index in range(blocks))
        self.encoder = nn.Sequential(*layers)
        self.delta = nn.Linear(channels, 1)
        self.activity = nn.Linear(channels, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(value)[:, :, -1]
        return self.delta(encoded).squeeze(1), self.activity(encoded).squeeze(1)


def device_for(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in the container")
    return torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); targets=[]; deltas=[]; confidence=[]
    for x, y, _ in loader:
        prediction, logits = model(x.to(device))
        targets.append(y.numpy()); deltas.append(prediction.cpu().numpy()); confidence.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(targets), np.concatenate(deltas), np.concatenate(confidence)


def static_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    zero = float(np.mean(np.abs(actual)))
    mae = float(mean_absolute_error(actual, predicted))
    return {
        "mae_delta_ms": mae, "rmse_delta_ms": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)), "zero_delta_baseline_mae_ms": zero,
        "improvement_over_zero_percent": float((1 - mae / zero) * 100),
        "predicted_delta_std_ms": float(np.std(predicted)), "true_delta_std_ms": float(np.std(actual)),
    }


@torch.no_grad()
def permutation_importance(model: nn.Module, dataset: SequenceDataset, names: list[str], device: torch.device, batch_size: int, seed: int) -> pd.DataFrame:
    rng=np.random.default_rng(seed); count=min(10000,len(dataset)); chosen=rng.choice(len(dataset),size=count,replace=False)
    x=np.stack([dataset[int(i)][0].numpy() for i in chosen]); y=np.array([dataset[int(i)][1].item() for i in chosen])
    def mae(values:np.ndarray)->float:
        predictions=[]
        for start in range(0,len(values),batch_size):
            prediction,_=model(torch.from_numpy(values[start:start+batch_size]).to(device));predictions.append(prediction.cpu().numpy())
        return float(np.mean(np.abs(y-np.concatenate(predictions))))
    baseline=mae(x);records=[]
    for channel,name in enumerate(names):
        changed=x.copy();order=rng.permutation(len(changed));changed[:,channel,:]=changed[order,channel,:]
        permuted=mae(changed);records.append({"feature":name,"importance_mae_increase_ms":permuted-baseline,"permuted_mae_ms":permuted,"baseline_mae_ms":baseline})
    return pd.DataFrame(records).sort_values("importance_mae_increase_ms",ascending=False)


def train(args: argparse.Namespace, model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, device: torch.device, pos_weight: float) -> tuple[nn.Module, list[dict], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    history=[]; best=math.inf; best_epoch=0; stale=0; best_state=None
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss=0.0; seen=0; epoch_started=time.monotonic()
        for batch_number, (x, y, active) in enumerate(train_loader, start=1):
            x=x.to(device); y=y.to(device); active=active.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, logits=model(x)
            weight=1 + args.dynamic_weight * torch.clamp(torch.abs(y) / args.dynamic_threshold, max=3)
            regression=(nn.functional.smooth_l1_loss(prediction, y, reduction="none") * weight).mean()
            loss=regression + 0.25*bce(logits, active)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            train_loss += float(loss.detach()) * len(y); seen += len(y)
            if batch_number % args.log_every == 0 or batch_number == len(train_loader):
                elapsed=time.monotonic()-epoch_started; rate=batch_number/max(elapsed,1e-6); remaining=(len(train_loader)-batch_number)/max(rate,1e-6)
                print(f"epoch {epoch:02d} batch {batch_number}/{len(train_loader)} loss={train_loss/seen:.5f} ETA={remaining:.0f}s",flush=True)
        actual, predicted, _ = predict_loader(model, validation_loader, device)
        validation_mae=float(np.mean(np.abs(actual-predicted))); scheduler.step(validation_mae)
        history.append({"epoch":epoch,"train_loss":train_loss/seen,"validation_delta_mae_ms":validation_mae,"learning_rate":optimizer.param_groups[0]["lr"]})
        print(f"epoch {epoch:02d}: loss={train_loss/seen:.5f} val_delta_mae={validation_mae:.5f} m/s")
        if validation_mae < best - 1e-5:
            best=validation_mae; best_epoch=epoch; stale=0
            best_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            torch.save({"epoch":epoch,"model_state":best_state,"history":history,"validation_delta_mae_ms":best},args.output_dir/"best_training_checkpoint.pt")
        else:
            stale += 1
            if stale >= args.early_stopping_rounds:
                print(f"early stopping at epoch {epoch}; best epoch {best_epoch}"); break
    if best_state is None: raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state); return model,history,best_epoch


@torch.no_grad()
def predict_session(model: nn.Module, session: Session, sequence_rows: int, history_rows: int, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    delta=np.full(len(session.features),np.nan,np.float32); confidence=np.full(len(session.features),np.nan,np.float32)
    endpoints=[]
    for segment in np.unique(session.segment_id):
        rows=np.flatnonzero(session.segment_id==segment)
        endpoints.extend(rows[history_rows:])
    for offset in range(0,len(endpoints),batch_size):
        batch=endpoints[offset:offset+batch_size]
        x=np.stack([session.features[e-sequence_rows+1:e+1].T for e in batch])
        prediction,logits=model(torch.from_numpy(x).to(device))
        delta[batch]=prediction.cpu().numpy(); confidence[batch]=torch.sigmoid(logits).cpu().numpy()
    return delta,confidence


def blackout_rollout(sessions: list[Session], split: str, predictions: dict[int,tuple[np.ndarray,np.ndarray]], durations: list[int], horizon_rows: int, history_rows: int, threshold: float, shrinkage: float, model_name: str) -> tuple[dict,pd.DataFrame,pd.DataFrame]:
    episodes=[]; samples=[]
    for session_index,session in enumerate(sessions):
        if session.split!=split: continue
        delta,confidence=predictions[session_index]
        for segment in np.unique(session.segment_id):
            rows=np.flatnonzero(session.segment_id==segment); first=int(rows[0]); last=int(rows[-1])
            for duration in durations:
                duration_rows=duration*10; last_start=last-duration_rows
                for start in range(first+history_rows,last_start+1,duration_rows):
                    end=start+duration_rows; endpoints=np.arange(start+horizon_rows,end+1,horizon_rows)
                    applied=np.where(confidence[endpoints]>=threshold,np.clip(delta[endpoints]*shrinkage,-3.0,3.0),0.0)
                    predicted=[float(session.velocity_ms[start])]
                    for change in applied: predicted.append(max(0.0,predicted[-1]+float(change)))
                    truth_rows=np.r_[start,endpoints]; truth=session.velocity_ms[truth_rows].astype(float); estimate=np.asarray(predicted)
                    times=session.elapsed[truth_rows]-session.elapsed[start]; errors=(estimate-truth)*3.6
                    true_dense=session.velocity_ms[start:end+1].astype(float); dense_times=session.elapsed[start:end+1]-session.elapsed[start]
                    true_distance=float(np.trapezoid(true_dense,x=dense_times)); predicted_distance=float(np.trapezoid(estimate,x=times)); distance_error=predicted_distance-true_distance
                    episode_id=f"{model_name}:{session.driver}:{session.key}:{segment}:{duration}:{start}"
                    episodes.append({"model":model_name,"episode_id":episode_id,"driver_id":session.driver,"session_key":session.key,"segment_id":int(segment),"duration_seconds":duration,"velocity_mae_kmh":float(np.mean(abs(errors[1:]))),"velocity_rmse_kmh":float(np.sqrt(np.mean(errors[1:]**2))),"final_velocity_error_kmh":float(errors[-1]),"final_velocity_absolute_error_kmh":float(abs(errors[-1])),"within_5_kmh_percent":float(np.mean(abs(errors[1:])<=5)*100),"true_distance_m":true_distance,"predicted_distance_m":predicted_distance,"distance_error_m":distance_error,"absolute_distance_error_m":abs(distance_error),"distance_drift_percent":abs(distance_error)/true_distance*100 if true_distance>=1 else math.nan})
                    for row,t,actual,pred,error,conf in zip(truth_rows,times,truth,estimate,errors,np.r_[np.nan,confidence[endpoints]]):
                        samples.append({"model":model_name,"episode_id":episode_id,"duration_seconds":duration,"elapsed_seconds":float(t),"archive_row":int(row),"true_speed_kmh":float(actual*3.6),"predicted_speed_kmh":float(pred*3.6),"velocity_error_kmh":float(error),"confidence":float(conf)})
    episode_frame=pd.DataFrame(episodes); sample_frame=pd.DataFrame(samples); summary={}
    for duration,group in episode_frame.groupby("duration_seconds"):
        errors=sample_frame.loc[(sample_frame.duration_seconds==duration)&(sample_frame.elapsed_seconds>0),"velocity_error_kmh"].to_numpy()
        summary[str(int(duration))]={"episodes":int(len(group)),"velocity_mae_kmh":float(np.mean(abs(errors))),"velocity_rmse_kmh":float(np.sqrt(np.mean(errors**2))),"final_velocity_absolute_error_kmh":float(group.final_velocity_absolute_error_kmh.mean()),"final_velocity_error_bias_kmh":float(group.final_velocity_error_kmh.mean()),"within_5_kmh_percent":float(np.mean(abs(errors)<=5)*100),"absolute_distance_error_m":float(group.absolute_distance_error_m.mean()),"distance_error_bias_m":float(group.distance_error_m.mean()),"distance_drift_percent":float(group.distance_drift_percent.replace([np.inf,-np.inf],np.nan).mean())}
    return summary,episode_frame,sample_frame


def plot_outputs(episodes:pd.DataFrame,samples:pd.DataFrame,summary:dict,baseline:dict,history:list[dict],output:Path)->None:
    durations=sorted(episodes.duration_seconds.unique()); figure,axes=plt.subplots(len(durations),1,figsize=(13,3.1*len(durations)),squeeze=False)
    for axis,duration in zip(axes[:,0],durations):
        choices=episodes[episodes.duration_seconds==duration].sort_values("velocity_mae_kmh"); episode=choices.iloc[len(choices)//2].episode_id; shown=samples[samples.episode_id==episode]
        axis.plot(shown.elapsed_seconds,shown.true_speed_kmh,label="Ground truth");axis.plot(shown.elapsed_seconds,shown.predicted_speed_kmh,label="Gated causal CNN");axis.set_title(f"Representative {duration}s blackout");axis.set_ylabel("Velocity (km/h)");axis.grid(alpha=.25)
    axes[-1,0].set_xlabel("Blackout time (s)");axes[0,0].legend();figure.tight_layout();figure.savefig(output/"blackout_velocity_examples.png",dpi=140);plt.close(figure)
    grouped=samples[samples.elapsed_seconds>0].assign(abs_error=lambda x:x.velocity_error_kmh.abs()).groupby(["duration_seconds","elapsed_seconds"]).abs_error.mean().reset_index();figure,axis=plt.subplots(figsize=(11,5))
    for duration,values in grouped.groupby("duration_seconds"):axis.plot(values.elapsed_seconds,values.abs_error,label=f"{duration}s")
    axis.set(xlabel="Blackout time (s)",ylabel="Mean absolute velocity error (km/h)",title="Velocity error accumulation");axis.grid(alpha=.25);axis.legend();figure.tight_layout();figure.savefig(output/"velocity_error_over_time.png",dpi=140);plt.close(figure)
    figure,axis=plt.subplots(figsize=(7,7))
    for duration,values in episodes.groupby("duration_seconds"):axis.scatter(values.true_distance_m,values.predicted_distance_m,label=f"{duration}s",alpha=.65)
    maximum=max(episodes.true_distance_m.max(),episodes.predicted_distance_m.max());axis.plot([0,maximum],[0,maximum],"k--");axis.set(xlabel="Ground-truth distance (m)",ylabel="Predicted distance (m)",title="Held-out blackout distance");axis.grid(alpha=.25);axis.legend();figure.tight_layout();figure.savefig(output/"distance_ground_truth_vs_prediction.png",dpi=140);plt.close(figure)
    x=sorted(int(v) for v in summary);figure,axes=plt.subplots(1,2,figsize=(12,4.5));axes[0].plot(x,[summary[str(v)]["velocity_mae_kmh"] for v in x],"o-",label="CNN");axes[0].plot(x,[baseline[str(v)]["velocity_mae_kmh"] for v in x],"o-",label="Hold");axes[1].plot(x,[summary[str(v)]["distance_drift_percent"] for v in x],"o-",label="CNN");axes[1].plot(x,[baseline[str(v)]["distance_drift_percent"] for v in x],"o-",label="Hold");axes[0].set_ylabel("Velocity MAE (km/h)");axes[1].set_ylabel("Distance drift (%)")
    for axis in axes:axis.set_xlabel("Blackout duration (s)");axis.grid(alpha=.25);axis.legend()
    figure.tight_layout();figure.savefig(output/"error_drift_vs_duration.png",dpi=140);plt.close(figure)
    frame=pd.DataFrame(history);figure,axis=plt.subplots(figsize=(8,4));axis.plot(frame.epoch,frame.validation_delta_mae_ms,"o-");axis.set(xlabel="Epoch",ylabel="Validation delta MAE (m/s)",title="CNN training history");axis.grid(alpha=.25);figure.tight_layout();figure.savefig(output/"training_history.png",dpi=140);plt.close(figure)


def write_report(report:dict,output:Path)->None:
    lines=["# Gated causal 1D CNN dead-reckoning experiment","",f"Best epoch: **{report['best_epoch']}**. Selected confidence threshold: **{report['selected_gate']['confidence_threshold']}**; delta shrinkage: **{report['selected_gate']['delta_shrinkage']}**.","","Driver E trains the network, Driver A selects the checkpoint and rollout gate, and Driver B remains untouched until final evaluation.","","| Blackout | Episodes | CNN velocity MAE | Hold MAE | Final abs. error | Within +/-5 | Distance error | Drift |","|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for duration,value in report["test_blackouts"].items():
        hold=report["constant_speed_baseline"][duration];lines.append(f"| {duration}s | {value['episodes']} | {value['velocity_mae_kmh']:.2f} km/h | {hold['velocity_mae_kmh']:.2f} km/h | {value['final_velocity_absolute_error_kmh']:.2f} km/h | {value['within_5_kmh_percent']:.1f}% | {value['absolute_distance_error_m']:.1f} m | {value['distance_drift_percent']:.1f}% |")
    lines += ["","## Leakage controls","",f"- Finite causal history: {report['maximum_causal_history_rows']} rows; fully purged at every continuous-segment start.","- The 30-second accelerometer bias estimate uses only preceding phone IMU samples.","- No Euler orientation features, GNSS, ECU, speed, position, driver ID, or session ID enter the network.","- Gate and shrinkage selection use Driver A blackouts only; the candidate set includes rejecting every model update.","- Every Driver B blackout starts from one true speed and then uses smartphone IMU only.",""]
    (output/"experiment_report.md").write_text("\n".join(lines),encoding="utf-8")


def main()->None:
    args=arguments();seed_everything(args.seed)
    if args.smoke:
        args.epochs=min(args.epochs,2);args.max_rows_per_session=args.max_rows_per_session or 5000;args.channels=min(args.channels,16);args.blocks=min(args.blocks,2)
        if args.output_dir==DEFAULT_OUTPUT:args.output_dir=Path("artifacts/ecu_delta_v_causal_cnn_smoke")
    if args.threads>0:torch.set_num_threads(args.threads)
    device=device_for(args);horizon_rows=round(args.horizon*10);sequence_rows=round(args.sequence_seconds*10);bias_rows=round(args.bias_seconds*10);history_rows=bias_rows+sequence_rows;durations=sorted({int(v) for v in args.blackout_durations.split(",")});started=time.monotonic();args.output_dir.mkdir(parents=True,exist_ok=True)
    print(f"device={device}; loading and causally featurizing sessions...",flush=True)
    sessions,feature_names=load_sessions(args,bias_rows);mean,std=fit_scaler(sessions);normalize(sessions,mean,std)
    print("building leakage-safe sequence indices...",flush=True)
    examples={split:eligible_examples(sessions,split,history_rows,horizon_rows,args.sample_stride) for split in ("train","validation","test")}
    datasets={split:SequenceDataset(sessions,index,sequence_rows,horizon_rows,args.dynamic_threshold) for split,index in examples.items()}
    generator=torch.Generator().manual_seed(args.seed)
    loaders={split:DataLoader(dataset,batch_size=args.batch_size,shuffle=(split=="train"),num_workers=args.workers,generator=generator if split=="train" else None) for split,dataset in datasets.items()}
    train_targets=np.array([datasets["train"][i][1].item() for i in range(len(datasets["train"]))]);positive=float(np.mean(abs(train_targets)>=args.dynamic_threshold));pos_weight=(1-positive)/max(positive,1e-4)
    print(f"examples: train={len(datasets['train'])}, validation={len(datasets['validation'])}, test={len(datasets['test'])}; starting training...",flush=True)
    model=DeltaVelocityCNN(len(feature_names),args.channels,args.blocks,args.kernel_size).to(device);model,history,best_epoch=train(args,model,loaders["train"],loaders["validation"],device,pos_weight)
    torch.save({"model_state":model.state_dict(),"model_config":{"inputs":len(feature_names),"channels":args.channels,"blocks":args.blocks,"kernel_size":args.kernel_size},"feature_names":feature_names,"mean":mean,"std":std,"sequence_rows":sequence_rows,"bias_rows":bias_rows,"horizon_rows":horizon_rows},args.output_dir/"evaluation_model.pt")
    print("training finished; selecting the rollout gate on Driver A...",flush=True)
    validation_actual,validation_raw,validation_conf=predict_loader(model,loaders["validation"],device)
    predictions={i:predict_session(model,s,sequence_rows,history_rows,device,args.batch_size) for i,s in enumerate(sessions) if s.split=="validation"}
    candidates=[]
    for threshold in [0.0,0.3,0.5,0.7,0.85,1.1]:
        for shrinkage in [0.5,0.75,1.0]:
            summary,_,_=blackout_rollout(sessions,"validation",predictions,durations,horizon_rows,history_rows,threshold,shrinkage,"candidate")
            score=float(np.mean([v["velocity_mae_kmh"] for v in summary.values()]));candidates.append({"threshold":threshold,"shrinkage":shrinkage,"mean_validation_blackout_mae_kmh":score,"blackouts":summary})
    selected=min(candidates,key=lambda x:x["mean_validation_blackout_mae_kmh"]);threshold=selected["threshold"];shrinkage=selected["shrinkage"]
    predictions.update({i:predict_session(model,s,sequence_rows,history_rows,device,args.batch_size) for i,s in enumerate(sessions) if s.split=="test"})
    test_actual,test_raw,test_conf=predict_loader(model,loaders["test"],device);test_applied=np.where(test_conf>=threshold,np.clip(test_raw*shrinkage,-3,3),0)
    test_summary,episodes,samples=blackout_rollout(sessions,"test",predictions,durations,horizon_rows,history_rows,threshold,shrinkage,"gated_causal_cnn")
    baseline,baseline_episodes,baseline_samples=blackout_rollout(sessions,"test",predictions,durations,horizon_rows,history_rows,2.0,0.0,"hold_last_speed")
    importance=permutation_importance(model,datasets["validation"],feature_names,device,args.batch_size,args.seed);importance.to_csv(args.output_dir/"feature_importance.csv",index=False)
    episodes.to_csv(args.output_dir/"blackout_episodes.csv",index=False);samples.to_csv(args.output_dir/"blackout_velocity_samples.csv",index=False);baseline_episodes.to_csv(args.output_dir/"constant_speed_blackout_episodes.csv",index=False);baseline_samples.to_csv(args.output_dir/"constant_speed_velocity_samples.csv",index=False);pd.DataFrame(history).to_csv(args.output_dir/"training_history.csv",index=False);pd.DataFrame(candidates).drop(columns="blackouts").to_csv(args.output_dir/"gate_selection.csv",index=False)
    report={"experiment":"gated causal 1D CNN delta velocity","device":str(device),"split":"Driver E train, Driver A validation, Driver B untouched test","row_counts":{k:int(len(v)) for k,v in examples.items()},"horizon_seconds":args.horizon,"sequence_seconds":args.sequence_seconds,"bias_window_seconds":args.bias_seconds,"maximum_causal_history_rows":history_rows,"features":feature_names,"orientation_euler_features":False,"best_epoch":best_epoch,"activity_positive_fraction_train":positive,"validation_static_raw":static_metrics(validation_actual,validation_raw),"gate_selection_rule":"lowest mean Driver A blackout velocity MAE across requested durations","gate_candidates":candidates,"selected_gate":{"confidence_threshold":threshold,"delta_shrinkage":shrinkage},"test_static_gated":static_metrics(test_actual,test_applied),"test_blackouts":test_summary,"constant_speed_baseline":baseline,"ground_truth_speed_used_after_blackout_initialization":False,"elapsed_seconds":time.monotonic()-started}
    (args.output_dir/"metrics.json").write_text(json.dumps(report,indent=2)+"\n");plot_outputs(episodes,samples,test_summary,baseline,history,args.output_dir)
    top=importance.head(19).sort_values("importance_mae_increase_ms");figure,axis=plt.subplots(figsize=(9,7));axis.barh(top.feature,top.importance_mae_increase_ms);axis.set(xlabel="Validation MAE increase after permutation (m/s)",title="CNN channel permutation importance");figure.tight_layout();figure.savefig(args.output_dir/"feature_importance.png",dpi=140);plt.close(figure);write_report(report,args.output_dir)
    print(f"Training complete; best epoch {best_epoch}; gate threshold={threshold}, shrinkage={shrinkage}")
    for duration in durations:
        value=test_summary[str(duration)];hold=baseline[str(duration)];print(f"{duration:>3}s: MAE={value['velocity_mae_kmh']:.2f} km/h (hold={hold['velocity_mae_kmh']:.2f}), distance error={value['absolute_distance_error_m']:.1f} m")
    print(f"Artifacts: {args.output_dir}")


if __name__=="__main__":main()
