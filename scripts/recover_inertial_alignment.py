#!/usr/bin/env python3
"""Recover phone-IMU/ECU alignment using independent acceleration and yaw events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags


DEFAULT_INPUT = Path("artifacts/synchronization_verification/manifest.csv")
DEFAULT_OUTPUT = Path("artifacts/inertial_alignment_recovery")
AXES = ("x_or_yaw", "y_or_pitch", "z_or_roll")


def numeric(frame: pd.DataFrame, columns) -> np.ndarray:
    return frame.iloc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def smooth(values: np.ndarray, rows: int) -> np.ndarray:
    return pd.DataFrame(values).rolling(rows, center=True, min_periods=rows).mean().to_numpy()


def lag_profile(left: np.ndarray, right: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    lags = np.arange(-max_lag, max_lag + 1)
    if len(left) < 500 or np.std(left) < 1e-7 or np.std(right) < 1e-7:
        return lags, np.full(len(lags), np.nan)
    left = (left - left.mean()) / left.std()
    right = (right - right.mean()) / right.std()
    full = correlate(left, right, mode="full", method="fft") / len(left)
    full_lags = correlation_lags(len(left), len(right), mode="full")
    selected = (full_lags >= -max_lag) & (full_lags <= max_lag)
    return full_lags[selected], full[selected]


def best_channel(features: np.ndarray, target: np.ndarray, max_lag: int) -> dict:
    candidates = []
    for axis in range(features.shape[1]):
        lags, values = lag_profile(features[:, axis], target, max_lag)
        if not np.isfinite(values).any():
            continue
        position = int(np.nanargmax(np.abs(values)))
        candidates.append((abs(values[position]), axis, int(lags[position]), float(values[position]), lags, values))
    if not candidates:
        return {"axis": None, "feature_lag_rows": 0, "correlation": None, "peak_margin": None, "lags": np.arange(-max_lag, max_lag + 1), "profile": np.full(2 * max_lag + 1, np.nan)}
    _, axis, lag, value, lags, profile = max(candidates)
    distant = np.abs(lags - lag) > 10
    alternative = float(np.nanmax(np.abs(profile[distant]))) if np.any(distant) else 0.0
    return {"axis": axis, "feature_lag_rows": lag, "correlation": value, "peak_margin": abs(value) - alternative, "lags": lags, "profile": profile}


def aligned_correlation(left: np.ndarray, right: np.ndarray, vehicle_offset: int) -> float:
    if vehicle_offset >= 0:
        left, right = left[: len(left) - vehicle_offset or None], right[vehicle_offset:]
    else:
        left, right = left[-vehicle_offset:], right[:vehicle_offset]
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 100 or np.std(left[valid]) < 1e-7 or np.std(right[valid]) < 1e-7:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-lag-rows", type=int, default=100)
    parser.add_argument("--smooth-rows", type=int, default=5)
    parser.add_argument("--acc-threshold", type=float, default=0.25)
    parser.add_argument("--yaw-threshold", type=float, default=0.35)
    parser.add_argument("--lag-agreement-rows", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    source = pd.read_csv(args.manifest)
    records, sweep_records = [], []

    for position, record in enumerate(source.itertuples(index=False), start=1):
        phone = pd.read_csv(record.smartphone_path, encoding_errors="replace", low_memory=False)
        vehicle = pd.read_csv(record.vehicle_path, encoding_errors="replace", low_memory=False)
        count = min(len(phone), len(vehicle)); phone = phone.iloc[:count]; vehicle = vehicle.iloc[:count]
        linear = numeric(phone, slice(9, 12)) - numeric(phone, slice(12, 15))
        gyro = numeric(phone, slice(15, 18))
        ecu_acceleration = numeric(vehicle, [16])[:, 0] * 9.80665
        ecu_yaw = np.deg2rad(numeric(vehicle, [14])[:, 0])
        acceleration = best_channel(smooth(linear, args.smooth_rows), ecu_acceleration, args.max_lag_rows)
        yaw = best_channel(smooth(gyro, args.smooth_rows), ecu_yaw, args.max_lag_rows)
        acc_corr = acceleration["correlation"]
        yaw_corr = yaw["correlation"]
        agreement = abs(acceleration["feature_lag_rows"] - yaw["feature_lag_rows"])
        away_from_edge = max(abs(acceleration["feature_lag_rows"]), abs(yaw["feature_lag_rows"])) < args.max_lag_rows - 2
        high = bool(acc_corr is not None and yaw_corr is not None and abs(acc_corr) >= args.acc_threshold and abs(yaw_corr) >= args.yaw_threshold and agreement <= args.lag_agreement_rows and away_from_edge)
        moderate = bool(not high and acc_corr is not None and yaw_corr is not None and away_from_edge and (
            (abs(acc_corr) >= 0.12 and abs(yaw_corr) >= 0.60 and agreement <= 7)
            or (abs(acc_corr) >= 0.25 and abs(yaw_corr) >= 0.80 and agreement <= 15)
        ))
        if high or moderate:
            acc_weight, yaw_weight = abs(acc_corr), abs(yaw_corr)
            feature_lag = int(round((acceleration["feature_lag_rows"] * acc_weight + yaw["feature_lag_rows"] * yaw_weight) / (acc_weight + yaw_weight)))
            vehicle_offset = -feature_lag
            tier = "high" if high else "moderate"
        else:
            feature_lag = 0; vehicle_offset = 0; tier = "insufficient"
        aligned_acc = aligned_correlation(smooth(linear, args.smooth_rows)[:, acceleration["axis"]], ecu_acceleration, vehicle_offset) if acceleration["axis"] is not None else float("nan")
        aligned_yaw = aligned_correlation(smooth(gyro, args.smooth_rows)[:, yaw["axis"]], ecu_yaw, vehicle_offset) if yaw["axis"] is not None else float("nan")
        recovered = high or moderate
        result = record._asdict()
        result.update({
            "original_quality": record.quality, "alignment_evidence": tier, "training_eligible": recovered,
            "quality": "verified_row_alignment" if recovered else "review",
            "alignment_offset_rows": vehicle_offset, "alignment_offset_seconds": vehicle_offset * 0.1,
            "acceleration_axis": AXES[acceleration["axis"]] if acceleration["axis"] is not None else None,
            "acceleration_peak_corr": acc_corr, "acceleration_feature_lag_rows": acceleration["feature_lag_rows"], "acceleration_peak_margin": acceleration["peak_margin"],
            "yaw_axis": AXES[yaw["axis"]] if yaw["axis"] is not None else None,
            "yaw_peak_corr": yaw_corr, "yaw_feature_lag_rows": yaw["feature_lag_rows"], "yaw_peak_margin": yaw["peak_margin"],
            "inertial_lag_agreement_rows": agreement, "aligned_acceleration_corr": aligned_acc, "aligned_yaw_corr": aligned_yaw,
        })
        records.append(result)
        for signal, details in (("acceleration", acceleration), ("yaw", yaw)):
            for lag, value in zip(details["lags"], details["profile"]):
                sweep_records.append({"driver_id": record.driver_id, "session_key": record.session_key, "signal": signal, "feature_lag_rows": int(lag), "correlation": value})
        if recovered:
            figure, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(acceleration["lags"] / 10, acceleration["profile"]); axes[0].axvline(feature_lag / 10, color="red", linestyle="--"); axes[0].set_title(f"Acceleration: r={acc_corr:.3f}")
            axes[1].plot(yaw["lags"] / 10, yaw["profile"]); axes[1].axvline(feature_lag / 10, color="red", linestyle="--"); axes[1].set_title(f"Yaw: r={yaw_corr:.3f}")
            for axis in axes: axis.set_xlabel("Phone-feature lag (s)"); axis.set_ylabel("Correlation"); axis.grid(alpha=0.25)
            figure.suptitle(f"Driver {record.driver_id} / {record.session_key}: selected vehicle offset {vehicle_offset / 10:+.1f}s ({tier})")
            figure.tight_layout(); figure.savefig(plot_dir / f"driver_{record.driver_id}_{record.session_key}.png", dpi=130); plt.close(figure)
        if position % 12 == 0 or position == len(source): print(f"processed {position}/{len(source)} sessions", flush=True)

    manifest = pd.DataFrame(records).sort_values(["driver_id", "session_key"])
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    pd.DataFrame(sweep_records).to_csv(args.output_dir / "lag_sweep.csv", index=False)
    summary = manifest.groupby(["driver_id", "alignment_evidence"]).agg(sessions=("session_key", "size"), rows=("paired_rows", "sum")).reset_index()
    report = {
        "method": "independent 0.5-second-smoothed phone acceleration and gyroscope agreement against ECU longitudinal acceleration and yaw rate",
        "max_lag_rows": args.max_lag_rows, "thresholds": {"acceleration_absolute_correlation": args.acc_threshold, "yaw_absolute_correlation": args.yaw_threshold, "lag_agreement_rows": args.lag_agreement_rows},
        "sessions_total": int(len(manifest)), "sessions_high": int((manifest.alignment_evidence == "high").sum()), "sessions_moderate": int((manifest.alignment_evidence == "moderate").sum()),
        "sessions_training_eligible": int(manifest.training_eligible.sum()), "eligible_rows_before_edge_trim": int(manifest.loc[manifest.training_eligible, "paired_rows"].sum()),
        "by_driver_and_tier": summary.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    table_lines = ["| driver_id | alignment_evidence | sessions | rows |", "|---|---|---:|---:|"]
    table_lines.extend(f"| {row.driver_id} | {row.alignment_evidence} | {row.sessions} | {row.rows} |" for row in summary.itertuples(index=False))
    lines = ["# Inertial alignment recovery", "", "Phone GPS is used only as prior coarse evidence. Final offsets require independent phone acceleration and yaw-rate agreement with ECU motion events.", "", f"Training-eligible sessions: **{report['sessions_training_eligible']} / {report['sessions_total']}** ({report['sessions_high']} high, {report['sessions_moderate']} moderate).", "", *table_lines, "", "Positive `alignment_offset_rows` means vehicle row `i + offset` is paired with phone row `i`. Every original row index is retained by the preparation stage.", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2)); print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__": main()
