#!/usr/bin/env python3
"""Prepare verified synchronized sessions for ECU-supervised DR training.

This stage does not fit a model. It creates compact aligned session archives,
continuous-segment metadata, leakage-safe split plans, and target diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MANIFEST = Path("artifacts/synchronization_verification/manifest.csv")
DEFAULT_OUTPUT = Path("artifacts/ecu_training_dataset")


PHONE_COLUMNS = {
    "phone_latitude_deg": 0,
    "phone_longitude_deg": 1,
    "phone_gps_speed_ms": 3,
    "phone_gps_accuracy_m": 4,
    "acc_x_ms2": 9,
    "acc_y_ms2": 10,
    "acc_z_ms2": 11,
    "gravity_x_ms2": 12,
    "gravity_y_ms2": 13,
    "gravity_z_ms2": 14,
    "gyro_x_rads": 15,
    "gyro_y_rads": 16,
    "gyro_z_rads": 17,
    "mag_x_ut": 18,
    "mag_y_ut": 19,
    "mag_z_ut": 20,
    "orientation_yaw_deg": 21,
    "orientation_pitch_deg": 22,
    "orientation_roll_deg": 23,
}

VEHICLE_COLUMNS = {
    "ecu_latitude_deg": 2,
    "ecu_longitude_deg": 3,
    "ecu_velocity_kmh": 4,
    "ecu_heading_deg": 5,
    "ecu_vertical_velocity_kmh": 7,
    "ecu_steering_angle_deg": 9,
    "ecu_yaw_rate_degs": 14,
    "ecu_indicated_speed_kmh": 15,
    "ecu_longitudinal_accel_ms2": 16,
    "ecu_lateral_accel_ms2": 17,
}


def finite_numeric(frame: pd.DataFrame, column: int) -> np.ndarray:
    return pd.to_numeric(frame.iloc[:, column], errors="coerce").to_numpy(float)


def stable_hash(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_splits(manifest: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    result = manifest.copy()
    driver_rows = result.groupby("driver_id")["paired_rows"].sum().sort_values(ascending=False)
    drivers = driver_rows.index.tolist()
    if len(drivers) < 3:
        raise ValueError("At least three verified drivers are required for driver holdout")
    mapping = {
        drivers[0]: "train",
        drivers[1]: "validation",
        **{driver: "test" for driver in drivers[2:]},
    }
    result["driver_holdout_split"] = result["driver_id"].map(mapping)

    drive_split = {}
    for driver, group in result.groupby("driver_id", sort=True):
        keys = sorted(group["session_key"], key=lambda key: stable_hash(f"{driver}:{key}", seed))
        count = len(keys)
        if count == 1:
            assignments = ["train"]
        elif count == 2:
            assignments = ["train", "test"]
        else:
            validation_count = max(1, round(count * 0.15))
            test_count = max(1, round(count * 0.15))
            train_count = count - validation_count - test_count
            assignments = ["train"] * train_count + ["validation"] * validation_count + ["test"] * test_count
        drive_split.update({key: split for key, split in zip(keys, assignments)})
    result["drive_holdout_split"] = result["session_key"].map(drive_split)
    return result, {
        "driver_holdout_mapping": mapping,
        "driver_holdout_rationale": "largest verified driver train, second largest validation, remaining drivers test",
        "drive_holdout_seed": seed,
        "drive_holdout_policy": "deterministic 70/15/15-ish whole-session split within each driver",
    }


def segment_rows(
    phone_time: pd.Series,
    vehicle_time: np.ndarray,
    velocity_kmh: np.ndarray,
    critical_valid: np.ndarray,
    max_gap_seconds: float,
    max_velocity_jump_kmh: float,
    min_segment_rows: int,
) -> tuple[np.ndarray, dict]:
    phone_delta = phone_time.diff().dt.total_seconds().to_numpy()
    vehicle_delta = np.diff(vehicle_time, prepend=np.nan)
    velocity_delta = np.diff(velocity_kmh, prepend=np.nan)
    boundary = (
        ~critical_valid
        | ~np.roll(critical_valid, 1)
        | (phone_delta <= 0)
        | (phone_delta > max_gap_seconds)
        | (vehicle_delta <= 0)
        | (vehicle_delta > max_gap_seconds)
        | (np.abs(velocity_delta) > max_velocity_jump_kmh)
    )
    boundary[0] = True
    raw_segment = np.cumsum(boundary) - 1
    raw_segment[~critical_valid] = -1
    counts = pd.Series(raw_segment[raw_segment >= 0]).value_counts()
    accepted = set(counts[counts >= min_segment_rows].index.tolist())
    mapping = {old: new for new, old in enumerate(sorted(accepted))}
    segment = np.array([mapping.get(value, -1) for value in raw_segment], dtype=np.int32)
    return segment, {
        "phone_time_breaks": int(np.sum((phone_delta[1:] <= 0) | (phone_delta[1:] > max_gap_seconds))),
        "vehicle_time_breaks": int(np.sum((vehicle_delta[1:] <= 0) | (vehicle_delta[1:] > max_gap_seconds))),
        "velocity_jump_breaks": int(np.sum(np.abs(velocity_delta[1:]) > max_velocity_jump_kmh)),
        "invalid_critical_rows": int(np.sum(~critical_valid)),
        "raw_segments": int(len(counts)),
        "accepted_segments": int(len(accepted)),
        "short_or_invalid_rows_excluded": int(np.sum(segment < 0)),
    }


def distribution(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    return {
        "samples": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "mean_absolute": float(np.mean(np.abs(values))),
        "percentiles": {
            str(value): float(np.percentile(values, value))
            for value in (0, 1, 10, 25, 50, 75, 90, 99, 100)
        },
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 20 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def prepare_session(
    record,
    output_sessions: Path,
    min_segment_rows: int,
    max_gap_seconds: float,
    max_velocity_jump_kmh: float,
) -> tuple[dict, list[dict], dict]:
    smartphone = pd.read_csv(record.smartphone_path, encoding_errors="replace", low_memory=False)
    vehicle = pd.read_csv(record.vehicle_path, encoding_errors="replace", low_memory=False)
    original_count = min(len(smartphone), len(vehicle))
    offset = int(getattr(record, "alignment_offset_rows", 0))
    phone_start = max(0, -offset)
    vehicle_start = max(0, offset)
    count = min(len(smartphone) - phone_start, len(vehicle) - vehicle_start)
    phone_source_rows = np.arange(phone_start, phone_start + count, dtype=np.int64)
    vehicle_source_rows = np.arange(vehicle_start, vehicle_start + count, dtype=np.int64)
    smartphone = smartphone.iloc[phone_source_rows].reset_index(drop=True)
    vehicle = vehicle.iloc[vehicle_source_rows].reset_index(drop=True)
    phone_time = pd.to_datetime(
        smartphone.iloc[:, 8], format="%Y-%m-%d %H:%M:%S:%f", errors="coerce"
    )
    vehicle_time = finite_numeric(vehicle, 1)

    arrays: dict[str, np.ndarray] = {
        "source_row": phone_source_rows,
        "vehicle_source_row": vehicle_source_rows,
        "phone_timestamp_ns": phone_time.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        "phone_elapsed_seconds": finite_numeric(smartphone, 7) / 1000.0,
        "vehicle_time_seconds": vehicle_time,
    }
    arrays.update({name: finite_numeric(smartphone, index) for name, index in PHONE_COLUMNS.items()})
    arrays.update({name: finite_numeric(vehicle, index) for name, index in VEHICLE_COLUMNS.items()})
    arrays["ecu_longitudinal_accel_ms2"] = arrays["ecu_longitudinal_accel_ms2"] * 9.80665
    arrays["ecu_lateral_accel_ms2"] = arrays["ecu_lateral_accel_ms2"] * 9.80665

    critical_names = [
        "acc_x_ms2", "acc_y_ms2", "acc_z_ms2",
        "gravity_x_ms2", "gravity_y_ms2", "gravity_z_ms2",
        "gyro_x_rads", "gyro_y_rads", "gyro_z_rads", "ecu_velocity_kmh",
    ]
    critical_valid = np.ones(count, dtype=bool)
    for name in critical_names:
        critical_valid &= np.isfinite(arrays[name])
    critical_valid &= np.isfinite(vehicle_time) & phone_time.notna().to_numpy()
    critical_valid &= (arrays["ecu_velocity_kmh"] >= 0) & (arrays["ecu_velocity_kmh"] <= 180)
    segment, segment_audit = segment_rows(
        phone_time,
        vehicle_time,
        arrays["ecu_velocity_kmh"],
        critical_valid,
        max_gap_seconds,
        max_velocity_jump_kmh,
        min_segment_rows,
    )
    arrays["segment_id"] = segment
    keep = segment >= 0
    kept = {name: value[keep] for name, value in arrays.items()}
    archive_name = f"driver_{record.driver_id}_{record.session_key}.npz"
    archive_path = output_sessions / archive_name
    np.savez_compressed(archive_path, **{
        name: value.astype(np.float32) if value.dtype.kind == "f" else value
        for name, value in kept.items()
    })

    segment_records = []
    for segment_id in np.unique(segment[keep]):
        rows = np.flatnonzero(segment == segment_id)
        archive_rows = np.flatnonzero(kept["segment_id"] == segment_id)
        duration = float(vehicle_time[rows[-1]] - vehicle_time[rows[0]])
        segment_records.append(
            {
                "driver_id": record.driver_id,
                "session_key": record.session_key,
                "segment_uid": f"{record.driver_id}:{record.session_key}:{int(segment_id):03d}",
                "segment_id": int(segment_id),
                "source_row_start": int(arrays["source_row"][rows[0]]),
                "source_row_end": int(arrays["source_row"][rows[-1]]),
                "vehicle_source_row_start": int(arrays["vehicle_source_row"][rows[0]]),
                "vehicle_source_row_end": int(arrays["vehicle_source_row"][rows[-1]]),
                "archive_row_start": int(archive_rows[0]),
                "archive_row_end": int(archive_rows[-1]),
                "rows": int(len(rows)),
                "duration_seconds": duration,
                "driver_holdout_split": record.driver_holdout_split,
                "drive_holdout_split": record.drive_holdout_split,
                "archive_path": str(archive_path),
            }
        )
    velocity = arrays["ecu_velocity_kmh"]
    indicated = arrays["ecu_indicated_speed_kmh"]
    return (
        {
            "driver_id": record.driver_id,
            "session_key": record.session_key,
            "archive_path": str(archive_path),
            "archive_sha256": file_sha256(archive_path),
            "input_rows": count,
            "original_paired_rows": original_count,
            "alignment_offset_rows": offset,
            "alignment_offset_seconds": offset * 0.1,
            "kept_rows": int(np.sum(keep)),
            "excluded_rows": int(np.sum(~keep)),
            "segments": len(segment_records),
            "ecu_velocity_vs_indicated_corr": correlation(velocity, indicated),
            "driver_holdout_split": record.driver_holdout_split,
            "drive_holdout_split": record.drive_holdout_split,
            **segment_audit,
        },
        segment_records,
        kept,
    )


def audit_targets(
    prepared: list[tuple[dict, dict]],
    horizons_seconds: list[float],
    history_rows: int,
    blackout_seconds: list[int],
) -> tuple[dict, list[dict], list[dict]]:
    target_report = {}
    eligibility_records = []
    training_eligibility = []
    for horizon in horizons_seconds:
        horizon_rows = round(horizon * 10)
        targets = []
        split_targets = defaultdict(list)
        feature_pairs = defaultdict(lambda: [[], []])
        for session_meta, arrays in prepared:
            segment = arrays["segment_id"]
            velocity_ms = arrays["ecu_velocity_kmh"] / 3.6
            acceleration = np.column_stack([arrays["acc_x_ms2"], arrays["acc_y_ms2"], arrays["acc_z_ms2"]])
            gravity = np.column_stack([arrays["gravity_x_ms2"], arrays["gravity_y_ms2"], arrays["gravity_z_ms2"]])
            gyro = np.column_stack([arrays["gyro_x_rads"], arrays["gyro_y_rads"], arrays["gyro_z_rads"]])
            linear = acceleration - gravity
            gravity_unit = gravity / np.maximum(np.linalg.norm(gravity, axis=1)[:, None], 1e-6)
            linear_vertical = np.sum(linear * gravity_unit, axis=1)
            candidate = {
                "linear_x": linear[:, 0],
                "linear_y": linear[:, 1],
                "linear_z": linear[:, 2],
                "linear_vertical": linear_vertical,
                "linear_horizontal": np.linalg.norm(linear - linear_vertical[:, None] * gravity_unit, axis=1),
                "gyro_norm": np.linalg.norm(gyro, axis=1),
                "ecu_longitudinal_accel_reference": arrays["ecu_longitudinal_accel_ms2"],
            }
            for segment_id in np.unique(segment):
                index = np.flatnonzero(segment == segment_id)
                if len(index) <= max(history_rows, horizon_rows):
                    continue
                delta = velocity_ms[index[horizon_rows:]] - velocity_ms[index[:-horizon_rows]]
                targets.append(delta)
                split_targets[session_meta["driver_holdout_split"]].append(delta)
                for name, values in candidate.items():
                    rolling = pd.Series(values[index]).rolling(horizon_rows, min_periods=horizon_rows).mean().to_numpy()
                    aligned = rolling[horizon_rows:]
                    feature_pairs[name][0].append(aligned)
                    feature_pairs[name][1].append(delta)
        combined = np.concatenate(targets)
        correlations = {
            name: correlation(np.concatenate(parts[0]), np.concatenate(parts[1]))
            for name, parts in feature_pairs.items()
        }
        target_report[str(horizon)] = {
            "horizon_rows": horizon_rows,
            "distribution_ms": distribution(combined),
            "distribution_by_driver_holdout_split_ms": {
                split: distribution(np.concatenate(values)) for split, values in split_targets.items()
            },
            "interval_mean_correlations": dict(
                sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
            ),
        }
        for session_meta, arrays in prepared:
            for segment_id in np.unique(arrays["segment_id"]):
                archive_rows = np.flatnonzero(arrays["segment_id"] == segment_id)
                first_offset = max(history_rows, horizon_rows)
                eligible_count = max(0, len(archive_rows) - first_offset)
                training_eligibility.append(
                    {
                        "driver_id": session_meta["driver_id"],
                        "session_key": session_meta["session_key"],
                        "segment_id": int(segment_id),
                        "horizon_seconds": horizon,
                        "horizon_rows": horizon_rows,
                        "history_purge_rows": history_rows,
                        "first_eligible_archive_row": (
                            int(archive_rows[0] + first_offset) if eligible_count else -1
                        ),
                        "last_eligible_archive_row": (
                            int(archive_rows[-1]) if eligible_count else -1
                        ),
                        "eligible_examples": eligible_count,
                        "driver_holdout_split": session_meta["driver_holdout_split"],
                        "drive_holdout_split": session_meta["drive_holdout_split"],
                    }
                )

    for session_meta, arrays in prepared:
        segment = arrays["segment_id"]
        for segment_id in np.unique(segment):
            length = int(np.sum(segment == segment_id))
            for duration in blackout_seconds:
                duration_rows = duration * 10
                episodes = max(0, (length - history_rows) // duration_rows)
                eligibility_records.append(
                    {
                        "driver_id": session_meta["driver_id"],
                        "session_key": session_meta["session_key"],
                        "segment_id": int(segment_id),
                        "driver_holdout_split": session_meta["driver_holdout_split"],
                        "drive_holdout_split": session_meta["drive_holdout_split"],
                        "blackout_seconds": duration,
                        "nonoverlapping_episodes": episodes,
                    }
                )
    return target_report, eligibility_records, training_eligibility


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(config: dict, sessions: pd.DataFrame, segments: pd.DataFrame, eligibility: pd.DataFrame, output: Path) -> None:
    split_summary = sessions.groupby(["driver_holdout_split", "driver_id"]).agg(sessions=("session_key", "size"), rows=("kept_rows", "sum")).reset_index()
    blackout_summary = eligibility.groupby(["driver_holdout_split", "blackout_seconds"])["nonoverlapping_episodes"].sum().reset_index()
    target_rows = []
    for horizon, details in config["target_audit"].items():
        dist = details["distribution_ms"]
        target_rows.append({"horizon_s": horizon, "samples": dist["samples"], "mean_abs_dv_ms": round(dist["mean_absolute"], 4), "std_dv_ms": round(dist["std"], 4)})
    lines = [
        "# ECU-ground-truth training dataset",
        "",
        f"Prepared **{len(sessions)}** verified sessions, **{int(sessions['kept_rows'].sum()):,}** usable rows, and **{len(segments)}** continuous segments.",
        "No model was trained in this stage.",
        "",
        "## Primary driver holdout",
        "",
        markdown_table(split_summary),
        "",
        "## Δv targets",
        "",
        markdown_table(pd.DataFrame(target_rows)),
        "",
        "## Non-overlapping blackout capacity",
        "",
        markdown_table(blackout_summary),
        "",
        "## Integrity rules",
        "",
        f"- Maximum feature-history purge: {config['history_rows']} rows ({config['history_rows']/10:.1f} s).",
        "- Δv targets are generated only inside one continuous segment.",
        "- Train/validation/test assignments are whole sessions or whole drivers.",
        "- No random-row split is produced.",
        "- ECU velocity is reference/label data and is not an inference feature.",
        "- Smartphone GPS fields are retained only for diagnostics and blackout initialization studies.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-segment-rows", type=int, default=200)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    parser.add_argument("--max-velocity-jump-kmh", type=float, default=5.0)
    parser.add_argument("--history-rows", type=int, default=100)
    parser.add_argument("--horizons", default="0.5,1.0")
    parser.add_argument("--blackouts", default="10,30,60,120")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--driver-split", help="Optional explicit TRAIN,VALIDATION,TEST driver IDs, e.g. E,A,B")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_output = args.output_dir / "sessions"
    session_output.mkdir(parents=True, exist_ok=True)
    source_manifest = pd.read_csv(args.manifest)
    if "training_eligible" in source_manifest:
        selected = source_manifest.loc[source_manifest["training_eligible"].astype(bool)].copy()
        selection_policy = "training_eligible == true from inertial alignment recovery"
    else:
        selected = source_manifest.loc[source_manifest["quality"] == "verified_row_alignment"].copy()
        selection_policy = "quality == verified_row_alignment"
    if selected.empty:
        raise SystemExit("No verified_row_alignment sessions found")
    selected, split_config = assign_splits(selected, args.seed)
    if args.driver_split:
        driver_parts = [value.strip() for value in args.driver_split.split(",")]
        if len(driver_parts) != 3:
            raise ValueError("--driver-split must contain exactly TRAIN,VALIDATION,TEST")
        available = set(selected["driver_id"].astype(str))
        if not set(driver_parts).issubset(available):
            raise ValueError(f"Requested drivers {driver_parts} are not all available: {sorted(available)}")
        explicit_mapping = {driver_parts[0]: "train", driver_parts[1]: "validation", driver_parts[2]: "test"}
        selected["driver_holdout_split"] = selected["driver_id"].astype(str).map(explicit_mapping)
        selected = selected.loc[selected["driver_holdout_split"].notna()].copy()
        split_config["driver_holdout_mapping"] = explicit_mapping
        split_config["driver_holdout_rationale"] = "explicit command-line mapping"

    session_records = []
    segment_records = []
    prepared = []
    for record in selected.itertuples(index=False):
        session_meta, segments, arrays = prepare_session(
            record,
            session_output,
            args.min_segment_rows,
            args.max_gap_seconds,
            args.max_velocity_jump_kmh,
        )
        session_records.append(session_meta)
        segment_records.extend(segments)
        prepared.append((session_meta, arrays))

    horizons = sorted({float(value) for value in args.horizons.split(",")})
    blackouts = sorted({int(value) for value in args.blackouts.split(",")})
    target_audit, eligibility_records, training_eligibility_records = audit_targets(
        prepared, horizons, args.history_rows, blackouts
    )
    sessions_frame = pd.DataFrame(session_records).sort_values(["driver_id", "session_key"])
    segments_frame = pd.DataFrame(segment_records).sort_values(["driver_id", "session_key", "segment_id"])
    eligibility_frame = pd.DataFrame(eligibility_records)
    training_eligibility_frame = pd.DataFrame(training_eligibility_records)
    sessions_frame.to_csv(args.output_dir / "sessions.csv", index=False)
    segments_frame.to_csv(args.output_dir / "segments.csv", index=False)
    eligibility_frame.to_csv(args.output_dir / "blackout_eligibility.csv", index=False)
    training_eligibility_frame.to_csv(
        args.output_dir / "training_eligibility.csv", index=False
    )

    config = {
        "format": "one compressed NPZ archive per synchronized session",
        "source_manifest": str(args.manifest),
        "selection_policy": selection_policy,
        "session_count": len(session_records),
        "usable_rows": int(sessions_frame["kept_rows"].sum()),
        "continuous_segments": len(segment_records),
        "minimum_segment_rows": args.min_segment_rows,
        "maximum_gap_seconds": args.max_gap_seconds,
        "maximum_velocity_jump_kmh_per_sample": args.max_velocity_jump_kmh,
        "history_rows": args.history_rows,
        "candidate_horizons_seconds": horizons,
        "blackout_durations_seconds": blackouts,
        "split_configuration": split_config,
        "target": "ecu_velocity_kmh and in-segment delta velocity",
        "inference_feature_policy": "smartphone IMU only; ECU/GPS reference columns prohibited",
        "inference_source_columns": [
            "acc_x_ms2", "acc_y_ms2", "acc_z_ms2",
            "gravity_x_ms2", "gravity_y_ms2", "gravity_z_ms2",
            "gyro_x_rads", "gyro_y_rads", "gyro_z_rads",
            "orientation_yaw_deg", "orientation_pitch_deg", "orientation_roll_deg",
        ],
        "orientation_policy": "angles may only transform IMU vectors; do not expose angles directly to model",
        "reference_only_columns": [
            "phone_latitude_deg", "phone_longitude_deg", "phone_gps_speed_ms",
            "phone_gps_accuracy_m", "vehicle_time_seconds", "ecu_latitude_deg",
            "ecu_longitude_deg", "ecu_velocity_kmh", "ecu_heading_deg",
            "ecu_vertical_velocity_kmh", "ecu_steering_angle_deg",
            "ecu_yaw_rate_degs", "ecu_indicated_speed_kmh",
            "ecu_longitudinal_accel_ms2", "ecu_lateral_accel_ms2",
        ],
        "archive_columns": sorted(prepared[0][1].keys()),
        "target_audit": target_audit,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    folds = {
        "leave_one_driver_out": [
            {"test_driver": driver, "train_drivers": [other for other in sorted(selected["driver_id"].unique()) if other != driver]}
            for driver in sorted(selected["driver_id"].unique())
        ]
    }
    (args.output_dir / "driver_folds.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    write_report(config, sessions_frame, segments_frame, eligibility_frame, args.output_dir)
    print(f"Prepared {len(sessions_frame)} sessions")
    print(f"Usable rows: {int(sessions_frame['kept_rows'].sum()):,}")
    print(f"Continuous segments: {len(segments_frame)}")
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
