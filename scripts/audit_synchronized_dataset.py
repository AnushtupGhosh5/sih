#!/usr/bin/env python3
"""Create a quality manifest for every categorized synchronized S/V session."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("data/Synchronised V abd S datasets/Categorised IOVNB Dataset")
DEFAULT_OUTPUT = Path("artifacts/synchronized_dataset_audit")


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def haversine_metres(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * np.arctan2(np.sqrt(value), np.sqrt(1 - value))


def driver_id(path: Path) -> str:
    match = re.search(r"\(Driver\s+([^)]+)\)", str(path))
    return match.group(1).strip() if match else "unknown"


def find_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for smartphone in sorted(root.rglob("S-*.csv")):
        key = smartphone.stem[2:].lower()
        vehicles = [
            candidate
            for candidate in smartphone.parent.glob("*.csv")
            if candidate.stem[:2].lower() == "v-" and candidate.stem[2:].lower() == key
        ]
        if len(vehicles) == 1:
            pairs.append((key, smartphone, vehicles[0]))
    return pairs


def audit_pair(key: str, smartphone_path: Path, vehicle_path: Path) -> dict:
    smartphone = pd.read_csv(smartphone_path, encoding_errors="replace", low_memory=False)
    vehicle = pd.read_csv(vehicle_path, encoding_errors="replace", low_memory=False)
    if smartphone.shape[1] < 24 or vehicle.shape[1] < 29:
        raise ValueError(f"Unexpected schema in {smartphone_path} or {vehicle_path}")
    count = min(len(smartphone), len(vehicle))
    smartphone = smartphone.iloc[:count]
    vehicle = vehicle.iloc[:count]

    phone_time = pd.to_datetime(
        smartphone.iloc[:, 8], format="%Y-%m-%d %H:%M:%S:%f", errors="coerce"
    )
    phone_delta = phone_time.diff().dt.total_seconds().to_numpy()
    vehicle_time = pd.to_numeric(vehicle.iloc[:, 1], errors="coerce").to_numpy()
    vehicle_delta = np.diff(vehicle_time, prepend=np.nan)
    phone_speed = pd.to_numeric(smartphone.iloc[:, 3], errors="coerce").to_numpy() * 3.6
    vehicle_velocity = pd.to_numeric(vehicle.iloc[:, 4], errors="coerce").to_numpy()
    vehicle_acceleration = pd.to_numeric(vehicle.iloc[:, 16], errors="coerce").to_numpy() * 9.80665

    acceleration = smartphone.iloc[:, 9:12].apply(pd.to_numeric, errors="coerce").to_numpy()
    gravity = smartphone.iloc[:, 12:15].apply(pd.to_numeric, errors="coerce").to_numpy()
    linear = acceleration - gravity
    gravity_norm = np.linalg.norm(gravity, axis=1)
    gravity_unit = gravity / np.maximum(gravity_norm[:, None], 1e-6)
    linear_vertical = np.sum(linear * gravity_unit, axis=1)
    linear_horizontal = np.linalg.norm(
        linear - linear_vertical[:, None] * gravity_unit, axis=1
    )
    angle_frame = smartphone.iloc[:, 21:24].apply(pd.to_numeric, errors="coerce")
    angle_missing = int(angle_frame.isna().any(axis=1).sum())
    linear_candidates = {
        "linear_x": linear[:, 0],
        "linear_y": linear[:, 1],
        "linear_z": linear[:, 2],
        "linear_vertical": linear_vertical,
        "linear_horizontal": linear_horizontal,
    }
    acceleration_correlations = {
        name: correlation(values, vehicle_acceleration)
        for name, values in linear_candidates.items()
    }
    finite_correlations = {
        name: value for name, value in acceleration_correlations.items() if np.isfinite(value)
    }

    gps_separation = haversine_metres(
        pd.to_numeric(smartphone.iloc[:, 0], errors="coerce").to_numpy(),
        pd.to_numeric(smartphone.iloc[:, 1], errors="coerce").to_numpy(),
        pd.to_numeric(vehicle.iloc[:, 2], errors="coerce").to_numpy(),
        pd.to_numeric(vehicle.iloc[:, 3], errors="coerce").to_numpy(),
    )
    session_dir = smartphone_path.parent.name
    speed_corr = correlation(phone_speed, vehicle_velocity)
    if not np.isfinite(speed_corr):
        speed_corr = None
    quality = (
        "strong"
        if speed_corr is not None and speed_corr >= 0.8 and np.median(gps_separation) <= 100
        else "review"
    )
    return {
        "session_key": key,
        "driver_id": driver_id(smartphone_path),
        "session_directory": session_dir,
        "smartphone_path": str(smartphone_path),
        "vehicle_path": str(vehicle_path),
        "smartphone_rows": int(len(smartphone)),
        "vehicle_rows": int(len(vehicle)),
        "paired_rows": int(count),
        "row_count_delta": int(len(smartphone) - len(vehicle)),
        "phone_period_seconds": float(np.nanmedian(phone_delta[1:])),
        "vehicle_period_seconds": float(np.nanmedian(vehicle_delta[1:])),
        "phone_bad_time_steps": int(np.sum((phone_delta[1:] <= 0) | (phone_delta[1:] > 0.5))),
        "vehicle_bad_time_steps": int(np.sum((vehicle_delta[1:] <= 0) | (vehicle_delta[1:] > 1.0))),
        "vehicle_large_gaps": int(np.sum(vehicle_delta[1:] > 1.0)),
        "phone_speed_invalid": int(np.sum(~np.isfinite(phone_speed))),
        "vehicle_velocity_invalid": int(np.sum(~np.isfinite(vehicle_velocity))),
        "phone_speed_vs_vehicle_velocity_corr": speed_corr,
        "gps_row_separation_mean_m": float(np.nanmean(gps_separation)),
        "gps_row_separation_median_m": float(np.nanmedian(gps_separation)),
        "gps_row_separation_p95_m": float(np.nanpercentile(gps_separation, 95)),
        "best_linear_acceleration_corr": (
            max(finite_correlations.values(), key=abs) if finite_correlations else None
        ),
        "best_linear_acceleration_feature": (
            max(finite_correlations, key=lambda name: abs(finite_correlations[name]))
            if finite_correlations
            else None
        ),
        "orientation_missing_rows": angle_missing,
        "quality": quality,
    }


def write_report(manifest: pd.DataFrame, output: Path) -> None:
    by_driver = (
        manifest.groupby("driver_id")
        .agg(
            sessions=("session_key", "size"),
            paired_rows=("paired_rows", "sum"),
            median_speed_corr=("phone_speed_vs_vehicle_velocity_corr", "median"),
            strong_sessions=("quality", lambda values: int((values == "strong").sum())),
            review_sessions=("quality", lambda values: int((values == "review").sum())),
        )
        .reset_index()
    )
    def markdown_table(frame: pd.DataFrame, include_index: bool = False) -> str:
        table = frame.reset_index() if include_index else frame.copy()
        columns = [str(column) for column in table.columns]
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in table.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    lines = [
        "# Synchronized dataset audit",
        "",
        f"Detected **{len(manifest)}** normalized S/V session pairs and **{int(manifest['paired_rows'].sum()):,}** paired rows.",
        "",
        "The manifest keeps every pair. `strong` is only a triage label: row-aligned",
        "speed correlation is at least 0.8 and median GPS separation is at most 100 m.",
        "No session is silently discarded.",
        "",
        "## By driver",
        "",
        markdown_table(by_driver),
        "",
        "## Quality totals",
        "",
        markdown_table(manifest["quality"].value_counts().to_frame("sessions"), include_index=True),
        "",
        "## Next use",
        "",
        "Use ECU velocity as the 10 Hz target only after each pair passes this audit.",
        "Train/test splits should be made by complete drive or driver, never by rows.",
        "Review low-correlation pairs with timestamp/trajectory alignment before use.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.root)
    if not pairs:
        raise SystemExit(f"No synchronized S/V CSV pairs found below {args.root}")
    records = [audit_pair(*pair) for pair in pairs]
    manifest = pd.DataFrame(records).sort_values(["driver_id", "session_key"])
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_report(manifest, args.output_dir)
    print(f"Audited {len(manifest)} synchronized pairs")
    print(f"Paired rows: {int(manifest['paired_rows'].sum()):,}")
    print(manifest["quality"].value_counts().to_string())
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
