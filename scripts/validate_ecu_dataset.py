#!/usr/bin/env python3
"""Independently validate prepared ECU-supervised archives and split metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path("artifacts/ecu_training_dataset")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    config = json.loads((args.dataset_dir / "config.json").read_text())
    sessions = pd.read_csv(args.dataset_dir / "sessions.csv")
    segments = pd.read_csv(args.dataset_dir / "segments.csv")
    eligibility = pd.read_csv(args.dataset_dir / "training_eligibility.csv")
    failures = []
    checks = {
        "archives_checked": 0,
        "arrays_checked": 0,
        "segments_checked": 0,
        "training_ranges_checked": 0,
    }

    overlap = set(config["inference_source_columns"]) & set(config["reference_only_columns"])
    if overlap:
        failures.append(f"Inference/reference column overlap: {sorted(overlap)}")

    for session in sessions.itertuples(index=False):
        archive_path = Path(session.archive_path)
        if not archive_path.exists():
            failures.append(f"Missing archive: {archive_path}")
            continue
        if sha256(archive_path) != session.archive_sha256:
            failures.append(f"Checksum mismatch: {archive_path}")
        with np.load(archive_path) as archive:
            lengths = {name: len(archive[name]) for name in archive.files}
            checks["arrays_checked"] += len(lengths)
            if len(set(lengths.values())) != 1:
                failures.append(f"Array length mismatch: {archive_path}")
                continue
            row_count = next(iter(lengths.values()))
            if row_count != session.kept_rows:
                failures.append(f"Manifest row mismatch: {archive_path}")
            source_row = archive["source_row"]
            vehicle_source_row = archive["vehicle_source_row"] if "vehicle_source_row" in archive.files else source_row
            expected_offset = int(getattr(session, "alignment_offset_rows", 0))
            if not np.all(vehicle_source_row - source_row == expected_offset):
                failures.append(f"Alignment offset provenance mismatch: {archive_path}")
            segment_id = archive["segment_id"]
            vehicle_time = archive["vehicle_time_seconds"]
            velocity = archive["ecu_velocity_kmh"]
            for segment_value in np.unique(segment_id):
                index = np.flatnonzero(segment_id == segment_value)
                checks["segments_checked"] += 1
                if not np.all(np.diff(index) == 1):
                    failures.append(f"Non-contiguous archive segment: {archive_path}:{segment_value}")
                if not np.all(np.diff(source_row[index]) == 1):
                    failures.append(f"Non-contiguous source rows: {archive_path}:{segment_value}")
                if not np.all(np.diff(vehicle_source_row[index]) == 1):
                    failures.append(f"Non-contiguous vehicle source rows: {archive_path}:{segment_value}")
                delta_time = np.diff(vehicle_time[index])
                if np.any((delta_time <= 0) | (delta_time > config["maximum_gap_seconds"] + 1e-5)):
                    failures.append(f"Time discontinuity inside segment: {archive_path}:{segment_value}")
                if np.any(np.abs(np.diff(velocity[index])) > config["maximum_velocity_jump_kmh_per_sample"] + 1e-4):
                    failures.append(f"Velocity jump inside segment: {archive_path}:{segment_value}")
        checks["archives_checked"] += 1

    for segment in segments.itertuples(index=False):
        expected = segment.archive_row_end - segment.archive_row_start + 1
        if expected != segment.rows:
            failures.append(f"Segment range mismatch: {segment.segment_uid}")

    for item in eligibility.itertuples(index=False):
        checks["training_ranges_checked"] += 1
        if item.eligible_examples == 0:
            continue
        expected = item.last_eligible_archive_row - item.first_eligible_archive_row + 1
        if expected != item.eligible_examples:
            failures.append(
                f"Eligibility range mismatch: {item.driver_id}:{item.session_key}:{item.segment_id}:{item.horizon_seconds}"
            )
        segment = segments.loc[
            (segments["driver_id"] == item.driver_id)
            & (segments["session_key"] == item.session_key)
            & (segments["segment_id"] == item.segment_id)
        ].iloc[0]
        if item.first_eligible_archive_row - segment.archive_row_start < item.history_purge_rows:
            failures.append(
                f"History purge violation: {item.driver_id}:{item.session_key}:{item.segment_id}:{item.horizon_seconds}"
            )

    driver_split_counts = sessions.groupby(["driver_id", "driver_holdout_split"]).size()
    if len(driver_split_counts) != sessions["driver_id"].nunique():
        failures.append("A driver appears in more than one driver-holdout split")

    result = {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "dataset_dir": str(args.dataset_dir),
    }
    (args.dataset_dir / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.dataset_dir / "validation.md").write_text(
        "# Prepared dataset validation\n\n"
        f"Status: **{result['status']}**\n\n"
        f"Archives checked: {checks['archives_checked']}\n\n"
        f"Continuous segments checked: {checks['segments_checked']}\n\n"
        f"Training ranges checked: {checks['training_ranges_checked']}\n\n"
        f"Failures: {len(failures)}\n",
        encoding="utf-8",
    )
    print(f"Validation {result['status']}: {len(failures)} failures")
    print(json.dumps(checks))
    if failures:
        for failure in failures[:20]:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
