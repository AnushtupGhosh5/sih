#!/usr/bin/env python3
"""Audit the synchronized Y1 smartphone and vehicle recordings.

The script intentionally uses only the Python standard library so the audit is
reproducible before a machine-learning environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_DIR = Path(
    "data/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/Y (Driver D)/Y1"
)


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    return header, rows


def number(value: str) -> float | None:
    try:
        result = float(value.strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def quantiles(values: Iterable[float]) -> dict[str, float] | None:
    ordered = sorted(values)
    if not ordered:
        return None

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
        return ordered[index]

    return {
        "min": ordered[0],
        "p01": at(0.01),
        "p50": at(0.50),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def numeric_summary(header: list[str], rows: list[list[str]]) -> list[dict]:
    result = []
    for column, name in enumerate(header):
        values = [parsed for row in rows if (parsed := number(row[column])) is not None]
        if not values:
            continue
        result.append(
            {
                "index": column,
                "name": name.strip(),
                "valid": len(values),
                "invalid": len(rows) - len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
                "unique": len(set(values)),
            }
        )
    return result


def delta_summary(times: list[float], expected_period: float = 0.1) -> dict:
    deltas = [right - left for left, right in zip(times, times[1:])]
    anomalies = [
        {"after_row": index, "delta_seconds": delta}
        for index, delta in enumerate(deltas)
        if delta <= 0 or delta > max(1.0, expected_period * 5)
    ]
    return {
        "start": times[0],
        "end": times[-1],
        "span_seconds": times[-1] - times[0],
        "delta_seconds": quantiles(deltas),
        "negative_deltas": sum(delta < 0 for delta in deltas),
        "zero_deltas": sum(delta == 0 for delta in deltas),
        "large_gap_count": sum(delta > max(1.0, expected_period * 5) for delta in deltas),
        "anomalies": anomalies,
    }


def correlation(left: list[float | None], right: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(left, right) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    mean_x = statistics.fmean(x for x, _ in pairs)
    mean_y = statistics.fmean(y for _, y in pairs)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x, _ in pairs)
        * sum((y - mean_y) ** 2 for _, y in pairs)
    )
    return numerator / denominator if denominator else None


def haversine_metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_phone_timestamp(raw: str) -> float:
    return datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M:%S:%f").timestamp()


def markdown_report(audit: dict) -> str:
    phone = audit["smartphone"]
    vehicle = audit["vehicle"]
    alignment = audit["row_alignment_diagnostic"]
    phone_elapsed = phone["elapsed_time_ms"]
    phone_clock = phone["date_timestamp"]
    vehicle_clock = vehicle["time_since_day_start"]

    return f"""# Y1 data audit

Generated from the raw Y1 files by `scripts/audit_y1.py`.

## Inventory

- Smartphone samples: {phone['rows']:,}
- Vehicle samples: {vehicle['rows']:,}
- Smartphone columns: {phone['columns']}
- Vehicle columns: {vehicle['columns']}

## Timing

- Smartphone `TIME SINCE START` span: {phone_elapsed['span_seconds']:.3f} s
- Smartphone elapsed-time resets: {phone_elapsed['negative_deltas']}
- Smartphone wall-clock span: {phone_clock['span_seconds']:.3f} s
- Smartphone wall-clock large gaps: {phone_clock['large_gap_count']}
- Vehicle clock span: {vehicle_clock['span_seconds']:.3f} s
- Vehicle clock large gaps: {vehicle_clock['large_gap_count']}

## Row-alignment diagnostic

- Mean row-matched GPS separation: {alignment['gps_separation_metres']['mean']:.1f} m
- Median row-matched GPS separation: {alignment['gps_separation_metres']['p50']:.1f} m
- 95th percentile separation: {alignment['gps_separation_metres']['p95']:.1f} m
- Maximum separation: {alignment['gps_separation_metres']['max']:.1f} m
- Smartphone GPS speed vs vehicle GNSS velocity correlation: {alignment['speed_correlation']:.3f}
- Smartphone yaw-rate vs vehicle yaw-rate correlation: {alignment['yaw_rate_correlation']:.3f}

Equal row counts therefore do **not** establish valid sample alignment. Timestamp
repair and segment-level synchronization are required before supervised training.

## Data-quality notes

- Smartphone GPS latitude and longitude contain no missing values.
- Smartphone GPS speed has {phone['gps_speed_invalid']} invalid values.
- Smartphone GPS orientation has {phone['gps_orientation_invalid']} invalid values.
- Vehicle clutch position is constant across the recording.
- The vehicle GNSS velocity and indicated speed correlation is {vehicle['velocity_indicated_speed_correlation']:.3f}.
- Y1 is one driver/session, so it supports a proof of concept but not a driver-independent test.
"""


def build_audit(data_dir: Path) -> dict:
    phone_header, phone_rows = read_csv(data_dir / "S-Y1.txt")
    vehicle_header, vehicle_rows = read_csv(data_dir / "V-Y1.txt")

    phone_elapsed = [float(row[7]) / 1000.0 for row in phone_rows]
    phone_dates = [parse_phone_timestamp(row[8]) for row in phone_rows]
    vehicle_times = [float(row[1]) for row in vehicle_rows]

    distances = [
        haversine_metres(float(s[0]), float(s[1]), float(v[2]), float(v[3]))
        for s, v in zip(phone_rows, vehicle_rows)
    ]
    distance_ordered = sorted(distances)

    phone_speed = [number(row[3]) for row in phone_rows]
    vehicle_velocity = [number(row[4]) for row in vehicle_rows]
    phone_yaw_rate = [number(row[15]) for row in phone_rows]
    vehicle_yaw_rate = [
        value * math.pi / 180 if (value := number(row[14])) is not None else None
        for row in vehicle_rows
    ]

    vehicle_velocity_values = [float(row[4]) for row in vehicle_rows]
    indicated_speed_values = [float(row[15]) for row in vehicle_rows]

    return {
        "source_directory": str(data_dir),
        "smartphone": {
            "rows": len(phone_rows),
            "columns": len(phone_header),
            "elapsed_time_ms": delta_summary(phone_elapsed),
            "date_timestamp": delta_summary(phone_dates),
            "gps_speed_invalid": sum(value is None for value in phone_speed),
            "gps_orientation_invalid": sum(number(row[5]) is None for row in phone_rows),
            "numeric_columns": numeric_summary(phone_header, phone_rows),
        },
        "vehicle": {
            "rows": len(vehicle_rows),
            "columns": len(vehicle_header),
            "time_since_day_start": delta_summary(vehicle_times),
            "velocity_indicated_speed_correlation": correlation(
                vehicle_velocity_values, indicated_speed_values
            ),
            "constant_numeric_columns": [
                item["name"]
                for item in numeric_summary(vehicle_header, vehicle_rows)
                if item["unique"] == 1
            ],
            "numeric_columns": numeric_summary(vehicle_header, vehicle_rows),
        },
        "row_alignment_diagnostic": {
            "paired_rows": min(len(phone_rows), len(vehicle_rows)),
            "gps_separation_metres": {
                "mean": statistics.fmean(distances),
                "p50": statistics.median(distances),
                "p95": distance_ordered[round(0.95 * (len(distance_ordered) - 1))],
                "max": max(distances),
            },
            "speed_correlation": correlation(phone_speed, vehicle_velocity),
            "yaw_rate_correlation": correlation(phone_yaw_rate, vehicle_yaw_rate),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--json", type=Path, default=Path("reports/y1_audit.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/y1_audit.md"))
    args = parser.parse_args()

    audit = build_audit(args.data_dir)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown_report(audit), encoding="utf-8")
    print(f"Wrote {args.json}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
