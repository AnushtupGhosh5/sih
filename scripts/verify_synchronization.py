#!/usr/bin/env python3
"""Estimate and verify row lag between every synchronized S/V CSV pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_synchronized_dataset import find_pairs, driver_id, haversine_metres


DEFAULT_ROOT = Path("data/Synchronised V abd S datasets/Categorised IOVNB Dataset")
DEFAULT_OUTPUT = Path("artifacts/synchronization_verification")


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 20 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def shifted_metrics(
    smartphone: pd.DataFrame, vehicle: pd.DataFrame, lag: int
) -> tuple[float, float, float]:
    count = min(len(smartphone), len(vehicle))
    left_start = max(0, -lag)
    left_end = min(count, count - lag)
    right_start = left_start + lag
    right_end = left_end + lag
    phone_speed = pd.to_numeric(smartphone.iloc[left_start:left_end, 3], errors="coerce").to_numpy() * 3.6
    vehicle_speed = pd.to_numeric(vehicle.iloc[right_start:right_end, 4], errors="coerce").to_numpy()
    speed_corr = correlation(phone_speed, vehicle_speed)
    phone_lat = pd.to_numeric(smartphone.iloc[left_start:left_end, 0], errors="coerce").to_numpy()
    phone_lon = pd.to_numeric(smartphone.iloc[left_start:left_end, 1], errors="coerce").to_numpy()
    vehicle_lat = pd.to_numeric(vehicle.iloc[right_start:right_end, 2], errors="coerce").to_numpy()
    vehicle_lon = pd.to_numeric(vehicle.iloc[right_start:right_end, 3], errors="coerce").to_numpy()
    separation = haversine_metres(phone_lat, phone_lon, vehicle_lat, vehicle_lon)
    return speed_corr, float(np.nanmedian(separation)), float(np.nanpercentile(separation, 95))


def audit_pair(key: str, smartphone_path: Path, vehicle_path: Path, max_lag: int) -> tuple[dict, list[dict]]:
    smartphone = pd.read_csv(smartphone_path, encoding_errors="replace", low_memory=False)
    vehicle = pd.read_csv(vehicle_path, encoding_errors="replace", low_memory=False)
    count = min(len(smartphone), len(vehicle))
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        speed_corr, median_separation, p95_separation = shifted_metrics(smartphone, vehicle, lag)
        candidates.append(
            {
                "session_key": key,
                "lag_rows": lag,
                "lag_seconds_approx": lag * 0.1,
                "speed_correlation": speed_corr,
                "gps_median_separation_m": median_separation,
                "gps_p95_separation_m": p95_separation,
            }
        )
    candidates_frame = pd.DataFrame(candidates)
    finite = candidates_frame.loc[candidates_frame["speed_correlation"].notna()]
    zero = candidates_frame.loc[candidates_frame["lag_rows"] == 0].iloc[0]
    if finite.empty:
        best = zero
        improvement = float("nan")
    else:
        best = finite.sort_values(
            ["speed_correlation", "gps_median_separation_m"], ascending=[False, True]
        ).iloc[0]
        improvement = float(best["speed_correlation"] - zero["speed_correlation"])
    # Do not automatically shift on this diagnostic. Phone GPS speed is sparse
    # and latency-prone; its best correlation lag is not a trustworthy clock.
    # A session is row-verified only from its zero-lag evidence plus trajectory
    # consistency. Nonzero lag is retained for manual review.
    if zero["speed_correlation"] >= 0.8 and zero["gps_median_separation_m"] <= 100:
        quality = "verified_row_alignment"
    elif np.isfinite(best["speed_correlation"]) and best["speed_correlation"] >= 0.8 and improvement >= 0.05:
        quality = "lag_candidate_manual_review"
    elif zero["speed_correlation"] >= 0.8:
        quality = "strong_speed_manual_review"
    else:
        quality = "review"
    record = {
        "session_key": key,
        "driver_id": driver_id(smartphone_path),
        "session_directory": smartphone_path.parent.name,
        "smartphone_path": str(smartphone_path),
        "vehicle_path": str(vehicle_path),
        "paired_rows": count,
        "zero_lag_speed_corr": float(zero["speed_correlation"]),
        "zero_lag_gps_median_separation_m": float(zero["gps_median_separation_m"]),
        "best_lag_rows": int(best["lag_rows"]),
        "best_lag_seconds_approx": float(best["lag_seconds_approx"]),
        "best_speed_corr": float(best["speed_correlation"]) if np.isfinite(best["speed_correlation"]) else None,
        "best_gps_median_separation_m": float(best["gps_median_separation_m"]),
        "best_gps_p95_separation_m": float(best["gps_p95_separation_m"]),
        "speed_corr_improvement": improvement,
        "quality": quality,
    }
    return record, candidates


def write_report(manifest: pd.DataFrame, output: Path) -> None:
    summary = (
        manifest.groupby("driver_id")
        .agg(
            sessions=("session_key", "size"),
            median_zero_lag_corr=("zero_lag_speed_corr", "median"),
            median_best_corr=("best_speed_corr", "median"),
            lag_candidates=("quality", lambda values: int((values == "lag_candidate_manual_review").sum())),
            row_verified=("quality", lambda values: int((values == "verified_row_alignment").sum())),
            review_sessions=("quality", lambda values: int((values == "review").sum())),
        )
        .reset_index()
    )
    def table(frame: pd.DataFrame) -> str:
        columns = [str(column) for column in frame.columns]
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    lines = [
        "# Synchronization verification",
        "",
        f"Evaluated **{len(manifest)}** S/V pairs over ±50 samples (approximately ±5 seconds).",
        "Lag is selected using smartphone GPS speed ×3.6 versus ECU velocity; GPS trajectory separation is reported as a secondary check.",
        "",
        "## By driver",
        "",
        table(summary),
        "",
        "## Quality totals",
        "",
        table(manifest["quality"].value_counts().to_frame("sessions").reset_index().rename(columns={"index": "quality"})),
        "",
        "## Interpretation",
        "",
        "Sessions marked `verified_row_alignment` are the first candidates for row-level ECU targets.",
        "Lag candidates are diagnostic only: phone GPS speed latency can create a false best lag, so no automatic shift is applied.",
        "Sessions marked `strong_speed_manual_review` or `review` should not enter the main training set until investigated.",
        "A high speed correlation is necessary but not sufficient: timestamp continuity and target plausibility still need checking.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-lag", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(args.root)
    records = []
    lag_records = []
    for pair in pairs:
        record, candidates = audit_pair(*pair, args.max_lag)
        records.append(record)
        lag_records.extend(candidates)
    manifest = pd.DataFrame(records).sort_values(["driver_id", "session_key"])
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    pd.DataFrame(lag_records).to_csv(args.output_dir / "lag_sweep.csv", index=False)
    write_report(manifest, args.output_dir)
    print(f"Verified {len(manifest)} synchronized pairs")
    print(manifest["quality"].value_counts().to_string())
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
