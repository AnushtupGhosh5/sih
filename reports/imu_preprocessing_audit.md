# Smartphone IMU preprocessing audit

## Outcome

The poor cross-driver dead-reckoning result is reproducible and is not explained
by a speed-unit error, random time-series split, target construction error, or
distance-integration error. The current Euler orientation transform is not
physically trustworthy and should not be used in the next model, but removing
that representation cannot recover the result: raw and gravity-corrected phone
axes, oracle fixed-axis projections, same-driver controls, and causal
pre-blackout calibration were all tested independently.

## Verified assumptions

- The smartphone GPS speed column is numerically in m/s despite a CSV header
  that says km/h. Multiplication by 3.6 gives a median zero-row-lag correlation
  of 0.920 with ECU velocity over the 36 admitted sessions (minimum 0.801).
- The admitted synchronized sessions have median same-row GPS trajectory
  separation of 30.2 m. No automatic lag correction was applied.
- ECU velocity is converted from km/h to m/s before forming
  `v[t] - v[t-10]` for the 1-second target.
- ECU longitudinal acceleration has correlation 0.79-0.84 with the endpoint
  1-second delta target, and 0.88-0.91 after matching the interval average.
  This confirms target timing and units are coherent.
- The expected best ECU-acceleration lag is about -0.5 seconds because the
  endpoint delta represents the preceding 1-second interval.
- Stateful rollout begins from exactly one ground-truth speed. Subsequent
  updates receive IMU features only. Distance uses trapezoidal integration over
  the stored phone elapsed timestamps.
- Features are rebuilt inside each continuous segment and the first 100 rows
  are purged. Driver and drive tests use complete-session partitions, never
  randomly split rows.

## Gravity and orientation

The gravity vector is internally consistent: mean magnitude is approximately
9.807 m/s2 and is almost entirely on phone Z. Gravity subtraction is therefore
reasonable.

The reported orientation pitch averages roughly -82 degrees while gravity
remains on phone Z. Consequently, treating these fields as ordinary SciPy
yaw/pitch/roll Euler angles is not justified. The original experiment retained
raw and gravity-corrected phone axes as well, so this questionable transform did
not remove the source signal. Future experiments should exclude the Euler-world
features unless the dataset's exact orientation convention is established.

## Signal diagnostics

Phone linear-acceleration noise is large relative to vehicle acceleration:

| Driver | Phone X std | Phone Y std | Phone Z std | ECU longitudinal std |
|---|---:|---:|---:|---:|
| A | 1.10 | 1.08 | 0.58 | 0.74 m/s2 |
| B | 1.25 | 1.27 | 0.63 | 0.82 m/s2 |
| E | 2.17 | 1.87 | 0.96 | 0.69 m/s2 |

An oracle linear combination of causal 1-second mean phone XYZ, fitted and
scored within each driver, reaches correlation only 0.10 for A, 0.19 for B,
and 0.10 for E. Searches over +/-30 seconds do not reveal one consistent
missing IMU lag. Longer 2-10 second averaging remains weak.

## Whole-drive same-driver controls

- Driver A: best iteration 0; test delta-v MAE 0.470 m/s versus 0.468 m/s for
  zero delta. The learned blackout rollout is indistinguishable from holding
  speed.
- Driver E: best iteration 433; test delta-v MAE 0.378 m/s versus 0.391 m/s for
  zero delta. Sixty-second velocity MAE improves from 14.44 to 9.95 km/h, which
  shows some repeatable nonlinear signal for this driver's sessions, but it is
  still not operationally sufficient.
- Driver B has only one admitted session, so a leakage-safe same-driver
  whole-drive split is impossible.

Artifacts are in `artifacts/ecu_delta_v_within_driver_A/` and
`artifacts/ecu_delta_v_within_driver_E/`.

## Causal pre-blackout calibration

A local ridge mapping from causal 1-second phone XYZ mean/std to delta velocity
was fitted from only the 120 GNSS-valid seconds preceding every blackout.
Calibration length and regularization were selected on Driver A, then frozen
before evaluating Driver B.

On Driver B it changes mean episode velocity MAE versus holding speed as follows:

| Blackout | Causal calibration | Hold speed |
|---:|---:|---:|
| 10 s | 6.38 km/h | 6.48 km/h |
| 30 s | 12.00 km/h | 11.97 km/h |
| 60 s | 17.06 km/h | 14.96 km/h |
| 120 s | 24.16 km/h | 16.30 km/h |

This is accelerometer bias accumulation. Full metrics and episode records are
in `artifacts/causal_phone_calibration/`.

## Decision for the next experiment

Preserve all current results. Do not improve them by relaxing the split.
Exclude the unverified Euler transform, retain causal raw/gravity-relative
signals, and next test a temporal denoising model with an explicit bias/state
estimator and hold-speed residual gate. The gate must allow the learned update
only when validation evidence predicts acceleration reliably; otherwise the
system should retain the last-speed baseline rather than integrate noise.
