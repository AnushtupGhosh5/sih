# Inertial alignment recovery and signal ceiling

## Main finding

Sparse smartphone GPS was not a reliable proxy for phone-IMU timing in the
manually synchronized IO-VNBD pairs. Eleven sessions previously admitted at
zero row lag have strong phone acceleration and yaw peaks agreeing at the same
nonzero inertial lag. Several offsets are 1-9 seconds, enough to attach vehicle
delta-velocity labels to the wrong phone motion event.

The replacement alignment requires agreement between two independent signals:

- gravity-corrected phone acceleration versus ECU longitudinal acceleration;
- phone gyroscope versus ECU yaw rate.

High confidence requires absolute correlations of at least 0.25 and 0.35,
respectively, lag agreement within five rows, and a peak away from the search
boundary. A labeled moderate tier admits very strong yaw evidence under a
slightly wider agreement tolerance. Driver D is not admitted.

## Recovered data

- Original pairs: 72.
- Inertially aligned sessions: 20.
- High confidence: 17.
- Moderate confidence: 3.
- Prepared rows after alignment/quality trimming: 406,807.
- Drivers: A (4 sessions), B (1), E (15); Driver D excluded.
- Continuous segments: 45.
- Independent archive validation failures: 0.

The corrected dataset is `artifacts/ecu_training_dataset_inertial_aligned/`.
Every archive stores original `source_row` and `vehicle_source_row`; validation
checks that their difference equals the selected session offset.

## Signal change

For the 1-second target, the best simple phone-axis correlation increased from
about 0.08 in the old dataset to 0.31 after inertial alignment. ECU
longitudinal-acceleration consistency remained approximately 0.90.

The corrected CatBoost diagnostic no longer stops at four trees. It reaches:

- Driver A validation delta-v R2: 0.407;
- validation MAE improvement over zero delta: 24.4%;
- Driver B test delta-v R2: 0.233;
- test MAE improvement over zero delta: 5.4%.

This proves that corrected smartphone IMU contains recoverable vehicle-motion
information. Recursive output still has a roughly +0.04 m/s per-second bias,
which causes unacceptable long-horizon drift.

## Recoverability ceilings

The causal 120-second pre-blackout ridge calibration improves Driver B at short
horizons but diverges at long horizons. A validation-selected global CatBoost
bias/shrinkage correction also remains above the target.

An intentionally leaky per-blackout affine oracle was used only as an upper
bound. Its median distance drift is:

| Blackout | Median oracle drift |
|---:|---:|
| 10 s | 2.8% |
| 30 s | 8.6% |
| 60 s | 12.6% |
| 120 s | 16.6% |

Therefore, sub-10% behavior exists in the phone signal for short outages, but
affine calibration alone cannot meet the requirement at 60-120 seconds. The
next model must learn episode-specific bias from pre-blackout context and train
through multi-step velocity/distance integration. NHC and map constraints will
still be required for robust long outages.

## Artifacts

- `artifacts/inertial_alignment_recovery/`: manifest, full lag sweeps and plots.
- `artifacts/ecu_training_dataset_inertial_aligned/`: corrected archives and provenance.
- `artifacts/causal_phone_calibration_inertial_aligned/`: causal calibration results.
- `artifacts/ecu_delta_v_catboost_inertial_aligned/`: corrected nonlinear baseline.
- `artifacts/recoverability_ceiling_inertial_aligned/`: bias-corrected and leaky-oracle ceilings.
