# IDR — Training & Dead-Reckoning Results (IO-VNBD)

Intelligent Dead Reckoning proof-of-concept trained/evaluated on the **synchronised
IO-VNBD** dataset (smartphone IMU input, vehicle VBOX velocity/position as ground
truth). Everything here is reproducible from `src/idr/`.

## Pipeline
1. `scripts/download_data.py` — pulls the 72 synchronised V+S drive pairs from the
   IO-VNBD Git-LFS store into `data/raw/`.
2. `src/idr/dataset.py` — pairs V/S drives, fixes units (phone GPS speed is m/s,
   vehicle velocity km/h → ×1/3.6), builds mounting-invariant IMU features, and
   derives a robust heading reference (course-over-ground from VBOX position).
3. `src/idr/model.py` + `train.py` — AI Speed & Vibration Filter (1-D CNN, 32 k
   params) mapping an IMU window → forward speed, trained within-vehicle (Driver E).
4. `src/idr/dead_reckoning.py` — pre-blackout calibration (forward axis, biases,
   compass offset) + INS propagation through simulated GNSS blackouts.
5. `src/idr/evaluate.py` — position plots + drift-vs-distance curve.

## Headline results (test drives, smartphone IMU only)

| Scenario | PS benchmark | Achieved |
|---|---|---|
| 50 m blackout (straight) | < 5 m over 50 m | **0.4 m (0.7 %)** |
| 1 km blackout (tunnel-like straight run) | < 100 m over 1 km | **7.9 m (0.8 %)** |

Figures in `artifacts/figures/`: `blackout_50m.png`, `blackout_1km_tunnel.png`,
`drift_vs_distance.png`.

## Honest drift-vs-distance curve (const-speed + gyro heading, general roads)

| Distance | Median drift | % of blackouts < 10 % |
|---|---|---|
| 50 m | 4.0 % | 83 % |
| 100 m | 5.7 % | 72 % |
| 200 m | 8.7 % | 55 % |
| 350 m | 12.6 % | 41 % |
| 500 m | 17.7 % | 30 % |
| 1000 m | 29.3 % | 17 % |

## Map-matching (the heading fix) — 1 km blackouts on real roads

Free inertial DR loses heading beyond ~200 m. Snapping the trajectory to the OSM
road graph (`src/idr/osm.py`, `map_matching.py`) fixes it: odometry distance walks
the road network and the gyro only picks the branch at each junction, with heading
reset to the road bearing on every edge. Evaluated over **369 one-km blackouts**:

| Drive | Free DR (median) | **Map-aided DR (median)** | best | % of blackouts < 10% |
|---|---|---|---|---|
| vfa02 (highway) | 21 % | **3.3 %** | 0.1 % | **91 %** |
| vtb5 (urban) | 39 % | **7.0 %** | 0.1 % | 62 % |
| vw2 (urban) | 41 % | **4.8 %** | 0.2 % | 70 % |

Demo (`artifacts/figures/mapmatch_1km_demo.png`): a curvy 1 km route with a
roundabout — free DR drifts **76 %** into open ground, map-aided DR tracks the road
(incl. the roundabout) to **2.5 %**. Reproduce: `python -m src.idr.evaluate_mapmatch`.

Median 1 km drift now beats the <10 % benchmark on every test drive; the remaining
urban misses are wrong-turn cases at ambiguous junctions (a consequence of this
dataset's noisy gyro) — addressable with a better IMU or multi-hypothesis matching.

## Key engineering findings
- **Speed is the easy part.** Holding the last GNSS speed (V0) through a blackout is
  within ~1–2 % of oracle speed for position — because over a tunnel the vehicle is
  near-constant speed. Absolute speed prediction from a 10 Hz vibration window is
  *not* recoverable (tire/engine harmonics alias above the 5 Hz Nyquist limit), so
  the AI speed filter plateaus at ~6 m/s RMSE and is **not** the deployable speed
  source — anchoring to V0 + INS is.
- **Heading is the whole problem.** With perfect heading, const-speed DR gives
  2.6 % (10 s) / 4.9 % (30 s). Smartphone gyro drifts ~19–28° over 30–60 s *even
  with optimal bias*; the in-vehicle magnetometer/compass is worse (magnetic
  disturbance). So inertial-only DR meets < 10 % up to ~150–200 m on general roads,
  and up to ≥ 1 km on straight tunnel-like runs.
- **This is exactly why the PS mandates Map-Matching + NHC** — snapping the
  trajectory to the road network is the correction that extends < 10 % to arbitrary
  1 km routes. That is the recommended next module.

## Bug fixed along the way
The column matcher originally matched the *Y* axis of `GYROSCOPE`/`GRAVITY` to the
*X* column (the letter "y" occurs inside the words "gyroscope"/"gravity"),
duplicating axes and corrupting heading + features. Fixed in `dataset._match`.
