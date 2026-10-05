# IDR navigation app (Flutter, Android)

The phone side of the Intelligent Dead Reckoning system. The whole engine
runs on the device in pure Dart, fed by the phone's accelerometer, gyroscope,
magnetometer and GNSS: no OBD-II, no server, no ML runtime.

## What runs on the phone (`lib/engine/`)

| Module | Problem-statement item | What it does |
|---|---|---|
| `alignment.dart`, `calibration.dart` | In-Vehicle Alignment & Calibration Engine | Gravity gives the vertical axis. The forward axis is the closed-form rotation that best maps GNSS kinematics (dv/dt forward, −speed·heading-rate lateral) onto the measured horizontal acceleration, so a turn or an acceleration pins it to about 1°. Yields the phone's pitch / roll / yaw on the mount (shown in the UI), the gyro yaw sign and bias, and the compass offset. A knock on the mount invalidates it. |
| `fusion_filter.dart` | GNSS+INS Fusion Engine · Seamless GNSS Deficit Handler | A 2-D extended Kalman filter: east, north, heading, speed, gyro bias, accelerometer bias. It predicts from the aligned IMU at sensor rate and takes GNSS position / speed / course when the fix is healthy, zero-velocity updates at stops, and (optionally) the compass. Fixes are chi-square gated against the filter, so multipath jumps are refused; three consistent refused fixes in a row re-anchor the filter. In an outage it simply keeps predicting: the switch is seamless by construction and the covariance (the 2σ circle on the map) says how far it may be off. |
| speed path in `navigation_engine.dart` | AI Speed & Vibration Filter | Forward speed from the accelerometer with a continuously learned bias (learned only during gentle, straight driving), pothole shocks dropped, zero-velocity detection at stops, and a constant-speed prior in quiet stretches of an outage that re-learns the residual bias, so only genuine braking / acceleration events are integrated. Road vibration RMS is shown live. |
| `road_graph.dart`, `map_matcher.dart` | Advanced Map-Matching & Kinematic Constraints | Offline OSM road graph; during an outage odometry walks the graph, the gyro picks the branch at junctions, heading resets to the road bearing on every edge (non-holonomic: no sideways motion). Falls back to free inertial at the edge of the downloaded area. |
| `gnss_monitor.dart` | Seamless GNSS Deficit Handler | GOOD / DEGRADED / DENIED from fix accuracy and age; no accepted fix for 2 s (or the demo switch) declares the outage. |
| `navigation_engine.dart` | Real-time Navigation Interface (10 Hz) | Orchestrates everything and emits the solution at 10 Hz. |

`services/osm_service.dart` downloads the drivable road network within 2 km
while GNSS is healthy (parsed in a background isolate) and caches it on disk.
`services/log_service.dart` records sensors, fixes and the solution to CSV.

## Using it

1. Install the APK, grant location and sensor permissions, mount the phone.
2. Drive for ~30 s with GNSS. The status strip shows `ALIGNED · CAL 0.xx ·
   MAP n.nk`; the MOUNT read-out shows the phone's pitch / roll / yaw.
3. Enter a tunnel. Within about 2 s of the last good fix the banner flips to
   *Dead Reckoning*, the vehicle keeps moving from the IMU, the orange trail
   is the reckoned path, the hollow dot the free-inertial solution when the
   road constraint is active, and the circle the filter's 2σ uncertainty.
   For a demo without a tunnel, tap the GPS-off button while driving.
4. When GNSS returns the strip reports `LAST DR … m · … m OFF (… %)`: the
   measured drift against the first good fix, i.e. the SIH benchmark, live.
5. The tune button opens diagnostics: biases, uncertainties, fix statistics,
   calibration quality, and switches for the demo outage, compass pull,
   follow mode and the drive logger. Log files land in
   `Android/data/<package>/files/idr_logs/` and replay through the desktop
   pipeline.

## Tests

```bash
flutter test
```

`test/engine/` replays synthetic drives (speed/heading profiles, random phone
mount, sensor noise, gyro bias, road vibration, pothole shocks, multipath)
through the engine and checks, among others: forward axis recovered to within
a few degrees and mount angles within 4°; a 40 s blackout with a turn under
4 % drift; braking and a full stop-and-go inside a tunnel captured; pothole
shocks harmless; GNSS loss declared automatically within the stale timeout
while the position keeps moving; multipath jumps refused; a steady GNSS
offset accepted after three consistent fixes; the filter learns gyro and
accelerometer biases; map-aided reckoning pinned to the road.
