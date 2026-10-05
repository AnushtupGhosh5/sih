# IDR navigation app (Flutter, Android)

The phone-side of the Intelligent Dead Reckoning system: the full dead-reckoning
engine runs on the device in pure Dart, fed by the phone's accelerometer,
gyroscope, magnetometer and GNSS. No OBD-II, no server, no ML runtime needed.

## What runs on the phone (`lib/engine/`)

| File | Role | Ported from |
|---|---|---|
| `alignment.dart` | In-vehicle alignment & calibration engine: gravity → vertical axis, PCA / forward-acceleration → forward axis, knock detection | `src/alignment/alignment_engine.py` |
| `calibration.dart` | Pre-blackout calibration on the last 30 s of GNSS-available data: forward axis by correlation with GNSS dv/dt, accel bias, gyro yaw sign & bias, compass offset | `calibrate()` in `src/idr/dead_reckoning.py` |
| `dead_reckoning.dart` | Free INS propagation: hold-last-GNSS speed (or inertial), bias-corrected gyro heading about the live vertical, ZUPT at stops, optional weak compass pull | `dead_reckon()` in `src/idr/dead_reckoning.py` |
| `road_graph.dart` | Routable OSM road graph in local ENU metres with a grid index | `RoadNetwork` in `src/idr/osm.py` |
| `map_matcher.dart` | Map-aided dead reckoning: odometry walks the road graph, gyro picks the branch at junctions, heading resets to the road bearing on every edge | `map_aided_dr()` in `src/idr/map_matching.py` |
| `gnss_monitor.dart` | GNSS deficit handler: GOOD / DEGRADED / DENIED from fix accuracy and age | `src/sensor_fusion/gnss_quality.py` |
| `navigation_engine.dart` | Orchestrator: shadow reckoner seeded at every good fix (so the seconds before an outage is detected are covered), instant switch to dead reckoning, map anchoring, recovery statistics when GNSS returns | — |

`services/osm_service.dart` downloads the drivable road network within 2.5 km
of the vehicle from Overpass while GNSS is healthy and caches it on disk, so
the map constraint is available offline inside the tunnel.

## Using it

1. Install the APK, grant location and sensor permissions, mount the phone.
2. Drive for ~30 s with GNSS (any mount orientation). The status strip shows
   `ALIGNED · CAL 0.xx · MAP n.nk` once the mount is calibrated and the road
   graph is loaded.
3. Enter a tunnel, or tap **SIMULATE TUNNEL**. The banner flips to *Dead
   Reckoning*, the vehicle keeps moving from the IMU, the orange trail shows
   the reckoned path and a hollow ghost marks the free-inertial solution when
   the map-aided one is displayed.
4. When GNSS returns (or you tap **END TUNNEL**) the strip reports
   `LAST DR <distance> m · <error> m OFF (<drift> %)`, i.e. the measured drift
   against the first good fix: the SIH benchmark, live.

## Tests

```bash
flutter test
```

`test/engine/` replays synthetic drives (known speed/heading profile, random
phone mount, sensor noise and gyro bias) through the engine:

* calibration recovers the forward axis (dot > 0.95), the yaw sign and bias
  from the pre-blackout window;
* a 40 s blackout with a 60° turn ends under 3 % drift;
* map-aided reckoning stays on the road while free inertial drifts with an
  uncalibrated bias step;
* the simulate-tunnel switch enters dead reckoning within one sample and
  recovers after two good fixes.
