<div align="center">

# Intelligent Dead Reckoning (IDR) with GNSS Fusion

**Phone-only navigation that keeps working when GNSS disappears.**

Smart India Hackathon 2026 · Problem Statement **SIH26168** (ISRO) · Team **KANABI**

![Platform](https://img.shields.io/badge/platform-Android-3DDC84)
![Engine](https://img.shields.io/badge/engine-on--device%20Dart-0175C2)
![Training](https://img.shields.io/badge/training-Python%20%7C%20PyTorch-EE4C2C)
![Dataset](https://img.shields.io/badge/dataset-IO--VNBD-6f42c1)
![Tests](https://img.shields.io/badge/engine%20tests-37%20passing-brightgreen)

</div>

---

## The problem

GNSS signals vanish in tunnels, underpasses, multi-level parking, urban canyons and under jamming. Phone navigation then freezes or jumps. Most vehicles on Indian roads, including trucks, older cars and two-wheelers, have no factory inertial navigation and no OBD-II speed feed. They have only a phone on the dashboard.

## Our solution

IDR turns an ordinary smartphone into a self-contained navigator. It uses the phone's **accelerometer, gyroscope and magnetometer** plus an **offline OpenStreetMap road graph**. It needs no OBD-II, no wheel sensor and no fixed mount.

When GNSS is lost, the app switches to dead reckoning automatically, the vehicle marker keeps moving, and the path stays on the road. When GNSS returns, the app measures its own drift.

![1 km GNSS blackout on real roads: plain inertial drifts 76 %, map-aided IDR 2.5 %](docs/figures/mapmatch_1km_demo.png)

*IO-VNBD drive vfa02, 1 km GNSS blackout with a roundabout. Green is ground truth, red dashed is the plain phone IMU (76 % drift), blue is IDR with map matching (2.5 % drift).*

---

## Results

All results use the **smartphone IMU only**, with GNSS blackouts simulated on **held-out IO-VNBD test drives**.

| Scenario | PS benchmark | Achieved |
|---|---|---|
| 50 m blackout, motorway | < 5 m | **0.4 m** (0.7 %) |
| 1 km straight, tunnel-like run at ~100 km/h | < 100 m | **7.9 m** (0.8 %) |
| 1 km on real roads with turns, map-aided, median | < 10 % | **3.3 %** highway · **4.8–7.0 %** urban |
| 1 km blackouts finishing under 10 % drift (369 blackouts) | — | **17 % → 74 %** (plain inertial → map-aided) |

| Drive | Plain inertial, median | **Map-aided, median** | Under 10 % |
|---|---|---|---|
| vfa02 (highway) | 21 % | **3.3 %** | 91 % |
| vtb5 (urban) | 39 % | **7.0 %** | 62 % |
| vw2 (urban) | 41 % | **4.8 %** | 70 % |

| 1 km tunnel-like run | Plain inertial drift vs. blackout distance |
|---|---|
| ![](docs/figures/blackout_1km_tunnel.png) | ![](docs/figures/drift_vs_distance.png) |

### What the data taught us

- **Speed is the easy part.** Holding the last GNSS speed through a tunnel costs only 1–2 % extra position error, because vehicles in tunnels move at nearly constant speed. Absolute speed cannot be read from 10 Hz vibration, because tyre and engine harmonics alias above the 5 Hz Nyquist limit.
- **Heading is the real problem.** A phone gyroscope drifts 19–28° in 30–60 s, even with its bias optimally removed. Plain inertial navigation stays under 10 % only up to about 200 m.
- **So we constrain heading with the road.** Map matching extends sub-10 % performance from about 200 m to 1 km on real roads.

Full numbers and method: [RESULTS.md](RESULTS.md).

---

## How it works

```
                        ┌──────────── DESKTOP (once) ─────────────┐
 IO-VNBD drives ──► sync & clean ──► features ──► train models ──► simulated-blackout evaluation ──► export

                        ┌──────────── PHONE (live) ───────────────┐
 phone IMU ──► align & calibrate ──► Kalman fusion ◄── GNSS fix
                                          │
                                   outage detector ──► map matching on offline OSM ──► position · speed · heading @ 10 Hz
```

| Problem-statement requirement | Our implementation |
|---|---|
| **In-vehicle alignment & calibration** | Gravity gives the vertical axis. The forward axis comes from fitting GNSS kinematics to the measured acceleration. The app reports the phone's pitch, roll and yaw on its mount, and recalibrates if the phone is knocked. |
| **AI speed & vibration filter** | A 1-D CNN (32 k parameters) studied on IO-VNBD. On the phone: accelerometer speed with a learned bias, pothole-shock rejection, zero-velocity detection at stops, and a constant-speed prior in quiet tunnel stretches. |
| **Map matching & kinematic constraints** | During an outage, odometry walks the offline road graph. The gyro picks the branch at junctions, and heading resets to the road bearing on every segment, a non-holonomic constraint. |
| **GNSS+INS fusion engine** | A Kalman filter over position, heading, speed, gyro bias and accelerometer bias. It learns the biases while GNSS is healthy and rejects multipath jumps by innovation gating. |
| **Seamless GNSS deficit handler** | An outage is declared automatically about 2 s after the last accepted fix. The filter keeps predicting, so position output never stops. Recovery happens on the first good fix. |
| **Real-time navigation interface** | A 10 Hz Android HUD with a smooth vehicle marker, mode banner, dead-reckoned trail, uncertainty circle, vehicle-frame traces, diagnostics panel and drive logger. |

---

## Repository layout

| Path | Contents |
|---|---|
| [`mobile_app/ui/`](mobile_app/ui) | Flutter Android app. The full navigation engine runs on the phone in pure Dart under `lib/engine/`. See its [README](mobile_app/ui/README.md). |
| [`src/idr/`](src/idr) | IO-VNBD loader, IMU features, CNN speed model, dead reckoning, OSM road graph, map matching, evaluation and figures. |
| [`src/alignment/`](src/alignment) | Python reference for the in-vehicle alignment engine, with a test. |
| [`src/sensor_fusion/`](src/sensor_fusion) | Python reference fusion: 15-state error-state EKF, INS mechanisation, NHC/ZUPT, GNSS quality gating, LSTM IMU-error model, tests. |
| [`training/sensor_fusion/`](training/sensor_fusion) | LSTM training, evaluation and ONNX/TFLite export. |
| [`scripts/`](scripts) | Dataset audits, inertial alignment recovery and training experiments. See [TRAINING.md](TRAINING.md). |
| [`reports/`](reports) | Data-quality and alignment audits. |
| [`demo/`](demo) | Browser replay of a real IO-VNBD drive through a 1 km blackout. Open `demo/index.html`. |
| [`docs/figures/`](docs/figures) | Result plots used in this README. |
| [`edge_engine/`](edge_engine) | Placeholder for the edge build for external and FOG IMUs at ~200 Hz. Planned. |

---

## Quick start

### Android app

Download the latest APK from [Releases](../../releases), install it, and grant location and sensor permissions.

1. Mount the phone and drive for about 30 s with GNSS. The status strip shows `ALIGNED · CAL · MAP` when calibration and the road graph are ready.
2. Enter a tunnel. For a demo without one, tap the **GPS-off** button while moving.
3. When GNSS returns, read the measured drift in the status strip: `LAST DR … m · … m OFF (… %)`.

Build from source:

```bash
cd mobile_app/ui
flutter pub get
flutter test               # 37 engine tests on synthetic drives
flutter build apk --release
```

### Desktop pipeline

```bash
pip install -r requirements.txt
python scripts/download_data.py          # synchronised IO-VNBD drives
python -m src.idr.train                  # CNN speed and vibration model
python -m src.idr.evaluate               # 50 m and 1 km blackout plots, drift vs distance
python -m src.idr.evaluate_mapmatch      # 369 one-km blackouts, map-aided vs plain inertial
python -m src.idr.demo_export            # regenerate the browser replay
```

Docker wrappers for the longer experiments are described in [TRAINING.md](TRAINING.md).

---

## Status

| Done | Next |
|---|---|
| Dead reckoning and map matching validated on IO-VNBD | Real-vehicle tunnel test with the built-in drive logger |
| Android app with the full engine on the phone | Threshold tuning on our own Indian-road drives, including two-wheelers |
| 37 automated engine tests on synthetic drives | 200 Hz edge build for external and FOG IMUs |
| Reproducible desktop pipeline and browser replay demo | Multi-hypothesis map matching for ambiguous junctions |

The app has not yet been driven through a real tunnel. All quantitative results above come from IO-VNBD with simulated blackouts.

---

## Team KANABI

| Member | Focus |
|---|---|
| Abhilash | Alignment and calibration |
| Ritu | Dead reckoning, map matching, evaluation |
| Anushtup | Sensor fusion |
| Nilesh | Mobile app and integration |

## References

- Onyekpe U. et al. *IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning.* Data in Brief, 2021. [github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)
- Newson P., Krumm J. *Hidden Markov Map Matching Through Noise and Sparseness.* ACM SIGSPATIAL, 2009.
- Brossard M., Barrau A., Bonnabel S. *AI-IMU Dead-Reckoning.* IEEE Transactions on Intelligent Vehicles, 2020.
- Herath S., Yan H., Furukawa Y. *RoNIN: Robust Neural Inertial Navigation in the Wild.* ICRA, 2020.
- Solà J. *Quaternion kinematics for the error-state Kalman filter.* arXiv:1711.02508, 2017.
- Groves P. *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed., 2013.
- © OpenStreetMap contributors.
