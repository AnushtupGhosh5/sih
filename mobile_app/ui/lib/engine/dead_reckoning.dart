import 'dart:math' as math;

import 'calibration.dart';
import 'geo.dart';
import 'vec3.dart';

enum SpeedMode {
  /// Hold the last GNSS speed (the deployed mode: within 1–2 % of oracle
  /// speed over a tunnel on IO-VNBD).
  holdLastGnss,

  /// Integrate the calibrated forward acceleration (needs a good forward axis).
  inertial,
}

/// Free inertial dead reckoning in a local ENU frame (port of `dead_reckon()`
/// in `src/idr/dead_reckoning.py`, run sample by sample).
class DeadReckoner {
  DeadReckoner({
    required this.lat0,
    required this.lon0,
    required double v0,
    required double h0,
    required this.cal,
    this.speedMode = SpeedMode.holdLastGnss,
    this.compassGain = 0.0,
    this.accelTau = 0.3,
    this.resumeSeconds = 3.0,
  })  : v = v0,
        _vHold = v0,
        heading = h0,
        _resume = resumeSeconds;

  final double lat0, lon0;
  final Calibration cal;
  final SpeedMode speedMode;

  /// Fraction of the compass error applied per 0.1 s (Python `fuse_gain`).
  final double compassGain;
  final double accelTau;
  final double resumeSeconds;

  double east = 0, north = 0;
  double v; // m/s
  double heading; // radians, clockwise from north
  double distance = 0; // odometry metres since seed
  double elapsed = 0; // seconds since seed

  final double _vHold;
  double _aFilt = 0;
  double _resume;
  bool _wasStationary = false;

  ({double lat, double lon}) get position => enuToLatLon(east, north, lat0, lon0);

  /// Current yaw rate for a raw gyro sample using the calibration.
  double yawRate(Vec3 gyro, Vec3 ghat) => cal.yawSign * gyro.dot(ghat) - cal.yawBias;

  void step({
    required Vec3 lin,
    required Vec3 gyro,
    required Vec3 ghat,
    required double dt,
    double? compassHeading,
    bool stationary = false,
  }) {
    final yaw = yawRate(gyro, ghat);

    // ── speed ──────────────────────────────────────────────────────────
    if (stationary) {
      v = 0;
      _wasStationary = true;
    } else if (speedMode == SpeedMode.inertial && cal.hasForwardAxis) {
      final a = lin.dot(cal.fwd!) - cal.accelBias;
      _aFilt += (a - _aFilt) * (dt / (accelTau + dt));
      v = math.max(0.0, v + _aFilt * dt);
    } else {
      if (_wasStationary) {
        _wasStationary = false;
        _resume = 0;
      }
      if (_resume < resumeSeconds) {
        _resume += dt;
        v = math.min(_vHold, v + _vHold * dt / resumeSeconds);
      } else {
        v = _vHold;
      }
    }

    // ── heading ────────────────────────────────────────────────────────
    heading += yaw * dt;
    if (compassHeading != null && compassGain > 0) {
      final g = math.min(1.0, compassGain * dt * 10.0);
      heading += g * wrapPi(compassHeading - heading);
    }
    heading = wrapPi(heading);

    // ── position ───────────────────────────────────────────────────────
    east += v * math.sin(heading) * dt;
    north += v * math.cos(heading) * dt;
    distance += v * dt;
    elapsed += dt;
  }
}
