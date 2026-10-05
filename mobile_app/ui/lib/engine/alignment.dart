import 'dart:collection';

import 'vec3.dart';

enum AlignmentState { uncalibrated, zAligned, fullyAligned }

/// In-Vehicle Alignment & Calibration Engine (port of
/// `src/alignment/alignment_engine.py`).
///
/// Estimates the rotation from the phone frame to the vehicle frame
/// (forward, right, down) from IMU data alone:
///   1. gravity while stationary gives the vertical axis,
///   2. the dominant horizontal acceleration while driving straight gives
///      the forward axis (PCA when GNSS is absent, mean forward-acceleration
///      direction when GNSS confirms the vehicle is speeding up),
///   3. the lateral axis completes a right-handed frame.
/// A sudden gyro burst (phone knocked or re-mounted) invalidates the matrix.
class InVehicleAligner {
  InVehicleAligner({this.samplingRate = 10.0})
      : windowSize = (samplingRate * 2).round(),
        pcaWindowSize = (samplingRate * 10).round();

  final double samplingRate;
  final int windowSize;
  final int pcaWindowSize;

  double staticAccelVarThresh = 0.05; // (m/s^2)^2
  double staticGyroVarThresh = 0.01; // (rad/s)^2
  double turnYawRateThresh = 0.05; // rad/s
  double forwardAccelThresh = 0.5; // m/s^2
  double gravity = 9.81;

  AlignmentState state = AlignmentState.uncalibrated;

  final ListQueue<Vec3> _accel = ListQueue<Vec3>();
  final ListQueue<Vec3> _gyro = ListQueue<Vec3>();
  final List<Vec3> _horiz = <Vec3>[];

  Vec3? zVeh, xVeh, yVeh;
  Mat3 rotation = Mat3.identity;

  /// Incremented every time a displacement forces a re-calibration.
  int displacementCount = 0;

  bool get isAligned => state == AlignmentState.fullyAligned;

  void reset() {
    state = AlignmentState.uncalibrated;
    _accel.clear();
    _gyro.clear();
    _horiz.clear();
    zVeh = xVeh = yVeh = null;
    rotation = Mat3.identity;
  }

  /// Process one IMU frame. [gpsSpeed] in m/s when GNSS is available,
  /// [gpsAccel] the GNSS-derived longitudinal acceleration (m/s^2) if known.
  /// Returns true once the rotation matrix is available.
  bool processFrame(Vec3 accel, Vec3 gyro, {double? gpsSpeed, double? gpsAccel}) {
    _push(_accel, accel, windowSize);
    _push(_gyro, gyro, windowSize);
    if (_accel.length < windowSize) return false;

    switch (state) {
      case AlignmentState.uncalibrated:
        _attemptZAlignment(gpsSpeed);
      case AlignmentState.zAligned:
        _attemptXAlignment(gpsSpeed, gpsAccel);
      case AlignmentState.fullyAligned:
        _checkDisplacement();
    }
    return isAligned;
  }

  /// Rotate a raw phone-frame vector into the vehicle frame (identity until
  /// fully aligned).
  Vec3 transform(Vec3 raw) => isAligned ? rotation.apply(raw) : raw;

  // ── phases ───────────────────────────────────────────────────────────

  void _attemptZAlignment(double? gpsSpeed) {
    final accelVar = _varianceSum(_accel);
    final gyroVar = _varianceSum(_gyro);
    final stationary = gpsSpeed != null
        ? (gpsSpeed < 0.2 && accelVar < staticAccelVarThresh)
        : (accelVar < staticAccelVarThresh && gyroVar < staticGyroVarThresh);
    if (!stationary) return;
    final g = _mean(_accel);
    if (g.norm > 0) {
      zVeh = g.normalized;
      state = AlignmentState.zAligned;
    }
  }

  void _attemptXAlignment(double? gpsSpeed, double? gpsAccel) {
    final z = zVeh!;
    final accel = _accel.last;
    final gyro = _gyro.last;

    final yawRate = gyro.dot(z);
    final isLinear = yawRate.abs() < turnYawRateThresh;
    final lin = accel - z * gravity;
    final aHoriz = lin.projectOntoPlane(z);

    if (!isLinear) return; // curved: centripetal force would skew the axis

    if (gpsSpeed != null) {
      // GNSS confirms motion: collect horizontal acceleration while the
      // vehicle is speeding up, which points along the forward axis.
      final speedingUp = gpsAccel != null ? gpsAccel > 0.3 : aHoriz.norm > forwardAccelThresh;
      if (gpsSpeed > 1.0 && speedingUp && aHoriz.norm > 0.2) {
        _horiz.add(aHoriz);
        if (_horiz.length >= samplingRate.round()) {
          final m = _mean(_horiz);
          if (m.norm > 1e-6) {
            xVeh = m.normalized;
            _finalize();
          }
        }
      }
    } else {
      _horiz.add(aHoriz);
      if (_horiz.length >= pcaWindowSize) _applyPcaForXAxis();
    }
  }

  void _applyPcaForXAxis() {
    final axis = _principalAxis(_horiz);
    // Disambiguate forward/backward: braking spikes are sharper than
    // acceleration, so if the largest negative projection beats the largest
    // positive one the axis already points forward; otherwise flip it.
    var minP = double.infinity, maxP = -double.infinity;
    for (final v in _horiz) {
      final p = v.dot(axis);
      if (p < minP) minP = p;
      if (p > maxP) maxP = p;
    }
    xVeh = (minP.abs() > maxP ? axis : -axis).normalized;
    _finalize();
  }

  void _finalize() {
    final z = zVeh!;
    var y = z.cross(xVeh!).normalized;
    var x = y.cross(z).normalized;
    xVeh = x;
    yVeh = y;
    rotation = Mat3(x, y, z);
    state = AlignmentState.fullyAligned;
    _horiz.clear();
  }

  void _checkDisplacement() {
    final gyroVar = _varianceSum(_gyro);
    if (gyroVar > staticGyroVarThresh * 50) {
      displacementCount++;
      state = AlignmentState.uncalibrated;
      _horiz.clear();
      rotation = Mat3.identity;
      zVeh = xVeh = yVeh = null;
    }
  }

  // ── helpers ──────────────────────────────────────────────────────────

  static void _push(ListQueue<Vec3> q, Vec3 v, int max) {
    q.addLast(v);
    while (q.length > max) {
      q.removeFirst();
    }
  }

  static Vec3 _mean(Iterable<Vec3> vs) {
    var s = Vec3.zero;
    var n = 0;
    for (final v in vs) {
      s = s + v;
      n++;
    }
    return n == 0 ? Vec3.zero : s * (1.0 / n);
  }

  /// Sum over axes of the per-axis variance (numpy: var(axis=0).sum()).
  static double _varianceSum(Iterable<Vec3> vs) {
    final m = _mean(vs);
    var sx = 0.0, sy = 0.0, sz = 0.0;
    var n = 0;
    for (final v in vs) {
      final d = v - m;
      sx += d.x * d.x;
      sy += d.y * d.y;
      sz += d.z * d.z;
      n++;
    }
    if (n == 0) return 0;
    return (sx + sy + sz) / n;
  }

  /// First principal component of 3-D data via power iteration on the
  /// covariance matrix.
  static Vec3 _principalAxis(List<Vec3> data) {
    final m = _mean(data);
    var cxx = 0.0, cxy = 0.0, cxz = 0.0, cyy = 0.0, cyz = 0.0, czz = 0.0;
    for (final d in data) {
      final v = d - m;
      cxx += v.x * v.x;
      cxy += v.x * v.y;
      cxz += v.x * v.z;
      cyy += v.y * v.y;
      cyz += v.y * v.z;
      czz += v.z * v.z;
    }
    var v = const Vec3(1.0, 0.7, 0.3).normalized;
    for (var i = 0; i < 80; i++) {
      final w = Vec3(
        cxx * v.x + cxy * v.y + cxz * v.z,
        cxy * v.x + cyy * v.y + cyz * v.z,
        cxz * v.x + cyz * v.y + czz * v.z,
      );
      if (w.norm < 1e-12) break;
      v = w.normalized;
    }
    return v;
  }
}
