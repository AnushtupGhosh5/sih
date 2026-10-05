import 'dart:math' as math;

import 'geo.dart';

/// Loosely coupled 2-D GNSS+INS fusion filter (extended Kalman filter).
///
/// State: east, north (m), heading (rad, clockwise from north), forward
/// speed (m/s), gyro yaw-rate bias (rad/s), forward accelerometer bias
/// (m/s²). Prediction uses the yaw rate about the vertical and the forward
/// acceleration from the aligned IMU; updates come from GNSS position, speed
/// and course while the fix is healthy, from zero-velocity detections at
/// stops, and optionally from the compass. In a GNSS outage the filter just
/// keeps predicting, so the switch to dead reckoning is seamless by
/// construction, and the covariance says how far the solution may be off.
class FusionFilter {
  FusionFilter({
    this.sigmaGyro = 0.02,
    this.sigmaAccel = 0.8,
    this.sigmaAccelHold = 2.5,
    this.sigmaGyroBiasWalk = 2e-4,
    this.sigmaAccelBiasWalk = 1e-2,
    this.sigmaPosWalk = 0.2,
    this.driftFraction = 0.03,
    this.maxSpeed = 60.0,
    this.gateChi2 = 13.8,
    this.maxGyroBias = 0.05,
    this.maxAccelBias = 0.6,
    this.biasLearnMaxInnovation = 0.8,
  });

  /// Physical bounds on the bias states (a phone gyro drifts well under
  /// 3 deg/s; a 3 degree mount-tilt error is 0.5 m/s² of gravity leakage).
  final double maxGyroBias;
  final double maxAccelBias;

  /// Speed innovations larger than this (m/s) are applied to the speed only,
  /// never to the accelerometer bias: a big disagreement is a tracking
  /// transient, not a sensor bias.
  final double biasLearnMaxInnovation;

  /// Systematic dead-reckoning drift as a fraction of distance travelled,
  /// used to grow the position covariance during outages.
  final double driftFraction;

  /// Yaw-rate white noise (rad/s per sample).
  final double sigmaGyro;

  /// Forward-acceleration white noise (m/s² per sample) when the accelerometer drives speed.
  final double sigmaAccel;

  /// Speed process noise (m/s² per sample) when no forward axis is known (constant-speed model).
  final double sigmaAccelHold;
  final double sigmaGyroBiasWalk; // rad/s per sqrt(s)
  final double sigmaAccelBiasWalk; // m/s² per sqrt(s)
  final double sigmaPosWalk; // m per sqrt(s), kinematic model error
  final double maxSpeed;

  /// Chi-square gate on the GNSS position innovation (2 dof, 99.9 %).
  final double gateChi2;

  static const int dim = 6;
  final List<double> x = List<double>.filled(dim, 0);
  List<List<double>> P = _zeros();
  bool initialized = false;

  double lastInnovationM = 0;
  int rejectedInRow = 0;
  int accepted = 0;
  int rejected = 0;

  double get east => x[0];
  double get north => x[1];
  double get heading => x[2];
  double get speed => x[3];
  double get gyroBias => x[4];
  double get accelBias => x[5];

  double get posSigma => math.sqrt(math.max(0, P[0][0] + P[1][1]));
  double get headingSigma => math.sqrt(math.max(0, P[2][2]));
  double get speedSigma => math.sqrt(math.max(0, P[3][3]));

  void init({
    required double e,
    required double n,
    double? heading,
    required double speed,
    double gyroBias = 0,
    double accelBias = 0,
    double posSigma = 5.0,
    double headingSigma = 0.1,
    double speedSigma = 1.0,
    double gyroBiasSigma = 0.02,
    double accelBiasSigma = 0.3,
  }) {
    x[0] = e;
    x[1] = n;
    x[2] = heading ?? 0;
    x[3] = speed;
    x[4] = gyroBias;
    x[5] = accelBias;
    P = _zeros();
    P[0][0] = posSigma * posSigma;
    P[1][1] = posSigma * posSigma;
    P[2][2] = heading == null ? math.pi * math.pi : headingSigma * headingSigma;
    P[3][3] = speedSigma * speedSigma;
    P[4][4] = gyroBiasSigma * gyroBiasSigma;
    P[5][5] = accelBiasSigma * accelBiasSigma;
    initialized = true;
    rejectedInRow = 0;
  }

  /// Snap the position to a fix after repeated gross disagreement.
  void reinitPosition(double e, double n, double sigma) {
    x[0] = e;
    x[1] = n;
    for (var i = 0; i < dim; i++) {
      P[0][i] = P[i][0] = 0;
      P[1][i] = P[i][1] = 0;
    }
    P[0][0] = P[1][1] = sigma * sigma;
    rejectedInRow = 0;
  }

  /// The forward axis changed (new calibration or the phone moved): the
  /// accelerometer bias is a property of that axis.
  void resetAccelBias(double ba, double sigma) {
    x[5] = ba.clamp(-maxAccelBias, maxAccelBias);
    for (var i = 0; i < dim; i++) {
      P[5][i] = P[i][5] = 0;
    }
    P[5][5] = sigma * sigma;
  }

  /// Propagate by one IMU sample. [omega] is the sign-corrected yaw rate
  /// (rad/s), [aFwd] the forward acceleration (m/s², null when no forward
  /// axis is known yet), [deadband] a soft dead-band applied to the
  /// bias-corrected acceleration (used during outages to stop sensor noise
  /// from random-walking the speed).
  /// With [learnAccelBias] false the speed/bias coupling is left out of the
  /// Jacobian, so speed innovations do not move the bias estimate (used
  /// during turns, where any residual axis error leaks centripetal
  /// acceleration into the forward channel).
  /// [outageSeconds] > 0 adds position process noise that grows with time
  /// and speed, so the covariance reflects the systematic drift of dead
  /// reckoning (about [driftFraction] of distance) rather than only sensor
  /// white noise.
  void predict({
    required double omega,
    double? aFwd,
    required double dt,
    double deadband = 0,
    bool learnAccelBias = true,
    double outageSeconds = 0,
  }) {
    if (!initialized || dt <= 0) return;
    final e0 = x[0], n0 = x[1], h = x[2], v = x[3], bg = x[4], ba = x[5];
    final sh = math.sin(h), ch = math.cos(h);
    final useAccel = aFwd != null;
    var a = 0.0;
    if (useAccel) {
      a = aFwd - ba;
      if (deadband > 0) a = softDeadband(a, deadband);
    }

    x[0] = e0 + v * sh * dt;
    x[1] = n0 + v * ch * dt;
    x[2] = wrapPi(h + (omega - bg) * dt);
    x[3] = (v + a * dt).clamp(0.0, maxSpeed);

    final F = _identity();
    F[0][2] = v * ch * dt;
    F[0][3] = sh * dt;
    F[1][2] = -v * sh * dt;
    F[1][3] = ch * dt;
    F[2][4] = -dt;
    if (useAccel && learnAccelBias) F[3][5] = -dt;

    final sa = useAccel ? sigmaAccel : sigmaAccelHold;
    final drift = driftFraction * v;
    final qPos = sigmaPosWalk * sigmaPosWalk * dt + drift * drift * outageSeconds * dt;
    final q = <double>[
      qPos,
      qPos,
      sigmaGyro * sigmaGyro * dt * dt,
      sa * sa * dt * dt,
      sigmaGyroBiasWalk * sigmaGyroBiasWalk * dt,
      sigmaAccelBiasWalk * sigmaAccelBiasWalk * dt,
    ];
    final fp = _mul(F, P);
    P = _mul(fp, _transpose(F));
    for (var i = 0; i < dim; i++) {
      P[i][i] += q[i];
    }
  }

  /// GNSS position update in the local ENU frame. Returns false when the fix
  /// fails the innovation gate (multipath jump, stale position).
  bool updatePosition(double ze, double zn, double sigma, {bool gate = true}) {
    if (!initialized) return false;
    final y = [ze - x[0], zn - x[1]];
    lastInnovationM = math.sqrt(y[0] * y[0] + y[1] * y[1]);
    final r = sigma * sigma;
    final s00 = P[0][0] + r, s01 = P[0][1], s10 = P[1][0], s11 = P[1][1] + r;
    var det = s00 * s11 - s01 * s10;
    if (det.abs() < 1e-12) det = 1e-12;
    final gamma = (y[0] * (s11 * y[0] - s01 * y[1]) + y[1] * (-s10 * y[0] + s00 * y[1])) / det;
    if (gate && gamma > gateChi2) {
      rejectedInRow++;
      rejected++;
      return false;
    }
    _update(const [0, 1], y, [r, r]);
    rejectedInRow = 0;
    accepted++;
    return true;
  }

  void updateSpeed(double zv, double sigma) {
    if (!initialized) return;
    final y = zv - x[3];
    if (y.abs() > biasLearnMaxInnovation) {
      // Decouple the accelerometer bias for this update.
      final row = List<double>.generate(dim, (i) => P[5][i]);
      for (var i = 0; i < dim; i++) {
        if (i != 5) P[5][i] = P[i][5] = 0;
      }
      _update(const [3], [y], [sigma * sigma]);
      for (var i = 0; i < dim; i++) {
        if (i != 5) P[5][i] = P[i][5] = row[i];
      }
    } else {
      _update(const [3], [y], [sigma * sigma]);
    }
    if (x[3] < 0) x[3] = 0;
    _clampBiases();
  }

  void _clampBiases() {
    x[4] = x[4].clamp(-maxGyroBias, maxGyroBias);
    x[5] = x[5].clamp(-maxAccelBias, maxAccelBias);
  }

  void updateCourse(double zh, double sigma) {
    if (!initialized) return;
    _update(const [2], [wrapPi(zh - x[2])], [sigma * sigma]);
    x[2] = wrapPi(x[2]);
    _clampBiases();
  }

  /// Zero-velocity update at a detected stop.
  void updateZupt({double sigma = 0.05}) => updateSpeed(0, sigma);

  // ── internals ────────────────────────────────────────────────────────

  /// Joseph-form measurement update for a measurement that observes the
  /// state components [idx] directly (H is a selection matrix) with
  /// independent noise variances [rDiag].
  void _update(List<int> idx, List<double> y, List<double> rDiag) {
    final m = idx.length;
    // S = H P H^T + R
    final S = List.generate(m, (j) => List.generate(m, (k) => P[idx[j]][idx[k]] + (j == k ? rDiag[j] : 0.0)));
    final sInv = _inverseSmall(S);
    // K = P H^T S^-1  (dim x m)
    final K = List.generate(dim, (i) => List.generate(m, (k) {
          var s = 0.0;
          for (var j = 0; j < m; j++) {
            s += P[i][idx[j]] * sInv[j][k];
          }
          return s;
        }));
    // x += K y
    for (var i = 0; i < dim; i++) {
      var s = 0.0;
      for (var j = 0; j < m; j++) {
        s += K[i][j] * y[j];
      }
      x[i] += s;
    }
    // P = (I - K H) P (I - K H)^T + K R K^T
    final ikh = _identity();
    for (var i = 0; i < dim; i++) {
      for (var j = 0; j < m; j++) {
        ikh[i][idx[j]] -= K[i][j];
      }
    }
    final left = _mul(_mul(ikh, P), _transpose(ikh));
    for (var i = 0; i < dim; i++) {
      for (var k = 0; k < dim; k++) {
        var s = 0.0;
        for (var j = 0; j < m; j++) {
          s += K[i][j] * rDiag[j] * K[k][j];
        }
        left[i][k] += s;
      }
    }
    P = left;
    // keep symmetric
    for (var i = 0; i < dim; i++) {
      for (var k = i + 1; k < dim; k++) {
        final v = 0.5 * (P[i][k] + P[k][i]);
        P[i][k] = P[k][i] = v;
      }
    }
  }

  static double softDeadband(double a, double band) {
    if (a > band) return a - band;
    if (a < -band) return a + band;
    return 0;
  }

  static List<List<double>> _zeros() => List.generate(dim, (_) => List<double>.filled(dim, 0));

  static List<List<double>> _identity() {
    final m = _zeros();
    for (var i = 0; i < dim; i++) {
      m[i][i] = 1;
    }
    return m;
  }

  static List<List<double>> _transpose(List<List<double>> a) =>
      List.generate(dim, (i) => List.generate(dim, (j) => a[j][i]));

  static List<List<double>> _mul(List<List<double>> a, List<List<double>> b) {
    final out = _zeros();
    for (var i = 0; i < dim; i++) {
      for (var k = 0; k < dim; k++) {
        final aik = a[i][k];
        if (aik == 0) continue;
        for (var j = 0; j < dim; j++) {
          out[i][j] += aik * b[k][j];
        }
      }
    }
    return out;
  }

  static List<List<double>> _inverseSmall(List<List<double>> s) {
    if (s.length == 1) {
      final v = s[0][0].abs() < 1e-12 ? 1e-12 : s[0][0];
      return [
        [1 / v]
      ];
    }
    var det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
    if (det.abs() < 1e-12) det = 1e-12;
    return [
      [s[1][1] / det, -s[0][1] / det],
      [-s[1][0] / det, s[0][0] / det],
    ];
  }
}
