import 'dart:collection';
import 'dart:math' as math;

import 'signal.dart';
import 'vec3.dart';

/// One IMU sample recorded while GNSS was healthy, with the GNSS speed and
/// course that were current at that moment.
class CalSample {
  const CalSample({
    required this.t,
    required this.lin,
    required this.gyro,
    required this.ghat,
    required this.speed,
    required this.courseRad,
    this.azimuthRad,
    double? fixT,
  }) : fixT = fixT ?? t;

  final double t; // seconds
  final double fixT; // time of the GNSS fix the held speed/course came from
  final Vec3 lin; // linear acceleration (gravity removed), phone frame
  final Vec3 gyro; // rad/s, phone frame
  final Vec3 ghat; // unit gravity direction, phone frame
  final double speed; // GNSS speed m/s (held between fixes)
  final double courseRad; // GNSS course, radians clockwise from north
  final double? azimuthRad; // phone compass azimuth, if available
}

/// Result of the pre-blackout calibration (port of `calibrate()` in
/// `src/idr/dead_reckoning.py`).
class Calibration {
  const Calibration({
    required this.ghat,
    required this.fwd,
    required this.accelBias,
    required this.yawSign,
    required this.yawBias,
    required this.forwardCorr,
    required this.yawCorr,
    required this.azOffset,
    required this.compassR,
    required this.windowSeconds,
    required this.fromData,
  });

  /// Default yaw sign for Android: the accelerometer reports +g along the
  /// axis pointing UP, so gyro·ĝ is the counter-clockwise rate about the
  /// up axis and heading (clockwise from north) changes with the opposite sign.
  static const double defaultYawSign = -1.0;

  final Vec3 ghat;
  final Vec3? fwd; // forward axis in the phone frame (null if not found)
  final double accelBias; // residual forward-acceleration bias (m/s^2)
  final double yawSign;
  final double yawBias; // rad/s
  final double forwardCorr; // correlation of forward accel with GNSS dv/dt
  final double yawCorr; // correlation of gyro yaw with GNSS heading rate
  final double azOffset; // compass azimuth -> vehicle heading offset (rad)
  final double compassR; // consistency of that offset (0..1)
  final double windowSeconds;
  final bool fromData;

  bool get hasForwardAxis => fwd != null && forwardCorr > 0.3;
  bool get compassUsable => compassR > 0.8;

  factory Calibration.fallback(Vec3 ghat) => Calibration(
        ghat: ghat,
        fwd: null,
        accelBias: 0,
        yawSign: defaultYawSign,
        yawBias: 0,
        forwardCorr: 0,
        yawCorr: 0,
        azOffset: 0,
        compassR: 0,
        windowSeconds: 0,
        fromData: false,
      );

  @override
  String toString() =>
      'Calibration(fwdCorr=${forwardCorr.toStringAsFixed(2)}, yawSign=$yawSign, '
      'yawBias=${yawBias.toStringAsExponential(2)}, compassR=${compassR.toStringAsFixed(2)}, '
      'window=${windowSeconds.toStringAsFixed(1)}s)';
}

/// Keeps a rolling window of GNSS-available samples and, on demand, estimates
/// the forward axis, accelerometer bias, gyro yaw sign/bias and compass
/// offset exactly as the desktop pipeline does before a simulated blackout.
class BlackoutCalibrator {
  BlackoutCalibrator({
    this.windowSeconds = 30.0,
    this.minSeconds = 8.0,
    this.fs = 10.0,
    this.lowpassHz = 0.5,
  });

  final double windowSeconds;
  final double minSeconds;
  final double fs;
  final double lowpassHz;

  final ListQueue<CalSample> _buf = ListQueue<CalSample>();

  int get length => _buf.length;
  double get seconds => _buf.length < 2 ? 0 : _buf.last.t - _buf.first.t;

  void add(CalSample s) {
    _buf.addLast(s);
    while (_buf.isNotEmpty && s.t - _buf.first.t > windowSeconds) {
      _buf.removeFirst();
    }
  }

  void clear() => _buf.clear();

  /// Returns null when the window is too short to calibrate.
  Calibration? compute() {
    if (_buf.length < 2) return null;
    final samples = _buf.toList(growable: false);
    final t0 = samples.first.t, t1 = samples.last.t;
    if (t1 - t0 < minSeconds) return null;

    // Resample (sample-and-hold) onto a uniform grid at [fs].
    final dt = 1.0 / fs;
    final n = ((t1 - t0) * fs).floor() + 1;
    final lin = <Vec3>[], gyr = <Vec3>[], gh = <Vec3>[];
    final sp = <double>[], head = <double>[], fixT = <double>[];
    final az = <double?>[];
    var j = 0;
    for (var i = 0; i < n; i++) {
      final tg = t0 + i * dt;
      while (j + 1 < samples.length && samples[j + 1].t <= tg) {
        j++;
      }
      final s = samples[j];
      lin.add(s.lin);
      gyr.add(s.gyro);
      gh.add(s.ghat);
      sp.add(s.speed);
      head.add(s.courseRad);
      fixT.add(s.fixT);
      az.add(s.azimuthRad);
    }

    // Gravity direction and a horizontal basis.
    var gsum = Vec3.zero;
    for (final g in gh) {
      gsum = gsum + g;
    }
    final ghat = gsum.normalized;
    final (e1, e2) = horizontalBasis(ghat);

    final c1 = butterLowpassFiltfilt([for (final v in lin) v.dot(e1)], lowpassHz, fs);
    final c2 = butterLowpassFiltfilt([for (final v in lin) v.dot(e2)], lowpassHz, fs);

    // GNSS-derived longitudinal acceleration and heading rate as piecewise-
    // constant rates between consecutive fixes (their integrals are exact).
    // Statistics are restricted to the span between the first and last fix
    // inside the window: a partial interval at either edge would otherwise
    // mis-state the heading/speed change by up to one fix interval and bias
    // the gyro/accelerometer bias estimates.
    final spR = _intervalRates(sp, fixT, dt);
    final hdR = _intervalRates(unwrap(head), fixT, dt);
    final dv = butterLowpassFiltfilt(spR.rates, lowpassHz, fs);
    final dh = hdR.rates;
    final minSpan = (fs * 3).round();
    final (s0, s1) = spR.last - spR.first >= minSpan ? (spR.first, spR.last) : (0, n);
    final (h0, h1) = hdR.last - hdR.first >= minSpan ? (hdR.first, hdR.last) : (0, n);

    // Forward axis: horizontal direction whose acceleration best tracks dv/dt.
    Vec3? fwd;
    var bestR = -2.0;
    var bestTh = 0.0;
    final dvSpan = dv.sublist(s0, s1);
    final moving = sp.reduce(math.max) > 2.0 && std(dvSpan) > 0.05;
    if (moving) {
      for (var k = 0; k < 180; k++) {
        final th = 2 * math.pi * k / 180;
        final ct = math.cos(th), st = math.sin(th);
        final af = List<double>.generate(s1 - s0, (i) => ct * c1[s0 + i] + st * c2[s0 + i]);
        if (std(af) < 1e-6) continue;
        final r = pearson(af, dvSpan);
        if (r > bestR) {
          bestR = r;
          bestTh = th;
        }
      }
      if (bestR > -2) {
        fwd = (e1 * math.cos(bestTh) + e2 * math.sin(bestTh)).normalized;
      }
    }
    var accelBias = 0.0;
    if (fwd != null) {
      final aF = butterLowpassFiltfilt([for (final v in lin) v.dot(fwd)], lowpassHz, fs);
      accelBias = mean(List<double>.generate(s1 - s0, (i) => aF[s0 + i] - dv[s0 + i]));
    }

    // Gyro yaw about the live vertical: sign so integrated yaw follows the
    // GNSS heading, bias = mean residual.
    final yaw = List<double>.generate(h1 - h0, (i) => gyr[h0 + i].dot(gh[h0 + i]));
    final dhSpan = dh.sublist(h0, h1);
    final rYaw = pearson(yaw, dhSpan);
    // The physical sign is fixed by the sensor convention (defaultYawSign);
    // the data only overrides it when a clear turn makes the correlation
    // unambiguous.
    final double sign;
    if (rYaw.isNaN || rYaw.abs() < 0.5) {
      sign = Calibration.defaultYawSign; // too little turning to decide
    } else {
      sign = rYaw > 0 ? 1.0 : -1.0;
    }
    final yawBias = mean(List<double>.generate(h1 - h0, (i) => sign * yaw[i] - dhSpan[i]));

    // Compass azimuth -> vehicle heading offset.
    final diffs = <double>[];
    for (var i = 0; i < n; i++) {
      final a = az[i];
      if (a != null && sp[i] > 1.0) diffs.add(head[i] - a);
    }
    final azOffset = diffs.isEmpty ? 0.0 : circularMean(diffs);
    final compassR = diffs.length < fs * 3 ? 0.0 : circularR(diffs);

    return Calibration(
      ghat: ghat,
      fwd: fwd,
      accelBias: accelBias,
      yawSign: sign,
      yawBias: yawBias,
      forwardCorr: bestR > -2 ? bestR : 0,
      yawCorr: rYaw.isNaN ? 0 : rYaw,
      azOffset: azOffset,
      compassR: compassR,
      windowSeconds: t1 - t0,
      fromData: true,
    );
  }

  /// Rate of change of a sample-and-hold GNSS series, piecewise constant
  /// between fix arrivals (indices where [marks], the fix timestamp, changes).
  /// [first]/[last] bound the span covered by complete fix intervals; the
  /// partial edges are filled with the adjacent interval's rate.
  static ({List<double> rates, int first, int last}) _intervalRates(
      List<double> x, List<double> marks, double dt) {
    final n = x.length;
    final rates = List<double>.filled(n, 0);
    final knots = <int>[];
    for (var i = 1; i < n; i++) {
      if (marks[i] != marks[i - 1]) knots.add(i);
    }
    if (knots.length < 2) return (rates: rates, first: 0, last: n);
    for (var j = 0; j + 1 < knots.length; j++) {
      final a = knots[j], b = knots[j + 1];
      final r = (x[b] - x[a]) / ((b - a) * dt);
      for (var i = a; i < b; i++) {
        rates[i] = r;
      }
    }
    for (var i = 0; i < knots.first; i++) {
      rates[i] = rates[knots.first];
    }
    for (var i = knots.last; i < n; i++) {
      rates[i] = rates[knots.last - 1];
    }
    return (rates: rates, first: knots.first, last: knots.last);
  }

  /// Two orthonormal vectors spanning the plane perpendicular to [ghat].
  static (Vec3, Vec3) horizontalBasis(Vec3 ghat) {
    var ref = Vec3.unitX;
    if (ghat.dot(ref).abs() > 0.9) ref = Vec3.unitY;
    final e1 = ghat.cross(ref).normalized;
    final e2 = ghat.cross(e1).normalized;
    return (e1, e2);
  }
}
