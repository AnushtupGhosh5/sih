import 'dart:math' as math;

import 'geo.dart';

double mean(List<double> x) {
  if (x.isEmpty) return 0;
  var s = 0.0;
  for (final v in x) {
    s += v;
  }
  return s / x.length;
}

double variance(List<double> x) {
  if (x.length < 2) return 0;
  final m = mean(x);
  var s = 0.0;
  for (final v in x) {
    final d = v - m;
    s += d * d;
  }
  return s / x.length;
}

double std(List<double> x) => math.sqrt(variance(x));

/// Pearson correlation; 0 when either series is constant.
double pearson(List<double> a, List<double> b) {
  final n = math.min(a.length, b.length);
  if (n < 2) return 0;
  final ma = mean(a.sublist(0, n)), mb = mean(b.sublist(0, n));
  var sab = 0.0, saa = 0.0, sbb = 0.0;
  for (var i = 0; i < n; i++) {
    final da = a[i] - ma, db = b[i] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  if (saa < 1e-18 || sbb < 1e-18) return 0;
  return sab / math.sqrt(saa * sbb);
}

/// Unwrap a sequence of angles (radians) so consecutive differences stay
/// within (-pi, pi].
List<double> unwrap(List<double> a) {
  if (a.isEmpty) return <double>[];
  final out = List<double>.filled(a.length, 0);
  out[0] = a[0];
  for (var i = 1; i < a.length; i++) {
    out[i] = out[i - 1] + wrapPi(a[i] - a[i - 1]);
  }
  return out;
}

/// numpy-style gradient for a uniformly sampled series with spacing [dt].
List<double> gradient(List<double> x, double dt) {
  final n = x.length;
  final g = List<double>.filled(n, 0);
  if (n < 2) return g;
  g[0] = (x[1] - x[0]) / dt;
  g[n - 1] = (x[n - 1] - x[n - 2]) / dt;
  for (var i = 1; i < n - 1; i++) {
    g[i] = (x[i + 1] - x[i - 1]) / (2 * dt);
  }
  return g;
}

/// Circular mean of angles in radians.
double circularMean(List<double> angles) {
  if (angles.isEmpty) return 0;
  var s = 0.0, c = 0.0;
  for (final a in angles) {
    s += math.sin(a);
    c += math.cos(a);
  }
  return math.atan2(s, c);
}

/// Mean resultant length of angles (1 = perfectly consistent, 0 = random).
double circularR(List<double> angles) {
  if (angles.isEmpty) return 0;
  var s = 0.0, c = 0.0;
  for (final a in angles) {
    s += math.sin(a);
    c += math.cos(a);
  }
  return math.sqrt(s * s + c * c) / angles.length;
}

/// Turn a sample-and-hold series (e.g. 1 Hz GNSS speed held at IMU rate)
/// into a piecewise-linear one between the points where the value changes.
List<double> interpolateHeld(List<double> x) {
  final n = x.length;
  if (n < 3) return List<double>.from(x);
  final knots = <int>[0];
  for (var i = 1; i < n; i++) {
    if (x[i] != x[i - 1]) knots.add(i);
  }
  if (knots.last != n - 1) knots.add(n - 1);
  final out = List<double>.from(x);
  for (var k = 0; k + 1 < knots.length; k++) {
    final i0 = knots[k], i1 = knots[k + 1];
    if (i1 - i0 < 2) continue;
    final v0 = x[i0], v1 = x[i1];
    for (var i = i0; i <= i1; i++) {
      out[i] = v0 + (v1 - v0) * (i - i0) / (i1 - i0);
    }
  }
  return out;
}

/// Second-order Butterworth low-pass applied forward and backward
/// (zero-phase, like scipy.signal.filtfilt with a 2nd-order filter).
List<double> butterLowpassFiltfilt(List<double> x, double fc, double fs) {
  if (x.length < 4) return List<double>.from(x);
  final ff = fc / fs;
  final ita = 1.0 / math.tan(math.pi * ff);
  final q = math.sqrt(2.0);
  final b0 = 1.0 / (1.0 + q * ita + ita * ita);
  final b1 = 2 * b0;
  final b2 = b0;
  final a1 = 2.0 * (ita * ita - 1.0) * b0;
  final a2 = -(1.0 - q * ita + ita * ita) * b0;

  List<double> run(List<double> s) {
    final y = List<double>.filled(s.length, 0);
    var x1 = s[0], x2 = s[0], y1 = s[0], y2 = s[0];
    for (var i = 0; i < s.length; i++) {
      final v = b0 * s[i] + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
      x2 = x1;
      x1 = s[i];
      y2 = y1;
      y1 = v;
      y[i] = v;
    }
    return y;
  }

  final fwd = run(x);
  final bwd = run(fwd.reversed.toList());
  return bwd.reversed.toList();
}
