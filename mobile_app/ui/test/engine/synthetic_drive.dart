import 'dart:math' as math;

import 'package:sih_navigation/engine/geo.dart';
import 'package:sih_navigation/engine/navigation_engine.dart';
import 'package:sih_navigation/engine/vec3.dart';

class TruthPoint {
  const TruthPoint(this.t, this.e, this.n, this.lat, this.lon, this.headingRad, this.speed);
  final double t, e, n, lat, lon, headingRad, speed;
}

/// Generates phone IMU samples and GNSS fixes for a vehicle with a known
/// speed and heading-rate profile, seen through an arbitrary phone mount.
///
/// Vehicle frame: x forward, y left, z up (right-handed). Android sensor
/// conventions: the accelerometer reports +g along the up axis at rest and
/// the gyro is positive counter-clockwise about each axis.
class SyntheticDrive {
  SyntheticDrive({
    required this.mount,
    this.fs = 20.0,
    this.lat0 = 28.6139,
    this.lon0 = 77.2090,
    int seed = 7,
  }) : _rng = math.Random(seed);

  /// Rotation from the vehicle frame into the phone frame.
  final Mat3 mount;
  final double fs;
  final double lat0, lon0;
  final math.Random _rng;

  final List<ImuSample> imu = <ImuSample>[];
  final List<GnssFix> gnss = <GnssFix>[];
  final List<TruthPoint> truth = <TruthPoint>[];

  double gauss(double sigma) {
    final u1 = math.max(1e-12, _rng.nextDouble()), u2 = _rng.nextDouble();
    return sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2);
  }

  Vec3 get trueForwardInPhone => mount.apply(Vec3.unitX);
  Vec3 get trueUpInPhone => mount.apply(Vec3.unitZ);

  void simulate({
    required double duration,
    required double Function(double t) speed,
    required double Function(double t) headingRate,
    double h0 = 0,
    double accelNoise = 0.15,
    double vibrationPerMs = 0.03, // extra accel noise per m/s of speed (road vibration)
    double gyroNoise = 0.004,
    Vec3 gyroBias = const Vec3(0.002, -0.001, 0.003),
    Vec3 Function(double t)? gyroBiasAt,
    double gnssAccuracy = 5.0,
    double gnssPosNoise = 1.0,
    double tStart = 1000.0,
  }) {
    final dt = 1.0 / fs;
    final n = (duration * fs).round();
    var e = 0.0, nn = 0.0, h = h0;
    var nextGnss = 0.0;
    final gravPhone = mount.apply(const Vec3(0, 0, 9.81));
    for (var k = 0; k <= n; k++) {
      final tt = k * dt;
      final t = tStart + tt;
      final v = speed(tt);
      final a = (speed(tt + dt) - speed(math.max(0.0, tt - dt))) / (tt < dt ? dt : 2 * dt);
      final hr = headingRate(tt);

      final aVeh = Vec3(a, -v * hr, 9.81);
      final gyroVeh = Vec3(0, 0, -hr);
      final northVeh = Vec3(math.cos(h), math.sin(h), 0);
      final magVeh = northVeh * 25.0 + const Vec3(0, 0, -40.0);

      final sA = accelNoise + vibrationPerMs * v;
      final noiseA = Vec3(gauss(sA), gauss(sA), gauss(sA));
      final noiseG = Vec3(gauss(gyroNoise), gauss(gyroNoise), gauss(gyroNoise));
      final accelPhone = mount.apply(aVeh) + noiseA;
      imu.add(ImuSample(
        t: t,
        accel: accelPhone,
        linear: accelPhone - gravPhone,
        gyro: mount.apply(gyroVeh) + (gyroBiasAt?.call(tt) ?? gyroBias) + noiseG,
        mag: mount.apply(magVeh),
      ));
      final ll = enuToLatLon(e, nn, lat0, lon0);
      truth.add(TruthPoint(t, e, nn, ll.lat, ll.lon, h, v));

      if (tt >= nextGnss - 1e-9) {
        nextGnss += 1.0;
        final p = enuToLatLon(e + gauss(gnssPosNoise), nn + gauss(gnssPosNoise), lat0, lon0);
        gnss.add(GnssFix(
          t: t,
          lat: p.lat,
          lon: p.lon,
          speedMs: math.max(0.0, v + gauss(0.1)),
          courseDeg: wrapDeg(radToDeg(h)),
          accuracyM: gnssAccuracy,
        ));
      }

      h += hr * dt;
      e += v * math.sin(h) * dt;
      nn += v * math.cos(h) * dt;
    }
  }

  TruthPoint truthAt(double t) {
    var best = truth.first;
    for (final p in truth) {
      if ((p.t - t).abs() < (best.t - t).abs()) best = p;
    }
    return best;
  }
}

/// Replays a drive through the engine, withholding GNSS fixes inside the
/// given blackout windows (seconds since drive start). [onState] receives the
/// engine state after every IMU sample.
void replayDrive(
  NavigationEngine engine,
  SyntheticDrive drive, {
  List<(double, double)> blackouts = const [],
  void Function(double tt, NavState s)? onState,
  void Function(double tt)? beforeSample,
}) {
  final t0 = drive.imu.first.t;
  var gi = 0;
  for (final s in drive.imu) {
    final tt = s.t - t0;
    beforeSample?.call(tt);
    while (gi < drive.gnss.length && drive.gnss[gi].t <= s.t) {
      final f = drive.gnss[gi++];
      final ft = f.t - t0;
      final withheld = blackouts.any((b) => ft >= b.$1 && ft < b.$2);
      if (!withheld) engine.onGnss(f);
    }
    engine.onImu(s);
    final st = engine.current;
    if (st != null) onState?.call(tt, st);
  }
}
