import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:sih_navigation/engine/geo.dart';
import 'package:sih_navigation/engine/navigation_engine.dart';
import 'package:sih_navigation/engine/road_graph.dart';
import 'package:sih_navigation/engine/vec3.dart';

import 'synthetic_drive.dart';

/// Speed profile (m/s): stop, accelerate, cruise, slow down, accelerate,
/// cruise. Plenty of longitudinal acceleration for the calibration window.
double _speed(double t) {
  if (t < 3) return 0;
  if (t < 13) return 1.5 * (t - 3);
  if (t < 25) return 15;
  if (t < 30) return 15 - 1.4 * (t - 25);
  if (t < 35) return 8;
  if (t < 40) return 8 + 1.4 * (t - 35);
  return 15;
}

/// Heading rate (rad/s, clockwise positive): a 90 degree right turn at
/// 45-53 s (inside the calibration window), a 60 degree left turn during the
/// blackout at 70-76 s.
double _headingRate(double t) {
  if (t >= 45 && t < 53) return (math.pi / 2) / 8;
  if (t >= 70 && t < 76) return -(math.pi / 3) / 6;
  return 0;
}

void main() {
  final mount = Mat3.fromEuler(pitch: 30 * math.pi / 180, roll: 10 * math.pi / 180, yaw: 45 * math.pi / 180);

  test('calibration + free dead reckoning through a 40 s blackout', () {
    final drive = SyntheticDrive(mount: mount, seed: 11)
      ..simulate(duration: 110, speed: _speed, headingRate: _headingRate);
    final engine = NavigationEngine();

    NavState? atEnd;
    var maxBlackoutDistance = 0.0;
    var sawDr = false;
    Calibration? cal;
    replayDrive(engine, drive, blackouts: [(60, 100)], onState: (tt, s) {
      if (tt > 63 && tt < 100) {
        expect(s.mode, NavMode.deadReckoning, reason: 'should be dead reckoning at $tt s');
        sawDr = true;
        cal ??= s.calibration;
        maxBlackoutDistance = math.max(maxBlackoutDistance, s.blackoutDistanceM);
      }
      if ((tt - 99.9).abs() < 0.03) atEnd = s;
    });

    expect(sawDr, isTrue);
    expect(atEnd, isNotNull);

    // Calibration learned the mount from data.
    expect(cal, isNotNull);
    expect(cal!.fromData, isTrue);
    expect(cal!.yawSign, -1.0);
    expect(cal!.yawCorr.abs(), greaterThan(0.8));
    expect(cal!.forwardCorr, greaterThan(0.6));
    expect(cal!.fwd!.dot(drive.trueForwardInPhone), greaterThan(0.95));
    expect(cal!.compassUsable, isTrue);

    // Position error at the end of the blackout vs. ground truth.
    final truth = drive.truthAt(drive.imu.first.t + 99.9);
    final err = distanceM(atEnd!.lat, atEnd!.lon, truth.lat, truth.lon);
    final distance = 15.0 * 37; // ~37 s of dead reckoning at 15 m/s
    expect(maxBlackoutDistance, greaterThan(500));
    expect(err / distance, lessThan(0.03), reason: 'drift ${err.toStringAsFixed(1)} m over $distance m');

    // GNSS returns: recovery stats recorded, mode back to fusion.
    final last = engine.current!;
    expect(last.mode, NavMode.gnssIns);
    expect(last.lastRecovery, isNotNull);
    expect(last.lastRecovery!.driftPct, lessThan(5));
    expect(last.trail, isEmpty);
  });

  test('map-aided reckoning stays on the road and beats free inertial', () {
    // Straight drive north. The gyro bias steps after GNSS is lost, so the
    // pre-blackout calibration cannot remove it: free DR curves away while
    // the road graph pins the solution.
    final drive = SyntheticDrive(mount: Mat3.identity, seed: 2)
      ..simulate(
        duration: 90,
        speed: (t) => t < 2 ? 0 : math.min(20, 2.0 * (t - 2)),
        headingRate: (_) => 0,
        gyroBiasAt: (t) => t < 42 ? Vec3.zero : const Vec3(0, 0, -0.02), // 1.1 deg/s drift
      );
    final graph = RoadGraph.build(
      lat0: drive.lat0,
      lon0: drive.lon0,
      nodes: {1: (e: 0.0, n: -200.0), 2: (e: 0.0, n: 1000.0), 3: (e: 0.0, n: 3000.0)},
      edges: [(1, 2), (2, 3)],
    );
    final engine = NavigationEngine()..roadGraph = graph;

    NavState? atEnd;
    replayDrive(engine, drive, blackouts: [(40, 85)], onState: (tt, s) {
      if (tt > 44 && tt < 85) {
        expect(s.mode, NavMode.deadReckoning);
        expect(s.mapAided, isTrue);
      }
      if ((tt - 84.9).abs() < 0.03) atEnd = s;
    });
    expect(atEnd, isNotNull);
    final truth = drive.truthAt(drive.imu.first.t + 84.9);
    final onRoad = latLonToEnu(atEnd!.lat, atEnd!.lon, drive.lat0, drive.lon0);
    expect(onRoad.e.abs(), lessThan(1.0)); // pinned to the road
    expect((onRoad.n - truth.n).abs() / 800, lessThan(0.03)); // odometry error small
    final ghost = atEnd!.freeDrPosition!;
    final free = latLonToEnu(ghost.lat, ghost.lon, drive.lat0, drive.lon0);
    expect(free.e.abs(), greaterThan(onRoad.e.abs() + 5)); // free DR wandered off
  });

  test('simulate-tunnel switch enters and leaves dead reckoning', () {
    final drive = SyntheticDrive(mount: Mat3.identity, seed: 4)
      ..simulate(duration: 60, speed: (t) => t < 2 ? 0 : math.min(12, 1.5 * (t - 2)), headingRate: (_) => 0);
    final engine = NavigationEngine();
    final modes = <double, NavMode>{};
    replayDrive(
      engine,
      drive,
      beforeSample: (tt) {
        if ((tt - 20).abs() < 0.01) engine.simulateBlackout = true;
        if ((tt - 40).abs() < 0.01) engine.simulateBlackout = false;
      },
      onState: (tt, s) {
        if ((tt - 20.2).abs() < 0.03 || (tt - 39.9).abs() < 0.03 || (tt - 43).abs() < 0.03) {
          modes[(tt * 10).roundToDouble() / 10] = s.mode;
        }
      },
    );
    expect(modes[20.2], NavMode.deadReckoning); // switched within one sample
    expect(modes[39.9], NavMode.deadReckoning);
    expect(modes[43.0], NavMode.gnssIns); // two good fixes after the switch
    final r = engine.current!.lastRecovery!;
    expect(r.distanceM, greaterThan(200));
    expect(r.errorM, lessThan(10));
  });

  test('blackout that starts at a standstill: stop detected, then speed integrated from zero', () {
    // Drive, brake to a stop at 45-50 s, GNSS lost at 52 s while stopped,
    // pull away at 60 s inside the outage, cruise at 12 m/s until 100 s.
    double speed(double t) {
      if (t < 3) return 0;
      if (t < 13) return 1.5 * (t - 3);
      if (t < 45) return 15;
      if (t < 50) return 15 - 3.0 * (t - 45);
      if (t < 60) return 0;
      if (t < 66) return 2.0 * (t - 60);
      return 12;
    }

    final drive = SyntheticDrive(mount: mount, seed: 21)
      ..simulate(duration: 105, speed: speed, headingRate: (t) => (t >= 20 && t < 28) ? (math.pi / 2) / 8 : 0);
    final engine = NavigationEngine();
    NavState? atEnd;
    var stoppedSpeedSeen = double.infinity;
    replayDrive(engine, drive, blackouts: [(52, 101)], onState: (tt, s) {
      if (tt > 57 && tt < 60) stoppedSpeedSeen = math.min(stoppedSpeedSeen, s.speedMs);
      if ((tt - 99.9).abs() < 0.03) atEnd = s;
    });
    expect(stoppedSpeedSeen, lessThan(0.05)); // ZUPT zeroed the held speed
    expect(atEnd, isNotNull);
    expect(atEnd!.mode, NavMode.deadReckoning);
    expect(atEnd!.speedMs, closeTo(12, 2.0)); // integrated back up from zero
    final truth = drive.truthAt(drive.imu.first.t + 99.9);
    final err = distanceM(atEnd!.lat, atEnd!.lon, truth.lat, truth.lon);
    final travelled = 6 * 6 + 12 * 34; // ~444 m since pulling away
    expect(err / travelled, lessThan(0.10), reason: 'drift ${err.toStringAsFixed(1)} m over $travelled m');
  });

  test('never dead-reckons before the first fix', () {
    final drive = SyntheticDrive(mount: Mat3.identity, seed: 1)
      ..simulate(duration: 5, speed: (_) => 0, headingRate: (_) => 0);
    final engine = NavigationEngine();
    for (final s in drive.imu) {
      engine.onImu(s);
      expect(engine.mode, isNot(NavMode.deadReckoning));
      expect(engine.current!.hasFix, isFalse);
    }
  });
}
