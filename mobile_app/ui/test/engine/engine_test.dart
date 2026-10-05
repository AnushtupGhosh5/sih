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

double _err(NavState s, SyntheticDrive d, double tt) {
  final tr = d.truthAt(d.imu.first.t + tt);
  return distanceM(s.lat, s.lon, tr.lat, tr.lon);
}

void main() {
  final mount = Mat3.fromEuler(pitch: 30 * math.pi / 180, roll: 10 * math.pi / 180, yaw: 45 * math.pi / 180);

  test('calibration, fusion and a 40 s blackout with a turn', () {
    final drive = SyntheticDrive(mount: mount, seed: 11)
      ..simulate(duration: 110, speed: _speed, headingRate: _headingRate);
    final engine = NavigationEngine();

    NavState? atEnd, after;
    var maxBlackoutDistance = 0.0;
    var sawDr = false;
    replayDrive(engine, drive, blackouts: [(60, 100)], onState: (tt, s) {
      if (tt > 62.5 && tt < 100) {
        expect(s.mode, NavMode.deadReckoning, reason: 'should be dead reckoning at $tt s');
        sawDr = true;
        maxBlackoutDistance = math.max(maxBlackoutDistance, s.blackoutDistanceM);
      }
      if ((tt - 99.9).abs() < 0.03) atEnd = s;
      if ((tt - 100.5).abs() < 0.03) after = s;
    });
    expect(sawDr, isTrue);
    expect(atEnd, isNotNull);

    // Calibration learned the mount from data.
    final cal = atEnd!.calibration!;
    expect(cal.fromData, isTrue);
    expect(cal.yawSign, -1.0);
    expect(cal.forwardCorr, greaterThan(0.6));
    expect(engine.forwardAxis!.dot(drive.trueForwardInPhone), greaterThan(0.95));
    final mountAngles = atEnd!.mount!;
    expect(mountAngles.pitchDeg, closeTo(30, 4));
    expect(mountAngles.rollDeg, closeTo(10, 4));
    expect(mountAngles.yawDeg, closeTo(45, 4));

    // The fusion filter learned the gyro bias before the outage.
    final expectedBias = -const Vec3(0.002, -0.001, 0.003).dot(drive.trueUpInPhone);
    expect(engine.filter.gyroBias, closeTo(expectedBias, 0.002));

    // Position error at the end of the blackout vs. ground truth.
    final err = _err(atEnd!, drive, 99.9);
    const distance = 15.0 * 37; // ~37 s of dead reckoning at 15 m/s
    expect(maxBlackoutDistance, greaterThan(500));
    expect(err / distance, lessThan(0.04), reason: 'drift ${err.toStringAsFixed(1)} m over $distance m');
    expect(atEnd!.uncertaintyM, greaterThan(5)); // honest: uncertainty grew

    // GNSS returns: recovery stats recorded, mode back to fusion at once.
    expect(after, isNotNull);
    expect(after!.mode, NavMode.gnssIns);
    expect(after!.lastRecovery, isNotNull);
    expect(after!.lastRecovery!.driftPct, lessThan(5));
    expect(after!.trail, isEmpty);
    expect(_err(after!, drive, 100.5), lessThan(6));
  });

  test('map-aided reckoning stays on the road and beats free inertial', () {
    // Straight drive north. The gyro bias steps after GNSS is lost, so the
    // filter cannot have learned it: free inertial curves away while the
    // road graph pins the solution.
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
    expect((onRoad.n - truth.n).abs() / 850, lessThan(0.04)); // odometry error small
    final ghost = atEnd!.freeDrPosition!;
    final free = latLonToEnu(ghost.lat, ghost.lon, drive.lat0, drive.lon0);
    expect(free.e.abs(), greaterThan(onRoad.e.abs() + 5)); // free inertial wandered off
  });

  test('simulate-tunnel switch enters dead reckoning at once and recovers on the first good fix', () {
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
        for (final mark in [20.2, 39.9, 41.5]) {
          if ((tt - mark).abs() < 0.03) modes[mark] = s.mode;
        }
      },
    );
    expect(modes[20.2], NavMode.deadReckoning); // switched within one sample
    expect(modes[39.9], NavMode.deadReckoning);
    expect(modes[41.5], NavMode.gnssIns); // first good fix after the switch
    final r = engine.current!.lastRecovery!;
    expect(r.distanceM, greaterThan(200));
    expect(r.errorM, lessThan(10));
  });

  test('blackout that starts at a standstill: stop detected, then speed integrated from zero', () {
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
    expect(stoppedSpeedSeen, lessThan(0.1)); // ZUPT held the speed at zero
    expect(atEnd, isNotNull);
    expect(atEnd!.mode, NavMode.deadReckoning);
    expect(atEnd!.speedMs, closeTo(12, 2.0)); // integrated back up from zero
    final err = _err(atEnd!, drive, 99.9);
    const travelled = 6 * 6 + 12 * 34; // ~444 m since pulling away
    expect(err / travelled, lessThan(0.10), reason: 'drift ${err.toStringAsFixed(1)} m over $travelled m');
  });

  test('braking inside the tunnel is captured (hold-last-speed would miss it)', () {
    double speed(double t) {
      if (t < 3) return 0;
      if (t < 13) return 1.5 * (t - 3);
      if (t < 50) return 15;
      if (t < 54) return 15 - 2.0 * (t - 50);
      return 7;
    }

    final drive = SyntheticDrive(mount: mount, seed: 31)
      ..simulate(duration: 105, speed: speed, headingRate: (t) => (t >= 20 && t < 28) ? (math.pi / 2) / 8 : 0);
    final engine = NavigationEngine();
    NavState? atEnd;
    replayDrive(engine, drive, blackouts: [(40, 101)], onState: (tt, s) {
      if ((tt - 99.9).abs() < 0.03) atEnd = s;
    });
    expect(atEnd, isNotNull);
    expect(atEnd!.mode, NavMode.deadReckoning);
    expect(atEnd!.speedMs, closeTo(7, 1.5));
    final err = _err(atEnd!, drive, 99.9);
    const travelled = 15.0 * 10 + 11.0 * 4 + 7.0 * 46; // ~516 m in the outage
    expect(err / travelled, lessThan(0.06), reason: 'drift ${err.toStringAsFixed(1)} m over $travelled m');
    // For reference: holding 15 m/s for the last 46 s would be ~370 m off.
  });

  test('pothole shocks do not corrupt speed or position', () {
    final drive = SyntheticDrive(mount: mount, seed: 41)
      ..simulate(
        duration: 90,
        speed: (t) => t < 3 ? 0 : math.min(15, 1.5 * (t - 3)),
        headingRate: (t) => (t >= 20 && t < 28) ? (math.pi / 2) / 8 : 0,
        shockEverySeconds: 2.5,
      );
    final engine = NavigationEngine();
    var worstSpeedErrGnss = 0.0;
    NavState? atEnd;
    replayDrive(engine, drive, blackouts: [(50, 80)], onState: (tt, s) {
      expect(s.lat.isFinite && s.lon.isFinite && s.speedMs.isFinite, isTrue);
      if (tt > 25 && tt < 50) {
        final tr = drive.truthAt(drive.imu.first.t + tt);
        worstSpeedErrGnss = math.max(worstSpeedErrGnss, (s.speedMs - tr.speed).abs());
      }
      if ((tt - 79.9).abs() < 0.03) atEnd = s;
    });
    expect(engine.current!.shockCount, greaterThan(20));
    expect(worstSpeedErrGnss, lessThan(1.5));
    expect(atEnd!.speedMs, closeTo(15, 2.5));
    expect(_err(atEnd!, drive, 79.9) / (15.0 * 30), lessThan(0.08));
  });

  test('GNSS loss is detected automatically within the stale timeout and the dot keeps moving', () {
    final drive = SyntheticDrive(mount: mount, seed: 51)
      ..simulate(duration: 60, speed: (t) => t < 3 ? 0 : math.min(15, 1.5 * (t - 3)), headingRate: (_) => 0);
    final engine = NavigationEngine();
    final stale = engine.monitor.staleSeconds;
    NavState? at41, atDetect, at45, at40;
    replayDrive(engine, drive, blackouts: [(40, 61)], onState: (tt, s) {
      if ((tt - 40.0).abs() < 0.03) at40 = s;
      if ((tt - 41.0).abs() < 0.03) at41 = s;
      if ((tt - (40 + stale + 0.3)).abs() < 0.03) atDetect = s;
      if ((tt - 45.0).abs() < 0.03) at45 = s;
    });
    expect(at41!.mode, isNot(NavMode.deadReckoning)); // one missed fix is not an outage
    expect(atDetect!.mode, NavMode.deadReckoning); // declared right after the timeout
    expect(distanceM(at40!.lat, at40!.lon, at45!.lat, at45!.lon), greaterThan(60)); // kept moving
    expect(at45!.blackoutDistanceM, greaterThan(30));
    expect(_err(at45!, drive, 45.0), lessThan(5));
  });

  test('multipath jumps are rejected instead of teleporting the vehicle', () {
    // Two fixes 80 m off (a reflection in an urban canyon), then normal fixes.
    final drive = SyntheticDrive(mount: mount, seed: 61)
      ..simulate(
        duration: 60,
        speed: (t) => t < 3 ? 0 : math.min(15, 1.5 * (t - 3)),
        headingRate: (_) => 0,
        gnssOffsetAt: (t) => (t >= 30 && t < 32) ? (e: 80.0, n: 0.0) : (e: 0.0, n: 0.0),
      );
    final engine = NavigationEngine();
    var worstErr = 0.0;
    var sawDegraded = false;
    NavState? at40;
    replayDrive(engine, drive, onState: (tt, s) {
      if (tt >= 30 && tt < 40) {
        worstErr = math.max(worstErr, _err(s, drive, tt));
        if (tt > 30.5 && tt < 32 && s.mode == NavMode.degraded) sawDegraded = true;
      }
      if ((tt - 40.0).abs() < 0.03) at40 = s;
    });
    expect(engine.current!.rejectedFixes, 2);
    expect(sawDegraded, isTrue);
    expect(worstErr, lessThan(15), reason: 'vehicle must not jump to the 80 m multipath fixes');
    expect(at40!.mode, NavMode.gnssIns);
  });

  test('a steady GNSS offset wins after three consistent fixes (the filter was wrong, not GNSS)', () {
    final drive = SyntheticDrive(mount: mount, seed: 62)
      ..simulate(
        duration: 60,
        speed: (t) => t < 3 ? 0 : math.min(15, 1.5 * (t - 3)),
        headingRate: (_) => 0,
        gnssOffsetAt: (t) => t >= 30 ? (e: 60.0, n: 0.0) : (e: 0.0, n: 0.0),
      );
    final engine = NavigationEngine();
    NavState? at31, at36;
    replayDrive(engine, drive, onState: (tt, s) {
      if ((tt - 31.0).abs() < 0.03) at31 = s;
      if ((tt - 36.0).abs() < 0.03) at36 = s;
    });
    // Not yet: one odd fix is ignored.
    expect(_err(at31!, drive, 31.0), lessThan(15));
    // Three consistent fixes later the solution follows the fix stream.
    final tr = drive.truthAt(drive.imu.first.t + 36.0);
    final p = latLonToEnu(at36!.lat, at36!.lon, drive.lat0, drive.lon0);
    expect((p.e - (tr.e + 60)).abs(), lessThan(12));
    expect(at36!.mode, NavMode.gnssIns);
  });

  test('fused solution while GNSS is healthy is smooth and accurate', () {
    final drive = SyntheticDrive(mount: mount, seed: 71)
      ..simulate(duration: 70, speed: _speed, headingRate: (t) => (t >= 45 && t < 53) ? (math.pi / 2) / 8 : 0);
    final engine = NavigationEngine();
    var sumPos = 0.0, sumSpd = 0.0, sumHead = 0.0;
    var n = 0;
    replayDrive(engine, drive, onState: (tt, s) {
      if (tt < 20 || tt > 68) return;
      final tr = drive.truthAt(drive.imu.first.t + tt);
      final e = distanceM(s.lat, s.lon, tr.lat, tr.lon);
      sumPos += e * e;
      sumSpd += (s.speedMs - tr.speed) * (s.speedMs - tr.speed);
      final dh = wrapPi(degToRad(s.headingDeg) - tr.headingRad);
      if (tr.speed > 2) sumHead += dh * dh;
      n++;
    });
    expect(math.sqrt(sumPos / n), lessThan(2.5));
    expect(math.sqrt(sumSpd / n), lessThan(0.6));
    expect(radToDeg(math.sqrt(sumHead / n)), lessThan(6));
    expect(engine.current!.uncertaintyM, lessThan(12));
  });

  test('never dead-reckons before the first fix', () {
    final drive = SyntheticDrive(mount: Mat3.identity, seed: 1)
      ..simulate(duration: 5, speed: (_) => 0, headingRate: (_) => 0);
    final engine = NavigationEngine();
    for (final s in drive.imu) {
      engine.onImu(s);
      expect(engine.mode, isNot(NavMode.deadReckoning));
      expect(engine.current!.hasFix, isFalse);
      expect(engine.current!.fwdAccelHistory.length, lessThanOrEqualTo(100));
    }
  });
}
